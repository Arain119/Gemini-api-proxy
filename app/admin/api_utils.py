import asyncio
import json
import time
import uuid
import logging
import os
import sys
import base64
import mimetypes
import random
import hashlib
import itertools
from collections import deque
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from google.genai import types
from fastapi import HTTPException

from app.admin.database import Database
from app.admin.api_models import ChatCompletionRequest, ChatMessage, ContentPart
from app.admin.cli_auth import (
    call_gemini_with_cli_account,
    upload_file_with_cli_account,
    delete_file_with_cli_account,
)

# 配置日志
logger = logging.getLogger(__name__)

# 防自动化检测注入器
class GeminiAntiDetectionInjector:
    """
    防自动化检测的 Unicode 字符注入器
    """
    def __init__(self):
        # Unicode符号库
        self.safe_symbols = [
            '∙', '∘', '∞', '≈', '≠', '≤', '≥', '±', '∓', '×', '÷', '∂', '∆', '∇',
            '○', '●', '◯', '◦', '◉', '◎', '⦿', '⊙', '⊚', '⊛', '⊜', '⊝',
            '□', '■', '▢', '▣', '▤', '▥', '▦', '▧', '▨', '▩', '▪', '▫',
            '△', '▲', '▴', '▵', '▶', '▷', '▸', '▹', '►', '▻', '▼', '▽',
            '◀', '◁', '◂', '◃', '◄', '◅', '◆', '◇', '◈', '◉', '◊',
            '☆', '★', '⭐', '✦', '✧', '✩', '✪', '✫', '✬', '✭', '✮', '✯',
            '✰', '✱', '✲', '✳', '✴', '✵', '✶', '✷', '✸', '✹', '✺', '✻',
            '→', '←', '↑', '↓', '↔', '↕', '↖', '↗', '↘', '↙', '↚', '↛',
            '⇒', '⇐', '⇑', '⇓', '⇔', '⇕', '⇖', '⇗', '⇘', '⇙', '⇚', '⇛',
            '‖', '‗', '‰', '‱', '′', '″', '‴', '‵', '‶', '‷', '‸', '‹', '›',
            '‼', '‽', '‾', '‿', '⁀', '⁁', '⁂', '⁃', '⁆', '⁇', '⁈', '⁉',
            '※', '⁎', '⁑', '⁒', '⁓', '⁔', '⁕', '⁖', '⁗', '⁘', '⁙', '⁚',
            '⊕', '⊖', '⊗', '⊘', '⊙', '⊚', '⊛', '⊜', '⊝', '⊞', '⊟', '⊠',
            '⋄', '⋅', '⋆', '⋇', '⋈', '⋉', '⋊', '⋋', '⋌', '⋍', '⋎', '⋏'
        ]
        self.invisible_symbols = ['\u200B', '\u200C', '\u2060']
        self.request_history = set()
        self.max_history_size = 5000

    def inject_symbols(self, text: str, strategy: str = 'auto') -> str:
        if not text.strip(): return text
        symbol_count = random.randint(2, 4)
        if strategy == 'invisible':
            symbols = random.sample(self.invisible_symbols, min(2, len(self.invisible_symbols)))
        elif strategy == 'mixed':
            visible_count = random.randint(1, 2)
            invisible_count = 1
            symbols = (random.sample(self.safe_symbols, visible_count) +
                       random.sample(self.invisible_symbols, invisible_count))
        else:
            symbols = random.sample(self.safe_symbols, min(symbol_count, len(self.safe_symbols)))
        strategies = ['prefix', 'suffix', 'wrap']
        if strategy == 'auto': strategy = random.choice(strategies)
        if strategy == 'prefix': return ''.join(symbols) + ' ' + text
        elif strategy == 'suffix': return text + ' ' + ''.join(symbols)
        elif strategy == 'wrap':
            mid = len(symbols) // 2
            prefix = ''.join(symbols[:mid])
            suffix = ''.join(symbols[mid:])
            return f"{prefix} {text} {suffix}" if prefix and suffix else f"{text} {suffix}"
        else: return text + ' ' + ''.join(symbols)

    def process_content(self, content: Union[str, List]) -> Union[str, List]:
        content_hash = hashlib.md5(str(content).encode()).hexdigest()
        strategy = random.choice(['mixed', 'invisible', 'prefix', 'suffix']) if content_hash in self.request_history else 'auto'
        self.request_history.add(content_hash)
        if len(self.request_history) > self.max_history_size:
            self.request_history = set(list(self.request_history)[self.max_history_size // 2:])
        if isinstance(content, str): return self.inject_symbols(content, strategy)
        elif isinstance(content, list):
            processed = []
            for item in content:
                if isinstance(item, dict):
                    processed_item = item.copy()
                    if 'text' in processed_item:
                        processed_item['text'] = self.inject_symbols(processed_item['text'], strategy)
                    processed.append(processed_item)
                else: processed.append(item)
            return processed
        return content

    def get_statistics(self) -> Dict:
        return {
            'available_symbols': len(self.safe_symbols),
            'invisible_symbols': len(self.invisible_symbols),
            'request_history_size': len(self.request_history),
            'max_history_size': self.max_history_size
        }

class UserRateLimiter:
    """处理单个用户密钥的速率限制"""
    def __init__(self, db: Database, user_key_info: Dict):
        self.db = db
        self.user_key_info = user_key_info

    def check_rate_limits(self):
        """检查用户是否超出速率限制"""
        user_id = self.user_key_info['id']
        
        # -1 表示无限制
        rpm_limit = self.user_key_info.get('rpm_limit', -1)
        tpm_limit = self.user_key_info.get('tpm_limit', -1)
        rpd_limit = self.user_key_info.get('rpd_limit', -1)

        # 检查每分钟请求数 (RPM) 和每分钟令牌数 (TPM)
        if rpm_limit != -1 or tpm_limit != -1:
            usage_minute = self.db.get_user_key_usage_stats(user_id, 'minute')
            if rpm_limit != -1 and usage_minute['requests'] >= rpm_limit:
                raise HTTPException(status_code=429, detail=f"Rate limit exceeded for RPM: {rpm_limit}")
            if tpm_limit != -1 and usage_minute['tokens'] >= tpm_limit:
                raise HTTPException(status_code=429, detail=f"Rate limit exceeded for TPM: {tpm_limit}")

        # 检查每天请求数 (RPD)
        if rpd_limit != -1:
            usage_day = self.db.get_user_key_usage_stats(user_id, 'day')
            if usage_day['requests'] >= rpd_limit:
                raise HTTPException(status_code=429, detail=f"Rate limit exceeded for RPD: {rpd_limit}")


def decrypt_response(hex_string: str) -> str:
    if not isinstance(hex_string, str) or not hex_string or len(hex_string) % 8 != 0: return hex_string
    try:
        if not all(c in '0123456789abcdef' for c in hex_string.lower()): return hex_string
        txt = ''
        for i in range(0, len(hex_string), 8):
            codepoint = 0
            hex_block = hex_string[i:i+8]
            for j in range(0, 8, 2):
                byte_hex = hex_block[j:j+2]
                byte_val = int(byte_hex, 16)
                decrypted_byte = byte_val ^ 0x5A
                codepoint = (codepoint << 8) | decrypted_byte
            txt += chr(codepoint)
        return txt
    except (ValueError, TypeError): return hex_string

class RateLimitCache:
    def __init__(self, max_entries: int = 10000, default_window: int = 60):
        self.cache: Dict[str, Dict[str, deque]] = {}
        self.max_entries = max_entries
        self.default_window = default_window
        self.lock = asyncio.Lock()

    @staticmethod
    def _prune_queue(queue: deque, cutoff: float) -> None:
        while queue and queue[0][0] <= cutoff:
            queue.popleft()

    def _ensure_model_locked(self, model_name: str) -> Dict[str, deque]:
        metrics = self.cache.get(model_name)
        if metrics is None:
            metrics = {'requests': deque(), 'tokens': deque()}
            self.cache[model_name] = metrics
        return metrics

    async def cleanup_expired(self, window_seconds: int = None):
        if window_seconds is None:
            window_seconds = self.default_window
        cutoff_time = time.time() - window_seconds
        async with self.lock:
            for model_name in list(self.cache.keys()):
                metrics = self.cache.get(model_name)
                if not metrics:
                    continue
                self._prune_queue(metrics['requests'], cutoff_time)
                self._prune_queue(metrics['tokens'], cutoff_time)
                if not metrics['requests'] and not metrics['tokens']:
                    self.cache.pop(model_name, None)

    async def add_usage(self, model_name: str, requests: int = 1, tokens: int = 0):
        now = time.time()
        cutoff_time = now - self.default_window
        async with self.lock:
            metrics = self._ensure_model_locked(model_name)
            metrics['requests'].append((now, requests))
            metrics['tokens'].append((now, tokens))

            self._prune_queue(metrics['requests'], cutoff_time)
            self._prune_queue(metrics['tokens'], cutoff_time)

            if self.max_entries > 0:
                while len(metrics['requests']) > self.max_entries:
                    metrics['requests'].popleft()
                while len(metrics['tokens']) > self.max_entries:
                    metrics['tokens'].popleft()

    async def get_current_usage(self, model_name: str, window_seconds: int = None) -> Dict[str, int]:
        if window_seconds is None:
            window_seconds = self.default_window
        cutoff_time = time.time() - window_seconds
        async with self.lock:
            metrics = self.cache.get(model_name)
            if not metrics:
                return {'requests': 0, 'tokens': 0}

            self._prune_queue(metrics['requests'], cutoff_time)
            self._prune_queue(metrics['tokens'], cutoff_time)

            total_requests = sum(value for _, value in metrics['requests'])
            total_tokens = sum(value for _, value in metrics['tokens'])
            return {'requests': total_requests, 'tokens': total_tokens}

def _health_check_candidates(key_info: Dict[str, Any]) -> List[str]:
    source_type = (key_info.get("source_type") or "cli_api_key").lower()

    if source_type == "cli_oauth":
        return [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-pro-preview-05-06",
        ]

    if source_type == "cli_api_key":
        return [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]

    return ["gemini-2.5-flash-lite"]


def _extract_http_exception(exc: HTTPException) -> tuple:
    status_code = exc.status_code

    detail = exc.detail
    if isinstance(detail, dict):
        try:
            message = json.dumps(detail, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive
            message = str(detail)
    else:
        message = str(detail)

    return status_code, message or str(exc)


async def check_gemini_key_health(key_info: Dict[str, Any], db: Optional[Database] = None, timeout: int = 10) -> Dict[str, Any]:
    start_time = time.time()
    if db is None:
        raise HTTPException(status_code=500, detail="Database required for CLI health check")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Hello"}],
            }
        ],
        "generation_config": {
            "temperature": 0.0,
            "top_p": 0.1,
        },
    }

    attempted: List[str] = []
    last_status = None
    last_error = None

    for model_name in _health_check_candidates(key_info):
        attempted.append(model_name)
        try:
            await call_gemini_with_cli_account(
                db,
                key_info,
                payload,
                model_name,
                timeout=float(timeout),
            )
            return {
                "healthy": True,
                "response_time": time.time() - start_time,
                "status_code": 200,
                "error": None,
                "model": model_name,
            }
        except asyncio.TimeoutError:
            last_status = None
            last_error = "Timeout"
            break
        except HTTPException as exc:
            status_code, message = _extract_http_exception(exc)
            last_status = status_code
            last_error = message
            if status_code == 404:
                continue
            break
        except Exception as exc:  # pragma: no cover - defensive
            last_status = None
            last_error = str(exc)
            break

    return {
        "healthy": False,
        "response_time": time.time() - start_time,
        "status_code": last_status,
        "error": last_error,
        "attempted_models": attempted,
    }

async def keep_alive_ping():
    try:
        render_url = os.getenv('RENDER_EXTERNAL_URL')
        target_url = f"{render_url}/wake" if render_url else "http://localhost:8000/wake"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(target_url, timeout=30) as response:
                    logger.info(f"Keep-alive ping {'successful' if response.status == 200 else 'warning'}: {response.status}")
        except ImportError:
            import urllib.request
            with urllib.request.urlopen(target_url, timeout=30) as response:
                logger.info(f"Keep-alive ping {'successful' if response.status == 200 else 'warning'}: {response.status}")
    except Exception as e:
        logger.warning(f"Keep-alive ping failed: {e}")

def init_anti_detection_config(db: Database):
    try:
        db.set_config('anti_detection_enabled', 'true')
        logger.info("Anti-detection system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize anti-detection system: {e}")

async def upload_file_to_gemini(
    db: Database,
    key_info: Dict[str, Any],
    file_content: bytes,
    mime_type: str,
    filename: str,
) -> Optional[str]:
    source_type = (key_info.get('source_type') or 'cli_api_key').lower()
    if source_type not in {'cli_api_key', 'cli_oauth'}:
        logger.error("Attempted to upload file with unsupported key type: %s", source_type)
        return None

    try:
        result = await upload_file_with_cli_account(
            db,
            key_info,
            filename=filename,
            mime_type=mime_type,
            file_content=file_content,
            timeout=float(db.get_config('request_timeout', '60')),
        )
    except HTTPException as exc:
        logger.error("CLI file upload failed: %s", exc.detail)
        return None
    except Exception as exc:  # pragma: no cover
        logger.error("Unexpected CLI file upload failure: %s", exc)
        return None

    file_obj = result.get("file") or {}
    file_uri = file_obj.get("uri") or file_obj.get("name")
    if file_uri:
        logger.info("File uploaded to Gemini successfully: %s", file_uri)
    else:
        logger.error("CLI upload response missing URI: %s", result)
    return file_uri


async def delete_file_from_gemini(
    db: Database,
    key_info: Dict[str, Any],
    file_uri: str,
) -> bool:
    source_type = (key_info.get('source_type') or 'cli_api_key').lower()
    if source_type not in {'cli_api_key', 'cli_oauth'}:
        logger.error("Attempted to delete file with unsupported key type: %s", source_type)
        return False

    try:
        await delete_file_with_cli_account(
            db,
            key_info,
            file_uri=file_uri,
            timeout=float(db.get_config('request_timeout', '60')),
        )
        logger.info("File deleted from Gemini successfully: %s", file_uri)
        return True
    except HTTPException as exc:
        logger.error("CLI file deletion failed: %s", exc.detail)
        return False
    except Exception as exc:  # pragma: no cover
        logger.error("Unexpected CLI file deletion failure: %s", exc)
        return False

def get_actual_model_name(db: Database, request_model: str) -> str:
    candidates = []
    resolved = db.resolve_model_name(request_model)
    if resolved:
        candidates.append(resolved)
    if request_model not in candidates:
        candidates.append(request_model)

    all_configs = db.get_all_model_configs()

    for candidate in candidates:
        for config in all_configs:
            if config['model_name'] == candidate:
                logger.info(f"Found model by model_name: {candidate}")
                return config['model_name']

    for candidate in candidates:
        for config in all_configs:
            if config.get('display_name') == candidate:
                logger.info(
                    "Found model by display_name: '%s', mapping to model_name: '%s'",
                    candidate,
                    config['model_name'],
                )
                return config['model_name']

    default_model = db.get_config('default_model_name', 'gemini-2.5-flash-lite')
    logger.warning(
        "Model '%s' not found by name or display name, falling back to default: %s",
        request_model,
        default_model,
    )
    return default_model

def inject_prompt_to_messages(db: Database, messages: List[ChatMessage]) -> List[ChatMessage]:
    inject_config = db.get_inject_prompt_config()
    if not inject_config['enabled'] or not inject_config['content']: return messages
    content = inject_config['content']
    position = inject_config['position']
    new_messages = copy.deepcopy(messages)

    def _prepend_text(message: ChatMessage, text: str):
        if isinstance(message.content, str):
            message.content = f"{text}\n\n{message.content}" if message.content else text
        elif isinstance(message.content, list):
            message.content = [{"type": "text", "text": text}] + list(message.content)
        else:
            message.content = f"{text}\n\n{message.get_text_content()}"

    def _append_text(message: ChatMessage, text: str):
        if isinstance(message.content, str):
            message.content = f"{message.content}\n\n{text}" if message.content else text
        elif isinstance(message.content, list):
            message.content = list(message.content) + [{"type": "text", "text": text}]
        else:
            message.content = f"{message.get_text_content()}\n\n{text}"

    if position == 'system':
        system_msg = next((msg for msg in new_messages if msg.role == 'system'), None)
        if system_msg:
            _prepend_text(system_msg, content)
        else:
            new_messages.insert(0, ChatMessage(role='system', content=content))
    elif position == 'user_prefix':
        user_msg = next((msg for msg in new_messages if msg.role == 'user'), None)
        if user_msg:
            _prepend_text(user_msg, content)
    elif position == 'user_suffix':
        user_msg = next((msg for msg in reversed(new_messages) if msg.role == 'user'), None)
        if user_msg:
            _append_text(user_msg, content)
    anti_truncation_cfg = db.get_anti_truncation_config()
    if anti_truncation_cfg.get('enabled'):
        user_msg = next((msg for msg in reversed(new_messages) if msg.role == 'user'), None)
        if user_msg:
            suffix = "请以 [finish] 结尾"
            _append_text(user_msg, suffix)
    return new_messages

def get_thinking_config(db: Database, request: ChatCompletionRequest, model_name: str) -> Dict:
    thinking_config = {}
    
    # 1. Check if thinking is globally disabled
    global_thinking_enabled = db.get_config('thinking_enabled', 'true').lower() == 'true'
    if not global_thinking_enabled:
        return {"thinkingBudget": 0}

    # 2. Determine budget and include_thoughts based on priority
    budget = None
    include_thoughts = None

    # Priority 1: User-provided thinking_config.thinking_budget
    if request.thinking_config and request.thinking_config.thinking_budget is not None:
        budget = request.thinking_config.thinking_budget

    if request.thinking_config and request.thinking_config.include_thoughts is not None:
        include_thoughts = request.thinking_config.include_thoughts

    # Priority 2: User-provided reasoning_effort (if budget is not already set)
    # The logic in api_models.py handles "low" and "medium" by creating a thinking_config,
    # so we only need to handle "high" here as a special case.
    if budget is None and request.reasoning_effort == "high":
        if 'pro' in request.model:
            budget = 32768  # Pro max
        else:  # Default to flash max for other models like flash
            budget = 24576  # Flash max

    # Priority 3: Model-level defaults (if budget is still not set)
    if budget is None:
        model_config = db.get_model_config(model_name)
        if model_config:
            budget = model_config.get('default_thinking_budget', -1)
            if include_thoughts is None:
                include_thoughts = bool(model_config.get('include_thoughts_default', 1))

    # Priority 4: Global config from DB (if budget is still not set)
    if budget is None:
        budget = int(db.get_config('thinking_budget', '-1'))

    if include_thoughts is None:
        include_thoughts = db.get_config('include_thoughts', 'false').lower() == 'true'

    # 3. Build the final config dictionary
    if budget == 0:
        return {"thinkingBudget": 0}

    if budget is not None and budget != 0:
        thinking_config["thinkingBudget"] = budget
    
    if include_thoughts:
        thinking_config["includeThoughts"] = include_thoughts

    return thinking_config

def process_multimodal_content(item: Union[Dict[str, Any], ContentPart], file_storage: Dict) -> Optional[Dict]:
    try:
        if isinstance(item, ContentPart):
            item_dict = item.model_dump(exclude_none=True)
        else:
            item_dict = item

        file_data = item_dict.get('file_data') or item_dict.get('fileData')
        inline_data = item_dict.get('inline_data') or item_dict.get('inlineData')
        if hasattr(file_data, "model_dump"):
            file_data = file_data.model_dump(exclude_none=True)
        if hasattr(inline_data, "model_dump"):
            inline_data = inline_data.model_dump(exclude_none=True)
        if inline_data:
            mime_type = inline_data.get('mimeType') or inline_data.get('mime_type')
            data = inline_data.get('data')
            if mime_type and data: return {"inlineData": {"mimeType": mime_type, "data": data}}
        elif file_data:
            mime_type = file_data.get('mimeType') or file_data.get('mime_type')
            file_uri = file_data.get('fileUri') or file_data.get('file_uri')
            if mime_type and file_uri: return {"fileData": {"mimeType": mime_type, "fileUri": file_uri}}
        elif item_dict.get('type') == 'file' and 'file_id' in item_dict:
            file_id = item_dict['file_id']
            if file_id in file_storage:
                file_info = file_storage[file_id]
                if file_info.get('format') == 'inlineData':
                    return {"inlineData": {"mimeType": file_info['mime_type'], "data": file_info['data']}}
                elif file_info.get('format') == 'fileData':
                    if 'gemini_file_uri' in file_info:
                        return {"fileData": {"mimeType": file_info['mime_type'], "fileUri": file_info['gemini_file_uri']}}
                    elif 'file_uri' in file_info:
                        logger.warning(f"Using local file URI for file {file_id}, this may not work with Gemini")
                        return {"fileData": {"mimeType": file_info['mime_type'], "fileUri": file_info['file_uri']}}
            else: logger.warning(f"File ID {file_id} not found in storage")
        if item_dict.get('type') == 'image_url' and 'image_url' in item_dict:
            image_url = item_dict['image_url'].get('url', '')
            if image_url.startswith('data:'):
                try:
                    header, data = image_url.split(',', 1)
                    mime_type = header.split(';')[0].split(':')[1]
                    return {"inlineData": {"mimeType": mime_type, "data": data}}
                except Exception as e: logger.warning(f"Failed to parse data URL: {e}")
            else: logger.warning("HTTP URLs not supported for images, use file upload instead")
        logger.warning(f"Unsupported multimodal content format: {item_dict}")
        return None
    except Exception as e:
        logger.error(f"Error processing multimodal content: {e}")
        return None

def estimate_token_count(text: str) -> int:
    return len(text) // 4

def should_apply_anti_detection(db: Database, request: ChatCompletionRequest, anti_detection_injector: GeminiAntiDetectionInjector, enable_anti_detection: bool = True) -> bool:
    if not enable_anti_detection or not db.get_config('anti_detection_enabled', 'true').lower() == 'true': return False
    disable_for_tools = db.get_config('anti_detection_disable_for_tools', 'true').lower() == 'true'
    if disable_for_tools and (request.tools or request.tool_choice):
        logger.info("Anti-detection disabled for tool calls")
        return False
    token_threshold = int(db.get_config('anti_detection_token_threshold', '5000'))
    total_tokens = 0
    for msg in request.messages:
        if isinstance(msg.content, str):
            total_tokens += estimate_token_count(msg.content)
        elif isinstance(msg.content, list):
            total_tokens += sum(estimate_token_count(item.get('text', '')) for item in msg.content if isinstance(item, dict) and item.get('type') == 'text')
    if total_tokens < token_threshold:
        logger.info(f"Anti-detection skipped: token count {total_tokens} below threshold {token_threshold}")
        return False
    logger.info(f"Anti-detection enabled: token count {total_tokens} exceeds threshold {token_threshold}")
    return True

def openai_to_gemini(db: Database, request: ChatCompletionRequest, anti_detection_injector: GeminiAntiDetectionInjector, file_storage: Dict, enable_anti_detection: bool = True) -> Dict:
    contents = []
    tool_declarations = []
    tool_config = None
    
    # 1. Convert OpenAI tools to Gemini FunctionDeclarations
    if request.tools:
        for tool in request.tools:
            if tool.get("type") == "function":
                func_info = tool.get("function", {})
                tool_declarations.append(
                    types.FunctionDeclaration(
                        name=func_info.get("name"),
                        description=func_info.get("description"),
                        parameters=func_info.get("parameters")
                    )
                )
    
    gemini_tools = [types.Tool(function_declarations=tool_declarations)] if tool_declarations else None

    # 2. Convert OpenAI tool_choice to Gemini ToolConfig
    if request.tool_choice:
        if isinstance(request.tool_choice, str):
            if request.tool_choice == "none":
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingMode.NONE)
                )
            elif request.tool_choice == "auto":
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingMode.AUTO)
                )
        elif isinstance(request.tool_choice, dict):
            func_name = request.tool_choice.get("function", {}).get("name")
            if func_name:
                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingMode.ANY,
                        allowed_function_names=[func_name]
                    )
                )

    # 3. Process messages and handle tool calls/responses
    anti_detection_enabled = should_apply_anti_detection(db, request, anti_detection_injector, enable_anti_detection)
    
    # Track tool_call_id to function_name mapping
    tool_call_id_to_name = {}

    for msg in request.messages:
        # First pass to find assistant tool calls and build the map
        if msg.role == "assistant" and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("type") == "function":
                    tool_call_id_to_name[tool_call.get("id")] = tool_call.get("function", {}).get("name")

    for msg in request.messages:
        # In Gemini, 'tool' role messages are sent as 'user' role
        role = "user" if msg.role in ["system", "user", "tool"] else "model"
        parts = []

        if msg.role == "tool":
            func_name = tool_call_id_to_name.get(msg.tool_call_id)
            if func_name:
                # Ensure content is a serializable dict
                try:
                    response_content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                except json.JSONDecodeError:
                    response_content = {"content": msg.content}

                parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=func_name,
                        response=response_content
                    )
                ))
            else:
                logger.warning(f"Could not find function name for tool_call_id: {msg.tool_call_id}")
                continue # Skip this tool message if we can't map it
        
        elif msg.role == "assistant" and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # This is a tool call request from the model, convert to Gemini's FunctionCall
            for tool_call in msg.tool_calls:
                if tool_call.get("type") == "function":
                    func = tool_call.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {} # Default to empty dict if arguments are not valid JSON
                    parts.append(types.Part(
                        function_call=types.FunctionCall(
                            name=func.get("name"),
                            args=args
                        )
                    ))
        
        elif isinstance(msg.content, str):
            text_content = anti_detection_injector.inject_symbols(msg.content) if anti_detection_enabled and msg.role == 'user' else msg.content
            parts.append({"text": f"[System]: {text_content}" if msg.role == "system" else text_content})
        
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, str):
                    text_content = anti_detection_injector.inject_symbols(item) if anti_detection_enabled and msg.role == 'user' else item
                    parts.append({"text": text_content})
                elif isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_content = anti_detection_injector.inject_symbols(item.get('text', '')) if anti_detection_enabled and msg.role == 'user' else item.get('text', '')
                        parts.append({"text": text_content})
                    elif item.get('type') in ['image', 'image_url', 'audio', 'video', 'document']:
                        multimodal_part = process_multimodal_content(item, file_storage)
                        if multimodal_part: parts.append(multimodal_part)
                elif isinstance(item, ContentPart):
                    if item.type in (None, 'text', 'input_text'):
                        text_value = item.text or ""
                        if text_value:
                            text_value = anti_detection_injector.inject_symbols(text_value) if anti_detection_enabled and msg.role == 'user' else text_value
                            parts.append({"text": text_value})
                    else:
                        multimodal_part = process_multimodal_content(item, file_storage)
                        if multimodal_part:
                            parts.append(multimodal_part)

        if parts:
            contents.append({"role": role, "parts": parts})

    thinking_config = get_thinking_config(db, request, request.model)
    thinking_cfg_obj = types.ThinkingConfig(**thinking_config) if thinking_config else None
    
    generation_config = types.GenerateContentConfig(
        temperature=request.temperature,
        top_p=request.top_p,
        candidate_count=request.n,
        thinking_config=thinking_cfg_obj,
        max_output_tokens=request.max_tokens,
        stop_sequences=request.stop,
    )
    
    gemini_request = {
        "contents": contents,
        "generation_config": generation_config
    }
    if gemini_tools:
        gemini_request["tools"] = gemini_tools
    if tool_config:
        gemini_request["tool_config"] = tool_config
        
    return gemini_request

def extract_thoughts_and_content(gemini_response: Dict) -> tuple[str, str, List[Dict]]:
    thoughts, content = "", ""
    tool_calls = []
    
    # Assuming we only process the first candidate for tool calls for simplicity
    candidate = gemini_response.get("candidates", [{}])[0]
    
    if candidate and candidate.get("content", {}).get("parts"):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part and part["text"]:
                if part.get("thought", False):
                    thoughts += part["text"]
                else:
                    content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name"),
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False)
                    }
                })
            
    return thoughts.strip(), content.strip(), tool_calls

def gemini_to_openai(gemini_response: Dict, request: ChatCompletionRequest, usage_info: Dict = None) -> Dict:
    choices = []
    thoughts, content, tool_calls = extract_thoughts_and_content(gemini_response)
    
    # We'll process based on the first candidate
    candidate = gemini_response.get("candidates", [{}])[0]
    finish_reason = map_finish_reason(candidate.get("finishReason", "STOP"))

    message = {"role": "assistant"}
    
    if content:
        message["content"] = content
    else:
        # Per OpenAI spec, content is null if tool_calls are present
        message["content"] = None

    if thoughts:
        message["reasoning"] = thoughts
        
    if tool_calls:
        message["tool_calls"] = tool_calls

    choices.append({
        "index": 0,
        "message": message,
        "finish_reason": finish_reason
    })
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "created": int(time.time()),
        "model": request.model, "choices": choices,
        "usage": usage_info or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

def map_finish_reason(gemini_reason: str) -> str:
    return {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "TOOL_CALL": "tool_calls",
        "OTHER": "stop"
    }.get(gemini_reason, "stop")

def validate_file_for_gemini(file_content: bytes, mime_type: str, filename: str, supported_mime_types: set, max_file_size: int, max_inline_size: int) -> Dict[str, Any]:
    file_size = len(file_content)
    if mime_type not in supported_mime_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type}")
    if file_size > max_file_size:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {max_file_size // (1024 * 1024)}MB")
    return {"size": file_size, "mime_type": mime_type, "use_inline": file_size <= max_inline_size, "filename": filename}
