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
import io
import itertools
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from functools import lru_cache

from google import genai
from google.genai import types
from fastapi import HTTPException

from database import Database
from api_models import ChatCompletionRequest, ChatMessage

# 配置日志
logger = logging.getLogger(__name__)

# GenAI Client 缓存
@lru_cache(maxsize=32)
def get_cached_client(api_key: str) -> genai.Client:
    """按 key 复用 google-genai Client，减小握手 & 日志开销"""
    return genai.Client(api_key=api_key)

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
    def __init__(self, max_entries: int = 10000):
        self.cache: Dict[str, Dict[str, List[tuple]]] = {}
        self.max_entries = max_entries
        self.lock = asyncio.Lock()

    async def cleanup_expired(self, window_seconds: int = 60):
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        async with self.lock:
            for model_name in list(self.cache.keys()):
                if model_name in self.cache:
                    self.cache[model_name]['requests'] = [(t, v) for t, v in self.cache[model_name]['requests'] if t > cutoff_time]
                    self.cache[model_name]['tokens'] = [(t, v) for t, v in self.cache[model_name]['tokens'] if t > cutoff_time]

    async def add_usage(self, model_name: str, requests: int = 1, tokens: int = 0):
        async with self.lock:
            if model_name not in self.cache: self.cache[model_name] = {'requests': [], 'tokens': []}
            current_time = time.time()
            self.cache[model_name]['requests'].append((current_time, requests))
            self.cache[model_name]['tokens'].append((current_time, tokens))

    async def get_current_usage(self, model_name: str, window_seconds: int = 60) -> Dict[str, int]:
        async with self.lock:
            if model_name not in self.cache: return {'requests': 0, 'tokens': 0}
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            self.cache[model_name]['requests'] = [(t, v) for t, v in self.cache[model_name]['requests'] if t > cutoff_time]
            self.cache[model_name]['tokens'] = [(t, v) for t, v in self.cache[model_name]['tokens'] if t > cutoff_time]
            total_requests = sum(v for _, v in self.cache[model_name]['requests'])
            total_tokens = sum(v for _, v in self.cache[model_name]['tokens'])
            return {'requests': total_requests, 'tokens': total_tokens}

async def check_gemini_key_health(api_key: str, timeout: int = 10) -> Dict[str, Any]:
    start_time = time.time()
    try:
        client = get_cached_client(api_key)
        await asyncio.wait_for(
            client.aio.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents="Hello",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            ),
            timeout=timeout
        )
        return {"healthy": True, "response_time": time.time() - start_time, "status_code": 200, "error": None}
    except asyncio.TimeoutError:
        return {"healthy": False, "response_time": timeout, "status_code": None, "error": "Timeout"}
    except Exception as e:
        status_code = None
        error_message = str(e)
        # Try to extract HTTP status code from the exception message
        import re
        match = re.match(r"(\d{3})", error_message)
        if match:
            status_code = int(match.group(1))
        
        return {"healthy": False, "response_time": time.time() - start_time, "status_code": status_code, "error": error_message}

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

async def upload_file_to_gemini(file_content: bytes, mime_type: str, filename: str, gemini_key: str) -> Optional[str]:
    try:
        client = get_cached_client(gemini_key)
        file_stream = io.BytesIO(file_content)
        upload_result = await client.aio.files.upload(
            file=file_stream,
            config={"mimeType": mime_type, "displayName": filename, "name": f"files/{uuid.uuid4().hex}_{filename}"}
        )
        file_uri = getattr(upload_result, "uri", None)
        if file_uri:
            logger.info(f"File uploaded to Gemini successfully: {file_uri}")
            return file_uri
        else:
            logger.error("No URI returned from google-genai upload result")
            return None
    except Exception as e:
        logger.error(f"Error uploading file to Gemini: {str(e)}")
        return None

async def delete_file_from_gemini(file_uri: str, gemini_key: str) -> bool:
    try:
        client = get_cached_client(gemini_key)
        await client.aio.files.delete(name=file_uri)
        logger.info(f"File deleted from Gemini successfully: {file_uri}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file from Gemini: {str(e)}")
        return False

def get_actual_model_name(db: Database, request_model: str) -> str:
    # 首先尝试直接按 model_name 匹配
    all_configs = db.get_all_model_configs()
    for config in all_configs:
        if config['model_name'] == request_model:
            logger.info(f"Found model by model_name: {request_model}")
            return request_model

    # 如果找不到，再尝试按 display_name 匹配
    for config in all_configs:
        if config['display_name'] == request_model:
            logger.info(f"Found model by display_name: '{request_model}', mapping to model_name: '{config['model_name']}'")
            return config['model_name']

    # 如果都找不到，返回默认模型
    default_model = db.get_config('default_model_name', 'gemini-2.5-flash-lite')
    logger.warning(f"Model '{request_model}' not found by model_name or display_name, falling back to default: {default_model}")
    return default_model

def inject_prompt_to_messages(db: Database, messages: List[ChatMessage]) -> List[ChatMessage]:
    inject_config = db.get_inject_prompt_config()
    if not inject_config['enabled'] or not inject_config['content']: return messages
    content = inject_config['content']
    position = inject_config['position']
    new_messages = messages.copy()
    if position == 'system':
        system_msg = next((msg for msg in new_messages if msg.role == 'system'), None)
        if system_msg:
            system_msg.content = f"{content}\n\n{system_msg.get_text_content()}"
        else:
            new_messages.insert(0, ChatMessage(role='system', content=content))
    elif position == 'user_prefix':
        user_msg = next((msg for msg in new_messages if msg.role == 'user'), None)
        if user_msg: user_msg.content = f"{content}\n\n{user_msg.get_text_content()}"
    elif position == 'user_suffix':
        user_msg = next((msg for msg in reversed(new_messages) if msg.role == 'user'), None)
        if user_msg: user_msg.content = f"{user_msg.get_text_content()}\n\n{content}"
    anti_truncation_cfg = db.get_anti_truncation_config()
    if anti_truncation_cfg.get('enabled'):
        user_msg = next((msg for msg in reversed(new_messages) if msg.role == 'user'), None)
        if user_msg:
            suffix = "请以 [finish] 结尾"
            if isinstance(user_msg.content, str): user_msg.content = f"{user_msg.content}\n\n{suffix}"
            elif isinstance(user_msg.content, list): user_msg.content.append(suffix)
    return new_messages

def get_thinking_config(db: Database, request: ChatCompletionRequest) -> Dict:
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
    
    # Priority 2: User-provided reasoning_effort (if budget is not already set)
    # The logic in api_models.py handles "low" and "medium" by creating a thinking_config,
    # so we only need to handle "high" here as a special case.
    if budget is None and request.reasoning_effort == "high":
        if 'pro' in request.model:
            budget = 32768  # Pro max
        else:  # Default to flash max for other models like flash
            budget = 24576  # Flash max

    # Priority 3: Global config from DB (if budget is still not set)
    if budget is None:
        budget = int(db.get_config('thinking_budget', '-1'))

    # Determine include_thoughts (user request > global config)
    if request.thinking_config and request.thinking_config.include_thoughts is not None:
        include_thoughts = request.thinking_config.include_thoughts
    else:
        include_thoughts = db.get_config('include_thoughts', 'false').lower() == 'true'

    # 3. Build the final config dictionary
    if budget is not None and budget != 0:
        thinking_config["thinkingBudget"] = budget
    
    if include_thoughts:
        thinking_config["includeThoughts"] = include_thoughts

    return thinking_config

def process_multimodal_content(item: Dict, file_storage: Dict) -> Optional[Dict]:
    try:
        file_data = item.get('file_data') or item.get('fileData')
        inline_data = item.get('inline_data') or item.get('inlineData')
        if inline_data:
            mime_type = inline_data.get('mimeType') or inline_data.get('mime_type')
            data = inline_data.get('data')
            if mime_type and data: return {"inlineData": {"mimeType": mime_type, "data": data}}
        elif file_data:
            mime_type = file_data.get('mimeType') or file_data.get('mime_type')
            file_uri = file_data.get('fileUri') or file_data.get('file_uri')
            if mime_type and file_uri: return {"fileData": {"mimeType": mime_type, "fileUri": file_uri}}
        elif item.get('type') == 'file' and 'file_id' in item:
            file_id = item['file_id']
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
        if item.get('type') == 'image_url' and 'image_url' in item:
            image_url = item['image_url'].get('url', '')
            if image_url.startswith('data:'):
                try:
                    header, data = image_url.split(',', 1)
                    mime_type = header.split(';')[0].split(':')[1]
                    return {"inlineData": {"mimeType": mime_type, "data": data}}
                except Exception as e: logger.warning(f"Failed to parse data URL: {e}")
            else: logger.warning("HTTP URLs not supported for images, use file upload instead")
        logger.warning(f"Unsupported multimodal content format: {item}")
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

        if parts:
            contents.append({"role": role, "parts": parts})

    thinking_config = get_thinking_config(db, request)
    thinking_cfg_obj = types.ThinkingConfig(**thinking_config) if thinking_config else None
    
    generation_config = types.GenerateContentConfig(
        temperature=request.temperature, top_p=request.top_p, candidate_count=request.n,
        thinking_config=thinking_cfg_obj, max_output_tokens=request.max_tokens,
        stop_sequences=request.stop,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
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
