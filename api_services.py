# api_services.py
import asyncio
import json
import time
import uuid
import logging
import os
import copy
import itertools
import re
from typing import Coroutine, Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime
from zoneinfo import ZoneInfo
import httpx
from urllib.parse import quote_plus

from google.genai import types
from fastapi import HTTPException

from database import Database
from api_models import (ChatCompletionRequest, ChatMessage, EmbeddingRequest, EmbeddingResponse, EmbeddingData, EmbeddingUsage,
                          GeminiEmbeddingRequest, GeminiEmbeddingResponse, EmbeddingValue)
from api_utils import (
    GeminiAntiDetectionInjector,
    map_finish_reason,
    decrypt_response,
    check_gemini_key_health,
    RateLimitCache,
    openai_to_gemini,
)
from cli_auth import (
    call_gemini_with_cli_account,
    embed_with_cli_account,
    stream_gemini_with_cli_account,
    resolve_cli_model_name,
)

logger = logging.getLogger(__name__)

_rr_counter = itertools.count()
_rr_lock = asyncio.Lock()


def _normalize_source_type(key_info: Dict[str, Any]) -> str:
    source_type = (key_info.get('source_type') or 'cli_api_key').lower()
    if source_type == 'api_key':
        return 'cli_api_key'
    return source_type


def _is_cli_key(key_info: Dict[str, Any]) -> bool:
    return _normalize_source_type(key_info) == 'cli_oauth'


def _uses_cli_transport(key_info: Dict[str, Any]) -> bool:
    return _normalize_source_type(key_info) in {'cli_oauth', 'cli_api_key'}


def _get_cli_account_id(key_info: Dict[str, Any]) -> Optional[int]:
    metadata = key_info.get('metadata') or {}
    try:
        return int(metadata.get('cli_account_id')) if metadata.get('cli_account_id') is not None else None
    except (TypeError, ValueError):
        return None

async def update_key_performance_background(db: Database, key_id: int, success: bool, response_time: float, error_type: str = None):
    """
    在后台异步更新key性能指标，并实现熔断器逻辑，不阻塞主请求流程
    """
    try:
        key_info = db.get_gemini_key_by_id(key_id)
        if not key_info:
            return

        # EMA (Exponential Moving Average) a平滑因子
        alpha = 0.1  # 对新数据给予10%的权重

        # 更新EMA指标
        new_ema_success_rate = key_info['ema_success_rate'] * (1 - alpha) + (1 if success else 0) * alpha

        # 仅在成功时更新响应时间EMA
        new_ema_response_time = key_info['ema_response_time']
        if success:
            if key_info['ema_response_time'] == 0:
                 new_ema_response_time = response_time
            else:
                new_ema_response_time = key_info['ema_response_time'] * (1 - alpha) + response_time * alpha

        update_data = {
            "ema_success_rate": new_ema_success_rate,
            "ema_response_time": new_ema_response_time
        }

        current_time = int(time.time())

        if success:
            # 成功则重置失败计数和熔断状态
            update_data["consecutive_failures"] = 0
            update_data["breaker_status"] = "active"
            update_data["health_status"] = "healthy"
        else:
            # --- 熔断器逻辑 ---
            # 熔断窗口设为60秒
            breaker_window = 60
            # 熔断阈值设为2次
            breaker_threshold = 2

            last_failure = key_info.get('last_failure_timestamp', 0)
            consecutive_failures = key_info.get('consecutive_failures', 0)

            if current_time - last_failure < breaker_window:
                consecutive_failures += 1
            else:
                # 超出时间窗口，重置连续失败计数
                consecutive_failures = 1

            update_data["consecutive_failures"] = consecutive_failures
            update_data["last_failure_timestamp"] = current_time

            if consecutive_failures >= breaker_threshold:
                update_data["breaker_status"] = "tripped"
                logger.warning(f"Circuit breaker tripped for key #{key_id} after {consecutive_failures} failures.")

            # --- 区分失败类型 ---
            if error_type == "rate_limit":
                update_data["health_status"] = "rate_limited"
            else:
                update_data["health_status"] = "unhealthy"

            # 安排后台健康检查以实现自动恢复
            failover_config = db.get_failover_config()
            if not failover_config.get('background_health_check', True):
                logger.info("Background health check disabled, skipping schedule")
            else:
                asyncio.create_task(schedule_health_check(db, key_id, failover_config))

        db.update_gemini_key(key_id, **update_data)

    except Exception as e:
        logger.error(f"Background performance update failed for key {key_id}: {e}")


async def schedule_health_check(db: Database, key_id: int, failover_config: Optional[Dict[str, Any]] = None):
    """
    调度后台健康检测任务
    """
    try:
        # 获取配置中的延迟时间
        config = failover_config or db.get_failover_config()
        if not config.get('background_health_check', True):
            logger.info("Background health check disabled, aborting scheduled task")
            return

        delay = config.get('health_check_delay', 5)

        # 延迟指定时间后执行健康检测，避免立即重复检测
        await asyncio.sleep(delay)

        key_info = db.get_gemini_key_by_id(key_id)
        if key_info and key_info.get('status') == 1:  # 只检测激活的key
            health_result = await check_gemini_key_health(key_info, db)

            # 更新健康状态
            db.update_key_performance(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            # 记录健康检测历史
            db.record_daily_health_status(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            status = "healthy" if health_result['healthy'] else "unhealthy"
            logger.info(f"Background health check for key #{key_id}: {status}")

    except Exception as e:
        logger.error(f"Background health check failed for key {key_id}: {e}")


async def log_usage_background(db: Database, gemini_key_id: int, user_key_id: int, model_name: str, status: str, requests: int, tokens: int):
    """
    在后台异步记录使用量，不阻塞主请求流程
    """
    try:
        db.log_usage(gemini_key_id, user_key_id, model_name, status, requests, tokens)
    except Exception as e:
        logger.error(f"Background usage logging failed: {e}")


async def review_prompt_with_flashlite(
        db: Database,
        rate_limiter: RateLimitCache,
        original_request: ChatCompletionRequest,
        user_key_info: Dict,
        anti_detection: GeminiAntiDetectionInjector,
) -> Dict[str, Any]:
    """
    使用 gemini-2.5-flash-lite 对用户输入进行预审，判断是否需要联网搜索，以及是否应在搜索关键词中附加当前（UTC+08:00）时间。
    返回结构：{"should_search": bool, "append_current_time": bool, "search_query": Optional[str], "analysis": str}
    """
    default_decision = {
        "should_search": False,
        "append_current_time": False,
        "search_query": None,
        "analysis": ""
    }

    try:
        if not original_request.messages:
            return default_decision

        # 获取最近的对话上下文（最多 6 条）
        recent_messages = original_request.messages[-6:]
        conversation_blocks = []
        last_user_text = ""
        for msg in recent_messages:
            text_content = msg.get_text_content() if hasattr(msg, "get_text_content") else str(msg.content)
            if not text_content:
                continue
            if msg.role == "user":
                last_user_text = text_content
            conversation_blocks.append(f"{msg.role.upper()}: {text_content}")

        if not last_user_text:
            return default_decision

        conversation_text = "\n".join(conversation_blocks)
        current_time_str = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")

        system_instruction = (
            "你是一个负责请求前安全审查与策略规划的助手。"
            "请根据对话内容判断是否需要联网搜索最新信息，以及是否应在搜索关键词中附加当前的北京时间（UTC+08:00）。"
            "必须输出 JSON，对象字段如下：\n"
            "- should_search: 布尔值，是否需要触发联网搜索以获取实时资料；\n"
            "- search_query: 字符串，若需要搜索则给出建议的搜索主题，否则为 null；\n"
            "- append_current_time: 布尔值，若为 true 表示应在搜索关键词中追加当前北京时间；\n"
            "- analysis: 字符串，简要说明判断依据。"
        )

        user_prompt = (
            f"当前时间（北京时间）为 {current_time_str}。\n"
            f"以下是最近的对话：\n{conversation_text}\n"
            "请按照要求返回 JSON。不要添加额外说明或代码块标记。"
        )

        review_request = ChatCompletionRequest(
            model="gemini-2.5-flash-lite",
            messages=[
                ChatMessage(role="system", content=system_instruction),
                ChatMessage(role="user", content=user_prompt)
            ],
            temperature=0.1,
            top_p=0.1,
            max_tokens=256
        )

        gemini_request = openai_to_gemini(db, review_request, anti_detection, {}, enable_anti_detection=False)

        response = await make_request_with_fast_failover(
            db,
            rate_limiter,
            gemini_request,
            review_request,
            "gemini-2.5-flash-lite",
            user_key_info,
            _internal_call=True
        )

        ai_text = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not ai_text:
            return default_decision

        json_candidate = ai_text
        if "```" in json_candidate:
            # 清除可能的代码块包裹
            json_candidate = json_candidate.split("```", 1)[1]
            if "```" in json_candidate:
                json_candidate = json_candidate.split("```", 1)[0]
        json_candidate = json_candidate.strip()
        if not json_candidate.startswith("{"):
            start = json_candidate.find("{")
            end = json_candidate.rfind("}")
            if start != -1 and end != -1:
                json_candidate = json_candidate[start:end + 1]

        if not json_candidate:
            logger.info("Pre-input review returned empty response; using default decision")
            return default_decision

        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError as decode_error:
            preview = json_candidate[:200]
            logger.warning(
                "Pre-input review returned non-JSON payload (preview: %s): %s",
                preview,
                decode_error,
            )
            return default_decision

        decision = default_decision.copy()
        decision["should_search"] = bool(parsed.get("should_search"))
        decision["append_current_time"] = bool(parsed.get("append_current_time"))
        decision["analysis"] = str(parsed.get("analysis", ""))

        search_query = parsed.get("search_query")
        if isinstance(search_query, str) and search_query.strip():
            decision["search_query"] = search_query.strip()

        logger.info(
            "Pre-input review result: should_search=%s, append_current_time=%s, search_query=%s",
            decision["should_search"],
            decision["append_current_time"],
            decision["search_query"] or ""
        )
        return decision

    except Exception as e:
        logger.error(f"Failed to execute pre-input review: {e}")
        return default_decision


async def collect_gemini_response_directly(
        db: Database,
        key_info: Dict[str, Any],
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        use_stream: bool = True,
        timeout_seconds: Optional[float] = None,
        _internal_call: bool = False
) -> Dict:
    """
    从Google API收集完整响应
    """
    normalized_model = resolve_cli_model_name(db, model_name)
    url = f"https://generativelanguage.googleapis.com/v1beta/{normalized_model}:streamGenerateContent?alt=sse"

    # 确定超时时间
    if timeout_seconds is None:
        has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
        is_fast_failover = await should_use_fast_failover(db)
        if has_tool_calls:
            timeout = 60.0
        elif is_fast_failover:
            timeout = 60.0
        else:
            timeout = float(db.get_config('request_timeout', '60'))
    else:
        timeout = timeout_seconds

    if not _uses_cli_transport(key_info):
        raise HTTPException(status_code=503, detail="Non-CLI Gemini keys are no longer supported")

    if use_stream:
        logger.info("CLI transport uses buffered mode for direct collection; forcing non-stream mode")
        use_stream = False

    logger.info(f"Starting direct collection from: {url}")

    complete_content = ""
    thinking_content = ""
    total_tokens = 0
    finish_reason = "stop"
    processed_lines = 0
    tool_calls: List[Dict[str, Any]] = []

    def _process_part(part: Dict[str, Any]) -> None:
        nonlocal complete_content, thinking_content, total_tokens, tool_calls

        text = part.get("text", "")
        if text:
            total_tokens += len(text.split())
            if part.get("thought", False):
                thinking_content += text
            else:
                complete_content += text
            return

        function_call = part.get("functionCall") or part.get("function_call")
        if function_call:
            arguments = function_call.get("args")
            if arguments is None:
                arguments = function_call.get("arguments")

            if isinstance(arguments, (dict, list)):
                try:
                    arguments_str = json.dumps(arguments, ensure_ascii=False)
                except Exception:
                    arguments_str = str(arguments)
            else:
                arguments_str = str(arguments) if arguments is not None else "{}"

            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": function_call.get("name", ""),
                    "arguments": arguments_str or "{}"
                }
            })

    # 防截断相关变量
    anti_trunc_cfg = db.get_anti_truncation_config() if hasattr(db, 'get_anti_truncation_config') else {'enabled': False}
    full_response = ""
    saw_finish_tag = False
    start_time = time.time()

    try:
        data = await call_gemini_with_cli_account(
            db,
            key_info,
            gemini_request,
            model_name,
            timeout=float(timeout),
        )
        response_time = time.time() - start_time
        asyncio.create_task(update_key_performance_background(db, key_id, True, response_time))
        account_id = _get_cli_account_id(key_info)
        if account_id:
            db.touch_cli_account(account_id)

        for candidate in data.get("candidates", []):
            finish_reason_raw = candidate.get("finishReason", "stop")
            finish_reason = map_finish_reason(finish_reason_raw) if finish_reason_raw else "stop"
            for part in candidate.get("content", {}).get("parts", []):
                _process_part(part)

    except asyncio.TimeoutError as e:
        logger.warning(f"Direct request timeout/connection error: {str(e)}")
        response_time = time.time() - start_time
        asyncio.create_task(update_key_performance_background(db, key_id, False, response_time))
        raise Exception(f"Direct request failed: {str(e)}") from e
    except Exception as e:
        logger.error(f"Unexpected direct request error: {str(e)}")
        response_time = time.time() - start_time
        asyncio.create_task(update_key_performance_background(db, key_id, False, response_time))
        raise

    # 检查是否收集到内容
    if not complete_content.strip() and not tool_calls:
        logger.error(f"No content collected directly. Processed {processed_lines} lines")
        raise HTTPException(
            status_code=502,
            detail="No content received from Google API"
        )

    # Anti-truncation handling for non-stream response
    anti_trunc_cfg = db.get_anti_truncation_config() if hasattr(db, 'get_anti_truncation_config') else {'enabled': False}
    if anti_trunc_cfg.get('enabled') and not _internal_call:
        max_attempts = anti_trunc_cfg.get('max_attempts', 3)
        attempt = 0
        while True:
            trimmed = complete_content.rstrip()
            if trimmed.endswith('[finish]'):
                complete_content = trimmed[:-8].rstrip()
                break
            if attempt >= max_attempts:
                logger.info("Anti-truncation enabled but reached max attempts without [finish].")
                break
            attempt += 1
            logger.info(f"Anti-truncation attempt {attempt}: continue fetching content")
            # 构造新的请求，在末尾追加继续提示
            continuation_request = copy.deepcopy(gemini_request)
            continuation_request['contents'].append({
                "role": "user",
                "parts": [{
                    "text": "继续，请以 [finish] 结尾"
                }]
            })
            try:
                cont_data = await call_gemini_with_cli_account(
                    db,
                    key_info,
                    continuation_request,
                    model_name,
                    timeout=float(timeout),
                )
                for candidate in cont_data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        _process_part(part)
            except Exception as e:
                logger.warning(f"Anti-truncation continuation attempt failed: {e}")
                break

    # 分离思考和内容
    thinking_content_final = thinking_content.strip()
    complete_content_final = complete_content.strip()

    # 计算token使用量
    prompt_tokens = len(str(openai_request.messages).split())
    reasoning_tokens = len(thinking_content_final.split())
    completion_tokens = len(complete_content_final.split())

    # 如果启用了响应解密，则解密内容
    decryption_enabled = db.get_response_decryption_config().get('enabled', False)
    if decryption_enabled and not _internal_call:
        logger.info(f"Decrypting response. Original length: {len(complete_content_final)}")
        final_content = decrypt_response(complete_content_final)
        logger.info(f"Decrypted length: {len(final_content)}")
    else:
        final_content = complete_content_final

    # 构建最终响应
    message_content = final_content if final_content else None
    message = {
        "role": "assistant",
        "content": message_content
    }
    if thinking_content_final:
        message["reasoning"] = thinking_content_final
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }
    if reasoning_tokens > 0:
        usage["reasoning_tokens"] = reasoning_tokens
        usage["total_tokens"] += reasoning_tokens

    openai_response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": openai_request.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else finish_reason
        }],
        "usage": usage
    }

    logger.info(f"Successfully collected direct response: {len(final_content)} chars, {completion_tokens} tokens, {reasoning_tokens} reasoning tokens")
    return openai_response


async def make_gemini_request_single_attempt(
        db: Database,
        key_info: Dict[str, Any],
        key_id: int,
        gemini_request: Dict,
        model_name: str,
        timeout: float = 60.0
) -> Dict:
    start_time = time.time()
    if not _uses_cli_transport(key_info):
        raise HTTPException(status_code=503, detail="Non-CLI Gemini keys are no longer supported")

    try:
        response_dict = await call_gemini_with_cli_account(
            db,
            key_info,
            gemini_request,
            model_name,
            timeout=float(timeout),
        )

        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(db, key_id, True, response_time)
        )
        return response_dict

    except asyncio.TimeoutError:
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(db, key_id, False, response_time)
        )
        logger.warning(f"Key #{key_id} timeout after {response_time:.2f}s")
        raise HTTPException(status_code=504, detail="Request timeout")

    except HTTPException:
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(db, key_id, False, response_time)
        )
        raise

    except Exception as e:
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(db, key_id, False, response_time)
        )
        err_msg = str(e)
        if "rate_limit" in err_msg.lower() or "status: 429" in err_msg:
            logger.warning(f"Key #{key_id} is rate-limited (429). Marking as 'rate_limited'.")
            db.update_gemini_key_status(key_id, 'rate_limited')
            raise HTTPException(status_code=429, detail="Rate limited")
        logger.error(f"Key #{key_id} request error: {err_msg}")
        raise HTTPException(status_code=500, detail=err_msg)


async def make_request_with_fast_failover(
        db: Database,
        rate_limiter: RateLimitCache,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        max_key_attempts: int = None,
        _internal_call: bool = False
) -> Dict:
    """
    快速故障转移请求处理
    """
    available_keys = db.get_available_gemini_keys()

    # 防御性检查：确保 available_keys 不为 None
    if available_keys is None:
        logger.error("get_available_gemini_keys() returned None in fast failover")
        raise HTTPException(
            status_code=503,
            detail="Database error: unable to retrieve API keys"
        )

    if not available_keys:
        logger.error("No available keys for request")
        raise HTTPException(
            status_code=503,
            detail="No available API keys"
        )

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting fast failover with up to {max_key_attempts} key attempts for model {model_name}")

    failed_keys = []
    last_error = None

    track_usage = bool(user_key_info) and not _internal_call

    for attempt in range(max_key_attempts):
        try:
            # 选择下一个可用的key（排除已失败的）
            selection_result = await select_gemini_key_and_check_limits(
                db,
                rate_limiter,
                model_name,
                excluded_keys=set(failed_keys)
            )

            # 增强的空值检查
            if selection_result is None:
                logger.warning(f"select_gemini_key_and_check_limits returned None on attempt {attempt + 1}")
                break

            if 'key_info' not in selection_result:
                logger.error(f"Invalid selection_result format on attempt {attempt + 1}: missing 'key_info'")
                break

            key_info = selection_result['key_info']
            logger.info(f"Fast failover attempt {attempt + 1}: Using key #{key_info['id']}")

            # ====== 计算 should_stream_to_gemini ======
            stream_to_gemini_mode = db.get_stream_to_gemini_mode_config().get('mode', 'auto')
            has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
            if has_tool_calls:
                should_stream_to_gemini = False
            elif stream_to_gemini_mode == 'stream':
                should_stream_to_gemini = True
            elif stream_to_gemini_mode == 'non_stream':
                should_stream_to_gemini = False
            else:
                should_stream_to_gemini = True

            try:
                # 确定超时时间：工具调用或快速响应模式使用60秒，其他使用配置值
                has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
                is_fast_failover = await should_use_fast_failover(db)
                if has_tool_calls:
                    timeout_seconds = 60.0  # 工具调用强制60秒超时
                    logger.info("Using extended 60s timeout for tool calls")
                elif is_fast_failover:
                    timeout_seconds = 60.0  # 快速响应模式使用60秒超时
                    logger.info("Using extended 60s timeout for fast response mode")
                else:
                    timeout_seconds = float(db.get_config('request_timeout', '60'))

                # 从Google API收集完整响应
                logger.info(f"Using direct collection for non-streaming request with key #{key_info['id']}")
                if _is_cli_key(key_info):
                    should_stream_to_gemini = False

                # 收集响应
                response = await collect_gemini_response_directly(
                    db,
                    key_info,
                    key_info['id'],
                    gemini_request,
                    openai_request,
                    model_name,
                    use_stream=should_stream_to_gemini,
                    timeout_seconds=timeout_seconds,
                    _internal_call=_internal_call
                )

                logger.info(f"✅ Request successful with key #{key_info['id']} on attempt {attempt + 1}")

                # 从响应中获取token使用量
                usage = response.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                reasoning_tokens = usage.get('reasoning_tokens', 0)
                total_tokens = usage.get(
                    'total_tokens',
                    prompt_tokens + completion_tokens + reasoning_tokens
                )

                # 记录使用量
                if track_usage:
                    # 在后台记录使用量，不阻塞响应
                    asyncio.create_task(
                        log_usage_background(
                            db,
                            key_info['id'],
                            user_key_info['id'],
                            model_name,
                            'success',
                            1,
                            total_tokens
                        )
                    )

                # 更新速率限制
                if not _internal_call:
                    await rate_limiter.add_usage(model_name, 1, total_tokens)
                return response

            except HTTPException as e:
                failed_keys.append(key_info['id'])
                last_error = e

                logger.warning(f"❌ Key #{key_info['id']} failed: {e.detail}")

                # 记录失败的使用量
                if track_usage:
                    asyncio.create_task(
                        log_usage_background(
                            db,
                            key_info['id'],
                            user_key_info['id'],
                            model_name,
                            'failure',
                            1,
                            0
                        )
                    )

                if not _internal_call:
                    await rate_limiter.add_usage(model_name, 1, 0)

                # 如果是客户端错误（4xx），不继续尝试其他key
                if 400 <= e.status_code < 500:
                    logger.warning(f"Client error {e.status_code}, stopping failover")
                    raise e

                # 服务器错误或网络错误，继续尝试下一个key
                continue

        except Exception as e:
            logger.error(f"Unexpected error during failover attempt {attempt + 1}: {str(e)}")
            last_error = HTTPException(status_code=500, detail=str(e))
            continue

    # 所有key都失败了
    failed_count = len(failed_keys)
    logger.error(f"❌ All {failed_count} attempted keys failed for {model_name}")

    if last_error:
        raise last_error
    else:
        raise HTTPException(
            status_code=503,
            detail=f"All {failed_count} available API keys failed"
        )

async def _stream_cli_transport_response(
        *,
        db: Database,
        key_info: Dict[str, Any],
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        timeout: float,
        usage_collector: Optional[Dict[str, int]] = None,
) -> AsyncGenerator[bytes, None]:
    """通过 Gemini CLI 传输流式返回，并转换为 OpenAI SSE 片段。"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    total_tokens = 0
    reasoning_tokens = 0
    thinking_sent = False
    anti_trunc_cfg = db.get_anti_truncation_config() if hasattr(db, 'get_anti_truncation_config') else {'enabled': False}
    saw_finish_tag = False
    account_id = _get_cli_account_id(key_info)
    start_time = time.time()

    try:
        async for event in stream_gemini_with_cli_account(
            db,
            key_info,
            gemini_request,
            model_name,
            timeout=timeout,
        ):
            for candidate in event.get("candidates", []):
                content = candidate.get("content") or {}
                parts = content.get("parts", [])

                for part in parts:
                    function_call = part.get("functionCall") or part.get("function_call")
                    if function_call:
                        name = function_call.get("name", "")
                        arguments = function_call.get("arguments") or function_call.get("args") or {}
                        if not isinstance(arguments, str):
                            try:
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            except Exception:
                                arguments = str(arguments)

                        tool_chunk = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": openai_request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": arguments or "{}"
                                        }
                                    }]
                                },
                                "finish_reason": None,
                            }]
                        }
                        yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                        continue

                    text = part.get("text")
                    if not text:
                        continue

                    if anti_trunc_cfg.get('enabled'):
                        idx = text.find('[finish]')
                        if idx != -1:
                            text = text[:idx]
                            saw_finish_tag = True

                    is_thought = part.get("thought", False)
                    token_count = len(text.split())
                    total_tokens += token_count
                    if is_thought:
                        reasoning_tokens += token_count

                    if is_thought and not (openai_request.thinking_config and openai_request.thinking_config.include_thoughts):
                        continue

                    if is_thought and not thinking_sent:
                        thinking_header = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": openai_request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": "**Thinking Process:**\n"},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(thinking_header, ensure_ascii=False)}\n\n".encode('utf-8')
                        thinking_sent = True

                    if not is_thought and thinking_sent:
                        response_header = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": openai_request.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": "\n\n**Response:**\n"},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(response_header, ensure_ascii=False)}\n\n".encode('utf-8')
                        thinking_sent = False

                    chunk_data = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": openai_request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode('utf-8')

                finish_reason_raw = candidate.get("finishReason")
                if finish_reason_raw or saw_finish_tag:
                    finish_reason = map_finish_reason(finish_reason_raw) if finish_reason_raw else "stop"
                    finish_chunk = {
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": openai_request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }]
                    }
                    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"

                    response_time = time.time() - start_time
                    asyncio.create_task(update_key_performance_background(db, key_id, True, response_time))
                    if usage_collector is not None:
                        prompt_tokens = usage_collector.get('prompt_tokens', 0)
                        usage_collector['completion_tokens'] = total_tokens
                        usage_collector['reasoning_tokens'] = reasoning_tokens
                        usage_collector['total_tokens'] = prompt_tokens + total_tokens
                    if account_id:
                        db.touch_cli_account(account_id)
                    return

    except HTTPException:
        response_time = time.time() - start_time
        asyncio.create_task(update_key_performance_background(db, key_id, False, response_time))
        raise
    except Exception as exc:  # pragma: no cover - unexpected failures
        response_time = time.time() - start_time
        asyncio.create_task(update_key_performance_background(db, key_id, False, response_time))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 如果循环结束但没有明确的 finish，发送默认完成
    finish_chunk = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": openai_request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
    yield b"data: [DONE]\n\n"

    response_time = time.time() - start_time
    asyncio.create_task(update_key_performance_background(db, key_id, True, response_time))
    if usage_collector is not None:
        prompt_tokens = usage_collector.get('prompt_tokens', 0)
        usage_collector['completion_tokens'] = total_tokens
        usage_collector['reasoning_tokens'] = reasoning_tokens
        usage_collector['total_tokens'] = prompt_tokens + total_tokens
    if account_id:
        db.touch_cli_account(account_id)


async def stream_gemini_response_single_attempt(
        db: Database,
        rate_limiter: RateLimitCache,
        key_info: Dict[str, Any],
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        _internal_call: bool = False,
        usage_collector: Optional[Dict[str, int]] = None
) -> AsyncGenerator[bytes, None]:
    """使用 CLI 传输执行一次流式请求。"""
    if not _uses_cli_transport(key_info):
        raise HTTPException(status_code=503, detail="Non-CLI Gemini keys are no longer supported")

    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover(db)
    if has_tool_calls or is_fast_failover:
        timeout = 60.0
    else:
        timeout = float(db.get_config('request_timeout', '60'))

    logger.info(f"Starting CLI stream request to model: {model_name}")

    key_id = key_info['id']
    prompt_tokens = len(str(openai_request.messages).split())
    if usage_collector is not None:
        usage_collector['prompt_tokens'] = prompt_tokens
        usage_collector['completion_tokens'] = 0
        usage_collector['reasoning_tokens'] = 0
        usage_collector['total_tokens'] = prompt_tokens

    async for chunk in _stream_cli_transport_response(
        db=db,
        key_info=key_info,
        key_id=key_id,
        gemini_request=gemini_request,
        openai_request=openai_request,
        model_name=model_name,
        timeout=timeout,
        usage_collector=usage_collector,
    ):
        yield chunk



async def _keep_alive_generator(task: asyncio.Task) -> AsyncGenerator[bytes, Any]:
    """
    在后台任务执行期间周期性发送 keep-alive 心跳，任务完成后返回结果。
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except asyncio.TimeoutError:
            yield b": keep-alive\n\n"

    yield await task


async def stream_non_stream_keep_alive(
        db: Database,
        rate_limiter: RateLimitCache,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        _internal_call: bool = False
) -> AsyncGenerator[bytes, None]:
    """
    向 Gemini 使用非流式接口，但对客户端保持 SSE 流式格式。
    在等待后端响应时发送 keep-alive，然后一次性返回完整内容。
    """
    async def get_full_response():
        if await should_use_fast_failover(db):
            return await make_request_with_fast_failover(
                db, rate_limiter, gemini_request, openai_request, model_name,
                user_key_info=user_key_info, _internal_call=_internal_call
            )
        else:
            return await make_request_with_failover(
                db, rate_limiter, gemini_request, openai_request, model_name,
                user_key_info=user_key_info, _internal_call=_internal_call
            )

    task = asyncio.create_task(get_full_response())

    try:
        async for result in _keep_alive_generator(task):
            if isinstance(result, bytes):
                yield result  # This is a keep-alive chunk
            else:
                # This is the final complete response
                openai_response = result
                yield f"data: {json.dumps(openai_response, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

    except HTTPException as e:
        error_data = {"error": {"message": e.detail, "code": e.status_code}}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    except Exception as e:
        error_data = {"error": {"message": str(e), "code": 500}}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

# 配置管理函数
async def should_use_fast_failover(db: Database) -> bool:
    """检查是否应该使用快速故障转移"""
    config = db.get_failover_config()
    return config.get('fast_failover_enabled', True)

async def select_gemini_key_and_check_limits(db: Database, rate_limiter: RateLimitCache, model_name: str, excluded_keys: set = None) -> Optional[Dict]:
    """自适应选择可用的Gemini Key并检查模型限制"""
    if excluded_keys is None:
        excluded_keys = set()

    available_keys = db.get_available_gemini_keys()

    # 防御性检查：确保 available_keys 不为 None
    if available_keys is None:
        logger.error("get_available_gemini_keys() returned None")
        return None

    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        logger.warning("No available Gemini keys found after exclusions")
        return None

    model_config = db.get_model_config(model_name)
    if not model_config:
        logger.error(f"Model config not found for: {model_name}")
        return None

    logger.info(
        f"Model {model_name} limits: RPM={model_config['total_rpm_limit']}, TPM={model_config['total_tpm_limit']}, RPD={model_config['total_rpd_limit']}")
    logger.info(f"Available API keys: {len(available_keys)}")

    current_usage = await rate_limiter.get_current_usage(model_name)

    if (current_usage['requests'] >= model_config['total_rpm_limit'] or
            current_usage['tokens'] >= model_config['total_tpm_limit']):
        logger.warning(
            f"Model {model_name} has reached rate limits: requests={current_usage['requests']}/{model_config['total_rpm_limit']}, tokens={current_usage['tokens']}/{model_config['total_tpm_limit']}")
        return None

    day_usage = db.get_usage_stats(model_name, 'day')
    if day_usage['requests'] >= model_config['total_rpd_limit']:
        logger.warning(
            f"Model {model_name} has reached daily request limit: {day_usage['requests']}/{model_config['total_rpd_limit']}")
        return None

    strategy = db.get_config('load_balance_strategy', 'adaptive')

    if strategy == 'round_robin':
        async with _rr_lock:
            idx = next(_rr_counter) % len(available_keys)
            selected_key = available_keys[idx]
    elif strategy == 'least_used':
        # 按总请求数排序
        sorted_keys = sorted(available_keys, key=lambda k: k.get('total_requests', 0))
        selected_key = sorted_keys[0]
    else:  # adaptive strategy
        best_key = None
        best_score = -1.0

        for key_info in available_keys:
            # 使用新的EMA指标
            ema_success_rate = key_info.get('ema_success_rate', 1.0)
            ema_response_time = key_info.get('ema_response_time', 0.0)

            # 响应时间评分，10秒为基准，超过10秒评分为0
            time_score = max(0.0, 1.0 - (ema_response_time / 10.0))

            # 最终评分：成功率权重70%，时间权重30%
            score = ema_success_rate * 0.7 + time_score * 0.3

            # 增加近期失败惩罚
            last_failure = key_info.get('last_failure_timestamp', 0)
            time_since_failure = time.time() - last_failure
            if time_since_failure < 300: # 5分钟内失败过
                penalty = (300 - time_since_failure) / 300  # 惩罚力度随时间减小
                score *= (1 - penalty * 0.5) # 最高惩罚50%的分数

            if score > best_score:
                best_score = score
                best_key = key_info

        selected_key = best_key if best_key else available_keys[0]

    logger.info(f"Selected API key #{selected_key['id']} for model {model_name} (strategy: {strategy})")

    return {
        'key_info': selected_key,
        'model_config': model_config
    }


# 传统故障转移函数 - 使用 google-genai 替代 httpx
async def make_gemini_request_with_retry(
        db: Database,
        key_info: Dict[str, Any],
        key_id: int,
        gemini_request: Dict,
        model_name: str,
        max_retries: int = 3,
        timeout: float = None
) -> Dict:
    """带重试的Gemini API请求，记录性能指标"""
    if timeout is None:
        timeout = float(db.get_config('request_timeout', '60'))

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            if not _uses_cli_transport(key_info):
                raise HTTPException(status_code=503, detail="Non-CLI Gemini keys are no longer supported")

            response = await call_gemini_with_cli_account(
                db,
                key_info,
                gemini_request,
                model_name,
                timeout=float(timeout),
            )
            account_id = _get_cli_account_id(key_info)
            if account_id:
                db.touch_cli_account(account_id)

            response_time = time.time() - start_time
            db.update_key_performance(key_id, True, response_time)
            return response

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="Request timeout")
            logger.warning(f"Request timeout (attempt {attempt + 1}), retrying...")
            await asyncio.sleep(2 ** attempt)
            continue
        except HTTPException as exc:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if exc.status_code in (429, 500) and attempt < max_retries - 1:
                logger.warning(f"HTTP error {exc.status_code} on attempt {attempt + 1}: {exc.detail}")
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt == max_retries - 1:
                error_message = str(e)
                status_code = 500
                if "429" in error_message or "rate limit" in error_message.lower():
                    status_code = 429
                elif "403" in error_message or "permission" in error_message.lower():
                    status_code = 403
                elif "404" in error_message or "not found" in error_message.lower():
                    status_code = 404
                elif "400" in error_message or "invalid" in error_message.lower():
                    status_code = 400

                raise HTTPException(status_code=status_code, detail=error_message)
            logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}, retrying...")
            await asyncio.sleep(2 ** attempt)

    raise HTTPException(status_code=500, detail="Max retries exceeded")


async def make_request_with_failover(
        db: Database,
        rate_limiter: RateLimitCache,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        max_key_attempts: int = None,
        excluded_keys: set = None,
        _internal_call: bool = False
) -> Dict:
    """传统请求处理（保留用于兼容）"""
    if excluded_keys is None:
        excluded_keys = set()

    available_keys = db.get_available_gemini_keys()
    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        logger.error("No available keys for failover")
        raise HTTPException(
            status_code=503,
            detail="No available API keys"
        )

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    else:
        max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting failover with {max_key_attempts} key attempts for model {model_name}")

    # 确定超时时间：工具调用或快速响应模式使用60秒，其他使用配置值
    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover(db)
    if has_tool_calls:
        timeout_seconds = 60.0  # 工具调用强制60秒超时
        logger.info("Using extended 60s timeout for tool calls in traditional failover")
    elif is_fast_failover:
        timeout_seconds = 60.0  # 快速响应模式使用60秒超时
        logger.info("Using extended 60s timeout for fast response mode in traditional failover")
    else:
        timeout_seconds = float(db.get_config('request_timeout', '60'))

    last_error = None
    failed_keys = []

    track_usage = bool(user_key_info) and not _internal_call

    for attempt in range(max_key_attempts):
        try:
            selection_result = await select_gemini_key_and_check_limits(
                db,
                rate_limiter,
                model_name,
                excluded_keys=excluded_keys.union(set(failed_keys))
            )

            if not selection_result:
                logger.warning(f"No more available keys after {attempt} attempts")
                break

            key_info = selection_result['key_info']
            model_config = selection_result['model_config']

            logger.info(f"Attempt {attempt + 1}: Using key #{key_info['id']} for {model_name}")

            # ====== 计算 should_stream_to_gemini ======
            stream_to_gemini_mode = db.get_stream_to_gemini_mode_config().get('mode', 'auto')
            has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
            if has_tool_calls:
                should_stream_to_gemini = False
            elif stream_to_gemini_mode == 'stream':
                should_stream_to_gemini = True
            elif stream_to_gemini_mode == 'non_stream':
                should_stream_to_gemini = False
            else:
                should_stream_to_gemini = True

            try:
                # 直接从Google API收集完整响应（传统故障转移）
                logger.info(f"Using direct collection for non-streaming request with key #{key_info['id']} (traditional failover)")

                # 直接收集响应，避免SSE双重解析
                if _is_cli_key(key_info):
                    should_stream_to_gemini = False

                response = await collect_gemini_response_directly(
                    db,
                    key_info,
                    key_info['id'],
                    gemini_request,
                    openai_request,
                    model_name,
                    use_stream=should_stream_to_gemini,
                    timeout_seconds=timeout_seconds,
                    _internal_call=_internal_call
                )

                logger.info(f"✅ Request successful with key #{key_info['id']} on attempt {attempt + 1}")

                # 从响应中获取token使用量
                usage = response.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                reasoning_tokens = usage.get('reasoning_tokens', 0)
                total_tokens = usage.get(
                    'total_tokens',
                    prompt_tokens + completion_tokens + reasoning_tokens
                )

                if track_usage:
                    db.log_usage(
                        gemini_key_id=key_info['id'],
                        user_key_id=user_key_info['id'],
                        model_name=model_name,
                        status='success',
                        requests=1,
                        tokens=total_tokens
                    )
                    logger.info(
                        f"📊 Logged usage: gemini_key_id={key_info['id']}, user_key_id={user_key_info['id']}, model={model_name}, tokens={total_tokens}")

                if not _internal_call:
                    await rate_limiter.add_usage(model_name, 1, total_tokens)
                return response

            except HTTPException as e:
                failed_keys.append(key_info['id'])
                last_error = e

                db.update_key_performance(key_info['id'], False, 0.0)

                if track_usage:
                    db.log_usage(
                        gemini_key_id=key_info['id'],
                        user_key_id=user_key_info['id'],
                        model_name=model_name,
                        status='failure',
                        requests=1,
                        tokens=0
                    )

                if not _internal_call:
                    await rate_limiter.add_usage(model_name, 1, 0)

                logger.warning(f"❌ Key #{key_info['id']} failed with {e.status_code}: {e.detail}")

                if e.status_code < 500:
                    logger.warning(f"Client error {e.status_code}, stopping failover")
                    raise e

                continue

        except Exception as e:
            logger.error(f"Unexpected error during failover attempt {attempt + 1}: {str(e)}")
            last_error = HTTPException(status_code=500, detail=str(e))
            continue

    failed_count = len(failed_keys)
    logger.error(f"❌ All {failed_count} keys failed for {model_name}")

    if last_error:
        raise last_error
    else:
        raise HTTPException(
            status_code=503,
            detail=f"All {failed_count} available API keys failed"
        )


async def stream_with_failover(
        db: Database,
        rate_limiter: RateLimitCache,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        max_key_attempts: int = None,
        excluded_keys: set = None,
        _internal_call: bool = False
) -> AsyncGenerator[bytes, None]:
    """传统流式响应处理（保留用于兼容）。只支持 CLI 类型的上游。"""
    if excluded_keys is None:
        excluded_keys = set()

    available_keys = db.get_available_gemini_keys()
    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        error_data = {
            'error': {
                'message': 'No available API keys',
                'type': 'service_unavailable',
                'code': 503
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"
        return

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    else:
        max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting stream failover with {max_key_attempts} key attempts for {model_name}")

    failed_keys = []
    track_usage = bool(user_key_info) and not _internal_call

    for attempt in range(max_key_attempts):
        selection_result = await select_gemini_key_and_check_limits(
            db,
            rate_limiter,
            model_name,
            excluded_keys=excluded_keys.union(set(failed_keys))
        )

        if not selection_result:
            break

        key_info = selection_result['key_info']
        if not _uses_cli_transport(key_info):
            logger.warning("Skipping non-CLI key during streaming failover")
            failed_keys.append(key_info['id'])
            continue

        logger.info(f"Stream attempt {attempt + 1}: Using key #{key_info['id']}")

        stream_to_gemini_mode = db.get_stream_to_gemini_mode_config().get('mode', 'auto')
        has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
        if has_tool_calls:
            should_stream_to_gemini = False
        elif stream_to_gemini_mode == 'stream':
            should_stream_to_gemini = True
        elif stream_to_gemini_mode == 'non_stream':
            should_stream_to_gemini = False
        else:
            should_stream_to_gemini = True

        if not should_stream_to_gemini:
            logger.info("Streaming disabled by configuration; falling back to buffered mode")
            async for chunk in stream_non_stream_keep_alive(
                db,
                rate_limiter,
                gemini_request,
                openai_request,
                model_name,
                user_key_info=user_key_info,
                _internal_call=_internal_call,
            ):
                yield chunk
            return

        usage_summary: Dict[str, int] = {}
        try:
            async for chunk in stream_gemini_response_single_attempt(
                db,
                rate_limiter,
                key_info,
                gemini_request,
                openai_request,
                model_name,
                _internal_call=_internal_call,
                usage_collector=usage_summary,
            ):
                yield chunk

            total_tokens = usage_summary.get('total_tokens')
            if total_tokens is None:
                prompt_tokens = usage_summary.get('prompt_tokens', 0)
                completion_tokens = usage_summary.get('completion_tokens', 0)
                reasoning_tokens = usage_summary.get('reasoning_tokens', 0)
                total_tokens = prompt_tokens + completion_tokens + reasoning_tokens

            if track_usage:
                db.log_usage(
                    gemini_key_id=key_info['id'],
                    user_key_id=user_key_info['id'],
                    model_name=model_name,
                    status='success',
                    requests=1,
                    tokens=total_tokens
                )
            return

        except HTTPException as exc:
            failed_keys.append(key_info['id'])
            logger.warning(f"Stream key #{key_info['id']} failed: {exc.detail}")
            if track_usage:
                db.log_usage(
                    gemini_key_id=key_info['id'],
                    user_key_id=user_key_info['id'],
                    model_name=model_name,
                    status='failure',
                    requests=1,
                    tokens=0
                )
            if attempt < max_key_attempts - 1:
                retry_msg = {
                    'error': {
                        'message': f'Key #{key_info["id"]} failed, trying next key...',
                        'type': 'retry_info',
                        'retry_attempt': attempt + 1
                    }
                }
                yield f"data: {json.dumps(retry_msg, ensure_ascii=False)}\n\n".encode('utf-8')
                continue
            break

        except Exception as exc:
            failed_keys.append(key_info['id'])
            logger.error(f"Stream failover error on attempt {attempt + 1}: {exc}")
            if attempt < max_key_attempts - 1:
                continue
            break

    error_data = {
        'error': {
            'message': f'All {len(failed_keys)} available API keys failed',
            'type': 'all_keys_failed',
            'code': 503,
            'failed_keys': failed_keys
        }
    }
    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
    yield b"data: [DONE]\n\n"



async def record_hourly_health_check(db: Database):
    """每小时记录一次健康检测结果"""
    try:
        available_keys = db.get_available_gemini_keys()

        for key_info in available_keys:
            key_id = key_info['id']

            # 执行健康检测
            health_result = await check_gemini_key_health(key_info, db)

            # 记录到历史表
            db.record_daily_health_status(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            # 更新性能指标
            db.update_key_performance(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

        logger.info(f"✅ Hourly health check completed for {len(available_keys)} keys")

    except Exception as e:
        logger.error(f"❌ Hourly health check failed: {e}")


async def auto_cleanup_failed_keys(db: Database):
    """每日自动清理连续异常的API key"""
    try:
        # 获取配置
        cleanup_config = db.get_auto_cleanup_config()

        if not cleanup_config['enabled']:
            logger.info("🔒 Auto cleanup is disabled")
            return

        days_threshold = cleanup_config['days_threshold']
        min_checks_per_day = cleanup_config['min_checks_per_day']

        # 执行自动清理
        removed_keys = db.auto_remove_failed_keys(days_threshold, min_checks_per_day)

        if removed_keys:
            logger.warning(
                f"🗑️ Auto-removed {len(removed_keys)} failed keys after {days_threshold} consecutive unhealthy days:")
            for key in removed_keys:
                logger.warning(f"   - Key #{key['id']}: {key['key']} (failed for {key['consecutive_days']} days)")
        else:
            logger.info(f"✅ No keys need cleanup (threshold: {days_threshold} days)")

    except Exception as e:
        logger.error(f"❌ Auto cleanup failed: {e}")


def delete_unhealthy_keys(db: Database) -> Dict[str, Any]:
    """删除所有异常的Gemini密钥"""
    try:
        unhealthy_keys = db.get_unhealthy_gemini_keys()
        if not unhealthy_keys:
            return {"success": True, "message": "没有发现异常密钥", "deleted_count": 0}

        deleted_count = 0
        for key in unhealthy_keys:
            db.delete_gemini_key(key['id'])
            deleted_count += 1

        logger.info(f"Deleted {deleted_count} unhealthy Gemini keys.")
        return {"success": True, "message": f"成功删除 {deleted_count} 个异常密钥", "deleted_count": deleted_count}
    except Exception as e:
        logger.error(f"Error deleting unhealthy keys: {e}")
        raise HTTPException(status_code=500, detail="删除异常密钥时发生内部错误")


async def cleanup_database_records(db: Database):
    """每日自动清理旧的数据库记录"""
    try:
        logger.info("Starting daily database cleanup...")

        # 清理使用日志
        deleted_logs = db.cleanup_old_logs(days=1)
        logger.info(f"Cleaned up {deleted_logs} old usage log records.")

        # 清理健康检查历史
        deleted_history = db.cleanup_old_health_history(days=1)
        logger.info(f"Cleaned up {deleted_history} old health history records.")

        logger.info("✅ Daily database cleanup completed.")

    except Exception as e:
        logger.error(f"❌ Daily database cleanup failed: {e}")




# Add necessary imports for the new search function
from bs4 import BeautifulSoup



async def search_duckduckgo_and_scrape(query: str, num_results: int = 3):
    """Execute a DuckDuckGo HTML search and return enriched snippets for the top results."""

    logger.info(f"Starting DuckDuckGo WEB search and scrape for query: '{query}'")
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=15) as client:
            response = await client.get(search_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results_container = soup.find_all("div", class_="web-result")

            search_entries = []
            seen_urls = set()
            for res in results_container:
                if len(search_entries) >= num_results:
                    break

                link_element = res.find("a", class_="result__a") or res.find("a", class_="result__url")
                if not link_element or not link_element.get("href"):
                    continue

                href = link_element["href"].strip()
                if not href or href in seen_urls:
                    continue

                seen_urls.add(href)

                snippet_element = res.find("div", class_="result__snippet")
                snippet_text = snippet_element.get_text(" ", strip=True) if snippet_element else ""

                title_text = link_element.get_text(" ", strip=True)

                search_entries.append({
                    "url": href,
                    "title": title_text,
                    "serp_snippet": snippet_text,
                })

            if not search_entries:
                logger.warning(f"DuckDuckGo web search for '{query}' returned no URLs.")
                return ""

            fetch_tasks = [client.get(entry["url"]) for entry in search_entries]
            responses = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        detailed_results = []
        for entry, resp in zip(search_entries, responses):
            if not isinstance(resp, httpx.Response):
                logger.warning(f"Failed to fetch URL {entry['url']}: {resp}")
                continue

            page_soup = BeautifulSoup(resp.text, "html.parser")

            title = entry["title"] or (page_soup.title.string.strip() if page_soup.title else "No Title")

            meta_desc = ""
            meta_tag = page_soup.find("meta", attrs={"name": re.compile("^description$", re.IGNORECASE)})
            if meta_tag and meta_tag.get("content"):
                meta_desc = meta_tag["content"].strip()
            if not meta_desc:
                og_desc = page_soup.find("meta", attrs={"property": "og:description"})
                if og_desc and og_desc.get("content"):
                    meta_desc = og_desc["content"].strip()

            paragraphs = [p.get_text(" ", strip=True) for p in page_soup.find_all("p")]
            paragraphs = [text for text in paragraphs if len(text) > 40]

            list_items = [li.get_text(" ", strip=True) for li in page_soup.find_all("li")]
            list_items = [text for text in list_items if len(text) > 20][:5]

            combined_text = " ".join(paragraphs)
            sentences = re.split(r"(?<=[。！？!?\.])\s+", combined_text)
            key_sentences = []
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if len(clean_sentence) < 30:
                    continue
                key_sentences.append(clean_sentence)
                if len(key_sentences) >= 4:
                    break

            summary_block = ""
            if key_sentences:
                summary_block = "\n".join(f"- {s}" for s in key_sentences)
            elif paragraphs:
                summary_candidate = " ".join(paragraphs[:2])
                if len(summary_candidate) > 600:
                    summary_candidate = summary_candidate[:600].rstrip() + "…"
                summary_block = summary_candidate
            elif entry["serp_snippet"]:
                summary_block = entry["serp_snippet"]

            bullet_block = ""
            if list_items:
                bullet_block = "\n".join(f"- {item}" for item in list_items[:3])

            parts = [f"Source: {entry['url']}"]
            parts.append(f"Title: {title or 'No Title'}")
            if entry["serp_snippet"]:
                parts.append(f"Search Snippet: {entry['serp_snippet']}")
            if meta_desc:
                parts.append(f"Meta Description: {meta_desc}")
            if summary_block:
                if summary_block.startswith("-"):
                    parts.append("Key Points:\n" + summary_block)
                else:
                    parts.append(f"Summary: {summary_block}")
            if bullet_block:
                parts.append("Notable Items:\n" + bullet_block)

            detailed_results.append("\n".join(parts))

        if not detailed_results:
            logger.warning(f"Failed to extract detailed content for query '{query}'")
            return ""

        logger.info(f"Successfully scraped {len(detailed_results)} pages for query '{query}'.")
        return "\n\n".join(detailed_results)

    except Exception as e:
        logger.error(f"DuckDuckGo web search and scrape failed for query '{query}': {e}")
        return ""


async def _get_search_plan_from_ai(
    db: Database,
    rate_limiter: RateLimitCache,
    original_request: ChatCompletionRequest,
    original_user_prompt: str,
    user_key_info: Dict,
    anti_detection: Any,
    search_focus: Optional[str] = None,
    append_current_time: bool = False,
) -> Optional[Dict]:
    """
    Calls the AI to generate a search plan (queries and pages).
    """
    logger.info("Getting search plan from AI...")
    try:
        # Use the default model for this lightweight task
        planning_model = db.get_config('default_model_name', 'gemini-2.5-flash-lite')

        # Get current time and format it
        current_time_str = datetime.now(ZoneInfo("Asia/Shanghai")).strftime('%Y-%m-%d %H:%M:%S')

        planning_target = search_focus or original_user_prompt

        planning_prompt = (
            f"Current date is {current_time_str}. Based on the user's request, generate a JSON object with optimal search queries and the number of pages to crawl for each. "
            f"User Request: '{planning_target}'\n\n"
            "Rules:\n"
            "- Provide 1 to 3 distinct search queries.\n"
            "- Design the queries to surface detailed, authoritative sources (official statistics, regulatory filings, primary research, long-form analysis).\n"
            "- Include modifiers such as 'detailed data', 'comprehensive analysis', 'latest statistics', or domain-specific jargon when it helps retrieve richer information.\n"
        )

        if append_current_time:
            planning_prompt += "- When freshness matters, append the exact current Beijing time string to the query.\n"

        planning_prompt += (
            "- For each query, specify 'num_pages' between 2 and 5.\n"
            "- Your response MUST be a valid JSON object in the following format, with no other text or explanations:\n"
            '```json\n'
            '{\n'
            '  "search_tasks": [\n'
            '    {"query": "keyword1", "num_pages": 3},\n'
            '    {"query": "keyword2", "num_pages": 4}\n'
            '  ]\n'
            '}\n'
            '```'
        )

        # Create a new, simple request for the planning step
        planning_openai_request = ChatCompletionRequest(
            model=original_request.model,
            messages=[ChatMessage(role="user", content=planning_prompt)]
        )

        planning_gemini_request = openai_to_gemini(db, planning_openai_request, anti_detection, {}, False)

        # Make the internal call. _internal_call=True bypasses some logging/features.
        response_dict = await make_request_with_fast_failover(
            db, rate_limiter, planning_gemini_request, planning_openai_request, planning_model, user_key_info, _internal_call=True
        )

        ai_response_text = response_dict['choices'][0]['message']['content']

        # Clean up and parse JSON
        json_str = ai_response_text.strip()
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].strip()
        if '```' in json_str:
            json_str = json_str.split('```')[0].strip()

        plan = json.loads(json_str)

        if isinstance(plan, dict) and 'search_tasks' in plan and isinstance(plan['search_tasks'], list):
            logger.info(f"Successfully received search plan from AI: {plan}")
            return plan
        else:
            logger.warning(f"AI search plan has invalid structure: {plan}")
            return None

    except Exception as e:
        logger.error(f"Failed to get or parse search plan from AI: {e}")
        return None


async def execute_search_flow(
    db: Database,
    rate_limiter: RateLimitCache,
    original_request: ChatCompletionRequest,
    model_name: str,
    user_key_info: Dict,
    anti_detection: Any,
    file_storage: Dict,
    enable_anti_detection: bool = False,
    search_focus: Optional[str] = None,
    append_current_time: bool = False
) -> Dict:
    """
    执行搜索流程, 使用 Google 搜索和页面抓取.
    This version now uses an AI-driven planning step.
    """
    original_user_prompt = ""
    if original_request.messages:
        last_user_message = next((m for m in reversed(original_request.messages) if m.role == 'user'), None)
        if last_user_message:
            original_user_prompt = last_user_message.get_text_content().strip()

    if not original_user_prompt:
        raise HTTPException(status_code=400, detail="User prompt is missing for search.")

    logger.info(f"Starting AI-driven search flow for prompt: '{original_user_prompt}'")

    search_focus_text = search_focus.strip() if isinstance(search_focus, str) else ""

    # 1. Get search plan from AI
    search_plan = await _get_search_plan_from_ai(
        db,
        rate_limiter,
        original_request,
        original_user_prompt,
        user_key_info,
        anti_detection,
        search_focus=search_focus_text or None,
        append_current_time=append_current_time
    )

    search_tasks_to_run = []
    time_suffix = None
    if append_current_time:
        current_time = datetime.now(ZoneInfo("Asia/Shanghai"))
        time_suffix = current_time.strftime("%Y-%m-%d %H:%M:%S (UTC+08:00)")

        def _apply_time_suffix(value: Optional[str]) -> Optional[str]:
            if not value:
                return value
            trimmed = value.strip()
            if not trimmed:
                return trimmed
            if time_suffix and time_suffix not in trimmed:
                return f"{trimmed} {time_suffix}"
            return trimmed
    else:

        def _apply_time_suffix(value: Optional[str]) -> Optional[str]:
            if not value:
                return value
            return value.strip()

    if search_plan and search_plan.get('search_tasks'):
        logger.info("Executing AI-generated search plan.")
        for task in search_plan['search_tasks']:
            query = task.get('query')
            num_pages = int(task.get('num_pages', 3))
            if query:
                adjusted_query = _apply_time_suffix(query)
                search_tasks_to_run.append(search_duckduckgo_and_scrape(adjusted_query or query, num_results=num_pages))
    else:
        logger.warning("Failed to get AI search plan, falling back to default behavior.")
        search_config = db.get_search_config()
        num_pages = search_config.get('num_pages_per_query', 3)
        fallback_query = search_focus_text or original_user_prompt
        adjusted_fallback = _apply_time_suffix(fallback_query)
        search_tasks_to_run.append(
            search_duckduckgo_and_scrape(adjusted_fallback or original_user_prompt, num_results=num_pages)
        )

    # 2. Concurrently perform searches and scrape results
    if not search_tasks_to_run:
        raise HTTPException(status_code=500, detail="Search plan resulted in no tasks to execute.")

    logger.info(f"Executing {len(search_tasks_to_run)} search/scrape tasks concurrently.")
    search_results = await asyncio.gather(*search_tasks_to_run)

    # 3. Aggregate results and build context
    search_context = "\n\n".join(filter(None, search_results))

    if not search_context.strip():
        search_context = "No search results found."
        logger.warning("All search queries returned no usable results.")

    logger.info(f"Aggregated search context length: {len(search_context)} chars")

    # 4. Build the final prompt for the Gemini model
    final_prompt = (
        f"Please provide a comprehensive answer to the user's original request based on the following search results. "
        f"Synthesize the information from the sources and provide a clear, coherent response. "
        f"Do not simply list the results. Cite sources using [Source: URL] at the end of relevant sentences if possible.\n\n"
        f"--- User's Request ---\n{original_user_prompt}\n\n"
        f"--- Search Results ---\n{search_context}\n--- End of Search Results ---"
    )

    # 5. Modify the original request to include the new context-aware prompt
    final_gemini_request = openai_to_gemini(db, original_request, anti_detection, file_storage, enable_anti_detection)

    if final_gemini_request['contents']:
        for part in reversed(final_gemini_request['contents']):
            if part.get('role') == 'user':
                part['parts'] = [{'text': final_prompt}]
                break

    return final_gemini_request





async def create_embeddings(
    db: Database,
    rate_limiter: RateLimitCache,
    request: EmbeddingRequest,
    user_key_info: Dict
) -> EmbeddingResponse:
    """
    Create embeddings for the given input.
    """
    model_name = request.model
    contents = [request.input] if isinstance(request.input, str) else request.input

    config = {}
    if request.task_type:
        config['task_type'] = request.task_type
    if request.output_dimensionality:
        config['output_dimensionality'] = request.output_dimensionality

    selection_result = await select_gemini_key_and_check_limits(db, rate_limiter, model_name)
    if not selection_result:
        raise HTTPException(status_code=429, detail="Rate limit exceeded or no available keys.")

    key_info = selection_result['key_info']
    if not _uses_cli_transport(key_info):
        raise HTTPException(status_code=503, detail="Non-CLI Gemini keys are no longer supported")

    embedding_data: List[EmbeddingData] = []
    prompt_tokens = 0

    async def _run_embed(text_value: str, index: int) -> None:
        nonlocal prompt_tokens
        payload: Dict[str, Any] = {
            "content": {
                "parts": [{"text": text_value}]
            }
        }
        if config.get('task_type'):
            payload["taskType"] = config['task_type']
        if config.get('output_dimensionality'):
            payload["outputDimensionality"] = config['output_dimensionality']

        start_time = time.time()
        response = await embed_with_cli_account(
            db,
            key_info,
            payload,
            model_name,
            timeout=float(db.get_config('request_timeout', '60')),
        )
        response_time = time.time() - start_time
        asyncio.create_task(update_key_performance_background(db, key_info['id'], True, response_time))

        values = response.get("embedding", {}).get("values") or []
        embedding_data.append(EmbeddingData(embedding=values, index=index))
        prompt_tokens += len(text_value.split())

    try:
        for idx, item in enumerate(contents):
            text_value = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            await _run_embed(text_value, idx)

        usage = EmbeddingUsage(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,
        )

        asyncio.create_task(
            log_usage_background(
                db,
                key_info['id'],
                user_key_info['id'],
                model_name,
                'success',
                1,
                prompt_tokens,
            )
        )
        await rate_limiter.add_usage(model_name, 1, prompt_tokens)

        return EmbeddingResponse(
            data=embedding_data,
            model=model_name,
            usage=usage,
        )

    except HTTPException:
        raise
    except Exception as e:
        asyncio.create_task(
            update_key_performance_background(db, key_info['id'], False, 0.0, error_type="other")
        )
        asyncio.create_task(
            log_usage_background(
                db,
                key_info['id'],
                user_key_info['id'],
                model_name,
                'failure',
                1,
                0,
            )
        )
        await rate_limiter.add_usage(model_name, 1, 0)
        logger.error(f"Embedding creation failed for key #{key_info['id']}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def create_gemini_native_embeddings(
    db: Database,
    rate_limiter: RateLimitCache,
    request: GeminiEmbeddingRequest,
    model_name: str,
    user_key_info: Dict
) -> GeminiEmbeddingResponse:
    """
    Create embeddings using Gemini's native request and response format.
    """
    contents = [request.contents] if isinstance(request.contents, str) else request.contents

    config_dict = request.config.dict() if request.config else {}

    selection_result = await select_gemini_key_and_check_limits(db, rate_limiter, model_name)
    if not selection_result:
        raise HTTPException(status_code=429, detail="Rate limit exceeded or no available keys.")

    key_info = selection_result['key_info']
    if not _uses_cli_transport(key_info):
        raise HTTPException(status_code=503, detail="Non-CLI Gemini keys are no longer supported")

    embeddings: List[EmbeddingValue] = []
    prompt_tokens = 0

    async def _run_native_embed(raw_content: Any) -> None:
        nonlocal prompt_tokens
        payload: Dict[str, Any] = {
            "content": raw_content if isinstance(raw_content, dict) else {
                "parts": [{"text": str(raw_content)}]
            }
        }
        if config_dict.get('task_type'):
            payload["taskType"] = config_dict['task_type']
        if config_dict.get('output_dimensionality'):
            payload["outputDimensionality"] = config_dict['output_dimensionality']

        start_time = time.time()
        response = await embed_with_cli_account(
            db,
            key_info,
            payload,
            model_name,
            timeout=float(db.get_config('request_timeout', '60')),
        )
        response_time = time.time() - start_time
        asyncio.create_task(update_key_performance_background(db, key_info['id'], True, response_time))

        values = response.get("embedding", {}).get("values") or []
        embeddings.append(EmbeddingValue(values=values))
        prompt_tokens += len(str(raw_content).split())

    try:
        for item in contents:
            await _run_native_embed(item)

        asyncio.create_task(
            log_usage_background(
                db,
                key_info['id'],
                user_key_info['id'],
                model_name,
                'success',
                1,
                prompt_tokens
            )
        )
        await rate_limiter.add_usage(model_name, 1, prompt_tokens)

        return GeminiEmbeddingResponse(embeddings=embeddings)

    except HTTPException:
        raise
    except Exception as e:
        asyncio.create_task(
            update_key_performance_background(db, key_info['id'], False, 0.0, error_type="other")
        )
        asyncio.create_task(
            log_usage_background(
                db,
                key_info['id'],
                user_key_info['id'],
                model_name,
                'failure',
                1,
                0
            )
        )
        await rate_limiter.add_usage(model_name, 1, 0)

        logger.error(f"Native embedding creation failed for key #{key_info['id']}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_with_preprocessing(
    preprocessing_coro: Coroutine,
    final_streamer: Coroutine,
    db: Database,
    rate_limiter: RateLimitCache,
    request: ChatCompletionRequest,
    model_name: str,
    user_key_info: Dict
) -> AsyncGenerator[bytes, None]:
    """
    一个包装器，首先执行异步预处理，然后将结果传递给最终的流式处理函数。
    """
    try:
        # 1. 等待预处理协程完成，获取最终的 gemini_request
        final_gemini_request = await preprocessing_coro
        
        # 2. 使用预处理后的请求，调用并迭代最终的流式处理函数
        async for chunk in final_streamer(db, rate_limiter, final_gemini_request, request, model_name, user_key_info):
            yield chunk
            
    except HTTPException as e:
        # 捕获预处理或流式处理中的HTTP异常，并以流式错误格式返回
        error_data = {"error": {"message": e.detail, "code": e.status_code}}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    except Exception as e:
        # 捕获其他未知异常
        logger.error(f"Error in stream_with_preprocessing: {e}")
        error_data = {"error": {"message": str(e), "code": 500}}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

async def _execute_deepthink_preprocessing(
    db: Database,
    rate_limiter: RateLimitCache,
    original_request: ChatCompletionRequest,
    model_name: str,
    user_key_info: Dict,
    concurrency: int,
    anti_detection: Any,
    file_storage: Dict,
    enable_anti_detection: bool = False
) -> Dict:
    """
    执行完整的“反思式DeepThink”流程，返回最终的gemini_request。
    流程: 探索 -> 初步综合 -> 反思与二次规划 -> 最终综合
    """
    original_user_prompt = next((m.content for m in original_request.messages if m.role == 'user'), '')
    if not original_user_prompt:
        raise HTTPException(status_code=400, detail="User prompt is missing.")

    # 内部辅助函数，用于执行子请求
    async def _execute_sub_request(prompt: str, is_json: bool = False):
        try:
            temp_req = ChatCompletionRequest(model=original_request.model, messages=[ChatMessage(role="user", content=prompt)])
            gemini_req_body = openai_to_gemini(db, temp_req, anti_detection, file_storage, enable_anti_detection)
            if is_json:
                # Ensure generation_config exists and is a types.GenerationConfig object
                if "generation_config" not in gemini_req_body or gemini_req_body["generation_config"] is None:
                    gemini_req_body["generation_config"] = types.GenerationConfig()
                # Set the response_mime_type attribute on the GenerationConfig object
                gemini_req_body["generation_config"].response_mime_type = "application/json"

            response = await make_request_with_fast_failover(db, rate_limiter, gemini_req_body, temp_req, model_name, user_key_info, _internal_call=True)
            content = response['choices'][0]['message']['content']
            return json.loads(content) if is_json else content
        except Exception as e:
            logger.error(f"DeepThink sub-request failed for prompt '{prompt[:100]}...': {e}")
            return {"error": str(e)} if is_json else f"Error processing sub-request: {e}"

    # 内部辅助函数，用于执行搜索或推理
    async def _run_exploration_prompt(prompt: str):
        prompt = prompt.strip()
        if prompt.lower().startswith('[search]'):
            search_query = prompt[len('[search]'):].strip()
            logger.info(f"DeepThink is executing a search for: '{search_query}'")
            try:
                search_result = await search_duckduckgo_and_scrape(search_query, num_results=3)
                return f"Search results for '{search_query}':\n{search_result}" if search_result else f"No search results found for '{search_query}'."
            except Exception as e:
                logger.error(f"DeepThink sub-search failed for query '{search_query}': {e}")
                return f"Error during search for '{search_query}': {e}"
        else:
            return await _execute_sub_request(prompt)

    # --- 阶段一：并行探索 ---
    logger.info("--- DeepThink Phase 1: Parallel Exploration ---")
    prompt_gen_prompt = f"Based on the user's request, generate {concurrency} distinct and diverse thinking prompts to explore the problem from different angles. If a prompt requires real-time information or external data, include the keyword `[search]` at the beginning of that prompt string. Return the prompts as a JSON array of strings. User request: \"{original_user_prompt}\""
    initial_prompts = await _execute_sub_request(prompt_gen_prompt, is_json=True)
    if isinstance(initial_prompts, dict) and "error" in initial_prompts:
        raise HTTPException(status_code=500, detail=f"Failed to generate initial prompts: {initial_prompts['error']}")
    if not isinstance(initial_prompts, list):
        raise HTTPException(status_code=500, detail="Initial prompt generation did not return a list.")

    exploration_tasks = [_run_exploration_prompt(p) for p in initial_prompts]
    exploration_results = await asyncio.gather(*exploration_tasks)
    exploration_context = "\n\n".join([f"Exploration Result {i+1}:\n---\n{result}\n---" for i, result in enumerate(exploration_results)])

    # --- 阶段二：初步综合与自我反思 ---
    # 2.1 初步综合 (Drafting)
    logger.info("--- DeepThink Phase 2.1: Initial Synthesis (Drafting) ---")
    drafting_prompt = f"Original user request: \"{original_user_prompt}\"\n\nBased on the following exploration results, synthesize a comprehensive and well-structured 'preliminary answer draft'.\n\n{exploration_context}\n\nPreliminary Answer Draft:"
    draft_answer = await _execute_sub_request(drafting_prompt)
    if "Error processing sub-request" in draft_answer:
        raise HTTPException(status_code=500, detail="Failed to create initial draft.")

    # 2.2 批判性反思与二次规划 (Self-Correction)
    logger.info("--- DeepThink Phase 2.2: Critical Reflection & Secondary Planning ---")
    reflection_prompt = f"""
    You are a 'Reflector AI'. Your task is to critically analyze a preliminary answer and plan for its improvement.

    User's Original Request: "{original_user_prompt}"

    Preliminary Answer Draft:
    ---
    {draft_answer}
    ---

    Your Tasks:
    1.  **Analyze & Critique**: Evaluate the draft. Is it complete? Does it have logical gaps? Is the information sufficient?
    2.  **Propose Improvements**: Clearly state what is missing or could be improved.
    3.  **Plan Secondary Exploration**: Based on your critique, generate exactly 2 new, targeted exploration prompts to gather the missing information. If a prompt needs web search, prefix it with `[search]`.

    Return your response as a JSON object with three keys: "critique", "improvements", and "new_prompts" (which should be an array of 2 strings).
    Example: {{ "critique": "...", "improvements": "...", "new_prompts": ["[search] latest market trends for AI", "Explain the technical challenges of..."] }}
    """
    reflection_result = await _execute_sub_request(reflection_prompt, is_json=True)
    if isinstance(reflection_result, dict) and "error" in reflection_result:
        raise HTTPException(status_code=500, detail=f"Failed during reflection stage: {reflection_result['error']}")
    if not all(k in reflection_result for k in ["critique", "improvements", "new_prompts"]):
        raise HTTPException(status_code=500, detail="Reflection stage returned malformed JSON.")

    # --- 阶段三：最终答案生成 ---
    # 3.1 二次探索
    logger.info("--- DeepThink Phase 3.1: Secondary Exploration ---")
    secondary_prompts = reflection_result.get("new_prompts", [])
    secondary_tasks = [_run_exploration_prompt(p) for p in secondary_prompts]
    secondary_results = await asyncio.gather(*secondary_tasks)
    secondary_context = "\n\n".join([f"Secondary Exploration Result {i+1}:\n---\n{result}\n---" for i, result in enumerate(secondary_results)])

    # 3.2 最终综合 (Finalization)
    logger.info("--- DeepThink Phase 3.2: Final Synthesis ---")
    final_synthesis_prompt = f"""
    Your task is to generate a final, high-quality answer by integrating all available information.

    1.  **User's Original Request**: "{original_user_prompt}"

    2.  **Preliminary Draft**:
        ---
        {draft_answer}
        ---

    3.  **Critique and Improvements Suggested**:
        - Critique: {reflection_result.get('critique')}
        - Improvements: {reflection_result.get('improvements')}

    4.  **Additional Information from Secondary Exploration**:
        ---
        {secondary_context}
        ---

    Instructions:
    - Revise and enhance the preliminary draft using the critique and the new information.
    - Ensure the final answer is comprehensive, accurate, and directly addresses the user's original request.
    - Produce only the final, polished answer without any extra commentary.

    Final Answer:
    """

    # 更新原始请求以包含最终的综合提示
    final_request_messages = copy.deepcopy(original_request.messages)
    found_user = False
    for msg in reversed(final_request_messages):
        if msg.role == 'user':
            msg.content = final_synthesis_prompt
            found_user = True
            break
    if not found_user:
         final_request_messages.append(ChatMessage(role="user", content=final_synthesis_prompt))

    final_openai_request = original_request.copy(update={"messages": final_request_messages})

    # 4. 返回最终的gemini_request
    return openai_to_gemini(db, final_openai_request, anti_detection, file_storage, enable_anti_detection=enable_anti_detection)
