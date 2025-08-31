# api_services.py
import asyncio
import json
import time
import uuid
import logging
import os
import copy
import itertools
from typing import Coroutine, Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime
from zoneinfo import ZoneInfo
import httpx
from urllib.parse import quote_plus

from google.genai import types
from fastapi import HTTPException

from database import Database
from api_models import ChatCompletionRequest, ChatMessage
from api_utils import get_cached_client, map_finish_reason, decrypt_response, check_gemini_key_health, RateLimitCache, openai_to_gemini

logger = logging.getLogger(__name__)

_rr_counter = itertools.count()
_rr_lock = asyncio.Lock()

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
            asyncio.create_task(schedule_health_check(db, key_id))

        db.update_gemini_key(key_id, **update_data)

    except Exception as e:
        logger.error(f"Background performance update failed for key {key_id}: {e}")


async def schedule_health_check(db: Database, key_id: int):
    """
    调度后台健康检测任务
    """
    try:
        # 获取配置中的延迟时间
        config = db.get_failover_config()
        delay = config.get('health_check_delay', 5)

        # 延迟指定时间后执行健康检测，避免立即重复检测
        await asyncio.sleep(delay)

        key_info = db.get_gemini_key_by_id(key_id)
        if key_info and key_info.get('status') == 1:  # 只检测激活的key
            health_result = await check_gemini_key_health(key_info['key'])

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


async def collect_gemini_response_directly(
        db: Database,
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        use_stream: bool = True,
        _internal_call: bool = False
) -> Dict:
    """
    从Google API收集完整响应
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
    
    # 确定超时时间
    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover(db)
    if has_tool_calls:
        timeout = 60.0
    elif is_fast_failover:
        timeout = 60.0
    else:
        timeout = float(db.get_config('request_timeout', '60'))

    logger.info(f"Starting direct collection from: {url}")
    
    complete_content = ""
    thinking_content = ""
    total_tokens = 0
    finish_reason = "stop"
    processed_lines = 0

    # 防截断相关变量
    anti_trunc_cfg = db.get_anti_truncation_config() if hasattr(db, 'get_anti_truncation_config') else {'enabled': False}
    full_response = ""
    saw_finish_tag = False
    start_time = time.time()

    client = get_cached_client(gemini_key)
    try:
        if use_stream:
            # 使用 google-genai 的流式接口
            genai_stream = await client.aio.models.generate_content_stream(
                model=model_name,
                contents=gemini_request["contents"],
                config=gemini_request.get("generation_config")
            )
            async for chunk in genai_stream:
                data = chunk.to_dict() if hasattr(chunk, "to_dict") else json.loads(chunk.model_dump_json())
                for candidate in data.get("candidates", []):
                    content_data = candidate.get("content", {})
                    parts = content_data.get("parts", [])
                    for part in parts:
                        text = part.get("text", "")
                        if not text: continue
                        total_tokens += len(text.split())
                        is_thought = part.get("thought", False)
                        if is_thought:
                            thinking_content += text
                        else:
                            complete_content += text
                    finish_reason_raw = candidate.get("finishReason", "stop")
                    finish_reason = map_finish_reason(finish_reason_raw) if finish_reason_raw else "stop"
                    processed_lines += 1
            response_time = time.time() - start_time
            asyncio.create_task(update_key_performance_background(db, key_id, True, response_time))
        else:
            # 非流式直接调用
            response_obj = await client.aio.models.generate_content(
                model=model_name,
                contents=gemini_request["contents"],
                config=gemini_request.get("generation_config")
            )
            data = response_obj.to_dict() if hasattr(response_obj, "to_dict") else json.loads(response_obj.model_dump_json())
            for candidate in data.get("candidates", []):
                finish_reason_raw = candidate.get("finishReason", "stop")
                finish_reason = map_finish_reason(finish_reason_raw) if finish_reason_raw else "stop"
                for part in candidate.get("content", {}).get("parts", []):
                    text = part.get("text", "")
                    if text:
                        complete_content += text
                        total_tokens += len(text.split())
            response_time = time.time() - start_time
            asyncio.create_task(update_key_performance_background(db, key_id, True, response_time))

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
    if not complete_content.strip():
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
                cont_response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=continuation_request["contents"],
                    config=continuation_request.get("generation_config")
                )
                data = cont_response.to_dict() if hasattr(cont_response, "to_dict") else json.loads(cont_response.model_dump_json())
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        text = part.get("text", "")
                        if text:
                            complete_content += text
                            total_tokens += len(text.split())
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
    message = {
        "role": "assistant",
        "content": final_content
    }
    if thinking_content_final:
        message["reasoning"] = thinking_content_final

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
            "finish_reason": finish_reason
        }],
        "usage": usage
    }

    logger.info(f"Successfully collected direct response: {len(final_content)} chars, {completion_tokens} tokens, {reasoning_tokens} reasoning tokens")
    return openai_response


async def make_gemini_request_single_attempt(
        db: Database,
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        model_name: str,
        timeout: float = 60.0
) -> Dict:
    start_time = time.time()

    try:
        client = get_cached_client(gemini_key)
        async with asyncio.timeout(timeout):
            response_obj = await client.aio.models.generate_content(
                model=model_name,
                contents=gemini_request["contents"],
                config=gemini_request["generation_config"]
            )
        response_time = time.time() - start_time
        # SDK 对象转 dict
        response_dict = response_obj.to_dict() if hasattr(response_obj, "to_dict") else json.loads(response_obj.model_dump_json())
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

    except Exception as e:
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(db, key_id, False, response_time)
        )
        # google-genai 会在异常中封装详细信息
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
                
                # 收集响应
                response = await collect_gemini_response_directly(
                    db,
                    key_info['key'],
                    key_info['id'],
                    gemini_request,
                    openai_request,
                    model_name,
                    _internal_call=_internal_call
                )
                
                logger.info(f"✅ Request successful with key #{key_info['id']} on attempt {attempt + 1}")

                # 从响应中获取token使用量
                usage = response.get('usage', {})
                total_tokens = usage.get('completion_tokens', 0)
                prompt_tokens = usage.get('prompt_tokens', 0)

                # 记录使用量
                if user_key_info:
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
                await rate_limiter.add_usage(model_name, 1, total_tokens)
                return response

            except HTTPException as e:
                failed_keys.append(key_info['id'])
                last_error = e

                logger.warning(f"❌ Key #{key_info['id']} failed: {e.detail}")

                # 记录失败的使用量
                if user_key_info:
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

async def stream_gemini_response_single_attempt(
        db: Database,
        rate_limiter: RateLimitCache,
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        _internal_call: bool = False
) -> AsyncGenerator[bytes, None]:
    """
    单次流式请求尝试，失败立即抛出异常，使用 google-genai SDK 实现
    """
    # 确定超时时间：工具调用或快速响应模式使用60秒，其他使用配置值
    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover(db)
    if has_tool_calls:
        timeout = 60.0  # 工具调用强制60秒超时
        logger.info("Using extended 60s timeout for tool calls in streaming")
    elif is_fast_failover:
        timeout = 60.0  # 快速响应模式使用60秒超时
        logger.info("Using extended 60s timeout for fast response mode in streaming")
    else:
        timeout = float(db.get_config('request_timeout', '60'))

    logger.info(f"Starting single stream request to model: {model_name}")

    start_time = time.time()

    try:
        client = get_cached_client(gemini_key)
        async with asyncio.timeout(timeout):
            contents = gemini_request["contents"]
            # 流式接口直接使用contents和body参数
            genai_stream = await client.aio.models.generate_content_stream(
                model=model_name,
                contents=gemini_request["contents"],
                config=gemini_request.get("generation_config")
            )

            if False:  # legacy httpx code disabled after migration to google-genai
                    response_time = time.time() - start_time
                    asyncio.create_task(
                        update_key_performance_background(db, key_id, False, response_time)
                    )

                    error_text = await response.aread()
                    error_msg = error_text.decode() if error_text else f"HTTP {response.status_code}"
                    logger.error(f"Stream request failed with status {response.status_code}: {error_msg}")
                    raise Exception(f"Stream request failed: {error_msg}")

            stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(time.time())
            total_tokens = 0
            thinking_sent = False
            has_content = False
            processed_lines = 0
            # Anti-truncation related variables
            anti_trunc_cfg = db.get_anti_truncation_config() if hasattr(db, 'get_anti_truncation_config') else {'enabled': False}
            full_response = ""
            saw_finish_tag = False

            logger.info("Stream response started")

            async for chunk in genai_stream:
                    choices = chunk.candidates or []
                    for candidate in choices:
                        content = candidate.content or {}
                        parts = content.parts or []
                        
                        # Tool call streaming can be complex, aggregate parts first
                        # This logic assumes tool calls might be streamed chunk by chunk.
                        # A more robust implementation might need to accumulate parts across chunks.
                        
                        for i, part in enumerate(parts):
                            delta = {}
                            finish_reason_str = None

                            if hasattr(part, "text"):
                                text = part.text
                                if not text:
                                    continue
                                total_tokens += len(text.split())
                                has_content = True
                                
                                # Anti-truncation handling (stream) remains the same
                                text_to_send = text
                                if anti_trunc_cfg.get('enabled') and not _internal_call:
                                    idx = text.find('[finish]')
                                    if idx != -1:
                                        text_to_send = text[:idx]
                                        saw_finish_tag = True
                                full_response += text_to_send

                                is_thought = getattr(part, "thought", False)
                                if is_thought:
                                    delta["reasoning"] = text_to_send
                                else:
                                    delta["content"] = text_to_send
                            
                            elif hasattr(part, "function_call"):
                                fc = part.function_call
                                has_content = True
                                # OpenAI streams tool calls with an index
                                # We simulate this by creating a tool_call chunk per function call
                                tool_call_chunk = {
                                    "index": i, # Use part index as tool index
                                    "id": f"call_{uuid.uuid4().hex}",
                                    "type": "function",
                                    "function": {
                                        "name": fc.name,
                                        "arguments": json.dumps(fc.args, ensure_ascii=False)
                                    }
                                }
                                delta["tool_calls"] = [tool_call_chunk]

                            if delta:
                                chunk_data = {
                                    "id": stream_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": openai_request.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode('utf-8')

                        finish_reason = getattr(candidate, "finish_reason", None)
                        if finish_reason:
                            finish_chunk = {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": openai_request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": map_finish_reason(finish_reason)
                                }]
                            }
                            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                            yield "data: [DONE]\n\n".encode('utf-8')

                            logger.info(
                                f"Stream completed with finish_reason: {finish_reason}, tokens: {total_tokens}")

                            response_time = time.time() - start_time
                            asyncio.create_task(update_key_performance_background(db, key_id, True, response_time))
                            await rate_limiter.add_usage(model_name, 1, total_tokens)
                            return



            # 如果正常结束但没有内容，抛出异常
            if not has_content:
                logger.warning("Stream ended without content")
                raise Exception("Stream response had no content")

            # 正常结束，发送完成信号
            if has_content:
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
                yield "data: [DONE]\n\n".encode('utf-8')

                logger.info(
                    f"Stream ended naturally, tokens: {total_tokens}")

                response_time = time.time() - start_time
                asyncio.create_task(
                    update_key_performance_background(db, key_id, True, response_time)
                )

            await rate_limiter.add_usage(model_name, 1, total_tokens)



    # except Exception as e:  # 原 httpx 超时连接异常移除
    # Legacy httpx branch disabled after migration to google-genai
    except Exception as e:
        logger.warning(f"Stream timeout/connection error: {str(e)}")
        response_time = time.time() - start_time
        asyncio.create_task(
            update_key_performance_background(db, key_id, False, response_time)
        )
        raise Exception(f"Stream connection failed: {str(e)}")


async def stream_with_fast_failover(
        db: Database,
        rate_limiter: RateLimitCache,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        user_key_info: Dict = None,
        max_key_attempts: int = None,
        _internal_call: bool = False
) -> AsyncGenerator[bytes, None]:
    """
    流式响应快速故障转移
    """
    available_keys = db.get_available_gemini_keys()

    if not available_keys:
        error_data = {
            'error': {
                'message': 'No available API keys',
                'type': 'service_unavailable',
                'code': 503
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield "data: [DONE]\n\n".encode('utf-8')
        return

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting stream fast failover with up to {max_key_attempts} key attempts for {model_name}")

    failed_keys = []

    for attempt in range(max_key_attempts):
        try:
            selection_result = await select_gemini_key_and_check_limits(
                db,
                rate_limiter,
                model_name,
                excluded_keys=set(failed_keys)
            )

            if not selection_result:
                break

            key_info = selection_result['key_info']
            logger.info(f"Stream fast failover attempt {attempt + 1}: Using key #{key_info['id']}")

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

            success = False
            total_tokens = 0

            try:
                async for chunk in stream_gemini_response_single_attempt(
                        db,
                        rate_limiter,
                        key_info['key'],
                        key_info['id'],
                        gemini_request,
                        openai_request,
                        model_name,
                        _internal_call=_internal_call
                ):
                    yield chunk
                    success = True

                if success:
                    # 在后台记录使用量
                    if user_key_info:
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

                    await rate_limiter.add_usage(model_name, 1, total_tokens)
                    return

            except Exception as e:
                failed_keys.append(key_info['id'])
                logger.warning(f"Stream key #{key_info['id']} failed: {str(e)}")

                # 在后台更新性能指标
                asyncio.create_task(
                    update_key_performance_background(db, key_info['id'], False, 0.0)
                )

                # 记录失败的使用量
                if user_key_info:
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

                if attempt < max_key_attempts - 1:
                    logger.info(f"Key #{key_info['id']} failed, trying next key...")
                    # retry_msg = {
                    #     'error': {
                    #         'message': f'Key #{key_info["id"]} failed, trying next key...',
                    #         'type': 'retry_info',
                    #         'retry_attempt': attempt + 1
                    #     }
                    # }
                    # yield f"data: {json.dumps(retry_msg, ensure_ascii=False)}\n\n".encode('utf-8')
                    continue
                else:
                    break

        except Exception as e:
            logger.error(f"Stream failover error on attempt {attempt + 1}: {str(e)}")
            continue

    # 所有key都失败了
    error_data = {
        'error': {
            'message': f'All {len(failed_keys)} available API keys failed',
            'type': 'all_keys_failed',
            'code': 503,
            'failed_keys': failed_keys
        }
    }
    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
    yield "data: [DONE]\n\n".encode('utf-8')


async def _keep_alive_generator(task: asyncio.Task) -> AsyncGenerator[bytes, Any]:
    """
    一个通用的异步生成器，用于在后台任务运行时发送 keep-alive 心跳。
    任务完成后，它会 yield 任务的结果。
    """
    while not task.done():
        try:
            # 等待任务2秒，如果未完成则发送心跳
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except asyncio.TimeoutError:
            yield b": keep-alive\n\n"
    
    # 任务完成，返回结果
    yield await task


async def stream_with_preprocessing(
    preprocessing_coro: Coroutine,
    streaming_func: callable,
    db: Database,
    rate_limiter: RateLimitCache,
    openai_request: ChatCompletionRequest,
    model_name: str,
    user_key_info: Dict = None
) -> AsyncGenerator[bytes, None]:
    """
    在执行一个耗时的预处理任务时发送 keep-alive 心跳，然后流式传输最终结果。
    """
    task = asyncio.create_task(preprocessing_coro)
    
    async for result in _keep_alive_generator(task):
        if isinstance(result, bytes):
            yield result  # This is a keep-alive chunk
        else:
            # This is the final result from the preprocessing task
            modified_gemini_request = result
            if modified_gemini_request:
                # Now, stream the final response
                async for chunk in streaming_func(db, rate_limiter, modified_gemini_request, openai_request, model_name, user_key_info):
                    yield chunk
            else:
                # Handle cases where preprocessing failed
                error_data = {"error": {"message": "Preprocessing failed to produce a valid request.", "code": 500}}
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"


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
        gemini_key: str,
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
            # 复用缓存 client，避免重复创建
            client = get_cached_client(gemini_key)
            async with asyncio.timeout(timeout):
                genai_response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=gemini_request["contents"],
                    config=gemini_request["generation_config"]
                )
                
                response_time = time.time() - start_time
                # 更新key性能
                db.update_key_performance(key_id, True, response_time)
                
                # 将genai响应格式化为与旧代码兼容的格式
                response_json = genai_response.to_dict()
                return response_json

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="Request timeout")
            else:
                logger.warning(f"Request timeout (attempt {attempt + 1}), retrying...")
                await asyncio.sleep(2 ** attempt)
                continue
        except Exception as e:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt == max_retries - 1:
                # 提取错误消息
                error_message = str(e)
                status_code = 500
                
                # 尝试分析错误类型
                if "429" in error_message or "rate limit" in error_message.lower():
                    status_code = 429
                elif "403" in error_message or "permission" in error_message.lower():
                    status_code = 403
                elif "404" in error_message or "not found" in error_message.lower():
                    status_code = 404
                elif "400" in error_message or "invalid" in error_message.lower():
                    status_code = 400
                
                raise HTTPException(status_code=status_code, detail=error_message)
            else:
                logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}, retrying...")
                await asyncio.sleep(2 ** attempt)
                continue

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
                response = await collect_gemini_response_directly(
                    db,
                    key_info['key'],
                    key_info['id'],
                    gemini_request,
                    openai_request,
                    model_name,
                    _internal_call=_internal_call
                )

                logger.info(f"✅ Request successful with key #{key_info['id']} on attempt {attempt + 1}")

                # 从响应中获取token使用量
                usage = response.get('usage', {})
                total_tokens = usage.get('completion_tokens', 0)

                if user_key_info:
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

                await rate_limiter.add_usage(model_name, 1, total_tokens)
                return response

            except HTTPException as e:
                failed_keys.append(key_info['id'])
                last_error = e

                db.update_key_performance(key_info['id'], False, 0.0)

                if user_key_info:
                    db.log_usage(
                        gemini_key_id=key_info['id'],
                        user_key_id=user_key_info['id'],
                        model_name=model_name,
                        status='failure',
                        requests=1,
                        tokens=0
                    )

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
    """传统流式响应处理（保留用于兼容）"""
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
        yield "data: [DONE]\n\n".encode('utf-8')
        return

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    else:
        max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting stream failover with {max_key_attempts} key attempts for {model_name}")

    failed_keys = []

    for attempt in range(max_key_attempts):
        try:
            selection_result = await select_gemini_key_and_check_limits(
                db,
                rate_limiter,
                model_name,
                excluded_keys=excluded_keys.union(set(failed_keys))
            )

            if not selection_result:
                break

            key_info = selection_result['key_info']
            logger.info(f"Stream attempt {attempt + 1}: Using key #{key_info['id']}")

            success = False
            total_tokens = 0
            try:
                async for chunk in stream_gemini_response(
                        db,
                        rate_limiter,
                        key_info['key'],
                        key_info['id'],
                        gemini_request,
                        openai_request,
                        key_info,
                        model_name,
                        _internal_call=_internal_call
                ):
                    yield chunk
                    success = True

                if success:
                    if user_key_info:
                        db.log_usage(
                            gemini_key_id=key_info['id'],
                            user_key_id=user_key_info['id'],
                            model_name=model_name,
                            status='success',
                            requests=1,
                            tokens=total_tokens
                        )
                        logger.info(
                            f"📊 Logged stream usage: gemini_key_id={key_info['id']}, user_key_id={user_key_info['id']}, model={model_name}")

                    await rate_limiter.add_usage(model_name, 1, total_tokens)
                    return

            except Exception as e:
                failed_keys.append(key_info['id'])
                logger.warning(f"Stream key #{key_info['id']} failed: {str(e)}")

                db.update_key_performance(key_info['id'], False, 0.0)

                if user_key_info:
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
                else:
                    break

        except Exception as e:
            logger.error(f"Stream failover error on attempt {attempt + 1}: {str(e)}")
            continue

    error_data = {
        'error': {
            'message': f'All {len(failed_keys)} available API keys failed',
            'type': 'all_keys_failed',
            'code': 503,
            'failed_keys': failed_keys
        }
    }
    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
    yield "data: [DONE]\n\n".encode('utf-8')


async def stream_gemini_response(
        db: Database,
        rate_limiter: RateLimitCache,
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        key_info: Dict,
        model_name: str,
        _internal_call: bool = False
) -> AsyncGenerator[bytes, None]:
    """处理Gemini的流式响应，记录性能指标"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
    
    # 确定超时时间：工具调用或快速响应模式使用60秒，其他使用配置值
    has_tool_calls = bool(openai_request.tools or openai_request.tool_choice)
    is_fast_failover = await should_use_fast_failover(db)
    if has_tool_calls:
        timeout = 60.0  # 工具调用强制60秒超时
        logger.info("Using extended 60s timeout for tool calls in traditional streaming")
    elif is_fast_failover:
        timeout = 60.0  # 快速响应模式使用60秒超时
        logger.info("Using extended 60s timeout for fast response mode in traditional streaming")
    else:
        timeout = float(db.get_config('request_timeout', '60'))
    
    max_retries = int(db.get_config('max_retries', '3'))

    logger.info(f"Starting stream request to: {url}")

    start_time = time.time()

    for attempt in range(max_retries):
        try:
            client = get_cached_client(gemini_key)
            async with asyncio.timeout(timeout):
                genai_stream = await client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=gemini_request["contents"],
                    config=gemini_request.get("generation_config")
                )
                # 将 google-genai 流式响应包装为 SSE
                stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())
                total_tokens = 0
                thinking_sent = False
                processed_chunks = 0
                
                # 防截断相关变量
                anti_trunc_cfg = db.get_anti_truncation_config() if hasattr(db, 'get_anti_truncation_config') else {'enabled': False}
                full_response = ""
                continuation_attempted = False
                saw_finish_tag = False

                async for chunk in genai_stream:
                    processed_chunks += 1
                    choices = chunk.candidates or []
                    for candidate in choices:
                        content = candidate.content or {}
                        parts = content.parts or []
                        for part in parts:
                            if hasattr(part, "text"):
                                text = part.text
                                if not text:
                                    continue
                                total_tokens += len(text.split())
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

                        finish_reason = getattr(candidate, "finish_reason", None)
                        if finish_reason:
                            finish_chunk = {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": openai_request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": map_finish_reason(finish_reason)
                                }]
                            }
                            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                            yield "data: [DONE]\n\n".encode('utf-8')

                            response_time = time.time() - start_time
                            db.update_key_performance(key_id, True, response_time)
                            await rate_limiter.add_usage(model_name, 1, total_tokens)
                            return
                    if response.status_code != 200:
                        response_time = time.time() - start_time
                        db.update_key_performance(key_id, False, response_time)

                        # 如果是429错误，则标记为速率受限
                        if response.status_code == 429:
                            logger.warning(f"Stream key #{key_id} is rate-limited (429). Marking as 'rate_limited'.")
                            db.update_gemini_key_status(key_id, 'rate_limited')

                        error_text = await response.aread()
                        error_msg = error_text.decode() if error_text else "Unknown error"
                        logger.error(f"Stream request failed with status {response.status_code}: {error_msg}")
                        yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'api_error', 'code': response.status_code}}, ensure_ascii=False)}\n\n".encode(
                            'utf-8')
                        yield "data: [DONE]\n\n".encode('utf-8')
                        return

                    stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    created = int(time.time())
                    total_tokens = 0
                    thinking_sent = False
                    has_content = False
                    processed_lines = 0

                    logger.info(f"Stream response started, status: {response.status_code}")

                    try:
                        async for line in response.aiter_lines():
                            processed_lines += 1

                            if not line:
                                continue

                            if processed_lines <= 5:
                                logger.debug(f"Stream line {processed_lines}: {line[:100]}...")

                            if line.startswith("data: "):
                                json_str = line[6:]

                                if json_str.strip() == "[DONE]":
                                    logger.info("Received [DONE] signal from stream")
                                    break

                                if not json_str.strip():
                                    continue

                                try:
                                    data = json.loads(json_str)

                                    for candidate in data.get("candidates", []):
                                        content_data = candidate.get("content", {})
                                        parts = content_data.get("parts", [])

                                        for part in parts:
                                            if "text" in part:
                                                text = part["text"]
                                                if not text:
                                                    continue

                                                total_tokens += len(text.split())
                                                has_content = True
                                                # Anti-truncation handling
                                                if anti_trunc_cfg.get('enabled') and not _internal_call:
                                                    idx = text.find('[finish]')
                                                    if idx != -1:
                                                        text_to_send = text[:idx]
                                                        saw_finish_tag = True
                                                    else:
                                                        text_to_send = text
                                                else:
                                                    text_to_send = text
                                                full_response += text_to_send

                                                is_thought = part.get("thought", False)

                                                if is_thought and not (openai_request.thinking_config and
                                                                       openai_request.thinking_config.include_thoughts):
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
                                                    yield f"data: {json.dumps(thinking_header, ensure_ascii=False)}\n\n".encode(
                                                        'utf-8')
                                                    thinking_sent = True
                                                    logger.debug("Sent thinking header")
                                                elif not is_thought and thinking_sent:
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
                                                    yield f"data: {json.dumps(response_header, ensure_ascii=False)}\n\n".encode(
                                                        'utf-8')
                                                    thinking_sent = False
                                                    logger.debug("Sent response header")

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
                                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode(
                                                    'utf-8')

                                        finish_reason = candidate.get("finishReason")
                                        if finish_reason:
                                            finish_chunk = {
                                                "id": stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": openai_request.model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {},
                                                    "finish_reason": map_finish_reason(finish_reason)
                                                }]
                                            }
                                            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode(
                                                'utf-8')
                                            yield "data: [DONE]\n\n".encode('utf-8')

                                            logger.info(
                                                f"Stream completed with finish_reason: {finish_reason}, tokens: {total_tokens}")

                                            response_time = time.time() - start_time
                                            db.update_key_performance(key_id, True, response_time)
                                            await rate_limiter.add_usage(model_name, 1, total_tokens)
                                            return

                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSON decode error: {e}, line: {json_str[:200]}...")
                                    continue

                            elif line.startswith("event: "):
                                continue
                            elif line.startswith("id: ") or line.startswith("retry: "):
                                continue

                        if has_content:
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
                            yield "data: [DONE]\n\n".encode('utf-8')

                            logger.info(
                                f"Stream ended naturally, processed {processed_lines} lines, tokens: {total_tokens}")

                            response_time = time.time() - start_time
                            db.update_key_performance(key_id, True, response_time)

                        if not has_content:
                            logger.warning(
                                f"Stream response had no content after processing {processed_lines} lines, falling back to non-stream")
                            try:
                                fallback_response = await make_gemini_request_with_retry(
                                    db, gemini_key, key_id, gemini_request, model_name, 1, timeout=timeout
                                )

                                include_thoughts_fallback = openai_request.thinking_config and openai_request.thinking_config.include_thoughts
                                thoughts, content = extract_thoughts_and_content(fallback_response, include_thoughts_fallback)

                                if thoughts and openai_request.thinking_config and openai_request.thinking_config.include_thoughts:
                                    full_content = f"**Thinking Process:**\n{thoughts}\n\n**Response:**\n{content}"
                                else:
                                    full_content = content

                                if full_content:
                                    chunk_data = {
                                        "id": stream_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": openai_request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": full_content},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode('utf-8')

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
                                    total_tokens = len(full_content.split())

                                    logger.info(f"Fallback completed, tokens: {total_tokens}")

                            except Exception as e:
                                logger.error(f"Fallback request failed: {e}")
                                response_time = time.time() - start_time
                                db.update_key_performance(key_id, False, response_time)
                                yield f"data: {json.dumps({'error': {'message': 'Failed to get response', 'type': 'server_error'}}, ensure_ascii=False)}\n\n".encode(
                                    'utf-8')

                        await rate_limiter.add_usage(model_name, 1, total_tokens)
                        yield "data: [DONE]\n\n".encode('utf-8')
                        return

                    except Exception as e:
                        logger.warning(f"Stream connection error (attempt {attempt + 1}): {str(e)}")
                        response_time = time.time() - start_time
                        db.update_key_performance(key_id, False, response_time)
                        if attempt < max_retries - 1:
                            yield f"data: {json.dumps({'error': {'message': 'Connection interrupted, retrying...', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                                'utf-8')
                            await asyncio.sleep(1)
                            continue
                        else:
                            yield f"data: {json.dumps({'error': {'message': 'Stream connection failed after retries', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                                'utf-8')
                            yield "data: [DONE]\n\n".encode('utf-8')
                            return

        except Exception as e:
            logger.warning(f"Connection error (attempt {attempt + 1}): {str(e)}")
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt < max_retries - 1:
                yield f"data: {json.dumps({'error': {'message': f'Connection error, retrying... (attempt {attempt + 1})', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                yield f"data: {json.dumps({'error': {'message': 'Connection failed after all retries', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                yield "data: [DONE]\n\n".encode('utf-8')
                return
        except Exception as e:
            logger.error(f"Unexpected error in stream (attempt {attempt + 1}): {str(e)}")
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            else:
                yield f"data: {json.dumps({'error': {'message': 'Unexpected error occurred', 'type': 'server_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                yield "data: [DONE]\n\n".encode('utf-8')
                return

async def record_hourly_health_check(db: Database):
    """每小时记录一次健康检测结果"""
    try:
        available_keys = db.get_available_gemini_keys()

        for key_info in available_keys:
            key_id = key_info['id']

            # 执行健康检测
            health_result = await check_gemini_key_health(key_info['key'])

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
    """DeepThink的预处理部分，返回最终的gemini_request"""
    original_user_prompt = next((m.content for m in original_request.messages if m.role == 'user'), '')
    if not original_user_prompt:
        raise HTTPException(status_code=400, detail="User prompt is missing.")

    # 1. 生成N个新Prompt
    prompt_generation_request = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": f"Based on the user's request, generate {concurrency} distinct and diverse thinking prompts to explore the problem from different angles. Return the prompts as a JSON array of strings. User request: \"{original_user_prompt}\""
            }]
        }],
        "generation_config": { "temperature": 0.5, "maxOutputTokens": 2048, "response_mime_type": "application/json" }
    }
    
    try:
        temp_request = ChatCompletionRequest(model=original_request.model, messages=[{"role": "user", "content": "generate prompts"}])
        response = await make_request_with_fast_failover(db, rate_limiter, prompt_generation_request, temp_request, model_name, user_key_info, _internal_call=True)
        prompts = json.loads(response['choices'][0]['message']['content'])
        if not isinstance(prompts, list) or len(prompts) != concurrency:
            raise ValueError(f"Failed to generate {concurrency} valid prompts.")
    except Exception as e:
        logger.error(f"DeepThink prompt generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate thinking prompts.")

    # 2. 并发执行
    async def run_prompt(prompt):
        request_body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generation_config": gemini_request.get("generation_config")}
        try:
            temp_req = ChatCompletionRequest(model=original_request.model, messages=[{"role": "user", "content": prompt}])
            response = await make_request_with_fast_failover(db, rate_limiter, request_body, temp_req, model_name, user_key_info, _internal_call=True)
            return response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"DeepThink sub-request failed for prompt '{prompt}': {e}")
            return f"Error processing sub-request: {e}"

    tasks = [run_prompt(p) for p in prompts]
    answers = await asyncio.gather(*tasks)

    # 3. 综合提炼
    explorations = "\n\n".join([f"Exploration {i+1}:\n\"{answer}\"" for i, answer in enumerate(answers)])
    synthesis_prompt = f'Original user request: "{original_user_prompt}"\n\nBased on the original request and the following {concurrency} independent explorations, synthesize a final, high-quality answer.\n\n{explorations}\n\nSynthesized Answer:'
    
    # 更新原始请求以包含综合提示
    final_request_messages = copy.deepcopy(original_request.messages)
    # 替换或追加综合提示
    found_user = False
    for msg in reversed(final_request_messages):
        if msg.role == 'user':
            msg.content = synthesis_prompt
            found_user = True
            break
    if not found_user:
         final_request_messages.append({"role": "user", "content": synthesis_prompt})

    final_openai_request = original_request.copy(update={"messages": final_request_messages})
    
    # 4. 返回最终的gemini_request
    return openai_to_gemini(db, final_openai_request, anti_detection, file_storage, enable_anti_detection=enable_anti_detection)

async def execute_search_flow(
    db: Database,
    rate_limiter: RateLimitCache,
    original_request: ChatCompletionRequest,
    model_name: str,
    user_key_info: Dict,
    anti_detection: Any,
    file_storage: Dict,
    enable_anti_detection: bool = True
) -> Dict:
    """
    执行由模型驱动的自主搜索流程.
    """
    original_user_prompt = next((m.content for m in original_request.messages if m.role == 'user'), '')
    if not original_user_prompt:
        raise HTTPException(status_code=400, detail="User prompt is missing for search.")

    logger.info(f"Starting model-driven search flow for prompt: '{original_user_prompt}'")

    # 1. 由模型生成搜索查询
    search_queries = []
    try:
        generation_prompt = f"根据以下用户请求，生成一个或多个简洁有效的搜索查询，以全面回答该请求。以 JSON 数组的形式返回结果。用户请求：'{original_user_prompt}'"
        query_generation_request = {
            "contents": [{"role": "user", "parts": [{"text": generation_prompt}]}],
            "generation_config": {"temperature": 0.5, "maxOutputTokens": 2048, "response_mime_type": "application/json"}
        }
        temp_request = ChatCompletionRequest(model=model_name, messages=[{"role": "user", "content": "generate queries"}])
        response = await make_request_with_fast_failover(db, rate_limiter, query_generation_request, temp_request, model_name, user_key_info, _internal_call=True)
        
        queries_str = response['choices'][0]['message']['content']
        search_queries = json.loads(queries_str)
        
        if not isinstance(search_queries, list) or not all(isinstance(q, str) for q in search_queries):
            raise ValueError("Model did not return a valid JSON array of strings for search queries.")
        
        logger.info(f"Model generated {len(search_queries)} search queries: {search_queries}")

    except Exception as e:
        logger.error(f"Failed to generate search queries by model: {e}. Falling back to user prompt.")
        search_queries = [original_user_prompt]

    # 2. 并发执行搜索
    async def search_duckduckgo(query: str):
        try:
            async with httpx.AsyncClient() as client:
                params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
                headers = {"User-Agent": "GeminiProxy/1.7.0"}
                response = await client.get("https://api.duckduckgo.com/", params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                results = []
                if data.get("AbstractText"):
                    results.append(f"Source: {data.get('AbstractSource', 'N/A')}\nContent: {data.get('AbstractText')}")
                
                related_topics = data.get("RelatedTopics", [])
                for topic in related_topics:
                    if "Text" in topic and len(results) < 3: # 每个查询最多补充到3条
                        results.append(f"Result: {topic['Text']}")
                
                return f"--- Results for query '{query}' ---\n" + "\n\n".join(results) if results else ""
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed for query '{query}': {e}")
            return ""

    search_tasks = [search_duckduckgo(query) for query in search_queries]
    search_results = await asyncio.gather(*search_tasks)
    
    # 3. 聚合和格式化结果
    search_context = "\n\n".join(filter(None, search_results))
    if not search_context.strip():
        search_context = "No search results found."
        logger.warning("All search queries returned no usable results.")
    else:
        logger.info(f"Aggregated search context length: {len(search_context)} chars")

    # 4. 构建最终的综合提示
    beijing_time = datetime.now(ZoneInfo("Asia/Shanghai")).strftime('%Y-%m-%d %H:%M:%S')
    
    synthesis_prompt = f"""
Please provide a comprehensive answer to the user's original request based on the following search results.
The current Beijing time is {beijing_time}.

--- Search Results ---
{search_context}
--- End of Search Results ---

User's Original Request: "{original_user_prompt}"

Final Answer:
"""
    
    # 5. 更新原始请求的消息并返回最终的gemini_request
    final_messages = copy.deepcopy(original_request.messages)
    user_message_found = False
    for msg in reversed(final_messages):
        if msg.role == 'user':
            msg.content = synthesis_prompt
            user_message_found = True
            break
    
    if not user_message_found:
        final_messages.append(ChatMessage(role="user", content=synthesis_prompt))

    final_openai_request = original_request.copy(update={"messages": final_messages})
    
    return openai_to_gemini(db, final_openai_request, anti_detection, file_storage, enable_anti_detection=enable_anti_detection)
