# api_routes.py
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime

import psutil
import sys
from fastapi import (APIRouter, Depends, File, Header, HTTPException,
                     UploadFile)
from fastapi.responses import JSONResponse, StreamingResponse

from api_models import ChatCompletionRequest, EmbeddingRequest, GeminiEmbeddingRequest
from api_services import (_execute_deepthink_preprocessing, create_embeddings, create_gemini_native_embeddings, execute_search_flow, make_request_with_failover,
                          make_request_with_fast_failover,
                          should_use_fast_failover,
                          stream_non_stream_keep_alive,
                          stream_with_failover, stream_with_fast_failover,
                          delete_unhealthy_keys, stream_with_preprocessing)
from api_utils import (GeminiAntiDetectionInjector, check_gemini_key_health,
                       delete_file_from_gemini, get_actual_model_name,
                       inject_prompt_to_messages, openai_to_gemini,
                       upload_file_to_gemini, validate_file_for_gemini, UserRateLimiter)
from database import Database
from api_services import auto_cleanup_failed_keys
from dependencies import (get_anti_detection, get_db, get_keep_alive_enabled,
                          get_request_count, get_start_time, get_rate_limiter)
from api_utils import RateLimitCache

logger = logging.getLogger(__name__)

# Main Router for user-facing APIs
router = APIRouter()

# Admin Router for management APIs
admin_router = APIRouter(prefix="/admin", tags=["管理 API"])


# Mock file storage (as in the original single-file app)
file_storage = {}
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

MAX_INLINE_SIZE = 20 * 1024 * 1024
MAX_FILE_SIZE = 100 * 1024 * 1024
SUPPORTED_MIME_TYPES = {
    # Images
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
    # Audio
    "audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4", "audio/flac", "audio/aac",
    # Video
    "video/mp4", "video/x-msvideo", "video/quicktime", "video/webm",
# Documents
    "application/pdf", "text/plain", "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
}

# ===============================================================================
# General and v1 API Routes
# ===============================================================================

@router.get("/", summary="服务根端点", tags=["通用"])
async def root(
    db: Database = Depends(get_db),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
    return {
        "service": "Gemini API Proxy",
        "status": "running",
        "version": "1.6.0",
        "features": ["Gemini 2.5 Multimodal"],
        "keep_alive": keep_alive_enabled,
        "auto_cleanup": db.get_auto_cleanup_config()['enabled'],
        "anti_detection": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "fast_failover": db.get_failover_config()['fast_failover_enabled'],
        "docs": "/docs",
        "health": "/health"
    }

@router.get("/health", summary="服务健康检查", tags=["通用"])
async def health_check(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
    available_keys = len(db.get_available_gemini_keys())
    uptime = time.time() - start_time
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_keys": available_keys,
        "environment": "render" if os.getenv('RENDER_EXTERNAL_URL') else "local",
        "uptime_seconds": int(uptime),
        "request_count": request_count,
        "version": "1.6.0",
        "multimodal_support": "Gemini 2.5 Optimized",
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }

@router.get("/wake", summary="服务唤醒", tags=["通用"])
async def wake_up(
    db: Database = Depends(get_db),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
    return {
        "status": "awake",
        "timestamp": datetime.now().isoformat(),
        "message": "Service is active",
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }

@router.get("/status", summary="获取详细服务状态", tags=["通用"])
async def get_status(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled),
    anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection)
):
    process = psutil.Process(os.getpid())
    return {
        "service": "Gemini API Proxy",
        "status": "running",
        "version": "1.6.0",
        "render_url": os.getenv('RENDER_EXTERNAL_URL'),
        "python_version": sys.version,
        "models": db.get_supported_models(),
        "active_keys": len(db.get_available_gemini_keys()),
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "uptime_seconds": int(time.time() - start_time),
        "total_requests": request_count,
        "thinking_enabled": db.get_thinking_config()['enabled'],
        "multimodal_optimized": True,
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "anti_detection_stats": anti_detection.get_statistics(),
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }

@router.get("/metrics", summary="获取服务指标", tags=["通用"])
async def get_metrics(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled),
    anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection)
):
    process = psutil.Process(os.getpid())
    return {
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "active_connections": len(db.get_available_gemini_keys()),
        "uptime_seconds": int(time.time() - start_time),
        "requests_count": request_count,
        "database_size_mb": os.path.getsize(db.db_path) / 1024 / 1024 if os.path.exists(db.db_path) else 0,
        "keep_alive_enabled": keep_alive_enabled,
        "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "anti_detection_stats": anti_detection.get_statistics(),
        "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled']
    }

@router.get("/v1", summary="获取 v1 API 信息", tags=["通用"])
async def api_v1_info(
    db: Database = Depends(get_db),
    start_time: float = Depends(get_start_time),
    request_count: int = Depends(get_request_count),
    keep_alive_enabled: bool = Depends(get_keep_alive_enabled)
):
    render_url = os.getenv('RENDER_EXTERNAL_URL')
    base_url = render_url if render_url else 'https://your-service.onrender.com'
    return {
        "service": "Gemini API Proxy",
        "version": "1.6.0",
        "api_version": "v1",
        "compatibility": "OpenAI API v1",
        "description": "A high-performance proxy for Gemini API with OpenAI compatibility.",
        "status": "operational",
        "base_url": base_url,
        "features": [
            "Multi-key polling & load balancing", "OpenAI API compatibility", "Rate limiting & usage analytics",
            "Thinking mode support", "Optimized Gemini 2.5 multimodal", "Streaming responses", "Fast failover",
            "Real-time monitoring", "Health checking", "Adaptive load balancing", "Auto keep-alive",
            "Auto-cleanup unhealthy keys", "Anti-automation detection"
        ],
        "endpoints": {
            "chat_completions": "/v1/chat/completions", "models": "/v1/models", "files": "/v1/files",
            "api_info": "/v1", "health": "/health", "status": "/status", "admin": "/admin/*", "docs": "/docs"
        },
        "supported_models": db.get_supported_models(),
        "service_status": {
            "active_gemini_keys": len(db.get_available_gemini_keys()),
            "thinking_enabled": db.get_thinking_config().get('enabled', False),
            "thinking_budget": db.get_thinking_config().get('budget', -1),
            "uptime_seconds": int(time.time() - start_time),
            "total_requests": request_count,
            "keep_alive_enabled": keep_alive_enabled,
            "auto_cleanup_enabled": db.get_auto_cleanup_config()['enabled'],
            "auto_cleanup_threshold": db.get_auto_cleanup_config()['days_threshold'],
            "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
            "fast_failover_enabled": db.get_failover_config()['fast_failover_enabled'],
        },
        "multimodal_support": {
            "images": ["jpeg", "png", "gif", "webp", "bmp"],
            "audio": ["mp3", "wav", "ogg", "mp4", "flac", "aac"],
            "video": ["mp4", "avi", "mov", "webm", "quicktime"],
            "documents": ["pdf", "txt", "csv", "docx", "xlsx"]
        },
        "timestamp": datetime.now().isoformat()
    }

@router.post("/v1/files", summary="上传文件", tags=["用户 API"])
async def upload_file(
        file: UploadFile = File(..., description="要上传的文件，最大100MB"),
        authorization: str = Header(None, description="用户访问密钥，格式为 'Bearer sk-...'"),
        db: Database = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    user_key = db.validate_user_key(authorization.replace("Bearer ", ""))
    if not user_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    file_content = await file.read()
    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
    validation_result = validate_file_for_gemini(
        file_content, mime_type, file.filename,
        supported_mime_types=SUPPORTED_MIME_TYPES,
        max_file_size=MAX_FILE_SIZE,
        max_inline_size=MAX_INLINE_SIZE
    )
    file_id = f"file-{uuid.uuid4().hex}"
    file_info = {
        "id": file_id, "object": "file", "bytes": validation_result["size"], "created_at": int(time.time()),
        "filename": file.filename, "purpose": "multimodal", "mime_type": mime_type,
        "use_inline": validation_result["use_inline"]
    }

    if validation_result["use_inline"]:
        file_info["data"] = base64.b64encode(file_content).decode('utf-8')
        file_info["format"] = "inlineData"
    else:
        gemini_keys = db.get_available_gemini_keys()
        if not gemini_keys:
            raise HTTPException(status_code=503, detail="No available Gemini keys for file upload")
        gemini_key = gemini_keys[0]['key']
        gemini_file_uri = await upload_file_to_gemini(file_content, mime_type, file.filename, gemini_key)
        if gemini_file_uri:
            file_info["gemini_file_uri"] = gemini_file_uri
            file_info["gemini_key_used"] = gemini_key
            file_info["format"] = "fileData"
        else:
            raise HTTPException(status_code=500, detail="Failed to upload file to Gemini File API")

    file_storage[file_id] = file_info
    return {
        "id": file_id, "object": "file", "bytes": validation_result["size"], "created_at": file_info["created_at"],
        "filename": file.filename, "purpose": "multimodal", "format": file_info["format"]
    }

@router.get("/v1/files", summary="列出已上传的文件", tags=["用户 API"])
async def list_files(authorization: str = Header(None), db: Database = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    if not db.validate_user_key(authorization.replace("Bearer ", "")):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    files = [{"id": fid, **finfo} for fid, finfo in file_storage.items()]
    return {"object": "list", "data": files}

@router.get("/v1/files/{file_id}", summary="获取文件信息", tags=["用户 API"])
async def get_file(file_id: str, authorization: str = Header(None), db: Database = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    if not db.validate_user_key(authorization.replace("Bearer ", "")):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    return file_storage[file_id]

@router.delete("/v1/files/{file_id}", summary="删除文件", tags=["用户 API"])
async def delete_file(file_id: str, authorization: str = Header(None), db: Database = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    if not db.validate_user_key(authorization.replace("Bearer ", "")):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = file_storage[file_id]
    if "gemini_file_uri" in file_info:
        await delete_file_from_gemini(file_info["gemini_file_uri"], file_info["gemini_key_used"])
    
    del file_storage[file_id]
    return {"id": file_id, "object": "file", "deleted": True}

@router.post("/v1/chat/completions", summary="创建聊天补全", tags=["用户 API"])
async def chat_completions(
        request: ChatCompletionRequest,
        authorization: str = Header(None),
        db: Database = Depends(get_db),
        anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection),
        rate_limiter: RateLimitCache = Depends(get_rate_limiter)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    user_key_info = db.validate_user_key(authorization.replace("Bearer ", ""))
    if not user_key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 用户级别速率限制
    try:
        user_rate_limiter = UserRateLimiter(db, user_key_info)
        user_rate_limiter.check_rate_limits()
    except HTTPException as e:
        # 直接抛出 UserRateLimiter 中生成的异常
        raise e
    except Exception as e:
        # 捕获其他潜在错误
        logger.error(f"Error during user rate limit check: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during rate limit check")

    # 提前执行提示词注入，以确保在所有模式下都生效
    request.messages = inject_prompt_to_messages(db, request.messages)

    last_user_message = next((m.content for m in reversed(request.messages) if m.role == 'user'), None)
    actual_model_name = get_actual_model_name(db, request.model)
    
    # DeepThink Logic
    deepthink_config = db.get_deepthink_config()
    if deepthink_config.get('enabled') and last_user_message and '[deepthink' in last_user_message.lower():
        concurrency = deepthink_config.get('concurrency', 3)
        match = re.search(r'\[deepthink:(\d+)\]', last_user_message, re.IGNORECASE)
        if match:
            custom_concurrency = int(match.group(1))
            if 3 <= custom_concurrency <= 7: concurrency = custom_concurrency
            for msg in request.messages:
                if msg.role == 'user': msg.content = msg.content.replace(match.group(0), '').strip()
        else:
            for msg in request.messages:
                if msg.role == 'user': msg.content = re.sub(r'\[deepthink\]', '', msg.content, flags=re.IGNORECASE).strip()

        preprocessing_coro = _execute_deepthink_preprocessing(db, rate_limiter, request, actual_model_name, user_key_info, concurrency, anti_detection, file_storage, enable_anti_detection=False)

        if request.stream:
            final_streamer = stream_with_fast_failover if await should_use_fast_failover(db) else stream_with_failover
            return StreamingResponse(
                stream_with_preprocessing(preprocessing_coro, final_streamer, db, rate_limiter, request, actual_model_name, user_key_info),
                media_type="text/event-stream"
            )
        else:
            final_gemini_request = await preprocessing_coro
            response_func = make_request_with_fast_failover if await should_use_fast_failover(db) else make_request_with_failover
            response = await response_func(db, rate_limiter, final_gemini_request, request, actual_model_name, user_key_info)
            return JSONResponse(content=response)

    # Search Logic
    search_config = db.get_search_config()
    if search_config.get('enabled') and last_user_message and '[search]' in last_user_message.lower():
        logger.info("Search mode activated")
        for msg in request.messages:
            if msg.role == 'user': msg.content = re.sub(r'\[search\]', '', msg.content, flags=re.IGNORECASE).strip()
        
        preprocessing_coro = execute_search_flow(db, rate_limiter, request, actual_model_name, user_key_info, anti_detection, file_storage, enable_anti_detection=False)

        if request.stream:
            final_streamer = stream_with_fast_failover if await should_use_fast_failover(db) else stream_with_failover
            return StreamingResponse(
                stream_with_preprocessing(preprocessing_coro, final_streamer, db, rate_limiter, request, actual_model_name, user_key_info),
                media_type="text/event-stream"
            )
        else:
            final_gemini_request = await preprocessing_coro
            response_func = make_request_with_fast_failover if await should_use_fast_failover(db) else make_request_with_failover
            response = await response_func(db, rate_limiter, final_gemini_request, request, actual_model_name, user_key_info)
            return JSONResponse(content=response)
    
    # Regular flow
    gemini_request = openai_to_gemini(db, request, anti_detection, file_storage, enable_anti_detection=True)

    stream_mode = db.get_stream_mode_config().get('mode', 'auto')
    has_tool_calls = bool(request.tools or request.tool_choice)
    decryption_enabled = db.get_response_decryption_config().get('enabled', False)
    anti_truncation_enabled = db.get_anti_truncation_config().get('enabled', False)

    if request.stream and (decryption_enabled or has_tool_calls or anti_truncation_enabled):
        return StreamingResponse(stream_non_stream_keep_alive(db, rate_limiter, gemini_request, request, actual_model_name, user_key_info), media_type="text/event-stream")

    should_stream = request.stream
    if stream_mode == 'stream': should_stream = True
    elif stream_mode == 'non_stream': should_stream = False

    if should_stream:
        streamer = stream_with_fast_failover if await should_use_fast_failover(db) else stream_with_failover
        return StreamingResponse(streamer(db, rate_limiter, gemini_request, request, actual_model_name, user_key_info), media_type="text/event-stream")
    else:
        requester = make_request_with_fast_failover if await should_use_fast_failover(db) else make_request_with_failover
        response = await requester(db, rate_limiter, gemini_request, request, actual_model_name, user_key_info)
        return JSONResponse(content=response)

@router.get("/v1/models", summary="列出可用模型", tags=["用户 API"])
async def list_models(db: Database = Depends(get_db)):
    models = db.get_supported_models()
    model_list = [{"id": model, "object": "model", "created": int(time.time()), "owned_by": "google"} for model in models]
    return {"object": "list", "data": model_list}

@router.post("/v1/embeddings", summary="创建嵌入", tags=["用户 API"])
async def embeddings(
    request: EmbeddingRequest,
    authorization: str = Header(None),
    db: Database = Depends(get_db),
    rate_limiter: RateLimitCache = Depends(get_rate_limiter)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    user_key_info = db.validate_user_key(authorization.replace("Bearer ", ""))
    if not user_key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        user_rate_limiter = UserRateLimiter(db, user_key_info)
        user_rate_limiter.check_rate_limits()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during user rate limit check: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during rate limit check")

    response = await create_embeddings(db, rate_limiter, request, user_key_info)
    return JSONResponse(content=response.dict())

@router.post("/v1/models/{model_name:path}:embedContent", summary="创建原生Gemini嵌入", tags=["用户 API"])
async def gemini_native_embeddings(
    model_name: str,
    request: GeminiEmbeddingRequest,
    authorization: str = Header(None),
    db: Database = Depends(get_db),
    rate_limiter: RateLimitCache = Depends(get_rate_limiter)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    user_key_info = db.validate_user_key(authorization.replace("Bearer ", ""))
    if not user_key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        user_rate_limiter = UserRateLimiter(db, user_key_info)
        user_rate_limiter.check_rate_limits()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during user rate limit check: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during rate limit check")

    response = await create_gemini_native_embeddings(db, rate_limiter, request, model_name, user_key_info)
    return JSONResponse(content=response.dict())

# ===============================================================================
# Admin API Routes
# ===============================================================================

@admin_router.post("/health/check-all", summary="一键健康检测")
async def check_all_keys_health_endpoint(db: Database = Depends(get_db)):
    active_keys = [key for key in db.get_all_gemini_keys() if key['status'] == 1]
    if not active_keys:
        return {"success": True, "message": "No active keys to check"}
    
    tasks = [check_gemini_key_health(key['key']) for key in active_keys]
    results = await asyncio.gather(*tasks)
    
    healthy_count = 0
    output_results = []
    for key, result in zip(active_keys, results):
        db.update_key_performance(key['id'], result['healthy'], result['response_time'])
        db.record_daily_health_status(key['id'], result['healthy'], result['response_time'])
        status_code = result.get('status_code', 'N/A')
        logger.info(f"密钥 #{key['id']} 状态：{status_code}")
        if result['healthy']:
            healthy_count += 1
        output_results.append({"key_id": key['id'], **result})
        
    return {
        "success": True, "message": f"Health check completed: {healthy_count}/{len(active_keys)} keys healthy",
        "results": output_results
    }

@admin_router.get("/health/summary", summary="获取健康状态汇总")
async def get_health_summary(db: Database = Depends(get_db)):
    return {"success": True, "summary": db.get_keys_health_summary()}

@admin_router.get("/cleanup/status", summary="获取自动清理状态")
async def get_cleanup_status(db: Database = Depends(get_db)):
    config = db.get_auto_cleanup_config()
    return {
        "success": True, **config,
        "at_risk_keys": db.get_at_risk_keys(config['days_threshold'])
    }

@admin_router.post("/cleanup/config", summary="更新自动清理配置")
async def update_cleanup_config(request: dict, db: Database = Depends(get_db)):
    db.set_auto_cleanup_config(
        enabled=request.get('enabled'),
        days_threshold=request.get('days_threshold'),
        min_checks_per_day=request.get('min_checks_per_day')
    )
    return {"success": True, "message": "Config updated"}

@admin_router.post("/cleanup/manual", summary="手动执行清理")
async def manual_cleanup(db: Database = Depends(get_db)):
    try:
        await auto_cleanup_failed_keys(db)
        return {"success": True, "message": "Manual cleanup executed successfully"}
    except Exception as e:
        logger.error(f"Manual cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.get("/failover/stats", summary="获取故障转移统计")
async def get_failover_stats(db: Database = Depends(get_db)):
    try:
        health_summary = db.get_keys_health_summary()
        return {
            "success": True,
            "health_summary": health_summary,
            "config": db.get_failover_config(),
            "recommendations": {
                "optimal_max_attempts": min(max(health_summary.get('healthy', 0), 2), 5),
                "fast_failover_recommended": health_summary.get('unhealthy', 0) > 0,
                "background_check_recommended": True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get failover stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.post("/test/anti-detection", summary="测试防检测功能")
async def test_anti_detection(anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection)):
    try:
        test_texts = [ "请帮我分析这个问题", "使用中文回复：", "请告诉我", "我想说：" ]
        results = []
        for text in test_texts:
            processed = anti_detection.inject_symbols(text)
            results.append({
                "original": text,
                "processed": processed,
                "char_difference": len(processed) - len(text)
            })
        return {
            "success": True,
            "results": results,
            "total_symbols_available": len(anti_detection.safe_symbols)
        }
    except Exception as e:
        logger.error(f"Anti-detection test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@admin_router.get("/config/failover", summary="获取故障转移配置")
async def get_failover_config_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_failover_config()}

@admin_router.post("/config/failover", summary="更新故障转移配置")
async def update_failover_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_failover_config(
        fast_failover_enabled=request.get('fast_failover_enabled'),
        background_health_check=request.get('background_health_check'),
        health_check_delay=request.get('health_check_delay')
    )
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/anti-detection", summary="获取防检测配置")
async def get_anti_detection_config_endpoint(db: Database = Depends(get_db), anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection)):
    return {
        "success": True,
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "disable_for_tools": db.get_config('anti_detection_disable_for_tools', 'true').lower() == 'true',
        "token_threshold": int(db.get_config('anti_detection_token_threshold', '5000')),
        "statistics": anti_detection.get_statistics()
    }

@admin_router.post("/config/anti-detection", summary="更新防检测配置")
async def update_anti_detection_config_endpoint(request: dict, db: Database = Depends(get_db)):
    if (v := request.get('anti_detection_enabled')) is not None: db.set_config('anti_detection_enabled', str(v).lower())
    if (v := request.get('disable_for_tools')) is not None: db.set_config('anti_detection_disable_for_tools', str(v).lower())
    if (v := request.get('token_threshold')) is not None: db.set_config('anti_detection_token_threshold', str(v))
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/anti-truncation", summary="获取防截断配置")
async def get_anti_truncation_config_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_anti_truncation_config()}

@admin_router.post("/config/anti-truncation", summary="更新防截断配置")
async def update_anti_truncation_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_anti_truncation_config(enabled=request.get('enabled'))
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/response-decryption", summary="获取响应解密配置")
async def get_response_decryption_config_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_response_decryption_config()}

@admin_router.post("/config/response-decryption", summary="更新响应解密配置")
async def update_response_decryption_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_response_decryption_config(enabled=request.get('enabled'))
    return {"success": True, "message": "Config updated"}

@admin_router.get("/keep-alive/status", summary="获取 Keep-Alive 状态")
async def get_keep_alive_status(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_keep_alive_config()}

@admin_router.post("/keep-alive/toggle", summary="切换 Keep-Alive 状态")
async def toggle_keep_alive(request: dict, db: Database = Depends(get_db)):
    db.set_keep_alive_config(enabled=request.get('enabled'))
    return {"success": True, "message": "Keep-Alive status updated"}

@admin_router.post("/keep-alive/ping", summary="手动触发 Keep-Alive")
async def ping_keep_alive():
    # This endpoint is a placeholder to allow manual triggering if needed.
    # The actual keep-alive logic is handled by the background task.
    return {"success": True, "message": "Ping acknowledged. Keep-alive is managed by a background task."}

@admin_router.get("/keys/gemini", summary="获取所有Gemini密钥")
async def get_gemini_keys_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "keys": db.get_all_gemini_keys()}

@admin_router.post("/config/gemini-key", summary="添加Gemini密钥")
async def add_gemini_key_endpoint(request: dict, db: Database = Depends(get_db)):
    keys_str = request.get("key", "")
    keys = [k.strip() for k in re.split(r'[,;\s\n\r]+', keys_str) if k.strip()]
    added_count = 0
    for key in keys:
        if db.add_gemini_key(key):
            added_count += 1
    return {"success": True, "message": f"Added {added_count} keys."}

@admin_router.delete("/keys/gemini/unhealthy", summary="删除所有异常的Gemini密钥")
async def delete_unhealthy_keys_endpoint(db: Database = Depends(get_db)):
    result = delete_unhealthy_keys(db)
    return result


@admin_router.delete("/keys/gemini/{key_id}", summary="删除Gemini密钥")
async def delete_gemini_key_endpoint(key_id: int, db: Database = Depends(get_db)):
    if db.delete_gemini_key(key_id):
        return {"success": True, "message": "Key deleted"}
    raise HTTPException(404, "Key not found")

@admin_router.post("/keys/gemini/{key_id}/toggle", summary="切换Gemini密钥状态")
async def toggle_gemini_key_status_endpoint(key_id: int, db: Database = Depends(get_db)):
    if db.toggle_gemini_key_status(key_id):
        return {"success": True, "message": "Status toggled"}
    raise HTTPException(404, "Key not found")

@admin_router.get("/keys/user", summary="获取所有用户密钥")
async def get_user_keys_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "keys": db.get_all_user_keys()}

@admin_router.post("/config/user-key", summary="生成用户密钥")
async def generate_user_key_endpoint(request: dict, db: Database = Depends(get_db)):
    name = request.get("name", "API User")
    key = db.generate_user_key(name)
    return {"success": True, "key": key, "name": name}

@admin_router.delete("/keys/user/{key_id}", summary="删除用户密钥")
async def delete_user_key_endpoint(key_id: int, db: Database = Depends(get_db)):
    if db.delete_user_key(key_id):
        return {"success": True, "message": "Key deleted"}
    raise HTTPException(404, "Key not found")

@admin_router.post("/keys/user/{key_id}/toggle", summary="切换用户密钥状态")
async def toggle_user_key_status_endpoint(key_id: int, db: Database = Depends(get_db)):
    if db.toggle_user_key_status(key_id):
        return {"success": True, "message": "Status toggled"}
    raise HTTPException(404, "Key not found")

@admin_router.post("/keys/user/{key_id}/config", summary="更新用户密钥配置")
async def update_user_key_config_endpoint(key_id: int, request: dict, db: Database = Depends(get_db)):
    if db.update_user_key(key_id, **request):
        return {"success": True, "message": "Config updated"}
    raise HTTPException(404, "Key not found")

@admin_router.get("/models", summary="列出所有模型配置")
async def list_model_configs_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "models": db.get_all_model_configs()}

@admin_router.get("/models/{model_name}", summary="获取模型配置")
async def get_model_config_endpoint(model_name: str, db: Database = Depends(get_db)):
    config = db.get_model_config(model_name)
    if not config:
        raise HTTPException(404, "Model not found")
    return {"success": True, "model_name": model_name, **config}

@admin_router.post("/models/{model_name}", summary="更新模型配置")
async def update_model_config_endpoint(model_name: str, request: dict, db: Database = Depends(get_db)):
    allowed = ['display_name', 'single_api_rpm_limit', 'single_api_tpm_limit', 'single_api_rpd_limit', 'status']
    update_data = {k: v for k, v in request.items() if k in allowed}

    # 验证 display_name
    if 'display_name' in update_data and not update_data['display_name'].strip():
        update_data['display_name'] = None  # 如果为空或只有空格，则设置为NULL

    if not update_data:
        raise HTTPException(422, "No valid fields to update")
    
    try:
        if db.update_model_config(model_name, **update_data):
            return {"success": True, "message": "Model config updated"}
        else:
            # rowcount 为 0 可能意味着模型不存在
            raise HTTPException(404, "Model not found or no changes made")
    except Exception as e:
        # 捕获可能的数据库唯一性约束错误
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(400, "Display name already exists.")
        raise HTTPException(500, f"Failed to update model config: {e}")

@admin_router.get("/config", summary="获取所有系统配置")
async def get_all_config_endpoint(db: Database = Depends(get_db)):
    return {
        "success": True,
        "system_configs": db.get_all_configs(),
        "thinking_config": db.get_thinking_config(),
        "inject_config": db.get_inject_prompt_config(),
        "cleanup_config": db.get_auto_cleanup_config(),
        "failover_config": db.get_failover_config(),
        "anti_detection_config": {'enabled': db.get_config('anti_detection_enabled', 'true').lower() == 'true'},
        "stream_mode_config": db.get_stream_mode_config(),
        "stream_to_gemini_mode_config": db.get_stream_to_gemini_mode_config(),
        "deepthink_config": db.get_deepthink_config(),
        "search_config": db.get_search_config(),
    }

@admin_router.post("/config/thinking", summary="更新思考模式配置")
async def update_thinking_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_thinking_config(
        enabled=request.get('enabled'),
        budget=request.get('budget'),
        include_thoughts=request.get('include_thoughts')
    )
    return {"success": True, "message": "Config updated"}

@admin_router.post("/config/inject-prompt", summary="更新提示词注入配置")
async def update_inject_prompt_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_inject_prompt_config(
        enabled=request.get('enabled'),
        content=request.get('content'),
        position=request.get('position')
    )
    return {"success": True, "message": "Config updated"}

@admin_router.post("/config/stream-mode", summary="更新流式模式配置")
async def update_stream_mode_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_stream_mode_config(mode=request.get('mode'))
    return {"success": True, "message": "Config updated"}

@admin_router.post("/config/stream-to-gemini-mode", summary="更新向 Gemini 流式模式配置")
async def update_stream_to_gemini_mode_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_stream_to_gemini_mode_config(mode=request.get("mode"))
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/deepthink", summary="获取DeepThink配置")
async def get_deepthink_config_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_deepthink_config()}

@admin_router.post("/config/deepthink", summary="更新DeepThink配置")
async def update_deepthink_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_deepthink_config(
        enabled=request.get('enabled'),
        concurrency=request.get('concurrency')
    )
    return {"success": True, "message": "Config updated"}

@admin_router.get("/config/search", summary="获取搜索配置")
async def get_search_config_endpoint(db: Database = Depends(get_db)):
    return {"success": True, "config": db.get_search_config()}

@admin_router.post("/config/search", summary="更新搜索配置")
async def update_search_config_endpoint(request: dict, db: Database = Depends(get_db)):
    db.set_search_config(
        enabled=request.get('enabled'),
        num_queries=request.get('num_queries'),
        num_pages_per_query=request.get('num_pages_per_query')
    )
    return {"success": True, "message": "Config updated"}

@admin_router.post("/config/load-balance", summary="更新负载均衡策略")
async def update_load_balance_config_endpoint(request: dict, db: Database = Depends(get_db)):
    strategy = request.get('load_balance_strategy')
    if strategy not in ['adaptive', 'least_used', 'round_robin']:
        raise HTTPException(422, "Invalid load balance strategy")
    db.set_config('load_balance_strategy', strategy)
    return {"success": True, "message": "Config updated"}

@admin_router.get("/stats", summary="获取管理统计信息")
async def get_admin_stats_endpoint(db: Database = Depends(get_db), anti_detection: GeminiAntiDetectionInjector = Depends(get_anti_detection), keep_alive_enabled: bool = Depends(get_keep_alive_enabled)):
    health_summary = db.get_keys_health_summary()
    return {
        "gemini_keys": len(db.get_all_gemini_keys()),
        "active_gemini_keys": len(db.get_available_gemini_keys()),
        "healthy_gemini_keys": health_summary.get('healthy', 0),
        "user_keys": len(db.get_all_user_keys()),
        "active_user_keys": len([k for k in db.get_all_user_keys() if k['status'] == 1]),
        "supported_models": db.get_supported_models(),
        "usage_stats": db.get_all_usage_stats(),
        "thinking_config": db.get_thinking_config(),
        "inject_config": db.get_inject_prompt_config(),
        "cleanup_config": db.get_auto_cleanup_config(),
        "health_summary": health_summary,
        "keep_alive_enabled": keep_alive_enabled,
        "anti_detection_enabled": db.get_config('anti_detection_enabled', 'true').lower() == 'true',
        "anti_detection_stats": anti_detection.get_statistics(),
        "stream_mode_config": db.get_stream_mode_config(),
        "stream_to_gemini_mode_config": db.get_stream_to_gemini_mode_config(),
        "failover_config": db.get_failover_config(),
        "deepthink_config": db.get_deepthink_config(),
        "search_config": db.get_search_config()
    }


@admin_router.get("/stats/hourly", summary="获取过去24小时每小时的统计数据")
async def get_hourly_stats(db: Database = Depends(get_db)):
    try:
        stats = db.get_hourly_stats_for_last_24_hours()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Failed to get hourly stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve hourly stats")

@admin_router.get("/logs/recent", summary="获取最近的请求日志")
async def get_recent_logs(limit: int = 100, db: Database = Depends(get_db)):
    try:
        logs = db.get_recent_usage_logs(limit=limit)
        return {"success": True, "logs": logs}
    except Exception as e:
        logger.error(f"Failed to get recent logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent logs")
