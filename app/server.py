import asyncio
import logging
import os
import sys
import subprocess
import time
from asyncio import Queue
from contextlib import asynccontextmanager
from typing import Callable
from app.core.settings import settings

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.admin.api_routes import admin_router, router as api_router
from app.admin.api_services import (
    auto_cleanup_failed_keys,
    cleanup_database_records,
    record_hourly_health_check,
)
from app.admin.api_utils import (
    GeminiAntiDetectionInjector,
    RateLimitCache,
    keep_alive_ping,
)
from app.admin.cli_auth import close_cli_http_clients
from app.admin.database import Database
from app.admin.dependencies import (
    get_anti_detection,
    get_db,
    get_keep_alive_enabled,
    get_rate_limiter,
    get_request_count,
    get_start_time,
)
from app.runtime.streamlit import start_streamlit, stop_streamlit
from app.services.queue import request_queue
from app.runtime.proxy import router as streamlit_proxy_router

logger = logging.getLogger(__name__)

# 全局状态
request_count = 0
start_time = time.time()
scheduler: AsyncIOScheduler | None = None
keep_alive_enabled = False


def _get_env_keep_alive_default() -> str:
    value = os.getenv("ENABLE_KEEP_ALIVE")
    if value is None:
        value = "true" if os.getenv("RENDER") else "false"
    return str(value).lower()


ENV_KEEP_ALIVE_DEFAULT = _get_env_keep_alive_default()

streamlit_process: subprocess.Popen | None = None

# 初始化全局组件
db_queue: Queue = Queue(maxsize=10_000)
db = Database(db_queue=db_queue)
anti_detection = GeminiAntiDetectionInjector()
rate_limiter = RateLimitCache()


async def db_writer_worker(queue: Queue, db_instance: Database) -> None:
    logger.info("启动数据库写入后台任务")
    while True:
        try:
            task = await queue.get()
            if task is None:
                break

            operation, data = task
            if operation == "log_usage":
                db_instance.log_usage_sync(**data)

            queue.task_done()
        except asyncio.CancelledError:
            logger.info("数据库写入任务被取消")
            break
        except Exception as exc:
            logger.error("数据库写入任务异常: %s", exc)


def _build_scheduler(db_instance: Database) -> tuple[AsyncIOScheduler, int]:
    scheduler_instance = AsyncIOScheduler()
    keep_alive_interval = int(os.getenv("KEEP_ALIVE_INTERVAL", "10"))

    scheduler_instance.add_job(
        keep_alive_ping,
        "interval",
        minutes=keep_alive_interval,
        id="keep_alive",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )

    scheduler_instance.add_job(
        record_hourly_health_check,
        "interval",
        hours=1,
        id="hourly_health_check",
        max_instances=1,
        coalesce=True,
        kwargs={"db": db_instance},
    )

    scheduler_instance.add_job(
        auto_cleanup_failed_keys,
        "cron",
        hour=2,
        minute=0,
        id="daily_cleanup",
        max_instances=1,
        coalesce=True,
        kwargs={"db": db_instance},
    )

    scheduler_instance.add_job(
        cleanup_database_records,
        "cron",
        hour=3,
        minute=0,
        id="daily_db_cleanup",
        max_instances=1,
        coalesce=True,
        kwargs={"db": db_instance},
    )

    return scheduler_instance, keep_alive_interval


async def update_keep_alive_state(enabled: bool) -> None:
    global scheduler, keep_alive_enabled

    if enabled == keep_alive_enabled:
        return

    if enabled:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)

        scheduler, interval = _build_scheduler(db)
        scheduler.start()
        keep_alive_enabled = True
        logger.info("🟢 保活任务已启动，间隔 %s 分钟", interval)
        asyncio.create_task(keep_alive_ping())
    else:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
        scheduler = None
        keep_alive_enabled = False
        logger.info("🔴 保活任务已停止")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动中...")

    global streamlit_process

    db_worker_task = asyncio.create_task(db_writer_worker(db_queue, db))
    desired_state = str(
        db.get_config("keep_alive_enabled", ENV_KEEP_ALIVE_DEFAULT)
    ).lower() == "true"
    await update_keep_alive_state(desired_state)

    try:
        streamlit_process = start_streamlit()
        app.state.streamlit_process = streamlit_process
        logger.info("Streamlit 管理界面已启动，端口 %s", settings.streamlit_internal_port)
    except Exception as exc:  # pragma: no cover - 子进程启动失败不易单测
        streamlit_process = None
        app.state.streamlit_process = None
        logger.error("启动 Streamlit 失败: %s", exc)

    yield

    await update_keep_alive_state(False)
    await db_queue.put(None)
    await db_worker_task
    stop_streamlit(streamlit_process)
    streamlit_process = None
    app.state.streamlit_process = None
    await close_cli_http_clients()
    logger.info("服务已停止")


app = FastAPI(
    title="Gemini API Proxy 2.0",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.state.request_queue = request_queue


@app.middleware("http")
async def increment_request_counter(request: Request, call_next):
    global request_count
    response = await call_next(request)
    request_count += 1
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(admin_router)
app.include_router(streamlit_proxy_router)


def _override(dep: Callable, value: Callable):
    app.dependency_overrides[dep] = value


_override(get_db, lambda: db)
_override(get_start_time, lambda: start_time)
_override(get_request_count, lambda: request_count)
_override(get_keep_alive_enabled, lambda: keep_alive_enabled)
_override(get_anti_detection, lambda: anti_detection)
_override(get_rate_limiter, lambda: rate_limiter)

logger.info("✅ FastAPI 应用初始化完成")
