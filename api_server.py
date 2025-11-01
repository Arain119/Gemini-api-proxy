import asyncio
import time
import logging
import os
import sys
from contextlib import asynccontextmanager
from asyncio import Queue

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from database import Database
from api_routes import router as api_router, admin_router
from dependencies import (
    get_db,
    get_start_time,
    get_request_count,
    get_keep_alive_enabled,
    get_anti_detection,
    get_rate_limiter,
    get_cli_auth_manager,
)
from api_utils import GeminiAntiDetectionInjector, keep_alive_ping, RateLimitCache
from api_services import record_hourly_health_check, auto_cleanup_failed_keys, cleanup_database_records
from cli_auth import CliAuthManager, close_cli_http_clients

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Global variables
request_count = 0
start_time = time.time()
scheduler = None


def _get_env_keep_alive_default() -> str:
    """Return the keep-alive default derived from environment variables."""
    value = os.getenv('ENABLE_KEEP_ALIVE')
    if value is None:
        value = 'true' if os.getenv('RENDER') else 'false'
    return str(value).lower()


ENV_KEEP_ALIVE_DEFAULT = _get_env_keep_alive_default()
keep_alive_enabled = False

# Initialize database and anti-detection injector
db_queue = Queue(maxsize=10000)
db = Database(db_queue=db_queue)
anti_detection = GeminiAntiDetectionInjector()
rate_limiter = RateLimitCache()
cli_auth_manager = CliAuthManager(database_factory=lambda: db)

async def db_writer_worker(queue: Queue, db_instance: Database):
    """A worker that processes database write operations from a queue."""
    logger.info("Starting database writer worker...")
    while True:
        try:
            task = await queue.get()
            if task is None:  # Sentinel for stopping the worker
                break
            
            operation, data = task
            if operation == "log_usage":
                db_instance.log_usage_sync(**data)
            
            queue.task_done()
        except asyncio.CancelledError:
            logger.info("Database writer worker cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in db_writer_worker: {e}")

def _build_scheduler(db_instance: Database):
    """Create and configure the APScheduler instance used for keep-alive tasks."""
    scheduler_instance = AsyncIOScheduler()
    keep_alive_interval = int(os.getenv('KEEP_ALIVE_INTERVAL', '10'))

    scheduler_instance.add_job(
        keep_alive_ping, 'interval', minutes=keep_alive_interval,
        id='keep_alive', max_instances=1, coalesce=True, misfire_grace_time=30
    )

    scheduler_instance.add_job(
        record_hourly_health_check, 'interval', hours=1,
        id='hourly_health_check', max_instances=1, coalesce=True,
        kwargs={'db': db_instance}
    )

    scheduler_instance.add_job(
        auto_cleanup_failed_keys, 'cron', hour=2, minute=0,
        id='daily_cleanup', max_instances=1, coalesce=True,
        kwargs={'db': db_instance}
    )

    scheduler_instance.add_job(
        cleanup_database_records, 'cron', hour=3, minute=0,
        id='daily_db_cleanup', max_instances=1, coalesce=True,
        kwargs={'db': db_instance}
    )

    return scheduler_instance, keep_alive_interval


async def update_keep_alive_state(enabled: bool) -> None:
    """Enable or disable keep-alive background tasks at runtime."""
    global scheduler, keep_alive_enabled

    if enabled == keep_alive_enabled:
        logger.info(
            "Keep-alive is already %s.",
            "enabled" if keep_alive_enabled else "disabled"
        )
        return

    if enabled:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)

        scheduler, interval = _build_scheduler(db)
        scheduler.start()
        keep_alive_enabled = True
        logger.info(f"🟢 Keep-alive enabled (interval: {interval} minutes)")

        # Perform an initial ping in the background
        asyncio.create_task(keep_alive_ping())
    else:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down.")

        scheduler = None
        keep_alive_enabled = False
        logger.info("🔴 Keep-alive disabled")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info("🚀 Service starting up...")

    # Start the database writer worker
    db_worker_task = asyncio.create_task(db_writer_worker(db_queue, db))

    desired_keep_alive_state = str(
        db.get_config('keep_alive_enabled', ENV_KEEP_ALIVE_DEFAULT)
    ).lower() == 'true'
    await update_keep_alive_state(desired_keep_alive_state)

    yield

    # Shutdown the scheduler and related background tasks
    await update_keep_alive_state(False)

    # Stop the database writer worker
    await db_queue.put(None)
    await db_worker_task
    logger.info("Database writer worker shut down.")

    await close_cli_http_clients()

    logger.info("👋 Service shutting down...")

# Create FastAPI app instance
app = FastAPI(
    title="Gemini API Proxy",
    version="1.6.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.middleware("http")
async def increment_request_counter(request: Request, call_next):
    global request_count
    try:
        response = await call_next(request)
    except Exception:
        request_count += 1
        raise
    else:
        request_count += 1
        return response

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)
app.include_router(admin_router)

# Dependency overrides
def _get_db():
    return db

def _get_start_time():
    return start_time

def _get_request_count():
    return request_count

def _get_keep_alive_enabled():
    return keep_alive_enabled

def _get_anti_detection():
    return anti_detection

def _get_rate_limiter():
    return rate_limiter

def _get_cli_auth_manager():
    return cli_auth_manager

app.dependency_overrides[get_db] = _get_db
app.dependency_overrides[get_start_time] = _get_start_time
app.dependency_overrides[get_request_count] = _get_request_count
app.dependency_overrides[get_keep_alive_enabled] = _get_keep_alive_enabled
app.dependency_overrides[get_anti_detection] = _get_anti_detection
app.dependency_overrides[get_rate_limiter] = _get_rate_limiter
app.dependency_overrides[get_cli_auth_manager] = _get_cli_auth_manager

logger.info("✅ FastAPI app initialized with routes and dependencies.")
