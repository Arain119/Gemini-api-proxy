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
from dependencies import get_db, get_start_time, get_request_count, get_keep_alive_enabled, get_anti_detection, get_rate_limiter
from api_utils import GeminiAntiDetectionInjector, keep_alive_ping, RateLimitCache
from api_services import record_hourly_health_check, auto_cleanup_failed_keys, cleanup_database_records

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
# In Render environment, default to 'true', otherwise default to 'false'
default_keep_alive = 'true' if os.getenv('RENDER') else 'false'
keep_alive_enabled = os.getenv('ENABLE_KEEP_ALIVE', default_keep_alive).lower() == 'true'

# Initialize database and anti-detection injector
db_queue = Queue(maxsize=10000)
db = Database(db_queue=db_queue)
anti_detection = GeminiAntiDetectionInjector()
rate_limiter = RateLimitCache()

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    global scheduler, keep_alive_enabled
    logger.info("🚀 Service starting up...")

    # Start the database writer worker
    db_worker_task = asyncio.create_task(db_writer_worker(db_queue, db))

    # Initialize and start the scheduler on startup if keep-alive is enabled
    if keep_alive_enabled:
        scheduler = AsyncIOScheduler()
        
        keep_alive_interval = int(os.getenv('KEEP_ALIVE_INTERVAL', '10'))
        scheduler.add_job(
            keep_alive_ping, 'interval', minutes=keep_alive_interval,
            id='keep_alive', max_instances=1, coalesce=True, misfire_grace_time=30
        )
        
        scheduler.add_job(
            record_hourly_health_check, 'interval', hours=1,
            id='hourly_health_check', max_instances=1, coalesce=True,
            kwargs={'db': db}
        )
        
        scheduler.add_job(
            auto_cleanup_failed_keys, 'cron', hour=2, minute=0,
            id='daily_cleanup', max_instances=1, coalesce=True,
            kwargs={'db': db}
        )
        
        scheduler.add_job(
            cleanup_database_records, 'cron', hour=3, minute=0,
            id='daily_db_cleanup', max_instances=1, coalesce=True,
            kwargs={'db': db}
        )
        
        scheduler.start()
        logger.info(f"🟢 Keep-alive enabled (interval: {keep_alive_interval} minutes)")
        
        # Perform an initial ping
        asyncio.create_task(keep_alive_ping())
    else:
        logger.info("🔴 Keep-alive disabled")

    yield

    # Shutdown the scheduler on application exit
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down.")
    
    # Stop the database writer worker
    await db_queue.put(None)
    await db_worker_task
    logger.info("Database writer worker shut down.")

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

app.dependency_overrides[get_db] = _get_db
app.dependency_overrides[get_start_time] = _get_start_time
app.dependency_overrides[get_request_count] = _get_request_count
app.dependency_overrides[get_keep_alive_enabled] = _get_keep_alive_enabled
app.dependency_overrides[get_anti_detection] = _get_anti_detection
app.dependency_overrides[get_rate_limiter] = _get_rate_limiter

logger.info("✅ FastAPI app initialized with routes and dependencies.")
