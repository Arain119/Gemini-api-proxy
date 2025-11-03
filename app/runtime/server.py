import asyncio
import logging
import signal

import uvicorn

from app.core.logging import configure_logging
from app.core.settings import settings

logger = logging.getLogger(__name__)


def run() -> None:
    """本地运行入口。"""
    configure_logging(logging.DEBUG if settings.debug else logging.INFO)
    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload_enabled,
    )


async def serve_async() -> None:
    """提供给 Render 启动脚本的协程入口。"""
    configure_logging(logging.DEBUG if settings.debug else logging.INFO)

    config = uvicorn.Config(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload_enabled,
    )
    server = uvicorn.Server(config)

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _signal_handler(*_: int) -> None:
        logger.info("Received shutdown signal")
        loop.create_task(server.shutdown())
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler, sig)

    await server.serve()
    await stop_event.wait()


if __name__ == "__main__":
    run()
