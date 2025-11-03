import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

from fastapi import HTTPException

from app.core.settings import settings


class QueueToken:
    def __init__(self, semaphore: asyncio.Semaphore):
        self._semaphore = semaphore
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._semaphore.release()
            self._released = True

    async def __aenter__(self) -> "QueueToken":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()


class RequestQueue:
    """基于模型的轻量队列，控制并发与排队等待。"""

    def __init__(self, max_concurrency: int, wait_timeout_ms: int) -> None:
        self._max_concurrency = max(1, max_concurrency)
        self._wait_timeout = max(100, wait_timeout_ms) / 1000.0
        self._locks: Dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(self._max_concurrency)
        )

    @asynccontextmanager
    async def acquire(self, model_name: str) -> AsyncIterator[QueueToken]:
        semaphore = self._locks[model_name]
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=self._wait_timeout)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=429,
                detail=f"Request queue is full for model {model_name}. Please retry later.",
            ) from exc

        token = QueueToken(semaphore)
        try:
            yield token
        finally:
            token.release()


request_queue = RequestQueue(
    max_concurrency=settings.queue_max_concurrency,
    wait_timeout_ms=settings.queue_max_wait_ms,
)
