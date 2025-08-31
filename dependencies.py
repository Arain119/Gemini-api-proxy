from typing import Generator
from database import Database
from api_utils import GeminiAntiDetectionInjector, RateLimitCache

# Dependency stubs that will be overridden in the main application setup.

def get_db() -> Database:
    """Dependency for the database session."""
    raise NotImplementedError("get_db dependency must be overridden")

def get_start_time() -> float:
    """Dependency for the application start time."""
    raise NotImplementedError("get_start_time dependency must be overridden")

def get_request_count() -> int:
    """Dependency for the global request counter."""
    raise NotImplementedError("get_request_count dependency must be overridden")

def get_keep_alive_enabled() -> bool:
    """Dependency for the keep-alive status."""
    raise NotImplementedError("get_keep_alive_enabled dependency must be overridden")

def get_anti_detection() -> GeminiAntiDetectionInjector:
    """Dependency for the anti-detection injector instance."""
    raise NotImplementedError("get_anti_detection dependency must be overridden")

def get_rate_limiter() -> RateLimitCache:
    """Dependency for the rate limiter instance."""
    raise NotImplementedError("get_rate_limiter dependency must be overridden")
