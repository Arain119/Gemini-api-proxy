import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _get_list(name: str, default: List[str]) -> List[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "gemini-api-proxy"))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    debug: bool = field(default_factory=lambda: _get_bool("DEBUG", False))

    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_int("PORT", 8000))
    reload_enabled: bool = field(default_factory=lambda: _get_bool("RELOAD", False))

    admin_auth_token: str = field(default_factory=lambda: os.getenv("ADMIN_AUTH_TOKEN", "admin-secret"))
    gemini_auth_password: str = field(default_factory=lambda: os.getenv("GEMINI_AUTH_PASSWORD", "123456"))

    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./proxy.db"))
    sqlite_path: str = field(default_factory=lambda: os.getenv("SQLITE_PATH", "./proxy.db"))

    queue_max_concurrency: int = field(default_factory=lambda: _get_int("QUEUE_MAX_CONCURRENCY", 4))
    queue_max_wait_ms: int = field(default_factory=lambda: _get_int("QUEUE_MAX_WAIT_MS", 30_000))
    queue_poll_interval_ms: int = field(default_factory=lambda: _get_int("QUEUE_POLL_INTERVAL_MS", 50))

    model_thinking_default: int = field(default_factory=lambda: _get_int("DEFAULT_THINKING_BUDGET", -1))

    streamlit_internal_port: int = field(default_factory=lambda: _get_int("STREAMLIT_INTERNAL_PORT", 7000))
    api_base_url: Optional[str] = field(default_factory=lambda: os.getenv("API_BASE_URL"))
    streamlit_base_url: Optional[str] = field(default_factory=lambda: os.getenv("STREAMLIT_BASE_URL"))
    render_external_url: Optional[str] = field(default_factory=lambda: os.getenv("RENDER_EXTERNAL_URL"))
    cli_redirect_base_url: Optional[str] = field(default_factory=lambda: os.getenv("CLI_REDIRECT_BASE_URL"))

    cors_origins: List[str] = field(default_factory=lambda: _get_list("CORS_ORIGINS", ["*"]))

    @property
    def resolved_api_base_url(self) -> str:
        if self.api_base_url:
            return self.api_base_url.rstrip("/")
        if self.render_external_url:
            return self.render_external_url.rstrip("/")
        hostname = os.getenv("RENDER_SERVICE_URL")
        if hostname:
            return hostname.rstrip("/")
        host = (self.host or "").strip()
        if host in {"0.0.0.0", "::", "0:0:0:0", "", "localhost"}:
            return f"http://127.0.0.1:{self.port}"
        return f"http://{host}:{self.port}"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
