import asyncio
import asyncio
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi import HTTPException
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import credentials as google_credentials
from google_auth_oauthlib.flow import Flow

from database import Database
from api_models import CliAuthCompleteResponse

logger = logging.getLogger(__name__)

CLI_DEFAULT_CLIENT_ID = (
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
)
CLI_DEFAULT_CLIENT_SECRET = "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"
CLI_DEFAULT_REDIRECT_PATH = "/admin/cli-auth/callback"
CLI_LOOPBACK_DEFAULT_HOST = "127.0.0.1"
CLI_LOOPBACK_DEFAULT_PORT = 8765

GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
_MODEL_PREFIXES = ("models/", "tunedModels/", "cachedContents/")

CLI_DEFAULT_USER_AGENT = "GeminiCLIProxy/1.0"
CLI_CLIENT_HEADER = "gemini-cli-proxy/1.0"


@dataclass
class CliAuthSession:
    """Track state for an in-flight CLI OAuth authorization."""

    flow: Flow
    redirect_uri: str
    mode: str
    loop: Optional[asyncio.AbstractEventLoop] = None
    loopback_host: Optional[str] = None
    loopback_port: Optional[int] = None
    loopback_server: Optional[ThreadingHTTPServer] = None
    loopback_thread: Optional[threading.Thread] = None
    authorization_response: Optional[str] = None
    error: Optional[str] = None
    event: threading.Event = field(default_factory=threading.Event)


_httpx_client: Optional[httpx.AsyncClient] = None
_httpx_client_lock = asyncio.Lock()


async def _get_shared_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        async with _httpx_client_lock:
            if _httpx_client is None:
                limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
                _httpx_client = httpx.AsyncClient(timeout=None, limits=limits)
    return _httpx_client


async def close_cli_http_clients() -> None:
    global _httpx_client
    async with _httpx_client_lock:
        if _httpx_client is not None:
            await _httpx_client.aclose()
            _httpx_client = None


def _normalize_source_type(key_info: Dict[str, Any]) -> str:
    source_type = (key_info.get("source_type") or "cli_api_key").lower()
    if source_type == "api_key":
        # Backward compatibility for legacy rows.
        return "cli_api_key"
    return source_type


def _base_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": CLI_DEFAULT_USER_AGENT,
        "X-Goog-Api-Client": CLI_CLIENT_HEADER,
    }


def normalize_cli_model_name(model_name: str) -> str:
    """Ensure the model name carries the full resource prefix.

    Gemini CLI 及其后端 API 期望模型以 `models/` 或 `tunedModels/` 等资源路径
    开头，而数据库和 OpenAI 兼容请求通常只会提供裸模型名（例如
    `gemini-1.5-flash`). 这里统一补全缺失的前缀，避免 404。"""

    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")

    normalized = model_name.strip()
    if normalized.startswith(_MODEL_PREFIXES):
        return normalized

    return f"models/{normalized}"


def resolve_cli_model_name(db: Optional[Database], model_name: str) -> str:
    """Normalize并校验模型是否在系统支持列表中。

    用户在前端列表中只能看到 `database.get_supported_models()` 返回的型号，
    因此这里复用同一来源进行一次校验，避免误传入未开放的模型（例如
    `gemini-2.0-flash`)."""

    normalized = normalize_cli_model_name(model_name)

    if db is None:
        return normalized

    try:
        supported_models = db.get_supported_models()
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.error("Failed to load supported models: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load supported models") from exc

    normalized_supported = {normalize_cli_model_name(model) for model in supported_models}

    if normalized not in normalized_supported:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not enabled")

    return normalized


def _snake_to_camel(name: str) -> str:
    if not name or "_" not in name:
        return name
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() or "_" for part in parts[1:])


def _serialize_cli_payload(value: Any) -> Any:
    """Recursively convert google-genai objects and snake_case keys to JSON data."""

    if value is None:
        return None

    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        value = value.model_dump()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            new_key = _snake_to_camel(key)
            result[new_key] = _serialize_cli_payload(item)
        return result

    if isinstance(value, list):
        return [_serialize_cli_payload(item) for item in value if item is not None]

    return value


async def _prepare_cli_headers(
    db: Database,
    key_info: Dict[str, Any],
) -> Tuple[Dict[str, str], Optional[int], Optional[google_credentials.Credentials], Dict[str, Any]]:
    """Resolve authentication headers for CLI-backed keys.

    Returns a tuple of (headers, account_id, credentials, metadata_copy).
    """

    headers = _base_headers()
    metadata = dict(key_info.get("metadata") or {})
    source_type = _normalize_source_type(key_info)

    if source_type == "cli_oauth":
        account_id_raw = metadata.get("cli_account_id")
        if account_id_raw is None:
            raise HTTPException(status_code=500, detail="CLI key missing account reference")
        try:
            account_id = int(account_id_raw)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail="Invalid CLI account reference") from exc

        credentials, account = await ensure_cli_credentials(db, account_id)
        headers["Authorization"] = f"Bearer {credentials.token}"

        account_email = account.get("account_email")
        if account_email and metadata.get("account_email") != account_email:
            metadata["account_email"] = account_email

        return headers, account_id, credentials, metadata

    key_value = key_info.get("key")
    if not key_value:
        raise HTTPException(status_code=500, detail="Gemini API key missing")

    headers["X-Goog-Api-Key"] = key_value
    return headers, None, None, metadata


async def _ensure_cli_account_metadata(
    db: Database,
    key_info: Dict[str, Any],
    metadata: Dict[str, Any],
    credentials: Optional[google_credentials.Credentials],
    account_id: Optional[int],
) -> None:
    if not account_id:
        return

    metadata_changed = False
    account_email = metadata.get("account_email")

    if not account_email and credentials and getattr(credentials, "token", None):
        email = await fetch_account_email(credentials.token)
        if email:
            metadata["account_email"] = email
            metadata_changed = True
            db.update_cli_account_credentials(account_id, credentials.to_json(), email)

    if metadata_changed:
        key_id = key_info.get("id")
        if key_id:
            db.update_gemini_key(key_id, metadata=metadata)

    db.touch_cli_account(account_id)

# NOTE: Keep this list aligned with the scopes requested by the official
# gemini-cli project. Asking for extra scopes (like the deprecated
# `generative-language.retrieval`) causes Google OAuth to return
# `invalid_scope` and blocks users from signing in.
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


class CliAuthManager:
    """Manage OAuth flows that emulate the Gemini CLI login."""

    def __init__(self, *, database_factory: Optional[Any] = None) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, CliAuthSession] = {}
        self._completed: Dict[str, Dict[str, Any]] = {}
        self._database_factory = database_factory
        raw_mode = (os.getenv("GEMINI_CLI_CALLBACK_MODE") or "loopback").strip().lower()
        self._callback_mode = raw_mode if raw_mode in {"loopback", "remote"} else "loopback"

    # ------------------------------------------------------------------
    # OAuth flow setup helpers
    # ------------------------------------------------------------------

    def _build_client_config(self, redirect_uri: str) -> Dict[str, Any]:
        client_id = (os.getenv("GEMINI_CLI_CLIENT_ID") or CLI_DEFAULT_CLIENT_ID).strip()
        client_secret = (os.getenv("GEMINI_CLI_CLIENT_SECRET") or CLI_DEFAULT_CLIENT_SECRET).strip()

        if not client_id:
            client_id = CLI_DEFAULT_CLIENT_ID
        if not client_secret:
            client_secret = CLI_DEFAULT_CLIENT_SECRET

        return {
            "web": {
                "client_id": client_id,
                "project_id": "gemini-cli-proxy",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_secret": client_secret,
                "redirect_uris": [redirect_uri],
            }
        }

    def _resolve_remote_redirect_uri(self) -> str:
        explicit_redirect = os.getenv("GEMINI_CLI_REDIRECT_URI")
        if explicit_redirect:
            explicit_redirect = explicit_redirect.strip()
        if explicit_redirect:
            return explicit_redirect

        raw_client_id = os.getenv("GEMINI_CLI_CLIENT_ID")
        raw_client_secret = os.getenv("GEMINI_CLI_CLIENT_SECRET")

        client_id = (raw_client_id or CLI_DEFAULT_CLIENT_ID).strip()
        client_secret = (raw_client_secret or CLI_DEFAULT_CLIENT_SECRET).strip()

        if not client_id:
            client_id = CLI_DEFAULT_CLIENT_ID
        if not client_secret:
            client_secret = CLI_DEFAULT_CLIENT_SECRET
        using_default_client = (
            client_id == CLI_DEFAULT_CLIENT_ID and client_secret == CLI_DEFAULT_CLIENT_SECRET
        )

        if using_default_client:
            return f"http://localhost:8765{CLI_DEFAULT_REDIRECT_PATH}"

        base_url = os.getenv("EXTERNAL_BASE_URL")
        if base_url:
            return base_url.rstrip("/") + CLI_DEFAULT_REDIRECT_PATH

        render_external_url = os.getenv("RENDER_EXTERNAL_URL")
        if render_external_url:
            return render_external_url.strip().rstrip("/") + CLI_DEFAULT_REDIRECT_PATH

        render_hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME")
        if render_hostname:
            hostname = render_hostname.strip()
            if not hostname:
                return f"http://localhost:8765{CLI_DEFAULT_REDIRECT_PATH}"
            if "://" in hostname:
                base = hostname
            else:
                base = f"https://{hostname}"
            return base.rstrip("/") + CLI_DEFAULT_REDIRECT_PATH

        return f"http://localhost:8765{CLI_DEFAULT_REDIRECT_PATH}"

    def start_authorization(self) -> Dict[str, Any]:
        """Initialise the OAuth flow and return the authorization URL."""

        mode = (os.getenv("GEMINI_CLI_CALLBACK_MODE") or self._callback_mode).strip().lower()
        if mode not in {"loopback", "remote"}:
            mode = "loopback"
        self._callback_mode = mode

        if mode == "remote":
            return self._start_remote_authorization()
        return self._start_loopback_authorization()

    def _start_remote_authorization(self) -> Dict[str, Any]:
        redirect_uri = self._resolve_remote_redirect_uri()
        client_config = self._build_client_config(redirect_uri)

        flow = Flow.from_client_config(
            client_config,
            scopes=DEFAULT_SCOPES,
            redirect_uri=redirect_uri,
        )

        auth_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        session = CliAuthSession(flow=flow, redirect_uri=redirect_uri, mode="remote")

        with self._lock:
            self._sessions[state] = session

        logger.info("Started Gemini CLI OAuth flow in remote mode with state %s", state)
        return {
            "authorization_url": auth_url,
            "state": state,
            "redirect_uri": redirect_uri,
            "mode": "remote",
            "auto_finalize": False,
        }

    def _start_loopback_authorization(self) -> Dict[str, Any]:
        host = (os.getenv("GEMINI_CLI_LOOPBACK_HOST") or CLI_LOOPBACK_DEFAULT_HOST).strip() or CLI_LOOPBACK_DEFAULT_HOST

        port_candidates = []
        env_port = os.getenv("GEMINI_CLI_LOOPBACK_PORT")
        if env_port:
            try:
                parsed = int(env_port)
                if 0 <= parsed <= 65535:
                    port_candidates.append(parsed)
            except ValueError:
                logger.warning("Invalid GEMINI_CLI_LOOPBACK_PORT value '%s' - falling back to defaults", env_port)

        if CLI_LOOPBACK_DEFAULT_PORT not in port_candidates:
            port_candidates.append(CLI_LOOPBACK_DEFAULT_PORT)
        port_candidates.append(0)

        handler_cls = self._create_loopback_handler()
        server: Optional[ThreadingHTTPServer] = None
        bound_port: Optional[int] = None

        for candidate in port_candidates:
            try:
                server = ThreadingHTTPServer((host, candidate), handler_cls)
            except OSError as exc:
                logger.warning("Failed to bind loopback server on %s:%s (%s)", host, candidate, exc)
                continue
            else:
                bound_port = server.server_address[1]
                break

        if not server or bound_port is None:
            raise HTTPException(status_code=500, detail="Failed to start loopback callback server")

        redirect_uri = f"http://{host}:{bound_port}/"
        client_config = self._build_client_config(redirect_uri)
        flow = Flow.from_client_config(
            client_config,
            scopes=DEFAULT_SCOPES,
            redirect_uri=redirect_uri,
        )

        auth_url, state = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )

        loop = asyncio.get_running_loop()
        session = CliAuthSession(
            flow=flow,
            redirect_uri=redirect_uri,
            mode="loopback",
            loop=loop,
            loopback_host=host,
            loopback_port=bound_port,
            loopback_server=server,
        )

        server.expected_state = state  # type: ignore[attr-defined]

        with self._lock:
            self._sessions[state] = session

        thread = threading.Thread(target=self._serve_loopback, args=(state, session), daemon=True)
        session.loopback_thread = thread
        thread.start()

        logger.info(
            "Started Gemini CLI OAuth flow in loopback mode with state %s on %s:%s",
            state,
            host,
            bound_port,
        )

        return {
            "authorization_url": auth_url,
            "state": state,
            "redirect_uri": redirect_uri,
            "mode": "loopback",
            "loopback_host": host,
            "loopback_port": bound_port,
            "auto_finalize": bool(self._database_factory),
        }

    def _create_loopback_handler(self):
        manager = self

        class LoopbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - reduce noise
                logger.debug("Loopback OAuth callback: " + format, *args)

            def do_GET(self) -> None:  # noqa: N802
                state = getattr(self.server, "expected_state", None)  # type: ignore[attr-defined]
                manager._process_loopback_request(self, state)

        return LoopbackHandler

    def _serve_loopback(self, state: str, session: CliAuthSession) -> None:
        server = session.loopback_server
        if not server:
            return

        try:
            server.serve_forever(poll_interval=0.5)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Loopback server error for state %s: %s", state, exc)
        finally:
            try:
                server.server_close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def _process_loopback_request(self, handler: BaseHTTPRequestHandler, state: Optional[str]) -> None:
        if not state:
            self._send_loopback_response(handler, success=False, message="无法识别的授权会话。")
            return

        parsed = urlparse(handler.path)
        params = parse_qs(parsed.query)
        error = params.get("error", [None])[0]
        error_description = params.get("error_description", [None])[0]

        host_header = handler.headers.get("Host")
        if not host_header:
            server_address = handler.server.server_address  # type: ignore[attr-defined]
            host_header = f"{server_address[0]}:{server_address[1]}"
        authorization_response = f"http://{host_header}{handler.path}"

        if error:
            message = error_description or "Google OAuth 返回错误"
            self._register_loopback_error(state, message)
            self._send_loopback_response(handler, success=False, message=message, error_code=error)
            return

        if self._register_loopback_response(state, authorization_response):
            self._send_loopback_response(handler, success=True, message="授权成功，您可以关闭此窗口。")
        else:
            self._send_loopback_response(handler, success=False, message="授权会话已过期或不存在。")

    def _send_loopback_response(
        self,
        handler: BaseHTTPRequestHandler,
        *,
        success: bool,
        message: str,
        error_code: Optional[str] = None,
    ) -> None:
        status = 200 if success else 400
        title = "Gemini CLI 登录成功" if success else "Gemini CLI 登录失败"
        detail = f"错误代码：{error_code}" if error_code else ""
        html = f"""
        <!DOCTYPE html>
        <html lang=\"zh-CN\">
            <head>
                <meta charset=\"utf-8\">
                <title>{title}</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 2rem; }}
                    h1 {{ color: {'#059669' if success else '#dc2626'}; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                <p>{message}</p>
                {f'<p>{detail}</p>' if detail else ''}
                <p>此页面可以关闭。</p>
            </body>
        </html>
        """

        handler.send_response(status)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.end_headers()
        handler.wfile.write(html.encode("utf-8"))

    def _register_loopback_response(self, state: str, authorization_response: str) -> bool:
        with self._lock:
            session = self._sessions.get(state)

        if not session or session.mode != "loopback":
            return False

        session.authorization_response = authorization_response
        session.error = None
        session.event.set()

        server = session.loopback_server
        if server:
            threading.Thread(target=server.shutdown, daemon=True).start()

        logger.info("Received loopback OAuth callback for state %s", state)

        if self._database_factory:
            self._schedule_auto_finalize(state, session)

        return True

    def _register_loopback_error(self, state: str, message: str) -> None:
        with self._lock:
            session = self._sessions.get(state)

        if session:
            session.error = message
            session.event.set()
            server = session.loopback_server
            if server:
                threading.Thread(target=server.shutdown, daemon=True).start()

        logger.error("Loopback OAuth callback failed for state %s: %s", state, message)
        self._store_completed_result(state, error=message)

    def _schedule_auto_finalize(self, state: str, session: CliAuthSession) -> None:
        if not self._database_factory or not session.loop:
            return

        try:
            asyncio.run_coroutine_threadsafe(self._auto_finalize(state), session.loop)
        except RuntimeError as exc:  # pragma: no cover - event loop closed
            logger.warning("Failed to schedule auto-finalization for state %s: %s", state, exc)

    async def _auto_finalize(self, state: str) -> None:
        try:
            credentials = await self.complete_authorization(state)
        except HTTPException as exc:
            self._store_completed_result(state, error=str(exc.detail))
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            self._store_completed_result(state, error=str(exc))
            return

        db = self._database_factory() if callable(self._database_factory) else self._database_factory
        if not db:
            self._store_completed_result(state, error="数据库不可用，无法自动完成授权")
            return

        try:
            response = await finalize_cli_oauth(db=db, credentials=credentials, label=None, state=state)
        except HTTPException as exc:
            self._store_completed_result(state, error=str(exc.detail))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to auto-finalize CLI OAuth for state %s: %s", state, exc)
            self._store_completed_result(state, error=str(exc))
        else:
            self._store_completed_result(state, response=response)

    def _store_completed_result(
        self,
        state: str,
        *,
        response: Optional[CliAuthCompleteResponse] = None,
        error: Optional[str] = None,
    ) -> None:
        if not response and not error:
            return

        with self._lock:
            if response:
                self._completed[state] = {"response": response}
            else:
                self._completed[state] = {"error": error}

    def record_completed_result(self, state: str, response: CliAuthCompleteResponse) -> None:
        """Expose completion storage for manual flows."""

        self._store_completed_result(state, response=response)

    def get_status(self, state: str) -> Dict[str, Any]:
        with self._lock:
            completed = self._completed.get(state)
            if completed:
                if completed.get("response"):
                    response: CliAuthCompleteResponse = completed["response"]
                    return {
                        "state": state,
                        "status": "completed",
                        "account_email": response.account_email,
                        "auto_finalize": True,
                        "result": response.dict(),
                    }
                return {
                    "state": state,
                    "status": "failed",
                    "message": completed.get("error"),
                    "auto_finalize": True,
                }

            session = self._sessions.get(state)
            if not session:
                return {"state": state, "status": "unknown", "auto_finalize": bool(self._database_factory)}

            if session.error:
                return {
                    "state": state,
                    "status": "failed",
                    "message": session.error,
                    "auto_finalize": bool(self._database_factory),
                }

            if session.authorization_response:
                return {
                    "state": state,
                    "status": "callback_received",
                    "auto_finalize": bool(self._database_factory),
                }

            return {
                "state": state,
                "status": "pending",
                "auto_finalize": bool(self._database_factory),
            }

    def pop_completed_result(self, state: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._completed.pop(state, None)

    async def complete_authorization(
        self,
        state: str,
        *,
        code: Optional[str] = None,
        authorization_response: Optional[str] = None,
    ) -> google_credentials.Credentials:
        """Complete the OAuth flow and return Google credentials."""

        with self._lock:
            session = self._sessions.get(state)

        if not session:
            completed = self._completed.get(state)
            if completed and completed.get("response"):
                raise HTTPException(status_code=409, detail="Authorization already completed")
            raise HTTPException(status_code=400, detail="Invalid or expired authorization state")

        if not authorization_response and session.authorization_response:
            authorization_response = session.authorization_response

        def _fetch_token() -> None:
            if authorization_response:
                session.flow.fetch_token(authorization_response=authorization_response)
            elif code:
                session.flow.fetch_token(code=code)
            else:
                raise ValueError("Either authorization_response or code must be provided")

        try:
            await asyncio.to_thread(_fetch_token)
        except Exception as exc:  # pragma: no cover - pass through detailed errors
            logger.error("OAuth token exchange failed for state %s: %s", state, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        credentials = session.flow.credentials
        if not credentials:
            raise HTTPException(status_code=500, detail="OAuth flow did not return credentials")

        with self._lock:
            self._sessions.pop(state, None)

        self._cleanup_session(session)

        logger.info("Completed Gemini CLI OAuth flow for state %s", state)
        return credentials

    def _cleanup_session(self, session: CliAuthSession) -> None:
        server = session.loopback_server
        if server:
            try:
                threading.Thread(target=server.shutdown, daemon=True).start()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

        thread = session.loopback_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)


async def fetch_account_email(access_token: str) -> Optional[str]:
    """Fetch the authenticated account's email using Google UserInfo API."""

    if not access_token:
        return None

    try:
        client = await _get_shared_httpx_client()
        response = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10.0,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("email")
        logger.warning(
            "Failed to fetch account email: status=%s, body=%s",
            response.status_code,
            response.text,
        )
    except Exception as exc:  # pragma: no cover - network failure
        logger.warning("UserInfo request failed: %s", exc)
    return None


async def finalize_cli_oauth(
    *,
    db: Database,
    credentials,
    label: Optional[str],
    state: str,
) -> CliAuthCompleteResponse:
    """Store CLI OAuth credentials and register a corresponding Gemini key."""

    access_token = getattr(credentials, "token", None)
    email = await fetch_account_email(access_token) if access_token else None
    credentials_json = credentials.to_json()

    try:
        account_id = db.create_cli_account(credentials_json, email, label)
    except Exception as exc:  # pragma: no cover - database errors are logged downstream
        logger.error("Failed to store CLI OAuth credentials: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store CLI credentials") from exc

    metadata = {"cli_account_id": account_id}
    if email:
        metadata["account_email"] = email
    key_value = f"cli-account-{account_id}"

    if not db.add_gemini_key(key_value, source_type="cli_oauth", metadata=metadata):
        key_entry = db.get_gemini_key_by_value(key_value)
        if not key_entry:
            raise HTTPException(status_code=500, detail="Failed to register CLI-backed Gemini key")
    else:
        key_entry = db.get_gemini_key_by_value(key_value)

    if not key_entry:
        raise HTTPException(status_code=500, detail="Failed to load CLI-backed Gemini key")

    existing_metadata = dict(key_entry.get("metadata") or {})
    merged_metadata = {**existing_metadata, **metadata}
    if merged_metadata != existing_metadata:
        db.update_gemini_key(key_entry["id"], metadata=merged_metadata)
        key_entry = db.get_gemini_key_by_value(key_value)

    if email:
        db.update_cli_account_credentials(account_id, credentials_json, email)

    return CliAuthCompleteResponse(
        account_id=account_id,
        gemini_key_id=key_entry["id"],
        state=state,
        account_email=email,
    )


def _load_credentials(serialized: str) -> google_credentials.Credentials:
    info = json.loads(serialized)
    scopes = info.get("scopes") or DEFAULT_SCOPES
    return google_credentials.Credentials.from_authorized_user_info(info, scopes=scopes)


async def ensure_cli_credentials(
    db: Database, account_id: int
) -> Tuple[google_credentials.Credentials, Dict[str, Any]]:
    """Load and refresh stored CLI credentials if required."""

    account = db.get_cli_account(account_id)
    if not account or account.get("status") != 1:
        raise HTTPException(status_code=503, detail="CLI account is not active")

    try:
        credentials = _load_credentials(account["credentials"])
    except Exception as exc:  # pragma: no cover - invalid data
        logger.error("Failed to load CLI credentials for account %s: %s", account_id, exc)
        raise HTTPException(status_code=500, detail="Stored credentials are invalid") from exc

    if credentials.expired and credentials.refresh_token:
        logger.info("Refreshing access token for CLI account %s", account_id)

        def _refresh() -> google_credentials.Credentials:
            credentials.refresh(GoogleRequest())
            return credentials

        try:
            await asyncio.to_thread(_refresh)
        except Exception as exc:  # pragma: no cover - network failure
            logger.error("Failed to refresh CLI credentials: %s", exc)
            raise HTTPException(status_code=401, detail="Failed to refresh CLI credentials") from exc

        db.update_cli_account_credentials(
            account_id,
            credentials.to_json(),
            account.get("account_email"),
        )

    if not credentials.valid:
        raise HTTPException(status_code=401, detail="CLI credentials are not valid")

    return credentials, account


async def call_gemini_with_cli_account(
    db: Database,
    key_info: Dict[str, Any],
    payload: Dict[str, Any],
    model_name: str,
    *,
    timeout: float,
) -> Dict[str, Any]:
    """Send a request to the Gemini API using Gemini CLI authentication."""

    headers, account_id, credentials, metadata = await _prepare_cli_headers(db, key_info)

    normalized_model = resolve_cli_model_name(db, model_name)
    url = f"{GEMINI_API_BASE}/v1beta/{normalized_model}:generateContent"
    timeout_config = httpx.Timeout(timeout, read=timeout)

    serialized_payload = _serialize_cli_payload(payload)

    try:
        client = await _get_shared_httpx_client()
        response = await client.post(url, json=serialized_payload, headers=headers, timeout=timeout_config)
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("CLI-backed request failed for key %s: %s", key_info.get("id"), exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="CLI credentials rejected by Google")

    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="CLI transport rate limited")

    if response.status_code >= 500:
        raise HTTPException(status_code=502, detail="Upstream Gemini service error")

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()

    if account_id:
        await _ensure_cli_account_metadata(db, key_info, metadata, credentials, account_id)

    return data


async def stream_gemini_with_cli_account(
    db: Database,
    key_info: Dict[str, Any],
    payload: Dict[str, Any],
    model_name: str,
    *,
    timeout: float,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream Gemini responses using Gemini CLI authentication."""

    headers, account_id, credentials, metadata = await _prepare_cli_headers(db, key_info)

    normalized_model = resolve_cli_model_name(db, model_name)
    url = f"{GEMINI_API_BASE}/v1beta/{normalized_model}:streamGenerateContent?alt=sse"
    timeout_config = httpx.Timeout(timeout, read=timeout)

    serialized_payload = _serialize_cli_payload(payload)

    try:
        client = await _get_shared_httpx_client()
        async with client.stream("POST", url, json=serialized_payload, headers=headers, timeout=timeout_config) as response:
            if response.status_code == 401:
                raise HTTPException(status_code=401, detail="CLI credentials rejected by Google")
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail="CLI transport rate limited")
            if response.status_code >= 500:
                raise HTTPException(status_code=502, detail="Upstream Gemini service error")
            if response.status_code >= 400:
                content_bytes = await response.aread()
                detail = content_bytes.decode("utf-8", "ignore") if content_bytes else ""
                raise HTTPException(status_code=response.status_code, detail=detail)

            async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload_text = line[5:].strip()
                    if not payload_text or payload_text == "[DONE]":
                        if payload_text == "[DONE]":
                            break
                        continue
                    try:
                        event = json.loads(payload_text)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON SSE payload: %s", payload_text)
                        continue
                    yield event
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("CLI streaming request failed for key %s: %s", key_info.get("id"), exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        if account_id:
            await _ensure_cli_account_metadata(db, key_info, metadata, credentials, account_id)


async def embed_with_cli_account(
    db: Database,
    key_info: Dict[str, Any],
    payload: Dict[str, Any],
    model_name: str,
    *,
    timeout: float,
) -> Dict[str, Any]:
    """Call the Gemini embedContent API using Gemini CLI authentication."""

    headers, account_id, credentials, metadata = await _prepare_cli_headers(db, key_info)
    normalized_model = resolve_cli_model_name(db, model_name)
    url = f"{GEMINI_API_BASE}/v1beta/{normalized_model}:embedContent"
    timeout_config = httpx.Timeout(timeout, read=timeout)

    serialized_payload = _serialize_cli_payload(payload)

    try:
        client = await _get_shared_httpx_client()
        response = await client.post(url, json=serialized_payload, headers=headers, timeout=timeout_config)
    except Exception as exc:  # pragma: no cover - network failure
        logger.error("CLI embedding request failed for key %s: %s", key_info.get("id"), exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="CLI credentials rejected by Google")
    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="CLI transport rate limited")
    if response.status_code >= 500:
        raise HTTPException(status_code=502, detail="Upstream Gemini service error")
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()

    if account_id:
        await _ensure_cli_account_metadata(db, key_info, metadata, credentials, account_id)

    return data


async def upload_file_with_cli_account(
    db: Database,
    key_info: Dict[str, Any],
    *,
    filename: str,
    mime_type: str,
    file_content: bytes,
    timeout: float,
) -> Dict[str, Any]:
    """Upload a file via the Gemini CLI transport."""

    headers, account_id, credentials, metadata = await _prepare_cli_headers(db, key_info)
    boundary = f"----GeminiCLI{uuid.uuid4().hex}"
    metadata_part = json.dumps({
        "file": {
            "displayName": filename,
            "mimeType": mime_type,
        }
    })

    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{metadata_part}\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: {mime_type}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n\r\n"
    ).encode("utf-8") + file_content + f"\r\n--{boundary}--\r\n".encode("utf-8")

    headers = dict(headers)
    headers["Content-Type"] = f"multipart/related; boundary={boundary}"

    url = f"{GEMINI_API_BASE}/upload/v1beta/files?uploadType=multipart"
    timeout_config = httpx.Timeout(timeout, read=timeout)

    try:
        client = await _get_shared_httpx_client()
        response = await client.post(url, content=body, headers=headers, timeout=timeout_config)
    except Exception as exc:  # pragma: no cover
        logger.error("CLI file upload failed for key %s: %s", key_info.get("id"), exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()

    if account_id:
        await _ensure_cli_account_metadata(db, key_info, metadata, credentials, account_id)

    return data


async def delete_file_with_cli_account(
    db: Database,
    key_info: Dict[str, Any],
    *,
    file_uri: str,
    timeout: float,
) -> None:
    """Delete a file via the Gemini CLI transport."""

    headers, account_id, credentials, metadata = await _prepare_cli_headers(db, key_info)
    headers = dict(headers)
    headers.setdefault("Content-Type", "application/json")

    normalized_uri = file_uri
    if normalized_uri.startswith("https://"):
        # Extract the path part after the base URL
        normalized_uri = normalized_uri.split("/v1beta/")[-1]

    if not normalized_uri.startswith("files/"):
        normalized_uri = f"files/{normalized_uri}" if not normalized_uri.startswith("files") else normalized_uri

    url = f"{GEMINI_API_BASE}/v1beta/{normalized_uri}"
    timeout_config = httpx.Timeout(timeout, read=timeout)

    try:
        client = await _get_shared_httpx_client()
        response = await client.delete(url, headers=headers, timeout=timeout_config)
    except Exception as exc:  # pragma: no cover
        logger.error("CLI file deletion failed for key %s: %s", key_info.get("id"), exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text or "Failed to delete file")

    if account_id:
        await _ensure_cli_account_metadata(db, key_info, metadata, credentials, account_id)
