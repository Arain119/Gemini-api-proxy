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
CLI_LOOPBACK_DEFAULT_HOST = "127.0.0.1"
CLI_LOOPBACK_DEFAULT_PORT = 8765
CLI_LOOPBACK_REDIRECT_PATH = "/oauth2callback"
CLI_REMOTE_CALLBACK_PATH = "/admin/cli-auth/callback"

SIGN_IN_SUCCESS_URL = "https://developers.google.com/gemini-code-assist/auth_success_gemini"
SIGN_IN_FAILURE_URL = "https://developers.google.com/gemini-code-assist/auth_failure_gemini"

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
    callback_url: Optional[str] = None
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


def _extract_quota_project(
    credentials: Optional[google_credentials.Credentials],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Best-effort extraction of the quota project id for CLI requests."""

    if metadata:
        for key in ("quota_project_id", "project_number", "project_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if not credentials:
        return None

    quota_project = getattr(credentials, "quota_project_id", None)
    if isinstance(quota_project, str) and quota_project.strip():
        return quota_project.strip()

    project_id = getattr(credentials, "project_id", None)
    if isinstance(project_id, str) and project_id.strip():
        return project_id.strip()

    try:
        info = json.loads(credentials.to_json())
    except Exception:  # pragma: no cover - defensive fallback
        info = {}

    for key in ("quota_project_id", "project_id", "project_number", "quota_project"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _normalize_scopes(scopes: Any) -> Optional[list]:
    """Normalize scopes into a list of unique, non-empty strings."""

    if not scopes:
        return None

    if isinstance(scopes, str):
        scopes = [scope.strip() for scope in scopes.split()]
    elif isinstance(scopes, (tuple, set)):
        scopes = list(scopes)

    if not isinstance(scopes, list):
        return None

    normalized = []
    seen = set()
    for scope in scopes:
        if not isinstance(scope, str):
            continue
        stripped = scope.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        normalized.append(stripped)
    return normalized if normalized else None


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

        quota_project_id = _extract_quota_project(credentials, metadata)
        if quota_project_id:
            headers["X-Goog-User-Project"] = quota_project_id
            if metadata.get("quota_project_id") != quota_project_id:
                metadata["quota_project_id"] = quota_project_id
                key_id = key_info.get("id")
                if key_id:
                    try:
                        db.update_gemini_key(key_id, metadata=metadata)
                    except Exception as exc:  # pragma: no cover - database errors logged downstream
                        logger.warning(
                            "Failed to persist quota project id for key %s: %s", key_id, exc
                        )

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

    quota_project_id = metadata.get("quota_project_id")
    if not quota_project_id:
        quota_project_candidate = _extract_quota_project(credentials, metadata)
        if quota_project_candidate:
            metadata["quota_project_id"] = quota_project_candidate
            metadata_changed = True

    if metadata_changed:
        key_id = key_info.get("id")
        if key_id:
            db.update_gemini_key(key_id, metadata=metadata)

    db.touch_cli_account(account_id)

# NOTE: Keep this list aligned with the scopes requested by the official
# gemini-cli project. Requesting additional scopes (including
# `https://www.googleapis.com/auth/generative-language`) makes Google OAuth
# return `restricted_client` errors for the public desktop client we rely on.
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


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

    quota_project_id = _extract_quota_project(credentials)

    metadata = {"cli_account_id": account_id}
    if email:
        metadata["account_email"] = email
    if quota_project_id:
        metadata["quota_project_id"] = quota_project_id
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


async def import_cli_credentials(
    *,
    db: Database,
    credentials_json: str,
    label: Optional[str],
) -> CliAuthCompleteResponse:
    """Store imported CLI OAuth credentials and register a corresponding Gemini key."""

    try:
        info = json.loads(credentials_json)
        # Basic validation for the refresh token
        if "refresh_token" not in info:
            raise ValueError("Invalid credentials format. Missing required key: 'refresh_token'.")

        # If client_id/secret are missing, inject the default ones used by gcloud/gemini-cli
        if "client_id" not in info:
            info["client_id"] = CLI_DEFAULT_CLIENT_ID
        if "client_secret" not in info:
            info["client_secret"] = CLI_DEFAULT_CLIENT_SECRET

        # The library expects 'token_uri' and 'scopes' for proper loading.
        if "token_uri" not in info:
            info["token_uri"] = "https://oauth2.googleapis.com/token"
        scopes = _normalize_scopes(info.get("scopes"))
        if scopes is None:
            scopes = list(DEFAULT_SCOPES)
        else:
            required = [scope for scope in DEFAULT_SCOPES if scope not in scopes]
            scopes.extend(required)
        info["scopes"] = scopes

        credentials = _load_credentials(json.dumps(info))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid credentials JSON: {exc}") from exc

    # We must refresh to verify the credentials and get a valid access token.
    if not credentials.refresh_token:
        raise HTTPException(status_code=400, detail="Credentials must include a refresh_token.")

    try:
        # Run refresh in a thread to avoid blocking the event loop
        await asyncio.to_thread(credentials.refresh, GoogleRequest())
    except Exception as exc:
        logger.error("Failed to refresh imported CLI credentials: %s", exc)
        raise HTTPException(status_code=401, detail=f"Failed to refresh imported credentials. They might be expired or invalid: {exc}") from exc

    access_token = getattr(credentials, "token", None)
    email = await fetch_account_email(access_token) if access_token else None
    
    # Use the refreshed credentials JSON for storage
    refreshed_credentials_json = credentials.to_json()

    try:
        account_id = db.create_cli_account(refreshed_credentials_json, email, label)
    except Exception as exc:
        logger.error("Failed to store imported CLI OAuth credentials: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store CLI credentials") from exc

    quota_project_id = _extract_quota_project(credentials)

    metadata = {"cli_account_id": account_id}
    if email:
        metadata["account_email"] = email
    if quota_project_id:
        metadata["quota_project_id"] = quota_project_id
    key_value = f"cli-account-{account_id}"

    if not db.add_gemini_key(key_value, source_type="cli_oauth", metadata=metadata):
        key_entry = db.get_gemini_key_by_value(key_value)
        if not key_entry:
            raise HTTPException(status_code=500, detail="Failed to register CLI-backed Gemini key after import")
    else:
        key_entry = db.get_gemini_key_by_value(key_value)

    if not key_entry:
        raise HTTPException(status_code=500, detail="Failed to load CLI-backed Gemini key after import")

    existing_metadata = dict(key_entry.get("metadata") or {})
    merged_metadata = {**existing_metadata, **metadata}
    if merged_metadata != existing_metadata:
        db.update_gemini_key(key_entry["id"], metadata=merged_metadata)
        key_entry = db.get_gemini_key_by_value(key_value)

    if email:
        db.update_cli_account_credentials(account_id, refreshed_credentials_json, email)

    return CliAuthCompleteResponse(
        account_id=account_id,
        gemini_key_id=key_entry["id"],
        state="imported",
        account_email=email,
    )


def _load_credentials(serialized: str) -> google_credentials.Credentials:
    info = json.loads(serialized)
    scopes = _normalize_scopes(info.get("scopes")) or list(DEFAULT_SCOPES)
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
