import asyncio
import json
import logging
import os
import threading
import uuid
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import httpx
from fastapi import HTTPException
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import credentials as google_credentials
from google_auth_oauthlib.flow import Flow

from database import Database

logger = logging.getLogger(__name__)

CLI_DEFAULT_CLIENT_ID = (
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
)
CLI_DEFAULT_CLIENT_SECRET = "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"
CLI_DEFAULT_REDIRECT_PATH = "/admin/cli-auth/callback"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com"
_MODEL_PREFIXES = ("models/", "tunedModels/", "cachedContents/")

CLI_DEFAULT_USER_AGENT = "GeminiCLIProxy/1.0"
CLI_CLIENT_HEADER = "gemini-cli-proxy/1.0"


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

DEFAULT_SCOPES = [
    "openid",
    "email",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/generative-language.retrieval",
]


class CliAuthManager:
    """Manage OAuth flows that emulate the Gemini CLI login."""

    def __init__(self) -> None:
        self._flows: Dict[str, Flow] = {}
        self._lock = threading.Lock()

    def _resolve_redirect_uri(self) -> str:
        explicit_redirect = os.getenv("GEMINI_CLI_REDIRECT_URI")
        if explicit_redirect:
            return explicit_redirect

        base_url = os.getenv("EXTERNAL_BASE_URL")
        if base_url:
            return base_url.rstrip("/") + CLI_DEFAULT_REDIRECT_PATH

        return f"http://localhost:8765{CLI_DEFAULT_REDIRECT_PATH}"

    def _build_client_config(self) -> Dict[str, Any]:
        client_id = os.getenv("GEMINI_CLI_CLIENT_ID", CLI_DEFAULT_CLIENT_ID)
        client_secret = os.getenv("GEMINI_CLI_CLIENT_SECRET", CLI_DEFAULT_CLIENT_SECRET)

        redirect_uri = self._resolve_redirect_uri()

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

    def start_authorization(self) -> Dict[str, str]:
        """Initialise the OAuth flow and return the authorization URL."""

        client_config = self._build_client_config()
        redirect_uri = client_config["web"]["redirect_uris"][0]

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

        with self._lock:
            self._flows[state] = flow

        logger.info("Started Gemini CLI OAuth flow with state %s", state)
        return {"authorization_url": auth_url, "state": state, "redirect_uri": redirect_uri}

    async def complete_authorization(
        self,
        state: str,
        *,
        code: Optional[str] = None,
        authorization_response: Optional[str] = None,
    ) -> google_credentials.Credentials:
        """Complete the OAuth flow and return Google credentials."""

        with self._lock:
            flow = self._flows.pop(state, None)

        if not flow:
            raise HTTPException(status_code=400, detail="Invalid or expired authorization state")

        def _fetch_token() -> None:
            if authorization_response:
                flow.fetch_token(authorization_response=authorization_response)
            elif code:
                flow.fetch_token(code=code)
            else:
                raise ValueError("Either authorization_response or code must be provided")

        try:
            await asyncio.to_thread(_fetch_token)
        except Exception as exc:  # pragma: no cover - pass through detailed errors
            logger.error("OAuth token exchange failed: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        credentials = flow.credentials
        if not credentials:
            raise HTTPException(status_code=500, detail="OAuth flow did not return credentials")

        logger.info("Completed Gemini CLI OAuth flow for state %s", state)
        return credentials


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
