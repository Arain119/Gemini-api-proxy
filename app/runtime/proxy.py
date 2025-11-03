import asyncio
import logging

import httpx
import websockets
from fastapi import APIRouter, Request, Response, WebSocket
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState

from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from app.core.settings import settings


router = APIRouter()

logger = logging.getLogger(__name__)


async def _proxy_request(request: Request, sub_path: str) -> Response:
    target_url = str(
        request.url.replace(
            scheme="http",
            netloc=f"127.0.0.1:{settings.streamlit_internal_port}",
            path=sub_path,
        )
    )

    headers = dict(request.headers)
    headers.pop("host", None)
    body = await request.body()

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.request(
            request.method,
            target_url,
            content=body,
            headers=headers,
            params=request.query_params,
        )

    # 过滤 hop-by-hop 头
    excluded_headers = {
        "content-encoding",
        "transfer-encoding",
        "connection",
        "keep-alive",
    }
    response_headers = {
        key: value for key, value in resp.headers.items() if key.lower() not in excluded_headers
    }

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
    )


async def _proxy_websocket(websocket: WebSocket, sub_path: str) -> None:
    target_url = f"ws://127.0.0.1:{settings.streamlit_internal_port}{sub_path}"
    excluded_headers = {
        "host",
        "upgrade",
        "connection",
        "sec-websocket-key",
        "sec-websocket-protocol",
        "sec-websocket-version",
        "sec-websocket-extensions",
    }
    headers = [
        (key, value)
        for key, value in websocket.headers.items()
        if key.lower() not in excluded_headers
    ]

    subprotocols_header = websocket.headers.get("sec-websocket-protocol")
    subprotocols = (
        [proto.strip() for proto in subprotocols_header.split(",") if proto.strip()]
        if subprotocols_header
        else None
    )

    await websocket.accept(subprotocol=subprotocols[0] if subprotocols else None)

    try:
        async with websockets.connect(
            target_url,
            extra_headers=headers,
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
            subprotocols=subprotocols,
        ) as upstream:
            async def client_to_upstream() -> None:
                try:
                    while True:
                        message = await websocket.receive()
                        message_type = message.get("type")

                        if message_type == "websocket.disconnect":
                            await upstream.close()
                            break

                        text_data = message.get("text")
                        if text_data is not None:
                            await upstream.send(text_data)
                            continue

                        bytes_data = message.get("bytes")
                        if bytes_data is not None:
                            await upstream.send(bytes_data)
                except WebSocketDisconnect:
                    await upstream.close()

            async def upstream_to_client() -> None:
                try:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except (ConnectionClosed, ConnectionClosedOK):
                    await websocket.close()

            await asyncio.gather(client_to_upstream(), upstream_to_client())
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - 网络代理故障难以覆盖
        logger.error("Streamlit WebSocket 代理失败: %s", exc)
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close()


@router.api_route("/", methods=["GET", "HEAD", "OPTIONS"], include_in_schema=False)
async def proxy_root(request: Request):
    return await _proxy_request(request, "/")


@router.api_route("/admin{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_admin(request: Request, path: str):
    sub_path = path if path.startswith("/") else f"/{path}" if path else "/"
    return await _proxy_request(request, sub_path)


@router.api_route("/_stcore/{path:path}", methods=["GET", "POST", "OPTIONS", "HEAD"], include_in_schema=False)
async def proxy_streamlit_core(request: Request, path: str):
    target = f"/_stcore/{path}" if path else "/_stcore"
    return await _proxy_request(request, target)


@router.api_route("/streamlit/static{path:path}", methods=["GET"])
async def proxy_static(request: Request, path: str):
    target = f"/streamlit/static{path}" if path.startswith("/") else f"/streamlit/static/{path}"
    return await _proxy_request(request, target)


@router.websocket("/_stcore/stream")
async def proxy_streamlit_websocket(websocket: WebSocket):
    await _proxy_websocket(websocket, "/_stcore/stream")
