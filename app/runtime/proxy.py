import asyncio
import inspect
import logging
from typing import Optional

import httpx
import websockets
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket
from fastapi import WebSocketDisconnect
from starlette.websockets import WebSocketState

from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from app.core.settings import settings


router = APIRouter()

logger = logging.getLogger(__name__)

_CONNECT_HEADERS_PARAM = (
    "additional_headers"
    if "additional_headers" in inspect.signature(websockets.connect).parameters
    else "extra_headers"
)


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

    resp: Optional[httpx.Response] = None
    last_exc: Optional[Exception] = None
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.request(
                    request.method,
                    target_url,
                    content=body,
                    headers=headers,
                    params=request.query_params,
                )
            break
        except httpx.ConnectError as exc:
            last_exc = exc
            await asyncio.sleep(0.5 * (attempt + 1))
        except httpx.HTTPError as exc:
            logger.error("Streamlit HTTP 代理失败: %s", exc)
            raise HTTPException(status_code=502, detail="上游管理界面请求失败") from exc

    if resp is None:
        logger.error("无法连接到 Streamlit 服务 (%s): %s", target_url, last_exc)
        raise HTTPException(status_code=503, detail="管理控制台暂不可用，请稍后再试。")

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

    last_exc: Optional[Exception] = None

    for attempt in range(5):
        try:
            async with websockets.connect(
                target_url,
                ping_interval=None,
                ping_timeout=None,
                max_size=None,
                subprotocols=subprotocols,
                **({_CONNECT_HEADERS_PARAM: headers} if headers else {}),
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
                return
        except (ConnectionRefusedError, OSError) as exc:
            last_exc = exc
            await asyncio.sleep(0.5 * (attempt + 1))
        except WebSocketDisconnect:
            return
        except Exception as exc:  # pragma: no cover - 网络代理故障难以覆盖
            logger.error("Streamlit WebSocket 代理失败: %s", exc)
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()
            return

    logger.error("Streamlit WebSocket 连接失败 (%s): %s", target_url, last_exc)
    if websocket.application_state == WebSocketState.CONNECTED:
        await websocket.close(code=1013)


@router.api_route("/", methods=["GET", "HEAD", "OPTIONS"], include_in_schema=False)
async def proxy_root(request: Request):
    return await _proxy_request(request, "/")


@router.api_route("/admin{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_admin(request: Request, path: str):
    sub_path = path if path.startswith("/") else f"/{path}" if path else "/"
    return await _proxy_request(request, sub_path)


@router.api_route(
    "/_stcore/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def proxy_streamlit_core(request: Request, path: str):
    target = f"/_stcore/{path}" if path else "/_stcore"
    return await _proxy_request(request, target)


@router.api_route("/static{path:path}", methods=["GET"], include_in_schema=False)
async def proxy_static_root(request: Request, path: str):
    target = f"/static{path}" if path else "/static"
    return await _proxy_request(request, target)


@router.api_route("/streamlit/static{path:path}", methods=["GET"])
async def proxy_static(request: Request, path: str):
    target = f"/streamlit/static{path}" if path.startswith("/") else f"/streamlit/static/{path}"
    return await _proxy_request(request, target)


@router.websocket("/_stcore/stream")
async def proxy_streamlit_websocket(websocket: WebSocket):
    await _proxy_websocket(websocket, "/_stcore/stream")
