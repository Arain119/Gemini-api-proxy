import httpx
from fastapi import APIRouter, Request, Response

from app.core.settings import settings


router = APIRouter()


async def _proxy_request(request: Request, sub_path: str) -> Response:
    target_url = request.url.replace(
        scheme="http",
        netloc=f"127.0.0.1:{settings.streamlit_internal_port}",
        path=sub_path,
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


@router.api_route("/admin{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_admin(request: Request, path: str):
    sub_path = path if path.startswith("/") else f"/{path}" if path else "/"
    return await _proxy_request(request, sub_path)


@router.api_route("/streamlit/static{path:path}", methods=["GET"])
async def proxy_static(request: Request, path: str):
    target = f"/streamlit/static{path}" if path.startswith("/") else f"/streamlit/static/{path}"
    return await _proxy_request(request, target)
