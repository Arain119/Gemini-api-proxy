from fastapi import FastAPI, Request, Response

from .gemini_routes import router as gemini_router
from .openai_routes import router as openai_router


def _mount_preflight(app: FastAPI) -> None:
    @app.options("/{full_path:path}")
    async def handle_preflight(request: Request, full_path: str):  # noqa: WPS430
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            },
        )


def mount_proxy_routes(app: FastAPI) -> None:
    """在主应用中挂载 OpenAI/Gemini 兼容路由。"""
    _mount_preflight(app)

    app.include_router(openai_router)
    app.include_router(gemini_router)
