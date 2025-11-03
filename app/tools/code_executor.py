import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

RUNNER_PATH = Path(__file__).with_name("sandbox_runner.py")
DEFAULT_TIMEOUT = 5.0
MAX_CAPTURE_LENGTH = 4000


def _truncate(value: str) -> str:
    if len(value) <= MAX_CAPTURE_LENGTH:
        return value
    return value[:MAX_CAPTURE_LENGTH] + "\n...<truncated>"


async def execute_python_snippet(code: str, *, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Execute a Python snippet in a sandboxed subprocess.

    The snippet runs in isolated mode (``python -I``) with a restricted
    runner that limits available modules and builtins. Outputs are truncated
    to avoid overwhelming the caller.
    """

    if not code or not code.strip():
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": "代码内容为空"
        }

    env = {
        "PYTHONUNBUFFERED": "1",
        "PATH": os.getenv("PATH", ""),
    }

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        str(RUNNER_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(code.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        process.kill()
        return {
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": "代码执行超时"
        }

    if stderr:
        return {
            "ok": False,
            "stdout": "",
            "stderr": _truncate(stderr.decode("utf-8", "ignore")),
            "error": "代码执行失败"
        }

    try:
        payload = json.loads(stdout.decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        return {
            "ok": False,
            "stdout": _truncate(stdout.decode("utf-8", "ignore")),
            "stderr": "",
            "error": "沙盒返回数据异常"
        }

    result = {
        "ok": bool(payload.get("ok")),
        "stdout": _truncate(payload.get("stdout", "")),
        "stderr": _truncate(payload.get("stderr", "")),
    }

    if payload.get("error"):
        result["error"] = str(payload["error"])

    if payload.get("execution_time") is not None:
        result["execution_time"] = payload["execution_time"]

    return result
