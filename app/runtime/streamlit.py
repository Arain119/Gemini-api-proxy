import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from app.core.settings import settings


def _build_command() -> list[str]:
    entry_path = Path(__file__).resolve().parents[2] / "streamlit_app" / "main.py"
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(entry_path),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--server.port",
        str(settings.streamlit_internal_port),
        "--server.address",
        "127.0.0.1",
    ]


def start_streamlit() -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    env.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
    env.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
    env.setdefault("API_BASE_URL", settings.resolved_api_base_url)

    command = _build_command()

    process = subprocess.Popen(
        command,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    time.sleep(1.0)
    return process


def stop_streamlit(process: Optional[subprocess.Popen]) -> None:
    if not process:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
