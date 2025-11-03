import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """统一配置日志格式。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

