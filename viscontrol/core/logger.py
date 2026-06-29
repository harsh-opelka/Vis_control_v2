"""Loguru-based application logging.

Why a thin wrapper: every module imports ``logger`` from here so we can swap
backends (e.g. add Sentry) without touching call sites, and tests can redirect
output via ``configure_logger(log_dir=tmp_path)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

# Re-export so the rest of the app does ``from viscontrol.core.logger import logger``.
logger = _logger

_configured = False


def configure_logger(
    log_dir: Path,
    *,
    rotation_mb: int = 10,
    keep_files: int = 7,
    level: str = "INFO",
) -> None:
    """Install a stderr sink + a rotating ``app.log`` sink.

    Idempotent — re-calling replaces the previous sinks so tests can reconfigure
    on each run without leaking handlers.
    """
    global _configured
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        enqueue=False,
        backtrace=True,
        diagnose=False,
    )
    logger.add(
        log_dir / "app.log",
        level=level,
        rotation=f"{rotation_mb} MB",
        retention=keep_files,
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    _configured = True


def is_configured() -> bool:
    """True once :func:`configure_logger` has been called at least once."""
    return _configured
