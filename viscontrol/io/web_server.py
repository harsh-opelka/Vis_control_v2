"""Read-only FastAPI sidecar.

Spec is strict: GET endpoints only, no settings changes, optional basic-auth
behind a hashed password. The web server runs on its own thread so Qt is
never blocked.

Shared state (current frame jpeg, status snapshot, last log path) is passed
through thread-safe accessors. The main window calls
:meth:`set_latest_jpeg` and :meth:`set_status_snapshot` on every frame /
state change; the FastAPI routes simply read.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from viscontrol.core.logger import logger
from viscontrol.core.security import verify_pin


@dataclass
class StatusSnapshot:
    state: str = "WAITING"
    mode: str = "demo"
    profile_name: str = ""
    tuchabzug_running: bool = False
    stop_tuchabzug: bool = False
    fault_active: bool = False
    inspected: int = 0
    good: int = 0
    defects: int = 0
    last_inference_ms: float = 0.0
    extras: dict = field(default_factory=dict)


class WebServer:
    """FastAPI + uvicorn read-only sidecar."""

    def __init__(
        self,
        *,
        port: int,
        password_hash: str = "",
        log_dir: Path,
        today_csv_provider: Callable[[], Path],
    ) -> None:
        self._port = port
        self._password_hash = password_hash
        self._log_dir = log_dir
        self._today_csv_provider = today_csv_provider

        self._lock = threading.Lock()
        self._latest_jpeg: bytes = b""
        self._status = StatusSnapshot()
        self._thread: threading.Thread | None = None
        self._server = None  # uvicorn.Server
        self._ready = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_latest_jpeg(self, data: bytes) -> None:
        with self._lock:
            self._latest_jpeg = data

    def set_status_snapshot(self, snapshot: StatusSnapshot) -> None:
        with self._lock:
            self._status = snapshot

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("WebServer already started")
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="WebServer", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ---------- internals ----------

    def _run(self) -> None:
        try:
            import uvicorn
            from fastapi import Depends, FastAPI, HTTPException, Response, status
            from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
            from fastapi.security import HTTPBasic, HTTPBasicCredentials
        except ImportError:
            logger.error("FastAPI/uvicorn not installed; web server disabled")
            self._ready.set()
            return

        app = FastAPI(title="VisControl Dashboard", version="1.0")
        security = HTTPBasic(auto_error=False)

        def _auth(creds: HTTPBasicCredentials | None = Depends(security)) -> None:
            if not self._password_hash:
                return  # auth disabled
            if creds is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="auth required",
                    headers={"WWW-Authenticate": "Basic"},
                )
            if not verify_pin(creds.password, self._password_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="bad credentials",
                    headers={"WWW-Authenticate": "Basic"},
                )

        @app.get("/", response_class=HTMLResponse)
        def root(_=Depends(_auth)) -> HTMLResponse:  # noqa: ANN001
            with self._lock:
                snap = self._status
            html = _render_dashboard(snap)
            return HTMLResponse(html)

        @app.get("/image/latest.jpg")
        def latest_image(_=Depends(_auth)) -> Response:  # noqa: ANN001
            with self._lock:
                data = self._latest_jpeg
            if not data:
                return Response(status_code=204)
            return Response(content=data, media_type="image/jpeg")

        @app.get("/api/status.json")
        def status_json(_=Depends(_auth)) -> JSONResponse:  # noqa: ANN001
            with self._lock:
                snap = self._status
            return JSONResponse(
                {
                    "state": snap.state,
                    "mode": snap.mode,
                    "profile_name": snap.profile_name,
                    "signals": {
                        "TuchabzugRunning": snap.tuchabzug_running,
                        "StopTuchabzug": snap.stop_tuchabzug,
                        "FaultActive": snap.fault_active,
                    },
                    "counts": {
                        "inspected": snap.inspected,
                        "good": snap.good,
                        "defects": snap.defects,
                    },
                    "last_inference_ms": snap.last_inference_ms,
                }
            )

        @app.get("/logs/today.csv", response_class=PlainTextResponse)
        def today_csv(_=Depends(_auth)) -> PlainTextResponse:  # noqa: ANN001
            try:
                path = self._today_csv_provider()
                if not path.exists():
                    return PlainTextResponse("", media_type="text/csv")
                return PlainTextResponse(
                    path.read_text(encoding="utf-8"), media_type="text/csv"
                )
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=str(e)) from e

        config = uvicorn.Config(app, host="0.0.0.0", port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._ready.set()
        logger.info("Web dashboard listening on port {}", self._port)
        try:
            self._server.run()
        except Exception:  # noqa: BLE001
            logger.exception("uvicorn crashed")


def _render_dashboard(snap: StatusSnapshot) -> str:
    """Return a minimal auto-refreshing HTML dashboard."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VisControl — Dashboard</title>
<meta http-equiv="refresh" content="2">
<style>
 body {{ background: #FAFAF7; color: #2C2C2A; font-family: sans-serif; margin: 24px; }}
 h1 {{ color: #1B3E7F; margin-bottom: 4px; }}
 .sub {{ color: #5F5E5A; margin-bottom: 24px; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
         gap: 16px; margin-bottom: 24px; }}
 .card {{ background: white; border: 1px solid #E5E3DC; border-radius: 12px;
          padding: 16px; }}
 .label {{ color: #5F5E5A; font-size: 12px; text-transform: uppercase; }}
 .value {{ font-size: 28px; font-weight: 600; }}
 .state {{ font-size: 48px; font-weight: 800; }}
 .ok {{ color: #2D8659; }} .fault {{ color: #C8102E; }} .neutral {{ color: #5F5E5A; }}
 img {{ max-width: 100%; border-radius: 12px; border: 1px solid #E5E3DC; }}
</style>
</head>
<body>
<h1>VISCONTROL</h1>
<div class="sub">OPELKA · {snap.profile_name} · mode={snap.mode}</div>
<div class="state {_state_class(snap.state)}">{snap.state}</div>
<div class="grid">
  <div class="card"><div class="label">Inspected</div>
    <div class="value">{snap.inspected}</div></div>
  <div class="card"><div class="label">Good</div>
    <div class="value">{snap.good}</div></div>
  <div class="card"><div class="label">Defects</div>
    <div class="value">{snap.defects}</div></div>
  <div class="card"><div class="label">Inference</div>
    <div class="value">{snap.last_inference_ms:.1f} ms</div></div>
</div>
<div class="grid">
  <div class="card"><div class="label">TuchabzugRunning</div>
    <div class="value">{snap.tuchabzug_running}</div></div>
  <div class="card"><div class="label">StopTuchabzug</div>
    <div class="value">{snap.stop_tuchabzug}</div></div>
  <div class="card"><div class="label">FaultActive</div>
    <div class="value">{snap.fault_active}</div></div>
</div>
<img src="/image/latest.jpg" alt="latest frame"/>
</body>
</html>
"""


def _state_class(state: str) -> str:
    if state == "FAULT":
        return "fault"
    if state in ("READY",):
        return "ok"
    return "neutral"
