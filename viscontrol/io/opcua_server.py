"""OPC UA server bridge for the 3-variable PLC contract.

This is the canonical PLC-facing contract — see the README for the B&R
integrator. Three nodes live under namespace ``http://opelka.com/viscontrol``:

  - ``TuchabzugRunning``  (Bool, PLC writes, we read)
  - ``StopTuchabzug``     (Bool, we write, PLC reads)
  - ``FaultActive``       (Bool, we write, PLC reads)

We act as an OPC UA *server* (not a client) per the spec — the B&R PLC opens
a session to us. We poll ``TuchabzugRunning`` from a background thread and
invoke an injected callback on rising / falling edge so the rest of the app
can drive the state machine without knowing about asyncua.

Only used in Production mode. In Demo mode this server is never instantiated.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Optional

from viscontrol.core.logger import logger


class OpcuaServer:
    """Async OPC UA server, surfaced as a sync API."""

    NS_URI = "http://opelka.com/viscontrol"
    BROWSE_TUCH = "TuchabzugRunning"
    BROWSE_STOP = "StopTuchabzug"
    BROWSE_FAULT = "FaultActive"

    def __init__(
        self,
        endpoint: str,
        *,
        namespace: str | None = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        self._endpoint = endpoint
        self._namespace = namespace or self.NS_URI
        self._poll_interval = poll_interval_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._server = None
        self._var_tuchabzug = None
        self._var_stop = None
        self._var_fault = None
        self._last_tuch_value: bool | None = None

        # Set by the main window after construction.
        self.on_tuchabzug_change: Optional[Callable[[bool], None]] = None

        self._publish_lock = threading.Lock()
        self._pending_stop: bool | None = None
        self._pending_fault: bool | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._ready_event.is_set()

    def start(self) -> None:
        """Spin up the OPC UA server on its own asyncio thread."""
        if self._thread is not None:
            raise RuntimeError("OpcuaServer already started")
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="OpcuaServer", daemon=True,
        )
        self._thread.start()
        # Wait briefly for "ready" so the UI can show OPC UA dot accurately.
        self._ready_event.wait(timeout=10.0)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        self._ready_event.clear()

    def publish_outputs(self, *, stop_tuchabzug: bool, fault_active: bool) -> None:
        """Schedule a write of ``StopTuchabzug`` and ``FaultActive``.

        Thread-safe — the actual write happens on the OPC UA loop.
        """
        with self._publish_lock:
            self._pending_stop = bool(stop_tuchabzug)
            self._pending_fault = bool(fault_active)

    # ---------- internals ----------

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except Exception:  # noqa: BLE001
            logger.exception("OPC UA server crashed")
        finally:
            self._ready_event.set()  # release any waiter

    async def _serve(self) -> None:
        try:
            from asyncua import Server, ua  # type: ignore
        except ImportError as e:
            logger.error("asyncua not installed: {}", e)
            return
        self._server = Server()
        await self._server.init()
        self._server.set_endpoint(self._endpoint)
        ns = await self._server.register_namespace(self._namespace)

        objects = self._server.nodes.objects
        folder = await objects.add_object(ns, "VisControl")

        self._var_tuchabzug = await folder.add_variable(ns, self.BROWSE_TUCH, False)
        await self._var_tuchabzug.set_writable()  # PLC writes this
        self._var_stop = await folder.add_variable(ns, self.BROWSE_STOP, False)
        self._var_fault = await folder.add_variable(ns, self.BROWSE_FAULT, False)

        async with self._server:
            logger.info("OPC UA server listening at {}", self._endpoint)
            self._ready_event.set()
            while not self._stop_event.is_set():
                # 1) Sample TuchabzugRunning and fire callback on edge.
                try:
                    cur = bool(await self._var_tuchabzug.get_value())
                except Exception:  # noqa: BLE001
                    cur = self._last_tuch_value or False
                if self._last_tuch_value is None:
                    self._last_tuch_value = cur
                elif cur != self._last_tuch_value:
                    logger.info(
                        "OPC UA TuchabzugRunning {} -> {}", self._last_tuch_value, cur,
                    )
                    self._last_tuch_value = cur
                    if self.on_tuchabzug_change:
                        try:
                            self.on_tuchabzug_change(cur)
                        except Exception:  # noqa: BLE001
                            logger.exception("on_tuchabzug_change callback failed")
                # 2) Drain any pending publish_outputs.
                with self._publish_lock:
                    pending_stop = self._pending_stop
                    pending_fault = self._pending_fault
                    self._pending_stop = None
                    self._pending_fault = None
                try:
                    if pending_stop is not None:
                        await self._var_stop.write_value(pending_stop)
                    if pending_fault is not None:
                        await self._var_fault.write_value(pending_fault)
                except Exception:  # noqa: BLE001
                    logger.exception("OPC UA write failed")
                await asyncio.sleep(self._poll_interval)
        logger.info("OPC UA server stopped")
