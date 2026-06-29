"""Production PLC OPC UA CLIENT.

Connects to the PLC's own OPC UA server (not our server — we are the client).
Uses the synchronous ``opcua`` (python-opcua) library, matching the reference
implementation tested by the PLC engineer on the Jetson.

Only instantiated when mode='production'.  Demo mode is completely untouched.

Node map (all under the PLC server, grouped by section):
  ext_tuchabzug_status     Bool   WE READ   — ::TUA:toext_Tuchabzug_running
  ext_tuchabzug_stop       Bool   WE WRITE  — ::TUA:fromext_stop_Tuchabzug
  ext_error                UInt16 WE WRITE  — ::AsGlobalPV:fromext_Error_idx
  ext_viscontrol_alive     Bool   WE WRITE  — ::Signal:fromext_viscontrol_alive, livebit
  ext_error_quit           Bool   WE READ   — ::Signal:toext_Error_quit, operator ack
  ext_einlaufband_running  Bool   WE READ   — ::Einlauf:toext_Einlaufband_running

Thread model
------------
One background daemon thread owns ALL OPC UA reads and writes via a command
queue so there are never concurrent calls to the opcua client.  The main/GUI
thread only enqueues commands (non-blocking); the worker thread drains them and
then polls the readable node at ``poll_interval_s``.

The stop-pulse command sends True, sleeps ``stop_pulse_ms``, then sends False —
all on the worker thread so the GUI never blocks.

The livebit (ext_viscontrol_alive) is toggled by the worker thread itself every
1 second using a monotonic clock, with no separate thread required.  It resumes
automatically after reconnect.

Error latching: VisControl writes ext_error = fault_error_code once when a
belt fault is confirmed and then holds it — it is never auto-cleared when the
belt goes clean. It is only reset to 0 on the rising edge of ext_error_quit
(operator acknowledge), via the ``on_error_quit_change`` callback fired from
the worker thread (see MainWindow._on_plc_error_quit).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

from viscontrol.core.logger import logger

# Command tokens for the internal queue
_CMD_STOP_PULSE = "stop_pulse"
_CMD_SET_ERROR = "set_error"
_CMD_CLEAR_ERROR = "clear_error"
_SENTINEL = None  # stop_event wakeup / shutdown sentinel

_LIVEBIT_INTERVAL_S: float = 1.0


class PlcInterface:
    """Duck-typed interface satisfied by both PlcClient and DemoPlcStub.

    Defined as a plain base class (not Protocol) so isinstance() checks work
    without requiring typing_extensions on older Pythons.
    """

    def send_stop_pulse(self) -> None: ...
    def set_error(self, code: int) -> None: ...
    def clear_error(self) -> None: ...
    def read_tuchabzug_status(self) -> bool: ...
    def read_einlaufband_running(self) -> bool: ...


class DemoPlcStub(PlcInterface):
    """No-op stub for demo mode.

    Demo state is driven entirely by the Force-toggle and Simulate-Pulse
    buttons, which call state-machine methods directly — unchanged.  This
    stub exists only to satisfy the PlcInterface type contract.
    """

    def send_stop_pulse(self) -> None:
        pass

    def set_error(self, code: int) -> None:
        pass

    def clear_error(self) -> None:
        pass

    def read_tuchabzug_status(self) -> bool:
        return False

    def read_einlaufband_running(self) -> bool:
        return False


class PlcClient(PlcInterface):
    """OPC UA client that connects to the PLC server and drives the production flow.

    Lifecycle
    ---------
    Construct → start() → [use] → stop()

    Callbacks (set by MainWindow after construction)
    ------------------------------------------------
    ``on_tuchabzug_change(bool)``  — fired on rising or falling edge of
        ext_tuchabzug_status from the worker thread.
    """

    def __init__(
        self,
        url: str,
        node_ids: dict[str, str],
        *,
        poll_interval_s: float = 0.1,
        stop_pulse_ms: int = 100,
        reconnect_delay_s: float = 2.0,
    ) -> None:
        self._url = url
        self._node_ids = node_ids
        self._poll_interval = poll_interval_s
        self._stop_pulse_s = stop_pulse_ms / 1000.0
        self._reconnect_delay = reconnect_delay_s

        # OPC UA state (only touched by the worker thread)
        self._client = None
        self._nodes: dict = {}
        self._connected = False

        # Thread control
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._cmd_queue: queue.Queue = queue.Queue()

        # Cached readable values (updated by worker thread; safe to read from GUI thread)
        self._cached_tuchabzug: bool = False
        self._cached_einlaufband_running: bool = False

        # Callbacks injected by MainWindow
        self.on_tuchabzug_change: Optional[Callable[[bool], None]] = None
        # Fired on the rising edge of ext_error_quit (operator acknowledge).
        self.on_error_quit_change: Optional[Callable[[bool], None]] = None

    # ── public properties ────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("PlcClient already started")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="PlcClient", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._cmd_queue.put(_SENTINEL)  # wake up a blocked queue.get()
        self._thread.join(timeout=5.0)
        self._thread = None
        self._disconnect()

    # ── commands (called from GUI thread — non-blocking) ─────────────────────

    def send_stop_pulse(self) -> None:
        """Queue a stop pulse command (True → sleep → False on worker thread)."""
        self._cmd_queue.put((_CMD_STOP_PULSE, None))

    def set_error(self, code: int) -> None:
        """Queue writing ext_error = code."""
        self._cmd_queue.put((_CMD_SET_ERROR, int(code)))

    def clear_error(self) -> None:
        """Queue writing ext_error = 0."""
        self._cmd_queue.put((_CMD_CLEAR_ERROR, None))

    # ── readable cached values ────────────────────────────────────────────────

    def read_tuchabzug_status(self) -> bool:
        return self._cached_tuchabzug

    def read_einlaufband_running(self) -> bool:
        return self._cached_einlaufband_running

    # ── internals ─────────────────────────────────────────────────────────────

    def _disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._client = None
        self._nodes = {}
        self._connected = False

    def _connect(self) -> None:
        """Establish connection and resolve nodes. Raises on failure."""
        self._disconnect()
        try:
            from opcua import Client  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "python-opcua not installed. Run: pip install opcua"
            ) from exc
        self._client = Client(self._url)
        self._client.connect()
        self._nodes = {
            name: self._client.get_node(node_id)
            for name, node_id in self._node_ids.items()
        }
        self._connected = True

    def _ensure_connection_blocking(self) -> bool:
        """Block until connected (mirrors reference ``ensure_connection``).

        Returns True once connected, False if ``_stop_event`` is set.
        """
        while not self._stop_event.is_set():
            try:
                if not self._connected or self._client is None:
                    logger.info("PlcClient: connecting to {} ...", self._url)
                    self._connect()
                    logger.info("PlcClient: connected to {}", self._url)
                    return True
                else:
                    # Ping to confirm connection is still alive
                    self._nodes["ext_tuchabzug_status"].get_value()
                    return True
            except Exception as exc:
                self._connected = False
                logger.warning(
                    "PlcClient: connection failed: {}. Retry in {}s",
                    exc, self._reconnect_delay,
                )
                self._stop_event.wait(self._reconnect_delay)
        return False

    def _execute_command(self, cmd: tuple) -> None:
        """Execute one queued command on the worker thread.

        If the connection is lost during execution, ``_connected`` is set to
        False so the polling loop will reconnect before the next iteration.
        """
        kind, value = cmd
        if not self._connected:
            logger.warning("PlcClient: dropping {} command (disconnected)", kind)
            return
        try:
            if kind == _CMD_STOP_PULSE:
                self._do_stop_pulse()
            elif kind == _CMD_SET_ERROR:
                self._do_set_error(int(value))
            elif kind == _CMD_CLEAR_ERROR:
                self._do_set_error(0)
        except Exception as exc:
            logger.error("PlcClient: command {} failed: {}", kind, exc)
            self._connected = False

    def _do_stop_pulse(self) -> None:
        """Write ext_tuchabzug_stop True → sleep → False.

        Mirrors the reference ``pulse_stop_if_status_true``.
        Only fires if ext_tuchabzug_status is True.
        """
        try:
            from opcua import ua  # type: ignore[import]
        except ImportError:
            return
        status = bool(self._nodes["ext_tuchabzug_status"].get_value())
        if not status:
            logger.debug("PlcClient: stop pulse skipped (ext_tuchabzug_status=False)")
            return
        dv_true = ua.DataValue(ua.Variant(True, ua.VariantType.Boolean))
        dv_false = ua.DataValue(ua.Variant(False, ua.VariantType.Boolean))
        self._nodes["ext_tuchabzug_stop"].set_value(dv_true)
        logger.info("PlcClient: ext_tuchabzug_stop = True")
        time.sleep(self._stop_pulse_s)
        # Re-check connection after sleep before writing False
        if not self._connected:
            logger.warning("PlcClient: connection lost during stop pulse — skipping False write")
            return
        self._nodes["ext_tuchabzug_stop"].set_value(dv_false)
        logger.info("PlcClient: ext_tuchabzug_stop = False")

    def _do_set_error(self, code: int) -> None:
        try:
            from opcua import ua  # type: ignore[import]
        except ImportError:
            return
        if code < 0 or code > 65535:
            logger.error("PlcClient: ext_error value {} out of UInt16 range", code)
            return
        dv = ua.DataValue(ua.Variant(code, ua.VariantType.UInt16))
        self._nodes["ext_error"].set_value(dv)
        logger.info("PlcClient: ext_error = {}", code)

    def _do_write_livebit(self, value: bool) -> None:
        try:
            from opcua import ua  # type: ignore[import]
        except ImportError:
            return
        dv = ua.DataValue(ua.Variant(value, ua.VariantType.Boolean))
        self._nodes["ext_viscontrol_alive"].set_value(dv)
        logger.debug("PlcClient: ext_viscontrol_alive = {}", value)

    # ── worker thread ──────────────────────────────────────────────────────────

    def _thread_main(self) -> None:
        last_tuch: bool | None = None
        last_quit: bool | None = None
        _livebit: bool = False
        _last_livebit_time: float = 0.0  # forces immediate write after (re)connect

        while not self._stop_event.is_set():
            # Phase 1: ensure connected (blocks + retries until connected or stopped)
            if not self._connected:
                if not self._ensure_connection_blocking():
                    break  # stop_event set
                # Re-read initial values after (re)connect to avoid spurious edges
                last_tuch = None
                last_quit = None
                _last_livebit_time = 0.0

            # Phase 2: drain the command queue
            try:
                while True:
                    cmd = self._cmd_queue.get_nowait()
                    if cmd is _SENTINEL:
                        return
                    self._execute_command(cmd)
            except queue.Empty:
                pass

            # Phase 3: poll readable node + livebit heartbeat
            try:
                tuch = bool(self._nodes["ext_tuchabzug_status"].get_value())

                # Update cache (read by GUI thread via read_tuchabzug_status())
                self._cached_tuchabzug = tuch

                # Fire tuchabzug edge callbacks
                if last_tuch is None:
                    last_tuch = tuch
                elif tuch != last_tuch:
                    logger.info(
                        "PlcClient: ext_tuchabzug_status {} → {}",
                        last_tuch, tuch,
                    )
                    last_tuch = tuch
                    if self.on_tuchabzug_change is not None:
                        try:
                            self.on_tuchabzug_change(tuch)
                        except Exception:
                            logger.exception("on_tuchabzug_change callback error")

                # Einlaufband running — cached for read_einlaufband_running().
                self._cached_einlaufband_running = bool(
                    self._nodes["ext_einlaufband_running"].get_value()
                )

                # Error quit (operator acknowledge) — fire only on rising edge so a
                # held-down ack button doesn't repeatedly re-trigger the clear.
                quit_val = bool(self._nodes["ext_error_quit"].get_value())
                if last_quit is None:
                    last_quit = quit_val
                elif quit_val != last_quit:
                    last_quit = quit_val
                    if quit_val:
                        logger.info("PlcClient: ext_error_quit rising edge (operator ack)")
                        if self.on_error_quit_change is not None:
                            try:
                                self.on_error_quit_change(True)
                            except Exception:
                                logger.exception("on_error_quit_change callback error")

                # Livebit: toggle every 1 second on the worker thread
                now = time.monotonic()
                if now - _last_livebit_time >= _LIVEBIT_INTERVAL_S:
                    _livebit = not _livebit
                    self._do_write_livebit(_livebit)
                    _last_livebit_time = now

                self._stop_event.wait(self._poll_interval)

            except Exception as exc:
                logger.warning("PlcClient: poll error: {}", exc)
                self._connected = False

        self._disconnect()
        logger.info("PlcClient: worker thread exited")
