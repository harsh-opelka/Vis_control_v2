"""Application entry point.

Order of operations on startup (per build spec, step 15):

  1. Load config (default + local override) — picks up profile, language,
     orientation, calibration, mode.
  2. Configure logger.
  3. Build QApplication + install i18n translator.
  4. Detect unexpected reboot (shutdown marker presence).
  5. Build MainWindow — this also constructs camera / detector / SM and
     surfaces the startup banner.
  6. In Production mode, start the OPC UA server and read the initial
     ``TuchabzugRunning`` value to seed the SM state.
  7. Start the web sidecar if enabled.
  8. Log the ``startup`` event with the detected reason.
  9. Run the Qt event loop.
 10. On clean exit: mark shutdown, log the ``shutdown`` event, tear down
     background threads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from viscontrol.core.config import load_config
from viscontrol.core.event_log import Event, EventLog, EventType
from viscontrol.core.events import Mode
from viscontrol.core.logger import configure_logger, logger
from viscontrol.io.storage import detect_unexpected_reboot, mark_clean_shutdown


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _PROJECT_ROOT / "config"


def main() -> int:
    cfg = load_config(_CONFIG_DIR)
    log_dir = Path(cfg.storage.log_dir)
    configure_logger(
        log_dir,
        rotation_mb=cfg.storage.app_log_rotation_mb,
        keep_files=cfg.storage.app_log_keep_files,
    )
    logger.info("VisControl starting; mode={}", cfg.app.mode)

    event_log = EventLog(log_dir)
    was_unexpected, reason = detect_unexpected_reboot(log_dir)
    event_log.append(Event(
        event_type=EventType.STARTUP,
        profile_name=cfg.app.active_profile,
        extra={"reason": reason, "mode": cfg.app.mode},
    ))

    # Build QApplication.
    os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", ""))
    from PySide6.QtCore import QTimer  # noqa: PLC0415  (delay import for env handling)
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VisControl")
    app.setOrganizationName("OPELKA")

    from viscontrol.ui.i18n import install_translator  # noqa: PLC0415
    install_translator(cfg.app.language)  # type: ignore[arg-type]

    from viscontrol.ui.main_window import MainWindow  # noqa: PLC0415
    window = MainWindow(cfg, _CONFIG_DIR)

    # Production-mode services: PLC client + web (if enabled).
    # OpcuaServer (legacy server-mode bridge) is left dormant until removed.
    if cfg.app.mode == "production":
        try:
            from viscontrol.io.plc_client import PlcClient  # noqa: PLC0415
            plc = PlcClient(
                cfg.plc.url,
                {
                    "ext_tuchabzug_stop":      cfg.plc.node_ext_tuchabzug_stop,
                    "ext_tuchabzug_status":    cfg.plc.node_ext_tuchabzug_status,
                    "ext_error":               cfg.plc.node_ext_error,
                    "ext_viscontrol_alive":    cfg.plc.node_ext_viscontrol_alive,
                    "ext_error_quit":          cfg.plc.node_ext_error_quit,
                    "ext_einlaufband_running": cfg.plc.node_ext_einlaufband_running,
                },
                poll_interval_s=cfg.plc.poll_interval_s,
                stop_pulse_ms=cfg.plc.stop_pulse_ms,
                reconnect_delay_s=cfg.plc.reconnect_delay_s,
            )
            plc.start()
            window.attach_plc_client(plc)
            logger.info("PLC client started — connecting to {}", cfg.plc.url)
        except Exception:  # noqa: BLE001
            logger.exception("failed to start PLC client")

    if cfg.web.enabled:
        try:
            from viscontrol.io.web_server import WebServer  # noqa: PLC0415
            web = WebServer(
                port=cfg.web.port,
                password_hash=cfg.web.password_hash,
                log_dir=log_dir,
                today_csv_provider=event_log.today_path,
            )
            web.start()
            window.attach_web_server(web)
            logger.info("Web dashboard started on port {}", cfg.web.port)
        except Exception:  # noqa: BLE001
            logger.exception("failed to start web server")

    window.show()

    # If the spec's banner timeout is 0, hide immediately. Otherwise it
    # auto-hides via the QTimer set up in MainWindow.
    if cfg.ui.startup_banner_seconds <= 0:
        window.hide_startup_banner()

    def _on_about_to_quit() -> None:
        logger.info("VisControl shutting down")
        event_log.append(Event(
            event_type=EventType.SHUTDOWN,
            profile_name=cfg.app.active_profile,
        ))
        mark_clean_shutdown(log_dir)
        window.shutdown()

    app.aboutToQuit.connect(_on_about_to_quit)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
