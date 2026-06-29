# VisControl — OPELKA

Visual inspection control for the dough sheet line (Tucheinzug / Tuchabzug /
Gärtuch). Runs on Windows / Linux x64 dev machines and on the Jetson at the
line. Cameras are Basler `a2A`; the production PLC is a B&R unit talking OPC
UA over Ethernet.

The application launches immediately in Demo mode with a built-in
MockCamera, so an operator or engineer can verify the full UI/detection
loop without any hardware attached.

---

## 1. What works today on a PC (Demo mode)

Everything below runs from a clean checkout against synthetic dough images:

- `python -m viscontrol` launches the UI immediately.
- MockCamera streams synthetic frames at the configured FPS.
- Classical detector classifies blobs as `single` / `row_fused` /
  `column_fused` / `unknown`.
- Overlays render: green/yellow/red circles per classification, yellow
  dashed transfer line on the cloth ROI, "?" on unknown blobs.
- The profile dropdown switches the active profile and applies settings
  immediately.
- "Simulate Pulse" triggers a full cycle: WAITING → TRACKING → INSPECTING →
  READY/FAULT.
- The PLC signals panel reflects all three OPC UA variables in real time.
- In Demo, force-toggle buttons flip `TuchabzugRunning` manually.
- FAULT state shows a red banner and self-clears after `fault_clear_frames`
  consecutive clean frames.
- SERVICE view PIN gate works (default `0000`, hashed in `local.yaml`).
- Mode switching Demo ↔ Production from SERVICE.
- Language switching EN ↔ DE without restart.
- Installation Wizard walks through all 5 steps and saves a profile.
- Image orientation transformations apply equally to MockCamera and
  BaslerCamera.
- Web dashboard at `http://localhost:8080` shows live frame + signals.
- Daily event CSV log is appended on every event (`logs/events-YYYY-MM-DD.csv`).
- All unit tests pass (`py -m pytest -q`).

## 2. What needs real hardware

Tested only in stub form so far; production validation requires:

- **Real Basler camera** — `pypylon` import path, exposure / gain control,
  PixelFormat negotiation. Falls back to MockCamera automatically if no
  camera is detected.
- **Real B&R PLC over OPC UA** — see the contract in §6. A `simulate_plc.py`
  script ships in `scripts/` to stand in for the PLC during integration.

## 3. What's left after real dough images arrive

- Calibrate per recipe via the Installation Wizard (`Learn Reference`).
- Tune `fused_threshold` and `noise_threshold` per profile.
- Tune the `transfer_line_x` and `roi_split_x` per camera mounting.

## 4. Install (PC dev)

```bash
# 1) Clone, create a venv, install
python -m venv .venv
.venv\Scripts\activate                    # Windows
# source .venv/bin/activate               # Linux/Mac
pip install -e ".[dev]"

# 2) Compile translations once
# PySide6 ships its own lrelease binary. On Windows it sits at:
#   .venv\Lib\site-packages\PySide6\lrelease.exe
# Linux:
#   .venv/lib/python3.11/site-packages/PySide6/Qt/libexec/lrelease
pyside6-lrelease translations/viscontrol_de.ts -qm translations/viscontrol_de.qm

# 3) Generate synthetic dough images for the MockCamera (only needed once)
py scripts/generate_test_images.py

# 4) Run
python -m viscontrol
```

On first launch:
- Mode: Demo
- Camera: MockCamera (cycles `assets/test_images/`)
- Profile: `Default`
- Language: English
- SERVICE PIN: `0000`

## 5. Install (Jetson)

JetPack already ships an OpenCV build optimised for ARM. Don't install
`opencv-python-headless` on the Jetson; use the system one.

```bash
# System Python 3.11+ assumed.
sudo apt-get install python3-pyside6.qtcore python3-pyside6.qtgui \
                     python3-pyside6.qtwidgets python3-opencv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[basler]"     # excludes opencv-python-headless (uses system)
```

Notes:
- `pypylon` ships ARM64 wheels on Basler's site — install from there if pip
  fails.
- The Jetson defaults to ARM64; pure-Python deps (`asyncua`, `fastapi`,
  `uvicorn`, `pydantic`, `loguru`, `bcrypt`) install from PyPI as normal.
- Drop a `config/local.yaml` with `app.mode: production` to skip the Demo
  banner.

## 6. OPC UA contract — for the B&R integrator

VisControl runs an OPC UA **server**. The B&R PLC connects as a client.
Three variables live under the namespace `http://opelka.com/viscontrol`,
folder `VisControl/`:

| Variable             | Type | Direction      | Meaning                                                                 |
|----------------------|------|----------------|-------------------------------------------------------------------------|
| `TuchabzugRunning`   | Bool | PLC → VisControl | True while the Tuchabzug puller is active                              |
| `StopTuchabzug`      | Bool | VisControl → PLC | True asserts an immediate stop of the Tuchabzug puller                  |
| `FaultActive`        | Bool | VisControl → PLC | True when an unrecoverable inspection fault is held                     |

Default endpoint: `opc.tcp://0.0.0.0:4840/viscontrol/` (configurable via
`opcua.endpoint`).

Flow on every pulse:
1. PLC raises `TuchabzugRunning` → VisControl moves to TRACKING.
2. When a dough row crosses the configured transfer line, VisControl raises
   `StopTuchabzug`. The PLC stops the puller in response.
3. PLC lowers `TuchabzugRunning`. After a settling delay
   (`inspection.delay_after_pull_ms`, default 200 ms) VisControl moves to
   INSPECTING and runs the belt-side classifier.
4. If a fault is detected, VisControl raises `FaultActive`. The PLC handles
   alarming / line stop. `FaultActive` clears automatically when the belt
   image is clean for `inspection.fault_clear_frames` consecutive frames.

Both write nodes (`StopTuchabzug`, `FaultActive`) are pushed on every
state-machine transition, so a PLC reconnect always sees an up-to-date
value.

Run `py scripts/simulate_plc.py` to drive the contract from a workstation
during integration.

## 7. Web dashboard

Read-only, runs in-process on a background thread. Default URL is
`http://<host>:8080`. Endpoints:

| Path               | Returns                                          |
|--------------------|--------------------------------------------------|
| `/`                | HTML dashboard, auto-refresh every 2 s          |
| `/image/latest.jpg`| Most recent annotated belt frame (JPEG)         |
| `/api/status.json` | State, mode, profile, counts, signal snapshot   |
| `/logs/today.csv`  | Today's event CSV                                |

No POST endpoints. To require auth: in SERVICE → Web access, click
`Set web password…` and set a password. The password is hashed before
storage; basic-auth credentials are required on every endpoint once set.

## 8. PIN reset

If the SERVICE PIN is lost: edit `config/local.yaml`, clear the
`ui.service_pin_hash` field, and restart. On next launch the app re-seeds
the hash from the literal `0000` (the same code path runs on first launch
of a fresh install).

## 9. Troubleshooting

| Symptom                                     | Likely cause / fix                                          |
|---------------------------------------------|-------------------------------------------------------------|
| Status bar says "Camera disconnected"        | No Basler detected; check pylon SDK + USB/Ethernet link    |
| Status bar shows MockCamera in Production   | `pypylon` missing or no device found — auto-fallback kicks in. Reinstall `pypylon`. |
| OPC UA dot stays red in Production           | Port 4840 blocked by firewall, or another OPC UA server bound to it |
| Web dashboard 401                            | A web password is configured; supply it via basic auth     |
| German strings show in English               | `.qm` file missing; run `pyside6-lrelease` (see §4 step 2) |
| State machine stuck in TRACKING              | Cloth ROI never produces a centroid past `transfer_line_x`; check `roi_split_x` and `transfer_line_x` for the active profile |
| Detector flags every piece as `unknown`      | `noise_threshold` or `fused_threshold` too aggressive; run Wizard → Learn Reference |
| "unexpected_reboot" event on every startup   | App is being SIGKILL'd; ensure the systemd unit / shortcut calls Quit cleanly |

## 10. Layout

```
viscontrol/
  core/            config, profiles, security, state machine, events, logging
  detection/       classical OpenCV detector + calibration + pipeline
  io/              camera, OPC UA server, storage helpers, web sidecar
  ui/              PySide6 — theme, i18n, widgets, views, main_window
  main.py          startup recovery + service wiring (step 15)
  __main__.py      `python -m viscontrol` entry

config/
  default.yaml     version-controlled defaults
  local.yaml       user-edited overrides (gitignored)

assets/test_images/   synthetic dough images for MockCamera
scripts/
  generate_test_images.py   regenerate the synthetic dough frames
  simulate_plc.py           drive TuchabzugRunning from a workstation

tests/             pytest suites for core/, detection/, io/camera
translations/
  viscontrol_de.ts          source (human-edited)
  viscontrol_de.qm          compiled (built locally, not committed)

logs/              runtime: app.log (rotating), events-*.csv, defects/
```

## 11. Running the tests

```bash
py -m pytest -q
```

All suites — `core.config`, `core.security`, `core.profiles`,
`core.event_log`, `core.state_machine`, `detection`, `io.camera` —
should pass.

---

© OPELKA. Internal use.
