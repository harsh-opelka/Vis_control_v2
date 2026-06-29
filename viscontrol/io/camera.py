"""Camera abstraction layer.

The rest of the application sees a single :class:`Camera` interface. The actual
backend is one of:

- :class:`BaslerCamera`: real hardware via ``pypylon``. Imported lazily so the
  app still runs on dev machines without the SDK installed.
- :class:`MockCamera`: cycles through images in ``assets/test_images/`` at a
  configurable FPS. Also applies the orientation transform — useful for
  exercising the transform code without hardware.
- :class:`PlaybackCamera`: CALIBRATION TOOLING — replays a folder of frames
  recorded by ``io/recorder.py`` for offline detection tuning. See its
  docstring for why it does NOT re-apply the orientation transform.

Basler and Mock both apply :class:`OrientationTransform` to every grabbed frame
*before* emitting it, so downstream code never needs to know about rotation/flip.

:func:`make_camera` does the source resolution: ``"auto"`` tries Basler first,
falls back to mock; ``"basler"`` raises if pypylon is missing; ``"mock"`` always
returns a mock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol

import cv2
import numpy as np

from viscontrol.core.logger import logger
from viscontrol.core.profiles import ProductProfile


FrameCallback = Callable[[np.ndarray], None]


@dataclass
class OrientationTransform:
    """Rotate-then-flip transform applied to every grabbed frame.

    Why rotate-then-flip (not flip-then-rotate): the mockup describes the
    transform as "rotate the camera mount, then mirror if needed", which maps
    cleanly to ``rotate -> flip``. Keep the order consistent across MockCamera
    and BaslerCamera so the SERVICE orientation panel does the same thing for
    both.
    """

    rotation: Literal[0, 90, 180, 270] = 0
    flip_horizontal: bool = False

    def apply(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        if self.rotation == 90:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            out = cv2.rotate(out, cv2.ROTATE_180)
        elif self.rotation == 270:
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.flip_horizontal:
            out = cv2.flip(out, 1)
        return out


class Camera(Protocol):
    """Camera interface implemented by all backends."""

    is_connected: bool

    def start(self, on_frame: FrameCallback) -> None: ...
    def stop(self) -> None: ...
    def apply_profile(self, profile: ProductProfile) -> None: ...
    def grab(self) -> Optional[np.ndarray]: ...  # one-shot single frame, optional


# -------------- MockCamera --------------


class MockCamera:
    """File-backed camera for development.

    Cycles ``image_dir`` deterministically at ``fps`` frames per second. If
    ``image_dir`` is empty we synthesize a blank frame so the UI still gets
    *something* and the user sees an obvious "no test images" status banner.
    """

    is_connected = True
    backend_name = "mock"

    def __init__(
        self,
        image_dir: Path,
        *,
        fps: float = 5.0,
        transform: OrientationTransform | None = None,
        synthetic_size: tuple[int, int] = (1024, 1024),
    ) -> None:
        self._dir = Path(image_dir)
        self._fps = max(0.5, float(fps))
        self._transform = transform or OrientationTransform()
        self._synthetic_size = synthetic_size
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index = 0
        self._lock = threading.Lock()

    def set_transform(self, transform: OrientationTransform) -> None:
        with self._lock:
            self._transform = transform

    def _list_images(self) -> list[Path]:
        if not self._dir.exists():
            return []
        return sorted(
            p for p in self._dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        )

    def _next_frame(self) -> np.ndarray:
        images = self._list_images()
        if not images:
            # Synthetic blank: bright square in the middle so the user can see
            # orientation transforms working in the UI.
            h, w = self._synthetic_size
            frame = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), 200, -1)
            cv2.putText(
                frame, "NO TEST IMAGES",
                (w // 4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, 80, 3, cv2.LINE_AA,
            )
            return frame
        path = images[self._index % len(images)]
        self._index = (self._index + 1) % len(images)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("MockCamera failed to read {}", path)
            return self._next_frame()  # try the next one
        return img

    def grab(self) -> Optional[np.ndarray]:
        with self._lock:
            transform = self._transform
        return transform.apply(self._next_frame())

    def start(self, on_frame: FrameCallback) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("MockCamera already running")
        self._stop.clear()

        def _loop() -> None:
            period = 1.0 / self._fps
            logger.info("MockCamera loop start; fps={}", self._fps)
            while not self._stop.is_set():
                frame = self.grab()
                if frame is not None:
                    try:
                        on_frame(frame)
                    except Exception as e:  # noqa: BLE001
                        logger.exception("MockCamera frame callback failed: {}", e)
                self._stop.wait(period)
            logger.info("MockCamera loop stopped")

        self._thread = threading.Thread(target=_loop, name="MockCameraLoop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def apply_profile(self, profile: ProductProfile) -> None:
        # MockCamera ignores exposure/gain — but we still log so SERVICE can confirm
        # profile changes propagated.
        logger.debug(
            "MockCamera apply_profile(name={}, exposure_us={}, gain={})",
            profile.name, profile.camera_exposure_us, profile.camera_gain,
        )


# -------------- BaslerCamera --------------


class BaslerCamera:
    """Real Basler ``a2A`` camera via pypylon.

    Import is deferred until :meth:`__init__` so the app can still launch
    without the SDK on dev machines. On import failure we raise; the
    :func:`make_camera` factory catches that and falls back to MockCamera.
    """

    backend_name = "basler"

    def __init__(
        self,
        *,
        serial: str = "",
        pixel_format: str = "Mono8",
        transform: OrientationTransform | None = None,
    ) -> None:
        try:
            from pypylon import pylon  # type: ignore
        except ImportError as e:
            raise RuntimeError(f"pypylon not available: {e}") from e
        self._pylon = pylon
        self._serial = serial
        self._pixel_format = pixel_format
        self._transform = transform or OrientationTransform()
        self._camera = self._open(serial, pixel_format)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return bool(self._camera and self._camera.IsOpen())

    def _open(self, serial: str, pixel_format: str):  # noqa: ANN001
        pylon = self._pylon
        tlf = pylon.TlFactory.GetInstance()
        if serial:
            info = pylon.DeviceInfo()
            info.SetSerialNumber(serial)
            device = tlf.CreateDevice(info)
        else:
            devices = tlf.EnumerateDevices()
            if not devices:
                raise RuntimeError("no Basler cameras detected")
            # Prefer a2A models when available.
            preferred = [d for d in devices if d.GetModelName().startswith("a2A")]
            chosen = preferred[0] if preferred else devices[0]
            device = tlf.CreateDevice(chosen)
        camera = pylon.InstantCamera(device)
        camera.Open()
        try:
            camera.PixelFormat.SetValue(pixel_format)
        except Exception:  # noqa: BLE001
            logger.warning("Could not set PixelFormat={}, leaving default", pixel_format)
        return camera

    def set_transform(self, transform: OrientationTransform) -> None:
        with self._lock:
            self._transform = transform

    def apply_profile(self, profile: ProductProfile) -> None:
        if not self.is_connected:
            return
        try:
            self._camera.ExposureTime.SetValue(profile.camera_exposure_us)
        except Exception:  # noqa: BLE001
            try:
                self._camera.ExposureTimeAbs.SetValue(profile.camera_exposure_us)
            except Exception:  # noqa: BLE001
                logger.warning("Could not set ExposureTime to {}", profile.camera_exposure_us)
        try:
            self._camera.Gain.SetValue(profile.camera_gain)
        except Exception:  # noqa: BLE001
            try:
                self._camera.GainRaw.SetValue(int(profile.camera_gain))
            except Exception:  # noqa: BLE001
                logger.warning("Could not set Gain to {}", profile.camera_gain)

    def grab(self) -> Optional[np.ndarray]:
        if not self.is_connected:
            return None
        pylon = self._pylon
        try:
            result = self._camera.GrabOne(1000)
            if not result.GrabSucceeded():
                return None
            arr = result.GetArray()
            with self._lock:
                transform = self._transform
            return transform.apply(arr)
        except Exception as e:  # noqa: BLE001
            logger.exception("BaslerCamera grab failed: {}", e)
            return None

    def start(self, on_frame: FrameCallback) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("BaslerCamera already running")
        self._stop.clear()
        pylon = self._pylon

        def _loop() -> None:
            logger.info("BaslerCamera grab loop start")
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            try:
                while not self._stop.is_set() and self._camera.IsGrabbing():
                    try:
                        result = self._camera.RetrieveResult(
                            500, pylon.TimeoutHandling_ThrowException
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    if result.GrabSucceeded():
                        arr = result.GetArray()
                        with self._lock:
                            transform = self._transform
                        frame = transform.apply(arr)
                        try:
                            on_frame(frame)
                        except Exception as e:  # noqa: BLE001
                            logger.exception("Frame callback failed: {}", e)
                    result.Release()
            finally:
                try:
                    self._camera.StopGrabbing()
                except Exception:  # noqa: BLE001
                    pass
                logger.info("BaslerCamera grab loop stopped")

        self._thread = threading.Thread(target=_loop, name="BaslerCameraLoop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            if self._camera and self._camera.IsOpen():
                self._camera.Close()
        except Exception:  # noqa: BLE001
            pass


# -------------- PlaybackCamera (CALIBRATION TOOLING) --------------


class PlaybackCamera:
    """Replays a folder of recorded PNG frames as a camera source.

    Used for offline calibration: feeds previously-recorded raw frames (see
    ``io/recorder.py``) through the exact same ``on_frame`` callback the live
    cameras use, so the rest of the app (pipeline, tripwire, state machine)
    runs identically regardless of source.

    Frames are NOT re-run through :class:`OrientationTransform` — they were
    already transformed at record time (recording taps the same point this
    module emits from), so re-applying it here would double-transform.
    ``set_transform`` is kept as a no-op only so existing duck-typed callers
    (``main_window`` calls it unconditionally via ``hasattr``) don't need a
    special case.
    """

    backend_name = "playback"

    def __init__(self, folder: Path, *, fps: float = 5.0, loop: bool = True) -> None:
        self._dir = Path(folder)
        self._fps = max(0.1, float(fps))
        self._loop = loop
        self._frames: list[Path] = self._list_frames()
        self._index = 0
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_frame: FrameCallback | None = None

    @property
    def is_connected(self) -> bool:
        return len(self._frames) > 0

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def current_index(self) -> int:
        return self._index

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def set_loop(self, loop: bool) -> None:
        self._loop = loop

    def set_transform(self, transform: OrientationTransform) -> None:
        pass  # no-op — see class docstring

    def _list_frames(self) -> list[Path]:
        if not self._dir.exists():
            return []
        return sorted(self._dir.glob("frame_*.png"))

    def _read(self, idx: int) -> Optional[np.ndarray]:
        if not self._frames:
            return None
        path = self._frames[idx % len(self._frames)]
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning("PlaybackCamera failed to read {}", path)
        return img

    def grab(self) -> Optional[np.ndarray]:
        return self._read(self._index)

    def apply_profile(self, profile: ProductProfile) -> None:
        logger.debug("PlaybackCamera apply_profile ignored (name={})", profile.name)

    def start(self, on_frame: FrameCallback) -> None:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("PlaybackCamera already running")
        if not self._frames:
            logger.warning("PlaybackCamera: no frame_*.png files found in {}", self._dir)
        self._on_frame = on_frame
        self._stop.clear()
        self._paused.clear()

        def _loop() -> None:
            period = 1.0 / self._fps
            logger.info(
                "PlaybackCamera loop start; dir={} frames={} fps={} loop={}",
                self._dir, len(self._frames), self._fps, self._loop,
            )
            while not self._stop.is_set():
                if self._paused.is_set() or not self._frames:
                    self._stop.wait(0.1)
                    continue
                frame = self._read(self._index)
                if frame is not None:
                    try:
                        on_frame(frame)
                    except Exception as e:  # noqa: BLE001
                        logger.exception("PlaybackCamera frame callback failed: {}", e)
                self._index += 1
                if self._index >= len(self._frames):
                    if self._loop:
                        self._index = 0
                    else:
                        self._index = len(self._frames) - 1
                        self._paused.set()  # hold on the last frame
                self._stop.wait(period)
            logger.info("PlaybackCamera loop stopped")

        self._thread = threading.Thread(target=_loop, name="PlaybackCameraLoop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- playback transport (not part of the Camera protocol) ----

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def step(self) -> None:
        """Emit exactly one frame and advance, regardless of pause state.

        Intended to be called while paused, from the GUI thread, so the
        operator can hold on a single frame while tuning thresholds.
        """
        if not self._frames or self._on_frame is None:
            return
        frame = self._read(self._index)
        if frame is not None:
            try:
                self._on_frame(frame)
            except Exception as e:  # noqa: BLE001
                logger.exception("PlaybackCamera step callback failed: {}", e)
        self._index = (self._index + 1) % len(self._frames)


# -------------- Factory --------------


def make_camera(
    *,
    source: Literal["auto", "basler", "mock", "playback"],
    mock_image_dir: Path,
    mock_fps: float,
    pixel_format: str = "Mono8",
    basler_serial: str = "",
    transform: OrientationTransform | None = None,
    on_warning: Callable[[str], None] | None = None,
    playback_dir: Path | None = None,
    playback_fps: float = 5.0,
    playback_loop: bool = True,
) -> Camera:
    """Resolve the configured ``source`` into a concrete camera instance.

    ``on_warning`` is invoked with a user-visible string when we fall back
    from Basler to mock so the status bar can display the reason.
    ``playback_dir``/``playback_fps``/``playback_loop`` are only used when
    ``source == "playback"`` (CALIBRATION TOOLING — see PlaybackCamera).
    """
    transform = transform or OrientationTransform()

    if source == "playback":
        return PlaybackCamera(
            playback_dir or Path("."), fps=playback_fps, loop=playback_loop,
        )

    if source == "mock":
        return MockCamera(mock_image_dir, fps=mock_fps, transform=transform)

    if source == "basler":
        try:
            return BaslerCamera(
                serial=basler_serial, pixel_format=pixel_format, transform=transform,
            )
        except RuntimeError as e:
            logger.error("Basler requested but unavailable: {}", e)
            raise

    # auto
    try:
        return BaslerCamera(
            serial=basler_serial, pixel_format=pixel_format, transform=transform,
        )
    except RuntimeError as e:
        msg = f"Basler unavailable ({e}); using MockCamera"
        logger.warning(msg)
        if on_warning:
            on_warning(msg)
        return MockCamera(mock_image_dir, fps=mock_fps, transform=transform)
