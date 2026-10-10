"""Screen capture backends.

* ``MssCapture``  - portable GDI/X11 capture (python-mss). Always available.
* ``DxcamCapture`` - DXGI Desktop Duplication (Windows, optional ``dxcam``).
  Much cheaper per frame and, importantly, it reports when the screen has
  *not* changed since the last grab (``grab`` returns ``None``) - the
  automation loop then knows that the previous detection is still current
  and never mistakes an old frame for a new one.

Capture objects are created inside the automation worker thread and used
only there (GDI/DXGI handles are thread-affine).
"""

from __future__ import annotations

import logging
import sys
from typing import Protocol

import numpy as np

from .geometry import Rect, monitor_index_containing

log = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    pass


class Capture(Protocol):
    name: str

    def grab(self, rect: Rect) -> np.ndarray | None:
        """BGR/BGRA frame of ``rect`` or ``None`` when the screen did not change."""

    def monitors(self) -> list[Rect]:
        ...

    def close(self) -> None:
        ...


def _mss_factory():
    import mss  # local import keeps startup fast and tests independent

    return getattr(mss, "MSS", None) or mss.mss


def list_monitors() -> list[Rect]:
    """Physical monitor rectangles (index 0 = first monitor)."""
    try:
        with _mss_factory()() as sct:
            return [Rect.from_mss(m) for m in sct.monitors[1:]]
    except Exception as exc:  # pragma: no cover - depends on display
        log.debug("mss monitors failed: %s", exc)
        from .winapi import monitors_win32

        return monitors_win32()


class MssCapture:
    name = "mss"

    def __init__(self) -> None:
        try:
            self._sct = _mss_factory()()
        except Exception as exc:
            raise CaptureError(f"mss: не удалось инициализировать захват экрана: {exc}") from exc

    def monitors(self) -> list[Rect]:
        return [Rect.from_mss(m) for m in self._sct.monitors[1:]]

    def grab(self, rect: Rect) -> np.ndarray:
        try:
            shot = self._sct.grab(rect.to_mss())
        except Exception as exc:
            raise CaptureError(f"mss: ошибка захвата {rect}: {exc}") from exc
        w, h = shot.size
        return np.frombuffer(shot.raw, np.uint8).reshape(h, w, 4)

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:  # pragma: no cover
            pass


class DxcamCapture:
    """DXGI Desktop Duplication through the ``dxcam`` package (Windows only)."""

    name = "dxcam"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise CaptureError("dxcam доступен только в Windows")
        try:
            import dxcam  # noqa: F401
        except Exception as exc:
            raise CaptureError(f"dxcam не установлен: {exc}") from exc
        self._dxcam = dxcam
        self._factory = vars(dxcam).get("__factory")
        if self._factory is None:
            raise CaptureError("dxcam: неподдерживаемая версия")
        self._cams: dict[tuple[int, int], object] = {}
        self._outputs: list[tuple[int, int, Rect]] = []
        self._last_rect: Rect | None = None
        self._fallback = MssCapture()   # for regions spanning several monitors
        self._refresh_outputs()
        if not self._outputs:
            raise CaptureError("dxcam: не найдено ни одного монитора")

    def _refresh_outputs(self) -> None:
        outs = []
        for d, outputs in enumerate(self._factory.outputs):
            for o, out in enumerate(outputs):
                try:
                    out.update_desc()
                    rc = out.desc.DesktopCoordinates
                    outs.append((d, o, Rect.from_ltrb(rc.left, rc.top, rc.right, rc.bottom)))
                except Exception as exc:  # pragma: no cover - hardware specific
                    log.debug("dxcam output %s/%s: %s", d, o, exc)
        self._outputs = outs

    def monitors(self) -> list[Rect]:
        return self._fallback.monitors()

    def _camera(self, d: int, o: int):
        cam = self._cams.get((d, o))
        if cam is None:
            cam = self._dxcam.create(device_idx=d, output_idx=o, output_color="BGRA")
            self._cams[(d, o)] = cam
        return cam

    def grab(self, rect: Rect) -> np.ndarray | None:
        idx = monitor_index_containing(rect, [r for _, _, r in self._outputs])
        if idx is None:
            self._refresh_outputs()
            idx = monitor_index_containing(rect, [r for _, _, r in self._outputs])
        if idx is None:
            # region spans several monitors -> GDI for this frame
            self._last_rect = None
            return self._fallback.grab(rect)
        d, o, mon = self._outputs[idx]
        region = (rect.left - mon.left, rect.top - mon.top, rect.right - mon.left, rect.bottom - mon.top)
        try:
            frame = self._camera(d, o).grab(region=region, new_frame_only=True)
        except Exception as exc:
            raise CaptureError(f"dxcam: ошибка захвата: {exc}") from exc
        if frame is None:
            if rect != self._last_rect:
                # no new desktop frame yet, but we have nothing for this region
                self._last_rect = rect
                return self._fallback.grab(rect)
            return None
        self._last_rect = rect
        return frame

    def close(self) -> None:
        for cam in self._cams.values():
            try:
                cam.release()
            except Exception:  # pragma: no cover
                pass
        self._cams.clear()
        self._fallback.close()


def create_capture(backend: str = "auto") -> Capture:
    """Create the best available capture backend."""
    if backend in ("auto", "dxcam") and sys.platform == "win32":
        try:
            return DxcamCapture()
        except Exception as exc:
            if backend == "dxcam":
                log.warning("dxcam недоступен (%s), используется mss", exc)
            else:
                log.info("dxcam недоступен (%s), используется mss", exc)
    return MssCapture()
