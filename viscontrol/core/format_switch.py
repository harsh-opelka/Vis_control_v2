"""PLC-driven product / cloth switching (pure logic, no Qt, no I/O).

The PLC publishes two values (``gFormatSet.Name`` and
``gFormatSet.Beschickung.ClothColor``). :class:`FormatSwitcher` turns them into
a (product, cloth) pair, waits until the pair has been stable for
``stable_s`` AND the line is idle, then hands it to ``apply_fn``.

Invalid input (unknown product, ClothColor not 0/1, empty value, PLC read
error ``None``) never raises and never changes the applied settings; each
distinct bad value logs one WARNING.

Threading: :meth:`submit` and :meth:`tick` must be called from ONE thread (the
GUI thread). PlcClient callbacks come from its worker thread and reach
:meth:`submit` through a Qt signal.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from viscontrol.core.config import AppConfig
from viscontrol.core.logger import logger

CLOTH_BY_COLOR = {0: "white", 1: "black"}


def match_product(raw: Any, known: list[str]) -> Optional[str]:
    """Trimmed, case-insensitive match of the PLC product string; None if invalid."""
    if not isinstance(raw, str):
        return None
    key = raw.strip().casefold()
    if not key:
        return None
    for name in known:
        if name.strip().casefold() == key:
            return name
    return None


def match_cloth(raw: Any, known: list[str]) -> Optional[str]:
    """Map ClothColor (0 = white, 1 = black) to a cloth key; None if invalid."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    cloth = CLOTH_BY_COLOR.get(raw)
    return cloth if cloth in known else None


class FormatSwitcher:
    def __init__(
        self,
        cfg_fn: Callable[[], AppConfig],
        apply_fn: Callable[[str, str], None],
        is_idle_fn: Callable[[], bool],
        now_fn: Callable[[], float],
        stable_s: float = 1.0,
    ) -> None:
        # cfg_fn (not a cfg) because the wizard's Cancel replaces MainWindow._cfg.
        self._cfg_fn = cfg_fn
        self._apply_fn = apply_fn
        self._is_idle_fn = is_idle_fn
        self._now = now_fn
        self._stable_s = stable_s
        self.active_product: Optional[str] = None
        self.active_cloth: Optional[str] = None
        self._target: tuple[Optional[str], Optional[str]] = (None, None)
        self._target_since: float = 0.0
        self._warned: dict[str, Any] = {}
        # Read-only state for the sidebar info block.
        self.plc_seen: bool = False
        self.product_valid: bool = True
        self.cloth_valid: bool = True
        self.on_change: Optional[Callable[[], None]] = None  # GUI thread only

    @property
    def pending(self) -> bool:
        """True while a valid target differing from the applied settings waits."""
        product, cloth = self._target
        return (
            product is not None and cloth is not None
            and (product, cloth) != (self.active_product, self.active_cloth)
        )

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()

    @property
    def enabled(self) -> bool:
        cfg = self._cfg_fn()
        return bool(cfg.products) and bool(cfg.cloths)

    def apply_defaults(self) -> None:
        """Apply ``app.active_product`` / ``app.active_cloth`` (startup, no PLC yet)."""
        if not self.enabled:
            return
        cfg = self._cfg_fn()
        product = match_product(cfg.app.active_product, list(cfg.products))
        cloth = cfg.app.active_cloth if cfg.app.active_cloth in cfg.cloths else None
        if product is None or cloth is None:
            self._warn_once(
                "defaults",
                (cfg.app.active_product, cfg.app.active_cloth),
                "format switch: invalid default active_product={!r} / active_cloth={!r}; "
                "no product/cloth settings applied",
                cfg.app.active_product, cfg.app.active_cloth,
            )
            return
        self._apply(product, cloth)

    def submit(self, product_raw: Any, cloth_raw: Any) -> None:
        """New raw PLC values (``None`` = read error). Cheap; applies via tick()."""
        if not self.enabled:
            return
        cfg = self._cfg_fn()
        product = match_product(product_raw, list(cfg.products))
        cloth = match_cloth(cloth_raw, list(cfg.cloths))
        self.plc_seen = True
        self.product_valid = product is not None
        self.cloth_valid = cloth is not None
        if product is None:
            self._warn_once(
                "product", product_raw,
                "format switch: invalid PLC product {!r} (known: {}); keeping {!r}",
                product_raw, list(cfg.products), self.active_product,
            )
        else:
            self._warned.pop("product", None)
        if cloth is None:
            self._warn_once(
                "cloth", cloth_raw,
                "format switch: invalid PLC ClothColor {!r} (expected 0 or 1); keeping {!r}",
                cloth_raw, self.active_cloth,
            )
        else:
            self._warned.pop("cloth", None)
        # An invalid component keeps whatever is currently applied.
        target = (product or self.active_product, cloth or self.active_cloth)
        if target != self._target:
            self._target = target
            self._target_since = self._now()
        self.tick()
        self._notify()

    def tick(self) -> None:
        """Apply the pending target once it is stable and the line is idle."""
        product, cloth = self._target
        if product is None or cloth is None:
            return
        if (product, cloth) == (self.active_product, self.active_cloth):
            return
        if self._now() - self._target_since < self._stable_s:
            return
        if not self._is_idle_fn():
            return  # deferred; retried on the next tick
        self._apply(product, cloth)

    def _apply(self, product: str, cloth: str) -> None:
        try:
            self._apply_fn(product, cloth)
        except Exception:  # noqa: BLE001 - a bad apply must never stop the line
            logger.exception("format switch: applying product={} cloth={} failed", product, cloth)
            self._target = (self.active_product, self.active_cloth)
            return
        self.active_product, self.active_cloth = product, cloth
        self._target = (product, cloth)
        logger.info("PRODUCT/CLOTH SWITCH: product={} cloth={} (from PLC)", product, cloth)
        self._notify()

    def _warn_once(self, kind: str, raw: Any, msg: str, *args: Any) -> None:
        if kind in self._warned and self._warned[kind] == raw:
            return
        self._warned[kind] = raw
        logger.warning(msg, *args)
