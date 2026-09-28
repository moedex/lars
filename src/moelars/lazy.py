"""An engine that is built on first use and dropped after a quiet spell: `serve --idle-unload`.

The 30B default holds about 18 GB while loaded, and an agent calls it in bursts. With an
idle limit, the model is freed after that many seconds without a request, and the next
request builds it again (a few seconds with a warm page cache). Every call here runs on the
inference worker, under the server's one-at-a-time limiter, so a load, a request and an
unload never overlap.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any


def _release_device_memory() -> None:
    """Hand MLX's cached buffers back to the system; a no-op where MLX is not installed."""
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.clear_cache()


class LazyEngine:
    def __init__(self, factory: Callable[[], Any], idle_unload: float = 0.0,
                 clock: Callable[[], float] = time.monotonic, engine: Any | None = None) -> None:
        """`factory` builds the engine; `idle_unload` seconds without use drops it (0 keeps it).

        Pass `engine` to start loaded, as `serve` does without `--idle-unload`."""
        self._factory = factory
        self.idle_unload = idle_unload
        self._clock = clock
        self._engine = engine
        self._last_used = clock()
        self.model_id: str | None = engine.model_id if engine is not None else None
        self.version: str | None = getattr(engine, "version", None)
        self.loads = 0

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def get(self) -> Any:
        """The engine, built if it is not loaded; marks it used."""
        if self._engine is None:
            self._engine = self._factory()
            self.loads += 1
            self.model_id = self._engine.model_id
            self.version = getattr(self._engine, "version", None)
        self._last_used = self._clock()
        return self._engine

    def idle_for(self) -> float:
        return self._clock() - self._last_used

    def maybe_unload(self) -> bool:
        """Drop the engine if it has been idle past the limit; True when it did."""
        if self._engine is None or not self.idle_unload or self.idle_for() < self.idle_unload:
            return False
        self._engine = None
        gc.collect()
        _release_device_memory()
        return True


def parse_duration(text: str) -> float:
    """Seconds from `900`, `90s`, `15m` or `1h`; 0 turns idle unloading off."""
    text = text.strip().lower()
    for suffix, factor in (("h", 3600.0), ("m", 60.0), ("s", 1.0)):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * factor
    return float(text)
