"""Several LoRA adapters on one base model, swapped in place between passes.

The model gets LoRA layers for the union of the adapters' keys (an attention-plus-experts
adapter covers an attention-only one). Each adapter's weights stay in memory, with zeros
for keys it does not adapt, so a LoRA layer it lacks adds nothing. `use` reassigns the
LoRA arrays without copying, so switching adapters costs no model pass. Adapters must
share rank, scale and adapted blocks, which is what `lars.train.lora` writes by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AdapterSet:
    def __init__(self, model: Any, adapter_dirs: list[str]) -> None:
        import mlx.core as mx
        from mlx_lm.tuner.utils import linear_to_lora_layers

        if len(adapter_dirs) < 2:
            raise ValueError("an adapter set needs at least two adapters; load one with adapter_path")
        self._mx = mx
        self.model = model
        self.names = [Path(d).name for d in adapter_dirs]
        configs = [json.loads((Path(d) / "adapter_config.json").read_text()) for d in adapter_dirs]
        shared = {(c["num_layers"], c["lora_parameters"]["rank"], c["lora_parameters"]["scale"]) for c in configs}
        if len(shared) != 1:
            raise ValueError(f"adapters differ in blocks, rank or scale: {sorted(shared)}")
        keys = sorted({k for c in configs for k in c["lora_parameters"]["keys"]})
        num_layers, rank, scale = shared.pop()
        linear_to_lora_layers(model, num_layers, {"rank": rank, "scale": scale, "dropout": 0.0, "keys": keys})

        weights = [dict(mx.load(str(Path(d) / "adapters.safetensors"))) for d in adapter_dirs]
        union = sorted({k for w in weights for k in w})
        shapes = {k: next(w[k] for w in weights if k in w) for k in union}
        self._weights = [
            [(k, w[k] if k in w else mx.zeros(shapes[k].shape, dtype=shapes[k].dtype)) for k in union]
            for w in weights
        ]
        self.active: int | None = None
        self.use(0)

    def __len__(self) -> int:
        return len(self._weights)

    def use(self, index: int) -> None:
        if index != self.active:
            self.model.load_weights(self._weights[index], strict=False)
            self.active = index
