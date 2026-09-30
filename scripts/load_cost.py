"""Model load time, peak memory, and warm latency for one three-question request, per model.

    uv run python scripts/load_cost.py mlx-community/Qwen3-4B-Instruct-2507-4bit mlx-community/Qwen3.5-9B-MLX-4bit

`MODEL@ADAPTER` loads a LoRA adapter directory on top, unfused, and `MODEL@A,B` serves two
adapters of one base model as an averaged ensemble:

    uv run python scripts/load_cost.py mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit \
        mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit@checkpoints/lora-30b
"""

import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from lars.backends.mlx import MLXBackend
from lars.engine import Engine, EnsembleEngine
from lars.schema import SystemOneRequest

REQUEST = SystemOneRequest(
    state="Help! My payouts have been failing for 3 days. I am losing sales.",
    questions={
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, payouts", "technical": "Bugs, outages",
                                    "sales": "Pricing"}},
        "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
        "tone": {"type": "score", "instructions": "How angry is the sender?",
                 "criteria": ["calm", "annoyed", "furious"]},
    },
)


def main() -> int:
    out = {}
    for model in sys.argv[1:]:
        path, _, adapter = model.partition("@")
        adapters = [a for a in adapter.split(",") if a]
        mx.reset_peak_memory()
        started = time.perf_counter()
        backend = MLXBackend(path, adapter=adapters if len(adapters) > 1 else (adapters[0] if adapters else None))
        engine = EnsembleEngine(backend, [None] * len(adapters)) if len(adapters) > 1 else Engine(backend)
        load_s = time.perf_counter() - started
        engine.evaluate(REQUEST)  # warm-up
        started, n = time.perf_counter(), 20
        for _ in range(n):
            engine.evaluate(REQUEST)
        out[model] = {
            "load_s": round(load_s, 1),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
            "three_question_request_ms": round((time.perf_counter() - started) / n * 1000, 1),
        }
        print(json.dumps({model: out[model]}), flush=True)
        del engine
        gc.collect()
        mx.clear_cache()  # so the next model's peak starts from an empty pool
    Path("logs").mkdir(exist_ok=True)
    Path("logs/load-cost.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
