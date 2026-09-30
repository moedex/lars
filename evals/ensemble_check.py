"""Run two adapters as a live ensemble over the suite's test rows and compare with the simulated blend.

    uv run python evals/ensemble_check.py --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit \
        --adapter checkpoints/a --slug <suite slug of a> --adapter checkpoints/b --slug <suite slug of b>

Each adapter uses the per-config calibrators its suite run saved (`calibration/<config>.<slug>.json`),
and every row goes through `EnsembleEngine.evaluate`, the serving path. A row is right when
the averaged answer picks the target's top option, which is how `evals/cascade.py` scores its
`blend` policy from row dumps; the two macro accuracies should agree to within rounding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lars.backends.mlx import MLXBackend
from lars.calibration import Calibrator
from lars.engine import EnsembleEngine
from lars.evalset import read_examples
from lars.schema import ChoiceAnswer, NoulAnswer, SystemOneRequest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_suite import ALL_CONFIGS  # noqa: E402


def _hit(example, answer) -> bool:
    if isinstance(answer, NoulAnswer):
        return (answer.noul >= 0.5) == (example.label in {"1", "true", "yes"})
    if example.soft_label:
        top = max(example.soft_label.values())
        return float(example.soft_label.get(max(answer.probabilities, key=answer.probabilities.get), 0.0)) == top
    predicted = answer.choice if isinstance(answer, ChoiceAnswer) else max(answer.probabilities,
                                                                           key=answer.probabilities.get)
    return predicted == example.label


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", action="append", required=True)
    parser.add_argument("--slug", action="append", required=True, help="suite slug per adapter, in the same order")
    parser.add_argument("--configs", default=",".join(ALL_CONFIGS))
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--data-dir", default=str(HERE / "data"))
    parser.add_argument("--calibration-dir", default=str(HERE.parent / "calibration"))
    parser.add_argument("--out", default=None, help="write per-config accuracy JSON here")
    args = parser.parse_args()
    if len(args.adapter) != len(args.slug):
        raise SystemExit("give one --slug per --adapter")

    backend = MLXBackend(args.model, adapter=args.adapter)
    per_config = {}
    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        calibrators = []
        for slug in args.slug:
            path = Path(args.calibration_dir) / f"{cfg}.{slug}.json"
            calibrators.append(Calibrator.load(path) if path.exists() else None)
        engine = EnsembleEngine(backend, calibrators, max_rows=None, max_input_tokens=None)
        examples = list(read_examples(Path(args.data_dir) / f"{cfg}.test.jsonl"))[: args.rows]
        hits = [_hit(e, engine.evaluate(SystemOneRequest(state=e.state, questions={"q": e.question}))
                     .answers["q"]) for e in examples]
        per_config[cfg] = float(np.mean(hits))
        print(f"[{cfg}] live ensemble acc {per_config[cfg]:.3f}", flush=True)
    macro = float(np.mean(list(per_config.values())))
    print(f"macro accuracy {macro:.4f} over {len(per_config)} configs")
    if args.out:
        Path(args.out).write_text(json.dumps({"macro": macro, "per_config": per_config}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
