"""Fuse a LoRA adapter into a quantized base without re-quantizing it at 4 bits.

    uv run python scripts/fuse_mixed.py --model <4-bit base> --adapter <dir> --out <dir> [--bits 8|16]

`mlx_lm.fuse` either re-quantizes every fused layer at the base's precision, which rounds
most of a small LoRA update away (evals/RESULTS.md, "Fusing the adapter"), or dequantizes
the whole model. This fuses only the layers the adapter touched, keeps them in the base's
float dtype (`--bits 16`) or re-quantizes them at 8 bits, and leaves every other layer's
4-bit weights exactly as they were. mlx_lm reads the per-layer precision back from the
config's `quantization` entries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten
from mlx_lm.utils import load, save


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--bits",
        type=int,
        choices=(16, 8),
        default=8,
        help="precision of the fused layers: 16 keeps them in float, 8 re-quantizes them",
    )
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    # A local directory for the base, so load and save work offline from the cache.
    base = (
        args.model
        if Path(args.model).exists()
        else snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.py", "*.txt"])
    )
    model, tokenizer, config = load(base, adapter_path=args.adapter, return_config=True)
    fused = [(name, module.fuse(dequantize=True)) for name, module in model.named_modules() if hasattr(module, "fuse")]
    if not fused:
        raise SystemExit(f"no LoRA layers in {args.adapter}")
    if args.bits == 8:
        fused = [
            (name, nn.QuantizedLinear.from_linear(linear, group_size=args.group_size, bits=8)) for name, linear in fused
        ]
    model.update_modules(tree_unflatten(fused))

    # A fused layer's own entry overrides the base's precision on load; a float layer is
    # recorded as unquantized (False).
    entry = {"group_size": args.group_size, "bits": 8} if args.bits == 8 else False
    for key in ("quantization", "quantization_config"):
        if key in config:
            config[key] = {**config[key], **{name: entry for name, _ in fused}}
    save(Path(args.out), base, model, tokenizer, config, donate_model=False)
    print(f"fused {len(fused)} layers at {args.bits} bits into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
