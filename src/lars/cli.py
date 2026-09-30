"""Command line: serve, eval, calibrate, mcp-bridge, service."""

from __future__ import annotations

import argparse
import json
import sys

from lars import __version__
from lars.backends import load_backend
from lars.calibration import Calibrator
from lars.engine import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_ROWS, Engine, EnsembleEngine


def _apply_preset(args: argparse.Namespace) -> None:
    """Fill backend, model, adapters and calibrations from `--preset`; explicit flags win."""
    from lars.presets import PRESETS

    name = getattr(args, "preset", None)
    if not name:
        return
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; choose from {', '.join(sorted(PRESETS))}")
    preset = PRESETS[name]
    if args.backend in (None, "mock"):
        args.backend = preset.backend
    args.model = args.model or preset.model
    if not args.adapter:
        args.adapter = list(preset.adapters)
        args.calibration = args.calibration or list(preset.calibrations)


def _engine_from_args(args: argparse.Namespace) -> Engine | EnsembleEngine:
    from lars.presets import resolve_adapter, resolve_calibration, resolve_model

    _apply_preset(args)
    model = resolve_model(args.model)
    adapters = [resolve_adapter(a) for a in args.adapter or []]
    calibrations = [resolve_calibration(c) for c in args.calibration or []]
    calibrators = [Calibrator.load(c) if c else None for c in calibrations]
    budgets = {}
    if hasattr(args, "max_rows"):  # serve only; eval and calibrate score one row per question
        budgets = {"max_rows": args.max_rows or None, "max_input_tokens": args.max_input_tokens or None}
    named = load_named_calibrators(getattr(args, "calibration_dir", None))
    backend = load_backend(args.backend, model=model, template=args.template,
                           adapter=adapters if len(adapters) > 1 else (adapters[0] if adapters else None))
    if model != args.model:
        # Name the model by the reference it was served from (moedex/lars/4b), not its cache path.
        backend.model_name = backend.model_name.replace(model, args.model, 1)
    if len(adapters) > 1:
        if args.head:
            raise SystemExit("a pointer head cannot be combined with several adapters")
        if len(calibrators) not in (0, len(adapters)):
            raise SystemExit(f"give one --calibration per --adapter ({len(adapters)}), or none")
        if named:
            raise SystemExit("--calibration-dir is not supported with several adapters yet")
        return EnsembleEngine(backend, calibrators or [None] * len(adapters), version=__version__, **budgets)
    if len(calibrators) > 1:
        raise SystemExit("several --calibration files need as many --adapter directories")
    head = None
    if args.head:
        from lars.heads import PointerHeadScorer

        head = PointerHeadScorer.load(args.head, args.projection or str(args.head).replace(".npz", ".projection.npy"))
    return Engine(backend, calibrator=calibrators[0] if calibrators else None, version=__version__, head=head,
                  named_calibrators=named, **budgets)


def _expected_model_id(args: argparse.Namespace) -> str:
    """The engine's model ID before it is built, so `/healthz` can name it while unloaded."""
    from pathlib import Path

    name = "lars-mock" if args.backend == "mock" else "+".join(
        [args.model or "", *(Path(a).name for a in args.adapter or [])])
    return f"lars-{__version__}+{args.backend}:{name}"


def load_named_calibrators(directory: str | None) -> dict[str, Calibrator]:
    """Every `<name>.json` in `directory`, keyed by name: the per-decision calibrators a request can choose."""
    if not directory:
        return {}
    from pathlib import Path

    path = Path(directory).expanduser()
    if not path.is_dir():
        raise SystemExit(f"--calibration-dir {directory} is not a directory")
    return {f.stem: Calibrator.load(f) for f in sorted(path.glob("*.json"))}


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", default=None, help="named configuration, e.g. 30b (see lars.presets)")
    parser.add_argument("--backend", default="mock", choices=["mock", "mlx", "llamacpp"])
    parser.add_argument("--model", default=None, help="Model path or Hugging Face id for the backend")
    parser.add_argument("--template", default=None, help="Chat template name: plain, chatml, gemma, llama3")
    parser.add_argument("--calibration", action="append", default=None,
                        help="calibrator JSON from `lars calibrate` (a Hub `<org>/<repo>/<file>` works too); "
                             "repeat once per --adapter for an ensemble")
    parser.add_argument("--adapter", action="append", default=None,
                        help="LoRA adapter directory from `python -m lars.train.lora`, or a Hugging Face repo ID; "
                             "repeat to serve several adapters of one base model as an averaged ensemble")
    parser.add_argument("--head", default=None, help="Pointer head npz from `python -m lars.train.residual`")
    parser.add_argument("--projection", default=None, help="projection.npy from feature extraction")


def _check_local_paths(args: argparse.Namespace) -> None:
    """Fail at startup, not on the first request, when a local adapter or calibrator is missing."""
    from pathlib import Path

    from lars.presets import _is_hub_id

    for path in [*(args.adapter or []), args.head, args.projection]:
        if path and not _is_hub_id(path) and not Path(path).exists():
            raise SystemExit(f"not found: {path}")
    for path in args.calibration or []:
        if path and not _is_hub_id(path) and not Path(path).exists():
            raise SystemExit(f"not found: {path}")


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from lars.lazy import LazyEngine, parse_duration
    from lars.server import create_app

    idle = parse_duration(args.idle_unload)
    _apply_preset(args)
    _check_local_paths(args)
    if idle:
        # Loaded by the first request, not at startup: a login service costs nothing until used.
        holder = LazyEngine(lambda: _engine_from_args(args), idle_unload=idle, model_id=_expected_model_id(args))
        described = f"{args.backend}:{args.model} (loads on first request, unloads after {idle:.0f}s idle)"
    else:
        engine = _engine_from_args(args)
        holder = LazyEngine(lambda: engine, engine=engine)
        described = engine.model_id
    escalator, note = _escalator_from_args(args)
    print(f"escalation: {note}", file=sys.stderr)
    app = create_app(holder, max_body_bytes=args.max_body_bytes, mcp=args.mcp, escalator=escalator,
                     escalation_note=note)
    endpoints = f"http://{args.host}:{args.port}" + (" (MCP at /mcp)" if args.mcp else "")
    print(f"LARS {__version__} serving {described} on {endpoints}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _escalator_from_args(args: argparse.Namespace):
    """The escalator and a one-line note. The key is resolved here, once: without one, nothing is escalated."""
    if not args.escalate:
        return None, "off (start with --escalate to send low-confidence answers to a hosted model)"
    from lars.escalate import Escalator, resolve_api_key

    key = resolve_api_key()
    if not key:
        return None, ("off: --escalate was given but no API key was found (ANTHROPIC_API_KEY, "
                      "LARS_ANTHROPIC_KEY_FILE, or ~/.config/lars/anthropic-key)")
    named = load_named_calibrators(getattr(args, "calibration_dir", None))
    decisions = {name: c.escalate for name, c in named.items() if c.escalate}
    escalator = Escalator(key, model=args.escalate_model, below=args.escalate_below, weight=args.escalate_weight,
                          decisions=decisions)
    scope = ", ".join(f"{n} below {d['below']}" for n, d in sorted(decisions.items())) or "no decision opts in"
    default = f"; other answers below {args.escalate_below}" if args.escalate_below else ""
    return escalator, f"on ({args.escalate_model}): {scope}{default}"


def cmd_mcp_bridge(args: argparse.Namespace) -> int:
    from lars.bridge import run_bridge

    run_bridge(args.url, api_key_file=args.api_key_file)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from lars.evalset import evaluate, read_examples

    engine = _engine_from_args(args)
    examples = list(read_examples(args.data))
    if args.limit:
        examples = examples[: args.limit]
    result = evaluate(engine, examples)
    print(json.dumps(result.__dict__, indent=2))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from lars.evalset import calibrate, read_examples

    engine = _engine_from_args(args)
    examples = list(read_examples(args.data))
    if args.limit:
        examples = examples[: args.limit]
    calibrator = calibrate(engine, examples, source=args.data)
    if args.escalate_below:
        calibrator.escalate = {"below": args.escalate_below, "weight": args.escalate_weight}
    calibrator.save(args.out)
    print(json.dumps({"temperatures": calibrator.temperatures, "saved": args.out}, indent=2))
    return 0


def add_escalation_args(parser: argparse.ArgumentParser) -> None:
    from lars.escalate import DEFAULT_BELOW, DEFAULT_MODEL, DEFAULT_WEIGHT

    parser.add_argument("--escalate", action="store_true",
                        help="send answers below --escalate-below to a hosted Claude model (needs an API key at "
                             "startup and lars-engine[escalate]); the state text leaves the machine for those")
    parser.add_argument("--escalate-model", default=DEFAULT_MODEL)
    parser.add_argument("--escalate-below", type=float, default=DEFAULT_BELOW,
                        help="server-wide threshold for questions whose decision sets none; 0 (the default) "
                             "escalates only decisions whose calibrator opts in")
    parser.add_argument("--escalate-weight", type=float, default=DEFAULT_WEIGHT,
                        help="weight of the hosted pick in the served probabilities")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lars", description="Limited but Accurate Response System")
    parser.add_argument("--version", action="version", version=f"lars {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP server")
    _add_backend_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8600)
    serve.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                       help="model rows one request may plan (options x permutations x ablations); 0 for no limit")
    serve.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS,
                       help="input tokens one request may need; 0 for no limit")
    serve.add_argument("--max-body-bytes", type=int, default=1_000_000, help="largest request body accepted")
    serve.add_argument("--calibration-dir", default=None,
                       help="directory of per-decision calibrators (<name>.json); a request picks one with "
                            "lars.calibrator or lars.calibrators")
    serve.add_argument("--mcp", action="store_true", help="also serve MCP over Streamable HTTP at /mcp "
                                                           "(needs lars-engine[mcp])")
    add_escalation_args(serve)
    serve.add_argument("--idle-unload", default="0", help="free the model after this long without a request "
                       "(900, 15m, 1h) and load it on the next one; 0 keeps it loaded")
    serve.set_defaults(func=cmd_serve)

    bridge = sub.add_parser("mcp-bridge", help="stdio MCP server that forwards to a running `lars serve`")
    bridge.add_argument("--url", default="http://127.0.0.1:8600")
    bridge.add_argument("--api-key-file", default=None,
                        help="file holding the server's API key (default: LARS_API_KEY, then the service's key)")
    bridge.set_defaults(func=cmd_mcp_bridge)

    from lars.service import add_parser as add_service_parser

    add_service_parser(sub, _add_backend_args)

    ev = sub.add_parser("eval", help="Score a labeled JSONL set")
    _add_backend_args(ev)
    ev.add_argument("--data", required=True)
    ev.add_argument("--limit", type=int, default=0)
    ev.set_defaults(func=cmd_eval)

    cal = sub.add_parser("calibrate", help="Fit temperatures on a labeled JSONL set")
    _add_backend_args(cal)
    cal.add_argument("--data", required=True)
    cal.add_argument("--out", required=True)
    cal.add_argument("--limit", type=int, default=0)
    cal.add_argument("--escalate-below", type=float, default=0.0,
                     help="make this decision escalate answers below this to the hosted model "
                          "(set from evals/escalation_check.py); 0 keeps it local")
    cal.add_argument("--escalate-weight", type=float, default=0.5)
    cal.set_defaults(func=cmd_calibrate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
