"""`moelars service`: run `moelars serve --mcp` as a launchd agent that starts at login (macOS).

    moelars service install --backend mlx --model <model> --adapter <dir> --calibration <file>
    moelars service status | start | stop | restart | logs | uninstall

`install` writes ~/Library/LaunchAgents/dev.moelars.serve.plist and loads it. The agent runs
the Python that ran `install`, so install from a fixed environment (`uv tool install
'.[mlx,mcp]'`), not a development checkout that `uv sync` can change under it. It starts at
login with HF_HUB_OFFLINE=1, so login never waits on the network; `install` checks the
weights are already cached. The model loads on the first request and unloads after
`--idle-unload` (15 minutes by default) without one. Requests need the API key in
~/.config/moelars/api-key, created on first install (mode 600); the plist names the file and
never holds the key. launchd restarts the server if it crashes, at most every 30 s.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

LABEL = "dev.moelars.serve"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG_DIR = Path.home() / "Library" / "Logs" / "moelars"
API_KEY_FILE = Path.home() / ".config" / "moelars" / "api-key"


def ensure_api_key(path: Path = API_KEY_FILE) -> Path:
    """Create the key file (random, mode 600, directory 700) if there is none; keep an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if not path.exists() or not path.read_text().strip():
        path.write_text(secrets.token_urlsafe(32) + "\n")
    os.chmod(path, 0o600)
    return path


def _absolute(value: str | None) -> str | None:
    """A local path made absolute (launchd starts the agent in /); a Hub ID unchanged."""
    if value and Path(value).expanduser().exists():
        return str(Path(value).expanduser().resolve())
    return value


def serve_argv(args: argparse.Namespace) -> list[str]:
    """The `moelars serve` arguments the agent runs, from the options given to `install`."""
    argv = ["serve", "--mcp", "--host", args.host, "--port", str(args.port), "--idle-unload", args.idle_unload]
    if args.preset:
        argv += ["--preset", args.preset]
    if not (args.preset and args.backend == "mock"):
        argv += ["--backend", args.backend]
    for flag, value in (("--model", _absolute(args.model)), ("--template", args.template),
                        ("--head", _absolute(args.head)), ("--projection", _absolute(args.projection))):
        if value:
            argv += [flag, value]
    for adapter in args.adapter or []:
        argv += ["--adapter", _absolute(adapter)]
    for calibration in args.calibration or []:
        argv += ["--calibration", _absolute(calibration)]
    return argv


def build_plist(program: list[str], process_type: str = "Standard", api_key_file: Path = API_KEY_FILE,
                log_dir: Path = LOG_DIR) -> dict[str, Any]:
    return {
        "Label": LABEL,
        "ProgramArguments": program,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},  # restart after a crash, not after `service stop`
        "ThrottleInterval": 30,
        "ProcessType": process_type,
        "WorkingDirectory": str(Path.home()),
        "EnvironmentVariables": {
            "HF_HUB_OFFLINE": "1",
            "MOELARS_API_KEY_FILE": str(api_key_file),
            "MOELARS_MLX_CACHE_GB": "4",
            "PYTHONUNBUFFERED": "1",
        },
        "StandardOutPath": str(log_dir / "serve.log"),
        "StandardErrorPath": str(log_dir / "serve.err.log"),
    }


def check_cached(args: argparse.Namespace) -> list[str]:
    """Hub IDs that are not in the local cache: an offline start would fail on them."""
    from moelars.presets import PRESETS, _is_hub_id

    ids = [args.model] if args.model else []
    ids += list(args.adapter or [])
    if args.preset and args.preset in PRESETS:
        preset = PRESETS[args.preset]
        ids += [preset.model] if not args.model else []
        ids += list(preset.adapters) if not args.adapter else []
    missing = []
    for repo in [i for i in ids if i and _is_hub_id(i)]:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=repo, local_files_only=True)
        except Exception:  # noqa: BLE001 - any failure means an offline start cannot load it
            missing.append(repo)
    return missing


def _launchctl(*argv: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *argv], capture_output=True, text=True, check=check)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def loaded() -> bool:
    return _launchctl("print", f"{_domain()}/{LABEL}").returncode == 0


def _health(port: int) -> dict[str, Any] | None:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            return json.loads(response.read())
    except OSError:
        return None


def _port_from_plist() -> int:
    if not PLIST.exists():
        return 8600
    program = plistlib.loads(PLIST.read_bytes())["ProgramArguments"]
    return int(program[program.index("--port") + 1]) if "--port" in program else 8600


def cmd_install(args: argparse.Namespace) -> int:
    if sys.platform != "darwin":
        raise SystemExit("moelars service uses launchd, so it runs on macOS only")
    if args.backend == "mock" and not args.preset:
        raise SystemExit("give --preset, or --backend and --model: a login service for the mock backend is a mistake")
    missing = [] if args.skip_cache_check else check_cached(args)
    if missing:
        raise SystemExit("not in the local Hugging Face cache, so an offline start would fail: "
                         + ", ".join(missing) + "\ndownload them once (run `moelars serve` with the same options), "
                         "or pass --skip-cache-check")
    program = [sys.executable, "-m", "moelars", *serve_argv(args)]
    if "/.venv/" in sys.executable:
        print("warning: installing from a project virtualenv; `uv sync` there will change the running service. "
              "A fixed install is steadier: uv tool install '.[mlx,mcp]'", file=sys.stderr)
    plist = build_plist(program, process_type=args.process_type)
    if args.dry_run:
        sys.stdout.write(plistlib.dumps(plist).decode())
        return 0
    key_file = ensure_api_key()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    if loaded():
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    PLIST.write_bytes(plistlib.dumps(plist))
    result = _launchctl("bootstrap", _domain(), str(PLIST))
    if result.returncode != 0:
        raise SystemExit(f"launchctl bootstrap failed: {result.stderr.strip()}")
    url = f"http://127.0.0.1:{args.port}"
    print(f"installed {PLIST}\nserving {url} (MCP at {url}/mcp); logs in {LOG_DIR}\nAPI key in {key_file}\n")
    print("Claude Code:\n  claude mcp add --transport http --scope user moelars "
          f"{url}/mcp --header \"Authorization: Bearer $(cat {key_file})\"")
    print("Any stdio MCP client (Codex, others):\n  command: " + " ".join([sys.executable, "-m", "moelars"])
          + f" mcp-bridge --url {url}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    if not PLIST.exists():
        print("not installed (moelars service install ...)")
        return 1
    health = _health(_port_from_plist())
    state = "loaded in launchd" if loaded() else "not loaded (moelars service start)"
    print(f"{LABEL}: {state}")
    print(f"server: {health}" if health else "server: not answering on /healthz")
    return 0 if health else 1


def cmd_start(args: argparse.Namespace) -> int:
    if not PLIST.exists():
        raise SystemExit("not installed (moelars service install ...)")
    if loaded():
        _launchctl("kickstart", f"{_domain()}/{LABEL}")
    else:
        _launchctl("bootstrap", _domain(), str(PLIST), check=True)
    for _ in range(40):
        if _health(_port_from_plist()):
            print("started")
            return 0
        time.sleep(0.5)
    print(f"started, but /healthz is not answering yet; see {LOG_DIR / 'serve.err.log'}")
    return 1


def cmd_stop(_: argparse.Namespace) -> int:
    """Unload the agent until `start` or the next login."""
    if loaded():
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    print("stopped (starts again at next login; `moelars service uninstall` to remove it)")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    if not loaded():
        return cmd_start(args)
    _launchctl("kickstart", "-k", f"{_domain()}/{LABEL}", check=True)
    print("restarted")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    for name in ("serve.err.log", "serve.log"):
        path = LOG_DIR / name
        if path.exists():
            print(f"==> {path} <==")
            print("".join(path.read_text().splitlines(keepends=True)[-args.lines:]), end="")
    return 0


def cmd_uninstall(_: argparse.Namespace) -> int:
    if loaded():
        _launchctl("bootout", f"{_domain()}/{LABEL}")
    if PLIST.exists():
        PLIST.unlink()
    print(f"removed {PLIST}; the API key stays in {API_KEY_FILE} and logs in {LOG_DIR}")
    return 0


def add_parser(sub: Any, add_backend_args: Any) -> None:
    service = sub.add_parser("service", help="Run the server at login as a launchd agent (macOS)")
    actions = service.add_subparsers(dest="action", required=True)
    install = actions.add_parser("install", help="write and load the launchd agent")
    add_backend_args(install)
    install.add_argument("--host", default="127.0.0.1")
    install.add_argument("--port", type=int, default=8600)
    install.add_argument("--idle-unload", default="15m", help="free the model after this long idle; 0 keeps it")
    install.add_argument("--process-type", default="Standard", choices=["Standard", "Interactive", "Adaptive"])
    install.add_argument("--skip-cache-check", action="store_true")
    install.add_argument("--dry-run", action="store_true", help="print the plist and change nothing")
    install.set_defaults(func=cmd_install)
    for name, fn, text in (("status", cmd_status, "is it loaded and answering"),
                           ("start", cmd_start, "load it now"), ("stop", cmd_stop, "unload it until next login"),
                           ("restart", cmd_restart, "restart the server (after changing weights or code)"),
                           ("uninstall", cmd_uninstall, "unload it and remove the plist")):
        actions.add_parser(name, help=text).set_defaults(func=fn)
    logs = actions.add_parser("logs", help="tail the server logs")
    logs.add_argument("-n", "--lines", type=int, default=40)
    logs.set_defaults(func=cmd_logs)
