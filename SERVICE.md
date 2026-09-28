# moe-LARS as a login service with an MCP endpoint

Status: draft spec, not built. Goal: moe-LARS starts with the Mac, stays out of the way, and
coding agents (Claude Code, Codex, anything that speaks MCP) can ask it typed questions and
get calibrated answers back.

## Shape

```
launchd (LaunchAgent, at login)
  └── moelars serve --preset ... --mcp --idle-unload 15m      one process, 127.0.0.1:8600
        ├── POST /v1/systemone     existing HTTP API (SDKs, scripts)
        ├── GET  /healthz, /v1/models
        └── /mcp                   MCP over Streamable HTTP, same engine, same one-at-a-time queue

agents ── MCP over HTTP ──> http://127.0.0.1:8600/mcp
agents without HTTP MCP ── stdio ──> moelars mcp-bridge ── HTTP ──> the same server
```

**One process for every agent.** MCP's stdio transport has each client spawn its own server.
Laya does that because its model is 0.8 GB; ours is about 18 GB, so five agent sessions
would load it five times. The service loads it once and exposes MCP over Streamable HTTP
from the same FastAPI app. The existing limiter already runs one inference at a time, so
concurrent agents queue instead of racing on the model.

## 1. MCP endpoint

- `moelars serve --mcp` mounts an MCP server at `/mcp`, built with the official `mcp` Python
  SDK (`FastMCP(...).streamable_http_app()`), in a new optional extra: `moelars[mcp]`. The
  tools call the same `Engine` object as `/v1/systemone`, through the same limiter.
- Stateless mode (`stateless_http=True`): every tool call is independent, so there is no
  session state to lose on restart.

Tools. There are few of them, and each says what it's good for and where it's weak, because
the agent chooses tools from these descriptions:

| tool | input | output |
|---|---|---|
| `moelars_check` | `text`, `claim` | `p_yes`, `confidence` |
| `moelars_classify` | `text`, `question`, `options` (name → description) | `choice`, `probabilities`, `confidence` |
| `moelars_decide` | the full System One request: `state`, `questions` (noul, choice, score, multi), optional `moelars` extensions | the full System One response |
| `moelars_status` | none | model, adapter, calibrator, loaded or unloaded, queue depth, version |

- The two simple tools cover most agent uses: "does this log show a failure?", "is this
  diff risky?", "which area does this issue belong to?" `moelars_decide` is for callers who
  want several questions answered over one state in one pass.
- Results come back as MCP structured content with an output schema, plus a short text
  rendering.
- Tool descriptions state the limits honestly: no world knowledge beyond the base model's
  (mmlu 0.775), about 0.3 s per question, the input token cap, and the fact that
  probabilities are calibrated on jev-bench-like tasks. Agents should read `confidence` as
  "how sure", not as a guarantee on inputs unlike the benchmark.
- Errors (too many tokens, bad schema, model unavailable) become MCP tool errors carrying
  the existing `{message, error_type}` text, so the agent can correct and retry.

## 2. Security

- Bind `127.0.0.1` only; no LAN exposure by default.
- Validate the `Origin` header on `/mcp`, rejecting any origin that isn't absent or
  localhost. The MCP spec requires this for local HTTP servers: without it, any web page
  open in a browser could reach the model through DNS rebinding. The same check goes on
  `/v1/systemone`.
- Optional bearer token: `MOELARS_API_KEY` already gates the HTTP API, and `/mcp` reuses it.
  Clients send it as a header (for Claude Code, `claude mcp add --header`).
- Logs record request IDs, token counts and latency, never state text. Agents will send
  code and logs.

## 3. Memory: load on demand, unload when idle

You'd rather it use less than 18 GB, and an agent calls it in bursts.

- `--idle-unload 15m`: drop the model after 15 idle minutes; the next request loads it again.
  Measured loads with a warm page cache take 0.8 to 2.9 s, so the first question after an idle
  spell costs about 3 s and the rest cost the usual 0.3 s.
- `/healthz` and `moelars_status` answer while unloaded, without loading.
- **Needs verifying on real MLX:** dropping the model and calling `mx.clear_cache()` must
  actually return the roughly 18 GB to the OS. If MLX holds on to it, unloading means
  restarting the worker process instead, a larger change.
- `--idle-unload 0` keeps the model resident, for when latency matters more.

## 4. Autostart with launchd

**A LaunchAgent, starting at login, not a LaunchDaemon starting at boot.** A daemon runs
before any login, as root or a system user. It could work, but it would also need its own
copy of the Hugging Face cache and its own Python environment, and Metal access from a
daemon is worth verifying first. An agent runs as you, with your cache, venv and GPU
session, which is all a coding agent needs, since agents only run once you're logged in.

- `moelars service install [serve options]` writes
  `~/Library/LaunchAgents/dev.moelars.serve.plist` and loads it with
  `launchctl bootstrap gui/$(id -u)`. The plist records:
  - **Program:** the absolute path of the installed `moelars` executable, plus the serve
    options.
  - **Start and restart:** `RunAtLoad`, and `KeepAlive` on crash (`SuccessfulExit: false`)
    with `ThrottleInterval: 30`, so a crashing server can't spin.
  - **Environment:** `HF_HUB_OFFLINE=1`, so boot never waits on the network. The weights
    must already be in the cache; `install` checks that.
  - **Also in the environment:** `MOELARS_MLX_CACHE_GB`, and `MOELARS_API_KEY` if you set
    one.
  - **Logs:** `~/Library/Logs/moelars/serve.log` and `serve.err.log`.
- Also `moelars service start | stop | restart | status | logs | uninstall`, thin wrappers
  over `launchctl` (`kickstart -k`, `bootout`, `print`).
- **Needs verifying:** `ProcessType`. launchd may run a background agent at lower priority,
  which could slow GPU work; measure latency as `Standard` against `Interactive`.
- Run it from a fixed install (`uv tool install ./` into its own environment), not the repo
  checkout. Then `uv sync` during development can't swap code under the running service,
  and an upgrade is `uv tool install` plus `moelars service restart`.

## 5. What it serves

- Today: `--adapter checkpoints/lora-30b-c-s1 --calibration calibration/served/lora-30b-c-s1.json`
  (the current single-adapter default).
- Once the adapter is on the Hub: `--preset 30b`, cached locally for offline boot.
- Changing it is `moelars service install` with new options, then `restart`. There's no hot
  swap; a restart costs a few seconds.
- Whatever wins the current work (corpus D, the two-seed average) gets in the same way.

## 6. Living with training

- Serving (18.6 GB) plus attention LoRA training (33 GB) fits in 128 GB, but they share the
  GPU: training slows down and requests wait. With `--idle-unload`, an unused service holds
  no memory during overnight runs.
- For the heavy runs (the experts preset peaks at 93 GB), the queue scripts call
  `moelars service stop` at the start and `start` at the end, next to their existing
  free-memory guard.

## 7. Agent setup

- Claude Code:
  `claude mcp add --transport http --scope user moelars http://127.0.0.1:8600/mcp`
  (add `--header "Authorization: Bearer $MOELARS_API_KEY"` if a key is set).
- Codex: `[mcp_servers.moelars]` with `url = "http://127.0.0.1:8600/mcp"` in
  `~/.codex/config.toml`, if your Codex version supports Streamable HTTP servers; verify
  before relying on it. Otherwise use the bridge:
  `command = "moelars"`, `args = ["mcp-bridge", "--url", "http://127.0.0.1:8600"]`.
- `moelars mcp-bridge` is a stdio MCP server that holds no model and forwards each call to
  the service. It's for clients that only speak stdio. Starting it costs nothing, and if the
  service is down it says so in a tool error.

## 8. Tests

- CI (mock backend):
  - an MCP client over Streamable HTTP lists the tools and calls each one;
  - answers match `/v1/systemone` for the same request;
  - a foreign `Origin` is rejected, and a missing or wrong bearer token is refused;
  - the bridge forwards calls and reports a stopped service;
  - the plist is checked against a golden file;
  - idle unload and reload are checked with a fake clock.
- On the Mac, once:
  - memory actually comes back after unload;
  - the service starts after a reboot and login, and reconnects after a crash;
  - the `ProcessType` latency check;
  - Claude Code and Codex each complete a real tool call.

## 9. Work, roughly

| piece | size |
|---|---|
| `/mcp` mount, four tools, output schemas, Origin check, tests | about half a day |
| idle unload and reload, plus the real-MLX memory check | 2 to 3 hours, more if MLX keeps the memory |
| `moelars service` commands and the plist | 2 to 3 hours |
| `mcp-bridge` | about an hour |
| docs (README section, agent setup) | about an hour |

About a day and a half in all, with no GPU needed except the memory and latency checks.
None of it blocks the release work, or is blocked by it.

## Open questions

1. Login start is enough; boot before login isn't needed?
2. Idle unload by default, at 15 minutes?
3. Which agents besides Claude Code and Codex should the setup section cover?
4. A bearer token by default, or localhost with an Origin check only?
