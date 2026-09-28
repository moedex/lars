"""Labeled CI failures from GitLab, for the `ci_failure` decision: did fixing it need a code change?

    uv run python scripts/extract_ci_failures.py --since 2026-07-28 --out data/tc/ci_failure

Each failed job becomes one row, in the calibration JSONL format:

- state: the job's name, stage, GitLab failure reason, and the tail of its log. ANSI codes and
  section markers are stripped, and anything that looks like a secret, token, e-mail address
  or connection string is redacted.
- question: a noul, CLAIM below.
- label: read from what happened next, with no human in the loop.
  - "0" (not a code change): the same job passed on a retry in the same pipeline, or a later
    pipeline on the same ref and the same commit succeeded.
  - "1" (code change): the failure reproduced on the same commit (a retry of the job failed
    again, or a later pipeline on that commit failed too), and the next success on the ref
    came on a different commit.
  - Everything else is left out. That covers a failure with no later success, and a single
    failure followed by a new commit that passed. The second case was most of the first
    extraction (93% "code"), because on merge-request pipelines nearly every follow-up is a
    new commit, so a flaky failure plus any push read as a code change. `--loose` keeps
    those as "1" to reproduce that run.

- features: numbers from outside the log, which calibrate fuses with the model (see FEATURES
  below). Zero-shot on the log alone the model barely beat always-"1", so most of the signal
  is expected to be in the history of the project and job, and in a few cheap cues.

Rows are split by time: the newest `--test-fraction` go to `test.jsonl`, the rest to
`calibration.jsonl`. Output goes under data/, which git ignores. Logs can hold internal
details, so keep the output off shared storage.

The GitLab token comes from `glab config get token --host <host>` and is never printed.

To add or recompute features on an extraction already on disk, without GitLab:

    uv run python scripts/extract_ci_failures.py --refeature data/tc/ci_failure

At serve time a caller gets the same numbers for a new failure from `live_features`, reading
the extraction as its history, and passes them to moelars_check as `features` (with
`decision="ci_failure"`):

    from extract_ci_failures import live_features   # scripts/ on sys.path
    features = live_features(["data/tc/ci_failure/calibration.jsonl", "data/tc/ci_failure/test.jsonl"],
                             project="group/app", job_name="e2e:chrome", failure_reason="script_failure",
                             log_tail=tail, created_at=job["created_at"], pipeline=job["pipeline"]["id"])

Each row's meta records when its label was settled (settled_at), so history only counts labels
a caller could have known when the row's job was created. Extractions made before that field
existed fall back to created_at and keep a pipeline's rows out of each other's history.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import httpx

CLAIM = ("Fixing this CI failure needed a change to the code or configuration in the repository, "
         "not just a rerun (it is not infrastructure, network, runner or a flaky test).")
QUESTION = {"type": "noul", "instructions": CLAIM}
TAIL_LINES = 60
TAIL_CHARS = 5000

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
SECTION = re.compile(r"section_(start|end):\d+:[\w.-]+(\[[^\]]*\])?\r?")
REDACTIONS = [
    # Credentials in a URL (https://user:pass@host) go first, whole: the e-mail rule below would
    # otherwise take "pass@host" and leave the part of the password before any "!" or ":".
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s@]+@"), r"\1[REDACTED]@"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    # After the bearer rule, so "Authorization: Bearer <token>" loses the token, not just "Bearer".
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|authorization)(\s*[:=]\s*|\s+)(\S+)"),
     r"\1\2[REDACTED]"),
    # Any assignment to a name with "pass" in it (DB_PASS=, ;Pass=). Only "=": "Passed: 853" is
    # a test summary, not a secret.
    (re.compile(r"(?i)\b(\w*pass\w*)(\s*=\s*)([^;\"'\s]+)"), r"\1\2[REDACTED]"),
    # mysql -phunter2: the password glued to the flag.
    (re.compile(r"(?i)(\b(?:mysql\w*|mariadb\w*)\b[^\n]*?\s-p)(?=\S)([^\s-]\S*)"), r"\1[REDACTED]"),
    # Headers. Anything passed with curl -H / --header loses its whole value ("Basic abc" is two
    # words); elsewhere only a header whose name says it is a credential (X-Auth, X-Api-Key,
    # Cookie). Not any "...Key...:" name: "KeyNotFoundException: The given key" is the error.
    (re.compile(r"(?i)((?:\s-H|--header)[=\s]*([\"']?)[\w-]+\s*:\s*)[^\"'\n]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(\bX-[\w-]*(?:auth|key|token|secret|session|signature)[\w-]*\s*:\s*|"
                r"\b(?:set-)?cookie\s*:\s*)[^\s\"']+"), r"\1[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"), "[GITLAB_TOKEN]"),
    (re.compile(r"(?i)(Password|Pwd)=[^;\"'\s]+"), r"\1=[REDACTED]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Za-z0-9+/_-]{40,}={0,2}\b"), "[LONG_TOKEN]"),
]


# Features. Every row gets all of them, because calibrate fits a fusion only when every row has
# features, and serve-time fusion needs every fitted name.
#
# History rates are the past share of label-"0" (a rerun fixed it) failures, counting only
# failures whose label was settled strictly before the row was created (meta.settled_at, see
# known_at), so a row never sees its own label, one settled by the same event, or anything a
# caller at serve time could not have known yet.
# They are smoothed additively, rate = (zeros + STRENGTH * prior) / (count + STRENGTH):
# - the global prior is the share of "0" in all earlier rows, itself smoothed toward
#   BASE_RERUN_RATE, near the ~0.32 share of "0" in the first strict extraction;
# - a project's rate is smoothed toward the global prior;
# - a job name's rate within its project is smoothed toward that project's rate, so a job
#   with little history falls back to its project rather than to the whole fleet.
# With no history every rate is BASE_RERUN_RATE. STRENGTH = 5 means five past failures weigh
# as much as the prior. *_history is log1p(count) of the past failures behind each rate, so
# the fusion can learn how far to trust it.
BASE_RERUN_RATE = 0.3
STRENGTH = 5.0
FEATURES = ("failure_reason_not_script", "project_rerun_rate", "project_history", "job_rerun_rate",
            "job_history", "name_e2e", "name_deploy", "log_timeout", "log_connection",
            "log_registry_auth", "log_no_space", "log_oom")

NAME_E2E = re.compile(r"(?i)(e2e|end[-_ ]?to[-_ ]?end|\bui\b|ui[-_:]|integration|selenium|playwright|cypress)")
NAME_DEPLOY = re.compile(r"(?i)(deploy|registry|publish|release|docker|image|push)")
LOG_CUES = {
    "log_timeout": re.compile(r"(?i)(timed? ?out|timeout|deadline exceeded)"),
    "log_connection": re.compile(r"(?i)(connection (reset|refused|closed)|econnreset|econnrefused|"
                                 r"could not resolve host|temporary failure in name resolution|broken pipe|"
                                 r"network is unreachable)"),
    # A 401/403 on the same line as a registry, image pull/push or docker login: an expired
    # or missing credential on the runner, not the change under test.
    "log_registry_auth": re.compile(r"(?i)^(?=.*\b(401|403|unauthorized|forbidden|denied)\b)"
                                    r"(?=.*(registry|docker|pull|push|manifest|nuget|npm|feed)).*$", re.MULTILINE),
    "log_no_space": re.compile(r"(?i)no space left on device|disk (is )?full"),
    "log_oom": re.compile(r"(?i)(out of memory|\boom\b|oomkilled|cannot allocate memory|exit code 137|"
                          r"killed signal 9|javascript heap out of memory)"),
}


def cue_features(job_name: str, failure_reason: str | None, log_tail: str) -> dict[str, float]:
    """0/1 cues from the job name, GitLab's failure reason and the log tail. A failure_reason
    other than script_failure (runner_system_failure, stuck_or_timeout_failure, ...) means the
    job died around the script, which reruns usually fix."""
    reason = (failure_reason or "").strip()
    cues = {"failure_reason_not_script": float(bool(reason) and reason not in ("script_failure", "unknown")),
            "name_e2e": float(bool(NAME_E2E.search(job_name))),
            "name_deploy": float(bool(NAME_DEPLOY.search(job_name)))}
    cues.update({name: float(bool(pattern.search(log_tail))) for name, pattern in LOG_CUES.items()})
    return cues


class History:
    """Counts of past labeled failures, for the smoothed rerun-fix rates."""

    def __init__(self) -> None:
        self.total = [0, 0]  # [count, zeros]
        self.project: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self.job: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    def add(self, project: str, job_name: str, label: str, sign: int = 1) -> None:
        zero = int(str(label) == "0")
        for counts in (self.total, self.project[project], self.job[(project, job_name)]):
            counts[0] += sign
            counts[1] += sign * zero

    def features(self, project: str, job_name: str) -> dict[str, float]:
        def smooth(counts: list[int], prior: float) -> float:
            return (counts[1] + STRENGTH * prior) / (counts[0] + STRENGTH)

        glob = smooth(self.total, BASE_RERUN_RATE)
        proj_counts = self.project.get(project, [0, 0])
        job_counts = self.job.get((project, job_name), [0, 0])
        proj = smooth(proj_counts, glob)
        return {"project_rerun_rate": proj, "project_history": math.log1p(proj_counts[0]),
                "job_rerun_rate": smooth(job_counts, proj), "job_history": math.log1p(job_counts[0])}


def parse_time(value: str) -> datetime:
    """An ISO timestamp as an aware UTC datetime, so "12:00:00Z", "12:00:00.500Z" and
    "12:00:00+00:00" order by time, not as strings. A naive time is taken as UTC, which is what
    GitLab writes."""
    moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def parse_state(state: str) -> tuple[str, str | None, str]:
    """(job name, failure reason, log tail) back out of `state_for`'s text, so features can be
    recomputed from rows on disk. The name ends at the line's last " (stage ", so a stage with
    ")" in it ("build (win)") does not lose it. Rows extracted now also keep the name and
    reason in meta; `job_fields` prefers those."""
    head, _, tail = state.partition("log tail:\n")
    name = re.search(r"^job: (.*) \(stage .*\)$", head, re.MULTILINE)
    reason = re.search(r"^failure_reason: (.*)$", head, re.MULTILINE)
    return (name.group(1) if name else "", reason.group(1).strip() if reason else None, tail)


def job_fields(row: dict) -> tuple[str, str | None, str]:
    job_name, reason, tail = parse_state(row["state"])
    meta = row.get("meta", {})
    return meta.get("job_name", job_name), meta.get("failure_reason", reason), tail


def known_at(row: dict) -> datetime:
    """When a row's label became known, which is when it may enter another row's history.
    meta.settled_at is the finish of the run that settled it (the passing retry or pipeline, or
    the success on a new commit). Older extractions lack it and fall back to created_at, which
    is too early; `same_run` then keeps the worst of that out."""
    return parse_time(row["meta"].get("settled_at") or row["meta"]["created_at"])


def same_run(row: dict) -> tuple | None:
    """Rows whose labels one event settles, when settled_at is unknown: every failed attempt of
    a job in one pipeline gets its label from the same retry or later pipeline, so attempt 1's
    label in attempt 2's history would be attempt 2's own. Keyed on the pipeline; None when the
    row has settled_at (its time is then exact) or no pipeline."""
    meta = row["meta"]
    if meta.get("settled_at") or meta.get("pipeline") is None:
        return None
    return (meta["project"], meta["pipeline"])


def add_features(rows: list[dict]) -> list[dict]:
    """Sorts rows by meta.created_at and sets each row's "features". A row's history is the rows
    whose label was known strictly before it was created (`known_at`), less rows of the same run
    (`same_run`), so a row never sees itself or a label settled with its own. A pure function
    of the rows."""
    rows = sorted(rows, key=lambda r: parse_time(r["meta"]["created_at"]))
    events = sorted(rows, key=known_at)
    history, by_run = History(), defaultdict(list)
    added = 0
    for row in rows:
        created = parse_time(row["meta"]["created_at"])
        while added < len(events) and known_at(events[added]) < created:
            event = events[added]
            history.add(event["meta"]["project"], job_fields(event)[0], event["label"])
            if (run := same_run(event)) is not None:
                by_run[run].append(event)
            added += 1
        job_name, reason, tail = job_fields(row)
        run = same_run(row)
        peers = [peer for peer in by_run.get(run, []) if peer is not row] if run else []
        for peer in peers:
            history.add(peer["meta"]["project"], job_fields(peer)[0], peer["label"], sign=-1)
        row["features"] = {**cue_features(job_name, reason, tail),
                           **history.features(row["meta"]["project"], job_name)}
        for peer in peers:
            history.add(peer["meta"]["project"], job_fields(peer)[0], peer["label"])
    return rows


def read_rows(paths: Iterable[str | Path]) -> list[dict]:
    rows = []
    for path in paths:
        with Path(path).open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def live_features(history_paths: Iterable[str | Path], project: str, job_name: str,
                  failure_reason: str | None, log_tail: str, created_at: str | None = None,
                  pipeline: int | None = None) -> dict[str, float]:
    """Features for a failure not in the history, the same numbers add_features would give it.
    `history_paths` are extraction JSONL files (calibration.jsonl, test.jsonl); only rows whose
    label was known before `created_at` (ISO, any offset) count, or all of them when it is None.
    Pass the failure's `pipeline` id to leave out rows of the same pipeline that lack a
    settled_at, as add_features does. The log tail should be cleaned with clean_log, as
    extracted rows were."""
    cutoff = parse_time(created_at) if created_at is not None else None
    run = (project, pipeline) if pipeline is not None else None
    history = History()
    for row in read_rows(history_paths):
        if cutoff is not None and not known_at(row) < cutoff:
            continue
        if run is not None and same_run(row) == run:
            continue
        history.add(row["meta"]["project"], job_fields(row)[0], row["label"])
    return {**cue_features(job_name, failure_reason, log_tail), **history.features(project, job_name)}


def write_split(rows: list[dict], out: Path, test_fraction: float) -> int:
    """Features, then the time split: calibration.jsonl gets the oldest rows. Returns its size."""
    rows = add_features(rows)
    cut = int(len(rows) * (1 - test_fraction))
    for name, part in (("calibration", rows[:cut]), ("test", rows[cut:])):
        with (out / f"{name}.jsonl").open("w") as handle:
            for row in part:
                handle.write(json.dumps(row) + "\n")
    return cut


def redact(line: str) -> str:
    for pattern, replacement in REDACTIONS:
        line = pattern.sub(replacement, line)
    return line


def clean_log(text: str) -> str:
    """The log's last TAIL_LINES non-blank lines, at most TAIL_CHARS, markup stripped and secrets
    redacted. Each line is redacted whole, before the tail is cut: cutting first can take the key
    word off the front of "password=hunter2" and leave the value for no rule to match. A line the
    character cut splits is dropped for the same reason."""
    lines = [SECTION.sub("", ANSI.sub("", line)).rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    lines = [redact(line) for line in lines if line.strip()][-TAIL_LINES:]
    tail = "\n".join(lines)
    if len(tail) > TAIL_CHARS:
        tail = tail[-TAIL_CHARS:].partition("\n")[2]
    return tail


def state_for(job: dict, log_tail: str) -> str:
    return (f"job: {job['name']} (stage {job['stage']})\n"
            f"failure_reason: {job.get('failure_reason') or 'unknown'}\n"
            f"log tail:\n{log_tail}")


class GitLab:
    def __init__(self, host: str) -> None:
        token = subprocess.run(["glab", "config", "get", "token", "--host", host], capture_output=True,
                               text=True, check=True).stdout.strip()
        self.http = httpx.Client(base_url=f"https://{host}/api/v4", headers={"PRIVATE-TOKEN": token}, timeout=60)

    def get(self, path: str, **params) -> httpx.Response:
        response = self.http.get(path, params=params)
        for attempt in range(4):
            if response.status_code != 429 and response.status_code < 500:
                break
            time.sleep(2 ** attempt)
            response = self.http.get(path, params=params)
        response.raise_for_status()
        return response

    def pages(self, path: str, **params):
        page = 1
        while True:
            response = self.get(path, per_page=100, page=page, **params)
            items = response.json()
            yield from items
            if not response.headers.get("x-next-page"):
                return
            page += 1


def label_outcomes(pipelines: list[dict]) -> dict[int, tuple[str, dict]]:
    """Pipeline id -> (label, the later pipeline that settled it) for failed pipelines with a
    later success on their ref. The label is "0" (fixed without a code change: a later success
    on the same commit settles it), "1" (reproduced on its commit, then fixed by a new one: the
    new commit's success settles it) or "new_commit" (failed once, then a new commit passed:
    ambiguous)."""
    by_ref: dict[str, list[dict]] = defaultdict(list)
    for p in pipelines:
        by_ref[p["ref"]].append(p)
    outcomes: dict[int, tuple[str, dict]] = {}
    for runs in by_ref.values():
        runs.sort(key=lambda p: p["id"])
        for i, p in enumerate(runs):
            if p["status"] != "failed":
                continue
            later_success = next((q for q in runs[i + 1:] if q["status"] == "success"), None)
            if later_success is None:
                continue
            same_commit = next((q for q in runs[i + 1:] if q["status"] == "success" and q["sha"] == p["sha"]), None)
            if same_commit is not None:
                outcomes[p["id"]] = ("0", same_commit)
                continue
            before_fix = runs[i + 1: runs.index(later_success)]
            reproduced = any(q["status"] == "failed" and q["sha"] == p["sha"] for q in before_fix)
            outcomes[p["id"]] = ("1" if reproduced else "new_commit", later_success)
    return outcomes


def label_failures(pipelines: list[dict]) -> dict[int, str]:
    """Pipeline id -> label, as label_outcomes without the settling pipeline."""
    return {pid: label for pid, (label, _) in label_outcomes(pipelines).items()}


def finished(item: dict) -> str:
    """When a job or pipeline finished, the moment its outcome was known."""
    return item.get("finished_at") or item.get("updated_at") or item["created_at"]


def update_summary(out: Path, **fields) -> dict:
    """Merges fields into out/summary.json, keeping what an earlier run wrote there."""
    path = out / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary.update(fields)
    path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="gitlab.tcdevops.com")
    parser.add_argument("--since", help="ISO date; pipelines updated after it")
    parser.add_argument("--out", default="data/tc/ci_failure")
    parser.add_argument("--projects", default=None, help="comma list of project ids or paths (default: all active)")
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--loose", action="store_true",
                        help='label a single failure fixed by a new commit "1" (noisy; see the docstring)')
    parser.add_argument("--refeature", metavar="DIR",
                        help="recompute features for DIR/calibration.jsonl and DIR/test.jsonl in place, no GitLab")
    args = parser.parse_args()
    if args.refeature:
        out = Path(args.refeature)
        rows = read_rows([out / "calibration.jsonl", out / "test.jsonl"])
        cut = write_split(rows, out, args.test_fraction)
        update_summary(out, rows=len(rows), calibration=cut, test=len(rows) - cut, test_fraction=args.test_fraction,
                       features=list(FEATURES))
        print(json.dumps({"rows": len(rows), "calibration": cut, "test": len(rows) - cut, "features": FEATURES}))
        return 0
    if not args.since:
        parser.error("--since is required unless --refeature")
    gl = GitLab(args.host)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.projects:
        projects = [{"id": p.strip(), "path_with_namespace": p.strip()} for p in args.projects.split(",")]
    else:
        projects = list(gl.pages("/projects", simple="true", last_activity_after=f"{args.since}T00:00:00Z",
                                 order_by="last_activity_at"))
    print(f"{len(projects)} projects active since {args.since}", flush=True)

    rows, stats = [], Counter()
    for project in projects:
        pid = str(project["id"]).replace("/", "%2F")
        try:
            pipelines = list(gl.pages(f"/projects/{pid}/pipelines", updated_after=f"{args.since}T00:00:00Z"))
        except httpx.HTTPStatusError:
            stats["project_unreadable"] += 1
            continue
        outcomes = label_outcomes(pipelines)
        stats["failed_pipelines"] += sum(p["status"] == "failed" for p in pipelines)
        for p in pipelines:
            if p["status"] != "failed":
                continue
            if p["id"] not in outcomes:
                stats["unknown_outcome"] += 1
                continue
            jobs = list(gl.pages(f"/projects/{pid}/pipelines/{p['id']}/jobs", include_retried="true"))
            passed: dict[str, str] = {}  # job name -> when its first passing attempt finished
            for j in jobs:
                if j["status"] == "success":
                    passed[j["name"]] = min(passed.get(j["name"], finished(j)), finished(j), key=parse_time)
            failed_attempts = Counter(j["name"] for j in jobs if j["status"] == "failed")
            pipeline_label, settler = outcomes[p["id"]]
            for job in jobs:
                if job["status"] != "failed" or job.get("allow_failure"):
                    continue
                # A retry of this job that passed in the same pipeline settles it: a rerun was enough.
                if job["name"] in passed:
                    label, settled_at = "0", passed[job["name"]]
                else:
                    label, settled_at = pipeline_label, finished(settler)
                if label == "new_commit":
                    # A retry that failed again reproduces it on this commit just as a re-run pipeline does.
                    label = "1" if failed_attempts[job["name"]] >= 2 or args.loose else None
                if label is None:
                    stats["ambiguous"] += 1
                    continue
                try:
                    trace = gl.get(f"/projects/{pid}/jobs/{job['id']}/trace").text
                except httpx.HTTPStatusError:
                    stats["trace_unreadable"] += 1
                    continue
                rows.append({"state": state_for(job, clean_log(trace)), "question": QUESTION, "label": label,
                             "meta": {"project": project["path_with_namespace"], "pipeline": p["id"],
                                      "ref": p["ref"], "sha": p["sha"], "job": job["id"], "job_name": job["name"],
                                      "failure_reason": job.get("failure_reason") or "unknown",
                                      "created_at": job["created_at"], "settled_at": settled_at,
                                      "source": p["source"]}})
                stats[f"label_{label}"] += 1
                if len(rows) >= args.max_rows:
                    break
            if len(rows) >= args.max_rows:
                break
        print(f"{project['path_with_namespace']}: {len(rows)} rows so far", flush=True)
        if len(rows) >= args.max_rows:
            break

    cut = write_split(rows, out, args.test_fraction)
    summary = {"since": args.since, "rows": len(rows), "calibration": cut, "test": len(rows) - cut,
               "test_fraction": args.test_fraction, "claim": CLAIM, "features": list(FEATURES), **stats}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
