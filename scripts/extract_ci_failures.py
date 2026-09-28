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

Rows are split by time: the newest `--test-fraction` go to `test.jsonl`, the rest to
`calibration.jsonl`. Output goes under data/, which git ignores. Logs can hold internal
details, so keep the output off shared storage.

The GitLab token comes from `glab config get token --host <host>` and is never printed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
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
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    # After the bearer rule, so "Authorization: Bearer <token>" loses the token, not just "Bearer".
    (re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_-]?key|authorization)(\s*[:=]\s*|\s+)(\S+)"),
     r"\1\2[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[AWS_KEY]"),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"), "[GITLAB_TOKEN]"),
    (re.compile(r"(?i)(Password|Pwd)=[^;\"'\s]+"), r"\1=[REDACTED]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Za-z0-9+/_-]{40,}={0,2}\b"), "[LONG_TOKEN]"),
]


def clean_log(text: str) -> str:
    lines = [SECTION.sub("", ANSI.sub("", line)).rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if line.strip()]
    tail = "\n".join(lines[-TAIL_LINES:])[-TAIL_CHARS:]
    for pattern, replacement in REDACTIONS:
        tail = pattern.sub(replacement, tail)
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


def label_failures(pipelines: list[dict]) -> dict[int, str]:
    """Pipeline id -> "0" (fixed without a code change), "1" (reproduced on its commit, then fixed
    by a new one) or "new_commit" (failed once, then a new commit passed: ambiguous) for failed
    pipelines with a later success on their ref."""
    by_ref: dict[str, list[dict]] = defaultdict(list)
    for p in pipelines:
        by_ref[p["ref"]].append(p)
    labels: dict[int, str] = {}
    for runs in by_ref.values():
        runs.sort(key=lambda p: p["id"])
        for i, p in enumerate(runs):
            if p["status"] != "failed":
                continue
            later_success = next((q for q in runs[i + 1:] if q["status"] == "success"), None)
            if later_success is None:
                continue
            same_commit_success = any(q["status"] == "success" and q["sha"] == p["sha"] for q in runs[i + 1:])
            if same_commit_success or later_success["sha"] == p["sha"]:
                labels[p["id"]] = "0"
                continue
            before_fix = runs[i + 1: runs.index(later_success)]
            reproduced = any(q["status"] == "failed" and q["sha"] == p["sha"] for q in before_fix)
            labels[p["id"]] = "1" if reproduced else "new_commit"
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="gitlab.tcdevops.com")
    parser.add_argument("--since", required=True, help="ISO date; pipelines updated after it")
    parser.add_argument("--out", default="data/tc/ci_failure")
    parser.add_argument("--projects", default=None, help="comma list of project ids or paths (default: all active)")
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--loose", action="store_true",
                        help='label a single failure fixed by a new commit "1" (noisy; see the docstring)')
    args = parser.parse_args()
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
        labels = label_failures(pipelines)
        stats["failed_pipelines"] += sum(p["status"] == "failed" for p in pipelines)
        for p in pipelines:
            if p["status"] != "failed":
                continue
            if p["id"] not in labels:
                stats["unknown_outcome"] += 1
                continue
            jobs = list(gl.pages(f"/projects/{pid}/pipelines/{p['id']}/jobs", include_retried="true"))
            passed = {j["name"] for j in jobs if j["status"] == "success"}
            failed_attempts = Counter(j["name"] for j in jobs if j["status"] == "failed")
            for job in jobs:
                if job["status"] != "failed" or job.get("allow_failure"):
                    continue
                # A retry of this job that passed in the same pipeline settles it: a rerun was enough.
                label = "0" if job["name"] in passed else labels[p["id"]]
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
                                      "job": job["id"], "created_at": job["created_at"], "source": p["source"]}})
                stats[f"label_{label}"] += 1
                if len(rows) >= args.max_rows:
                    break
            if len(rows) >= args.max_rows:
                break
        print(f"{project['path_with_namespace']}: {len(rows)} rows so far", flush=True)
        if len(rows) >= args.max_rows:
            break

    rows.sort(key=lambda r: r["meta"]["created_at"])
    cut = int(len(rows) * (1 - args.test_fraction))
    for name, part in (("calibration", rows[:cut]), ("test", rows[cut:])):
        with (out / f"{name}.jsonl").open("w") as handle:
            for row in part:
                handle.write(json.dumps(row) + "\n")
    summary = {"since": args.since, "rows": len(rows), "calibration": cut, "test": len(rows) - cut,
               "claim": CLAIM, **stats}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
