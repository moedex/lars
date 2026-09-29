"""Labeled Shortcut work requests, for three routing decisions.

    SHORTCUT_API_TOKEN=... uv run python scripts/extract_shortcut_routing.py --since 2025-09-01

Decisions, each labeled by what people recorded on the story:

- `shortcut_product_area` (choice): the story's Product Area custom field.
- `shortcut_team` (choice): the team (Shortcut group) the story sits in.
- `shortcut_code_change` (noul): whether the story ended up with a linked merge request or
  branch. A story fixed through another story reads as "no", so this one is noisy.

Rows come from the Shortcut search API, one query per month (a search returns at most 1,000
stories), filtered by `--query` (default: work requests). The state is the story name plus its
description, with two kinds of lines removed:

- template lines that give the answer away or carry customer details, matched by their key
  (`Application:` restates the product area; `Category:`, account, domain, e-mail, ticket
  link and similar lines are customer or support-system details);
- anything left that looks like an e-mail address, a URL, a domain name that is not one of
  TurnCommerce's own brands, or a long number.

Comments are never read: they now carry AI triage summaries that would leak the answer.

Options with fewer than `--min-per-option` labeled stories are dropped (rows and option), since a
calibrator cannot be checked on them; the counts say how many. Stories are split by creation
time: the newest `--test-fraction` go to test.jsonl. Output goes under data/tc/shortcut/<decision>/,
which git ignores, and the script prints counts only. The token is read from SHORTCUT_API_TOKEN and
never printed; create one under Settings > API Tokens in Shortcut.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import httpx

API = "https://api.app.shortcut.com/api/v3"
TOKEN_ENV = "SHORTCUT_API_TOKEN"
PRODUCT_AREA_FIELD = "Product Area"
MAX_STATE_CHARS = 4000

# Template keys whose whole line is dropped, compared lower-case without spaces, dashes or underscores.
DROP_KEYS = {
    "application", "category", "account", "accountid", "accountname", "accountnumber", "impactedaccount",
    "customer", "customername", "customeremail", "email", "emailaddress", "user", "username", "login",
    "domain", "domains", "domainname", "impacteddomain", "impacteddomains", "ticket", "ticketlink",
    "supportticket", "supportticketlink", "zendesk", "freshdesk", "helpscout", "link", "url", "phone",
    "name", "contact", "reportedby", "requestedby", "requester", "orderid", "order", "invoice",
}
KEY_LINE = re.compile(r"^\s*[*_#>\-\s]*([A-Za-z][A-Za-z0-9 _/-]{0,40}?)[*_\s]*:")
OWN_BRANDS = {"namebright", "dropcatch", "hugedomains", "turncommerce", "sellerx", "wdn", "whois",
              "bettercontact", "premiumdomains", "tcdevops", "shortcut"}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
DOMAIN = re.compile(r"(?i)\b((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})\b")
LONG_NUMBER = re.compile(r"\b\d{6,}\b")
FILE_LIKE = re.compile(r"(?i)\.(cs|ts|js|py|json|cfm|sql|html|css|md|yml|yaml|xml|png|jpg|log|txt|dll|exe)$")


def _key(text: str) -> str:
    return re.sub(r"[\s_\-/]", "", text).lower()


def _domain(match: re.Match) -> str:
    host = match.group(1)
    labels = host.lower().split(".")
    if FILE_LIKE.search(host) or any(label in OWN_BRANDS for label in labels):
        return host
    if re.fullmatch(r"[\d.]+", host):  # version numbers like 1.2.3
        return host
    return "[DOMAIN]"


def clean_description(name: str, description: str) -> str:
    kept = []
    for line in (description or "").replace("\r\n", "\n").split("\n"):
        match = KEY_LINE.match(line)
        if match and _key(match.group(1)) in DROP_KEYS:
            continue
        kept.append(line.rstrip())
    text = f"{name.strip()}\n\n" + "\n".join(kept).strip()
    text = EMAIL.sub("[EMAIL]", text)
    text = URL.sub("[URL]", text)
    text = DOMAIN.sub(_domain, text)
    text = LONG_NUMBER.sub("[NUMBER]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:MAX_STATE_CHARS].strip()


def story_labels(story: dict, area_field_id: str | None, area_values: dict[str, str],
                 teams: dict[str, str]) -> dict[str, str | None]:
    area = None
    for field in story.get("custom_fields") or []:
        if field.get("field_id") == area_field_id:
            area = area_values.get(field.get("value_id")) or field.get("value")
    linked = bool(story.get("pull_requests") or story.get("branches") or story.get("merge_requests"))
    return {
        "shortcut_product_area": area,
        "shortcut_team": teams.get(story.get("group_id")) if story.get("group_id") else None,
        "shortcut_code_change": "1" if linked else "0",
    }


QUESTIONS = {
    "shortcut_product_area": "Which product area does this work request concern?",
    "shortcut_team": "Which team should own this work request?",
    "shortcut_code_change": "Resolving this work request will need a code change merged to a repository "
                            "(not only a data fix, configuration change, answer or duplicate).",
}


def build_rows(stories: list[dict], labeler, min_per_option: int) -> tuple[dict[str, list[dict]], dict]:
    """Rows per decision, with rare options dropped; stats are counts only."""
    stats: dict[str, Any] = {}
    rows: dict[str, list[dict]] = {}
    for decision, prompt in QUESTIONS.items():
        labeled = [(s, labeler(s)[decision]) for s in stories]
        labeled = [(s, lab) for s, lab in labeled if lab]
        counts = Counter(lab for _, lab in labeled)
        if decision == "shortcut_code_change":
            options = ["1", "0"]
            question = {"type": "noul", "instructions": prompt}
        else:
            options = sorted(o for o, n in counts.items() if n >= min_per_option and o.lower() != "unknown")
            question = {"type": "choice", "instructions": prompt, "criteria": {o: o for o in options}}
        kept = [(s, lab) for s, lab in labeled if lab in options]
        rows[decision] = [{"state": clean_description(s.get("name", ""), s.get("description", "")),
                           "question": question, "label": lab,
                           "meta": {"story": s["id"], "created_at": s["created_at"]}} for s, lab in kept]
        stats[decision] = {"labeled": len(labeled), "kept": len(kept), "options": len(options),
                           "dropped_rare_or_unknown": len(labeled) - len(kept)}
    return rows, stats


def split(rows: list[dict], test_fraction: float) -> tuple[list[dict], list[dict]]:
    rows = sorted(rows, key=lambda r: r["meta"]["created_at"])
    cut = int(len(rows) * (1 - test_fraction))
    return rows[:cut], rows[cut:]


class Shortcut:
    def __init__(self, token: str) -> None:
        self.http = httpx.Client(base_url=API, headers={"Shortcut-Token": token}, timeout=60)

    def get(self, path: str, **params) -> Any:
        response = self.http.get(path, params=params or None)
        for attempt in range(4):
            if response.status_code != 429 and response.status_code < 500:
                break
            time.sleep(2 ** attempt)
            response = self.http.get(path, params=params or None)
        response.raise_for_status()
        return response.json()

    def search(self, query: str) -> list[dict]:
        found, page = [], self.get("/search/stories", query=query, page_size=25, detail="full")
        while True:
            found.extend(page.get("data", []))
            nxt = page.get("next")
            if not nxt:
                return found
            page = self.get(nxt.removeprefix("/api/v3"))


def months(since: date, until: date):
    start = since.replace(day=1)
    while start <= until:
        end = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
        yield start, min(end, until)
        start = end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", required=True, help="ISO date: stories created on or after it")
    parser.add_argument("--until", default=None, help="ISO date (default today)")
    parser.add_argument("--query", default="label:WorkRequest", help="Shortcut search filter")
    parser.add_argument("--out", default="data/tc/shortcut")
    parser.add_argument("--min-per-option", type=int, default=15)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    args = parser.parse_args()
    token = os.environ.get(TOKEN_ENV)
    if not token:
        raise SystemExit(f"set {TOKEN_ENV} (Shortcut > Settings > API Tokens)")
    sc = Shortcut(token)

    fields = sc.get("/custom-fields")
    area = next((f for f in fields if f.get("name") == PRODUCT_AREA_FIELD), None)
    area_values = {v["id"]: v["value"] for v in (area or {}).get("values", [])}
    teams = {g["id"]: g["name"] for g in sc.get("/groups")}

    until = date.fromisoformat(args.until) if args.until else date.today()
    stories: dict[int, dict] = {}
    for start, end in months(date.fromisoformat(args.since), until):
        batch = sc.search(f"{args.query} created:{start.isoformat()}..{end.isoformat()}")
        stories.update({s["id"]: s for s in batch if not s.get("archived")})
        print(f"{start:%Y-%m}: {len(batch)} stories", flush=True)

    rows, stats = build_rows(list(stories.values()), lambda s: story_labels(s, area and area["id"], area_values,
                                                                            teams), args.min_per_option)
    summary: dict[str, Any] = {"since": args.since, "until": until.isoformat(), "query": args.query,
                               "stories": len(stories), "product_area_field_found": area is not None}
    for decision, decision_rows in rows.items():
        out = Path(args.out) / decision
        out.mkdir(parents=True, exist_ok=True)
        calibration, test = split(decision_rows, args.test_fraction)
        for name, part in (("calibration", calibration), ("test", test)):
            with (out / f"{name}.jsonl").open("w") as handle:
                for row in part:
                    handle.write(json.dumps(row) + "\n")
        summary[decision] = {**stats[decision], "calibration": len(calibration), "test": len(test),
                             "labels": dict(Counter(r["label"] for r in decision_rows).most_common())}
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
