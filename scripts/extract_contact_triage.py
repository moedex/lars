"""Labeled Premium Domains contact tickets, for the `contact_handling` choice and the `contact_spam` noul.

    uv run python scripts/extract_contact_triage.py --print-sql > export.sql   # the query to run yourself
    uv run python scripts/extract_contact_triage.py --export contacts.tsv      # CSV, TSV or JSONL
    MOELARS_TC_CONTACT_DSN=mysql://... uv run --with pymysql python scripts/extract_contact_triage.py --from-db

Labels come only from what a person did with the ticket, never from the AI pipeline's own output:

- `ai_triage_classification` is Haiku's Tier 1 output (admin/scheduled-tasks/tasks/ai-crm-processor.cfm).
  Its SPAM branch also writes `status='spam'`, and its ESCALATE branch writes `is_high_priority=1` with
  `priority_set_by='Neo (AI Triage)'`. So `status='spam'` alone, `is_high_priority` and the classification
  are model output. Training on them would teach moe-LARS to imitate Haiku, so they go in `meta` only,
  where they let Haiku be scored against the same human labels.
- `priority` is 'normal' on every row (the form inserts it and nothing in the CRM changes it): no signal.
- Human outcomes: an agent confirmed or marked spam (`closed_reason` 'spam_confirmed' via "Agree, Hide",
  or 'spam' via the spam button, with a person as closer); an agent sent a reply (`responded=1`, set by
  sendContactReply, or a human 'sent' row in contact_replies, or `closed_reason='replied'`); an agent
  closed it without replying (`closed_reason` resolved / no_response_needed / duplicate). `reply_count`
  is not a reply signal: inbound thread follow-ups from the customer bump it too.
- Left out: open tickets nobody has acted on, AI spam nobody confirmed, and tickets closed by the system
  (auto_reply bounces, auto_rule inbox rules) or by a bulk cleanup script.

`--export` reads a file the user produced with the SQL from `--print-sql` (a mysql client's TSV via
`mysql --batch`, a CSV from a GUI client, or JSONL). `--from-db` runs the same SQL itself, reading the DSN
from MOELARS_TC_CONTACT_DSN; the DSN is never printed. The SQL buckets `closed_by_name` into human / system
/ ai / bulk inside the database, so staff names never leave it. A ticket caught mid-archive is in both
tables; the copy with a human outcome wins, then the archive copy (`from_table`). A row without an id, or a
TSV/CSV row whose cell count differs from the header, stops the run with a count and line number only.

State is the subject and message (plus the source and the domain asked about), with PII redacted: e-mail
addresses, phone numbers, IP addresses, URLs with query strings, and the row's own contact name, e-mail
and phone wherever they appear. Contact columns are used for that and then dropped. Rows are split by
submission time: the newest `--test-fraction` go to test. Human spam confirmations stop in May (older
unconfirmed spam is deleted after 30 days by crm-archiver), so a plain time split can leave spam out of
test; `--per-label` takes the newest fraction of each label instead, and summary.json flags a label
missing from either split. Output goes under data/, which git ignores; only counts are printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

HANDLING_QUESTION = {
    "type": "choice",
    "instructions": "How should the support team of a premium domain name marketplace handle this incoming message?",
    "criteria": {
        "spam": "Junk nobody should answer: unsolicited marketing, SEO or link-building offers, scams, phishing, "
                "bot or template text.",
        "reply": "A real person who needs an answer from the team: a question about buying or pricing a domain, "
                 "a payment plan, a transfer, an order or their account.",
        "no_reply": "Genuine but needs no answer: mail meant for a domain's previous owner, a thank-you or "
                    "acknowledgement, an automated notice, a duplicate of another ticket.",
    },
}
SPAM_CLAIM = ("This message is spam or junk (unsolicited marketing, SEO or link-building offers, scams, phishing, "
              "bot text), not a real person writing about a domain, a purchase or their account.")
SPAM_QUESTION = {"type": "noul", "instructions": SPAM_CLAIM}
DECISIONS = {"contact_handling": HANDLING_QUESTION, "contact_spam": SPAM_QUESTION}

# Haiku's classes as the action the pipeline takes on them: MISDIRECTED and SIMPLE get a Sonnet draft,
# COMPLEX an Opus draft, ESCALATE is left for a person to answer, SPAM is closed.
HAIKU_HANDLING = {"SPAM": "spam", "MISDIRECTED": "reply", "SIMPLE": "reply", "COMPLEX": "reply", "ESCALATE": "reply"}

SPAM_REASONS = {"spam", "spam_confirmed"}
NO_REPLY_REASONS = {"resolved", "no_response_needed", "no response needed", "duplicate"}
MESSAGE_CHARS = 3000
DSN_ENV = "MOELARS_TC_CONTACT_DSN"

_SELECT = """SELECT s.id, '{kind}' AS from_table, s.submitted_at, s.source, s.subject, s.message, s.domain_name,
  s.status, s.closed_reason,
  CASE WHEN s.closed_by_name IS NULL OR s.closed_by_name = '' THEN 'none'
       WHEN s.closed_by_name LIKE 'System%' THEN 'system'
       WHEN s.closed_by_name LIKE 'Neo%' THEN 'ai'
       WHEN s.closed_by_name LIKE 'Bulk%' THEN 'bulk'
       ELSE 'human' END AS closed_by_kind,
  s.responded,
  EXISTS (SELECT 1 FROM premium_domains.contact_replies r WHERE r.submission_id = s.id AND r.sent_by_type = 'human'
          AND r.status = 'sent' AND r.reply_type IN ('reply', 'reply_all')) AS human_reply,
  s.ai_triage_classification, s.user_name, s.email, s.phone,
  COALESCE(ie.from_name, iea.from_name) AS from_name
FROM premium_domains.{table} s
LEFT JOIN premium_domains.contact_submissions_inbound_emails ie ON ie.id = s.inbound_email_id
LEFT JOIN premium_domains.contact_submissions_inbound_emails_archive iea ON iea.id = s.inbound_email_id"""
# Closed tickets move to the archive after 90 days and confirmed spam at once, so most human labels live there.
EXPORT_SQL = (_SELECT.format(table="contact_submissions", kind="live") + "\nUNION ALL\n"
              + _SELECT.format(table="contact_submissions_archive", kind="archive") + "\nORDER BY submitted_at;")

URL_WITH_QUERY = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']*\?[^\s<>\"']*")
EMAIL = re.compile(r"(?i)(?:mailto:)?[\w.+%-]+@[\w-]+(?:\.[\w-]+)+")
# Addresses written to dodge scrapers: "john(at)acme.com", "john [at] acme [dot] com", "priya at raman dot io",
# "john@acme dot com". A bracketed "at" is never prose, so any dot form may follow it. A bare " at " is
# (a sale at coolname.com), so it needs at least one spelled-out "dot" before it counts as an address.
_LABEL = r"[\w-]+"
_DOT_WORD = r"(?:\s*[(\[{<]\s*dot\s*[)\]}>]\s*|\s+dot\s+)"
_DOT_ANY = rf"(?:{_DOT_WORD}|\.)"
EMAIL_OBFUSCATED = re.compile(
    rf"(?i)(?<![\w.+%-])[\w.+%-]+(?:"
    rf"\s*[(\[{{<]\s*at\s*[)\]}}>]\s*{_LABEL}(?:{_DOT_ANY}{_LABEL})+"
    rf"|(?:\s+at\s+|\s*@\s*){_LABEL}(?:\.{_LABEL})*(?:{_DOT_WORD}{_LABEL}(?:\.{_LABEL})*)+"
    rf")(?![\w-])")
# Not part of a longer dotted run, so a phone written 01.23.45.67.89 is left for PHONE.
IPV4 = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d|\.\d)")
# Three or more colons keeps times like 10:30:00 out.
IPV6 = re.compile(r"(?i)(?<![\w:])(?:[0-9a-f]{0,4}:){3,7}[0-9a-f]{0,4}(?![\w:])")
# Not after a letter-or-digit, a currency sign, or a digit plus [.,] (the middle of 1,250,000 or 3.1415926);
# "Tel.4155550199" and "name,415-555-0199" still start a match.
PHONE = re.compile(r"(?<![\w$€£])(?<!\d[.,])\+?\(?\d[\d ().-]{5,}\d(?!\w|,\d)")
# Dates are cut out before PHONE runs, so "2026-09-28 415-555-0199" cannot merge into one match that reads
# as a date. The guards keep a dotted phone (01.23.45.67.89) from passing for a date.
DATE = re.compile(r"(?<!\d)(?<!\d[-./])(\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./](?:\d{4}|\d{2}))"
                  r"(?!\d|[-./]\d)")
# Mailbox local parts that are roles, not names: redacting them would blank ordinary words.
ROLE_WORDS = {"info", "sales", "support", "admin", "contact", "hello", "office", "mail", "team", "help", "service",
              "domains", "domain", "noreply", "reply", "billing", "accounts", "webmaster", "the", "and", "mr", "mrs"}
FREEMAIL = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "icloud.com", "live.com", "msn.com",
            "protonmail.com", "proton.me", "gmx.com", "mail.com", "yandex.com", "qq.com", "163.com", "me.com"}


def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text in {"NULL", "\\N"} else text


def _one_phone(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    if not 7 <= len(digits) <= 15:
        return text
    if re.fullmatch(r"\d+[.,]\d+", text):
        return text  # one decimal point: 3.14159265 or a measurement, not a phone
    if text.isdigit() and len(digits) < 10:
        return text  # a bare 1500000 is a price or an order number far more often than a local phone number
    return "[PHONE]"


def _phone_like(match: re.Match) -> str:
    """PHONE spans spaces, so two numbers side by side ("415-555-0199 212-555-0100", a signature block)
    come out as one match of more than 15 digits, which no single phone has. Cut such a run at spaces
    into pieces of at most 15 digits and judge each; a double space always ends a piece."""
    text = match.group(0)
    if len(re.sub(r"\D", "", text)) <= 15:
        return _one_phone(text)
    out, piece, count = [], "", 0
    for token in re.split(r"(\s+)", text):
        if not token.strip():
            if len(token) > 1:
                out += [_one_phone(piece), token]
                piece, count = "", 0
            elif piece:
                piece += token
            else:
                out.append(token)
            continue
        n = len(re.sub(r"\D", "", token))
        if piece and count + n > 15:
            stripped = piece.rstrip()
            out += [_one_phone(stripped), piece[len(stripped):]]
            piece, count = "", 0
        piece += token
        count += n
    out.append(_one_phone(piece))
    return "".join(out)


def _redact_phones(text: str) -> str:
    parts = DATE.split(text)  # the capture group puts each date at an odd index
    return "".join(part if i % 2 else PHONE.sub(_phone_like, part) for i, part in enumerate(parts))


def _own_patterns(row: dict) -> list[tuple[re.Pattern, str]]:
    """Patterns for this row's own contact details, which the generic rules can miss: a name typed
    in the sign-off, the e-mail's company domain, a phone number written with odd separators."""
    patterns: list[tuple[re.Pattern, str]] = []
    for key in ("email", "from_email"):
        address = _clean(row.get(key)).strip().lower()
        if "@" not in address:
            continue
        patterns.append((re.compile(re.escape(address), re.I), "[EMAIL]"))
        local, _, domain = address.partition("@")
        if domain and domain not in FREEMAIL and domain != _clean(row.get("domain_name")).strip().lower():
            patterns.append((re.compile(r"(?<![\w.-])" + re.escape(domain) + r"(?![\w-])", re.I), "[SENDER_DOMAIN]"))
    names: list[str] = []
    for key in ("user_name", "from_name", "name", "first_name", "last_name"):
        full = _clean(row.get(key)).strip().strip("\"'")
        if full:
            names.append(full)
    for key in ("email", "from_email"):
        local = _clean(row.get(key)).split("@")[0]
        names.extend(re.split(r"[._+\-\d]+", local))
    tokens = {t for n in names for t in re.split(r"[\s,]+", n) if len(t) >= 3 and t.lower() not in ROLE_WORDS}
    # Full names first, so "Jane Doe" becomes one [NAME] instead of two.
    for name in sorted({n for n in names if " " in n.strip()} | tokens, key=len, reverse=True):
        patterns.append((re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I), "[NAME]"))
    digits = re.sub(r"\D", "", _clean(row.get("phone")))
    if len(digits) >= 7:
        patterns.append((re.compile(r"[ ().-]*".join(re.escape(d) for d in digits[-7:])), "[PHONE]"))
    return patterns


def redact(text: str, row: dict) -> str:
    # Generic rules first: a name swapped out of an address would leave "[NAME]@host" for EMAIL to miss.
    text = URL_WITH_QUERY.sub("[URL]", text)
    text = EMAIL.sub("[EMAIL]", text)
    text = EMAIL_OBFUSCATED.sub("[EMAIL]", text)
    text = IPV4.sub("[IP]", text)
    text = IPV6.sub(lambda m: "[IP]" if re.search(r"[0-9a-f]", m.group(0), re.I) else m.group(0), text)
    text = _redact_phones(text)
    for pattern, replacement in _own_patterns(row):
        text = pattern.sub(replacement, text)
    return text


def state_for(row: dict) -> str:
    subject = redact(_clean(row.get("subject")).strip(), row)
    message = redact(_clean(row.get("message")).replace("\r\n", "\n").strip()[:MESSAGE_CHARS], row)
    lines = [f"source: {_clean(row.get('source')) or 'unknown'}"]
    domain = _clean(row.get("domain_name")).strip()
    if domain:
        lines.append(f"domain asked about: {domain}")
    if subject:
        lines.append(f"subject: {subject}")
    lines.append(f"message:\n{message}")
    return "\n".join(lines)


def _flag(value) -> bool:
    return _clean(value).strip().lower() in {"1", "true", "yes", "t"}


def closed_by_kind(row: dict) -> str:
    """The SQL export computes this; a raw export with closed_by_name gets the same buckets here."""
    kind = _clean(row.get("closed_by_kind")).strip().lower()
    if kind:
        return kind
    name = _clean(row.get("closed_by_name")).strip()
    if not name:
        return "none"
    for prefix, bucket in (("system", "system"), ("neo", "ai"), ("bulk", "bulk")):
        if name.lower().startswith(prefix):
            return bucket
    return "human"


def outcome(row: dict) -> tuple[str | None, str]:
    """(handling label or None, why). Reads only what a person did; the AI columns are never consulted."""
    reason = _clean(row.get("closed_reason")).strip().lower()
    kind = closed_by_kind(row)
    if kind in {"system", "bulk", "ai"}:
        return None, f"closed_by_{kind}"
    if reason in SPAM_REASONS and kind == "human":
        return "spam", "human_spam"
    if _flag(row.get("responded")) or _flag(row.get("human_reply")) or reason == "replied":
        return "reply", "human_reply"
    if kind == "human" and reason in NO_REPLY_REASONS:
        return "no_reply", "human_closed_without_reply"
    status = _clean(row.get("status")).strip().lower()
    if status == "spam":
        return None, "spam_unconfirmed"
    if status == "open":
        return None, "open_no_outcome"
    return None, f"other_{reason or 'no_reason'}"


def labels_for(row: dict, resolved_not_spam: bool) -> dict[str, str]:
    handling, why = outcome(row)
    if handling is None:
        return {}
    labels = {"contact_handling": handling}
    if handling == "spam":
        labels["contact_spam"] = "1"
    elif handling == "reply" or resolved_not_spam:
        # A reply is a person treating it as real; closing without one is weaker (spam had its own button).
        labels["contact_spam"] = "0"
    return labels


def haiku_label(decision: str, classification: str) -> str | None:
    if not classification:
        return None
    mapped = HAIKU_HANDLING.get(classification.upper())
    if decision == "contact_spam" and mapped:
        return "1" if mapped == "spam" else "0"
    return mapped


def read_export(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
    # Bytes, not read_text: universal newlines would turn the bare \r that `mysql --batch` leaves in a CRLF
    # form message into a row break, and the row would come back as two rows of shifted columns.
    text = path.read_bytes().decode("utf-8-sig")
    first = text.split("\n", 1)[0]
    if path.suffix == ".tsv" or ("\t" in first and "," not in first):
        # `mysql --batch` writes one line per row with \n, \t, \\ and \0 escaped (not \r), and NULL as NULL.
        # A file saved with CRLF line ends has \r on the header too; only then is \r\n the row break.
        lines = text.rstrip("\r\n").split("\r\n" if first.endswith("\r") else "\n")
        header = lines[0].split("\t")
        unescape = {"n": "\n", "t": "\t", "\\": "\\", "0": "\0", "r": "\r"}
        rows = []
        for number, line in enumerate(lines[1:], start=2):
            cells = line.split("\t")
            if len(cells) != len(header):  # the line number and counts only: never the row's text
                raise ValueError(f"{path.name} line {number}: {len(cells)} cells, header has {len(header)}")
            rows.append(dict(zip(header, [re.sub(r"\\(.)", lambda m: unescape.get(m.group(1), m.group(1)), c)
                                          for c in cells], strict=True)))
        return rows
    with path.open(newline="", encoding="utf-8-sig") as handle:  # newline="": csv handles \r inside quotes
        rows = list(csv.DictReader(handle))
    for number, row in enumerate(rows, start=2):
        if None in row or None in row.values():
            raise ValueError(f"{path.name} record {number}: cell count differs from the header")
    return rows


def read_db() -> list[dict]:
    dsn = os.environ.get(DSN_ENV)
    if not dsn:
        raise SystemExit(f"set {DSN_ENV} (mysql://user:password@host:port/premium_domains)")
    try:
        import pymysql
        import pymysql.cursors
    except ImportError as err:
        raise SystemExit("--from-db needs pymysql: uv run --with pymysql python scripts/...") from err
    url = urlparse(dsn)
    connection = pymysql.connect(host=url.hostname, port=url.port or 3306, user=unquote(url.username or ""),
                                 password=unquote(url.password or ""), database=url.path.lstrip("/") or None,
                                 charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor, read_timeout=120)
    try:
        with connection.cursor() as cursor:
            cursor.execute(EXPORT_SQL.rstrip(";"))
            return list(cursor.fetchall())
    finally:
        connection.close()


def split_by_time(rows: list[dict], test_fraction: float, per_label: bool) -> tuple[list[dict], list[dict]]:
    """Newest `test_fraction` to test, so calibration never sees the future. `per_label` cuts each label
    separately, for a label whose rows all sit in one stretch of time."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["label"] if per_label else "", []).append(row)
    calibration, test = [], []
    for part in groups.values():
        part.sort(key=lambda r: (r["meta"]["submitted_at"], r["meta"]["id"]))
        cut = int(round(len(part) * (1 - test_fraction)))
        calibration += part[:cut]
        test += part[cut:]
    key = lambda r: (r["meta"]["submitted_at"], r["meta"]["id"])  # noqa: E731
    return sorted(calibration, key=key), sorted(test, key=key)


def _copy_rank(row: dict) -> tuple[bool, bool]:
    """Which copy of a ticket found in both tables to keep: one with a human outcome first (a live copy
    can still be open while the archived one was answered), then the archive, which is the later state."""
    return outcome(row)[0] is not None, _clean(row.get("from_table")).strip().lower() == "archive"


def dedupe(raw_rows: list[dict], stats: Counter) -> list[dict]:
    """One row per ticket id, in first-seen order. Header case is folded (a GUI export may say "ID").
    An empty id is refused: every such row would otherwise collapse into the first as a duplicate."""
    rows = [{str(k).strip().lower(): v for k, v in raw.items()} for raw in raw_rows]
    missing = sum(not _clean(r.get("id")).strip() for r in rows)
    if missing:
        raise ValueError(f"{missing} of {len(rows)} rows have no id; export the id column (see --print-sql)")
    best: dict[str, dict] = {}
    for row in rows:
        ident = _clean(row["id"]).strip()
        if ident not in best:
            best[ident] = row
            continue
        stats["duplicate_id"] += 1  # a ticket caught mid-archive shows up in both tables
        kept = best[ident]
        if None not in (outcome(kept)[0], outcome(row)[0]) and outcome(kept)[0] != outcome(row)[0]:
            stats["duplicate_id_outcome_conflict"] += 1
        if _copy_rank(row) > _copy_rank(kept):
            best[ident] = row
    return list(best.values())


def build(raw_rows: list[dict], resolved_not_spam: bool = False) -> tuple[dict[str, list[dict]], Counter]:
    """Rows per decision, plus counts of what was kept and why the rest was left out."""
    stats: Counter = Counter()
    out: dict[str, list[dict]] = {name: [] for name in DECISIONS}
    for raw in dedupe(raw_rows, stats):
        ident = _clean(raw["id"]).strip()
        stats["rows_read"] += 1
        handling, why = outcome(raw)
        stats[f"outcome_{why}"] += 1
        labels = labels_for(raw, resolved_not_spam)
        if not labels:
            continue
        state = state_for(raw)
        classification = _clean(raw.get("ai_triage_classification")).strip().upper() or None
        for decision, label in labels.items():
            meta = {"id": ident, "submitted_at": _clean(raw.get("submitted_at")),
                    "source": _clean(raw.get("source")), "ai_triage_classification": classification,
                    "haiku_label": haiku_label(decision, classification or "")}
            out[decision].append({"state": state, "question": DECISIONS[decision], "label": label, "meta": meta})
    return out, stats


def split_summary(calibration: list[dict], test: list[dict]) -> dict:
    """Counts only: labels per split, and Haiku's agreement with the human label where it classified."""
    summary: dict = {}
    for name, part in (("calibration", calibration), ("test", test)):
        labels = Counter(r["label"] for r in part)
        scored = [r for r in part if r["meta"]["haiku_label"] is not None]
        summary[name] = {"rows": len(part), "labels": dict(sorted(labels.items())),
                         "haiku_scored": len(scored),
                         "haiku_agree": sum(r["meta"]["haiku_label"] == r["label"] for r in scored),
                         "haiku_confusion": dict(sorted(Counter(f"{r['label']}<-{r['meta']['haiku_label']}"
                                                                for r in scored).items()))}
    every = set(summary["calibration"]["labels"]) | set(summary["test"]["labels"])
    summary["labels_missing"] = {name: sorted(every - set(summary[name]["labels"])) for name in ("calibration", "test")}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--export", type=Path, help="CSV, TSV (mysql --batch) or JSONL produced with --print-sql")
    source.add_argument("--from-db", action="store_true", help=f"run the export SQL against ${DSN_ENV}")
    source.add_argument("--print-sql", action="store_true", help="print the export SQL and exit")
    parser.add_argument("--out", type=Path, default=Path("data/tc/contact_triage"))
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--per-label", action="store_true", help="time-split each label separately")
    parser.add_argument("--resolved-not-spam", action="store_true",
                        help='label tickets a person closed without replying "0" for contact_spam too')
    args = parser.parse_args()
    if args.print_sql:
        print(EXPORT_SQL)
        return 0
    try:  # these errors carry counts and line numbers only, so they are safe to print
        raw_rows = read_export(args.export) if args.export else read_db()
        decisions, stats = build(raw_rows, args.resolved_not_spam)
    except ValueError as err:
        raise SystemExit(f"error: {err}") from None
    summary: dict = {"test_fraction": args.test_fraction, "per_label": args.per_label,
                     "resolved_not_spam": args.resolved_not_spam, **dict(sorted(stats.items()))}
    for decision, rows in decisions.items():
        calibration, test = split_by_time(rows, args.test_fraction, args.per_label)
        folder = args.out / decision
        folder.mkdir(parents=True, exist_ok=True)
        for name, part in (("calibration", calibration), ("test", test)):
            with (folder / f"{name}.jsonl").open("w") as handle:
                for row in part:
                    handle.write(json.dumps(row) + "\n")
        summary[decision] = split_summary(calibration, test)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
