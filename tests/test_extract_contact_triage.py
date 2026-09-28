"""scripts/extract_contact_triage.py: redaction, human-outcome labels and the time split, on synthetic data only."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extract_contact_triage import (  # noqa: E402
    build,
    haiku_label,
    outcome,
    read_export,
    redact,
    split_by_time,
    state_for,
)


def ticket(i, **fields):
    row = {"id": str(i), "submitted_at": f"2026-{(i % 9) + 1:02d}-10 12:00:00", "source": "form",
           "subject": "", "message": "Is example.com still for sale?", "domain_name": "", "status": "open",
           "closed_reason": "NULL", "closed_by_kind": "none", "responded": "0", "human_reply": "0",
           "ai_triage_classification": "NULL", "user_name": "", "email": "", "phone": "", "from_name": "NULL"}
    row.update(fields)
    return row


def test_redaction_removes_contact_details_and_the_rows_own_name():
    row = ticket(1, user_name="Priya Raman", email="p.raman77@ramanventures.io", phone="+1 (415) 555-0199",
                 domain_name="coolname.com")
    message = ("Hi, Priya Raman here from Ramanventures.io. Reach me at p.raman77@ramanventures.io, "
               "priya@other-mail.net or 415.555.0199, and my colleague at +44 20 7946 0958.\n"
               "Server 203.0.113.42 and fe80::1ff:fe23:4567:890a logged it at 10:30:00 on 2026-09-28.\n"
               "Tracking link https://track.example.net/c?uid=abc&email=x plain https://coolname.com/about\n"
               "Budget $1,250,000 for coolname.com.\nThanks, Priya")
    out = redact(message, row)
    for leaked in ("Priya", "Raman", "p.raman77", "ramanventures", "other-mail.net", "415", "7946",
                   "203.0.113.42", "fe80", "uid=abc"):
        assert leaked.lower() not in out.lower(), leaked
    for kept in ("10:30:00", "2026-09-28", "$1,250,000", "coolname.com", "https://coolname.com/about"):
        assert kept in out, kept
    assert redact("offer 1500000, order 12345678, cell 4155550199", {}) == "offer 1500000, order 12345678, cell [PHONE]"
    assert "[NAME]" in out and "[EMAIL]" in out and "[PHONE]" in out and "[IP]" in out and "[URL]" in out


def test_adjacent_phones_dates_and_glued_punctuation_are_redacted():
    # Third-party numbers: the row knows nothing about them, so the generic rule has to catch each one.
    assert redact("Tel 415-555-0199 212-555-0100", {}) == "Tel [PHONE] [PHONE]"
    assert redact("T 415 555 0199  415 555 0100", {}) == "T [PHONE]  [PHONE]"
    assert redact("Call 2026-09-28 415-555-0199 thanks", {}) == "Call 2026-09-28 [PHONE] thanks"
    assert redact("on 12/05/2026 call 01.23.45.67.89", {}) == "on 12/05/2026 call [PHONE]"
    assert redact("Tel.4155550199", {}) == "Tel.[PHONE]"
    assert redact("name,415-555-0199", {}) == "name,[PHONE]"
    for kept in ("Budget $1,250,000 or 1,250,000", "pi is 3.14159265", "at 10:30:00 on 2026-09-28."):
        assert redact(kept, {}) == kept


def test_obfuscated_addresses_are_redacted_but_prose_with_at_is_not():
    assert redact("reach me: john(at)acme.com", {}) == "reach me: [EMAIL]"
    assert redact("priya at ramanventures dot io", {}) == "[EMAIL]"
    assert redact("john [at] acme [dot] co.uk ok", {}) == "[EMAIL] ok"
    assert redact("john@acme dot com", {}) == "[EMAIL]"
    assert redact("Saw it for sale at coolname.com today", {}) == "Saw it for sale at coolname.com today"


def test_name_from_inbound_email_header_and_email_local_part():
    row = ticket(2, source="email", from_name='"Olumide Bakare"', email="obakare@gmail.com")
    out = state_for({**row, "subject": "Olumide re: offer", "message": "Regards,\nOlumide B.\nsent by obakare"})
    assert "Olumide" not in out and "obakare" not in out
    assert "gmail.com" not in out  # the address is gone even though freemail domains are not redacted alone
    assert out.startswith("source: email\n")


def test_role_mailboxes_do_not_blank_ordinary_words():
    row = ticket(3, email="info@acme-holdings.com")
    out = redact("Please send more info about the sales process at acme-holdings.com", row)
    assert "more info about the sales process" in out
    assert "acme-holdings.com" not in out  # the sender's company domain identifies them


def test_labels_come_from_human_actions_only():
    human_spam = ticket(1, status="spam", closed_reason="spam_confirmed", closed_by_kind="human")
    agent_spam = ticket(2, status="spam", closed_reason="spam", closed_by_kind="human",
                        ai_triage_classification="MISDIRECTED")
    replied = ticket(3, status="open", responded="1", ai_triage_classification="SPAM")
    replied_row = ticket(4, status="closed", closed_reason="replied", closed_by_kind="human")
    closed_quiet = ticket(5, status="closed", closed_reason="resolved", closed_by_kind="human")
    assert outcome(human_spam)[0] == "spam"
    assert outcome(agent_spam)[0] == "spam"
    assert outcome(replied)[0] == "reply"
    assert outcome(replied_row)[0] == "reply"
    assert outcome(closed_quiet)[0] == "no_reply"
    # Left out: nobody acted, or only the machine did.
    ai_spam = ticket(6, status="spam", ai_triage_classification="SPAM")
    auto_reply = ticket(7, status="closed", closed_reason="auto_reply", closed_by_kind="system")
    bulk = ticket(8, status="spam", closed_reason="spam_confirmed", closed_by_kind="bulk")
    untouched = ticket(9, ai_triage_classification="ESCALATE", is_high_priority="1", priority="urgent")
    for row in (ai_spam, auto_reply, bulk, untouched):
        assert outcome(row)[0] is None, row["id"]


def test_model_output_columns_never_change_a_label():
    base = [ticket(1, status="spam", closed_reason="spam_confirmed", closed_by_kind="human"),
            ticket(2, responded="1"),
            ticket(3, status="closed", closed_reason="resolved", closed_by_kind="human"),
            ticket(4, status="spam"), ticket(5)]
    for classification in ("SPAM", "MISDIRECTED", "SIMPLE", "COMPLEX", "ESCALATE", "NULL"):
        variant = [{**r, "ai_triage_classification": classification, "is_high_priority": "1", "priority": "urgent",
                    "has_ai_response": "1", "priority_set_by": "Neo (AI Triage)", "reply_count": "4"} for r in base]
        decisions, _ = build(variant)
        assert [r["label"] for r in decisions["contact_handling"]] == ["spam", "reply", "no_reply"]
        assert [r["label"] for r in decisions["contact_spam"]] == ["1", "0"]
        for r in decisions["contact_handling"]:
            expected = None if classification == "NULL" else classification
            assert r["meta"]["ai_triage_classification"] == expected
    # status='spam' is what Haiku's SPAM branch writes: on its own it is not a label.
    decisions, stats = build([ticket(1, status="spam", ai_triage_classification="SPAM")])
    assert decisions["contact_handling"] == [] and stats["outcome_spam_unconfirmed"] == 1


def test_contact_columns_are_dropped_and_haiku_kept_in_meta():
    row = ticket(1, status="closed", closed_reason="replied", closed_by_kind="human", user_name="Zed Quux",
                 email="zed@quux.dev", phone="5550001111", ai_triage_classification="COMPLEX",
                 message="Zed here, call 5550001111")
    decisions, _ = build([row])
    out = decisions["contact_handling"][0]
    blob = json.dumps(out)
    assert "Zed" not in blob and "quux" not in blob and "5550001111" not in blob
    assert set(out) == {"state", "question", "label", "meta"}
    assert out["meta"]["ai_triage_classification"] == "COMPLEX" and out["meta"]["haiku_label"] == "reply"
    assert haiku_label("contact_spam", "SPAM") == "1" and haiku_label("contact_spam", "MISDIRECTED") == "0"
    assert haiku_label("contact_handling", "") is None


def test_resolved_without_reply_is_not_a_spam_label_unless_asked():
    quiet = ticket(1, status="closed", closed_reason="resolved", closed_by_kind="human")
    assert build([quiet])[0]["contact_spam"] == []
    assert [r["label"] for r in build([quiet], resolved_not_spam=True)[0]["contact_spam"]] == ["0"]


def test_time_split_puts_the_newest_rows_in_test():
    rows = [{"label": "spam" if i < 5 else "reply", "meta": {"submitted_at": f"2026-01-{i + 1:02d}", "id": str(i)}}
            for i in range(10)]
    calibration, test = split_by_time(rows, 0.3, per_label=False)
    assert [r["meta"]["id"] for r in test] == ["7", "8", "9"]
    assert max(r["meta"]["submitted_at"] for r in calibration) < min(r["meta"]["submitted_at"] for r in test)
    assert {r["label"] for r in test} == {"reply"}  # all spam is old: the plain split has none in test
    calibration, test = split_by_time(rows, 0.4, per_label=True)
    assert {r["label"] for r in test} == {"spam", "reply"}
    for label in ("spam", "reply"):
        newest_cal = max(r["meta"]["submitted_at"] for r in calibration if r["label"] == label)
        assert newest_cal < min(r["meta"]["submitted_at"] for r in test if r["label"] == label)


def test_duplicate_ids_across_live_and_archive_count_once():
    row = ticket(1, status="closed", closed_reason="replied", closed_by_kind="human")
    decisions, stats = build([row, dict(row)])
    assert len(decisions["contact_handling"]) == 1 and stats["duplicate_id"] == 1


def test_duplicates_keep_the_copy_with_the_human_outcome():
    live = ticket(1, from_table="live")  # stale: still open in the live table
    archived = ticket(1, from_table="archive", status="closed", closed_reason="replied", closed_by_kind="human")
    for rows in ([live, archived], [archived, live]):
        decisions, stats = build(rows)
        assert [r["label"] for r in decisions["contact_handling"]] == ["reply"] and stats["duplicate_id"] == 1
    # Both copies labelled: the archive holds the later state.
    live_quiet = ticket(2, from_table="live", status="closed", closed_reason="resolved", closed_by_kind="human")
    archive_spam = ticket(2, from_table="archive", status="spam", closed_reason="spam_confirmed",
                          closed_by_kind="human")
    decisions, stats = build([live_quiet, archive_spam])
    assert [r["label"] for r in decisions["contact_handling"]] == ["spam"]
    assert stats["duplicate_id_outcome_conflict"] == 1


def test_missing_ids_are_refused_and_header_case_is_folded():
    with pytest.raises(ValueError, match="2 of 2 rows have no id"):
        build([{**ticket(1), "id": ""}, {k: v for k, v in ticket(2).items() if k != "id"}])
    upper = {("ID" if k == "id" else k.upper()): v for k, v in ticket(3, responded="1").items()}
    decisions, _ = build([upper])
    assert decisions["contact_handling"][0]["meta"]["id"] == "3"


def test_reads_mysql_batch_tsv_and_csv(tmp_path):
    header = ["id", "submitted_at", "message", "closed_reason", "closed_by_kind", "status"]
    tsv = tmp_path / "x.tsv"
    tsv.write_text("\t".join(header) + "\n" + "\t".join(["7", "2026-03-01", "line one\\nline\\ttwo", "NULL", "none",
                                                        "open"]) + "\n")
    row = read_export(tsv)[0]
    assert row["message"] == "line one\nline\ttwo" and row["closed_reason"] == "NULL"
    with (tmp_path / "x.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(["8", "2026-03-02", "multi\nline, with comma", "", "none", "open"])
    assert read_export(tmp_path / "x.csv")[0]["message"] == "multi\nline, with comma"


def test_tsv_keeps_a_crlf_message_in_one_row(tmp_path):
    # mysql --batch escapes \n but not \r, so a CRLF textarea arrives as "line one\r\\nline two".
    header = ["id", "submitted_at", "source", "subject", "message", "domain_name", "status", "closed_reason",
              "closed_by_kind", "responded"]
    cells = ["7", "2026-03-01", "form", "Hi", "line one\r\\nline two", "x.com", "closed", "replied", "human", "1"]
    tsv = tmp_path / "x.tsv"
    tsv.write_bytes(("\t".join(header) + "\n" + "\t".join(cells) + "\n").encode())
    rows = read_export(tsv)
    assert len(rows) == 1 and rows[0]["message"] == "line one\r\nline two" and rows[0]["responded"] == "1"
    assert [r["label"] for r in build(rows)[0]["contact_handling"]] == ["reply"]
    assert "line one\nline two" in state_for(rows[0])
    # A file re-saved with CRLF line ends: \r\n is then the row break.
    tsv.write_bytes(("\t".join(header) + "\r\n" + "\t".join(cells) + "\r\n").encode())
    rows = read_export(tsv)
    assert len(rows) == 1 and rows[0]["message"] == "line one\r\nline two" and rows[0]["responded"] == "1"


def test_rows_with_the_wrong_cell_count_are_refused_without_their_text(tmp_path):
    tsv = tmp_path / "x.tsv"
    tsv.write_text("id\tmessage\n1\tsecret words\tstray\n")
    with pytest.raises(ValueError) as err:
        read_export(tsv)
    assert "line 2" in str(err.value) and "secret" not in str(err.value)
    bad_csv = tmp_path / "x.csv"
    bad_csv.write_text("id,message\n1,secret,stray\n")
    with pytest.raises(ValueError) as err:
        read_export(bad_csv)
    assert "secret" not in str(err.value)


def test_cli_writes_splits_and_prints_counts_only(tmp_path):
    rows = ([ticket(i, status="spam", closed_reason="spam_confirmed", closed_by_kind="human",
                    message=f"SEO offer {i}", email="secret.sender@spammy.biz") for i in range(1, 5)]
            + [ticket(i, responded="1", ai_triage_classification="SIMPLE", user_name="Hidden Person")
               for i in range(5, 9)])
    export = tmp_path / "export.jsonl"
    export.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_contact_triage.py"), "--export",
                             str(export), "--out", str(out), "--per-label"], capture_output=True, text=True, check=True)
    assert "SEO offer" not in result.stdout and "Hidden" not in result.stdout and "spammy" not in result.stdout
    summary = json.loads((out / "summary.json").read_text())
    assert summary["rows_read"] == 8
    assert summary["contact_handling"]["labels_missing"] == {"calibration": [], "test": []}
    lines = (out / "contact_handling" / "calibration.jsonl").read_text().splitlines()
    lines += (out / "contact_handling" / "test.jsonl").read_text().splitlines()
    assert len(lines) == 8 and all("spammy" not in line for line in lines)
    sql = subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_contact_triage.py"), "--print-sql"],
                         capture_output=True, text=True, check=True).stdout
    assert "contact_submissions_archive" in sql and "closed_by_name LIKE" in sql
