"""scripts/extract_shortcut_routing.py: redaction, labels and rare-option handling on synthetic stories."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from extract_shortcut_routing import build_rows, clean_description, split, story_labels  # noqa: E402

DESCRIPTION = """**Impacted Account:** 4481234 (Jane Roe)
**Impacted Domain:** janesbakery.com
Application: NB2
Category: Account Billing
Support Ticket Link: https://support.example.com/tickets/99812?x=1
**Steps**
1. Log in as jane.roe@gmail.com and open billing
2. Charge fails for order 55512345; see PaymentService.cs and namebright.com/account
Actual: error 500. Expected: charge goes through. Version 2.14.3."""


def test_clean_description_drops_template_lines_and_redacts():
    out = clean_description("Card charge fails on renewal", DESCRIPTION)
    for leaked in ("4481234", "Jane Roe", "janesbakery", "NB2", "Account Billing", "support.example.com",
                   "jane.roe", "55512345"):
        assert leaked not in out, leaked
    assert out.startswith("Card charge fails on renewal")
    assert "PaymentService.cs" in out and "namebright.com" in out and "2.14.3" in out  # own brand, file, version
    assert "[EMAIL]" in out and "[NUMBER]" in out


AREA_FIELD, AREA_NB2, AREA_DNS = "f-area", "v-nb2", "v-dns"


def _story(i, area, team, linked, created):
    return {"id": i, "name": f"story {i}", "description": "Steps: it breaks", "created_at": created,
            "group_id": team, "custom_fields": [{"field_id": AREA_FIELD, "value_id": area}] if area else [],
            "pull_requests": [{"id": 1}] if linked else []}


def test_labels_rare_options_and_time_split():
    values = {AREA_NB2: "NB2", AREA_DNS: "Dns", "v-unk": "Unknown"}
    teams = {"g-bc": "BaseCamp", "g-inf": "Infrastructure"}
    stories = [_story(i, AREA_NB2, "g-bc", i % 3 == 0, f"2026-0{1 + i % 9}-01") for i in range(20)]
    stories += [_story(100 + i, AREA_DNS, "g-inf", False, "2026-05-01") for i in range(3)]   # rare area
    stories += [_story(200, "v-unk", None, False, "2026-05-02"), _story(201, None, "g-bc", True, "2026-05-03")]
    labeler = lambda s: story_labels(s, AREA_FIELD, values, teams)  # noqa: E731
    assert labeler(stories[0]) == {"shortcut_product_area": "NB2", "shortcut_team": "BaseCamp",
                                   "shortcut_code_change": "1"}
    rows, stats = build_rows(stories, labeler, min_per_option=5)
    area = rows["shortcut_product_area"]
    assert {r["label"] for r in area} == {"NB2"} and area[0]["question"]["criteria"] == {"NB2": "NB2"}
    assert stats["shortcut_product_area"] == {"labeled": 24, "kept": 20, "options": 1, "dropped_rare_or_unknown": 4}
    assert stats["shortcut_team"]["options"] == 1  # Infrastructure has only 3 stories
    assert len(rows["shortcut_code_change"]) == len(stories)
    calibration, test = split(area, 0.3)
    assert max(r["meta"]["created_at"] for r in calibration) <= min(r["meta"]["created_at"] for r in test)
