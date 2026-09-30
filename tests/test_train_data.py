import json

import pytest

from lars.train.data import (
    Record,
    from_jev_bench,
    from_open_jev,
    from_tasksource_jev,
    read_records,
    split_by_group,
    write_records,
)


def test_open_jev_conversion_reorders_noul_and_splits_choice_keys():
    rows = [
        {"id": "n1", "kind": "noul", "question": "Is it urgent?", "options": ["no", "yes"], "target": [0.2, 0.8],
         "state_json": json.dumps("Please help now")},
        {"id": "c1", "kind": "choice", "question": "Category?", "options": ["billing: Charges", "bug_report: Broken"],
         "target": [0.0, 1.0], "state_json": json.dumps({"message": "it crashes"})},
        {"id": "s1", "kind": "score", "question": "Severity?", "options": ["Cosmetic", "Degraded", "Blocking"],
         "target": [0.0, 0.5, 0.5], "state_json": json.dumps("half broken")},
        {"id": "m1", "kind": "multilabel", "question": "?", "options": ["a", "b"], "target": [1, 1],
         "state_json": "\"x\""},
    ]
    records = list(from_open_jev(rows))
    assert [r.kind for r in records] == ["noul", "choice", "score"]
    assert records[0].options == ["yes", "no"] and records[0].target == [0.8, 0.2]
    assert records[1].options == ["billing", "bug_report"]
    assert records[1].descriptions == ["Charges", "Broken"]
    assert records[1].state == {"message": "it crashes"}
    assert records[1].to_question()["criteria"] == {"billing": "Charges", "bug_report": "Broken"}
    assert records[2].to_question()["criteria"] == ["Cosmetic", "Degraded", "Blocking"]


def test_tasksource_conversion_keeps_choice_only():
    rows = [
        {"id": "t1", "kind": "choice", "state": "text_A: a\ntext_B: b", "question": "Entailed?",
         "options": ["yes", "no", "maybe"], "target": [1.0, 0.0, 0.0], "source": "glue/rte"},
        {"id": "t2", "kind": "score", "state": "x", "question": "?", "options": ["a", "b"], "target": [1, 0]},
    ]
    records = list(from_tasksource_jev(rows))
    assert len(records) == 1 and records[0].source == "tasksource-jev/glue/rte"


def test_jev_bench_conversion_uses_soft_labels():
    score_question = {"type": "score", "instructions": "Sentiment?", "criteria": ["vneg", "neg", "neu", "pos", "vpos"]}
    rows = [
        {"state": json.dumps("great film"), "label": "4", "soft_label": None, "question": json.dumps(score_question)},
        {"state": "toxic text", "label": "1", "soft_label": json.dumps({"yes": 0.75, "no": 0.25}),
         "question": {"type": "noul", "instructions": "Toxic?"}},
        {"state": "hyp/prem", "label": "entailment", "soft_label": json.dumps({"entailment": 0.6, "neutral": 0.4}),
         "question": {"type": "choice", "instructions": "NLI?",
                      "criteria": {"entailment": "e", "neutral": "n", "contradiction": "c"}}},
    ]
    records = list(from_jev_bench(rows, "unit"))
    assert records[0].target == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert records[1].target == [0.75, 0.25]
    assert records[2].target == [0.6, 0.4, 0.0]


def test_roundtrip_and_group_split(tmp_path):
    records = [Record(f"r{i}", f"src{i % 4}", "noul", "s", "q", ["yes", "no"], [1.0, 0.0]) for i in range(20)]
    path = tmp_path / "r.jsonl"
    assert write_records(records, path) == 20
    loaded = list(read_records(path))
    assert loaded == records
    train, held = split_by_group(loaded, holdout_fraction=0.25, seed=1)
    assert len(held) == 5 and not ({r.source for r in train} & {r.source for r in held})


def test_named_holdout_sources_are_fixed_and_checked():
    records = [Record(f"{s}{i}", s, "noul", "x", "q", ["yes", "no"], [1.0, 0.0]) for s in "abcd" for i in range(3)]
    for seed in (0, 1, 2):
        train, held = split_by_group(records, 0.5, seed=seed, sources_to_hold=["b", "d"])
        assert {r.source for r in held} == {"b", "d"} and {r.source for r in train} == {"a", "c"}
    with pytest.raises(ValueError, match="not in the records"):
        split_by_group(records, 0.5, sources_to_hold=["z"])
