"""scripts/extract_ci_failures.py: redaction and outcome labels, on synthetic data only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import json  # noqa: E402
import math  # noqa: E402

import pytest  # noqa: E402
from extract_ci_failures import (  # noqa: E402
    BASE_RERUN_RATE,
    FEATURES,
    QUESTION,
    STRENGTH,
    add_features,
    clean_log,
    cue_features,
    label_failures,
    label_outcomes,
    live_features,
    main,
    parse_state,
    state_for,
)


def test_clean_log_strips_markup_and_redacts_secrets():
    log = ("\x1b[32;1msection_start:1700000000:step_script\r\x1b[0K$ dotnet test\n"
           "export API_KEY=abc123 PASSWORD: hunter2\n"
           "Server=db;User Id=sa;Password=s3cr3t;\n"
           "curl -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig'\n"
           "pushed by dev.person@example.com with glpat-ABCDEFGHIJKLMNOPQRSTUV\n"
           "key AKIAABCDEFGHIJKLMNOP and " + "Z" * 48 + "\n"
           "Failed! - Failed: 1, Passed: 853\n")
    out = clean_log(log)
    for secret in ("abc123", "hunter2", "s3cr3t", "eyJhbGci", "example.com", "glpat-", "AKIAABCD", "Z" * 48):
        assert secret not in out, secret
    assert "section_start" not in out and "\x1b" not in out
    assert "Failed: 1, Passed: 853" in out


@pytest.mark.parametrize("line, secret", [
    ("export DB_PASS=hunter2", "hunter2"),
    ("mysql -u root -phunter2 db", "hunter2"),
    ("Server=db;User ID=sa;Pass=hunter2;", "hunter2"),
    ('curl -H "X-Auth: abcdefgh12345678" https://api', "abcdefgh12345678"),
    ("curl -H 'Authorization: Basic dXNlcjpwYXNz' https://api", "dXNlcjpwYXNz"),
    ("git clone https://ci:p!ss@host.example.com/g/a.git", "p!"),
    ("Cookie: session=abcdef123456", "abcdef123456"),
])
def test_clean_log_redacts_other_secret_shapes(line, secret):
    assert secret not in clean_log(line)


def test_clean_log_redacts_before_cutting_the_tail():
    # The secret's line starts just before the character cut: cutting first would leave "rd=hunter2".
    filler = "\n".join("v " * 50 for _ in range(50))  # 50 lines of 99 characters
    out = clean_log("setup password=hunter2\n" + filler)
    assert "hunter2" not in out
    assert out.startswith("v ")  # the split first line is dropped, not kept in part


def test_clean_log_keeps_ordinary_lines():
    for line in ("Passed: 853", "KeyNotFoundException: The given key was not present", "mysql -u root -p db"):
        assert clean_log(line) == line


def test_labels_follow_what_fixed_the_failure():
    def p(i, ref, sha, status):
        return {"id": i, "ref": ref, "sha": sha, "status": status}

    pipelines = [
        p(1, "main", "a", "failed"), p(2, "main", "a", "success"),       # rerun on the same commit: not code
        p(3, "mr/1", "b", "failed"), p(4, "mr/1", "c", "success"),       # passed only after a new commit: code
        p(5, "mr/2", "d", "failed"), p(6, "mr/2", "d", "failed"),        # never passed: unknown, left out
        p(7, "mr/3", "e", "failed"), p(8, "mr/3", "f", "failed"), p(9, "mr/3", "f", "success"),
    ]
    labels = label_failures(pipelines)
    assert labels[1] == "0"
    assert labels[3] == "new_commit"  # failed once, then a new commit passed: ambiguous
    assert 5 not in labels and 6 not in labels
    assert labels[7] == "new_commit" and labels[8] == "0"
    reproduced = label_failures([p(1, "mr", "a", "failed"), p(2, "mr", "a", "failed"), p(3, "mr", "b", "success")])
    assert reproduced[1] == "1"  # failed again on the same commit before a new one fixed it


def _row(t, project, job, label, reason="script_failure", log="error CS0103: name not found", **meta):
    state = state_for({"name": job, "stage": "test", "failure_reason": reason}, log)
    return {"state": state, "question": QUESTION, "label": label,
            "meta": {"project": project, "created_at": f"2026-08-01T00:00:{t:02d}Z", **meta}}


def parse_job(row):
    return row["state"].split("\n")[0].split(" ")[1]


def test_no_history_gives_the_prior():
    (row,) = add_features([_row(0, "g/a", "unit", "1")])
    f = row["features"]
    assert set(f) == set(FEATURES)
    assert f["project_rerun_rate"] == f["job_rerun_rate"] == pytest.approx(BASE_RERUN_RATE)
    assert f["project_history"] == f["job_history"] == 0.0


def test_rates_use_only_strictly_earlier_rows():
    rows = [_row(1, "g/a", "e2e", "0"), _row(2, "g/a", "e2e", "0"), _row(3, "g/a", "unit", "1"),
            _row(3, "g/a", "e2e", "1"),  # same instant as the row before: neither sees the other
            _row(4, "g/b", "e2e", "0")]
    shuffled = [rows[i] for i in (4, 2, 0, 3, 1)]  # input order must not matter, only time
    out = {(r["meta"]["created_at"][-3:-1], parse_job(r)): r["features"] for r in add_features(shuffled)}
    glob = lambda zeros, n: (zeros + STRENGTH * BASE_RERUN_RATE) / (n + STRENGTH)  # noqa: E731
    smooth = lambda zeros, n, prior: (zeros + STRENGTH * prior) / (n + STRENGTH)  # noqa: E731
    # Row at t=3 (e2e): two earlier e2e rows in g/a, both "0"; its own "1" is not counted.
    g = glob(2, 2)
    proj = smooth(2, 2, g)
    f = out[("03", "e2e")]
    assert f["project_rerun_rate"] == pytest.approx(proj)
    assert f["job_rerun_rate"] == pytest.approx(smooth(2, 2, proj))
    assert f["job_history"] == pytest.approx(math.log1p(2))
    # Its same-time peer (unit) has the same project history and none of its own.
    assert out[("03", "unit")]["project_rerun_rate"] == pytest.approx(proj)
    assert out[("03", "unit")]["job_history"] == 0.0
    # Changing a later label cannot move an earlier row's features.
    flipped = [dict(r, label="1" if r["meta"]["created_at"].endswith("04Z") else r["label"]) for r in rows]
    again = {(r["meta"]["created_at"][-3:-1], parse_job(r)): r["features"] for r in add_features(flipped)}
    for key in [k for k in out if k[0] != "04"]:
        assert again[key] == out[key]
    # g/b has no history of its own at t=4: its rate is the global rate over the four earlier rows.
    assert out[("04", "e2e")]["project_rerun_rate"] == pytest.approx(glob(2, 4))
    assert out[("04", "e2e")]["project_history"] == 0.0


def test_live_features_match_the_extraction(tmp_path):
    rows = add_features([_row(1, "g/a", "e2e", "0"), _row(2, "g/a", "e2e", "1"), _row(3, "g/a", "e2e", "0")])
    path = tmp_path / "calibration.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = "error CS0103: name not found"
    assert live_features([path], "g/a", "e2e", "script_failure", log,
                         created_at=rows[2]["meta"]["created_at"]) == rows[2]["features"]
    after_all = live_features([path], "g/a", "e2e", "script_failure", log)
    assert after_all["job_history"] == pytest.approx(math.log1p(3))


def test_cues_fire_on_sample_lines():
    def cues(log, name="build", reason="script_failure"):
        return {k for k, v in cue_features(name, reason, log).items() if v}

    assert cues("error CS0103: The name 'x' does not exist") == set()
    assert cues("ok", reason="runner_system_failure") == {"failure_reason_not_script"}
    assert cues("ok", reason="unknown") == set()  # state_for writes "unknown" for a missing reason
    assert cues("ok", name="e2e:chrome") == {"name_e2e"}
    assert cues("ok", name="integration-tests") == {"name_e2e"}
    assert cues("ok", name="deploy_staging") == {"name_deploy"}
    assert cues("ERROR: Job failed: execution took longer than 1h0m0s seconds (timed out)") == {"log_timeout"}
    assert cues("read tcp 10.0.0.1:443: connection reset by peer") == {"log_connection"}
    assert cues("Error response from daemon: pull access denied, 401 Unauthorized for registry.example") \
        == {"log_registry_auth"}
    assert cues("error: 401 Unauthorized") == set()  # a 401 from the app under test is not a registry
    assert cues("write /builds/x: no space left on device") == {"log_no_space"}
    assert cues("FATAL ERROR: Reached heap limit - JavaScript heap out of memory") == {"log_oom"}
    assert cues("Container was OOMKilled, exit code 137") == {"log_oom"}


def test_settling_pipeline_is_the_run_that_decided_the_label():
    def p(i, sha, status):
        return {"id": i, "ref": "mr", "sha": sha, "status": status}

    outcomes = label_outcomes([p(1, "a", "failed"), p(2, "b", "failed"), p(3, "a", "success"), p(4, "b", "success")])
    assert outcomes[1][0] == "0" and outcomes[1][1]["id"] == 3  # its own commit's success, not the first success
    assert outcomes[2][0] == "0" and outcomes[2][1]["id"] == 4
    outcomes = label_outcomes([p(1, "a", "failed"), p(2, "a", "failed"), p(3, "b", "success")])
    assert outcomes[1][0] == "1" and outcomes[1][1]["id"] == 3  # known only once the new commit passed


@pytest.mark.parametrize("label", ["0", "1"])
def test_a_retry_does_not_see_the_label_settled_with_its_own(label):
    # Two failed attempts of one job in one pipeline share a label, settled by one later event.
    # Without settled_at the pipeline keeps them apart; the second attempt must get the prior.
    rows = add_features([_row(1, "g/a", "e2e", label, pipeline=7), _row(5, "g/a", "e2e", label, pipeline=7)])
    for row in rows:
        assert row["features"]["job_rerun_rate"] == pytest.approx(BASE_RERUN_RATE)
        assert row["features"]["job_history"] == 0.0
    # Another pipeline's earlier failure still counts.
    other = add_features([_row(1, "g/a", "e2e", label, pipeline=6), _row(5, "g/a", "e2e", label, pipeline=7)])
    assert other[1]["features"]["job_history"] == pytest.approx(math.log1p(1))


def test_history_waits_for_the_label_to_settle(tmp_path):
    early = _row(1, "g/a", "e2e", "0", pipeline=6, settled_at="2026-08-01T00:00:09Z")
    mid = _row(5, "g/a", "e2e", "1", pipeline=8, settled_at="2026-08-01T00:00:20Z")
    late = _row(10, "g/a", "e2e", "1", pipeline=9, settled_at="2026-08-01T00:00:30Z")
    rows = {r["meta"]["pipeline"]: r for r in add_features([early, mid, late])}
    assert rows[8]["features"]["job_history"] == 0.0  # created at :05, the :01 failure settled only at :09
    assert rows[9]["features"]["job_history"] == pytest.approx(math.log1p(1))  # :01 settled by :10; :05 not yet
    path = tmp_path / "calibration.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    assert live_features([path], "g/a", "e2e", "script_failure", "error CS0103: name not found",
                         created_at="2026-08-01T00:00:10Z") == rows[9]["features"]


def test_times_compare_as_times_not_strings(tmp_path):
    rows = [_row(0, "g/a", "e2e", "0"), _row(0, "g/a", "e2e", "0")]
    rows[0]["meta"]["created_at"] = "2026-08-01T12:00:00.500Z"
    rows[1]["meta"]["created_at"] = "2026-08-01T12:00:01Z"  # later, though it sorts first as a string
    out = add_features(rows)
    assert out[0]["meta"]["created_at"].endswith(".500Z")
    assert out[1]["features"]["job_history"] == pytest.approx(math.log1p(1))
    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps(out[0]) + "\n")
    for same_instant in ("2026-08-01T12:00:00.500+00:00", "2026-08-01T14:00:00.500+02:00"):
        assert live_features([path], "g/a", "e2e", None, "", created_at=same_instant)["job_history"] == 0.0
    assert live_features([path], "g/a", "e2e", None, "", created_at="2026-08-01T12:00:01")["job_history"] > 0


def test_job_name_survives_a_stage_with_parentheses():
    state = state_for({"name": "build", "stage": "build (win)", "failure_reason": None}, "x")
    assert parse_state(state) == ("build", "unknown", "x")


def test_refeature_updates_the_summary(tmp_path, monkeypatch):
    rows = [_row(t, "g/a", "e2e", str(t % 2)) for t in range(10)]
    (tmp_path / "calibration.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "test.jsonl").write_text("")
    (tmp_path / "summary.json").write_text(json.dumps({"since": "2026-08-01", "rows": 10, "calibration": 10,
                                                       "test": 0, "label_0": 5}))
    monkeypatch.setattr(sys, "argv", ["x", "--refeature", str(tmp_path), "--test-fraction", "0.2"])
    assert main() == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert (summary["calibration"], summary["test"], summary["test_fraction"]) == (8, 2, 0.2)
    assert summary["features"] == list(FEATURES)
    assert summary["since"] == "2026-08-01" and summary["label_0"] == 5
