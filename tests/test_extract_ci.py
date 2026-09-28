"""scripts/extract_ci_failures.py: redaction and outcome labels, on synthetic data only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from extract_ci_failures import clean_log, label_failures  # noqa: E402


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
