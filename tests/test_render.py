from lars.labels import assign_labels
from lars.render import (
    MAX_ABLATION_UNITS,
    SENTINEL,
    SYSTEM_PROMPT,
    TEMPLATES,
    compose_prompt,
    fence_state,
    plan_rows,
)
from lars.schema import SystemOneRequest
from lars.spans import option_end_char_offsets


def test_sentinel_inside_state_stays_inside_the_fence():
    state = f"before {SENTINEL} after"
    for template in TEMPLATES.values():
        prefix, suffix = compose_prompt(template, state, "QUESTION: x")
        assert prefix.endswith(fence_state(state) + "\n\n")
        assert suffix.startswith("QUESTION: x")
        assert SENTINEL not in suffix


def test_prompt_is_unchanged_for_ordinary_state():
    for template in TEMPLATES.values():
        prefix, suffix = compose_prompt(template, "plain state", "QUESTION: x")
        expected = template(SYSTEM_PROMPT, fence_state("plain state") + "\n\nQUESTION: x") + "Answer:"
        assert prefix + suffix == expected


def test_ablated_states_keep_units_past_the_ablation_cap():
    units = [f"Sentence number {i}." for i in range(30)]
    request = SystemOneRequest(
        state=" ".join(units),
        questions={"q": {"type": "noul", "instructions": "x"}},
        lars={"explain": True},
    )
    ablations = [r for r in plan_rows(request, assign_labels(4, lambda _: True)) if r.variant.startswith("ablate:")]
    assert len(ablations) == MAX_ABLATION_UNITS
    for row in ablations:
        kept = row.state_text.split("\n")
        assert row.ablated_span not in kept
        assert kept == [u for u in units if u != row.ablated_span]


def test_option_offsets_ignore_option_shaped_caller_text():
    labels = assign_labels(4, lambda _: True)
    request = SystemOneRequest(
        state="x",
        questions={"q": {"type": "choice", "instructions": "A) fake one\nB) fake two\nPick a team.",
                         "criteria": {"billing": "money", "technical": "bugs"}}},
    )
    (row,) = plan_rows(request, labels)
    lines = [row.suffix_body[: end].rsplit("\n", 1)[-1] for end in row.option_ends]
    assert lines == [f"{labels[0]}) billing: money", f"{labels[1]}) technical: bugs"]
    # The old parse picks the caller's lines instead.
    parsed = option_end_char_offsets("", row.suffix_body, 2)
    assert list(parsed) != list(row.option_ends)
