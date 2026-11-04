"""The gate is the part worth testing: a search is only as good as what it cannot talk its way past.

Skipped unless the [triage] extra is installed, since the search is scored on the triage benchmark.
"""
from __future__ import annotations

import pytest

pytest.importorskip("triage")
pytest.importorskip("langgraph.checkpoint.sqlite")

from evolve import Knobs, neighbours  # noqa: E402
from evolve.guard import PROBES, check_probes, frozen_parts_intact, probes_ok, run_probe  # noqa: E402
from evolve.knobs import NON_IDEMPOTENT, SHIPPED  # noqa: E402
from evolve.proposers import RuleProposer  # noqa: E402
from evolve.scorer import Score  # noqa: E402

# The tools a counterexample actually exercises. comment_ticket is not one of them: it is caught
# by the benchmark invariants instead (a run of the loop rejected both of its unsafe strategies
# that way, after they scored a perfect 75/75). Naming the gap is the point.
PROBED = ("rerun_job", "append_audit")

UNIQUE = Knobs(recovery=tuple((t, "unique_payload") for t in sorted(NON_IDEMPOTENT)))


def test_shipped_policy_pauses_rather_than_guessing():
    assert set(check_probes(SHIPPED).values()) == {"paused"}


def test_unique_payload_is_safe_on_every_counterexample():
    assert probes_ok(check_probes(UNIQUE))


def test_unique_payload_still_pauses_when_the_same_call_appears_twice():
    """The whole point of the guard clause: identical earlier call, so the evidence is not
    about this step and there is nothing safe to conclude."""
    r = check_probes(UNIQUE)
    assert r["repeated_action_never_ran"] == "paused"
    assert r["repeated_action_already_ran"] == "paused"


def test_unique_payload_resolves_a_one_off_action_without_a_person():
    r = check_probes(UNIQUE)
    assert r["unique_action_never_ran"] == "clean"
    assert r["unique_action_already_ran"] == "clean"


@pytest.mark.parametrize("strategy,probe,outcome", [
    ("assume_done", "repeated_action_never_ran", "dropped"),
    ("retry", "repeated_action_already_ran", "duplicated"),
])
def test_the_probes_actually_catch_the_unsafe_strategies(strategy, probe, outcome):
    """If these ever come back 'clean', the gate has stopped doing its job and every result
    the loop produces afterwards is worthless."""
    k = SHIPPED.with_recovery("rerun_job", strategy)
    assert check_probes(k)[probe] == outcome
    assert not probes_ok(check_probes(k))


def test_every_probe_is_decisive_for_some_policy():
    """A probe that passes for everything tests nothing."""
    policies = [SHIPPED, UNIQUE] + [SHIPPED.with_recovery(t, s) for t in NON_IDEMPOTENT
                                    for s in ("assume_done", "retry")]
    for probe in PROBES:
        outcomes = {run_probe(probe, k) for k in policies}
        assert outcomes & {"dropped", "duplicated"}, f"{probe.name} never fails for any policy"


@pytest.mark.parametrize("tool", PROBED)
@pytest.mark.parametrize("strategy", ("assume_done", "retry"))
def test_guessing_is_rejected_for_every_tool_a_probe_covers(tool, strategy):
    """The gate's job in one line. If this ever passes, the search can buy autonomy with a
    dropped or duplicated action and every result it produces afterwards is worthless."""
    assert not probes_ok(check_probes(SHIPPED.with_recovery(tool, strategy)))


def test_search_space_rejects_anything_outside_it():
    with pytest.raises(ValueError):
        Knobs(recovery=(("patch_procedure", "retry"),)).validate()  # idempotent tool, not a knob
    with pytest.raises(ValueError):
        Knobs(recovery=(("rerun_job", "ignore_everything"),)).validate()
    with pytest.raises(ValueError):
        Knobs(max_retries=99).validate()
    assert not all(frozen_parts_intact(Knobs(max_retries=99)).values())


def test_neighbours_are_one_knob_away_and_valid():
    seen = list(neighbours(SHIPPED))
    assert seen and len(seen) == len(set(seen))
    for k in seen:
        k.validate()
        diffs = (k.recovery != SHIPPED.recovery) + (k.max_retries != SHIPPED.max_retries) \
            + (k.min_confidence != SHIPPED.min_confidence)
        assert diffs == 1


def test_pause_is_canonical_so_the_loop_does_not_remeasure_the_same_policy():
    assert SHIPPED.with_recovery("rerun_job", "pause") == SHIPPED
    assert Knobs.from_dict({"recovery": {"rerun_job": "pause"}}) == SHIPPED


def test_rule_proposer_aims_at_the_tool_causing_the_interruptions():
    best = Score(knobs=SHIPPED, reached_gate=75, unattended=45, paused_on={"append_audit": 20, "rerun_job": 10})
    first = RuleProposer().propose(best, {SHIPPED})[0]
    assert first.recovery_map == {"append_audit": "unique_payload"}
