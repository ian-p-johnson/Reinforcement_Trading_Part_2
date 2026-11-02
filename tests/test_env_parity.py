"""Parity tests: the refactored env must reproduce golden traces exactly.

The fixtures in ``tests/golden/`` were generated from the pre-Track-A
implementation (see DECISIONS.md P1).  Any behaviour change must either match
them or deliberately regenerate them with a reviewed diff.
"""
from __future__ import annotations

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path setup)
from make_golden import compare
from scenarios import SCENARIOS, run_trace


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_golden_parity(scenario):
    trace, trades = run_trace(scenario)
    problems = compare(scenario.name, trace, trades)
    assert not problems, "\n".join(problems)


def test_edge_case_coverage():
    """The fixtures must actually exercise the semantics they claim to cover."""
    from make_golden import load_fixture

    _tr, trades = load_fixture("same_candle_sl_tp")
    assert len(trades) == 1
    assert trades["exit_reason"].iloc[0] == "SL", \
        "same-candle double touch must resolve to SL (pessimistic rule)"

    tr, _ = load_fixture("truncation_40")
    assert tr["truncated"].sum() == 1 and not tr["terminated"].any()

    tr, _ = load_fixture("data_end")
    assert tr["terminated"].sum() == 1

    tr, _ = load_fixture("random_start")
    assert int(tr["start_i"]) != 0

    tr, _ = load_fixture("flat")
    assert len(tr["reward"]) > 100
