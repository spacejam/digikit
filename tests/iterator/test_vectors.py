"""The ITERATOR reference vectors are self-consistent: stored streams match their digests, the control-call counts and
trigger samples agree with the timelines, and every MODE position is covered."""

from __future__ import annotations

import numpy as np
from support import iterator_vectors as iv


def test_manifest_shape() -> None:
    cs = iv.cases()
    assert len(cs) >= 100
    assert len({c.id for c in cs}) == len(cs)
    modes = {c.timeline[0].mode for c in iv.cases("mode")}
    assert modes == set(range(9))
    for fam in (
        "mode",
        "sweep",
        "pitch",
        "trigger",
        "long",
        "edge",
        "sequence",
        "fuzz",
    ):
        assert iv.cases(fam)


def test_stored_streams_match_digests() -> None:
    for c in iv.cases():
        assert len(c.samples) == len(c.periods) == c.stored_samples
        assert int(np.sum(c.periods.astype(np.uint64))) >= 0
        if c.complete:
            assert iv.digests_match(c, c.samples, c.periods), c.id
            assert c.call_samples[0] == 0
            assert len(c.call_samples) == c.n_calls
            assert int(np.sum(c.periods.astype(np.uint64))) >= c.duration_ticks


def test_trigger_samples_follow_gate_edges() -> None:
    for c in iv.cases():
        calls = iv.calls(c)
        edges = [
            int(c.call_samples[k])
            for k in range(len(calls))
            if calls[k].gate and (k == 0 or not calls[k - 1].gate)
        ]
        assert tuple(edges) == c.trigger_samples, c.id
