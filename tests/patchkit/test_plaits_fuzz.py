"""tools/patchkit/pk_plaits_fuzz.py: the sequence generator is deterministic and covers what the search promises, and
one tiny search on the compiled routine (model 21, a few sequences) is reproducible."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tools.patchkit import pk_plaits_fuzz as F


def test_generator_is_deterministic() -> None:
    a = [F.generate(7, 21, i) for i in range(30)]
    assert a == [F.generate(7, 21, i) for i in range(30)]
    assert json.dumps(a) != json.dumps([F.generate(8, 21, i) for i in range(30)])
    assert F.generate(7, 21, 0) != F.generate(7, 12, 0)


def test_generator_covers_the_promised_inputs() -> None:
    steps = [s for i in range(300) for s in F.generate(1, 21, i)["steps"]]
    for k in ("harmonics", "timbre", "morph", "tune"):
        vals = {s[k] for s in steps}
        assert (
            0 in vals and F.KNOB_MAX in vals and all(0 <= v <= F.KNOB_MAX for v in vals)
        ), k
    notes = {s["note"] for s in steps}
    assert {0, 127} <= notes and all(0 <= n <= 127 for n in notes)
    assert {0, 127} <= {s["velocity"] for s in steps}
    for i in range(300):
        seq = F.generate(1, 21, i)["steps"]
        assert seq[0]["trig"] == 1  # the voice's first note-on
        assert all(s["model"] in F.ENABLED for s in seq)
        assert seq[-1]["model"] == 21  # always ends switched into N
    assert any(
        s["model"] != 21 for i in range(50) for s in F.generate(1, 21, i)["steps"]
    )
    gaps = []
    for i in range(300):
        on = [b for b, s in enumerate(F.generate(1, 21, i)["steps"]) if s["trig"]]
        gaps += [y - x for x, y in zip(on, on[1:], strict=False)]
    assert 1 in gaps and max(gaps) > 16  # fast retriggers and long gaps


def test_dx7_sequences_carry_note_offs_and_the_others_do_not_change() -> None:
    """The DX7 models' steps carry `released` (0 at every note-on, 1 from a note-off until the next note-on; the gate is
    low there), from a generator of their own: the other models' sequences have no such key, so they are the ones they
    always were."""
    for model in F.DX7:
        saw_release = saw_hold = False
        for i in range(60):
            steps = F.generate(2, model, i)["steps"]
            assert all("released" in st for st in steps)
            assert all(st["released"] == 0 for st in steps if st["trig"])
            saw_release |= any(st["released"] for st in steps)
            saw_hold |= any(not st["released"] and not st["trig"] for st in steps)
            # released is 1 only after a note-on of the sequence and stays 1 to the next note-on
            up = False
            for st in steps:
                if st["trig"]:
                    up = False
                if up:
                    assert st["released"] == 1
                up = up or bool(st["released"])
        assert saw_release and saw_hold, model
    assert all(
        "released" not in st for i in range(40) for st in F.generate(2, 12, i)["steps"]
    )


def test_mutation_and_confirm_knobs() -> None:
    import random

    seq = F.generate(3, 12, 0)
    assert F.mutate(seq, random.Random(1)) == F.mutate(seq, random.Random(1))
    assert seq == F.generate(3, 12, 0)  # the original is not touched
    k = F.confirm_knobs(
        {"note": 60, "tune": 0x4000, "harmonics": 5, "timbre": 6, "morph": 7}
    )
    assert k == {"TUNE": 0x4000, "HARM": 5, "TIMB": 6, "MORPH": 7}
    hi = F.confirm_knobs(
        {"note": 127, "tune": 0x7F00, "harmonics": 0, "timbre": 0, "morph": 0}
    )
    assert hi["TUNE"] == 0x7C00
    assert (
        F.confirm_knobs(
            {"note": 0, "tune": 0, "harmonics": 0, "timbre": 0, "morph": 0}
        )["TUNE"]
        == 0x0400
    )
    assert F.compare(21, 71_204)["needs_confirm"] is False
    assert F.compare(21, 73_000)["needs_confirm"] is True


def test_tiny_run_on_the_compiled_routine_is_reproducible() -> None:
    """The first 5 blocks of a TERRAIN sequence on the compiled routine (about 15 s): counted from the note-on that
    latches the model, the same instructions twice on a rig whose voice is cleared in between, within the bound of
    MAX_BLOCK_INSTRUCTIONS_NEW[21][1]."""
    from tools.patchkit.plaits_rig import PlaitsRig

    seq = F.generate(5, 21, 0)
    seq["steps"] = seq["steps"][:5]
    rig = PlaitsRig(1)
    a = F.run_sequence(rig, seq)
    b = F.run_sequence(rig, seq)
    assert a == b and a[3] is True
    assert len(a[2]) == 5 and all(n > 1_000 for n in a[2])
    assert a[0] == 0 or 20_000 < a[0] < 82_000


def test_save_never_overwrites_and_names_the_run(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two saves of the same (model, seed, jobs, sequences) in one second keep both files; the name carries the run."""
    from tools.patchkit import pk_plaits_fuzz as P

    monkeypatch.setattr(P, "OUT", tmp_path)
    res = {
        "model": 9,
        "seed": 0,
        "jobs": 3,
        "sequences": 33,
        "worst": {"instructions": 1},
    }
    a = P.save(res)
    b = P.save({**res, "worst": {"instructions": 2}})
    assert a != b and a.exists() and b.exists()
    assert a.name.startswith("m9-seed0-j3-n33-") and a.suffix == ".json"
    assert '"instructions": 1' in a.read_text() and '"instructions": 2' in b.read_text()
