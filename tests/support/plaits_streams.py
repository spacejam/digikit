"""Control streams for one PLAITS voice, block by block (Task 9b): note-ons at different pitches with knob changes and
no model change (`retrigger_stream`), and model changes at note-ons and between them (`switch_stream`). Each block is
the row values of the controls plus the trigger latch. Used by tests/plaits/test_continuous_host.py (against upstream
rendered continuously) and tests/patchkit/test_plaits_module.py (the compiled core against the host)."""

from __future__ import annotations

from typing import Any

import numpy as np

from support.plaits_host import frame

Block = dict[str, int]


def retrigger_stream(model: int, seed: int, blocks: int = 240) -> list[Block]:
    """Note-ons at different pitches, 1 to 23 blocks apart (repeated notes, 1/32-step bursts and held notes), each with
    its own knob p-locks and velocity (0 and 127 included), knobs ramping within some notes (as an LFO would) and TUNE
    changes; MODEL fixed."""
    rng = np.random.default_rng(seed)
    out: list[Block] = []
    gaps = [1, 1, 2, 3, 5, 8, 13, 23, 4, 1, 6, 10]
    notes = [60, 48, 67, 36, 72, 55, 84, 43, 60, 60, 96, 24]
    velocities = [100, 127, 1, 64, 0, 110, 30, 127, 90, 5, 100, 75, 127]
    nxt, k = 0, 0
    velocity = 100
    knobs = {"harmonics": 0x4000, "timbre": 0x4000, "morph": 0x4000, "mix": 0}
    tune, note, ramp = 0x4000, 60, 0
    for b in range(blocks):
        trig = int(b == nxt)
        if trig:
            note = notes[k % len(notes)]
            velocity = velocities[k % len(velocities)]
            for name in knobs:
                knobs[name] = int(rng.integers(0, 0x7F01))
            if k % 3 == 2:
                tune = int(rng.integers(0x3800, 0x4800))
            ramp = int(rng.integers(-0x300, 0x300)) if k % 2 else 0
            nxt = b + gaps[k % len(gaps)]
            k += 1
        elif ramp:
            for name in ("timbre", "morph"):
                knobs[name] = int(np.clip(knobs[name] + ramp, 0, 0x7F00))
        out.append(
            {
                "note": note,
                "tune": tune,
                "model": model,
                "trig": trig,
                "velocity": velocity,
                **knobs,
            }
        )
    return out


def gate_stream(model: int, seed: int, blocks: int = 480) -> list[Block]:
    """Note-ons with the Digitakt's note-offs, for the DX7 models (the gate: plaits.c): each note is held for 1 to 70 blocks
    (`released` 0, the gate high), then released (`released` 1, the gate low) until the next note-on 0 to 60 blocks later (a
    gap of 0 is a note-on right after the note-off block: a retrigger during the release of the previous note's tail, and
    a note-on before the note-off is a retrigger of a held note), each note with its own pitch, velocity and knobs (HARMONICS
    walks across the patches)."""
    rng = np.random.default_rng(seed)
    out: list[Block] = []
    notes = [60, 48, 67, 36, 72, 55, 84, 43, 60, 60, 96, 24, 64]
    k = 0
    while len(out) < blocks:
        note = notes[k % len(notes)] + int(rng.integers(-2, 3))
        velocity = int(rng.choice([100, 127, 1, 64, 0, 110, 30]))
        knobs = {
            "harmonics": int(rng.integers(0, 0x7F01)),
            "timbre": int(
                rng.choice([0x4000, 0, 0x7F00, int(rng.integers(0, 0x7F01))])
            ),
            "morph": int(rng.choice([0x4000, 0, 0x7F00, int(rng.integers(0, 0x7F01))])),
            "mix": int(rng.choice([0, 0, 0x7F00, 0x2000])),
        }
        hold = int(rng.choice([1, 2, 5, 12, 30, 70, int(rng.integers(1, 71))]))
        gap = int(rng.choice([0, 0, 1, 4, 20, 60, int(rng.integers(0, 61))]))
        for b in range(hold + gap):
            out.append(
                {
                    "note": note,
                    "tune": 0x4000,
                    "model": model,
                    "trig": int(b == 0),
                    "velocity": velocity,
                    "released": int(b >= hold),
                    **knobs,
                }
            )
        k += 1
    return out[:blocks]


def switch_stream() -> list[Block]:
    """Model changes on one voice, 9 blocks per note: note-ons at blocks 0 and 5 of each (a retrigger of the same model
    at 5); at block 8 the MODEL row already shows the next note's model without a note-on (the running engine goes
    on); consecutive notes of one model (6 6, 12 12, 5 5, 7 7, 13 13, 15 15, 9 9, 17 17, 16 16, 23 23, 22 22, 10 10, 8 8, 11 11, 0 0, 18 18, 24 24) do not re-initialise the engine at the note-on
    (SPEECH and RINGS read it as Plaits' trigger); each note-on has its own velocity."""
    seq = [
        0,
        5,
        6,
        6,
        4,
        7,
        12,
        12,
        21,
        6,
        5,
        5,
        7,
        7,
        2,
        21,
        3,
        1,
        12,
        6,
        13,
        13,
        14,
        15,
        15,
        13,
        9,
        9,
        17,
        17,
        16,
        16,
        23,
        23,
        22,
        22,
        10,
        10,
        8,
        8,
        11,
        11,
        0,
        0,
        18,
        18,
        24,
        24,
        19,
        20,
        20,
        0,
    ]
    out: list[Block] = []
    for k, model in enumerate(seq):
        for b in range(9):
            out.append(
                {
                    "note": 45 + 3 * k + (b >= 5),
                    "tune": 0x4000,
                    "model": seq[k + 1] if b == 8 and k + 1 < len(seq) else model,
                    "trig": int(b in (0, 5)),
                    "harmonics": 0x2000 + 0x500 * k,
                    "timbre": 0x7F00 - 0x400 * k,
                    "morph": 0x4400 if b < 5 else 0x3C00,
                    "mix": 0x2000 * (k % 4),
                    "velocity": (100 + 37 * k + 50 * (b >= 5)) % 128,
                }
            )
    return out


def as_frame(s: Block) -> Any:
    return frame(
        s["note"],
        tune=s["tune"],
        model=s["model"],
        harmonics=s["harmonics"],
        timbre=s["timbre"],
        morph=s["morph"],
        mix=s["mix"],
        velocity=s.get("velocity", 100),
        released=s.get("released", 0),
    )
