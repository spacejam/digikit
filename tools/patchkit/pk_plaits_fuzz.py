"""Fuzz search for the worst-case CPU block of one PLAITS model, on the isolated compiled routine.

    uv run python -m tools.patchkit.pk_plaits_cpu --fuzz --model N [--seed S] [--budget-seconds T] [--confirm]

The emulator run of pk_plaits_cpu takes minutes to an hour per knob setting, so worst-case knobs picked by hand can be
missed (TERRAIN's post-jump path was). This searches on the compiled PLAITS object of machines/plaits-48k (mode 1)
on the concrete SHARC core (tools/patchkit/plaits_rig.py, the rig of the compiled-core tests), counting instructions
per block over random sequences, and keeps the worst. `--confirm` then hands its static knobs and note to one run of
the emulator path (pk_plaits_cpu).

A sequence is a list of blocks (one voice, one block each) as the machine sees them: the knob rows (TUNE, HARMONICS,
TIMBRE, MORPH, MIX), the note, the velocity, the MODEL row and the note-on latch. It follows the conventions of
plaits.c: the model is latched at a note-on (the engine is initialised at the voice's first note-on and at a note-on
that changes the model, and runs on from note to note otherwise); a MODEL change without a note-on leaves the old
engine running; every note-on is Plaits' RISING_EDGE (strike engines) in the block of the note-on, HIGH after; for the
DX7 models (DX7) a step also carries `released` (the Digitakt's note-off flag: 1 from a note-off until the next note-on,
and the gate is LOW then). A sequence may start on another model and switch into N by a note-on or by turning MODEL first. Only blocks while the
voice has MODEL N latched count, so the init block of the switch is included and another engine's blocks are not.

Sequence i of a (seed, model) is a pure function of them, and so is the whole search for a given (seed, jobs,
--sequences). With `--budget-seconds` alone the search is deadline-bound: the number of sequences reached depends on
the machine's speed and load, so the same seed is NOT reproducible (a run under load found 63,391 where an idle one
found 64,832); use `--sequences K` for a reproducible search. The result JSON goes to
out/runs/plaits-cpu/fuzz/mN-seedS-jJ-nK-<UTC time>.json and is never overwritten: "worst" and "sequence" never
contain time; "timing" does.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "runs" / "plaits-cpu" / "fuzz"
ENABLED = (0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 17, 18, 19, 20, 21, 24)
DX7 = (
    18,
    19,
    20,
    24,
)  # the models whose gate follows the note-off (plaits.c): their sequences carry "released"
KNOB_MAX = 0x7F00
NOTABLE = (0x0400, 0x0A00, 0x1500, 0x2000, 0x3C00, 0x4000, 0x4600, 0x7C00)
TUNE_PANEL = (
    0x0400,
    0x7C00,
)  # the lowest / highest TUNE the panel reaches (pk_plaits_cpu WORST)

# Worst block (instructions, mode 1 = the 48k object) recorded for each model: finding 06's PLAITS sections and the
# bounds' comments in tests/patchkit/test_plaits_module.py (MAX_BLOCK_INSTRUCTIONS*, measured on the extreme grid or
# the run-on grid, including the model-change block).
RECORDED_WORST = {
    0: 34_381,  # O5 shape/saw kernels (seed 1, 30 sequences); 75,626 before
    1: 42_307,
    2: 57_442,
    3: 80_917,
    4: 25_466,  # O5 harmonic/amplitude kernels (seed 1, 26 sequences); 88,156 before
    5: 41_168,
    6: 30_711,  # CHORD kernels + output-stage kernel (seed 1); 55,296 in C
    7: 34_083,
    9: 64_832,
    16: 66_817,
    23: 23_689,
    22: 72_064,
    10: 114_002,
    11: 85_757,  # the delay lines cleared through a pointer; 106,610 with indexed stores
    8: 46_189,  # continuous (unpatched) mode; 46,274 in the burst mode
    17: 33_953,  # O5 oscillator/downsampling kernels (seed 0, 35 sequences); 97,034 before
    12: 25_083,  # + output-stage kernel (seed 1); 31,472 before
    13: 17_999,  # drum kernels (lib/drums_sm.s) + output-stage kernel; 67,683 in C
    14: 26_340,  # 73,269 in C
    15: 27_959,  # 71,381 in C
    21: 71_204,
    18: 96_314,  # DX7 (one engine, four banks): seed 1, 67 sequences, the generic render_ops loop
}
TOLERANCE = 0.02  # a fuzz worst more than this above the recorded worst needs an emulator confirmation

# -- the generator (pure; tests/patchkit/test_plaits_fuzz.py)

FIELDS = ("note", "velocity", "tune", "harmonics", "timbre", "morph", "mix")


def _knob(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.2:
        return 0
    if r < 0.4:
        return KNOB_MAX
    if r < 0.55:
        return rng.choice(NOTABLE)
    return rng.randint(0, KNOB_MAX)


def _field(rng: random.Random, name: str) -> int:
    if name == "note":
        r = rng.random()
        return rng.choice((0, 127)) if r < 0.25 else rng.randint(0, 127)
    if name == "velocity":
        r = rng.random()
        return rng.choice((0, 1, 100, 127)) if r < 0.4 else rng.randint(0, 127)
    if name == "tune":
        r = rng.random()
        if r < 0.5:
            return 0x4000
        if r < 0.8:
            return rng.choice((0, KNOB_MAX))
        return rng.randint(0, KNOB_MAX)
    if name == "mix":
        return 0 if rng.random() < 0.8 else _knob(rng)
    return _knob(rng)


def _spacing(rng: random.Random) -> int:
    """Blocks until the next note-on: fast retriggers (1, 2), musical, and long."""
    r = rng.random()
    if r < 0.3:
        return 1
    if r < 0.5:
        return 2
    if r < 0.8:
        return rng.randint(3, 16)
    return rng.randint(17, 48)


def generate(seed: int, model: int, index: int) -> dict[str, Any]:
    """Sequence INDEX of (SEED, MODEL): {"model", "steps": [{model, trig, note, velocity, tune, harmonics, timbre,
    morph, mix}]}; "model" is the row value's model number of each step (row = model << 8)."""
    rng = random.Random(f"plaits-fuzz:{seed}:{model}:{index}")
    n = rng.randint(6, 40)
    state = {k: _field(rng, k) for k in FIELDS}
    p_jump = rng.choice(
        (0.0, 0.05, 0.3, 1.0)
    )  # how often the knobs and the note change from block to block
    steps: list[dict[str, int]] = []
    # the model sequence: start on another model (60 %), switch into N by a note-on or by turning MODEL first
    pre_model = (
        rng.choice([m for m in ENABLED if m != model]) if rng.random() < 0.6 else model
    )
    pre = rng.randint(1, 4) if pre_model != model else 0
    turn = rng.randint(1, 3) if pre and rng.random() < 0.3 else 0
    next_on = 0
    for b in range(n):
        if b and rng.random() < p_jump:
            for k in FIELDS:
                if rng.random() < 0.5:
                    state[k] = _field(rng, k)
        m = pre_model if b < pre else model
        trig = int(b == 0 or b == pre + turn or b == next_on)
        if trig:
            next_on = b + _spacing(rng)
        steps.append({"model": m, "trig": trig, **state})
    if model in DX7:
        add_releases(steps, random.Random(f"plaits-fuzz-rel:{seed}:{model}:{index}"))
    return {"model": model, "steps": steps}


def add_releases(steps: list[dict[str, int]], rng: random.Random) -> None:
    """The Digitakt's note-offs for a DX7 sequence (a separate generator, so that the other models' sequences are the
    ones they always were): after each note-on the note is held for a random number of blocks (1 to the next note-on, or
    never), then `released` is 1 until the next note-on."""
    hold = 0
    released = 0
    for st in steps:
        if st["trig"]:
            released = 0
            hold = rng.choice((1, 1, 2, 3, 5, 12, 30, 1000))
        elif hold:
            hold -= 1
            if hold == 0:
                released = 1
        st["released"] = released


def mutate(seq: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """A copy of SEQ with 1..3 fields of random blocks resampled, or a note-on moved or added."""
    steps = [dict(s) for s in seq["steps"]]
    for _ in range(rng.randint(1, 3)):
        i = rng.randrange(len(steps))
        r = rng.random()
        if r < 0.15 and steps[i]["model"] == seq["model"]:
            steps[i]["trig"] ^= 1
        elif r < 0.25 and "released" in steps[i]:
            steps[i]["released"] ^= 1
        else:
            name = rng.choice(FIELDS)
            steps[i][name] = _field(rng, name)
            if (
                rng.random() < 0.5
            ):  # a held change: the rest of the sequence keeps the new value
                for s in steps[i:]:
                    s[name] = steps[i][name]
    return {"model": seq["model"], "steps": steps}


def confirm_knobs(step: dict[str, int]) -> dict[str, int]:
    """The emulator path's static settings for a block: the emulator plays note 60 (the track's default trig), so
    the note is carried by TUNE (note = 60 + (TUNE - 0x4000) / 256, plaits.c), clamped to what the panel reaches;
    HARMONICS, TIMBRE and MORPH as they are (MIX is not on the tool's knob list; velocity is the default 100)."""
    eff = step["note"] + (step["tune"] - 0x4000) / 256
    tune = round(0x4000 + (eff - 60) * 256)
    tune = max(TUNE_PANEL[0], min(TUNE_PANEL[1], tune)) // 0x100 * 0x100
    return {
        "TUNE": tune,
        "HARM": step["harmonics"],
        "TIMB": step["timbre"],
        "MORPH": step["morph"],
    }


def compare(model: int, worst: int) -> dict[str, Any]:
    rec = RECORDED_WORST.get(model)
    if rec is None:
        return {"recorded": None, "delta_pct": None, "needs_confirm": True}
    d = 100.0 * (worst - rec) / rec
    return {
        "recorded": rec,
        "delta_pct": round(d, 2),
        "needs_confirm": d > 100 * TOLERANCE,
    }


# -- the search on the rig


def run_sequence(rig: Any, seq: dict[str, Any]) -> tuple[int, int, list[int], bool]:
    """Play SEQ on voice 0 of RIG from a never-triggered voice. Returns (worst counted block's instructions, its step
    index, the instructions of every block, all output finite). A block counts while the voice has seq["model"]
    latched."""
    import math

    rig.clear_voice(0)
    rig.active(0)
    rig.trig(0, False)
    rig.instructions.clear()
    latched: int | None = None
    worst, at, finite = 0, -1, True
    per: list[int] = []
    for i, s in enumerate(seq["steps"]):
        rig.controls(
            0,
            s["note"],
            tune=s["tune"],
            velocity=s["velocity"],
            model=s["model"] << 8,
            harmonics=s["harmonics"],
            timbre=s["timbre"],
            morph=s["morph"],
            mix=s["mix"],
        )
        rig.trig(0, bool(s["trig"]))
        rig.release(0, bool(s.get("released", 0)))
        if s["trig"]:
            latched = s["model"]
        out = rig.block(0)
        n = rig.instructions[-1]
        per.append(n)
        finite = finite and all(math.isfinite(x) for x in out["samples"])
        if latched == seq["model"] and n > worst:
            worst, at = n, i
    return worst, at, per, finite


def _worker(
    args: tuple[int, int, int, int, float | None, int | None, int],
) -> dict[str, Any]:
    """One process: its own rig, sequences w, w + jobs, ... until the deadline or the count."""
    seed, model, w, jobs, deadline, total, mode = args
    from tools.patchkit.plaits_rig import PlaitsRig

    rig = PlaitsRig(mode)
    rng = random.Random(f"plaits-fuzz-mut:{seed}:{model}:{w}")
    best: dict[str, Any] | None = None
    blocks = nseq = 0
    nonfinite: list[int] = []
    i = w
    t0 = time.monotonic()
    while True:
        if total is not None and i >= total:
            break
        if total is None and deadline is not None and time.time() >= deadline:
            break
        if best is not None and nseq >= 8 and rng.random() < 0.35:
            seq = mutate(best["sequence"], rng)
        else:
            seq = generate(seed, model, i)
        worst, at, per, finite = run_sequence(rig, seq)
        blocks += len(per)
        nseq += 1
        if not finite:
            nonfinite.append(i)
        if best is None or worst > best["instructions"]:
            best = {
                "instructions": worst,
                "block": at,
                "index": i,
                "sequence": seq,
                "block_instructions": per,
            }
        i += jobs
    return {
        "worker": w,
        "best": best,
        "blocks": blocks,
        "sequences": nseq,
        "nonfinite_sequences": nonfinite,
        "wall_s": time.monotonic() - t0,
    }


def fuzz(
    model: int,
    seed: int = 0,
    budget_s: float = 180.0,
    jobs: int = 1,
    sequences: int | None = None,
    mode: int = 1,
) -> dict[str, Any]:
    """The search; the result dict (see the module docstring). SEQUENCES bounds the count instead of the time."""
    t0 = time.monotonic()
    # the rig takes about 8 s to build in each worker: the budget counts from when the workers start sequences
    deadline = None if sequences is not None else time.time() + budget_s + 10
    tasks = [(seed, model, w, jobs, deadline, sequences, mode) for w in range(jobs)]
    if jobs == 1:
        parts = [_worker(tasks[0])]
    else:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            parts = pool.map(_worker, tasks)
    bests = [p["best"] for p in parts if p["best"] is not None]
    best = max(bests, key=lambda b: (b["instructions"], -b["index"]))
    blocks = sum(p["blocks"] for p in parts)
    wall = time.monotonic() - t0
    step = best["sequence"]["steps"][best["block"]]
    return {
        "model": model,
        "mode": mode,
        "seed": seed,
        "jobs": jobs,
        "sequences": sum(p["sequences"] for p in parts),
        "blocks": blocks,
        "worst": {
            "instructions": best["instructions"],
            "block": best["block"],
            "sequence_index": best["index"],
            "step": step,
            "confirm_knobs": confirm_knobs(step),
            **compare(model, best["instructions"]),
        },
        "sequence": best["sequence"],
        "block_instructions": best["block_instructions"],
        "nonfinite_sequences": sorted(
            i for p in parts for i in p["nonfinite_sequences"]
        ),
        "timing": {
            "wall_s": round(wall, 1),
            "blocks_per_minute": round(60 * blocks / max(wall, 1e-9), 1),
            "workers": [round(p["wall_s"], 1) for p in parts],
        },
    }


def save(res: dict[str, Any]) -> Path:
    """Write the result under a run id (model, seed, jobs, sequences reached, UTC time) and never over an earlier file:
    the figure that enabled a model must stay on disk (a later run under load finds a different worst)."""
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    stem = (
        f"m{res['model']}-seed{res['seed']}-j{res['jobs']}-n{res['sequences']}-{stamp}"
    )
    text = json.dumps(res, indent=1, sort_keys=True)
    for k in range(1000):
        path = OUT / (f"{stem}.json" if k == 0 else f"{stem}.{k}.json")
        try:
            with path.open("x") as f:
                f.write(text)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"no free file name for {stem}")


def summary(res: dict[str, Any]) -> str:
    w = res["worst"]
    s = w["step"]
    rec = (
        f"recorded {w['recorded']:,} ({w['delta_pct']:+.2f} %)"
        if w["recorded"]
        else "no recorded worst"
    )
    k = ",".join(f"{n}={v:#x}" for n, v in w["confirm_knobs"].items())
    t = res["timing"]
    return "\n".join(
        [
            f"PLAITS model {res['model']} fuzz (seed {res['seed']}, {res['sequences']} sequences, "
            f"{res['blocks']:,} blocks, {t['blocks_per_minute']:,.0f} blocks/min, {t['wall_s']:.0f} s)",
            f"  worst block {w['instructions']:,} instructions, {rec}; "
            f"{'NEEDS an emulator confirmation' if w['needs_confirm'] else 'within the recorded worst'}",
            f"  at sequence {w['sequence_index']} block {w['block']}: note {s['note']} velocity {s['velocity']} "
            f"TUNE {s['tune']:#x} HARM {s['harmonics']:#x} TIMB {s['timbre']:#x} MORPH {s['morph']:#x} "
            f"MIX {s['mix']:#x} trig {s['trig']}",
            f"  confirm with: --knobs {k}",
        ]
        + (
            [f"  NON-FINITE output in sequences {res['nonfinite_sequences']}"]
            if res["nonfinite_sequences"]
            else []
        )
    )


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--seed", type=int, default=0, help="fuzz: the seed (default 0)")
    ap.add_argument(
        "--budget-seconds",
        type=float,
        default=180.0,
        help="fuzz: wall-clock budget of the search (default 180)",
    )
    ap.add_argument(
        "--sequences",
        type=int,
        default=None,
        help="fuzz: stop after this many sequences instead of the budget (exactly reproducible)",
    )
    ap.add_argument(
        "--jobs", type=int, default=1, help="fuzz: worker processes (default 1)"
    )
    ap.add_argument(
        "--fuzz-mode",
        type=int,
        choices=(0, 1),
        default=1,
        help="fuzz: 1 = the 48k object (default), 0 = the 96k object",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PLAITS worst-case CPU fuzz (isolated core)"
    )
    ap.add_argument("--model", type=int, required=True)
    add_arguments(ap)
    a = ap.parse_args(argv)
    res = fuzz(a.model, a.seed, a.budget_seconds, a.jobs, a.sequences, a.fuzz_mode)
    print(summary(res))
    print(f"  saved {save(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
