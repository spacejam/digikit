"""Easy-wins finder for SHARC+ DSP code: profile a routine, report ranked optimisation opportunities.

    uv run python -m tools.patchkit.pk_dsp_waste --model N [--seed S] [--sequence FILE] [--top 20]
    uv run python -m tools.patchkit.pk_dsp_waste --stock [--image dt2-1.17] [--profile TSV|none] [--entry 0x1c642a]

`--model N`: the PLAITS routine on the isolated SHARC core (tools/patchkit/plaits_rig.py). The sequence is
deterministic: the fuzz worst of the model from out/runs/plaits-cpu/fuzz/mN-seed*.json (the largest, or `--seed S`),
else sequence 0 of the fixed seed 0 of tools/patchkit/pk_plaits_fuzz.py. The worst counted block of the sequence is
profiled instruction by instruction (counts per pc and adjacent pairs).

`--stock`: the stock voice renderer region of the image's section 7 blob: the call/jump closure of the voice loop
(FUN_1c642a, which calls the renderer 0x1c4ecf), from the program database out/sharcdb/IMAGE.sqlite. Dynamic when an
engine profile (SHARC_CYCMIX tsv, tools/sharc_dev/sharc_cpi.py) exists: `--profile` (default
out/runs/final-fix/cpu/stock-2500.tsv), executing the stock firmware on the emulator is not cheap, so no run is
started here. Without a profile the counts are loop-nesting weights (literal trip counts, else 8 per loop level, capped) and
the report says so.

Detectors and their assumptions: tools/patchkit/dsp_waste.py. The full result goes to
out/runs/dsp-waste/<target>.json. Observations only: nothing here changes DSP code.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from tools.patchkit import dsp_waste as W
from tools.patchkit import pk_toolchains

ROOT = pk_toolchains.ROOT
FUZZ = ROOT / "out" / "runs" / "plaits-cpu" / "fuzz"
OUT = ROOT / "out" / "runs" / "dsp-waste"
STOCK_TSV = ROOT / "out" / "runs" / "final-fix" / "cpu" / "stock-2500.tsv"
SHARCDB = ROOT / "out" / "sharcdb"
STOCK_ENTRIES = (
    0x1C642A,
)  # the stock voice loop; its closure includes the renderer 0x1c4ecf and 0x1c71ec


# -- target 1: PLAITS on the isolated core


def pick_sequence(
    model: int, seed: int | None
) -> tuple[dict[str, Any], int | None, str]:
    """(sequence, worst block index or None, where it came from)."""
    files = sorted(FUZZ.glob(f"m{model}-seed*.json"))
    if seed is not None:
        files = [f for f in files if f.name == f"m{model}-seed{seed}.json"]
    best: tuple[int, Path, dict[str, Any]] | None = None
    for f in files:
        d = json.loads(f.read_text())
        n = d["worst"]["instructions"]
        if best is None or n > best[0]:
            best = (n, f, d)
    if best is not None:
        return (
            best[2]["sequence"],
            best[2]["worst"]["block"],
            str(best[1].relative_to(ROOT)),
        )
    from tools.patchkit import pk_plaits_fuzz as F

    s = 0 if seed is None else seed
    return F.generate(s, model, 0), None, f"generated: seed {s}, sequence 0"


def _apply(rig: Any, s: dict[str, int]) -> None:
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


def worst_block(rig: Any, seq: dict[str, Any]) -> int:
    from tools.patchkit import pk_plaits_fuzz as F

    return int(F.run_sequence(rig, seq)[1])


def profile_block(
    rig: Any, seq: dict[str, Any], at: int
) -> tuple[collections.Counter, collections.Counter, int]:
    """Replay SEQ on a never-triggered voice and step block AT instruction by instruction."""
    counts: collections.Counter[int] = collections.Counter()
    pairs: collections.Counter[tuple[int, int]] = collections.Counter()
    prev: list[int | None] = [None]

    def hook(state: Any) -> None:
        pc = state.pc_sw
        counts[pc] += 1
        if prev[0] is not None:
            pairs[(prev[0], pc)] += 1
        prev[0] = pc

    rig.clear_voice(0)
    rig.active(0)
    rig.trig(0, False)
    rig.instructions.clear()
    for i, s in enumerate(seq["steps"][: at + 1]):
        _apply(rig, s)
        rig.block(0, on_step=hook if i == at else None)
    # the hook also saw the breakpoint the block stopped at, which was not executed
    last = prev[0]
    assert last is not None
    counts[last] -= 1
    if counts[last] == 0:
        del counts[last]
    for k in [k for k in pairs if k[1] == last]:
        pairs[k] -= 1
        if pairs[k] == 0:
            del pairs[k]
        break
    return counts, pairs, rig.instructions[-1]


def decode_range(rig: Any, base: int, nwords: int) -> list[W.Op]:
    from collections import Counter

    from tools.sharc_dev import sharcfn

    ops: list[W.Op] = []
    pc, end = base, base + nwords
    while pc < end:
        insn = rig.runner._decode(pc)
        if insn.kind == "unknown" or not insn.length_bytes:
            pc += 1
            continue
        text = sharcfn.render_instruction(pc, insn, rig.mem, set(), Counter(), [])[0]
        ops.append(W.build_op(pc, insn.length_bytes, insn.type_name, insn.fields, text))
        pc += insn.length_bytes // 2
    return ops


def plaits_program(
    rig: Any, counts: dict[int, float], pairs: dict[tuple[int, int], int]
) -> W.Program:
    ops: list[W.Op] = []
    syms: list[tuple[int, str]] = []
    for linked in (rig.core, rig.pl):
        n = len(linked.code_boot) // 2
        ops += decode_range(rig, linked.base_sw, n)
        syms += [
            (a, nm)
            for nm, a in linked.symbols.items()
            if linked.base_sw <= a < linked.base_sw + n
        ]
    have = {o.pc for o in ops}
    for (
        pc
    ) in counts:  # executed outside the two images (the displaced stock instruction)
        if pc not in have:
            ops += decode_range(rig, pc, 1)
    funcs = [(a, n) for a, n in syms if not n.startswith(".L")]
    return W.Program(ops, dict(counts), dict(pairs), funcs=funcs, labels=syms)


def run_plaits(
    model: int, seed: int | None, mode: int, seq_file: Path | None
) -> tuple[W.Program, dict[str, Any]]:
    from tools.patchkit.plaits_rig import PlaitsRig

    if seq_file is not None:
        d = json.loads(seq_file.read_text())
        seq, at, src = (
            d.get("sequence", d),
            d.get("worst", {}).get("block"),
            str(seq_file),
        )
    else:
        seq, at, src = pick_sequence(model, seed)
    rig = PlaitsRig(mode)
    if at is None:
        at = worst_block(rig, seq)
    counts, pairs, n = profile_block(rig, seq, at)
    assert n == sum(counts.values()), (n, sum(counts.values()))
    prog = plaits_program(rig, dict(counts), dict(pairs))
    meta = {
        "target": f"plaits-m{model}",
        "model": model,
        "mode": mode,
        "sequence_source": src,
        "block": at,
        "block_instructions": n,
        "weights": "dynamic (concrete core, one block, per instruction)",
    }
    return prog, meta


# -- target 2: the stock renderer region


def _closure(db: sqlite3.Connection, image: str, roots: list[int]) -> set[int]:
    seen: set[int] = set()
    todo = list(roots)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for (t,) in db.execute(
            "select distinct to_function from edges where image=? and from_function=? "
            "and kind in ('call','jump','cond_jump') and to_function is not null",
            (image, f),
        ):
            todo.append(t)
    return seen


def run_stock(
    image: str, profile: Path | None, entries: tuple[int, ...]
) -> tuple[W.Program, dict[str, Any]]:
    db = sqlite3.connect(f"file:{SHARCDB / (image + '.sqlite')}?mode=ro", uri=True)
    region = _closure(db, image, list(entries))
    ph = ",".join("?" * len(region))
    args = (image, *sorted(region))
    funcs = [
        (e, n)
        for e, n in db.execute(
            f"select entry_sw, name from functions where image=? and entry_sw in ({ph})",
            args,
        )
    ]
    delayed = {
        s
        for (s,) in db.execute(
            f"select from_sw from edges where image=? and delayed=1 and from_function in ({ph})",
            args,
        )
    }
    ops: list[W.Op] = []
    for sw, width, form, fields, text in db.execute(
        f"select sw, width, form, fields, mnemonic from insn where image=? and aligned=1 "
        f"and mnemonic is not null and function_sw in ({ph})",
        args,
    ):
        op = W.build_op(sw, width, form, json.loads(fields), text)
        if op.branch and sw in delayed:
            op.delayed = True
        ops.append(op)
    in_code = [
        v
        for (v,) in db.execute(
            "select distinct value from literals where image=? and in_code_range=1",
            (image,),
        )
    ]
    db.close()
    meta: dict[str, Any] = {
        "target": "stock",
        "image": image,
        "region_entries": ["%#x" % e for e in entries],
        "region_functions": len(region),
    }
    if profile is not None and profile.is_file():
        from tools.sharc_dev import sharc_cpi

        prof = sharc_cpi.Profile(profile)
        have = {o.pc for o in ops}
        counts = {
            pc: float(n)
            for pc, n in prof.counts.items()
            if pc in have and not prof.is_idle(pc)
        }
        pairs = {k: n for k, n in prof.pairs.items() if k[0] in have and k[1] in have}
        total = sum(n for pc, n in prof.counts.items() if not prof.is_idle(pc))
        meta.update(
            profile=str(profile.relative_to(ROOT)),
            weights="dynamic (engine profile of the stock firmware, whole run)",
            profile_total_instructions=total,
            region_share_pct=round(100 * sum(counts.values()) / total, 1),
        )
        prog = W.Program(ops, counts, pairs, funcs=funcs, extra_leaders=in_code)
    else:
        prog = W.Program(ops, {}, {}, funcs=funcs, extra_leaders=in_code)
        W.apply_static_weights(prog, entries=entries)
        meta["target"] = "stock-static"
        meta["weights"] = (
            "STATIC: loop-nesting weights (literal trip counts, else 8 per level, capped; units are weights, not cycles); no profile was found"
        )
    return prog, meta


# -- report


def build_report(prog: W.Program, meta: dict[str, Any], top: int) -> dict[str, Any]:
    findings = W.run_all(prog)
    instrs = prog.instructions()
    cycles = prog.cycles()
    return {
        **meta,
        "instructions": instrs,
        "estimated_cycles": cycles,
        "cycles_note": W.ASSUMPTIONS["cycles"],
        "class_mix": W.class_mix(prog),
        "functions": W.function_profile(prog, 30),
        "loops": W.loop_profile(prog, 30),
        "detector_totals": W.summarize(findings),
        "totals_note": "detectors overlap (one instruction can be claimed by several): do not add across detectors",
        "assumptions": W.ASSUMPTIONS,
        "top_groups": W.group_findings(findings)[: max(top, 20)],
        "top": [f.to_json() for f in findings[:top]],
        "findings": [f.to_json() for f in findings],
    }


def summary(rep: dict[str, Any], top: int) -> str:
    inst, cyc = rep["instructions"], rep["estimated_cycles"]
    lines = [
        f"{rep['target']}: {inst:,.0f} instructions, ~{cyc:,.0f} cycles (lower bound); weights: {rep['weights']}",
        "  mix: "
        + ", ".join(
            f"{k} {100 * v / max(inst, 1):.1f}%"
            for k, v in list(rep["class_mix"].items())[:7]
        ),
    ]
    for f in rep["functions"][:5]:
        lines.append(f"  fn {f['function']:<28} {f['share_pct']:5.1f}% of cycles")
    for lp in rep["loops"][:3]:
        lines.append(
            f"  loop {lp['kind']} {lp['function']} {lp['span']} depth {lp['depth']} "
            f"{lp['share_pct']}% of instrs, trip {lp['avg_trip']}"
        )
    lines.append(
        f"  top {top} opportunities, grouped by detector and function (estimated cycles saved of {cyc:,.0f}):"
    )
    for i, g in enumerate(rep["top_groups"][:top], 1):
        lines.append(
            f"  {i:2d}. {g['saving_cycles']:>10,.0f} {100 * g['saving_cycles'] / max(cyc, 1):4.1f}% "
            f"{g['detector']:<22} {g['confidence']:<6} {g['sites']:>4} sites in {g['function'][:30]}; "
            f"e.g. {g['best_site'][:44]}: {g['note'][:70]}"
        )
    lines.append(
        "  by detector: "
        + ", ".join(
            f"{k} {v['saving_cycles']:,.0f} ({v['findings']})"
            for k, v in rep["detector_totals"].items()
        )
    )
    return "\n".join(lines)


def save(rep: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{rep['target']}.json"
    path.write_text(json.dumps(rep, indent=1, sort_keys=True))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ranked easy-wins finder for SHARC+ code")
    ap.add_argument("--model", type=int, help="PLAITS model number (isolated core)")
    ap.add_argument(
        "--stock", action="store_true", help="the stock voice renderer region"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="plaits: the fuzz seed to take (default: the worst)",
    )
    ap.add_argument(
        "--sequence",
        type=Path,
        default=None,
        help="plaits: a fuzz result JSON or a bare sequence",
    )
    ap.add_argument(
        "--mode",
        type=int,
        choices=(0, 1),
        default=1,
        help="plaits: 1 = 48k object (default)",
    )
    ap.add_argument(
        "--image", default="dt2-1.17", help="stock: the program database image"
    )
    ap.add_argument(
        "--profile",
        default=str(STOCK_TSV),
        help="stock: engine profile TSV, or 'none' for static",
    )
    ap.add_argument(
        "--entry",
        default=",".join("%#x" % e for e in STOCK_ENTRIES),
        help="stock: region root functions",
    )
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)
    if (a.model is None) == (not a.stock):
        ap.error("give exactly one of --model N and --stock")
    if a.stock:
        prof = None if a.profile == "none" else Path(a.profile)
        prog, meta = run_stock(
            a.image, prof, tuple(int(x, 0) for x in a.entry.split(","))
        )
    else:
        prog, meta = run_plaits(a.model, a.seed, a.mode, a.sequence)
    rep = build_report(prog, meta, a.top)
    print(summary(rep, a.top))
    print(f"  saved {save(rep)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
