"""Where the stall cycles go: cycles against instructions per function and loop, for the PLAITS routine.

    uv run python -m tools.patchkit.pk_dsp_stalls --model N [--model M ...] [--seed S] [--top 8] [--loads]

A sibling of pk_dsp_waste (same block: the fuzz worst of the model, profiled on the isolated core,
tools/patchkit/plaits_rig.py). Per executed instruction it charges:

- the adjacent-pair stalls of tools/sharc_dev/sharc_timing.py, by kind: `dag` (a DAG register written by the
  previous instruction and used as an address, 4 cycles), `fwd` (a float result read by the next compute, 1),
  `mov` (a float or multiplier result moved out of the PE, 1), and `branch` (a taken branch, BTB hit assumed);
- instruction-cache misses for code in L2 (0xb80000..0xc0c000 short words) and data-cache misses for L2 and
  external data, each a 2-way 16 KB cache of 64-byte lines with LRU replacement (the cycle model's geometry,
  packages/emulator/sharc/src/cycmix.rs), at the effective address of every access. `cold` starts the block with
  empty caches; `warm` replays the block's own accesses once first (nothing else evicts). The emulator run sits
  between the two (stock code and data share the caches).

Cycles: nominal = instructions + pair stalls + 24 per miss; high = instructions + pair stalls + 48 per code miss
+ 96 per data miss (a dirty write-back each), cold caches (L2_FILL of sharc_timing.py; the manual gives no L2
latency). The manual's loads have no load-to-use stall (PRM Tables 4-35..4-43; "cache hits do not stall the
pipe", p. 8-7) and a miss stalls the core whatever the distance to the use: `--loads` prints the load-to-use
distance of each hot load anyway, so the claim can be checked against the code.

Observations only. JSON: out/runs/dsp-stalls/plaits-mN.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import Any

from tools.patchkit import dsp_waste as W
from tools.patchkit import pk_dsp_waste as D
from tools.sharc_dev import sharc_timing as T

OUT = D.ROOT / "out" / "runs" / "dsp-stalls"
CODE_L2 = (0xB80000, 0xC0C000)
KINDS = ("dag", "fwd", "mov", "branch")


class Cache:
    """2-way set-associative, LRU, 64-byte lines (cycmix.rs SetAssoc)."""

    def __init__(self, nbytes: int = 16 * 1024, line: int = 64) -> None:
        self.sets = nbytes // line // 2
        self.tags = [[-1, -1] for _ in range(self.sets)]
        self.mru = [0] * self.sets

    def access(self, line: int) -> bool:
        s = line % self.sets
        t = self.tags[s]
        if t[0] == line:
            self.mru[s] = 0
            return False
        if t[1] == line:
            self.mru[s] = 1
            return False
        v = 1 - self.mru[s]
        t[v] = line
        self.mru[s] = v
        return True


def far_line(addr: int) -> int | None:
    """A 64-byte line id for an L2 or external data address, None for L1, MMRs and anything else (the regions of
    cycmix.rs `bucket`). L2 in its normal-word (4 bytes per address) and short-word (2) spaces maps onto the byte
    space, so one line is one line whichever space reaches it."""
    if (
        0x20000000 <= addr < 0x20118000 or addr >= 0x60000000
    ):  # L2 and external, byte space
        return addr >> 6
    if 0x8000000 <= addr < 0x8046000:  # L2, normal word
        return (0x20000000 + 4 * (addr - 0x8000000)) >> 6
    if 0xB80000 <= addr < 0xC0C000:  # L2, short word
        return (0x20000000 + 2 * (addr - 0xB80000)) >> 6
    if (
        0x4000000 <= addr < 0x8000000 or 0x10000000 <= addr < 0x18000000
    ):  # external, word spaces
        return (1 << 40) | (addr >> 4)
    return None


def byte_space(addr: int) -> bool:
    return 0x20000000 <= addr < 0x30000000 or addr >= 0x60000000


def _ureg(state: Any, codes: dict[str, int], name: str) -> int | None:
    v = state.uregs.get(codes[name])
    val = getattr(v, "value", None)
    return None if val is None else val & 0xFFFFFFFF


def profile(
    rig: Any, seq: dict[str, Any], at: int, ops_of: dict[int, W.Op]
) -> dict[str, Any]:
    """Replay SEQ and step block AT: the pc trace and, per step, the effective addresses of its data accesses."""
    from tools.sharc_dev import sharc_trace as st

    codes = st.UREG_CODES
    trace: list[int] = []
    addrs: list[tuple[int, ...]] = []
    decoded: dict[int, W.Op | None] = {}

    def op_at(pc: int) -> W.Op | None:
        if pc not in decoded:
            o = ops_of.get(pc)
            if o is None:
                got = D.decode_range(rig, pc, 1)
                o = got[0] if got else None
            decoded[pc] = o
        return decoded[pc]

    def hook(state: Any) -> None:
        pc = state.pc_sw
        trace.append(pc)
        o = op_at(pc)
        got: list[int] = []
        if o is not None:
            for m in o.mem:
                if m.base is None:
                    if m.key is not None and isinstance(m.key[2], int):
                        got.append(m.key[2])
                    continue
                i = _ureg(state, codes, m.base)
                if i is None:
                    continue
                scale = 4 if byte_space(i) else 1
                if m.off is not None:
                    got.append((i + scale * m.off) & 0xFFFFFFFF)
                elif m.key is not None and isinstance(m.key[2], tuple):
                    mv = _ureg(state, codes, "M%d" % m.key[2][1])
                    if mv is not None:
                        mv = mv - (1 << 32) if mv & 0x80000000 else mv
                        got.append((i + scale * mv) & 0xFFFFFFFF)
                else:
                    got.append(i)
        addrs.append(tuple(got))

    rig.clear_voice(0)
    rig.active(0)
    rig.trig(0, False)
    rig.instructions.clear()
    for i, s in enumerate(seq["steps"][: at + 1]):
        D._apply(rig, s)
        rig.block(0, on_step=hook if i == at else None)
    # the hook also saw the breakpoint the block stopped at, which was not executed
    trace.pop()
    addrs.pop()
    assert len(trace) == rig.instructions[-1], (len(trace), rig.instructions[-1])
    return {"trace": trace, "addrs": addrs}


def simulate(
    trace: list[int], addrs: list[tuple[int, ...]], ops: dict[int, W.Op], warm: bool
) -> dict[str, Any]:
    icache, dcache = Cache(), Cache()
    imiss: collections.Counter[int] = collections.Counter()
    dmiss: collections.Counter[int] = collections.Counter()
    dacc: collections.Counter[int] = collections.Counter()
    for rnd in range(2 if warm else 1):
        last_line = -1
        for pc, aa in zip(trace, addrs, strict=True):
            count = rnd == (1 if warm else 0)
            if CODE_L2[0] <= pc < CODE_L2[1]:
                o = ops.get(pc)
                nb = o.nw * 2 if o is not None else 2
                first, last = pc * 2 // 64, (pc * 2 + nb - 1) // 64
                if first != last_line and icache.access(first) and count:
                    imiss[pc] += 1
                if last != first and icache.access(last) and count:
                    imiss[pc] += 1
                last_line = last
            else:
                last_line = -1
            for a in aa:
                ln = far_line(a)
                if ln is None:
                    continue
                if count:
                    dacc[pc] += 1
                if dcache.access(ln) and count:
                    dmiss[pc] += 1
    return {"imiss": imiss, "dmiss": dmiss, "dacc": dacc}


def pair_stalls(prog: W.Program) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = collections.defaultdict(
        lambda: dict.fromkeys(KINDS, 0.0)
    )
    tins = {pc: W.to_timing(o) for pc, o in prog.ops.items()}
    for (a, b), n in prog.pairs.items():
        ia, ib = tins.get(a), tins.get(b)
        if ia is None or ib is None:
            continue
        for k, f in (
            ("dag", T.dag_use),
            ("fwd", T.float_forward),
            ("mov", T.float_to_move),
        ):
            c = f(ia, ib)[1]
            if c:
                out[b][k] += c * n
    for pc, o in prog.ops.items():
        if o.branch in ("jump", "cond_jump", "call", "return", "indirect"):
            s = W.BRANCH_STALL[1 if o.delayed else 0]
            if s:
                out[pc]["branch"] += s * prog.taken(o)
    return out


def load_use(
    trace: list[int], ops: dict[int, W.Op], horizon: int = 12
) -> dict[int, collections.Counter[int]]:
    """Per load pc: how many instructions after it its destination register is first read (horizon = not seen)."""
    out: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    n = len(trace)
    for i, pc in enumerate(trace):
        o = ops.get(pc)
        if o is None:
            continue
        for m in o.mem:
            if m.direction != "load" or m.reg is None or not m.reg.startswith("R"):
                continue
            d = horizon
            for j in range(i + 1, min(n, i + horizon)):
                q = ops.get(trace[j])
                if q is not None and m.reg in q.uses:
                    d = j - i
                    break
                if q is not None and m.reg in q.defs:
                    break
            out[pc][d] += 1
    return out


def cost(n: float, st: dict[str, float], im: float, dm: float) -> tuple[float, float]:
    s = sum(st.values())
    return (
        n + s + T.L2_FILL[1] * (im + dm),
        n + s + T.L2_FILL[2] * im + 2 * T.L2_FILL[2] * dm,
    )


def analyse(
    model: int, seed: int | None, mode: int, top: int, loads: bool
) -> dict[str, Any]:
    from tools.patchkit.plaits_rig import PlaitsRig

    seq, at, src = D.pick_sequence(model, seed)
    rig = PlaitsRig(mode)
    if at is None:
        at = D.worst_block(rig, seq)
    # the static program first (decoded once), then the profiled block
    prog0 = D.plaits_program(rig, {}, {})
    p = profile(rig, seq, at, prog0.ops)
    trace, addrs = p["trace"], p["addrs"]
    counts = collections.Counter(trace)
    pairs = collections.Counter(zip(trace, trace[1:], strict=False))
    prog = D.plaits_program(rig, dict(counts), dict(pairs))
    ops = prog.ops
    st = pair_stalls(prog)
    cold = simulate(trace, addrs, ops, warm=False)
    warm = simulate(trace, addrs, ops, warm=True)

    def group(pcs: list[int]) -> dict[str, Any]:
        n = float(sum(counts[q] for q in pcs))
        s = {k: sum(st[q][k] for q in pcs if q in st) for k in KINDS}
        im = float(sum(cold["imiss"][q] for q in pcs))
        dm = float(sum(cold["dmiss"][q] for q in pcs))
        nom, high = cost(n, s, im, dm)
        return {
            "instructions": n,
            "stalls": {k: round(v) for k, v in s.items()},
            "imiss_cold": im,
            "dmiss_cold": dm,
            "imiss_warm": float(sum(warm["imiss"][q] for q in pcs)),
            "dmiss_warm": float(sum(warm["dmiss"][q] for q in pcs)),
            "far_data_accesses": float(sum(cold["dacc"][q] for q in pcs)),
            "cycles_nominal": round(nom),
            "cycles_high": round(high),
            "stall_cycles_high": round(high - n),
        }

    allpcs = list(counts)
    total = group(allpcs)
    by_fn: dict[str, list[int]] = collections.defaultdict(list)
    for q in allpcs:
        by_fn[prog.func_of(q)[1]].append(q)
    fns = sorted(
        ({"function": f, **group(q)} for f, q in by_fn.items()),
        key=lambda r: -r["stall_cycles_high"],
    )
    loops = []
    for lp in prog.loops():
        if lp.executed <= 0:
            continue
        body = [q for q in allpcs if lp.head <= q <= lp.end]
        loops.append(
            {
                "function": lp.func,
                "kind": lp.kind,
                "span": "%#x..%#x" % (lp.head, lp.end),
                **group(body),
            }
        )
    loops.sort(key=lambda r: -r["stall_cycles_high"])
    rep: dict[str, Any] = {
        "target": f"plaits-m{model}",
        "model": model,
        "sequence_source": src,
        "block": at,
        "total": total,
        "functions": fns[: max(top, 12)],
        "loops": loops[: max(top, 12)],
        "assumptions": __doc__,
    }
    if loads:
        lu = load_use(trace, ops)
        hot = sorted(lu, key=lambda q: -sum(lu[q].values()))[: 3 * top]
        rep["load_use"] = [
            {
                "pc": "%#x" % q,
                "where": prog.where(q),
                "text": ops[q].text[:60],
                "executions": sum(lu[q].values()),
                "distance": dict(sorted(lu[q].items())),
                "far_data_misses_cold": cold["dmiss"][q],
            }
            for q in hot
        ]
    return rep


def summary(rep: dict[str, Any], top: int) -> str:
    t = rep["total"]

    def line(r: dict[str, Any]) -> str:
        s = r["stalls"]
        return (
            f"{r['instructions']:>8,.0f} ins {r['cycles_high']:>8,} high (+{r['stall_cycles_high']:,}: "
            f"dag {s['dag']:,} fwd {s['fwd']:,} mov {s['mov']:,} br {s['branch']:,} "
            f"i$ {r['imiss_cold']:.0f}/{r['imiss_warm']:.0f} d$ {r['dmiss_cold']:.0f}/{r['dmiss_warm']:.0f} "
            f"of {r['far_data_accesses']:.0f})"
        )

    out = [f"{rep['target']} block {rep['block']}: {line(t)}  [misses cold/warm]"]
    for f in rep["functions"][:top]:
        out.append(f"  fn   {f['function'][:34]:<34} {line(f)}")
    for lp in rep["loops"][:top]:
        out.append(
            f"  loop {(lp['function'][:22] + ' ' + lp['span'])[:34]:<34} {line(lp)}"
        )
    for r in rep.get("load_use", [])[: 2 * top]:
        out.append(
            f"  load {r['pc']} {r['text'][:40]:<40} x{r['executions']} dist {r['distance']} d$ {r['far_data_misses_cold']}"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Stall cycles against instructions for PLAITS models"
    )
    ap.add_argument("--model", type=int, action="append", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", type=int, choices=(0, 1), default=1)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument(
        "--loads", action="store_true", help="load-to-use distances of the hot loads"
    )
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in a.model:
        rep = analyse(m, a.seed, a.mode, a.top, a.loads)
        print(summary(rep, a.top))
        path = OUT / f"{rep['target']}.json"
        path.write_text(json.dumps(rep, indent=1, sort_keys=True))
        t = rep["total"]
        rows.append((m, t["instructions"], t["cycles_high"], t["stall_cycles_high"]))
    if len(rows) > 1:
        print("ranking by stall cycles (high bound, cold caches):")
        for m, n, c, s in sorted(rows, key=lambda r: -r[3]):
            print(
                f"  model {m:>2}: {s:>8,} stall cycles, {n:>8,.0f} instructions, {c:>8,} cycles"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
