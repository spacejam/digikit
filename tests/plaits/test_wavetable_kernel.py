"""WavetableOscillator's sample-loop kernel (modules/machines/plaits/src/dsp/lib/wavetable_sm.s, CHORD) on the
isolated concrete SHARC+ core: wavetable_oscillator_render built for the DSP (the kernel) against the same file built
with -DPL_WT_C (selcc's C loop, the code the core digests were recorded from), bit for bit on out[] and the whole
oscillator state, over random states (with denormals, +-0, Inf and NaN in the words that do not index a table),
random parameters (frequencies outside the clamp included), random wave tables and run-on sequences; the kernel leaves
every register outside the scratch set as it found it; and wavetable_sm.s passes the strict encoding gate, has no
SIMD-mode transfer and stores no DAG register through its own DAG."""

from __future__ import annotations

import random
import struct
from functools import cache

import pytest

from tools.patchkit import pk_sharc, pk_toolchains

ROOT = pk_toolchains.ROOT
DSP = ROOT / "modules" / "machines" / "plaits" / "src" / "dsp"
LIB = DSP / "lib"
pytestmark = pytest.mark.skipif(
    pk_toolchains.selache_dir() is None,
    reason="selache not built (tools/patchkit/pk_toolchains.py selache)",
)

BASE_SW = 0xB92000
RETURN_SW = 0xBAF100
DATA = 0x200A0000
STACK = 0x200F0000
CALLER_I6 = 0x200F0100
GUARD = 0x5A5A5A5A
SCRATCH = {f"R{i}" for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12)} | {
    "I4",
    "I12",
    "LCNTR",
    "CURLCNTR",
    "LADDR",
}
RUNNING = {
    "PC",
    "PCSTK",
    "PCSTKP",
    "EMUCLK",
    "EMUCLK2",
    "TCOUNT",
    "FADDR",
    "DADDR",
    "MODE1STK",
}
HOOK = {
    "M5": 0,
    "M6": 1,
    "M7": 0xFFFFFFFF,
    "M13": 0,
    "M14": 1,
    "M15": 0xFFFFFFFF,
    "MODE1": 0x1011800,
    "MODE2": 1,
    "L6": 0,
    "L7": 0,
}
FLAGS = {"ASTATX", "ASTATY", "STKYX", "STKYY"}
ST, OUT, WAVES, OFFS = 0x200D0000, 0x200D0400, 0x200D1000, 0x200D0800
NUM_WAVES, WAVE_LEN, STATE_WORDS = 15, 132, 7


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@cache
def built(kernel: bool) -> tuple[pk_sharc.Linked, list[tuple[int, bytes]]]:
    defines = () if kernel else ("PL_WT_C",)
    objs = [
        pk_sharc.compile_c(
            (LIB / "wavetable_oscillator.c").read_text(),
            include_dirs=[LIB, DSP],
            defines=defines,
        )
    ]
    if kernel:
        objs.insert(0, pk_sharc.assemble((LIB / "wavetable_sm.s").read_text()))
    obj = pk_sharc.merge(objs)
    at, blobs, externs, data_at = DATA, [], {}, {}
    for sec, (blob, align, bss) in obj.data.items():
        at = -(-at // max(align, 8)) * max(align, 8)
        data_at[sec] = at
        blobs.append((at, bytes(len(blob)) if bss else blob))
        at += len(blob)
    for name, (sec, off, _g) in obj.data_symbols.items():
        externs[name] = data_at[sec] + off
    missing = {
        r.symbol
        for r in obj.relocs
        if r.symbol not in obj.symbols and r.symbol not in externs
    }
    assert not missing, missing
    linked = pk_sharc.link(obj, BASE_SW, externs)
    pk_sharc.verify(obj, linked, externs)
    return linked, blobs


class Core:
    def __init__(self, kernel: bool) -> None:
        from tools.sharc_dev import sharc_run as sr
        from tools.sharc_dev import sharc_trace as st
        from tools.sharc_dev.sharcldr import LoadedMemory

        self.st = st
        self.linked, blobs = built(kernel)
        payload = self.linked.code_boot
        blocks = [
            {
                "target_address": pk_sharc.SW_ALIAS_BASE + 2 * BASE_SW,
                "byte_count": len(payload),
                "fill": False,
                "payload_offset": 0,
                "payload_len": len(payload),
                "index": 0,
            }
        ]
        for at, blob in blobs:
            blocks.append(
                {
                    "target_address": at,
                    "byte_count": len(blob),
                    "fill": False,
                    "payload_offset": len(payload),
                    "payload_len": len(blob),
                    "index": len(blocks),
                }
            )
            payload += blob
        self.runner = sr.Runner(
            LoadedMemory(payload, blocks),
            BASE_SW,
            breakpoints=[RETURN_SW],
            approx_recips=True,
        )
        self.instructions = 0

    def poke(self, addr: int, word: int) -> None:
        assert self.st._dm_write(
            self.runner.state, addr, 4, self.st.Const(word & 0xFFFFFFFF)
        )

    def peek(self, addr: int) -> int:
        v = self.st._dm_read(self.runner.state, addr, 4)
        assert v is not None
        return int(v.value) & 0xFFFFFFFF

    def put(self, addr: int, words: list[int], guard: int = 2) -> None:
        for k in range(-guard, len(words) + guard):
            self.poke(addr + 4 * k, words[k] if 0 <= k < len(words) else GUARD)

    def get(self, addr: int, n: int, guard: int = 2) -> list[int]:
        return [self.peek(addr + 4 * k) for k in range(-guard, n + guard)]

    def call(
        self, name: str, args: list[int], regs: dict[str, int] | None = None
    ) -> dict[str, int | None]:
        self.poke(STACK, CALLER_I6)
        self.poke(STACK - 4, RETURN_SW - 1)
        r: dict[str | int, int | str] = {
            "I6": STACK,
            "I7": STACK - 8,
            "M5": 0,
            "M6": 1,
            "M7": -1,
            "M13": 0,
            "M14": 1,
            "M15": -1,
        }
        r.update(regs or {})
        for reg, a in zip(("R4", "R8", "R12"), args[:3], strict=False):
            r[reg] = a & 0xFFFFFFFF
        for k, a in enumerate(args[3:]):
            self.poke(STACK + 4 * (k + 1), a)
        run = self.runner.fresh_call(
            self.linked.symbols[name], regs=r, return_address=RETURN_SW
        )
        st = run.state
        self.before = {n: self._val(st, c) for n, c in self.st.UREG_CODES.items()}
        self.depth_before = (len(st.loops), len(st.status_stack), len(st.call_stack))
        res = run.run(5_000_000)
        assert res.halt.reason == "breakpoint" and res.final_pc_sw == RETURN_SW, (
            res.halt.to_json()
        )
        assert not run.state.loops, "loop stack not empty on return"
        self.runner = run
        st = run.state
        self.depth_after = (len(st.loops), len(st.status_stack), len(st.call_stack))
        self.instructions = res.instructions
        return {n: self._val(st, c) for n, c in self.st.UREG_CODES.items()}

    @staticmethod
    def _val(state: object, code: int) -> int | None:
        v = state.uregs.get(code)  # type: ignore[attr-defined]
        x = None if v is None else getattr(v, "value", None)
        return None if x is None else int(x) & 0xFFFFFFFF


SPECIAL = [
    0.0,
    -0.0,
    1e-39,
    -1e-39,
    3e38,
    -3e38,
    float("inf"),
    float("-inf"),
    float("nan"),
]


def rand_state(rng: random.Random, special: float) -> list[int]:
    def free() -> int:
        if rng.random() < special:
            return bits(rng.choice(SPECIAL))
        return bits(rng.uniform(-1.0, 1.0) * rng.choice([1.0, 0.1, 0.01, 3.0]))

    return [
        bits(rng.uniform(0.0, 0.999)),  # phase_
        bits(
            rng.choice([1e-7, 0.25, rng.uniform(1e-7, 0.25), rng.uniform(1e-5, 0.01)])
        ),  # frequency_
        free(),  # amplitude_
        bits(rng.uniform(0.0, NUM_WAVES - 1.0001)),  # waveform_
        free(),  # lp_
        free(),  # differentiator lp_
        free(),  # previous_
    ]


def rand_params(rng: random.Random) -> tuple[int, int, int]:
    f = rng.choice(
        [rng.uniform(0.0, 0.3), 0.0, -0.01, 0.5, rng.uniform(1e-5, 0.02), 0.25]
    )
    a = rng.choice([0.0, rng.uniform(0.0, 1.0), rng.uniform(-1.0, 1.0)])
    w = rng.choice([0.0, 1.0, rng.random()])
    return bits(f), bits(a), bits(w)


def setup_tables(core: Core, rng: random.Random) -> None:
    core.put(
        WAVES,
        [bits(rng.uniform(-1.0, 1.0)) for _ in range(NUM_WAVES * WAVE_LEN)],
        guard=0,
    )
    core.put(OFFS, [WAVE_LEN * k for k in range(NUM_WAVES)], guard=0)


def render(
    core: Core,
    state: list[int],
    params: tuple[int, int, int],
    out: list[int],
    size: int = 16,
):
    core.put(ST, state)
    core.put(OUT, out)
    f, a, w = params
    core.call(
        "wavetable_oscillator_render.", [ST, f, a, w, WAVES, OFFS, NUM_WAVES, OUT, size]
    )
    return core.get(OUT, size), core.get(ST, STATE_WORDS)


@pytest.fixture(scope="module")
def pair() -> tuple[Core, Core]:
    ker, ref = Core(True), Core(False)
    for core in (ker, ref):
        setup_tables(core, random.Random(42))
    return ker, ref


def test_render_equals_compiled_c_on_random_states(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    rng = random.Random(100)
    for i in range(60):
        special = 0.0 if i % 3 else 0.1
        st, prm = rand_state(rng, special), rand_params(rng)
        size = rng.choice([16, 16, 16, 1, 7, 0])
        out = [bits(rng.uniform(-1.0, 1.0)) for _ in range(size)]
        got, want = render(ker, st, prm, out, size), render(ref, st, prm, out, size)
        assert got == want, i


def test_render_runs_on(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    rng = random.Random(7)
    st = [0] * STATE_WORDS
    for core in (ker, ref):
        core.put(ST, st)
        core.call("wavetable_oscillator_init.", [ST])
    prm = rand_params(rng)
    for blk in range(60):
        if rng.random() < 0.3:
            prm = rand_params(rng)
        outs = []
        for core in (ker, ref):
            core.put(OUT, [0] * 16)
            f, a, w = prm
            core.call(
                "wavetable_oscillator_render.",
                [ST, f, a, w, WAVES, OFFS, NUM_WAVES, OUT, 16],
            )
            outs.append((core.get(OUT, 16), core.get(ST, STATE_WORDS)))
        assert outs[0] == outs[1], blk


def test_kernel_cost(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    rng = random.Random(3)
    st, prm = rand_state(rng, 0.0), rand_params(rng)
    n = []
    for core in (ker, ref):
        render(core, st, prm, [0] * 16)
        n.append(core.instructions)
    print(n)
    assert n[0] < n[1], n


@pytest.mark.parametrize("seed", [11, 12])
def test_render_preserves_the_dsp_state(pair: tuple[Core, Core], seed: int) -> None:
    ker, ref = pair
    rng = random.Random(seed)
    st, prm = rand_state(rng, 0.0), rand_params(rng)
    want = render(ref, st, prm, [0] * 16)
    ker.put(ST, st)
    ker.put(OUT, [0] * 16)
    regs: dict[str, int] = dict(HOOK)
    for p in range(16):
        regs[f"R{p}"] = rng.getrandbits(32)
        regs[f"S{p}"] = rng.getrandbits(32)
    for i in (0, 1, 2, 3, 5, 8, 9, 10, 11, 13, 14, 15):
        regs[f"I{i}"] = 0x200C0000 + 4 * rng.randrange(1000)
        regs[f"B{i}"] = 0x200C0000 + 4 * rng.randrange(1000)
        regs[f"L{i}"] = 4 * rng.randrange(1, 64)
    regs["B0"], regs["L0"] = OUT, 8
    regs["B1"], regs["L1"] = WAVES, 8
    for m in (0, 1, 2, 3, 4, 8, 9, 10, 11, 12):
        regs[f"M{m}"] = rng.randrange(-64, 64) & 0xFFFFFFFF
    if seed % 2 == 0:
        regs["M6"] = 3
    for u in ("USTAT1", "USTAT2", "USTAT3", "USTAT4", "PX1", "PX2"):
        regs[u] = rng.getrandbits(32)
    regs["L4"] = 0
    f, a, w = prm
    after = ker.call(
        "wavetable_oscillator_render.",
        [ST, f, a, w, WAVES, OFFS, NUM_WAVES, OUT, 16],
        regs,
    )
    before = dict(ker.before)
    before["I6"], before["I7"] = CALLER_I6, STACK
    changed = {
        n: (before[n], after[n])
        for n in before
        if n not in SCRATCH | FLAGS | RUNNING and before[n] != after[n]
    }
    assert not changed, changed
    assert ker.depth_after == (
        ker.depth_before[0],
        ker.depth_before[1],
        ker.depth_before[2] - 1,
    )
    assert (ker.get(OUT, 16), ker.get(ST, STATE_WORDS)) == want


def test_kernel_passes_the_strict_encoding_gate() -> None:
    """Every instruction of wavetable_sm.s has a precedent in stock code that belongs to a function (or in
    encoding_proven.toml), no data transfer runs in SIMD mode and no DAG register is stored through its own DAG."""
    import sqlite3

    from tools.patchkit import pk_encoding_gate as gate

    linked = pk_sharc.link(
        pk_sharc.assemble((LIB / "wavetable_sm.s").read_text()), BASE_SW, {}
    )

    class Mem:
        def read_sw(self, pc: int, size: int = 6) -> bytes | None:
            lo = 2 * (pc - BASE_SW)
            data = linked.code_boot
            return data[lo : lo + size] if lo + size <= len(data) else None

    db = ROOT / "out" / "sharcdb" / "dt2-1.17.sqlite"
    if not db.exists():
        pytest.skip("no stock instruction database")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT DISTINCT form, raw FROM insn WHERE aligned = 1 AND confidence = 'confident' "
        "AND function_sw IS NOT NULL"
    ).fetchall()
    con.close()
    strict = {
        gate.shape_of(f, int(r if isinstance(r, str) else r.hex(), 16)) for f, r in rows
    }
    ranges = [(BASE_SW, BASE_SW + len(linked.code_boot) // 2)]
    bad = gate.check_code(
        Mem(),
        ranges,
        strict | gate.proven_shapes(),
        gate.stock_control_words(db) | gate.proven_words(),
    )
    assert not bad, [gate.format_violation(v) for v in bad]
    assert gate.check_simd_memory(Mem(), ranges) == []
    assert gate.check_same_dag(Mem(), ranges) == []
