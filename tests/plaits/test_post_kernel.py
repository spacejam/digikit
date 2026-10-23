"""The voice's output-stage kernel (modules/machines/plaits/src/dsp/post_sm.s, plaits_stage_kernel) on the isolated
concrete SHARC+ core: plaits.c built for the DSP (render_engine calls the kernel per sub-block) against plaits.c built
with -DPL_POST_C (selcc's plaits_post, mix loop, plaits_dc_block and gain loop, the code the core digests were
recorded from), bit for bit on the block and on the voice's tail state (OUT, AUX, the DC blocker, both limiter peaks),
for every ported model's OUT/AUX gains (plain and limiter), random sub-block contents with denormals, +-0, Inf and NaN,
random mixes and run-on blocks. The engines are replaced by a stub that returns at once, so the OUT and AUX words the
test seeds are what the output stage sees. The kernel leaves every register outside the scratch set as it found it,
and post_sm.s passes the strict encoding gate, has no SIMD-mode transfer and stores no DAG register through its own
DAG."""

from __future__ import annotations

import random
import struct
from functools import cache

import pytest
from support import plaits_vectors as pv

from tools.patchkit import pk_sharc, pk_toolchains

ROOT = pk_toolchains.ROOT
DSP = ROOT / "modules" / "machines" / "plaits" / "src" / "dsp"
pytestmark = pytest.mark.skipif(
    pk_toolchains.selache_dir() is None,
    reason="selache not built (tools/patchkit/pk_toolchains.py selache)",
)

BASE_SW = 0xB92000
RETURN_SW = 0xBAF100
DATA = 0x200A0000
STACK = 0x200F0000
CALLER_I6 = 0x200F0100
FRAME, DST = 0x200E8000, 0x200E8400
GUARD = 0x5A5A5A5A
SUB = 16
TAIL = 36  # out[16], aux[16], dc.x1, dc.y1, out_peak, aux_peak
PORTED = tuple(
    pv.ENABLED
)  # every enabled model (its OUT/AUX gains: plain and limiter paths)
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
STUB_RETURN = """
.COMPRESS;
    I12 = DM(M7, I6);
    JUMP (M14, I12) (DB);
    RFRAME;
    NOP;
"""
LAYOUT_C = """
#include "plaits.h"
const int pl_t_voice_size = (int)sizeof(PlaitsVoice);
const int pl_t_out = (int)sizeof(PlaitsEngine);
int pl_t_code(void) { return 0; }
"""


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@cache
def built(
    kernel: bool,
) -> tuple[pk_sharc.Linked, list[tuple[int, bytes]], dict[str, int], int]:
    inc = [DSP / "engines", DSP / "lib", DSP]
    defines = () if kernel else ("PL_POST_C",)
    objs = [
        pk_sharc.compile_c(
            (DSP / "plaits.c").read_text(), include_dirs=inc, defines=defines
        ),
        pk_sharc.compile_c(LAYOUT_C, include_dirs=inc),
    ]
    if kernel:
        objs.insert(0, pk_sharc.assemble((DSP / "post_sm.s").read_text()))
    first = pk_sharc.merge(objs)
    missing = sorted(
        {
            r.symbol
            for r in first.relocs
            if r.symbol not in first.symbols and r.symbol not in first.data_symbols
        }
    )
    # every engine function plaits.c calls: one stub that returns at once
    stub = ".SECTION/PM seg_pmco;\n.NOCOMPRESS;\n"
    stub += "".join(f".GLOBAL {s};\n" for s in missing)
    stub += "".join(f"{s}:\n" for s in missing)
    obj = pk_sharc.merge([*objs, pk_sharc.assemble(stub + STUB_RETURN)])
    at, blobs, externs, data_at = DATA, [], {}, {}
    for sec, (blob, align, bss) in obj.data.items():
        at = -(-at // max(align, 8)) * max(align, 8)
        data_at[sec] = at
        blobs.append((at, bytes(len(blob)) if bss else blob))
        at += len(blob)
    for name, (sec, off, _g) in obj.data_symbols.items():
        externs[name] = data_at[sec] + off
    left = {
        r.symbol
        for r in obj.relocs
        if r.symbol not in obj.symbols and r.symbol not in externs
    }
    assert not left, left
    linked = pk_sharc.link(obj, BASE_SW, externs)
    pk_sharc.verify(obj, linked, externs)
    return linked, blobs, externs, -(-at // 64) * 64 + 0x100


class Core:
    def __init__(self, kernel: bool) -> None:
        from tools.sharc_dev import sharc_run as sr
        from tools.sharc_dev import sharc_trace as st
        from tools.sharc_dev.sharcldr import LoadedMemory

        self.st = st
        self.linked, blobs, self.externs, self.voice = built(kernel)
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
        self.size = self.peek(self.externs["pl_t_voice_size."])
        self.out_off = self.peek(self.externs["pl_t_out."])
        # selcc's offsetof gives 0: the tail follows the engine union, in plaits.h's order (out[16], aux[16], dc,
        # out_peak, aux_peak, accent, model, armed), as selcc's own byte offsets in render_engine show
        self.model_off = self.out_off + 4 * 37
        self.armed_off = self.out_off + 4 * 38
        assert self.voice + self.size < FRAME, (hex(self.voice), self.size)
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


def rand_tail(rng: random.Random, special: float) -> list[int]:
    def sample() -> int:
        if rng.random() < special:
            return bits(rng.choice(SPECIAL))
        return bits(rng.uniform(-1.0, 1.0) * rng.choice([1.0, 0.5, 0.01, 3.0, 40.0]))

    words = [sample() for _ in range(2 * SUB)]
    words += [bits(rng.uniform(-2.0, 2.0)), bits(rng.uniform(-2.0, 2.0))]  # dc
    words += [
        bits(rng.choice([0.5, rng.uniform(0.0, 0.99), rng.uniform(1.0, 30.0), 1.0]))
        for _ in range(2)
    ]  # peaks
    return words


def frame(rng: random.Random, model: int) -> list[int]:
    def knob() -> float:
        r = rng.random()
        return (
            0.0 if r < 0.15 else (32512.0 if r < 0.3 else float(rng.randint(0, 32512)))
        )

    return [
        bits(float(rng.randint(0, 127) << 8)),
        bits(knob()),
        bits(float(model << 8)),
        bits(knob()),
        bits(knob()),
        bits(knob()),
        bits(knob()),
        bits(float(rng.randint(0, 127) << 8)),
        1,
        0,  # PlaitsFrame.released: no DT2 note-off pulse (DX7 only)
    ]


def seed_voice(core: Core, model: int, tail: list[int]) -> None:
    core.put(core.voice, [0] * (core.size // 4), guard=0)
    core.poke(core.voice + core.model_off, model)
    core.poke(core.voice + core.armed_off, 1)
    core.put(core.voice + core.out_off, tail, guard=0)


def render(
    core: Core, fr: list[int], regs: dict[str, int] | None = None
) -> tuple[list[int], ...]:
    core.put(FRAME, fr, guard=0)
    core.put(DST, [GUARD ^ 3] * 32)
    core.call("plaits_render.", [core.voice, FRAME, 0, DST, 32, 48000], regs)
    return core.get(DST, 32), core.get(core.voice + core.out_off, TAIL, guard=0)


@pytest.fixture(scope="module")
def pair() -> tuple[Core, Core]:
    return Core(True), Core(False)


@pytest.mark.parametrize("model", PORTED)
def test_block_equals_compiled_c(pair: tuple[Core, Core], model: int) -> None:
    ker, ref = pair
    rng = random.Random(1000 + model)
    for i in range(6):
        tail, fr = rand_tail(rng, 0.0 if i % 2 else 0.1), frame(rng, model)
        got = []
        for core in (ker, ref):
            seed_voice(core, model, tail)
            got.append(render(core, fr))
        assert got[0] == got[1], (model, i)


def test_run_on_blocks(pair: tuple[Core, Core]) -> None:
    """The limiter models carry their peaks and the DC blocker its state from block to block; OUT and AUX re-seeded
    per block (the stub engine leaves them), loud enough to drive the limiter above 1 and back."""
    ker, ref = pair
    rng = random.Random(5)
    for model in (7, 9, 12, 0):
        tail = rand_tail(rng, 0.0)
        for core in (ker, ref):
            seed_voice(core, model, tail)
        for blk in range(12):
            loud = rng.choice([0.05, 1.0, 8.0, 60.0])
            io = [bits(rng.uniform(-loud, loud)) for _ in range(2 * SUB)]
            fr = frame(rng, model)
            got = []
            for core in (ker, ref):
                core.put(core.voice + core.out_off, io, guard=0)
                got.append(render(core, fr))
            assert got[0] == got[1], (model, blk)


def test_kernel_cost(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    rng = random.Random(3)
    for model in (0, 12):
        tail, fr = rand_tail(rng, 0.0), frame(rng, model)
        n = []
        for core in (ker, ref):
            seed_voice(core, model, tail)
            render(core, fr)
            n.append(core.instructions)
        print(model, n)
        assert n[0] < n[1], n


@pytest.mark.parametrize("seed", [11, 12])
def test_render_preserves_the_dsp_state(pair: tuple[Core, Core], seed: int) -> None:
    """plaits_render with the kernel from the hook's registers plus random values elsewhere (non-zero L and B on every
    DAG but the C stack's, M6 != 1 on the second seed): every register outside the scratch set, every MODE1/MODE2
    bit and the stack depths are as before, and the result equals the C's."""
    ker, ref = pair
    rng = random.Random(seed)
    model = 12 if seed % 2 else 9
    tail, fr = rand_tail(rng, 0.0), frame(rng, model)
    seed_voice(ref, model, tail)
    want = render(ref, fr)
    seed_voice(ker, model, tail)
    regs: dict[str, int] = dict(HOOK)
    for p in range(16):
        regs[f"R{p}"] = rng.getrandbits(32)
        regs[f"S{p}"] = rng.getrandbits(32)
    for i in (0, 1, 2, 3, 5, 8, 9, 10, 11, 13, 14, 15):
        regs[f"I{i}"] = 0x200C0000 + 4 * rng.randrange(1000)
        regs[f"B{i}"] = 0x200C0000 + 4 * rng.randrange(1000)
        regs[f"L{i}"] = 4 * rng.randrange(1, 64)
    regs["B0"], regs["L0"] = ker.voice + ker.out_off, 8
    regs["B1"], regs["L1"] = DST, 8
    for m in (0, 1, 2, 3, 4, 8, 9, 10, 11, 12):
        regs[f"M{m}"] = rng.randrange(-64, 64) & 0xFFFFFFFF
    if seed % 2 == 0:
        regs["M6"] = 3
    for u in ("USTAT1", "USTAT2", "USTAT3", "USTAT4", "PX1", "PX2"):
        regs[u] = rng.getrandbits(32)
    regs["L4"] = 0  # the C itself relies on L4 = 0
    got = render(ker, fr, regs)
    after = {n: ker._val(ker.runner.state, c) for n, c in ker.st.UREG_CODES.items()}
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
    assert got == want


def test_kernel_passes_the_strict_encoding_gate() -> None:
    """Every instruction of post_sm.s has a precedent in stock code that belongs to a function (or in
    encoding_proven.toml), no data transfer runs in SIMD mode and no DAG register is stored through its own DAG."""
    import sqlite3

    from tools.patchkit import pk_encoding_gate as gate

    linked = pk_sharc.link(
        pk_sharc.assemble((DSP / "post_sm.s").read_text()), BASE_SW, {}
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
