"""The voice wrapper's per-block path in assembly (modules/machines/plaits/src/dsp/wrap_sm.s: plaits_render at 48 kHz,
render_engine and render_sub) on the isolated concrete SHARC+ core, against plaits.c built with -DPL_WRAP_C (selcc's
plaits_render, the code the core digests were recorded from). Every engine and init function plaits.c calls is a
stub that logs its arguments (and the Params it is given), so the test checks the dispatch, the Params, the trigger
(SWARM, DX7 with and without a note-off, note-ons), note_on, the armed/active/ported tests, zeros, the stage
kernel's block and the voice's tail, bit for bit, at 48 kHz and at 96 kHz (the C). wrap_sm.s passes the strict
encoding gate; the call leaves every register outside the scratch set as it found it."""

from __future__ import annotations

import random
import struct
from functools import cache

import pytest

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
LOG_WORDS = 11
SCRATCH = {f"R{i}" for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12)} | {
    "I4",
    "I12",
    "LCNTR",
    "CURLCNTR",
    "LADDR",
}
RUNNING = {"PC", "PCSTK", "PCSTKP", "EMUCLK", "EMUCLK2", "TCOUNT", "FADDR", "DADDR"}
RUNNING |= {"MODE1STK"}
FLAGS = {"ASTATX", "ASTATY", "STKYX", "STKYY"}
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
SUPPORT_C = """
#include "plaits.h"
const int pl_t_voice_size = (int)sizeof(PlaitsVoice);
const int pl_t_out = (int)sizeof(PlaitsEngine);
int pl_t_log[4096];
int pl_t_logn;
int pl_t_code(void) { return 0; }
"""
# a logging stub per function: its id, R4, R12, the six Params words at R8, an engine's stack arguments
STUB = """
{name}:
    R0 = DM(pl_t_logn.);
    R2 = LSHIFT R0 BY 2;
    R1 = pl_t_log.;
    R1 = R1 + R2;
    I4 = R1;
    R2 = {ident};
    DM(0, I4) = R2;
    DM(1, I4) = R4;
    DM(2, I4) = R12;
{stack}
    I4 = R8;
    R2 = DM(0, I4);
    R3 = DM(1, I4);
    R5 = DM(2, I4);
    R6 = DM(3, I4);
    R7 = DM(4, I4);
    R12 = DM(5, I4);
    I4 = R1;
    DM(3, I4) = R2;
    DM(4, I4) = R3;
    DM(5, I4) = R5;
    DM(6, I4) = R6;
    DM(7, I4) = R7;
    DM(8, I4) = R12;
    R1 = {words};
    R0 = R0 + R1;
    DM(pl_t_logn.) = R0;
.COMPRESS;
    I12 = DM(M7, I6);
    JUMP (M14, I12) (DB);
    RFRAME;
    NOP;
.NOCOMPRESS;
"""

# the engines' two stack arguments (aux, size); other functions have none (their words would be the caller's frame)
STACK_ARGS = """
    R2 = DM(1, I6);
    DM(9, I4) = R2;
    R2 = DM(2, I6);
    DM(10, I4) = R2;"""


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@cache
def built(
    kernel: bool,
) -> tuple[pk_sharc.Linked, list[tuple[int, bytes]], dict[str, int], int]:
    inc = [DSP / "engines", DSP / "lib", DSP]
    defines = () if kernel else ("PL_WRAP_C",)
    asm = (DSP / "post_sm.s").read_text()
    if kernel:
        asm += "\n" + (DSP / "wrap_sm.s").read_text()
    objs = [
        pk_sharc.assemble(asm),
        pk_sharc.compile_c(
            (DSP / "plaits.c").read_text(), include_dirs=inc, defines=defines
        ),
        pk_sharc.compile_c(SUPPORT_C, include_dirs=inc),
    ]
    first = pk_sharc.merge(objs)
    missing = sorted(
        {
            r.symbol
            for r in first.relocs
            if r.symbol not in first.symbols and r.symbol not in first.data_symbols
        }
    )
    missing = [s for s in missing if s != "plaits_voice_mem."]
    stub = (
        ".SECTION/PM seg_pmco;\n.NOCOMPRESS;\n.EXTERN pl_t_log.;\n.EXTERN pl_t_logn.;\n"
    )
    stub += "".join(f".GLOBAL {s};\n" for s in missing)
    stub += "".join(
        STUB.format(
            name=s,
            ident=i + 1,
            words=LOG_WORDS,
            stack=STACK_ARGS if s.endswith("_render.") else "",
        )
        for i, s in enumerate(missing)
    )
    obj = pk_sharc.merge([*objs, pk_sharc.assemble(stub)])
    at, blobs, externs, data_at = DATA, [], {}, {}
    for sec, (blob, align, bss) in obj.data.items():
        at = -(-at // max(align, 8)) * max(align, 8)
        data_at[sec] = at
        blobs.append((at, bytes(len(blob)) if bss else blob))
        at += len(blob)
    for name, (sec, off, _g) in obj.data_symbols.items():
        externs[name] = data_at[sec] + off
    voice = -(-at // 64) * 64 + 0x100
    externs.setdefault("plaits_voice_mem.", voice)
    left = {
        r.symbol
        for r in obj.relocs
        if r.symbol not in obj.symbols and r.symbol not in externs
    }
    assert not left, left
    linked = pk_sharc.link(obj, BASE_SW, externs)
    pk_sharc.verify(obj, linked, externs)
    return linked, blobs, externs, voice


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
        assert self.voice + self.size < FRAME
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
        res = run.run(5_000_000)
        assert res.halt.reason == "breakpoint" and res.final_pc_sw == RETURN_SW, (
            res.halt.to_json()
        )
        assert not run.state.loops, "loop stack not empty on return"
        self.runner = run
        self.instructions = res.instructions
        return {n: self._val(run.state, c) for n, c in self.st.UREG_CODES.items()}

    @staticmethod
    def _val(state: object, code: int) -> int | None:
        v = state.uregs.get(code)  # type: ignore[attr-defined]
        x = None if v is None else getattr(v, "value", None)
        return None if x is None else int(x) & 0xFFFFFFFF

    def log(self) -> list[int]:
        n = self.peek(self.externs["pl_t_logn."])
        return [self.peek(self.externs["pl_t_log."] + 4 * k) for k in range(n)]


def frame(rng: random.Random, model: int) -> list[int]:
    def knob() -> float:
        r = rng.random()
        if r < 0.1:
            return rng.choice([-5.0, 40000.0, float("nan"), -0.0])
        return (
            0.0 if r < 0.2 else (32512.0 if r < 0.3 else float(rng.randint(0, 32512)))
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
        rng.choice([1, 1, 1, 0, 0x55]),
        rng.choice([0, 0, 1, 7]),
    ]


def case(
    core: Core, voice: list[int], fr: list[int], trig: int, rate: int, regs=None
) -> tuple[object, ...]:
    core.put(core.voice, voice, guard=0)
    core.put(FRAME, fr, guard=0)
    core.put(DST, [GUARD ^ 3] * 64)
    core.poke(core.externs["pl_t_logn."], 0)
    n = 32 if rate == 48000 else 64
    after = core.call("plaits_render.", [core.voice, FRAME, trig, DST, n, rate], regs)
    return (
        after["R0"],
        core.get(DST, 64),
        core.get(core.voice + core.out_off, (core.size - core.out_off) // 4, guard=0),
        core.log(),
    )


def voice_words(rng: random.Random, core: Core, model: int) -> list[int]:
    words = [rng.getrandbits(32) for _ in range(core.size // 4)]
    t = core.out_off // 4
    for k in range(t, t + 32):
        words[k] = bits(rng.uniform(-1.0, 1.0))
    words[t + 32] = words[t + 33] = 0
    words[t + 34] = words[t + 35] = bits(0.5)
    words[t + 36] = bits(rng.random())
    words[t + 37] = model
    words[t + 38] = rng.choice([1, 1, 1, 0])
    words[t + 39] = rng.choice([0, 1])
    return words


@pytest.fixture(scope="module")
def pair() -> tuple[Core, Core]:
    return Core(True), Core(False)


@pytest.mark.parametrize("group", range(4))
def test_blocks_equal_compiled_c(pair: tuple[Core, Core], group: int) -> None:
    ker, ref = pair
    rng = random.Random(70 + group)
    for i in range(40):
        model = rng.choice([*range(25), 25, -1, 8, 18, 19, 20, 24])
        latched = model if rng.random() < 0.8 else rng.randrange(25)
        voice = voice_words(rng, ker, latched)
        fr = frame(rng, model)
        trig = rng.choice([0, 0, 1, 2])
        rate = 48000 if i % 10 else 96000
        got = [case(core, voice, fr, trig, rate) for core in (ker, ref)]
        assert got[0] == got[1], (group, i, model, trig, rate)


def test_preserves_the_dsp_state(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    rng = random.Random(5)
    voice = voice_words(rng, ker, 13)
    voice[ker.out_off // 4 + 38] = 1
    fr = frame(rng, 13)
    fr[8] = 1
    want = case(ref, voice, fr, 1, 48000)
    regs: dict[str, int] = dict(HOOK)
    for p in range(16):
        regs[f"R{p}"] = rng.getrandbits(32)
    for i in (0, 1, 2, 3, 5, 8, 9, 10, 11, 13, 14, 15):
        regs[f"I{i}"] = 0x200C0000 + 4 * rng.randrange(1000)
    got = case(ker, voice, fr, 1, 48000, regs)
    assert got == want
    before = dict(ker.before)
    after = {n: ker._val(ker.runner.state, c) for n, c in ker.st.UREG_CODES.items()}
    before["I6"], before["I7"] = CALLER_I6, STACK
    changed = {
        n: (before[n], after[n])
        for n in before
        if n not in SCRATCH | FLAGS | RUNNING and before[n] != after[n]
    }
    assert not changed, changed


def test_kernel_cost(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    rng = random.Random(9)
    voice = voice_words(rng, ker, 13)
    voice[ker.out_off // 4 + 38] = 1
    fr = frame(rng, 13)
    fr[8] = 1
    n = []
    for core in (ker, ref):
        case(core, voice, fr, 0, 48000)
        n.append(core.instructions)
    print(n)
    assert n[0] < n[1], n


def test_kernel_passes_the_strict_encoding_gate() -> None:
    import sqlite3

    from tools.patchkit import pk_encoding_gate as gate

    externs = {
        s: DATA
        for s in (
            "plaits_render_c.",
            "plaits_note_on_c.",
            "plaits_stage_kernel.",
            "plaits_out_gain.",
            "plaits_aux_gain.",
            "plaits_gain.",
        )
    }
    src = (DSP / "wrap_sm.s").read_text()
    obj = pk_sharc.assemble(src)
    for r in obj.relocs:
        if r.symbol not in obj.symbols:
            externs.setdefault(r.symbol, BASE_SW + 0x8000)
    for name in ("plaits_render_c.", "plaits_note_on_c.", "plaits_stage_kernel."):
        externs[name] = BASE_SW + 0x8000
    linked = pk_sharc.link(obj, BASE_SW, externs)

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
    assert gate.check_delay_slots(Mem(), ranges) == []
    assert gate.check_loop_ends(Mem(), ranges) == []
