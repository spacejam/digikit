"""DX7's voice kernel (modules/machines/plaits/src/dsp/lib/dx_sm.s: dx_voice_kernel) on the isolated concrete SHARC+
core: six_op_render built for the DSP against the same file built with -DPL_DX_C (selcc's C, the code the core digests
were recorded from), bit for bit on out[], aux[] and the engine state, over run-on sequences from init in every bank
(random patches, so every algorithm's renderers, note-ons, holds and releases, random knobs), sub-block sizes 1, 7
and 16, and every algorithm with every feedback amount poked into the loaded patch (the only way to reach the
feedback paths of renderers 6 and 7: no bank patch with a 3-op loop has feedback > 0); the kernel leaves every
register outside the scratch set as it found it and passes the strict encoding gate."""

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
ST = 0x200C0000
PRM, OUT, AUX = (0x200D0000 + 0x400 * i for i in range(3))
STATE_WORDS = 4096  # more than sizeof(SixOp) / 4
EXTERNS: dict[bool, dict[str, int]] = {}


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


# SixOp: five words, temp[96], acc[32], then voice[2] of 276 words; in a voice, phase[6] at 23 and the patch at 132
# (algorithm at 266, feedback at 267).
VOICE0, VOICE_WORDS, V_PHASE, V_ALGORITHM, V_FEEDBACK = 133, 276, 23, 266, 267
V_PITCH_LEVEL = 121 + 7  # pitch_env.level[4]


@cache
def built(kernel: bool) -> tuple[pk_sharc.Linked, list[tuple[int, bytes]]]:
    defines = () if kernel else ("PL_DX_C",)
    objs = [
        pk_sharc.compile_c(
            (DSP / "engines" / "six_op.c").read_text(),
            include_dirs=[DSP / "engines", LIB, DSP],
            defines=defines,
        ),
        pk_sharc.compile_c((LIB / "tables.c").read_text(), include_dirs=[LIB, DSP]),
        pk_sharc.compile_c((LIB / "dx_tables.c").read_text(), include_dirs=[LIB, DSP]),
        pk_sharc.compile_c((LIB / "dx7_banks.c").read_text(), include_dirs=[LIB, DSP]),
    ]
    if kernel:
        objs.insert(0, pk_sharc.assemble((LIB / "dx_sm.s").read_text()))
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
    EXTERNS[kernel] = externs
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


def rand_params(rng: random.Random, trig: int) -> list[int]:
    note = rng.choice([rng.uniform(0.0, 127.0), 127.0, 0.0, rng.uniform(30.0, 90.0)])
    h, t, m = (rng.choice([0.0, 1.0, 0.5, rng.random()]) for _ in range(3))
    return [bits(note), bits(h), bits(t), bits(m), trig, bits(rng.random())]


def render(core: Core, prm: list[int], size: int = 16):
    core.put(PRM, prm)
    core.put(OUT, [GUARD] * size)
    core.put(AUX, [GUARD] * size)
    core.call("six_op_render.", [ST, PRM, OUT, AUX, size])
    return core.get(OUT, size), core.get(AUX, size)


def start(pair: tuple[Core, Core], bank: int, seed: int) -> None:
    garbage = [random.Random(seed).getrandbits(32) for _ in range(STATE_WORDS)]
    for core in pair:
        core.put(ST, garbage)
        core.call("six_op_init.", [ST])
        core.call("six_op_load_bank.", [ST, bank])


@pytest.fixture(scope="module")
def pair() -> tuple[Core, Core]:
    return Core(True), Core(False)


@pytest.mark.parametrize("bank", [0, 1, 2, 3])
def test_render_runs_on(pair: tuple[Core, Core], bank: int) -> None:
    ker, ref = pair
    rng = random.Random(40 + bank)
    start(pair, bank, bank)
    prm = rand_params(rng, 5)
    cost = [0, 0]
    for blk in range(45):
        r = rng.random()
        trig = 5 if r < 0.25 else (0 if r < 0.4 else 4)
        prm = (
            rand_params(rng, trig)
            if trig == 5 or rng.random() < 0.2
            else prm[:4] + [trig] + prm[5:]
        )
        size = rng.choice([16, 16, 16, 7, 1])
        assert render(ker, prm, size) == render(ref, prm, size), blk
        cost[0] += ker.instructions
        cost[1] += ref.instructions
        assert ker.get(ST, STATE_WORDS, 0) == ref.get(ST, STATE_WORDS, 0), blk
    assert cost[0] < cost[1] * 0.7, cost  # the kernel ran


@pytest.mark.parametrize("bank", [0, 1, 2, 3])
def test_every_patch_of_the_bank(pair: tuple[Core, Core], bank: int) -> None:
    """Each of the 32 patches struck and held (every algorithm and feedback setting of the bank)."""
    ker, ref = pair
    rng = random.Random(70 + bank)
    start(pair, bank, 100 + bank)
    for patch in range(32):
        prm = rand_params(rng, 5)
        prm[1] = bits((patch + 0.5) / 32.0 / 1.02)
        for blk, trig in enumerate((5, 4, 4)):
            prm[4] = trig
            assert render(ker, prm) == render(ref, prm), (patch, blk)
    assert ker.get(ST, STATE_WORDS, 0) == ref.get(ST, STATE_WORDS, 0)


@pytest.mark.parametrize("group", range(4))
def test_every_algorithm_and_feedback(pair: tuple[Core, Core], group: int) -> None:
    """Algorithms 8 group .. 8 group + 7, each with feedback amounts 0, 7 and one in 1..6 poked into both voices' loaded
    patch, random phases and pitch envelope levels, then struck again, held and released: every renderer, including 6 and 7 with feedback > 0."""
    ker, ref = pair
    rng = random.Random(900 + group)
    start(pair, group, 300 + group)
    for algorithm in range(8 * group, 8 * group + 8):
        prm = rand_params(rng, 5)
        prm[1] = bits(rng.random())
        for core in (ker, ref):
            render(core, prm)
            render(core, prm[:4] + [4] + prm[5:])
        for fb in (0, 7, rng.randrange(1, 7)):
            phases = [rng.getrandbits(32) for _ in range(6)]
            pitch = [bits(rng.uniform(-0.5, 0.5)) for _ in range(4)]
            for core in (ker, ref):
                for vi in range(2):
                    base = ST + 4 * (VOICE0 + VOICE_WORDS * vi)
                    core.poke(base + 4 * V_ALGORITHM, algorithm)
                    core.poke(base + 4 * V_FEEDBACK, fb)
                    for k, w in enumerate(phases):
                        core.poke(base + 4 * (V_PHASE + k), w)
                    for k, w in enumerate(pitch):
                        core.poke(base + 4 * (V_PITCH_LEVEL + k), w)
            for blk, trig in enumerate((4, 5, 4, 0)):
                prm = prm[:4] + [trig] + prm[5:]
                if trig == 5:
                    prm[2] = bits(rng.random())
                    prm[5] = bits(rng.random())
                size = rng.choice([16, 16, 7, 1])
                assert render(ker, prm, size) == render(ref, prm, size), (
                    algorithm,
                    fb,
                    blk,
                )
            assert ker.get(ST, STATE_WORDS, 0) == ref.get(ST, STATE_WORDS, 0), (
                algorithm,
                fb,
            )


def test_kernel_cost(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    start(pair, 0, 5)
    prm = [bits(60.0), bits(0.3), bits(0.5), bits(0.5), 5, bits(0.8)]
    n = []
    for core in (ker, ref):
        render(core, prm)
        total = 0
        for _ in range(6):
            render(core, prm[:4] + [4] + prm[5:])
            total += core.instructions
        n.append(total)
    print(n)
    assert n[0] < n[1], n


@pytest.mark.parametrize("seed", [11, 12])
def test_render_preserves_the_dsp_state(pair: tuple[Core, Core], seed: int) -> None:
    ker, ref = pair
    rng = random.Random(seed)
    start(pair, seed % 4, seed)
    prm = rand_params(rng, 5)
    want = render(ref, prm)
    ker.put(PRM, prm)
    ker.put(OUT, [GUARD] * 16)
    ker.put(AUX, [GUARD] * 16)
    regs: dict[str, int] = dict(HOOK)
    for p in range(16):
        regs[f"R{p}"] = rng.getrandbits(32)
        regs[f"S{p}"] = rng.getrandbits(32)
    for i in (0, 1, 2, 3, 5, 8, 9, 10, 11, 13, 14, 15):
        regs[f"I{i}"] = 0x200C0000 + 4 * rng.randrange(1000)
        regs[f"B{i}"] = 0x200C0000 + 4 * rng.randrange(1000)
        regs[f"L{i}"] = 4 * rng.randrange(1, 64)
    for m in (0, 1, 2, 3, 4, 8, 9, 10, 11, 12):
        regs[f"M{m}"] = rng.randrange(-64, 64) & 0xFFFFFFFF
    if seed % 2 == 0:
        regs["M6"] = 3
    for u in ("USTAT1", "USTAT2", "USTAT3", "USTAT4", "PX1", "PX2"):
        regs[u] = rng.getrandbits(32)
    regs["L4"] = 0
    after = ker.call("six_op_render.", [ST, PRM, OUT, AUX, 16], regs)
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
    assert (ker.get(OUT, 16), ker.get(AUX, 16)) == want


def test_kernel_passes_the_strict_encoding_gate() -> None:
    """Every instruction of dx_sm.s has a precedent in stock code that belongs to a function (or in
    encoding_proven.toml), no data transfer runs in SIMD mode and no DAG register is stored through its own DAG."""
    import sqlite3

    from tools.patchkit import pk_encoding_gate as gate

    linked = pk_sharc.link(
        pk_sharc.assemble((LIB / "dx_sm.s").read_text()),
        BASE_SW,
        {"pl_lut_sine.": DATA, "pl_dx_op_level.": DATA, "pl_dx_op_increment.": DATA},
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
    assert gate.check_delay_slots(Mem(), ranges) == []
    assert gate.check_loop_ends(Mem(), ranges) == []
