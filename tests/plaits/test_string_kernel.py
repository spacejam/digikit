"""STRING's kernels (modules/machines/plaits/src/dsp/lib/string_sm.s: string_loop_kernel, string_excite_kernel) on the
isolated concrete SHARC+ core: string_engine_render built for the DSP against the same file built with -DPL_STRING_C
(selcc's C, the code the core digests were recorded from), bit for bit on out[], aux[] and the engine state, over
random states (both non-linearity modes, saturated and fresh write counts with the never-written words of the lines
holding garbage in the kernel build and zeros in the C's), sub-block sizes 0, 1, 7 and 16, run-on sequences from init
over garbage memory; the kernels leave every register outside the scratch set as they found them and pass the
strict encoding gate."""

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
VW = 1303  # words of a StringVoice
STATE_WORDS = 3 * VW + 22
FS_WORD, FA_WORD = 1301, 1302  # fresh_string_, fresh_stretch_ in a voice
EXTERNS: dict[bool, dict[str, int]] = {}


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@cache
def built(kernel: bool) -> tuple[pk_sharc.Linked, list[tuple[int, bytes]]]:
    defines = () if kernel else ("PL_STRING_C",)
    objs = [
        pk_sharc.compile_c(
            (DSP / "engines" / "string_engine.c").read_text(),
            include_dirs=[DSP / "engines", LIB, DSP],
            defines=defines,
        ),
        pk_sharc.compile_c((LIB / "tables.c").read_text(), include_dirs=[LIB, DSP]),
        pk_sharc.compile_c((LIB / "svf.c").read_text(), include_dirs=[LIB, DSP]),
    ]
    if kernel:
        objs.insert(0, pk_sharc.assemble((LIB / "string_sm.s").read_text()))
    objs.append(pk_sharc.assemble((LIB / "svf_sm.s").read_text()))
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


def rand_voice(rng: random.Random, fresh: bool) -> tuple[list[int], list[int]]:
    """A voice for the kernel build and the same voice as the C sees it (cleared where never written)."""
    f = lambda lo, hi: bits(rng.uniform(lo, hi))  # noqa: E731
    line = [f(-1.0, 1.0) for _ in range(1024)]
    stretch = [f(-1.0, 1.0) for _ in range(256)]
    g = rng.uniform(0.01, 0.9)
    r = 2.0
    exc = [
        bits(g),
        bits(r),
        bits(1.0 / (1.0 + r * g + g * g)),
        f(-0.5, 0.5),
        f(-0.5, 0.5),
    ]
    damp = [bits(0.3), bits(2.0), bits(0.5), f(-0.5, 0.5), f(-0.5, 0.5)]
    swp, awp = rng.randrange(1024), rng.randrange(256)
    fs = (
        rng.choice([0, 1, 5, 300, 1023, 1024, 1500])
        if fresh
        else 1024 + rng.randrange(3)
    )
    fa = rng.choice([0, 3, 100, 255, 256]) if fresh else 256 + rng.randrange(3)
    tail = [
        f(-0.2, 0.2),
        f(-0.2, 0.2),  # dc_x_, dc_y_
        f(4.0, 1000.0),
        f(-0.4, 0.4),
        f(-0.3, 0.3),
        f(0.0, 1.0),  # delay_, dispersion_noise_, curved_bridge_, src_phase_
        f(-0.5, 0.5),
        f(-0.5, 0.5),  # out_sample_
        swp,
        awp,
        rng.choice([0, 0, 3, 40, 300]),
        fs,
        fa,
    ]
    ker = line + stretch + exc + damp + tail
    ref = list(ker)
    for o in range(1024):
        if (o - 1) % 1024 >= fs:
            ref[(swp + o) % 1024] = 0
    for o in range(256):
        if (o - 1) % 256 >= fa:
            ref[1024 + (awp + o) % 256] = 0
    return ker, ref


def rand_state(rng: random.Random, fresh: bool) -> tuple[list[int], list[int]]:
    ker: list[int] = []
    ref: list[int] = []
    for _ in range(3):
        k, r = rand_voice(rng, fresh)
        ker += k
        ref += r
    eng = [bits(rng.choice([0.0, rng.uniform(0.0005, 0.2)])) for _ in range(3)]
    eng += [bits(rng.uniform(0.0, 0.2)) for _ in range(16)]
    eng += [rng.randrange(16), rng.randrange(3), rng.getrandbits(32)]
    return ker + eng, ref + eng


def rand_params(rng: random.Random) -> list[int]:
    note = rng.choice([rng.uniform(0.0, 127.0), 127.0, 0.0, rng.uniform(20.0, 100.0)])
    h = rng.choice([0.0, 0.1, 0.25, 0.5, 0.9, 1.0, rng.random()])
    t, m = (rng.choice([0.0, 1.0, 0.5, 0.97, rng.random()]) for _ in range(2))
    return [
        bits(note),
        bits(h),
        bits(t),
        bits(m),
        rng.choice([0, 1, 4, 5]),
        bits(rng.random()),
    ]


def render(core: Core, state: list[int] | None, prm: list[int], size: int = 16):
    if state is not None:
        core.put(ST, state)
    core.put(PRM, prm)
    core.put(OUT, [GUARD] * size)
    core.put(AUX, [GUARD] * size)
    core.call("string_engine_render.", [ST, PRM, OUT, AUX, size])
    return core.get(OUT, size), core.get(AUX, size)


def written(core: Core) -> list[int]:
    """The engine state without the fresh counters and with the words of each line the kernel counts as written."""
    st = core.get(ST, STATE_WORDS, guard=0)
    res: list[int] = []
    for v in range(3):
        w = st[v * VW : (v + 1) * VW]
        swp, awp, fs, fa = w[1298], w[1299], w[FS_WORD], w[FA_WORD]
        res += [w[(swp + o) % 1024] for o in range(1024) if (o - 1) % 1024 < fs]
        res += [w[1024 + (awp + o) % 256] for o in range(256) if (o - 1) % 256 < fa]
        res += w[1280:FS_WORD]
    return res + st[3 * VW :]


def compare_state(ker: Core, ref: Core) -> None:
    sk = ker.get(ST, STATE_WORDS, guard=0)
    sr = ref.get(ST, STATE_WORDS, guard=0)
    for v in range(3):
        w = sk[v * VW : (v + 1) * VW]
        q = sr[v * VW : (v + 1) * VW]
        assert w[1280:FS_WORD] == q[1280:FS_WORD], v
        swp, awp, fs, fa = w[1298], w[1299], w[FS_WORD], w[FA_WORD]
        for o in range(1024):
            if (o - 1) % 1024 < fs:
                assert w[(swp + o) % 1024] == q[(swp + o) % 1024], (v, o)
        for o in range(256):
            if (o - 1) % 256 < fa:
                assert w[1024 + (awp + o) % 256] == q[1024 + (awp + o) % 256], (v, o)
    assert sk[3 * VW :] == sr[3 * VW :]


@pytest.fixture(scope="module")
def pair() -> tuple[Core, Core]:
    return Core(True), Core(False)


@pytest.mark.parametrize("fresh", [False, True])
def test_render_equals_compiled_c_on_random_states(
    pair: tuple[Core, Core], fresh: bool
) -> None:
    ker, ref = pair
    rng = random.Random(100 + fresh)
    for i in range(24):
        sk, sr = rand_state(rng, fresh)
        prm = rand_params(rng)
        size = rng.choice([16, 16, 16, 1, 7, 0])
        assert render(ker, sk, prm, size) == render(ref, sr, prm, size), i
        compare_state(ker, ref)
        for _ in range(3):
            prm = rand_params(rng) if rng.random() < 0.5 else prm
            assert render(ker, None, prm) == render(ref, None, prm), i
        compare_state(ker, ref)


def test_render_runs_on_from_init_over_garbage(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    rng = random.Random(7)
    garbage = [rng.getrandbits(32) for _ in range(STATE_WORDS)]
    for core in (ker, ref):
        core.put(ST, garbage)
        core.call("string_engine_init.", [ST])
    prm = rand_params(rng)
    for blk in range(110):
        if rng.random() < 0.15:
            prm = rand_params(rng)
        prm[4] = rng.choice([1, 5]) if rng.random() < 0.1 else (prm[4] & 4)
        assert render(ker, None, prm) == render(ref, None, prm), blk
    compare_state(ker, ref)


def test_kernel_cost(pair: tuple[Core, Core]) -> None:
    ker, ref = pair
    rng = random.Random(3)
    sk, sr = rand_state(rng, False)
    prm = rand_params(rng)
    prm[1] = bits(0.0)
    n = []
    for core, st in ((ker, sk), (ref, sr)):
        render(core, st, prm)
        n.append(core.instructions)
    print(n)
    assert n[0] < n[1], n


@pytest.mark.parametrize("seed", [11, 12])
def test_render_preserves_the_dsp_state(pair: tuple[Core, Core], seed: int) -> None:
    ker, ref = pair
    rng = random.Random(seed)
    sk, sr = rand_state(rng, seed == 12)
    prm = rand_params(rng)
    prm[1] = bits(0.0 if seed == 11 else 0.9)
    want = render(ref, sr, prm)
    ker.put(ST, sk)
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
    after = ker.call("string_engine_render.", [ST, PRM, OUT, AUX, 16], regs)
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
    """Every instruction of string_sm.s has a precedent in stock code that belongs to a function (or in
    encoding_proven.toml), no data transfer runs in SIMD mode and no DAG register is stored through its own DAG."""
    import sqlite3

    from tools.patchkit import pk_encoding_gate as gate

    linked = pk_sharc.link(
        pk_sharc.assemble((LIB / "string_sm.s").read_text()), BASE_SW, {}
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
