"""PLAITS models 18, 19, 20 and 24, the DX7 models (modules/machines/plaits/src/dsp/engines/six_op.c): the C99 port of
Plaits' six-operator FM engine (SixOpEngine with plaits/dsp/fm/*) against upstream (tests/plaits/oracle/run_oracle.cc,
../eurorack/plaits at 48 kHz), bit for bit on OUT and AUX, for all four banks: upstream's three (models 18, 19, 20,
fm_patches_table) and the owner's (model 24, data/tyler-favorites.syx, loaded through upstream's own LoadUserData).

What is compared
  - the generated tables (lib/gen_dx_tables.py) against upstream: the bank bytes (the built-in banks through
    fm_patches_table, the owner's bank against the .syx), the unit conversions of dx_units.h for every 0..99 value, the
    float tables, the compiled algorithms (render call and modulator of every algorithm and operator), and the three
    constants of the 48 kHz voice against numpy float32;
  - every patch of every bank, each played note-on, held, note-off, retriggered during the release, with the knobs of
    its own (HARMONICS picks the patch: patch k at h = (k + 0.5) / 32.64);
  - the knob corners at four pitches, sub-blocks of 16 and 12; randomised gate sequences (note-ons at different
    spacings with the gate high, low and high again, velocities, pitch and knob jumps, patch changes while notes ring);
  - extremes (notes -64 to 190, every corner): finite and equal; index ranges (the pitch-ratio tables, lut_cube_root, the
    modulation input of SinePM); tuning (A 440 within a cent through an all-integer-ratio patch); state size.
Not ported (six_op.h): the sustain mode (TRIGGER_UNPATCHED) and the LFO's Scrub: the oracle is never driven with the
unpatched trigger."""

from __future__ import annotations

import ctypes
import re
from typing import Any

import numpy as np
import pytest
from support.plaits_host import DSP, FP, Params, Port, build_port, owner_bank
from support.plaits_oracle import Oracle, Step, build, same

Arr = Any
MODELS = [18, 19, 20, 24]  # banks 0..3
RISING_HIGH, HIGH, LOW = 1 | 4, 4, 0
V100 = float(np.float32(np.float32(100 << 8) * np.float32(3.0757874e-05)))
F = np.float32


@pytest.fixture(scope="module")
def port(tmp_path_factory: pytest.TempPathFactory) -> Port:
    p = build_port(tmp_path_factory.mktemp("port"))
    p.lib.six_op_load_bank.argtypes = [ctypes.c_void_p, ctypes.c_int]
    p.lib.six_op_seed.argtypes = [ctypes.c_void_p, ctypes.c_int]
    p.lib.port_dx7_byte.argtypes = [ctypes.c_int, ctypes.c_int]
    p.lib.port_dx_word.argtypes = [ctypes.c_char_p, ctypes.c_int]
    p.lib.port_dx_word.restype = ctypes.c_uint
    p.lib.port_dx7_patch.argtypes = [ctypes.c_float]
    p.lib.port_dx7_patch_word.argtypes = [ctypes.c_int]
    p.lib.port_dx7_patch_word.restype = ctypes.c_uint
    p.lib.port_dx7_patch_thr.argtypes = [ctypes.c_int]
    p.lib.port_dx7_patch_thr.restype = ctypes.c_uint
    return p


@pytest.fixture(scope="module")
def checked(tmp_path_factory: pytest.TempPathFactory) -> Port:
    p = build_port(tmp_path_factory.mktemp("port-idx"), ("PL_INDEX_CHECK",))
    p.lib.six_op_load_bank.argtypes = [ctypes.c_void_p, ctypes.c_int]
    for name in ("port_index_lo", "port_index_hi", "port_index_n"):
        getattr(p.lib, name).argtypes = [ctypes.c_int]
    return p


@pytest.fixture(scope="module")
def oracle(tmp_path_factory: pytest.TempPathFactory) -> Oracle:
    o = build(tmp_path_factory.mktemp("run-oracle"))
    o.lib.run_oracle_dx_unit.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    o.lib.run_oracle_dx_lut.argtypes = [ctypes.c_int, ctypes.c_int]
    o.lib.run_oracle_dx_lut.restype = ctypes.c_float
    o.lib.run_oracle_bank_byte.argtypes = [ctypes.c_int, ctypes.c_int]
    o.lib.run_oracle_patch_index.argtypes = [ctypes.c_float]
    o.lib.run_oracle_algorithm.argtypes = [ctypes.c_int, ctypes.c_int] + [
        ctypes.POINTER(ctypes.c_int)
    ] * 5
    return o


def bits(x: float) -> int:
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


# ---- the port and the oracle on one engine


def six_op(port: Port, bank: int, steps: list[Step], size: int = 16) -> tuple[Arr, Arr]:
    """six_op_init, six_op_load_bank(bank), one render of `size` samples per step (track 0's generator)."""
    st = ctypes.create_string_buffer(port.six_op_size)
    port.lib.six_op_init(st)
    port.lib.six_op_load_bank(st, bank)
    out = np.zeros(len(steps) * size, dtype=np.float32)
    aux = np.zeros_like(out)
    for k, s in enumerate(steps):
        p = Params(*s)
        port.lib.six_op_render(
            st,
            ctypes.byref(p),
            out[k * size :].ctypes.data_as(FP),
            aux[k * size :].ctypes.data_as(FP),
            size,
        )
    return out, aux


def check(
    port: Port, oracle: Oracle, bank: int, steps: list[Step], size: int = 16
) -> tuple[Arr, Arr]:
    got = six_op(port, bank, steps, size)
    want = oracle.engine(MODELS[bank], steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1]), (bank, size)
    assert np.isfinite(got[0]).all()
    return got


def patch_h(k: int) -> float:
    """HARMONICS for patch k: the quantiser's value h * 32.64 in the middle of [k, k + 1)."""
    return float(F((k + 0.5) / 32.64))


def note_stream(
    note: float,
    h: float,
    t: float,
    m: float,
    hold: int,
    rel: int,
    notes: int = 2,
    accent: float = V100,
) -> list[Step]:
    """`notes` note-ons: each RISING_EDGE | HIGH for 1 sub-block, HIGH for `hold` sub-blocks in all, then the gate LOW for
    `rel` sub-blocks, then the next note-on (a retrigger during the release tail of the last)."""
    out: list[Step] = []
    for _ in range(notes):
        for b in range(hold + rel):
            trig = RISING_HIGH if b == 0 else (HIGH if b < hold else LOW)
            out.append((note, h, t, m, trig, accent))
    return out


# ---- the memoryless patch quantiser (owner decision, 2026-10-23: PORTING drift 28)


def generator() -> Any:
    """lib/gen_dx_tables.py as a module (the table's source of truth)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_dx_tables", DSP / "lib" / "gen_dx_tables.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_patch_of_every_harmonics_word_is_upstreams_quantiser_without_hysteresis(
    port: Port, oracle: Oracle
) -> None:
    """All 32,513 words of HARMONICS (0 .. 0x7f00): the port's patch = upstream's own HysteresisQuantizer2 with the
    hysteresis 0 (the oracle) = the generator's float32 model, and every patch 0..31 occurs."""
    gen = generator()
    seen = set()
    for w in range(0x7F01):
        h = float(gen.unit_of_word(w))
        q = port.lib.port_dx7_patch(h)
        assert q == oracle.lib.run_oracle_patch_index(h) == gen.patch_of_word(w), w
        seen.add(q)
    assert seen == set(range(32))


def test_the_patch_tables_are_the_generated_ones(port: Port) -> None:
    """pl_dx7_patch_word is the boundary (the smallest word of each patch), pl_dx7_patch_thr the float unit() of it, and the
    ColdFire's src/cf/dx7_names.s is regenerated byte for byte from the same list and the banks' names."""
    gen = generator()
    thr = gen.patch_words()
    assert thr[0] == 0 and len(thr) == 32 and thr == sorted(set(thr))
    for k in range(32):
        assert port.lib.port_dx7_patch_word(k) == thr[k]
        assert port.lib.port_dx7_patch_thr(k) == bits(gen.unit_of_word(thr[k]))
        if k:
            assert (
                gen.patch_of_word(thr[k]) == k
                and gen.patch_of_word(thr[k] - 1) == k - 1
            )
    banks = [bytes(port.lib.port_dx7_byte(b, j) for j in range(4096)) for b in range(4)]
    want = gen.names_asm(thr, gen.name_rows(banks))
    have = (DSP.parent / "cf" / "dx7_names.s").read_text()
    assert have == want


def test_the_patch_does_not_depend_on_the_direction_of_the_knob(
    port: Port, oracle: Oracle
) -> None:
    """Notes played at the word before and at each boundary, coming from below and from above (the oracle is built with
    the hysteresis 0; upstream's 0.005 would pick the old patch on one of the two ways): bit for bit, all four banks."""
    gen = generator()
    thr = gen.patch_words()
    for bank in range(4):
        steps: list[Step] = []
        for k in range(1, 32):
            lo, hi = (
                float(gen.unit_of_word(thr[k] - 1)),
                float(gen.unit_of_word(thr[k])),
            )
            for h in (lo, hi, lo, hi, hi, lo):
                steps += note_stream(60.0, h, 0.5, 0.5, 6, 3, 1)
        check(port, oracle, bank, steps)


# ---- the tables


def test_banks_equal_upstream_and_the_syx(port: Port, oracle: Oracle) -> None:
    """Banks 0..2 are fm_patches_table's three banks byte for byte (compiled from upstream's resources.cc); bank 3 is the
    owner's .syx (header, checksum and length checked by the generator) as upstream's Patch::Unpack reads it."""
    for b in range(3):
        for j in range(4096):
            assert port.lib.port_dx7_byte(b, j) == oracle.lib.run_oracle_bank_byte(b, j)
    syx = (DSP.parents[1] / "data" / "tyler-favorites.syx").read_bytes()
    assert len(syx) == 4104 and syx[:6].hex() == "f04300092000" and syx[-1] == 0xF7
    assert syx[4102] == (-sum(syx[6:4102])) & 0x7F
    assert owner_bank() == syx[6:4102]
    assert bytes(port.lib.port_dx7_byte(3, j) for j in range(4096)) == syx[6:4102]
    assert syx[6 + 118 : 6 + 128] == b"*PPG*Vol.1"


def test_unit_tables_equal_upstream(port: Port, oracle: Oracle) -> None:
    """The generated 0..99 conversion tables are dx_units.h's functions evaluated by upstream, bit for bit."""
    kinds = {
        0: b"l",
        1: b"P",
        2: b"O",
        3: b"I",
        4: b"F",
    }
    a, b = ctypes.c_float(), ctypes.c_float()
    for kind, name in kinds.items():
        for v in range(100):
            oracle.lib.run_oracle_dx_unit(kind, v, ctypes.byref(a), ctypes.byref(b))
            got = port.lib.port_dx_word(name, v)
            want = int(a.value) if kind == 0 else bits(a.value)
            assert got == want, (kind, v, got, want)
    for v in range(100):
        oracle.lib.run_oracle_dx_unit(5, v, ctypes.byref(a), ctypes.byref(b))
        assert port.lib.port_dx_word(b"D", v) == bits(a.value), v
        assert port.lib.port_dx_word(b"E", v) == bits(b.value), v
    for kind, name, n in ((0, b"k", 32), (1, b"a", 4), (2, b"p", 8), (3, b"q", 17)):
        for i in range(n):
            assert port.lib.port_dx_word(name, i) == bits(
                oracle.lib.run_oracle_dx_lut(kind, i)
            ), (kind, i)
    assert port.lib.port_dx_word(b"q", 17) == bits(1.0)  # the guard entry


def test_compiled_algorithms_equal_upstream(port: Port, oracle: Oracle) -> None:
    """pl_dx_calls and pl_dx_modmask are upstream's compiled Algorithms<6> (render call per operator, is_modulator)."""
    r, n, i, o, m = (ctypes.c_int() for _ in range(5))
    starts = 0
    for a in range(32):
        mask = 0
        for op in range(6):
            oracle.lib.run_oracle_algorithm(
                a,
                op,
                ctypes.byref(r),
                ctypes.byref(n),
                ctypes.byref(i),
                ctypes.byref(o),
                ctypes.byref(m),
            )
            word = port.lib.port_dx_word(b"c", a * 6 + op)
            mask |= m.value << op
            if word:
                starts += 1
                assert (word & 0xFF, (word >> 8) & 0xFF) == (r.value, n.value), (a, op)
                assert ((word >> 16) & 0xFF, word >> 24) == (i.value, o.value), (a, op)
        assert port.lib.port_dx_word(b"m", a) == mask, a
    # every operator is covered by exactly one chain: the chain lengths sum to six in each algorithm
    for a in range(32):
        assert (
            sum(
                (port.lib.port_dx_word(b"c", a * 6 + op) >> 8) & 0xFF for op in range(6)
            )
            == 6
        ), a
    assert starts > 32


def test_voice_constants_are_the_48k_floats() -> None:
    """DX_ONE_HZ, DX_A0 and DX_ENV_SCALE are what upstream's Voice::Init computes in float at 48,000 Hz."""
    src = (DSP / "engines" / "six_op.c").read_text()
    got = {
        k: F(re.search(rf"#define {k} ([0-9.e+-]+)f", src).group(1))  # type: ignore[union-attr]
        for k in ("DX_ONE_HZ", "DX_A0", "DX_ENV_SCALE")
    }
    one_hz = F(1.0) / F(48000.0)
    assert got["DX_ONE_HZ"] == one_hz
    assert got["DX_A0"] == F(55.0) / F(48000.0)
    assert got["DX_ENV_SCALE"] == F(44100.0) * one_hz
    assert "47872" not in src


# ---- the engine


@pytest.mark.parametrize("bank", range(4))
def test_every_patch_equals_upstream(port: Port, oracle: Oracle, bank: int) -> None:
    """Each of the 32 patches of the bank (HARMONICS), a note held 180 sub-blocks, released for 120, retriggered at another
    pitch and held 100 more: bit for bit, and sounding."""
    sounded = 0
    for k in range(32):
        h = patch_h(k)
        steps = note_stream(60.0, h, 0.5, 0.5, 180, 120, 1)
        steps += note_stream(67.0, h, 0.5, 0.5, 100, 40, 1)
        out, _ = check(port, oracle, bank, steps)
        sounded += int(float(np.abs(out).max()) > 0.01)
    assert sounded >= 28, sounded  # a few banks hold a very quiet voice


@pytest.mark.parametrize("size", [16, 12])
def test_knob_corners_equal_upstream(port: Port, oracle: Oracle, size: int) -> None:
    """HARMONICS 0, between patches, 1 and TIMBRE and MORPH at their corners, four pitches, all four banks: the engine's
    brightness (the modulator level) and envelope time scale."""
    for bank in range(4):
        for h in (0.0, 0.31, 0.5, 1.0):
            for t in (0.0, 0.5, 1.0):
                for m in (0.0, 0.3, 0.5, 1.0):
                    for note in (24.0, 60.0, 96.0, 127.0):
                        steps = note_stream(note, h, t, m, 40, 20, 2)
                        check(port, oracle, bank, steps, size)


def random_steps(seed: int, n: int) -> list[Step]:
    """Gates (note-on, high, low, high again without an edge), note and knob jumps, accents, patch changes while a note
    rings (HARMONICS jumping across patches)."""
    rng = np.random.default_rng(seed)
    steps: list[Step] = []
    cur = (60.0, 0.4, 0.6, 0.3)
    gate = HIGH
    for _ in range(n):
        if rng.random() < 0.12:
            cur = (
                float(rng.choice([-64.0, 24.0, 48.0, 60.0, 72.0, 100.0, 127.0, 190.0])),
                float(rng.choice([0.0, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.49999, 0.5, 0.5001, 1.0, rng.random()])),
            )
        r = rng.random()
        if r < 0.08:
            gate = RISING_HIGH
        elif r < 0.14:
            gate = LOW
        elif r < 0.2:
            gate = HIGH
        if gate == RISING_HIGH:
            trig, gate = RISING_HIGH, HIGH
        else:
            trig = gate
        steps.append((*cur, trig, float(rng.random())))
    return steps


def test_random_gates_and_jumps_equal_upstream(port: Port, oracle: Oracle) -> None:
    for seed in range(16):
        check(port, oracle, seed % 4, random_steps(seed, 3000))
    for seed in range(4):
        check(port, oracle, seed, random_steps(100 + seed, 3000), 12)


def test_fast_retriggers_equal_upstream(port: Port, oracle: Oracle) -> None:
    """A note-on every 1, 2, 3 and 5 sub-blocks, with and without a note-off between (the two voices alternate, the
    first note's tail rings on while the next starts), four banks."""
    for bank in range(4):
        for every in (1, 2, 3, 5, 8):
            for rel in (0, 1):
                steps: list[Step] = []
                for b in range(600):
                    trig = (
                        RISING_HIGH
                        if b % every == 0
                        else (LOW if rel and b % every == every - 1 else HIGH)
                    )
                    steps.append(
                        (
                            55.0 + (b // every) % 7,
                            patch_h((b // every) % 32),
                            0.6,
                            0.4,
                            trig,
                            V100,
                        )
                    )
                check(port, oracle, bank, steps)


def test_extremes_are_finite_and_equal_upstream(port: Port, oracle: Oracle) -> None:
    for bank in range(4):
        for note in (-64.0, -20.0, 0.0, 120.0, 150.0, 190.0):
            for h, t, m in (
                (0, 0, 0),
                (1, 1, 1),
                (1, 0, 1),
                (0, 1, 0),
                (0.5, 0.67, 0.5),
            ):
                for acc in (0.0, 1.0):
                    steps = note_stream(note, h, t, m, 60, 30, 2, acc)
                    check(port, oracle, bank, steps)


def test_indices_in_range(checked: Port) -> None:
    """The pitch-ratio tables (256 entries) are read inside them (notes -64 to 190, the pitch envelopes' swings and the
    fixed-frequency operators' ratios up to 2^8), lut_cube_root at velocity 1.0 reads its 17th (guard) entry, and every
    modulation input to SinePM stays inside (-32, 32) (it would wrap the phase otherwise): over all 128 patches."""
    checked.lib.port_index_reset()
    for bank in range(4):
        for k in range(32):
            for note, acc in ((-64.0, 1.0), (60.0, 1.0), (127.0, 0.5), (190.0, 1.0)):
                steps = note_stream(note, patch_h(k), 1.0, 0.5, 60, 20, 1, acc)
                six_op(checked, bank, steps)
    ranges = {
        t: (
            checked.lib.port_index_lo(t),
            checked.lib.port_index_hi(t),
            checked.lib.port_index_n(t),
        )
        for t in (41, 42, 43, 44)
    }
    print("DX7 index ranges", ranges)
    assert all(n > 0 for _, _, n in ranges.values())  # the recorder is hit
    assert ranges[41][0] >= 0 and ranges[41][1] <= 255
    assert ranges[42][0] >= 0 and ranges[42][1] == 16 and ranges[42][1] <= 17
    assert ranges[43][0] > -32 * 1024 and ranges[43][1] < 32 * 1024
    assert ranges[44][0] >= 0 and ranges[44][1] <= 255


def peak_ratio(x: Arr, f: float) -> float:
    """The spectral peak of x (Hann window, parabolic interpolation on a zero-padded FFT) over f, as a pitch ratio."""
    n = len(x)
    w = np.hanning(n)
    pad = 1 << 20
    spec = np.abs(np.fft.rfft(x * w, pad))
    k = int(np.argmax(spec))
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-30)
    off = 0.5 * (a - c) / (a - 2 * b + c)
    return float((k + off) * 48000.0 / pad / f)


def test_tuning_a440(port: Port) -> None:
    """Note 69 plays 440 Hz: a patch whose six operators all have a whole-number ratio (coarse 1..8, fine 0, detune 7, ratio
    mode), a flat pitch envelope (levels 50) and no vibrato plays a signal that is periodic at f0, so the strongest partial of
    the held note is a whole multiple of 440 Hz within a cent. (A 47,872 Hz build is 4.6 cents flat and fails: the mutation
    table of the report.)"""
    checked_patches = 0
    for bank in range(4):
        for k in range(32):
            base = k * 128
            ok = True
            for op in range(6):
                o = base + op * 17
                b15 = port.lib.port_dx7_byte(bank, o + 15)
                b16 = port.lib.port_dx7_byte(bank, o + 16)
                b12 = port.lib.port_dx7_byte(bank, o + 12)
                if (
                    (b15 & 1)
                    or b16 != 0
                    or ((b12 >> 3) & 0xF) != 7
                    or not 1 <= ((b15 >> 1) & 0x1F) <= 8
                ):
                    ok = False
            if any(
                port.lib.port_dx7_byte(bank, base + 106 + j) != 50 for j in range(4)
            ):
                ok = False
            if port.lib.port_dx7_byte(bank, base + 114) != 0:  # no vibrato
                ok = False
            if not ok:
                continue
            steps = note_stream(69.0, patch_h(k), 0.5, 0.5, 1500, 0, 1)
            out, _ = six_op(port, bank, steps)
            seg = out[16 * 400 : 16 * 1400]
            pk = peak_ratio(seg, 440.0)
            mult = round(pk)
            cents = 1200 * np.log2(pk / mult)
            checked_patches += 1
            assert float(np.abs(seg).max()) > 0.02 and mult >= 1, (bank, k, pk)
            assert abs(cents) < 1.0, (bank, k, pk, cents)
    print("tuning patches", checked_patches)
    assert checked_patches >= 3


def test_state_bytes(port: Port) -> None:
    """The engine's state is under the union's size (DUST's 16 KB buffer); six_op.h's layout, per voice 0.0x KB."""
    print("six_op state bytes", port.six_op_size, "union", port.engine_union_size)
    assert port.six_op_size < port.engine_union_size


def test_note_on_and_off_shape_the_sound(port: Port) -> None:
    """The gate is read: the same patch with the gate held to the end and with it low from sub-block 150 differ after it
    (the release stage), and a patch that sustains is quieter in its release than held (organ-like banks aside)."""
    held = six_op(port, 0, note_stream(60.0, patch_h(10), 0.5, 0.5, 1500, 0, 1))[0]
    released = six_op(port, 0, note_stream(60.0, patch_h(10), 0.5, 0.5, 150, 1350, 1))[
        0
    ]
    n = 16 * 150
    assert same(held[:n], released[:n])
    assert not same(held[n:], released[n:])
