"""PLAITS model 10, DUST (modules/machines/plaits/src/dsp/engines/dust.c): the C99 port of Plaits' ParticleEngine (six
particles, the low-pass and the diffuser on the FxEngine with its 8192 x 16-bit delay memory, packed two to a word) against
upstream (tests/plaits/oracle/run_oracle.cc, ../eurorack/plaits at 48 kHz), bit for bit on OUT and AUX, over the knob corners,
sweeps of all four parameters with irregular note-ons (the sync), sub-blocks of 16 and 12. stmlib's global Random is one state
per engine seeded 0x21 (the oracle seeds 0x21 when it selects the engine). The one deliberate difference from upstream's text:
the particle's `0.5f / Sqrt(q * f * Sqrt(density))` uses the port's Newton reciprocal square root (tests/support/plaits_oracle.py
dust_override, fail-closed); a test bounds it against the true formula. Also: only RISING_EDGE of the trigger matters, the
accent is not read, the delay offsets stay in their lines, the engine is bounded and finite, tracks seed different noise."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import pytest
from support.plaits_host import FP, Params, Port, build_port
from support.plaits_oracle import Oracle, Step, build, port_engine, same

Arr = Any
MODEL = 10
RISING_HIGH, HIGH = 1 | 4, 4
V100 = float(np.float32(np.float32(100 << 8) * np.float32(3.0757874e-05)))
CORNERS = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture(scope="module")
def port(tmp_path_factory: pytest.TempPathFactory) -> Port:
    return build_port(tmp_path_factory.mktemp("port"))


@pytest.fixture(scope="module")
def checked(tmp_path_factory: pytest.TempPathFactory) -> Port:
    p = build_port(tmp_path_factory.mktemp("port-idx"), ("PL_INDEX_CHECK",))
    for name in ("port_index_lo", "port_index_hi", "port_index_n"):
        getattr(p.lib, name).argtypes = [ctypes.c_int]
    return p


@pytest.fixture(scope="module")
def oracle(tmp_path_factory: pytest.TempPathFactory) -> Oracle:
    return build(tmp_path_factory.mktemp("run-oracle"))


def notes_on(
    every: int, n: int, note: float, h: float, t: float, m: float
) -> list[Step]:
    """n sub-blocks, a note-on every `every` sub-blocks (RISING_EDGE | HIGH there, HIGH between)."""
    return [
        (note, h, t, m, RISING_HIGH if k % every == 0 else HIGH, V100) for k in range(n)
    ]


def check(
    port: Port, oracle: Oracle, steps: list[Step], size: int = 16
) -> tuple[Arr, Arr]:
    got = port_engine(port, "dust", steps, size)
    want = oracle.engine(MODEL, steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    return got


@pytest.mark.parametrize("size", [16, 12])
def test_equals_upstream_at_knob_corners(port: Port, oracle: Oracle, size: int) -> None:
    """Every HARMONICS, TIMBRE, MORPH corner at 4 pitches, note-ons 1, 7 and 400 sub-blocks apart (the diffuser's buffer
    wraps about once in 8192 samples: 400 blocks of 16 are 6,400)."""
    for h in CORNERS:
        for t in CORNERS:
            for m in CORNERS:
                for note, every in ((24.0, 400), (60.0, 7), (96.0, 1), (127.0, 400)):
                    check(port, oracle, notes_on(every, 420, note, h, t, m), size)


def irregular(seed: int, n: int = 4000) -> list[Step]:
    """Knobs and pitch jumping between sub-blocks (also several in a row), with note-ons at random."""
    rng = np.random.default_rng(seed)
    steps: list[Step] = []
    cur = (60.0, 0.4, 0.6, 0.3)
    for _ in range(n):
        if rng.random() < 0.15:
            cur = (
                float(rng.choice([-64.0, 24.0, 48.0, 60.0, 72.0, 100.0, 127.0, 190.0])),
                float(rng.choice([0.0, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.49999, 0.5, 1.0, rng.random()])),
            )
        steps.append(
            (*cur, RISING_HIGH if rng.random() < 0.1 else HIGH, float(rng.random()))
        )
    return steps


def test_jumps_equal_upstream(port: Port, oracle: Oracle) -> None:
    for seed in range(12):
        check(port, oracle, irregular(seed))


def test_sweeps_equal_upstream(port: Port, oracle: Oracle) -> None:
    """All four parameters sweeping (MORPH across 0.5: diffusion to resonance) over the wrapper's note range."""
    steps: list[Step] = []
    n = 12000
    for k in range(n):
        x = k / n
        steps.append(
            (
                -64.0 + 254.0 * (0.5 - 0.5 * math.cos(6 * math.pi * x)),
                0.5 + 0.5 * math.sin(11 * x),
                0.5 + 0.5 * math.cos(13 * x),
                0.5 + 0.5 * math.cos(7 * x),
                RISING_HIGH if k % 997 == 0 else HIGH,
                V100,
            )
        )
    check(port, oracle, steps)


def test_extreme_notes_are_finite_and_equal_upstream(
    port: Port, oracle: Oracle
) -> None:
    for note in (-64.0, -20.0, 0.0, 120.0, 150.0, 190.0):
        for h, t, m in ((0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 0), (0.5, 0.67, 0.5)):
            check(port, oracle, notes_on(50, 300, note, h, t, m))


def test_only_the_rising_edge_matters(port: Port) -> None:
    """The trigger's HIGH and LOW bits (and the accent) are not read: only RISING_EDGE (sync)."""
    base = notes_on(9, 600, 60.0, 0.5, 0.8, 0.3)
    a = port_engine(port, "dust", base)
    for variant in (
        [(*s[:4], 5 if s[4] == 5 else 0, s[5]) for s in base],
        [(*s[:4], 1 if s[4] == 5 else 3 & 2, s[5]) for s in base],
        [(*s[:5], 0.0) for s in base],
        [(*s[:5], 1.0) for s in base],
    ):
        b = port_engine(port, "dust", variant)
        assert same(b[0], a[0]) and same(b[1], a[1])
    c = port_engine(port, "dust", notes_on(100000, 600, 60.0, 0.5, 0.8, 0.3))
    assert not same(c[0], a[0])  # the sync does change the sequence


def test_indices_in_range(checked: Port) -> None:
    """The diffuser's integral delay offsets: ap4 (table 35) inside its 444-sample line, the main delay (36) inside its
    3411-sample line (the LFO value is 0..1), over knob corners, random jumps and a long run."""
    checked.lib.port_index_reset()
    for h in CORNERS:
        for t in CORNERS:
            for m in CORNERS:
                port_engine(checked, "dust", notes_on(50, 200, 60.0, h, t, m))
    for seed in range(6):
        port_engine(checked, "dust", irregular(seed, 2000))
    port_engine(checked, "dust", notes_on(100000, 60000, 60.0, 0.3, 0.9, 0.0))
    # table 37: the delay memory's saturation paths (+1 above 32767, -1 below -32768) are both taken (11,542 times over this
    # sweep), so the bit-exact tests above cover the clamps
    assert checked.lib.port_index_n(37) > 0
    assert checked.lib.port_index_lo(37) == -1 and checked.lib.port_index_hi(37) == 1
    for table, lo_ok, hi_ok in ((35, 400 - 43, 400 + 43), (36, 3070 - 340, 3070 + 340)):
        lo, hi = checked.lib.port_index_lo(table), checked.lib.port_index_hi(table)
        print("DUST index table", table, lo, hi, checked.lib.port_index_n(table))
        assert checked.lib.port_index_n(table) > 0  # the recorder is hit
        assert lo >= lo_ok and hi <= hi_ok
    # base + offset + 1 stays inside the reserved memory (8192 samples)
    assert 4688 + 3070 + 340 + 1 < 8192


def test_runs_on_and_is_bounded(port: Port) -> None:
    """Densest particles, diffusion at its most (MORPH 0) and the longest reverb, held 20 s: finite, bounded."""
    for note, h, t, m in (
        (48.0, 1.0, 1.0, 0.0),
        (127.0, 1.0, 1.0, 1.0),
        (60.0, 0.5, 0.5, 0.2),
    ):
        out, aux = port_engine(port, "dust", notes_on(100000, 60000, note, h, t, m))
        assert np.isfinite(out).all() and np.isfinite(aux).all()
        print(
            f"note {note} h {h} t {t} m {m}: peaks {np.abs(out).max():.2f} {np.abs(aux).max():.2f}"
        )
        assert float(np.abs(out).max()) < 40.0 and float(np.abs(aux).max()) < 40.0


def test_state_bytes(port: Port) -> None:
    """DUST's state is exactly six particles (pre-gain and Svf: 6 x 24 B), the post filter (20), the diffuser's lp, two LFO
    words, write pointer and generator (20), 4,096 words of delay memory (16,384) and the DSP kernel's sample count since
    init (fresh_, 4: lib/dust_sm.s does not clear the memory): 16,572 B, the largest engine, so the union is DUST's; 16
    voices are the 273,984 bytes (17,124 per voice since the DX7 gate state) the `.PKBSS plaits_voice_mem.` of plaits_entry.s reserves (plaits.c checks it at compile
    time, the rig at load)."""
    print("dust state bytes", port.dust_size, "union", port.engine_union_size)
    assert port.dust_size == 6 * 24 + 20 + 20 + 16384 + 4 == 16572
    assert port.engine_union_size == port.dust_size
    assert port.voice_size * 16 == 273984


def test_tracks_seed_different_particles(port: Port) -> None:
    """dust_seed(track): track 0 keeps upstream's 0x21 (bit-exact with the oracle above); another track renders other
    particles and the same track renders the same again."""

    def render(track: int | None) -> Arr:
        st = ctypes.create_string_buffer(port.dust_size)
        port.lib.dust_init(st)
        if track is not None:
            port.lib.dust_seed.argtypes = [ctypes.c_void_p, ctypes.c_int]
            port.lib.dust_seed(st, track)
        out = np.zeros(16 * 40, dtype=np.float32)
        aux = np.zeros_like(out)
        for k in range(40):
            p = Params(60.0, 0.5, 0.8, 0.5, RISING_HIGH if k == 0 else HIGH, V100)
            port.lib.dust_render(
                st,
                ctypes.byref(p),
                out[16 * k :].ctypes.data_as(FP),
                aux[16 * k :].ctypes.data_as(FP),
                16,
            )
        return out

    base = render(None)
    assert same(render(0), base)
    assert not same(render(1), base) and not same(render(1), render(2))
    assert same(render(7), render(7))


def test_rsqrt_gain_within_bound_of_upstream_formula(port: Port) -> None:
    """The particle's pre-gain 0.5 * rsqrt(q * f * (density * rsqrt(density))) against upstream's 0.5 / Sqrt(q * f *
    Sqrt(density)) in double precision, over the engine's domain (density 5e-6..0.06 at the knob extremes, q 0.5..1e6, f
    1e-4..0.25): relative error at most 2e-6 (the measured figure is printed)."""
    port.lib.port_pl_rsqrt.restype = ctypes.c_float
    port.lib.port_pl_rsqrt.argtypes = [ctypes.c_float]
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(40000):
        d = float(np.exp2(rng.uniform(-18, -3)))
        q = float(0.5 + np.exp2(rng.uniform(-10, 20)) * rng.random())
        f = float(np.exp2(rng.uniform(-14, -2)))
        d32 = np.float32(d)
        sd = np.float32(d32 * np.float32(port.lib.port_pl_rsqrt(float(d32))))
        arg = np.float32(np.float32(np.float32(q) * np.float32(f)) * sd)
        got = float(np.float32(0.5) * np.float32(port.lib.port_pl_rsqrt(float(arg))))
        want = 0.5 / math.sqrt(float(q) * float(f) * math.sqrt(float(d32)))
        worst = max(worst, abs(got - want) / want)
    print(f"DUST pre-gain relative error {worst:.2e}")
    assert worst < 2e-6
