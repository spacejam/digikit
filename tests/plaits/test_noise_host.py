"""PLAITS model 9, NOISE (modules/machines/plaits/src/dsp/engines/noise.c): the C99 port of Plaits' NoiseEngine (two
clocked noise sources through a multimode filter and two band-pass filters) against upstream (tests/plaits/oracle/
run_oracle.cc, ../eurorack/plaits at 48 kHz), bit for bit on OUT and AUX, over the knob corners, notes over the
wrapper's whole range, sweeps of all four parameters with irregular note-ons (the sync), sub-blocks of 16 and 12.
stmlib's global Random is one state per engine seeded 0x21 (the oracle seeds 0x21 when it selects the engine), so a
single voice reproduces upstream's sequence exactly. The one deliberate difference from upstream's text is the oracle's
per-sample 1 / Sqrt(x): Newton's reciprocal square root of the port (tests/support/plaits_oracle.py rsqrt_noise_engine,
fail-closed). Also: the trigger matters only through RISING_EDGE (the wrapper's note-on), the engine is bounded and finite
at the extremes, and it is in tune."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import pytest
from support.plaits_host import FP, Params, Port, build_port
from support.plaits_oracle import Oracle, Step, build, port_engine, same

Arr = Any
RISING_HIGH, HIGH, LOW = 1 | 4, 4, 0
V100 = float(np.float32(np.float32(100 << 8) * np.float32(3.0757874e-05)))
CORNERS = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.fixture(scope="module")
def port(tmp_path_factory: pytest.TempPathFactory) -> Port:
    return build_port(tmp_path_factory.mktemp("port"))


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
    got = port_engine(port, "noise", steps, size)
    want = oracle.engine(9, steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    return got


@pytest.mark.parametrize("size", [16, 12])
def test_equals_upstream_at_knob_corners(port: Port, oracle: Oracle, size: int) -> None:
    """Every HARMONICS, TIMBRE, MORPH corner at 4 pitches, note-ons 1, 2 and 23 sub-blocks apart."""
    for h in CORNERS:
        for t in CORNERS:
            for m in (0.0, 0.5, 1.0):
                for note in (24.0, 60.0, 96.0, 127.0):
                    steps = (
                        notes_on(1, 4, note, h, t, m)
                        + notes_on(2, 8, note, h, t, m)
                        + notes_on(23, 40, note, h, t, m)
                    )
                    check(port, oracle, steps, size)


def test_long_runs_equal_upstream(port: Port, oracle: Oracle) -> None:
    """A note-on in every sub-block, and 1/32 notes at 300, 120 and 60 BPM, 6 s each, at a high clock and resonance."""
    for every in (1, 75, 188, 375):
        check(port, oracle, notes_on(every, 18000, 48.0, 0.3, 0.7, 0.6))


def test_sweeps_with_note_ons_equal_upstream(port: Port, oracle: Oracle) -> None:
    """All four parameters moving, note-ons at irregular spacing, notes over the wrapper's range -64..190."""
    rng = np.random.default_rng(9)
    steps: list[Step] = []
    nxt = 0
    n = 9000
    for k in range(n):
        x = k / n
        trig = RISING_HIGH if k == nxt else HIGH
        if k == nxt:
            nxt = k + int(rng.choice([1, 2, 3, 7, 30, 120, 400]))
        steps.append(
            (
                -64.0 + 254.0 * (0.5 - 0.5 * math.cos(6 * math.pi * x)),
                0.5 + 0.5 * math.sin(11 * x),
                x,
                0.5 + 0.5 * math.cos(7 * x),
                trig,
                V100,
            )
        )
    check(port, oracle, steps)


def test_parameter_jumps_equal_upstream(port: Port, oracle: Oracle) -> None:
    """Knobs and notes jumping between the extremes every few sub-blocks (the interpolators and the filters'
    coefficients move by their whole range in one block)."""
    rng = np.random.default_rng(90)
    steps: list[Step] = []
    for k in range(6000):
        if k % 5 == 0:
            cur = (
                float(rng.choice([-64.0, 0.0, 33.0, 60.0, 127.0, 190.0])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
            )
        steps.append((*cur, RISING_HIGH if k % 17 == 0 else HIGH, V100))
    check(port, oracle, steps)


def test_extreme_notes_are_finite_and_equal_upstream(
    port: Port, oracle: Oracle
) -> None:
    for note in (-64.0, -20.0, 0.0, 120.0, 150.0, 190.0):
        for h, t, m in ((0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 0), (0.5, 0.67, 0.5)):
            check(port, oracle, notes_on(5, 200, note, h, t, m))


def test_trigger_reads_only_the_rising_edge(port: Port) -> None:
    """The engine reads `trigger & RISING_EDGE` (the sync) and `trigger & UNPATCHED` (not ported: the wrapper never sets
    it): HIGH or LOW after the edge is the same, and RISING_EDGE without HIGH is the same as with it; the edge itself
    changes the stream (it restarts the clocks)."""
    base = notes_on(40, 400, 60.0, 0.3, 0.6, 0.6)
    a = port_engine(port, "noise", base)
    low = [(*s[:4], LOW if s[4] == HIGH else s[4], s[5]) for s in base]
    bare = [(*s[:4], 1 if s[4] == RISING_HIGH else s[4], s[5]) for s in base]
    for variant in (low, bare):
        b = port_engine(port, "noise", variant)
        assert same(b[0], a[0]) and same(b[1], a[1])
    none = port_engine(port, "noise", [(*s[:4], HIGH, s[5]) for s in base])
    assert not same(none[0], a[0])


def test_accent_changes_nothing(port: Port) -> None:
    """NOISE does not read the accent (upstream's EngineParameters.accent is unused here)."""
    base = notes_on(40, 400, 60.0, 0.3, 0.6, 0.6)
    a = port_engine(port, "noise", base)
    for acc in (0.0, 1.0):
        b = port_engine(port, "noise", [(*s[:5], acc) for s in base])
        assert same(b[0], a[0]) and same(b[1], a[1])


def test_runs_on_and_is_bounded(port: Port) -> None:
    """Noise at the loudest settings for 20 s, note-ons every block: finite, and the peak of the last 5 s is no larger
    than 1.5x the peak of the 5 s before (no runaway; the resonant filters are gain-compensated)."""
    for every, note, m in ((1, 48.0, 1.0), (75, 96.0, 0.9), (100000, 24.0, 1.0)):
        out, aux = port_engine(port, "noise", notes_on(every, 60000, note, 0.5, 1.0, m))
        assert np.isfinite(out).all() and np.isfinite(aux).all()
        for x in (out, aux):
            p3 = float(np.abs(x[16 * 30000 : 16 * 45000]).max())
            p4 = float(np.abs(x[16 * 45000 :]).max())
            print(f"every {every} note {note} morph {m}: peaks {p3:.2f} then {p4:.2f}")
            assert p4 <= 1.5 * p3 + 1e-3


def peak_hz(x: Arr, lo_hz: float, hi_hz: float) -> float:
    x = x[4800:].astype(np.float64) * np.hanning(len(x) - 4800)
    spec = np.abs(np.fft.rfft(x, 1 << 21))
    lo, hi = (int(f * (1 << 21) / 48000) for f in (lo_hz, hi_hz))
    k = lo + int(np.argmax(spec[lo:hi]))
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * 48000 / (1 << 21)


def test_resonance_in_tune(port: Port) -> None:
    """MORPH 1 (maximum resonance), HARMONICS 0.5 (the second filter at the same note), TIMBRE 1 (a fast clock): the
    filters ring at the note: AUX's peak for note 69 between 400 and 480 Hz is 440 Hz within 5 cents (a0 from exactly
    48 kHz; the band-pass's own width and the tangent approximation limit the figure)."""
    aux = port_engine(port, "noise", notes_on(100000, 18000, 69.0, 0.5, 1.0, 1.0))[1]
    f = peak_hz(aux, 400, 480)
    print(f"noise note 69: {f:.3f} Hz")
    assert abs(1200 * math.log2(f / 440.0)) < 5.0


def test_state_bytes(port: Port) -> None:
    """The engine's per-voice state (two clocked noises, three filters, four interpolator states, the generator)."""
    print("noise state bytes", port.noise_size, "union", port.engine_union_size)
    assert port.noise_size <= port.engine_union_size


def test_pl_rsqrt_within_measured_bound(port: Port) -> None:
    """The port's reciprocal square root (Newton, no sqrt instruction on the core) against the true one over the
    engine's domain ((0.5 + q) * 40 * f0 from 1e-8 to 1e3) and wider: at most 2.25 ulp (the measured worst over 1.5 million points is 2.18); 0 and negative inputs give 0."""
    port.lib.port_pl_rsqrt.restype = ctypes.c_float
    port.lib.port_pl_rsqrt.argtypes = [ctypes.c_float]
    assert port.lib.port_pl_rsqrt(0.0) == 0.0 and port.lib.port_pl_rsqrt(-1.0) == 0.0
    rng = np.random.default_rng(1)
    xs = np.concatenate(
        [np.exp2(np.linspace(-40, 40, 8001)), np.exp2(rng.uniform(-40, 40, 100000))]
    ).astype(np.float32)
    got = np.array([port.lib.port_pl_rsqrt(float(x)) for x in xs], dtype=np.float32)
    want = (1.0 / np.sqrt(xs.astype(np.float64))).astype(np.float32)
    worst = float(np.max(np.abs(got - want) / np.spacing(want)))
    print(f"pl_rsqrt worst error {worst:.2f} ulp")
    assert worst <= 2.25


def test_tracks_seed_different_noise(port: Port) -> None:
    """noise_seed(track): track 0 keeps upstream's 0x21 (bit-exact with the oracle above); another track renders other
    noise (two NOISE tracks are not identical) and the same track renders the same noise again."""

    def render(track: int | None) -> Arr:
        st = ctypes.create_string_buffer(port.noise_size)
        port.lib.noise_init(st)
        if track is not None:
            port.lib.noise_seed.argtypes = [ctypes.c_void_p, ctypes.c_int]
            port.lib.noise_seed(st, track)
        out = np.zeros(16 * 40, dtype=np.float32)
        aux = np.zeros_like(out)
        for k in range(40):
            p = Params(60.0, 0.5, 1.0, 0.5, RISING_HIGH if k == 0 else HIGH, V100)
            port.lib.noise_render(
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
