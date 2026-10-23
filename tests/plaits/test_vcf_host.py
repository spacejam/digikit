"""PLAITS model 16, VCF-VA (modules/machines/plaits/src/dsp/engines/vcf.c): the C99 port of Plaits'
VirtualAnalogVCFEngine (a variable-shape oscillator and a sub oscillator through a two-stage state variable filter with
soft clipping) against upstream (tests/plaits/oracle/run_oracle.cc, ../eurorack/plaits at 48 kHz), bit for bit on OUT (low
pass) and AUX (high pass), over the knob corners, notes over the wrapper's whole range, sweeps and jumps of all four
parameters, sub-blocks of 16 and 12. The engine runs on between notes and reads neither the trigger nor the accent.
Also: finite and bounded at the extremes, the cutoff's table index stays inside lut_pitch_ratio_high (index recorder,
table 29), and it is in tune."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import pytest
from support.plaits_host import Port, build_port
from support.plaits_oracle import Oracle, Step, build, port_engine, same

Arr = Any
MODEL = 16
HIGH = 4
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


def held(n: int, note: float, h: float, t: float, m: float) -> list[Step]:
    return [(note, h, t, m, HIGH, 0.8)] * n


def check(
    port: Port, oracle: Oracle, steps: list[Step], size: int = 16
) -> tuple[Arr, Arr]:
    got = port_engine(port, "vcf", steps, size)
    want = oracle.engine(MODEL, steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    return got


@pytest.mark.parametrize("size", [16, 12])
def test_equals_upstream_at_knob_corners(port: Port, oracle: Oracle, size: int) -> None:
    """Every HARMONICS, TIMBRE, MORPH corner at 4 pitches, 60 sub-blocks each."""
    for h in CORNERS:
        for t in CORNERS:
            for m in CORNERS:
                for note in (24.0, 60.0, 96.0, 127.0):
                    check(port, oracle, held(60, note, h, t, m), size)


def test_sweeps_equal_upstream(port: Port, oracle: Oracle) -> None:
    """All four parameters moving, notes over the wrapper's range -64..190."""
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
                HIGH,
                0.8,
            )
        )
    check(port, oracle, steps)


def jump_steps(seed: int, n: int = 3000) -> list[Step]:
    """Knobs and notes jumping between the extremes every few sub-blocks."""
    rng = np.random.default_rng(seed)
    steps: list[Step] = []
    cur = (60.0, 0.5, 0.5, 0.5)
    for k in range(n):
        if k % int(rng.integers(1, 6)) == 0:
            cur = (
                float(rng.choice([-64.0, 0.0, 33.0, 60.0, 100.0, 127.0, 190.0])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
            )
        steps.append((*cur, HIGH, 0.8))
    return steps


def test_parameter_jumps_equal_upstream(port: Port, oracle: Oracle) -> None:
    for seed in range(12):
        check(port, oracle, jump_steps(seed))


def test_extreme_notes_are_finite_and_equal_upstream(
    port: Port, oracle: Oracle
) -> None:
    for note in (-64.0, -20.0, 0.0, 120.0, 150.0, 190.0):
        for h, t, m in ((0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 0), (0.5, 0.67, 0.5)):
            check(port, oracle, held(200, note, h, t, m))


def test_trigger_and_accent_change_nothing(port: Port) -> None:
    """The engine reads neither the trigger nor the accent: a note-on every 40 sub-blocks, LOW, or a different accent,
    renders the same stream."""
    base = held(400, 60.0, 0.3, 0.6, 0.6)
    a = port_engine(port, "vcf", base)
    for variant in (
        [(*s[:4], 5 if k % 40 == 0 else 0, s[5]) for k, s in enumerate(base)],
        [(*s[:5], 0.0) for s in base],
        [(*s[:5], 1.0) for s in base],
    ):
        b = port_engine(port, "vcf", variant)
        assert same(b[0], a[0]) and same(b[1], a[1])


def test_runs_on_and_is_bounded(port: Port) -> None:
    """Both outputs are soft-clipped: |x| <= 1 (plus a little for the stage-2 blend) at the loudest settings, 20 s held."""
    for note, h, t, m in (
        (48.0, 1.0, 0.5, 1.0),
        (96.0, 0.0, 0.9, 0.0),
        (127.0, 0.9, 1.0, 0.5),
        (24.0, 0.1, 0.1, 0.2),
    ):
        out, aux = port_engine(port, "vcf", held(60000, note, h, t, m))
        assert np.isfinite(out).all() and np.isfinite(aux).all()
        assert float(np.abs(out).max()) <= 1.5 and float(np.abs(aux).max()) <= 1.001


def test_indices_in_range(checked: Port) -> None:
    """The cutoff's lut_pitch_ratio_high index (table 29) is inside the 256 entries over the corners, a grid of
    notes -64..190 and jump sequences."""
    checked.lib.port_index_reset()
    for h in CORNERS:
        for t in CORNERS:
            for m in CORNERS:
                for note in (-64.0, 0.0, 60.0, 127.0, 190.0):
                    port_engine(checked, "vcf", held(30, note, h, t, m))
    for seed in range(20):
        port_engine(checked, "vcf", jump_steps(seed))
    lo, hi = checked.lib.port_index_lo(29), checked.lib.port_index_hi(29)
    print("VCF-VA cutoff table index:", lo, hi, checked.lib.port_index_n(29))
    assert checked.lib.port_index_n(29) > 0  # the recorder is hit
    assert lo >= 0 and hi <= 255


def test_in_tune(port: Port) -> None:
    """A saw (MORPH 0.25), cutoff well above it (TIMBRE 1) and no resonance (HARMONICS 0.5): OUT's strongest line is the
    note: 440 Hz at note 69 within 1 cent (a0 from exactly 48 kHz)."""
    out = port_engine(port, "vcf", held(6000, 69.0, 0.5, 1.0, 0.25))[0]
    x = out[4800:].astype(np.float64) * np.hanning(len(out) - 4800)
    spec = np.abs(np.fft.rfft(x, 1 << 21))
    lo, hi = (int(f * (1 << 21) / 48000) for f in (400, 480))
    k = lo + int(np.argmax(spec[lo:hi]))
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    f = (k + 0.5 * (a - c) / (a - 2 * b + c)) * 48000 / (1 << 21)
    print(f"VCF-VA note 69: {f:.3f} Hz")
    assert abs(1200 * math.log2(f / 440.0)) < 1.0


def test_state_bytes(port: Port) -> None:
    """The engine's per-voice state: two filters, two oscillators and five interpolator states."""
    print("vcf state bytes", port.vcf_size, "union", port.engine_union_size)
    assert port.vcf_size <= port.engine_union_size
