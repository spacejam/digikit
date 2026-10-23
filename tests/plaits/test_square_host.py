"""PLAITS model 23, SQUARE (modules/machines/plaits/src/dsp/engines/square.c): the C99 port of Plaits' ChiptuneEngine in
its clocked mode (the owner's decision: every note-on is a TRIGGER_RISING_EDGE and advances the arpeggiator one note;
no envelope, no unclocked chord mode) against upstream (tests/plaits/oracle/run_oracle.cc, ../eurorack/plaits at 48 kHz,
trigger stream RISING_EDGE | HIGH in the note-on's sub-block and HIGH after, as the wrapper builds it), bit for bit on
OUT (the arpeggiated square voice) and AUX (the NES triangle bass). The arpeggiator is stepped over sequences of
note-ons: every chord and pattern (up, down, up-down, random; ranges 1, 2, 4), fast retriggers (a note-on in every
sub-block), and chord and pattern changes in the middle of an arpeggio. Also: the arpeggio really steps through the
chord's sorted notes, only the rising edge is read, the arpeggiator's note and octave indices stay in range, and it is in
tune."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import pytest
from support.plaits_host import Port, build_port
from support.plaits_oracle import Oracle, Step, build, port_engine, same

Arr = Any
MODEL = 23
HIGH, EDGE = 4, 5  # HIGH, RISING_EDGE | HIGH
CORNERS = [0.0, 0.25, 0.5, 0.75, 1.0]
# a HARMONICS value inside each of the 11 chords (the quantizer: 1.02 x, 11 steps) and a TIMBRE inside each of the 12
# patterns (12 steps)
CHORDS = [(i + 0.5) / 11 / 1.02 for i in range(11)]
PATTERNS = [(i + 0.5) / 12 for i in range(12)]


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


def clocked(
    n: int, every: int, note: float, h: float, t: float, m: float
) -> list[Step]:
    """n sub-blocks, a note-on (the rising edge) every `every` of them, the first at 0."""
    return [(note, h, t, m, EDGE if k % every == 0 else HIGH, 0.8) for k in range(n)]


def check(
    port: Port, oracle: Oracle, steps: list[Step], size: int = 16
) -> tuple[Arr, Arr]:
    got = port_engine(port, "square", steps, size)
    want = oracle.engine(MODEL, steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    return got


@pytest.mark.parametrize("size", [16, 12])
def test_every_chord_and_pattern_equals_upstream(
    port: Port, oracle: Oracle, size: int
) -> None:
    """All 11 chords x 12 patterns (the arpeggio's modes and ranges), MORPH corners, 3 pitches, note-ons 4 and 10
    sub-blocks apart: 24 note-ons, enough to go round every pattern at least once."""
    for h in CHORDS:
        for t in PATTERNS:
            for m, note, every in ((0.0, 48.0, 4), (0.5, 60.0, 10), (1.0, 84.0, 4)):
                check(port, oracle, clocked(240, every, note, h, t, m), size)


def test_knob_corners_equal_upstream(port: Port, oracle: Oracle) -> None:
    for h in CORNERS:
        for t in CORNERS:
            for m in CORNERS:
                for note in (24.0, 60.0, 96.0, 127.0):
                    check(port, oracle, clocked(80, 5, note, h, t, m))


@pytest.mark.parametrize("every", [1, 2, 3, 7, 40])
def test_fast_retriggers_equal_upstream(port: Port, oracle: Oracle, every: int) -> None:
    """A note-on in every sub-block (1/32 notes at 300 BPM are slower than this), and sparse ones; the random mode
    (patterns 9..11) draws one word per note-on."""
    for h in (CHORDS[4], CHORDS[10], CHORDS[1], CHORDS[0]):
        for t in (PATTERNS[1], PATTERNS[4], PATTERNS[7], PATTERNS[9], PATTERNS[11]):
            check(port, oracle, clocked(300, every, 60.0, h, t, 0.5))


def mid_arpeggio_changes(seed: int, n: int = 4000) -> list[Step]:
    """Chord (HARMONICS), pattern (TIMBRE), waveform (MORPH) and pitch changing between note-ons, in the middle of an
    arpeggio, with irregular note-on spacing (also two in a row)."""
    rng = np.random.default_rng(seed)
    steps: list[Step] = []
    cur = (60.0, 0.4, 0.0, 0.5)
    k = 0
    gap = 1
    while k < n:
        if rng.random() < 0.5:
            cur = (
                float(rng.choice([-64.0, 24.0, 48.0, 60.0, 72.0, 100.0, 127.0, 190.0])),
                float(rng.choice([0.0, 1.0, rng.random(), *CHORDS[:3]])),
                float(rng.choice([0.0, 1.0, rng.random(), *PATTERNS[::3]])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
            )
        for j in range(gap):
            steps.append((*cur, EDGE if j == 0 else HIGH, 0.8))
        k += gap
        gap = int(rng.choice([1, 1, 2, 3, 5, 9, 17]))
    return steps


def test_chord_and_pattern_changes_mid_arpeggio_equal_upstream(
    port: Port, oracle: Oracle
) -> None:
    for seed in range(12):
        check(port, oracle, mid_arpeggio_changes(seed))


def test_sweeps_equal_upstream(port: Port, oracle: Oracle) -> None:
    """HARMONICS, TIMBRE and MORPH sweeping while notes arrive every 6 sub-blocks, over the wrapper's note range."""
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
                EDGE if k % 6 == 0 else HIGH,
                0.8,
            )
        )
    check(port, oracle, steps)


def test_extreme_notes_are_finite_and_equal_upstream(
    port: Port, oracle: Oracle
) -> None:
    for note in (-64.0, -20.0, 0.0, 120.0, 150.0, 190.0):
        for h, t, m in ((0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 0), (0.5, 0.67, 0.5)):
            check(port, oracle, clocked(200, 7, note, h, t, m))


def test_only_the_rising_edge_and_no_accent(port: Port) -> None:
    """The clocked branch reads RISING_EDGE only: HIGH or LOW after the edge, and any accent, render the same stream."""
    base = clocked(600, 9, 60.0, CHORDS[4], PATTERNS[2], 0.6)
    a = port_engine(port, "square", base)
    for variant in (
        [(*s[:4], 1 if s[4] == EDGE else 0, s[5]) for s in base],
        [(*s[:4], 1 if s[4] == EDGE else 4, s[5]) for s in base],
        [(*s[:5], 0.0) for s in base],
        [(*s[:5], 1.0) for s in base],
    ):
        b = port_engine(port, "square", variant)
        assert same(b[0], a[0]) and same(b[1], a[1])


def blocks_f0(x: Arr, n: int, block: int) -> list[float]:
    """The frequency of the last 3/4 of each of the `n` blocks, from zero-crossing times (rising)."""
    out = []
    for k in range(n):
        seg = x[k * block + block // 4 : (k + 1) * block].astype(np.float64)
        s = np.signbit(seg)
        idx = np.nonzero(s[:-1] & ~s[1:])[0]
        assert len(idx) >= 3, k
        a = idx[0] - seg[idx[0]] / (seg[idx[0] + 1] - seg[idx[0]])
        b = idx[-1] - seg[idx[-1]] / (seg[idx[-1] + 1] - seg[idx[-1]])
        out.append(48000.0 * (len(idx) - 1) / (b - a))
    return out


def test_arpeggio_steps_through_the_chord(port: Port) -> None:
    """A four-note chord (HARMONICS in chord 4), pattern UP, range 1 (TIMBRE 0), note-ons 100 sub-blocks apart: the
    voice's pitch goes up the chord's sorted notes (one per note-on) and wraps; pattern DOWN goes down; a note-on-free
    stretch keeps its note."""
    note = 60.0
    f0 = 261.6256
    block = 100 * 16
    nb = 12
    x, _ = port_engine(
        port, "square", clocked(nb * 100, 100, note, CHORDS[4], 0.0, 0.5)
    )
    f = blocks_f0(x, nb, block)
    ratios = [fi / f0 for fi in f]
    # chord 4 (the minor seventh: 0, 3, 7, 10 semitones, already sorted); the first note-on clocks
    # the arpeggiator from its power-up note 0 to note 1
    want = [2.0 ** (s / 12.0) for s in (3, 7, 10, 0, 3, 7, 10, 0, 3, 7, 10, 0)]
    assert np.allclose(ratios, want, rtol=3e-3), (ratios, want)
    # DOWN (pattern 3): starts by going from note 0 down: wraps to the top
    x, _ = port_engine(
        port, "square", clocked(nb * 100, 100, note, CHORDS[4], PATTERNS[3], 0.5)
    )
    f = blocks_f0(x, nb, block)
    ratios = [fi / f0 for fi in f]
    want = [2.0 ** (s / 12.0) for s in (10, 7, 3, 0, 10, 7, 3, 0, 10, 7, 3, 0)]
    assert np.allclose(ratios, want, rtol=3e-3), (ratios, want)
    # no note-on: the note stays
    steps = clocked(100, 100, note, CHORDS[4], 0.0, 0.5)
    steps += [(note, CHORDS[4], 0.0, 0.5, HIGH, 0.8)] * 300
    x, _ = port_engine(port, "square", steps)
    f = blocks_f0(x, 4, block)
    assert np.allclose(f, f[0], rtol=1e-3)


def test_indices_in_range(checked: Port) -> None:
    """The arpeggiator's note is in 0..3 and its octave in 0..3 (tables 30, 31), the NES triangle step in 0..31 (32), and the chord ratio rows (table 5)
    are in range, over every chord and pattern, mid-arpeggio changes and fast retriggers."""
    checked.lib.port_index_reset()
    for h in CHORDS:
        for t in PATTERNS:
            port_engine(checked, "square", clocked(160, 2, 60.0, h, t, 0.5))
    for seed in range(20):
        port_engine(checked, "square", mid_arpeggio_changes(seed, 2000))
    for table, lo_ok, hi_ok in ((30, 0, 3), (31, 0, 3), (32, 0, 31), (5, 0, 43)):
        lo, hi = checked.lib.port_index_lo(table), checked.lib.port_index_hi(table)
        print("SQUARE index table", table, lo, hi, checked.lib.port_index_n(table))
        assert checked.lib.port_index_n(table) > 0  # the recorder is hit
        assert lo >= lo_ok and hi <= hi_ok


def test_runs_on_and_is_bounded(port: Port) -> None:
    """Held for 20 s with a note-on every 3 sub-blocks (about 1/16 notes at 120 BPM x 4): finite and bounded."""
    for note, h, t, m in (
        (48.0, 1.0, 0.5, 1.0),
        (96.0, 0.0, 0.9, 0.0),
        (127.0, 0.9, 1.0, 0.5),
    ):
        out, aux = port_engine(port, "square", clocked(60000, 3, note, h, t, m))
        assert np.isfinite(out).all() and np.isfinite(aux).all()
        assert float(np.abs(out).max()) <= 1.6 and float(np.abs(aux).max()) <= 1.6


def test_in_tune(port: Port) -> None:
    """HARMONICS 0 (a one-note chord, ratio exactly 1), TIMBRE 0 (up, range 1: every note-on plays the root): OUT's
    strongest line is the note: 440 Hz at note 69 within 1 cent (a0 from exactly 48 kHz); the bass is an octave below."""
    out, aux = port_engine(port, "square", clocked(6000, 600, 69.0, 0.0, 0.0, 0.5))

    def peak(x: Arr, lo_hz: float, hi_hz: float) -> float:
        x = x[4800:].astype(np.float64) * np.hanning(len(x) - 4800)
        spec = np.abs(np.fft.rfft(x, 1 << 21))
        lo, hi = (int(f * (1 << 21) / 48000) for f in (lo_hz, hi_hz))
        k = lo + int(np.argmax(spec[lo:hi]))
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        return (k + 0.5 * (a - c) / (a - 2 * b + c)) * 48000 / (1 << 21)

    f = peak(out, 400, 480)
    print(f"SQUARE note 69: {f:.3f} Hz")
    assert abs(1200 * math.log2(f / 440.0)) < 1.0
    fb = peak(aux, 200, 240)
    assert (
        abs(1200 * math.log2(fb / 220.0)) < 6.0
    )  # the 5-bit NES triangle's staircase pulls its line a little


def test_state_bytes(port: Port) -> None:
    print("square state bytes", port.square_size, "union", port.engine_union_size)
    assert port.square_size <= port.engine_union_size
