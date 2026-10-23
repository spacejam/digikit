"""PLAITS model 22, STRING MACHINE (modules/machines/plaits/src/dsp/engines/string_machine.c, lib/ensemble.c): the C99
port of Plaits' StringMachineEngine against upstream (tests/plaits/oracle/run_oracle.cc, ../eurorack/plaits at 48 kHz),
bit for bit on OUT and AUX: the four divide-down voices of the chord (HARMONICS), the registration (MORPH), the pair of
low-pass filters and the ensemble (TIMBRE: cutoff, amount, depth) with its 1024-float circular delay buffer. Also: the
engine ignores the trigger and the accent, the registration row and the ensemble's delay offset stay in range, it is
bounded when held, in tune, and the state sizes are what the report says."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import pytest
from support.plaits_host import Port, build_port
from support.plaits_oracle import Oracle, Step, build, port_engine, same

Arr = Any
MODEL = 22
HIGH, EDGE = 4, 5  # HIGH, RISING_EDGE | HIGH
CORNERS = [0.0, 0.25, 0.5, 0.75, 1.0]
# a HARMONICS value inside each of the 11 chords (the quantizer: 1.02 x, 11 steps)
CHORDS = [(i + 0.5) / 11 / 1.02 for i in range(11)]


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
    """n sub-blocks of one setting, a note-on (the rising edge) at the first."""
    return [(note, h, t, m, EDGE if k == 0 else HIGH, 0.8) for k in range(n)]


def check(
    port: Port, oracle: Oracle, steps: list[Step], size: int = 16
) -> tuple[Arr, Arr]:
    got = port_engine(port, "string_machine", steps, size)
    want = oracle.engine(MODEL, steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    return got


@pytest.mark.parametrize("size", [16, 12])
def test_knob_corners_equal_upstream(port: Port, oracle: Oracle, size: int) -> None:
    for h in CORNERS:
        for t in CORNERS:
            for m in CORNERS:
                for note in (24.0, 60.0, 96.0, 127.0):
                    check(port, oracle, held(120, note, h, t, m), size)


def test_every_chord_and_registration_equals_upstream(
    port: Port, oracle: Oracle
) -> None:
    """All 11 chords x the 11 registrations (MORPH at each table row and between rows), TIMBRE at the three ensemble
    regimes (dry at 0.5, wet at the ends)."""
    for h in CHORDS:
        for k in range(21):
            for t in (0.0, 0.5, 0.9):
                check(port, oracle, held(60, 60.0, h, t, k / 20))


def irregular(seed: int, n: int = 4000) -> list[Step]:
    """Chord, registration, filter and pitch jumping between sub-blocks (also several in a row), with note-ons at random."""
    rng = np.random.default_rng(seed)
    steps: list[Step] = []
    cur = (60.0, 0.4, 0.3, 0.5)
    for _ in range(n):
        if rng.random() < 0.15:
            cur = (
                float(rng.choice([-64.0, 24.0, 48.0, 60.0, 72.0, 100.0, 127.0, 190.0])),
                float(rng.choice([0.0, 1.0, rng.random(), *CHORDS[:3]])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
            )
        steps.append((*cur, EDGE if rng.random() < 0.1 else HIGH, rng.random()))
    return steps


def test_jumps_equal_upstream(port: Port, oracle: Oracle) -> None:
    for seed in range(12):
        check(port, oracle, irregular(seed))


def test_sweeps_equal_upstream(port: Port, oracle: Oracle) -> None:
    """HARMONICS, TIMBRE and MORPH sweeping over the wrapper's note range: the ensemble's delay depth moves with TIMBRE."""
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


def test_extreme_notes_are_finite_and_equal_upstream(
    port: Port, oracle: Oracle
) -> None:
    for note in (-64.0, -20.0, 0.0, 120.0, 150.0, 190.0):
        for h, t, m in ((0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 0), (0.5, 0.67, 0.5)):
            check(port, oracle, held(200, note, h, t, m))


def test_trigger_and_accent_change_nothing(port: Port) -> None:
    """Sustained voices: the trigger bits and the accent are not read."""
    base = [
        (60.0, CHORDS[4], 0.7, 0.4, EDGE if k % 9 == 0 else HIGH, 0.8)
        for k in range(600)
    ]
    a = port_engine(port, "string_machine", base)
    for variant in (
        [(*s[:4], 0, s[5]) for s in base],
        [(*s[:4], 1, s[5]) for s in base],
        [(*s[:5], 0.0) for s in base],
        [(*s[:5], 1.0) for s in base],
    ):
        b = port_engine(port, "string_machine", variant)
        assert same(b[0], a[0]) and same(b[1], a[1])


def test_indices_in_range(checked: Port) -> None:
    """The registration row (table 33) is in 0..10 and the ensemble's delay offset (34) in 16..368, over knob corners,
    random jumps and a TIMBRE sweep; the chord ratio rows (table 5) are in range."""
    checked.lib.port_index_reset()
    for h in CORNERS:
        for t in CORNERS:
            for m in CORNERS:
                port_engine(checked, "string_machine", held(40, 60.0, h, t, m))
    for seed in range(10):
        port_engine(checked, "string_machine", irregular(seed, 2000))
    for table, lo_ok, hi_ok in ((33, 0, 10), (34, 16, 368), (5, 0, 43)):
        lo, hi = checked.lib.port_index_lo(table), checked.lib.port_index_hi(table)
        print(
            "STRING MACHINE index table", table, lo, hi, checked.lib.port_index_n(table)
        )
        assert checked.lib.port_index_n(table) > 0  # the recorder is hit
        assert lo >= lo_ok and hi <= hi_ok


def test_runs_on_and_is_bounded(port: Port) -> None:
    """Held for 20 s: finite and bounded (the ensemble's delay buffer wraps about 900 times)."""
    for note, h, t, m in (
        (48.0, 1.0, 1.0, 1.0),
        (96.0, 0.0, 0.0, 0.0),
        (127.0, 0.9, 0.5, 0.5),
    ):
        out, aux = port_engine(port, "string_machine", held(60000, note, h, t, m))
        assert np.isfinite(out).all() and np.isfinite(aux).all()
        assert float(np.abs(out).max()) <= 2.0 and float(np.abs(aux).max()) <= 2.0


def test_in_tune(port: Port) -> None:
    """HARMONICS 0 (a one-note chord, ratio exactly 1), TIMBRE 0.5 (the ensemble dry): the strongest line of OUT
    is the note, about 0.998 of 440 Hz at note 69 (Plaits detunes the divide-down by 0.998: -3.5 cents), within 1 cent
    of that, with a0 from exactly 48 kHz."""
    out, _ = port_engine(port, "string_machine", held(6000, 69.0, 0.0, 0.5, 0.0))
    x = out[4800:].astype(np.float64) * np.hanning(len(out) - 4800)
    spec = np.abs(np.fft.rfft(x, 1 << 21))
    lo, hi = (int(f * (1 << 21) / 48000) for f in (400, 480))
    k = lo + int(np.argmax(spec[lo:hi]))
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    f = (k + 0.5 * (a - c) / (a - 2 * b + c)) * 48000 / (1 << 21)
    print(f"STRING MACHINE note 69: {f:.3f} Hz")
    assert abs(1200 * math.log2(f / (440.0 * 0.998))) < 1.0


def test_state_bytes(port: Port) -> None:
    print(
        "string machine state bytes",
        port.string_machine_size,
        "union",
        port.engine_union_size,
    )
    assert port.string_machine_size <= port.engine_union_size
