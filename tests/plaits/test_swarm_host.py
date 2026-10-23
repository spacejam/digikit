"""PLAITS model 8, SWARM (modules/machines/plaits/src/dsp/engines/swarm.c): the C99 port of Plaits' SwarmEngine (eight grains,
each a polyBLEP sawtooth on OUT and a magic-circle sine on AUX under a random-length grain envelope; the grains are lanes of
arrays of eight) against upstream (tests/plaits/oracle/run_oracle.cc, ../eurorack/plaits at 48 kHz), bit for bit on OUT and
AUX, over the knob corners, sweeps and jumps of all four parameters with irregular note-ons (the burst), sub-blocks of 16 and
12. stmlib's global Random is one state per engine seeded 0x21 (the oracle seeds 0x21 when it selects the engine). No
deliberate difference from upstream's text. Both of upstream's modes are tested bit for bit (the oracle is driven with the same trigger bits):
the patched one (burst: a note-on restarts the grains) and, since the owner's decision of 2026-10-22, the continuous one
(TRIGGER_UNPATCHED), which is what the wrapper gives model 8 on every block (test_wrapper_runs_swarm_unpatched_whatever_the_note_ons)."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import pytest
from support.plaits_host import FP, Frame, Params, Port, build_port
from support.plaits_oracle import Oracle, Step, build, port_engine, same

Arr = Any
MODEL = 8
RISING_HIGH, HIGH = 1 | 4, 4
UNPATCHED = (
    2  # what the wrapper gives model 8 on every block (the owner's continuous mode)
)
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
    got = port_engine(port, "swarm", steps, size)
    want = oracle.engine(MODEL, steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    return got


@pytest.mark.parametrize("size", [16, 12])
def test_equals_upstream_at_knob_corners(port: Port, oracle: Oracle, size: int) -> None:
    """Every HARMONICS, TIMBRE, MORPH corner at 4 pitches, note-ons 1, 7 and 400 sub-blocks apart (several hundred grains)."""
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
                float(rng.choice([0.0, 0.49999, 0.5, 0.5001, 1.0, rng.random()])),
            )
        steps.append(
            (*cur, RISING_HIGH if rng.random() < 0.1 else HIGH, float(rng.random()))
        )
    return steps


def test_jumps_equal_upstream(port: Port, oracle: Oracle) -> None:
    for seed in range(12):
        check(port, oracle, irregular(seed))


def test_sweeps_equal_upstream(port: Port, oracle: Oracle) -> None:
    """All four parameters sweeping (MORPH across the 0.5..1 region where grains stop overlapping, size_ratio around 1) over
    the wrapper's note range."""
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
    """In the patched (burst) mode the trigger's HIGH and LOW bits (and the accent) are not read: only RISING_EDGE (the burst)."""
    base = notes_on(9, 600, 60.0, 0.5, 0.8, 0.3)
    a = port_engine(port, "swarm", base)
    for variant in (
        [(*s[:4], 5 if s[4] == 5 else 0, s[5]) for s in base],
        [(*s[:4], 1 if s[4] == 5 else 0, s[5]) for s in base],
        [(*s[:5], 0.0) for s in base],
        [(*s[:5], 1.0) for s in base],
    ):
        b = port_engine(port, "swarm", variant)
        assert same(b[0], a[0]) and same(b[1], a[1])
    c = port_engine(port, "swarm", notes_on(100000, 600, 60.0, 0.5, 0.8, 0.3))
    assert not same(c[0], a[0])  # the burst does change the sequence


def test_indices_in_range(checked: Port) -> None:
    """The pitch table at the widest spread (48 semitones x rank up to 4/3 = 64 above and below) is read inside its 256 entries,
    through the recorder of port_sizes.c that the other engines use. The grain window's Sine(0.5 * phase + 1.25) takes a
    phase in [0.75, 1.75] (positive, wrapped), so it never leaves the sine table."""
    checked.lib.port_index_reset()
    for h in CORNERS:
        for t in CORNERS:
            for m in CORNERS:
                port_engine(checked, "swarm", notes_on(50, 200, 60.0, h, t, m))
    for seed in range(6):
        port_engine(checked, "swarm", irregular(seed, 2000))
    lo, hi, n = (
        checked.lib.port_index_lo(38),
        checked.lib.port_index_hi(38),
        checked.lib.port_index_n(38),
    )
    print("SWARM pitch index", lo, hi, n)
    assert n > 0  # the recorder is hit
    assert lo >= 128 - 64 and hi <= 128 + 64  # inside the table's 256 entries


def test_runs_on_and_is_bounded(port: Port) -> None:
    """Widest spread, densest grains, held 20 s at a low, a mid and a high note: finite, bounded (8 grains of 1/8)."""
    for note, h, t, m in (
        (24.0, 1.0, 1.0, 0.0),
        (60.0, 1.0, 1.0, 1.0),
        (127.0, 0.5, 0.5, 0.2),
    ):
        out, aux = port_engine(port, "swarm", notes_on(100000, 60000, note, h, t, m))
        assert np.isfinite(out).all() and np.isfinite(aux).all()
        print(
            f"note {note} h {h} t {t} m {m}: peaks {np.abs(out).max():.2f} {np.abs(aux).max():.2f}"
        )
        assert float(np.abs(out).max()) < 4.0 and float(np.abs(aux).max()) < 4.0


def test_state_bytes(port: Port) -> None:
    """SWARM's state is 15 arrays of eight floats and the generator: 484 bytes (not the largest engine: the union stays DUST's)."""
    print("swarm state bytes", port.swarm_size, "union", port.engine_union_size)
    assert port.swarm_size == 15 * 8 * 4 + 4 == 484
    assert port.swarm_size < port.engine_union_size


def test_tracks_seed_different_grains(port: Port) -> None:
    """swarm_seed(track): track 0 keeps upstream's 0x21 (bit-exact with the oracle above); another track renders other
    grains and the same track renders the same again."""

    def render(track: int | None) -> Arr:
        st = ctypes.create_string_buffer(port.swarm_size)
        port.lib.swarm_init(st)
        if track is not None:
            port.lib.swarm_seed.argtypes = [ctypes.c_void_p, ctypes.c_int]
            port.lib.swarm_seed(st, track)
        out = np.zeros(16 * 40, dtype=np.float32)
        aux = np.zeros_like(out)
        for k in range(40):
            p = Params(60.0, 0.5, 0.8, 0.5, RISING_HIGH if k == 0 else HIGH, V100)
            port.lib.swarm_render(
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


def unpatched(steps: list[Step], edges: bool = False) -> list[Step]:
    """The same steps with TRIGGER_UNPATCHED (continuous mode); edges keeps the RISING_EDGE bit at the note-ons."""
    return [(*s[:4], UNPATCHED | (1 if edges and s[4] & 1 else 0), s[5]) for s in steps]


@pytest.mark.parametrize("size", [16, 12])
def test_continuous_mode_equals_upstream(port: Port, oracle: Oracle, size: int) -> None:
    """The mode the wrapper runs (TRIGGER_UNPATCHED: a fresh grain rate 0.5 to 2.0 each grain, no burst): knob corners, jumps and a
    sweep against upstream, bit for bit, also with a RISING_EDGE added at the note-ons (a burst start inside the unpatched mode)."""
    for h in CORNERS:
        for t in (0.0, 0.5, 1.0):
            for m in CORNERS:
                for note, every in ((24.0, 400), (60.0, 7), (127.0, 400)):
                    steps = notes_on(every, 420, note, h, t, m)
                    check(port, oracle, unpatched(steps), size)
                    check(port, oracle, unpatched(steps, True), size)
    for seed in range(6):
        check(port, oracle, unpatched(irregular(seed), True), size)


def test_continuous_mode_does_not_slow_down(port: Port) -> None:
    """Held 60 s with no note-on after the first: the grains keep their rate (burst mode slows with each grain, continuous does not):
    the output of the last 10 s is as busy as the first 10 s (RMS within 2x), and bounded."""
    steps = unpatched(notes_on(10**9, 180000, 60.0, 0.5, 0.8, 0.8))
    out, aux = port_engine(port, "swarm", steps)
    n = 16 * 30000  # 10 s of 48 kHz
    first, last = (
        float(np.sqrt(np.mean(out[:n] ** 2))),
        float(np.sqrt(np.mean(out[-n:] ** 2))),
    )
    print(f"continuous rms first 10 s {first:.4f} last 10 s {last:.4f}")
    assert 0.5 < last / first < 2.0
    assert np.isfinite(out).all() and float(np.abs(out).max()) < 4.0


def test_wrapper_runs_swarm_unpatched_whatever_the_note_ons(port: Port) -> None:
    """plaits.c gives model 8 TRIGGER_UNPATCHED alone on every block: a note-on neither bursts nor changes the swarm, so a voice
    with note-ons every 5 blocks renders exactly what a voice with only the first does (model 0 would differ)."""

    def render(every: int, model: int) -> Arr:
        v = port.voice()
        f = Frame(
            60.0 * 256,
            0x4000,
            model * 256.0,
            0.5 * 0x7F00,
            0.6 * 0x7F00,
            0.5 * 0x7F00,
            0.0,
            100.0 * 256,
            1,
        )
        return np.concatenate(
            [
                port.render(v, f, 1 if (k == 0 or (every and k % every == 0)) else 0, 1)
                for k in range(200)
            ]
        )

    assert same(render(5, 8), render(0, 8))
    assert not same(render(5, 9), render(0, 9))
