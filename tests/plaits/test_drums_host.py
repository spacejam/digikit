"""PLAITS models 13, 14 and 15, KICK, SNARE and HIHAT (modules/machines/plaits/src/dsp/engines/{kick,snare,hihat}.c,
lib/drums.h): the C99 ports of Plaits' BassDrumEngine, SnareDrumEngine and HiHatEngine against upstream
(tests/plaits/oracle/run_oracle.cc, ../eurorack/plaits at 48 kHz), bit for bit on OUT and AUX, over strike sequences:
Plaits' trigger as the wrapper builds it (RISING_EDGE | HIGH in a note-on's first sub-block, HIGH after; never UNPATCHED,
so every drum's sustain branch is never taken), repeats down to every sub-block, the 1/32-note spacing, accent at 0,
0.787 (velocity 100) and 1, knob corners and sweeps, and notes over the wrapper's whole range. The one deliberate
difference from upstream's text is the oracle's square root (the SNARE's two level constants: Newton's iteration of
lib/drums.h instead of the hardware vsqrt; tests/support/plaits_oracle.py sqrt_override). Also: energy stays bounded under
repeated strikes, the pitch is in tune (KICK, SNARE), and every note-on strikes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from support.plaits_host import Port, build_port, frame
from support.plaits_oracle import Oracle, Step, build, port_engine, same

Arr = Any
DRUMS = {13: "kick", 14: "snare", 15: "hihat"}
RISING_HIGH, HIGH, LOW = 1 | 4, 4, 0
V100 = float(np.float32(np.float32(100 << 8) * np.float32(3.0757874e-05)))
ACCENTS = (0.0, V100, 1.0)
MODELS = pytest.mark.parametrize("model", sorted(DRUMS))


@pytest.fixture(scope="module")
def port(tmp_path_factory: pytest.TempPathFactory) -> Port:
    return build_port(tmp_path_factory.mktemp("port"))


@pytest.fixture(scope="module")
def oracle(tmp_path_factory: pytest.TempPathFactory) -> Oracle:
    return build(tmp_path_factory.mktemp("run-oracle"))


def strikes(
    every: int, n: int, note: float, h: float, t: float, m: float, accent: float
) -> list[Step]:
    """n sub-blocks, a note-on every `every` sub-blocks (RISING_EDGE | HIGH there, HIGH between)."""
    return [
        (note, h, t, m, RISING_HIGH if k % every == 0 else HIGH, accent)
        for k in range(n)
    ]


def check(
    port: Port, oracle: Oracle, model: int, steps: list[Step], size: int = 16
) -> Arr:
    got = port_engine(port, DRUMS[model], steps, size)
    want = oracle.engine(model, steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    return got[0]


CORNERS = [0.0, 0.25, 0.5, 0.75, 1.0]


@MODELS
@pytest.mark.parametrize("accent", ACCENTS)
@pytest.mark.parametrize("size", [16, 12])
def test_strikes_equal_upstream_at_knob_corners(
    port: Port, oracle: Oracle, model: int, accent: float, size: int
) -> None:
    """Every HARMONICS, TIMBRE, MORPH corner at 4 pitches: strikes 1 sub-block, 1 block, and 23 sub-blocks apart."""
    for h in CORNERS:
        for t in (0.0, 0.5, 1.0):
            for m in (0.0, 0.5, 1.0):
                for note in (24.0, 60.0, 96.0, 127.0):
                    steps = (
                        strikes(1, 6, note, h, t, m, accent)
                        + strikes(2, 10, note, h, t, m, accent)
                        + strikes(23, 50, note, h, t, m, accent)
                    )
                    check(port, oracle, model, steps, size)


@MODELS
def test_fast_repeats_at_32nd_spacing_equal_upstream(
    port: Port, oracle: Oracle, model: int
) -> None:
    """1/32 notes at 300, 120 and 60 BPM (75, 188 and 375 sub-blocks of 16 at 48 kHz) and a strike in every sub-block,
    at the longest decay (MORPH 1) and accent 1, for 6 s each."""
    for every in (1, 75, 188, 375):
        check(port, oracle, model, strikes(every, 18000, 48.0, 0.3, 0.7, 1.0, 1.0))


@MODELS
def test_sweeps_with_strikes_equal_upstream(
    port: Port, oracle: Oracle, model: int
) -> None:
    """All four parameters moving, strikes at irregular spacing with changing accent, notes over the wrapper's range
    -64..190."""
    rng = np.random.default_rng(model)
    steps: list[Step] = []
    nxt, accent = 0, 0.8
    n = 9000
    for k in range(n):
        x = k / n
        trig = RISING_HIGH if k == nxt else HIGH
        if k == nxt:
            nxt = k + int(rng.choice([1, 2, 3, 7, 30, 120, 400]))
            accent = float(rng.choice([0.0, V100, 1.0, float(rng.random())]))
        steps.append(
            (
                -64.0 + 254.0 * (0.5 - 0.5 * math.cos(6 * math.pi * x)),
                x,
                0.5 + 0.5 * math.sin(5 * x),
                1.0 - x,
                trig,
                accent,
            )
        )
    check(port, oracle, model, steps)


@MODELS
def test_extreme_notes_are_finite_and_equal_upstream(
    port: Port, oracle: Oracle, model: int
) -> None:
    for note in (-64.0, -20.0, 0.0, 120.0, 150.0, 190.0):
        for h, t, m in ((0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 0), (0.5, 0.67, 0.5)):
            check(port, oracle, model, strikes(5, 200, note, h, t, m, 1.0))


@MODELS
def test_trigger_reads_only_the_rising_edge(port: Port, model: int) -> None:
    """The engines read `trigger & UNPATCHED` and `trigger & RISING_EDGE` only: HIGH or LOW after the edge is the same
    (so the gate length cannot matter), and RISING_EDGE without HIGH is the same as with it; a stream without an edge
    never strikes (a drum that was never struck is silent: KICK's synthetic drum has a DC offset only through its
    VCA, which stays at 0)."""
    name = DRUMS[model]
    base = strikes(40, 400, 60.0, 0.3, 0.6, 0.6, V100)
    a = port_engine(port, name, base)
    low = [(*s[:4], LOW if s[4] == HIGH else s[4], s[5]) for s in base]
    bare = [(*s[:4], 1 if s[4] == RISING_HIGH else s[4], s[5]) for s in base]
    for variant in (low, bare):
        b = port_engine(port, name, variant)
        assert same(b[0], a[0]) and same(b[1], a[1])
    none = port_engine(port, name, [(*s[:4], HIGH, s[5]) for s in base])
    struck = float(np.abs(a[0]).max())
    print(f"{name}: struck peak {struck:.3f}, never struck {np.abs(none[0]).max():.2e}")
    assert float(np.abs(none[0]).max()) < 0.1 * struck


@MODELS
def test_every_note_on_strikes(port: Port, model: int) -> None:
    """Through the wrapper: a second note-on, after the first has died away, strikes again (its block is far louder than
    the block before it), and the stream without it is identical up to that note-on."""
    kw: dict[str, int] = dict(model=model, morph=0)
    trigs = [int(b in (0, 80)) for b in range(120)]
    v = port.voice()
    x = np.concatenate([port.render(v, frame(60, **kw), t, 1) for t in trigs])
    v = port.voice()
    y = np.concatenate(
        [port.render(v, frame(60, **kw), int(b == 0), 1) for b in range(120)]
    )
    assert same(x[: 32 * 80], y[: 32 * 80])
    assert not same(x[32 * 80 : 32 * 81], y[32 * 80 : 32 * 81])
    after = float(np.abs(x[32 * 80 : 32 * 82]).max())
    before = float(np.abs(x[32 * 78 : 32 * 80]).max())
    print(
        f"{DRUMS[model]}: peak before the second note-on {before:.4f}, after {after:.4f}"
    )
    assert after > 3 * before


@MODELS
def test_accent_changes_the_strike(port: Port, oracle: Oracle, model: int) -> None:
    """Accent scales the strike (pulse height, envelope start): louder at 1 than at 0, bit exact at both."""
    lo = check(port, oracle, model, strikes(400, 400, 60.0, 0.3, 0.5, 0.5, 0.0))
    hi = check(port, oracle, model, strikes(400, 400, 60.0, 0.3, 0.5, 0.5, 1.0))
    r = float(
        np.sqrt(np.mean(hi.astype(np.float64) ** 2))
        / np.sqrt(np.mean(lo.astype(np.float64) ** 2))
    )
    print(f"{DRUMS[model]} RMS accent 1 / accent 0: {r:.2f}")
    assert r > 1.2


@MODELS
def test_repeated_strikes_stay_bounded(port: Port, model: int) -> None:
    """A strike every block for 20 s and every 1/32 at 300 BPM, at the longest decay, accent 1: the engine stays
    finite and its peak grows no further in the last 5 s than to 1.1x the peak of the 5 s before; through the wrapper
    (Clip16, then the output gain) the voice never exceeds full scale."""
    name = DRUMS[model]
    for every, note in ((2, 48.0), (75, 48.0), (2, 96.0)):
        out = port_engine(port, name, strikes(every, 60000, note, 0.5, 1.0, 1.0, 1.0))[
            0
        ]
        assert np.isfinite(out).all()
        p3 = float(np.abs(out[16 * 30000 : 16 * 45000]).max())
        p4 = float(np.abs(out[16 * 45000 :]).max())
        print(
            f"{name} every {every} sub-blocks, note {note}: raw peak 10-15 s {p3:.2f}, 15-20 s {p4:.2f}"
        )
        assert p4 <= 1.1 * p3
    v = port.voice()
    trigs = [int(b % 37 == 0) for b in range(1500 * 10)]
    x = np.concatenate(
        [
            port.render(
                v,
                frame(
                    48,
                    model=model,
                    harmonics=0x7F00,
                    timbre=0x7F00,
                    morph=0x7F00,
                    velocity=127,
                ),
                t,
                1,
            )
            for t in trigs
        ]
    )
    assert np.isfinite(x).all()
    print(
        f"{name} through the wrapper, strikes every 37 blocks for 10 s: peak {np.abs(x).max():.3f}"
    )
    assert float(np.abs(x).max()) <= 1.0


def peak_hz(out: Arr, lo_hz: float, hi_hz: float) -> float:
    x = out[4800:].astype(np.float64) * np.hanning(len(out) - 4800)
    spec = np.abs(np.fft.rfft(x, 1 << 21))
    lo, hi = (int(f * (1 << 21) / 48000) for f in (lo_hz, hi_hz))
    k = lo + int(np.argmax(spec[lo:hi]))
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * 48000 / (1 << 21)


@pytest.mark.parametrize("model,h,t,m", [(13, 0.0, 0.3, 0.9), (14, 0.0, 0.0, 0.9)])
def test_fundamental_in_tune(
    port: Port, model: int, h: float, t: float, m: float
) -> None:
    """KICK (HARMONICS 0: no FM) and SNARE (analog, 808 mode, snappy 0) ring at f0: the peak of a struck note 69
    between 400 and 480 Hz is 440 Hz within 2 cents (a0 from exactly 48 kHz)."""
    out = port_engine(port, DRUMS[model], strikes(100000, 9000, 69.0, h, t, m, V100))[0]
    f = peak_hz(out, 400, 480)
    print(f"{DRUMS[model]} note 69: {f:.3f} Hz")
    assert abs(1200 * math.log2(f / 440.0)) < 2.0


@MODELS
def test_state_bytes(port: Port, model: int) -> None:
    """The engine's per-voice state (oscillators, filters, envelopes, the engine's noise generator) fits the union."""
    size = getattr(port, f"{DRUMS[model]}_size")
    print(f"{DRUMS[model]} state bytes", size, "union", port.engine_union_size)
    assert size <= port.engine_union_size


def test_pl_sqrt_within_one_ulp_of_sqrt(port: Port) -> None:
    """The port's square root (Newton, no sqrt instruction on the core) against the true one over the SNARE's domain
    [0, 1] and wider: 0 and 1 exact, at most 1 ulp (2 allowed) elsewhere."""
    import ctypes

    port.lib.port_pl_sqrt.restype = ctypes.c_float
    port.lib.port_pl_sqrt.argtypes = [ctypes.c_float]
    assert port.lib.port_pl_sqrt(0.0) == 0.0 and port.lib.port_pl_sqrt(1.0) == 1.0
    xs = np.concatenate(
        [np.linspace(0.0, 1.0, 20001), np.exp2(np.linspace(-30, 30, 181))]
    ).astype(np.float32)
    got = np.array([port.lib.port_pl_sqrt(float(x)) for x in xs], dtype=np.float32)
    want = np.sqrt(xs)
    ulp = np.spacing(want)
    worst = float(np.max(np.abs(got - want) / np.where(ulp == 0, 1, ulp)))
    print(f"pl_sqrt worst error {worst:.2f} ulp")
    assert worst <= 2.0
