"""PLAITS model 11, STRING (modules/machines/plaits/src/dsp/engines/string_engine.c): the C99 port of Plaits' StringEngine
(three Karplus-Strong strings used in turn, struck by noise bursts, with the curved bridge and dispersion non-linearities)
against upstream (tests/plaits/oracle/run_oracle.cc, ../eurorack/plaits at 48 kHz), bit for bit on OUT and AUX, over strike
sequences: Plaits' trigger as the wrapper builds it (RISING_EDGE | HIGH in a note-on's first sub-block, HIGH after; never
UNPATCHED, so the sustain branch is never taken), repeats down to every sub-block, the 1/32-note spacing, accent at 0, 0.787
(velocity 100) and 1, knob corners (HARMONICS also on both sides of the 0.24..0.26 dead band), sweeps, and notes over the
wrapper's whole range (the lowest ones play the on-the-fly linear upsampler). stmlib's global Random is one state per engine
seeded 0x21 (the oracle seeds 0x21 when it selects the engine). The one difference is the division by an f0 of zero (the
first strike of an engine leaves the third string at f0 = 0): upstream's infinity is constrained to 1020, the port uses 1020
directly; every first strike here exercises it. Also: energy bounded under repeated strikes, table indices in range, the
fundamental in tune, every note-on strikes."""

from __future__ import annotations

import ctypes
import math
from typing import Any

import numpy as np
import pytest
from support.plaits_host import GAIN, Port, build_port, frame
from support.plaits_oracle import Oracle, Step, build, port_engine, same

Arr = Any
MODEL = 11
RISING_HIGH, HIGH, LOW = 1 | 4, 4, 0
V100 = float(np.float32(np.float32(100 << 8) * np.float32(3.0757874e-05)))
ACCENTS = (0.0, V100, 1.0)
# below 0.24: curved bridge; 0.24..0.26: none; above: dispersion
HARMS = [0.0, 0.1, 0.24, 0.25, 0.26, 0.4, 0.75, 1.0]


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


def strikes(
    every: int, n: int, note: float, h: float, t: float, m: float, accent: float
) -> list[Step]:
    """n sub-blocks, a note-on every `every` sub-blocks (RISING_EDGE | HIGH there, HIGH between)."""
    return [
        (note, h, t, m, RISING_HIGH if k % every == 0 else HIGH, accent)
        for k in range(n)
    ]


def check(port: Port, oracle: Oracle, steps: list[Step], size: int = 16) -> Arr:
    got = port_engine(port, "string_engine", steps, size)
    want = oracle.engine(MODEL, steps, size)
    assert same(got[0], want[0]) and same(got[1], want[1])
    assert np.isfinite(got[0]).all() and np.isfinite(got[1]).all()
    return got[0]


@pytest.mark.parametrize("accent", ACCENTS)
@pytest.mark.parametrize("size", [16, 12])
def test_strikes_equal_upstream_at_knob_corners(
    port: Port, oracle: Oracle, accent: float, size: int
) -> None:
    """Every HARMONICS, TIMBRE, MORPH corner at 4 pitches: strikes 1 sub-block, 2, and 23 sub-blocks apart (the three strings in
    turn, several times round)."""
    for h in HARMS:
        for t in (0.0, 0.5, 1.0):
            for m in (0.0, 0.5, 0.96, 1.0):
                for note in (24.0, 60.0, 96.0, 127.0):
                    steps = strikes(23, 100, note, h, t, m, accent) + strikes(
                        1, 6, note, h, t, m, accent
                    )
                    check(port, oracle, steps, size)


def test_fast_repeats_at_32nd_spacing_equal_upstream(
    port: Port, oracle: Oracle
) -> None:
    """1/32 notes at 300, 120 and 60 BPM (25, 62.5 and 125 ms: 75, 188 and 375 sub-blocks of 16 at 48 kHz) and a strike in every
    sub-block, in the three non-linearity regimes at the longest decay (MORPH 1) and accent 1, for 3 s each."""
    for every in (1, 2, 75, 188, 375):
        for h in (0.1, 0.25, 0.7):
            check(port, oracle, strikes(every, 9000, 48.0, h, 0.7, 1.0, 1.0))


def irregular(seed: int, n: int = 6000) -> list[Step]:
    """Knobs, pitch and accent jumping between sub-blocks (also several in a row), with note-ons at random spacing."""
    rng = np.random.default_rng(seed)
    steps: list[Step] = []
    cur = (60.0, 0.4, 0.6, 0.3)
    accent = V100
    for _ in range(n):
        if rng.random() < 0.1:
            cur = (
                float(
                    rng.choice(
                        [-64.0, 0.0, 24.0, 48.0, 60.0, 72.0, 100.0, 127.0, 190.0]
                    )
                ),
                float(rng.choice([0.0, 0.24, 0.25, 0.26, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.5, 1.0, rng.random()])),
                float(rng.choice([0.0, 0.94999, 0.95, 1.0, rng.random()])),
            )
        on = rng.random() < 0.08
        if on:
            accent = float(rng.choice([0.0, V100, 1.0, rng.random()]))
        steps.append((*cur, RISING_HIGH if on else HIGH, accent))
    return steps


def test_jumps_equal_upstream(port: Port, oracle: Oracle) -> None:
    for seed in range(12):
        check(port, oracle, irregular(seed))


def test_sweeps_with_strikes_equal_upstream(port: Port, oracle: Oracle) -> None:
    """All four parameters sweeping (HARMONICS through the three regimes, MORPH across the infinite-decay crossfade at 0.95),
    strikes at irregular spacing with changing accent, notes over the wrapper's range -64..190."""
    rng = np.random.default_rng(11)
    steps: list[Step] = []
    nxt, accent = 0, 0.8
    n = 12000
    for k in range(n):
        x = k / n
        trig = RISING_HIGH if k == nxt else HIGH
        if k == nxt:
            nxt = k + int(rng.choice([1, 2, 3, 7, 30, 120, 400]))
            accent = float(rng.choice([0.0, V100, 1.0, float(rng.random())]))
        steps.append(
            (
                -64.0 + 254.0 * (0.5 - 0.5 * math.cos(6 * math.pi * x)),
                0.5 + 0.5 * math.sin(11 * x),
                0.5 + 0.5 * math.cos(13 * x),
                0.5 + 0.5 * math.cos(7 * x),
                trig,
                accent,
            )
        )
    check(port, oracle, steps)


def test_extreme_notes_are_finite_and_equal_upstream(
    port: Port, oracle: Oracle
) -> None:
    for note in (-64.0, -20.0, 0.0, 10.0, 30.0, 120.0, 150.0, 190.0):
        for h, t, m in ((0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 0), (0.5, 0.67, 0.5)):
            check(port, oracle, strikes(5, 200, note, h, t, m, 1.0))


def test_trigger_reads_only_the_rising_edge(port: Port) -> None:
    """StringEngine reads `trigger & UNPATCHED` and `trigger & RISING_EDGE` only: HIGH or LOW after the edge is the same
    (so the gate length cannot matter), and RISING_EDGE without HIGH is the same as with it; a sub-block without the
    edge does not strike."""
    base = strikes(40, 400, 60.0, 0.5, 0.6, 0.6, V100)
    a = port_engine(port, "string_engine", base)[0]
    low = [(*s[:4], LOW if s[4] == HIGH else s[4], s[5]) for s in base]
    bare = [(*s[:4], 1 if s[4] == RISING_HIGH else s[4], s[5]) for s in base]
    assert same(port_engine(port, "string_engine", low)[0], a)
    assert same(port_engine(port, "string_engine", bare)[0], a)
    none = [(*s[:4], HIGH, s[5]) for s in base]
    assert not port_engine(port, "string_engine", none)[0].any()


def test_every_note_on_strikes(port: Port) -> None:
    """Through the wrapper: each note-on puts the filtered noise burst on AUX in its first block (MIX at AUX) and a string rings
    on OUT; the same stream without the second note-on differs from it only from that note-on on. Each strike is one of the
    three strings in turn, so three note-ons in a row all ring at once."""
    v = port.voice()
    trigs = [int(b in (0, 20)) for b in range(40)]
    fr = frame(60, model=MODEL, morph=0x2000)
    x = np.concatenate([port.render(v, fr, t, 1) for t in trigs])
    v = port.voice()
    y = np.concatenate([port.render(v, fr, int(b == 0), 1) for b in range(40)])
    assert same(x[: 32 * 20], y[: 32 * 20])
    assert not same(x[32 * 20 : 32 * 21], y[32 * 20 : 32 * 21])
    v = port.voice()
    fa = frame(60, model=MODEL, mix=0x7F00)
    a = np.concatenate([port.render(v, fa, t, 1) for t in trigs])
    for on in (0, 20):
        blk = np.abs(a[32 * on : 32 * on + 32])
        assert (
            blk.max() > 5 * np.abs(a[32 * on - 64 : 32 * on]).max()
            if on
            else blk.max() > 0.01
        )
    # three strikes, one per string: the second and third do not cut the first (the tail of the first is still there)
    v = port.voice()
    z = np.concatenate([port.render(v, fr, int(b in (0, 6, 12)), 1) for b in range(60)])
    assert np.abs(z[32 * 20 : 32 * 60]).max() > 1e-3


def test_accent_changes_the_strike(port: Port, oracle: Oracle) -> None:
    """Accent brightens and lengthens the string (StringVoice: brightness and damping += 0.25 accent (1 - x)): louder at 1
    than at 0, bit exact at both."""
    lo = check(port, oracle, strikes(400, 400, 60.0, 0.3, 0.5, 0.5, 0.0))
    hi = check(port, oracle, strikes(400, 400, 60.0, 0.3, 0.5, 0.5, 1.0))
    r = float(
        np.sqrt(np.mean(hi.astype(np.float64) ** 2))
        / np.sqrt(np.mean(lo.astype(np.float64) ** 2))
    )
    print(f"STRING RMS accent 1 / accent 0: {r:.2f}")
    assert r > 1.2


def test_repeated_strikes_stay_bounded(port: Port) -> None:
    """Energy under repeated strikes, at the longest decay (MORPH 1: the string never decays) and accent 1: a strike every
    block for 20 s, and every 1/32 at 300 BPM. Each string's feedback is clamped at +-20 before the DC blocker, so the raw
    output is under 3 x 21 whatever happens; the engine stays finite and its peak grows no further in the last 5 s than
    to 1.5x the peak of the 5 s before (fresh noise is injected at every strike and the string never decays at MORPH 1, so
    the peak wanders by the noise: measured 1.12 at worst); through the wrapper the OUT limiter (gain -1) keeps the voice under 1.25 x 0.8 x
    the output gain once it has caught up (after 0.5 s)."""
    for every, note, h in ((2, 48.0, 0.1), (75, 48.0, 0.5), (2, 96.0, 0.9)):
        out = port_engine(
            port, "string_engine", strikes(every, 60000, note, h, 1.0, 1.0, 1.0)
        )[0]
        assert np.isfinite(out).all()
        assert float(np.abs(out).max()) < 3 * 21.0
        p3 = float(np.abs(out[16 * 30000 : 16 * 45000]).max())
        p4 = float(np.abs(out[16 * 45000 :]).max())
        print(
            f"STRING every {every} sub-blocks, note {note}, h {h}: raw peak 10-15 s {p3:.2f}, 15-20 s {p4:.2f}"
        )
        assert p4 <= 1.5 * p3
    v = port.voice()
    trigs = [int(b % 37 == 0) for b in range(1500 * 10)]
    x = np.concatenate(
        [
            port.render(
                v,
                frame(
                    48,
                    model=MODEL,
                    harmonics=0x2600,
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
    late = float(np.abs(x[24000:]).max())
    print(
        f"STRING through the wrapper, strikes every 37 blocks for 10 s: peak after 0.5 s {late:.3f}"
    )
    assert late < 1.25 * 0.8 * float(GAIN)


def test_indices_in_range(checked: Port) -> None:
    """lut_svf_shift (257 entries) is read at the damping cutoff and the next entry: the cutoff is at most 128 (the infinite
    decay crossfade), at least 12; the stretch all-pass's delay stays inside its line of 256."""
    checked.lib.port_index_reset()
    for h in HARMS:
        for m in (0.0, 0.5, 0.95, 1.0):
            for t in (0.0, 1.0):
                for note in (24.0, 127.0):
                    port_engine(
                        checked, "string_engine", strikes(50, 200, note, h, t, m, 1.0)
                    )
    for seed in range(4):
        port_engine(checked, "string_engine", irregular(seed, 2000))
    for tbl in (39, 40):
        print(
            "STRING index",
            tbl,
            checked.lib.port_index_lo(tbl),
            checked.lib.port_index_hi(tbl),
            checked.lib.port_index_n(tbl),
        )
        assert checked.lib.port_index_n(tbl) > 0  # the recorder is hit
    assert checked.lib.port_index_lo(39) >= 12 and checked.lib.port_index_hi(39) <= 128
    assert checked.lib.port_index_lo(40) >= 4 and checked.lib.port_index_hi(40) < 256


def test_fundamental_in_tune(port: Port) -> None:
    """With HARMONICS in the dead band (structure 0.25: no non-linearity), TIMBRE and MORPH mid, the ringing string at note 69
    (the spectrum's peak between 400 and 480 Hz) is 440 Hz within 5 cents (a0 from exactly 48 kHz; measured 3.5 cents sharp, which is upstream's own: the
    string's pitch is the delay compensated by upstream's lut_svf_shift, an approximation, and the oracle test above is bit exact)."""
    out = port_engine(
        port, "string_engine", strikes(100000, 9000, 69.0, 0.25, 0.3, 0.5, V100)
    )[0]
    x = out[4800:].astype(np.float64) * np.hanning(len(out) - 4800)
    spec = np.abs(np.fft.rfft(x, 1 << 21))
    lo, hi = (int(f * (1 << 21) / 48000) for f in (400, 480))
    k = lo + int(np.argmax(spec[lo:hi]))
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    f = (k + 0.5 * (a - c) / (a - 2 * b + c)) * 48000 / (1 << 21)
    print(f"STRING note 69: {f:.3f} Hz")
    assert abs(1200 * math.log2(f / 440.0)) < 5.0


def test_state_bytes(port: Port) -> None:
    """Three strings of 1024 + 256 floats, their filters and counters, the f0 history: it fits in the union (DUST's)."""
    print(
        "string state bytes", port.string_engine_size, "union", port.engine_union_size
    )
    assert 3 * (1024 + 256) * 4 <= port.string_engine_size <= port.engine_union_size
    assert port.string_engine_size < port.engine_union_size


def test_tracks_seed_different_noise(port: Port) -> None:
    """string_engine_seed(track): track 0 keeps upstream's 0x21 (bit-exact with the oracle above); another track renders other
    noise and the same track renders the same again."""

    def render(track: int | None) -> Arr:
        from support.plaits_host import FP, Params

        st = ctypes.create_string_buffer(port.string_engine_size)
        port.lib.string_engine_init(st)
        if track is not None:
            port.lib.string_engine_seed.argtypes = [ctypes.c_void_p, ctypes.c_int]
            port.lib.string_engine_seed(st, track)
        out = np.zeros(16 * 40, dtype=np.float32)
        aux = np.zeros_like(out)
        for k in range(40):
            p = Params(60.0, 0.5, 0.8, 0.5, RISING_HIGH if k == 0 else HIGH, V100)
            port.lib.string_engine_render(
                st,
                ctypes.byref(p),
                out[16 * k :].ctypes.data_as(FP),
                aux[16 * k :].ctypes.data_as(FP),
                16,
            )
        return aux

    base = render(None)
    assert same(render(0), base)
    assert not same(render(1), base) and not same(render(1), render(2))
    assert same(render(7), render(7))
