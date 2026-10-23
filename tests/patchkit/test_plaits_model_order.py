"""The MODEL knob's order (owner, 2026-10-22): 0..20 as the engines, then 21 DX7-4, 22 TERRAIN, 23 STRING MACHINE,
24 SQUARE. The engine numbers are unchanged (digests, tests and vectors stay keyed by engine); the DSP wrapper maps
the knob position to the engine in `plaits_knob_engine` (plaits.c, used by plaits_track), the formatter names the
knob positions (src/cf/model_fmt.s) and the randomizer picks knob positions (src/cf/randomize.s: 0..24, no gaps).

- names in knob order, and the three agree on the order;
- each knob position, through the DSP entry `plaits_track`, renders the intended engine: the same samples as the
  engine's own wrapper render (`plaits_render` with the engine number), for two notes and a note-off; and 25 distinct
  outputs.

Skips without a C compiler for the host port."""

from __future__ import annotations

import ctypes
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from support.paths import REPO_ROOT
from support.plaits_host import FP, Port, build_port, frame

PLAITS = REPO_ROOT / "modules" / "machines" / "plaits"
ENGINE_OF_KNOB = [*range(21), 24, 21, 22, 23]
KNOB_NAMES = [
    "VA", "ASYM-TRI", "2FM", "FORMANT", "ADDITIVE", "WAVETABLE", "CHORD", "SPEECH", "SWARM", "NOISE", "DUST",
    "STRING", "RINGS", "KICK", "SNARE", "HIHAT", "VCF-VA", "PHASE-DIST", "DX7-1", "DX7-2", "DX7-3", "DX7-4",
    "TERRAIN", "STRING MACH", "SQUARE",
]  # fmt: skip
ENGINE_NAMES = {e: KNOB_NAMES[k] for k, e in enumerate(ENGINE_OF_KNOB)}


def test_the_formatter_names_the_knob_positions_in_order() -> None:
    text = (PLAITS / "src" / "cf" / "model_fmt.s").read_text()
    strings = dict(re.findall(r'^(pm\d+):\s*\.asciz\s+"([^"]*)"', text, re.M))
    assert [strings[f"pm{i}"] for i in range(25)] == KNOB_NAMES
    table = re.search(r"pm_names:(.*?)pm_err", text, re.S)
    assert table is not None
    assert re.findall(r"pm(\d+)", table.group(1)) == [str(i) for i in range(25)]


def test_the_randomizer_lists_every_knob_position_once() -> None:
    text = (PLAITS / "src" / "cf" / "randomize.s").read_text()
    body = text.split("pr_models:")[1].split("pr_end:")[0]
    nums = [int(x) for x in re.findall(r"\b(\d+)\b", re.sub(r"\|.*", "", body))]
    assert nums == list(range(25))


def test_the_knob_order_is_the_engine_order_except_the_last_four() -> None:
    assert ENGINE_OF_KNOB[:21] == list(range(21))
    assert sorted(ENGINE_OF_KNOB) == list(range(25))
    assert [ENGINE_NAMES[e] for e in (21, 22, 23, 24)] == [
        "TERRAIN",
        "STRING MACH",
        "SQUARE",
        "DX7-4",
    ]


@pytest.fixture(scope="module")
def so(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("cc") is None and shutil.which("gcc") is None:
        pytest.skip("no C compiler")
    d = tmp_path_factory.mktemp("order")
    return d / "libplaits_port.so" if build_port(d) else d / "libplaits_port.so"


def fresh(so: Path, n: int) -> Port:
    """A private copy of the library: plaits_track keeps its voices in statics."""
    copy = so.with_name(f"lib{n}.so")
    shutil.copy(so, copy)
    p = Port(copy)
    p.lib.plaits_track.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        FP,
        ctypes.c_int,
        ctypes.c_int,
    ]
    p.lib.plaits_track.restype = ctypes.c_int
    p.lib.plaits_knob_engine.argtypes = [ctypes.c_int]
    p.lib.plaits_knob_engine.restype = ctypes.c_int
    return p


def seq(p: Port, knob: int | None, engine: int | None) -> np.ndarray:
    """Six blocks: a note-on, held, a different note-on, a note-off block, via plaits_track (knob) or plaits_render."""
    v = p.voice() if engine is not None else None
    out = []
    plan = [(60, 1), (60, 0), (60, 0), (64, 1), (64, 0), (64, 0)]
    for note, trig in plan:
        buf = np.zeros(32, dtype=np.float32)
        if engine is not None:
            f = frame(note, model=engine, harmonics=0x3000, timbre=0x5000, morph=0x2000)
            p.lib.plaits_render(
                v, ctypes.byref(f), trig, buf.ctypes.data_as(FP), 32, 48000
            )
        else:
            assert knob is not None
            f = frame(note, model=knob, harmonics=0x3000, timbre=0x5000, morph=0x2000)
            p.lib.plaits_track(
                3, ctypes.byref(f), trig, buf.ctypes.data_as(FP), 32, 48000
            )
        out.append(buf)
    return np.concatenate(out)


def test_each_knob_position_renders_the_intended_engine(so: Path) -> None:
    outs = []
    for knob, engine in enumerate(ENGINE_OF_KNOB):
        p = fresh(so, 100 + knob)
        assert p.lib.plaits_knob_engine(knob) == engine
        got = seq(p, knob, None)
        # the reference is the engine's own wrapper; for engines 0..20 the knob is the engine number, so plaits_track
        # itself (some engines seed from the track number, which plaits_render has no access to)
        q = fresh(so, 200 + knob)
        ref = seq(q, engine, None) if engine <= 20 else seq(q, None, engine)
        assert np.array_equal(got, ref), (knob, KNOB_NAMES[knob])
        assert np.all(np.isfinite(got))
        outs.append(got)
    # the 25 knob positions are 25 different sounds (none is silent, none repeats another)
    assert all(np.abs(o).max() > 1e-4 for o in outs)
    for i in range(25):
        for j in range(i):
            assert not np.array_equal(outs[i], outs[j]), (i, j)
    # out of range: no remap
    p = fresh(so, 999)
    assert [p.lib.plaits_knob_engine(k) for k in (-1, 25, 26, 100)] == [-1, 25, 26, 100]
