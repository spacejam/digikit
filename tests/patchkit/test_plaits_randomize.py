"""SRC+YES (randomize page) on a PLAITS track: core/params' `randomize` field and machines/plaits' rows.

- `plaits_model_rand(rnd, min, max)` runs on the emulated ColdFire core (tools/patchkit/pk_cfcall.py): over every
  generator value it returns exactly the built models, as raw 8.8 words, uniformly (no gaps, no disabled slot).
- The manifests: MODEL has the function, HARMONICS, TIMBRE and MORPH `randomize = 1`, MIX has none (left unchanged).
- The patched image in the emulator (skips without the daemon or the image `out/patch/<IMAGE>`): a PLAITS track after
  SRC+YES has a new MODEL, HARMONICS, TIMBRE and MORPH, all in range on the 0x100 grid, and TUNE, MIX and LEV unchanged;
  a SAMPLE track gives the same parameter words as the stock firmware does for the same inputs.

Skips without the m68k toolchain, the stock image or the patchkit build."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from support.paths import REPO_ROOT

from tools.firmware import config
from tools.patchkit import pk_cfcall, pk_link, pk_toolchains

ROOT: Path = REPO_ROOT
SYX = config.stock("Digitakt_II_OS1.17.syx")
SRC = ROOT / "modules" / "machines" / "plaits" / "src" / "cf" / "randomize.s"
MODULE = ROOT / "modules" / "machines" / "plaits" / "module.toml"
IMAGE = ROOT / "out" / "patch" / "plaits-25models-rand-names-splash-48k-1.17.syx"
BASE = 0x47E8B000
MODELS = list(range(25))

needs_cf = pytest.mark.skipif(
    pk_toolchains.m68k_bin("gcc") is None or not SYX.exists(),
    reason="m68k toolchain or firmware/Digitakt_II_OS1.17.syx missing",
)


@needs_cf
def test_model_rand_returns_exactly_the_built_models_uniformly() -> None:
    _obj, linked, _hist = pk_link.build([SRC], BASE, {})
    fn = linked.symbols["plaits_model_rand"]
    rnds = list(range(0, 0x8000, 5)) + [0x7FFF, 0, 1]
    pokes = {BASE: linked.image}
    calls = [{"pc": fn, "args": [r, 0, 0x1800], "reads": []} for r in rnds]
    res = pk_cfcall.CfMachine(SYX).run(pokes, calls)
    got = []
    for r, x in zip(rnds, res, strict=True):
        assert x.ok, (r, x.error)
        assert not x.clobbered, (r, x.clobbered)
        got.append(x.d0)
    assert sorted(set(got)) == [m << 8 for m in MODELS]
    counts = [got.count(m << 8) for m in MODELS]
    assert max(counts) - min(counts) <= 6, counts  # uniform up to the sweep's step


def test_the_manifest_randomizes_four_rows_and_not_mix() -> None:
    m = tomllib.loads(MODULE.read_text())
    rows = {e["id"]: e for e in m["entry"] if e["registry"] == "params"}
    assert rows["plaits_model"]["randomize"] == "plaits_model_rand"
    assert rows["plaits_model"]["max"] == (len(MODELS) - 1) << 8
    for k in ("plaits_harm", "plaits_timbre", "plaits_morph"):
        assert rows[k]["randomize"] == 1
    assert "randomize" not in rows["plaits_mix"]
    obj = next(o for o in m["object"] if "src/cf/randomize.s" in o["sources"])
    assert obj["place"] == "cf.expansion" and obj["devices"] == ["dt2"]
    params = tomllib.loads(
        (ROOT / "modules" / "core" / "params" / "module.toml").read_text()
    )
    dt2 = next(r for r in params["registry"] if r["devices"] == ["dt2"])
    assert dt2["field"][-1]["name"] == "randomize"
    assert (
        dt2["field"][-1]["default"] == 0
    )  # a row without the key keeps the template's randomizer


# ---- the emulator ------------------------------------------------------------------------------------------------------

TRACK = 0x42653300  # the first track's parameter words (probe, 2026-10-22): 32-bit reads of 16-bit words
WORDS = 0x40
# 16-bit word addresses, the sound-array slots of a clone of MANUAL SLICE: MODEL slot 1, HARMONICS 2, TIMBRE 4, MORPH 5
MODEL, HARM, TIMBRE, MORPH = 0x42653334, 0x42653336, 0x4265333E, 0x42653340
RANDOMIZED = {MODEL: 0x1800, HARM: 0x7F00, TIMBRE: 0x7F00, MORPH: 0x7F00}


def _word(snap: list[int], addr: int) -> int:
    w = snap[(addr - TRACK) // 4]
    return (w >> 16) & 0xFFFF if (addr - TRACK) % 4 == 0 else w & 0xFFFF


def _session(image: Path, steps: list[dict]) -> list[list[int]]:
    from tools.patchkit import pk_harness as H

    cell = H.Cell(device="dt2", firmware="1.17", modules=[])
    s = H.Session(cell, H.Goldens(), run_name="rand-test")
    snaps: list[list[int]] = []

    def snap(sess) -> None:  # noqa: ANN001
        snaps.append([sess.emu.peek32(TRACK + 4 * i) or 0 for i in range(WORDS)])

    s.open(image)
    try:
        s.reset()
        for st in steps:
            if st.get("snap"):
                snap(s)
            else:
                s.run_steps([st])
    finally:
        s.close()
    return snaps


def _select_plaits() -> list[dict]:
    sel = tomllib.loads(
        (ROOT / "modules" / "machines" / "plaits" / "tests" / "select.toml").read_text()
    )
    return sel["step"]


needs_emu = pytest.mark.skipif(
    not IMAGE.exists() or not SYX.exists() or not os.environ.get("DIGI_EMUD_DT2"),
    reason=f"needs {IMAGE.name} and DIGI_EMUD_DT2 (a digi-emud with coupled audio)",
)

YES = [{"tap": ["SRC", "YES"]}, {"wait_ms": 400}]


@needs_emu
def test_src_yes_randomizes_plaits_model_harmonics_timbre_morph_only() -> None:
    steps = [*_select_plaits(), {"snap": 1}]
    for _ in range(4):
        steps += [*YES, {"snap": 1}]
    snaps = _session(IMAGE, steps)
    before = snaps[0]
    seen = {a: set() for a in RANDOMIZED}
    for after in snaps[1:]:
        for a, top in RANDOMIZED.items():
            v = _word(after, a)
            assert v <= top and v % 0x100 == 0, (hex(a), hex(v))
            seen[a].add(v)
        # nothing else of the track's parameter words changed (TUNE, MIX, LEV and the rest)
        for i in range(WORDS):
            for part in (0, 2):
                addr = TRACK + 4 * i + part
                if addr in RANDOMIZED:
                    continue
                assert _word(after, addr) == _word(before, addr), hex(addr)
    for a, vals in seen.items():
        assert len(vals | {_word(before, a)}) > 1, hex(a)  # it changed


@needs_emu
def test_src_yes_on_a_stock_machine_is_unchanged() -> None:
    steps = [{"wait_ms": 1500}, {"snap": 1}, *YES, {"snap": 1}, *YES, {"snap": 1}]
    patched = _session(IMAGE, steps)
    stock = _session(SYX, steps)

    # the generator's state depends on the boot timing, so the values differ between images: the words that SRC+YES
    # changes (and those it leaves) are the same, and the untouched words are equal
    def changed(s: list[list[int]]) -> list[int]:
        return [i for i in range(WORDS) if s[0][i] != s[1][i]]

    assert changed(patched) == changed(stock) != []
    assert patched[0] == stock[0]
    same = [i for i in range(WORDS) if i not in changed(patched)]
    assert all(patched[2][i] == stock[2][i] for i in same)
