"""PLAITS' SRC page in the emulator (owner requests, 2026-10-22): one SRC page, and the knob bar of MODEL spans its range.

- One SRC page (core/machines `one_page`, src/cf/dt2/onepage.s): on a PLAITS track the OLED frame after the first, second
  and third SRC press is the same; on a stock machine (SAMPLE) the frames equal the stock firmware's, and a second press
  still opens its second page.
- The MODEL bar (core/params, src/cf/dt2/barscale.s): with MODEL, HARMONICS, TIMBRE and MORPH all at their maxima, the
  fader of MODEL is as full as the others (the lit pixels of its column equal HARMONICS'); at MODEL = 0 it is emptier.

Skip without the image `out/patch/<IMAGE>` and DIGI_EMUD_DT2 (a digi-emud with coupled audio)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from support.paths import REPO_ROOT

from tools.firmware import config

SYX = config.stock("Digitakt_II_OS1.17.syx")
IMAGE = REPO_ROOT / "out" / "patch" / "plaits-25models-rand-names-splash-48k-1.17.syx"
SELECT = REPO_ROOT / "modules" / "machines" / "plaits" / "tests" / "select.toml"

pytestmark = pytest.mark.skipif(
    not IMAGE.exists() or not SYX.exists() or not os.environ.get("DIGI_EMUD_DT2"),
    reason=f"needs {IMAGE.name} and DIGI_EMUD_DT2 (a digi-emud with coupled audio)",
)


def _run(image: Path, steps: list[dict], shots: int) -> list[bytes]:
    """Run `steps` (a step {"shot": 1} takes the OLED frame) from the ready state; returns the frames."""
    from tools.patchkit import pk_harness as H

    s = H.Session(
        H.Cell(device="dt2", firmware="1.17", modules=[]),
        H.Goldens(),
        run_name="ui-test",
    )
    frames: list[bytes] = []
    s.open(image)
    try:
        s.reset()
        for st in steps:
            if st.get("shot"):
                frames.append(s.emu.frame()[1])
            else:
                s.run_steps([st])
    finally:
        s.close()
    assert len(frames) == shots
    return frames


def _select() -> list[dict]:
    return tomllib.loads(SELECT.read_text())["step"]


SRC = [{"tap": "SRC"}, {"wait_ms": 700}, {"shot": 1}]


def test_plaits_has_one_src_page() -> None:
    f = _run(IMAGE, [*_select(), *SRC, *SRC, *SRC], 3)
    assert f[0] == f[1] == f[2]


def test_a_stock_machine_keeps_its_pages() -> None:
    steps = [{"wait_ms": 1500}, *SRC, *SRC, *SRC]
    patched = _run(IMAGE, steps, 3)
    stock = _run(SYX, steps, 3)
    assert patched == stock
    assert patched[0] != patched[1]  # SAMPLE's second press shows its second page


def _lit(frame: bytes, x0: int, x1: int, y0: int, y1: int) -> int:
    return sum(
        frame[(7 - y // 8) + 8 * x] >> (y % 8) & 1
        for x in range(x0, x1)
        for y in range(y0, y1)
    )


# the full-knob view: the top row's faders of MODEL (x 55 .. 63) and HARMONICS (81 .. 89)
def test_the_model_bar_spans_its_range() -> None:
    hi = [{"turn": f"ENC {k}", "by": 4, "events": 200} for k in "BCEF"]
    lo = [{"turn": "ENC B", "by": -4, "events": 200}]
    f = _run(
        IMAGE,
        [*_select(), {"tap": "SRC"}, {"wait_ms": 500}, *hi, {"wait_ms": 1500}, {"shot": 1},
         *lo, {"wait_ms": 1500}, {"shot": 1}],
        2,
    )  # fmt: skip
    top, low = f
    model_hi, harm_hi = _lit(top, 55, 64, 14, 32), _lit(top, 81, 90, 14, 32)
    assert model_hi == harm_hi, (model_hi, harm_hi)
    assert _lit(low, 55, 64, 14, 32) < model_hi
