"""core/params' knob-bar scaling (src/cf/dt2/barscale.s) on the native ColdFire core: the value of an added code is mapped from
its row's range onto 0 .. 0x7f00 before the drawer sees it; a row of 0 .. 0x7f00, an empty range and every stock code are
untouched. Skips without the m68k toolchain or the stock image."""

from __future__ import annotations

import pytest
from support.paths import REPO_ROOT

from tools.firmware import config
from tools.patchkit import pk_cfunit, pk_toolchains

SYX = config.stock("Digitakt_II_OS1.17.syx")
SRC = REPO_ROOT / "modules" / "core" / "params" / "src" / "cf" / "dt2" / "barscale.s"
FIRST = 0x113
ROWS = 0x47EC0000
ROWSIZE = 36

pytestmark = pytest.mark.skipif(
    pk_toolchains.m68k_bin("gcc") is None or not SYX.exists(),
    reason="m68k toolchain or firmware/Digitakt_II_OS1.17.syx missing",
)


def row(mn: int, mx: int) -> bytes:
    # name, short, template, slot, min, max, default, formatter, randomize
    words = [0, 0, 0xFE, 1, mn, mx, 0, 0, 0]
    return b"".join(w.to_bytes(4, "big") for w in words)


@pytest.fixture(scope="module")
def unit():
    return pk_cfunit.CfUnit(
        SYX,
        [SRC],
        ["_pf_t_draw"],
        {"_pf_first": FIRST, "_pf_count": 4, "_pf_rows": ROWS, "_pf_rowsize": ROWSIZE},
        ["pf_drawone"],
    )


def test_values_map_onto_the_level_range(unit):
    rows = row(0, 0x1800) + row(0, 0x7F00) + row(0x100, 0x100) + row(0x200, 0x1200)
    cases = [
        (FIRST, 0x1800),  # MODEL at its maximum -> full
        (FIRST, 0xC00),  # the middle
        (FIRST, 0),
        (FIRST, -0x100),  # below the range
        (FIRST, 0x3000),  # above the range
        (FIRST + 1, 0x4000),  # a 0 .. 0x7f00 row: unchanged
        (FIRST + 2, 0x100),  # an empty range: unchanged
        (FIRST + 3, 0x1200),  # an offset range
        (FIRST + 3, 0x200),
        (0xFE, 0x1800),  # a stock code: unchanged
        (FIRST + 4, 0x1800),  # past the rows: unchanged
    ]
    res = unit.run_cases(
        {ROWS: rows},
        [
            {"routine": "pf_drawone", "stack": [unit.rts, 0x1234, c, v]}
            for c, v in cases
        ],
    )
    got = [r.stack[3] for r in res]
    assert [r.stub for r in res] == ["_pf_t_draw"] * len(cases)
    assert got[0] == 0x7F00
    assert abs(got[1] - 0x3F80) <= 1
    assert got[2:5] == [0, 0, 0x7F00]
    assert got[5:7] == [0x4000, 0x100]
    assert got[7:9] == [0x7F00, 0]
    assert got[9:] == [0x1800, 0x1800]
    assert all(r.stack[1] == 0x1234 for r in res)  # this and the code are untouched
    assert [r.stack[2] for r in res] == [c for c, _ in cases]
