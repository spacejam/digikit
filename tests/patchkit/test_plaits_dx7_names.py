"""DX7 patch names on PLAITS' HARMONICS (modules/machines/plaits/src/cf/dx7_text.s, dx7_names.s).

What is proved, and where:
  - the manifest: the object, its externs and the one hook on the knob-turn header's `jsr param_value_text`, whose anchor
    resolves on 1.17 to the `jsr` itself (6 bytes);
  - `plaits_dx7_name` on the emulated ColdFire core (tools/patchkit/pk_cfcall.py), exhaustively: every HARMONICS word 0 ..
    0x7f00 of all four banks writes the name of the patch the DSP plays (the port's `six_op_patch_index` on the same word as
    the wrapper gives it to the engine, tests/support/plaits_host.py) and nothing else;
  - the header hook `dx7_header_text` on the same core, with a fake view whose value getter returns MODEL: HARMONICS with a DX7
    model (knob 18 .. 21) gets the bank's name in param_value_text's buffer and returns through the site; every other model and
    every other parameter runs the displaced `jsr` (the number, as before); the registers the code around the site keeps
    (d2 - d7, a2 - a6) are unchanged;
  - the emulator (needs the image and DIGI_EMUD_DT2): tests/patchkit/test_plaits_dx7_names_emu below.

The table is one generated list (lib/gen_dx_tables.py) for the DSP and for the names: tests/plaits/test_six_op_host.py."""

from __future__ import annotations

import ctypes
import os
import struct
import tomllib
from pathlib import Path

import pytest
from support.paths import REPO_ROOT
from support.plaits_host import build_port

from tools.firmware import config
from tools.patchkit import pk_cfcall, pk_link, pk_toolchains

ROOT: Path = REPO_ROOT
SYX = config.stock("Digitakt_II_OS1.17.syx")
PLAITS = ROOT / "modules" / "machines" / "plaits"
TEXT = PLAITS / "src" / "cf" / "dx7_text.s"
NAMES = PLAITS / "src" / "cf" / "dx7_names.s"
MODULE = PLAITS / "module.toml"
BASE = 0x47E8B000
BUF = 0x4293B818  # param_value_text's static buffer (the hook writes the name there)
HARM, MODEL = (
    0x1C3,
    0x1C1,
)  # the codes the fake rows use (the real ones are extern values)
MODEL_CELL = 0x47EC0000
CELLS = 0x47EC0300
VIEW, VTABLE = 0x47EC0100, 0x47EC0200

needs_cf = pytest.mark.skipif(
    pk_toolchains.m68k_bin("gcc") is None or not SYX.exists(),
    reason="m68k toolchain or firmware/Digitakt_II_OS1.17.syx missing",
)

TRAMP = f"""
    .text
    .globl tramp
tramp:                          | tramp(view, code, value): the registers and stack of the header at the hook site
    lea     -8(%sp),%sp
    movem.l %d2/%a2,(%sp)
    movea.l 12(%sp),%a2
    move.l  16(%sp),%d2
    move.l  20(%sp),%d0
    move.l  %d0,-(%sp)
    move.l  %d2,-(%sp)
    jmp     dx7_header_text
    .globl t_ret
t_ret:                          | the address after the site: what param_value_text's return would have left
    addq.l  #8,%sp
    movem.l (%sp),%d2/%a2
    lea     8(%sp),%sp
    rts
    .globl t_tail
t_tail:                         | the displaced jsr: marks d0 and goes on
    move.l  #0xdeadbeef,%d0
    addq.l  #8,%sp
    movem.l (%sp),%d2/%a2
    lea     8(%sp),%sp
    rts
    .globl t_getter
t_getter:                       | the view's value getter (vtable + 0x54): returns the MODEL word of the cell
    move.l  {MODEL_CELL:#x},%d0
    rts
    .globl tramp_l
tramp_l:                        | tramp_l(view, code, value, buf, d0): the label hook's registers and stack (this = 0x5555)
    lea     -32(%sp),%sp
    movem.l %d2-%d3/%a2/%a5,(%sp)
    movea.l 36(%sp),%a5
    move.l  40(%sp),%d2
    move.l  44(%sp),%d1
    movea.l 48(%sp),%a0
    move.l  52(%sp),%d0
    lea     -0x60(%sp),%sp
    move.l  %a0,-(%sp)
    move.l  %d1,-(%sp)
    move.l  %d2,-(%sp)
    move.l  #0x5555,-(%sp)
    movea.l #0xa1a1a1a1,%a1
    jmp     dx7_label_text
    .globl t_lret
t_lret:                         | the `jsr (a1)` of the site: records a1 and the displaced store, marks d0
    move.l  %a1,{CELLS:#x}
    move.l  0x44(%sp),%d1
    move.l  %d1,{CELLS + 4:#x}
    move.l  #0xfeedf00d,%d0
    bra.s   t_lend
    .globl t_ltail
t_ltail:                        | the displaced instructions: records a1 and d0, marks d0
    move.l  %a1,{CELLS:#x}
    move.l  %d0,{CELLS + 4:#x}
    move.l  #0xdeadbeef,%d0
t_lend:
    lea     16(%sp),%sp
    lea     0x60(%sp),%sp
    movem.l (%sp),%d2-%d3/%a2/%a5
    lea     32(%sp),%sp
    rts
    .globl t_eff
t_eff:                          | the firmware's cell-value function (view, code, &locked): the MODEL cell; records view and code
    move.l  4(%sp),%d1
    move.l  %d1,{CELLS + 8:#x}
    move.l  8(%sp),%d1
    move.l  %d1,{CELLS + 12:#x}
    move.l  {MODEL_CELL:#x},%d0
    rts
    .globl t_set
t_set:                          | t_set(word)
    move.l  4(%sp),%d0
    move.l  %d0,{MODEL_CELL:#x}
    rts
"""


def build_cf(tmp: Path) -> tuple[pk_link.Linked, bytes]:
    tramp = tmp / "tramp.s"
    tramp.write_text(TRAMP)
    srcs = [TEXT, NAMES, tramp]
    zero = {
        "_dn_harm": HARM, "_dn_model": MODEL, "_dn_tail": 0, "_dn_ret": 0,
        "_dn_effval": 0, "_dn_ltail": 0, "_dn_lret": 0,
    }  # fmt: skip
    _o, linked, _h = pk_link.build(srcs, BASE, zero)
    sym = linked.symbols
    real = {
        **zero,
        "_dn_tail": sym["t_tail"],
        "_dn_ret": sym["t_ret"],
        "_dn_effval": sym["t_eff"],
        "_dn_ltail": sym["t_ltail"],
        "_dn_lret": sym["t_lret"],
    }
    _o, linked, _h = pk_link.build(srcs, BASE, real)
    return linked, linked.image


@pytest.fixture(scope="module")
def cf(tmp_path_factory: pytest.TempPathFactory) -> tuple[pk_link.Linked, bytes]:
    return build_cf(tmp_path_factory.mktemp("dx7-names"))


def gen():  # the generator as a module
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_dx_tables", PLAITS / "src" / "dsp" / "lib" / "gen_dx_tables.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bank_bytes() -> list[bytes]:
    """The four banks as the DSP holds them (lib/dx7_banks.c)."""
    import re

    text = (PLAITS / "src" / "dsp" / "lib" / "dx7_banks.c").read_text()
    head = text.index("pl_dx7_banks[4096]")
    words = [
        int(x, 16)
        for x in re.findall(r"0x([0-9a-f]{8})u", text[head : text.index("};", head)])
    ]
    assert len(words) == 4096
    raw = b"".join(struct.pack("<I", w) for w in words)
    return [raw[b * 4096 : (b + 1) * 4096] for b in range(4)]


# ---- the manifest -----------------------------------------------------------------------------------------------------


def test_the_manifest_has_the_object_and_one_hook_on_the_header() -> None:
    m = tomllib.loads(MODULE.read_text())
    obj = next(o for o in m["object"] if "src/cf/dx7_text.s" in o["sources"])
    assert obj["sources"] == ["src/cf/dx7_text.s", "src/cf/dx7_names.s"]
    assert obj["place"] == "cf.expansion" and obj["devices"] == ["dt2"]
    ex = obj["externs"]
    assert ex["_dn_harm"] == "params.plaits_harm.index"
    assert ex["_dn_model"] == "params.plaits_model.index"
    assert ex["_dn_tail"] == "hook.plaits.dx7.headertext.tail"
    assert ex["_dn_ret"] == "hook.plaits.dx7.headertext.return"
    assert ex["_dn_effval"] == "plaits.dx7.effval"
    assert ex["_dn_ltail"] == "hook.plaits.dx7.label.tail"
    assert ex["_dn_lret"] == "hook.plaits.dx7.label.return"
    hooks = {o["anchor"]: o for o in m["op"] if o["op"] == "hook"}
    assert set(hooks) == {"plaits.dx7.headertext", "plaits.dx7.label"}
    for a, to in (
        ("plaits.dx7.headertext", "dx7_header_text"),
        ("plaits.dx7.label", "dx7_label_text"),
    ):
        h = hooks[a]
        assert h["to"] == to and h["displaced"] == "run" and h["offset"] == 0
        assert h["devices"] == ["dt2"]
    names = {r["name"] for r in m["anchor_rule"]}
    assert {"plaits.dx7.label", "plaits.dx7.effval"} <= names
    rule = next(r for r in m["anchor_rule"] if r["name"] == "plaits.dx7.headertext")
    assert rule["kind"] == "cf_func" and rule["len"] == 6


def test_the_anchor_is_the_jsr_to_param_value_text_in_the_lockfile() -> None:
    lock = tomllib.loads((ROOT / "devices" / "ports" / "dt2-1.17.toml").read_text())
    a = next(x for x in lock["anchor"] if x["name"] == "plaits.dx7.headertext")
    assert (a["addr"], a["len"]) == (0x4006CE92, 6)
    if SYX.exists():
        from tools.firmware import (
            extract,  # noqa: F401  (the bytes are checked below through the section)
        )

        sec = ROOT / "out" / "sections" / "dt2-1.17" / "section_3_MAIN_OS.bin"
        if sec.exists():
            code = sec.read_bytes()[0x4006CE92 - 0x40000400 :][:6]
            assert code == bytes.fromhex("4eb9400c8952")  # jsr param_value_text
            lab = {x["name"]: x for x in lock["anchor"]}
            assert lab["plaits.dx7.label"]["addr"] == 0x40069CC8
            assert lab["plaits.dx7.effval"]["addr"] == 0x4006AA32
            raw = sec.read_bytes()
            assert raw[0x40069CC8 - 0x40000400 :][:8] == bytes.fromhex(
                "2269005c2f400044"
            )


# ---- the code on the emulated core -------------------------------------------------------------------------------------


@needs_cf
def test_every_harmonics_word_of_every_bank_names_the_patch_the_dsp_plays(
    cf: tuple[pk_link.Linked, bytes], tmp_path: Path
) -> None:
    linked, image = cf
    g = gen()
    port = build_port(tmp_path)
    port.lib.port_dx7_patch.argtypes = [ctypes.c_float]
    rows = g.name_rows(bank_bytes())
    fn = linked.symbols["plaits_dx7_name"]
    words = list(range(0x7F01))
    calls = [
        {"pc": fn, "args": [w, b, BUF + 0x100], "reads": [(BUF + 0x100, 16)]}
        for b in range(4)
        for w in words
    ]
    res = pk_cfcall.CfMachine(SYX).run({BASE: image, BUF + 0x100: b"\xaa" * 16}, calls)
    k = 0
    seen = [set() for _ in range(4)]
    for b in range(4):
        for w in words:
            x = res[k]
            k += 1
            assert x.ok and not x.clobbered, (b, w, x.error, x.clobbered)
            patch = port.lib.port_dx7_patch(float(g.unit_of_word(w)))
            want = rows[b][patch]
            got = x.reads[0].split(b"\0")[0].decode()
            assert got == want, (b, w, patch, got, want)
            assert x.reads[0][: len(want) + 1] == want.encode() + b"\0"
            assert (
                x.reads[0][11:] == b"\xaa" * 5
            )  # nothing written past the longest name and its NUL
            seen[b].add(got)
    for b in range(4):
        assert seen[b] == set(rows[b])  # every patch of every bank is reachable


@needs_cf
def test_the_header_names_harmonics_for_dx7_models_and_runs_the_stock_call_otherwise(
    cf: tuple[pk_link.Linked, bytes],
) -> None:
    linked, image = cf
    sym = linked.symbols
    g = gen()
    thr = g.patch_words()
    rows = g.name_rows(bank_bytes())
    pokes = {
        BASE: image,
        VIEW: struct.pack(">I", VTABLE),
        VTABLE + 0x54: struct.pack(">I", sym["t_getter"]),
        BUF: b"stock text\0\0",
    }
    calls: list[dict] = []
    cases: list[tuple[int, int, int]] = []  # (model knob, code, value)
    for model in range(25):
        for code, value in (
            (HARM, 0x4000),
            (HARM, thr[5]),
            (HARM, 0x7F00),
            (HARM, 0),
            (MODEL, 0x4000),
            (0x10, 0x4000),
        ):
            cases.append((model, code, value))
    for model, code, value in cases:
        calls.append({"pc": sym["t_set"], "args": [model << 8]})
        calls.append(
            {"pc": sym["tramp"], "args": [VIEW, code, value], "reads": [(BUF, 12)]}
        )
    res = pk_cfcall.CfMachine(SYX).run(pokes, calls)
    for i, (model, code, value) in enumerate(cases):
        x = res[2 * i + 1]
        assert x.ok and not x.clobbered, (model, code, value, x.error, x.clobbered)
        if code == HARM and 18 <= model <= 21:
            patch = sum(1 for k in range(1, 32) if value >= thr[k])
            want = rows[model - 18][patch]
            assert x.d0 == BUF, (model, value)
            assert x.reads[0].split(b"\0")[0].decode() == want, (model, value)
        else:
            assert x.d0 == 0xDEADBEEF, (model, code, value)  # the displaced jsr ran


@needs_cf
def test_the_label_shows_the_short_name_for_the_effective_dx7_model_and_the_number_otherwise(
    cf: tuple[pk_link.Linked, bytes],
) -> None:
    linked, image = cf
    sym = linked.symbols
    g = gen()
    thr = g.patch_words()
    short = g.short_rows(g.name_rows(bank_bytes()))
    assert all(len(n) <= 6 for r in short for n in r)
    pokes = {
        BASE: image,
        VIEW: struct.pack(">I", VTABLE),
        BUF: b"stock text\0\0",
    }
    calls: list[dict] = []
    cases: list[tuple[int, int, int]] = []
    for model in range(25):
        for code, value in (
            (HARM, 0),
            (HARM, thr[7]),
            (HARM, 0x4000),
            (HARM, 0x7F00),
            (MODEL, 0x4000),
            (0x10, 0x4000),
        ):
            cases.append((model, code, value))
    for model, code, value in cases:
        calls.append({"pc": sym["t_set"], "args": [model << 8]})
        calls.append(
            {
                "pc": sym["tramp_l"],
                "args": [VIEW, code, value, BUF, 0x12345678],
                "reads": [(BUF, 12), (CELLS, 16)],
            }
        )
    res = pk_cfcall.CfMachine(SYX).run(pokes, calls)
    for i, (model, code, value) in enumerate(cases):
        x = res[2 * i + 1]
        assert x.ok and not x.clobbered, (model, code, value, x.error, x.clobbered)
        cells = struct.unpack(">4I", x.reads[1])
        if code == HARM and 18 <= model <= 21:
            patch = sum(1 for k in range(1, 32) if value >= thr[k])
            assert x.d0 == 0xFEEDF00D, (model, value)
            assert x.reads[0].split(b"\0")[0].decode() == short[model - 18][patch], (
                model,
                value,
            )
            assert cells[0] == sym["dx7_nop"], "the jsr (a1) calls the do-nothing stub"
            assert cells[1] == 0x12345678, "the displaced store of d0 was made"
            assert cells[2:] == (VIEW, MODEL), (
                "MODEL is read through the page's own cell-value function"
            )
        else:
            assert x.d0 == 0xDEADBEEF, (model, code, value)
            assert (
                cells[0] == 0xA1A1A1A1 and cells[1] == 0x12345678
            )  # a1 and d0 restored for the displaced code


# ---- the emulator: PLAITS on track 1, the SRC page, MODEL and HARMONICS turned ----------------------------------------------

IMAGE = ROOT / "out" / "patch" / "plaits-25models-dx7names-splash-48k-1.17.syx"
TRACK = 0x42653300  # the first track's parameter words (finding 02): 32-bit reads of 16-bit words
MODEL_W, HARM_W = 0x42653334, 0x42653336
SELECT = PLAITS / "tests" / "select.toml"
MODEL_COLS = (44, 73)  # the columns of MODEL's own label under its knob

needs_emu = pytest.mark.skipif(
    not IMAGE.exists() or not SYX.exists() or not os.environ.get("DIGI_EMUD_DT2"),
    reason=f"needs {IMAGE.name} and DIGI_EMUD_DT2 (a digi-emud with coupled audio)",
)


def _word(emu, addr: int) -> int:  # noqa: ANN001
    w = emu.peek32(addr & ~3) or 0
    return (w >> 16) & 0xFFFF if addr % 4 == 0 else w & 0xFFFF


def _header_text(emu) -> str:  # noqa: ANN001
    raw = b"".join(struct.pack(">I", emu.peek32(BUF + 4 * i) or 0) for i in range(4))
    return raw.split(b"\0")[0].decode("latin1")


def _model_to(s, target: int) -> None:  # noqa: ANN001
    for _ in range(120):
        cur = _word(s.emu, MODEL_W) >> 8
        if cur == target:
            return
        err = target - cur
        by = 1 if err > 0 else -1
        s.turn_burst(
            "ENC B", by * (4 if abs(err) > 3 else 1), 3 if abs(err) <= 3 else 6
        )
    raise AssertionError(
        f"MODEL stays at {_word(s.emu, MODEL_W) >> 8}, wanted {target}"
    )


@needs_emu
def test_the_header_names_the_patch_for_dx7_models_and_shows_numbers_otherwise() -> (
    None
):
    from tools.patchkit import pk_harness as H

    g = gen()
    rows = g.name_rows(bank_bytes())
    thr = g.patch_words()

    def patch(w: int) -> int:
        return sum(1 for k in range(1, 32) if w >= thr[k])

    s = H.Session(
        H.Cell(device="dt2", firmware="1.17", modules=[]),
        H.Goldens(),
        run_name="dx7-names",
    )
    s.open(IMAGE)
    seen: dict[int, set[str]] = {}
    numbers: list[str] = []
    try:
        s.reset()
        s.run_steps(tomllib.loads(SELECT.read_text())["step"])
        s.run_steps([{"tap": "SRC"}, {"wait_ms": 500}])
        emu = s.emu
        assert emu is not None

        def sweep(model: int, bursts: list[tuple[int, int]]) -> list[tuple[int, str]]:
            """Turn HARMONICS by (detents, events) bursts; (HARMONICS word, header text) after each."""
            out = []
            for by, ev in bursts:
                s.turn_burst("ENC C", by, ev)
                out.append((_word(emu, HARM_W), _header_text(emu)))
            return out

        walk = [(-4, 60)] + [(1, 3)] * 30 + [(4, 4)] * 12
        for model in (0, 17, 18, 19, 20, 21, 22, 24, 21, 18):
            _model_to(s, model)
            assert _word(emu, MODEL_W) >> 8 == model
            for w, text in sweep(model, walk):
                if 18 <= model <= 21:
                    assert text == rows[model - 18][patch(w)], (model, hex(w), text)
                    seen.setdefault(model, set()).add(text)
                else:
                    assert text == str(w >> 8), (
                        model,
                        hex(w),
                        text,
                    )  # the number, as before
                    numbers.append(text)
        # the walk crossed many patches of every DX7 bank, and the other models only ever showed numbers
        assert all(len(seen[m]) >= 8 for m in (18, 19, 20, 21)), {
            m: len(v) for m, v in seen.items()
        }
        assert len(set(numbers)) > 8
        s.health("dx7 names")
    finally:
        s.close()


def _dump(emu, name: str) -> None:  # noqa: ANN001
    """DX7_DUMP=DIR: the whole OLED frame as a PNG (for looking at a failing comparison)."""
    d = os.environ.get("DX7_DUMP")
    if d:
        from tools.control.digi_client import oled_png

        (Path(d) / f"{name}.png").write_bytes(oled_png(emu.frame()[1]))


def _label(emu, name: str = "", cols: tuple[int, int] = (73, 103)) -> bytes:  # noqa: ANN001
    """The OLED pixels of HARMONICS' inline value label (x 73 .. 103, y 31 .. 40; the frame is 8 pages of 128 columns)."""
    frame = emu.frame()[1]
    if name:
        _dump(emu, name)
    return bytes(
        frame[(7 - y // 8) + 8 * x] >> (y % 8) & 1
        for x in range(*cols)
        for y in range(31, 40)
    )


@needs_emu
def test_the_inline_label_names_the_patch_normally_and_for_the_held_steps_effective_model() -> (
    None
):
    """The label under HARMONICS on the SRC page (6 characters): the short name of the DX7 model's patch, the number
    otherwise, in normal mode and while a trig is held (p-lock editing), where the model is the step's own MODEL p-lock
    when it has one and the track's otherwise. Each state's label equals the label of the same state set up as the track's
    own model in normal mode (the references), and the references differ from each other."""
    from tools.patchkit import pk_harness as H

    s = H.Session(
        H.Cell(device="dt2", firmware="1.17", modules=[]),
        H.Goldens(),
        run_name="dx7-label",
    )
    s.open(IMAGE)
    try:
        s.reset()
        s.run_steps(tomllib.loads(SELECT.read_text())["step"])
        s.run_steps([{"tap": "SRC"}, {"wait_ms": 500}])
        emu = s.emu
        assert emu is not None
        # references: the track's own model in normal mode with HARMONICS at its maximum (a recent knob turn makes the page
        # show values; idle it shows the parameter names)
        s.turn_burst("ENC C", 4, 250)
        assert _word(emu, HARM_W) == 0x7F00
        ref: dict[int, bytes] = {}
        refm: dict[
            int, bytes
        ] = {}  # MODEL's own label (its name), to see which model a step has locked
        for m in range(18, 25):
            _model_to(s, m)
            s.turn_burst(
                "ENC C", 4, 3
            )  # a touch of HARMONICS (already at its maximum) makes its label show the value
            s.run_ms(500)
            refm[m] = _label(emu, "", MODEL_COLS)
            if m in (18, 19, 24):
                ref[m] = _label(emu, f"ref{m}")
        assert len(set(refm.values())) == 7, "MODEL's labels look the same"
        assert len({ref[18], ref[19], ref[24]}) == 3, (
            "the three references look the same"
        )
        _model_to(s, 18)
        s.turn_burst("ENC C", -4, 250)
        assert _word(emu, HARM_W) == 0
        # the pattern: steps 3, 5, 7 with trigs; step 3 p-locks MODEL 19 (DX7-2), step 5 MODEL 24 (SQUARE), step 7 none
        s.do_tap("RECORD")
        s.run_ms(200)
        for k in (3, 5, 7):
            s.do_tap(f"TRIG {k}")
            s.run_ms(150)
        for k, bursts in (
            (3, 2),
            (5, 40),
        ):  # step 3: DX7-2; step 5: pushed to the top, SQUARE (24)
            emu.button(H.button(f"TRIG {k}"), True)
            s.run_ms(300)
            for _ in range(
                bursts
            ):  # the first burst of three events only opens the lock; each later one moves MODEL by one
                s.turn_burst("ENC B", 1, 3)
                s.run_ms(300)
            emu.button(H.button(f"TRIG {k}"), False)
            s.run_ms(300)
        s.run_ms(
            300
        )  # grid recording stays on: a held trig edits its step (with it off, a trig key plays its track)
        assert _word(emu, MODEL_W) >> 8 == 18, (
            "the p-locks left the track's MODEL alone"
        )

        def held(k: int) -> bytes:
            """Hold step k and p-lock HARMONICS to its maximum: the label shows the effective value."""
            emu.button(H.button(f"TRIG {k}"), True)
            s.run_ms(300)
            s.turn_burst("ENC C", 4, 250)
            s.run_ms(400)
            out = _label(emu, f"held{k}")
            emu.button(H.button(f"TRIG {k}"), False)
            s.run_ms(300)
            return out

        got7 = held(7)
        assert got7 == ref[18], "a step without a MODEL lock: the track's model (DX7-1)"
        assert held(3) == ref[19], (
            "the step's MODEL lock (DX7-2): its bank's patch name"
        )
        assert held(5) == ref[24], (
            "the step's MODEL lock (SQUARE, not a DX7 model): the number"
        )
        assert _word(emu, HARM_W) == 0, (
            "the HARMONICS p-locks left the track's value alone"
        )
        s.health("dx7 label")
    finally:
        s.close()
