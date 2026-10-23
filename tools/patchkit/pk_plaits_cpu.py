"""DSP load of the PLAITS machine on a built image: the worst block and the share of the budget (finding 06).

    uv run python -m tools.patchkit.pk_plaits_cpu --image out/patch/IMAGE.syx --model 12 [--tracks 1] \\
        [--knobs worst | HARM=0x7f00,TIMB=0x4000,...] [--windows 500,2500] [--no-cycles] [--max-seconds 3600]
    uv run python -m tools.patchkit.pk_plaits_cpu --fuzz --model 12 [--seed 0] [--budget-seconds 180] [--confirm]
    uv run python -m tools.patchkit.pk_plaits_cpu --image out/patch/IMAGE.syx --models 13,13,13,13,12,12,6 \\
        [--knobs worst] [--no-cycles]       # a mixed run: one track per model, worst knobs per model

`--fuzz` first: a random search for the worst block on the isolated compiled routine (tools/patchkit/pk_plaits_fuzz.py,
minutes, no emulator); `--confirm` then runs the worst sequence's static knobs and note through the emulator below.

Plays TRACKS tracks of PLAITS model MODEL on the emulator (one trig per track: PLAITS renders every block from its
first trig on while the stock voice is active), sets the knobs, and measures the DSP over a window of audio:

- busy instructions per block, from the engine's counters (`status.audio.dsp_tiers`); the load is projected with the
  stock mix's cycle factors (tools/patchkit/pk_dsp_budget.py), which is low by about 14 % at the high bound for
  compiled C (finding 06), so it is a quick figure only (`--no-cycles` stops here);
- the default: the cycle model of tools.sharc_dev.sharc_cpi (the engine's SHARC_CYCMIX profile, interpreter only,
  slow: about 10 minutes per second of audio) over two windows, the short one cancelling boot, reset and the panel
  steps: cycles per block = (long - short) / blocks between them, low / nominal / high, as a share of the 666,667
  cycles of a block. The verdict is the high bound against the budget (85 %, the safety rule of the all-engines plan:
  an engine over it ships silent).

The image's program database (`uv run python -m tools.sharc_dev.sharcdb build ...`, out/sharcdb/IMAGE.sqlite) is
needed for the cycle model. Results go to out/runs/plaits-cpu/ (never committed). The emulator is always stopped:
on a result, an error, Ctrl-C, SIGTERM and when --max-seconds runs out. Exit 0 when the high bound is within the
budget.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

from tools.patchkit import pk_dsp_budget, pk_plaits_fuzz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "runs" / "plaits-cpu"
DEFAULT_IMAGE = ROOT / "out" / "patch" / "plaits-13models-names-splash-48k-1.17.syx"
DEFAULT_MODS = ["ui/boot-splash", "machines/plaits-48k"]
BLOCK_RATE = pk_dsp_budget.BLOCK_RATE
BLOCK_CYCLES = pk_dsp_budget.BLOCK_CYCLES_EXACT
BUDGET_PCT = pk_dsp_budget.DEFAULT_BUDGET_PCT
FUNC_SRC_DOWNS = (
    7  # FUNC+SRC, 7 x DOWN, YES, YES: the machine list's PLAITS entry (pk_proof_plaits)
)
MODEL_SLOT = 26  # mirror index of knob B (MODEL: the model number << 8)

# knob name -> (mirror index, encoder) of the SRC page (pk_proof_plaits)
KNOBS: dict[str, tuple[int, str]] = {
    "TUNE": (25, "ENC A"),
    "HARM": (27, "ENC C"),
    "TIMB": (31, "ENC E"),
    "MORPH": (32, "ENC F"),
    "LEV": (34, "ENC H"),
}
# model -> the worst block's knobs found on the concrete core's extreme grid (tests/patchkit/test_plaits_module.py
# extreme tests). Add the model's line when its extreme test names the worst setting; models not listed run at
# the machine's defaults, which is NOT the worst case.
WORST: dict[int, dict[str, int]] = {
    # DX7 (one engine for 18, 19, 20 and 24): the fuzz worst of model 18
    18: {"TUNE": 0x4300, "HARM": 0x7F00, "TIMB": 0x37E4, "MORPH": 0x7F00},
    5: {"TUNE": 0x7C00, "HARM": 0x7F00, "TIMB": 0x4000, "MORPH": 0x4000},
    6: {"TUNE": 0x4400, "HARM": 0, "TIMB": 0x7F00, "MORPH": 0x3C00},
    7: {"TUNE": 0x7C00, "HARM": 0x0400, "TIMB": 0x7F00, "MORPH": 0x4000},
    # NOISE: HARMONICS and TIMBRE at the top as the fuzz worst has them (note 6, TUNE 0xa00). The fuzz worst has MORPH
    # 0x173c, but there the cycle model failed (negative cycles, see cycle_model_ok); with MORPH 0x7f00 it measured
    # 73.1 % (windows 100,500) and the instruction counts agree to 0.3 %.
    9: {"TUNE": 0x0A00, "HARM": 0x7F00, "TIMB": 0x7F00, "MORPH": 0x7F00},
    # PHASE-DIST: the fuzz worst's HARMONICS, TIMBRE and MORPH (0x7f00, 0x3c00, 0xa00). The fuzz worst is a transient at
    # note 85 (TUNE 0x5900); a held note costs most near the top of the range (rig, instructions per block: note 85
    # 90,456, 110 99,452, 120 99,119), so TUNE 0x7c00 (note 120) measures the steady state (within 1.8 % of the fuzz worst).
    # VCF-VA: the fuzz (task-vcf, seed 1) worst: a high note, resonance off (HARMONICS 0), the cutoff high, MORPH at the top
    16: {"TUNE": 0x7C00, "HARM": 0x0000, "TIMB": 0x624A, "MORPH": 0x7F00},
    # SQUARE: the fuzz (seed 1) worst: a very high note, HARMONICS in a four-note chord, TIMBRE 0x3c00 (a pattern that
    # clocks), MORPH at the top (the slave a long way above the master: the most sync resets and band-limiting steps)
    23: {"TUNE": 0x7C00, "HARM": 0x3A06, "TIMB": 0x3C00, "MORPH": 0x7F00},
    # DUST: the fuzz (seed 1) worst: a very high note (the particles' band-pass at its highest cutoffs), HARMONICS 0x2299,
    # TIMBRE at the top (the densest particles), MORPH 0 (the diffuser fully wet)
    # SWARM (continuous mode): the fuzz (seed 1) worst: the highest note, HARMONICS 0, TIMBRE 0x2b55, MORPH 0x4d3f (the cost is nearly flat in the knobs)
    8: {"TUNE": 0x7C00, "HARM": 0x0000, "TIMB": 0x2B55, "MORPH": 0x4D3F},
    # STRING: the fuzz (seed 1, 53 sequences) worst: note 59, HARMONICS at the top (dispersion at its deepest: the stretch all-pass and the
    # noise-modulated delay), TIMBRE 0x382a, MORPH 0x2bce, at the note-on that selects the model (the clear of the three strings'
    # 3,840 delay-line words is in that block)
    11: {"TUNE": 0x3F00, "HARM": 0x7F00, "TIMB": 0x382A, "MORPH": 0x2BCE},
    10: {"TUNE": 0x6D00, "HARM": 0x2299, "TIMB": 0x7F00, "MORPH": 0x0000},
    # STRING MACHINE: the fuzz (seed 1) worst: a mid note, HARMONICS in a four-note chord (a 4-voice chord with all of
    # the oscillators' band-limiting steps), TIMBRE 0 (the ensemble fully wet at its deepest), MORPH at the top
    22: {"TUNE": 0x6A00, "HARM": 0x3AD2, "TIMB": 0x0000, "MORPH": 0x7F00},
    17: {"TUNE": 0x7C00, "HARM": 0x7F00, "TIMB": 0x3C00, "MORPH": 0x0A00},
    12: {"MORPH": 0x7F00},
    # the drums are nearly flat in their knobs (finding 06): the decay at its longest (a window past the stock voice's
    # life reads the baseline; why MORPH matters to it is not established), the lowest TUNE the
    # panel reaches (0x400) for KICK and SNARE, the highest (0x7c00) with HARMONICS 0x7f00 for HIHAT
    13: {"TUNE": 0x0400, "HARM": 0, "TIMB": 0, "MORPH": 0x7F00},
    14: {"TUNE": 0x0400, "HARM": 0, "TIMB": 0, "MORPH": 0x7F00},
    15: {"TUNE": 0x7C00, "HARM": 0x7F00, "TIMB": 0, "MORPH": 0x7F00},
    21: {"HARM": 0x0A00, "TIMB": 0x7F00},
}


class Timeout(Exception):
    """--max-seconds ran out."""


# -- pure parts (tests/patchkit/test_plaits_cpu.py)


def parse_knobs(spec: str, model: int) -> dict[str, int]:
    """`worst` (the model's WORST line), `default` or '' (none), or `HARM=0x7f00,TIMB=0` (names of KNOBS)."""
    if spec in ("", "default"):
        return {}
    if spec == "worst":
        return dict(WORST.get(model, {}))
    out: dict[str, int] = {}
    for part in spec.split(","):
        name, eq, val = part.partition("=")
        name = name.strip().upper()
        if not eq or name not in KNOBS:
            raise ValueError(
                f"bad knob {part!r}: NAME=VALUE with NAME in {sorted(KNOBS)}"
            )
        v = int(val, 0)
        if not 0 <= v <= 0x7F00:
            raise ValueError(f"knob {name}: {v:#x} outside 0..0x7f00")
        out[name] = v
    return out


def parse_windows(spec: str) -> tuple[int, int]:
    """`SHORT,LONG` in ms of audio, SHORT < LONG (the cycle model's difference needs both)."""
    a, _, b = spec.partition(",")
    short, long_ = int(a), int(b)
    if not 0 < short < long_:
        raise ValueError(f"windows {spec!r}: need 0 < SHORT < LONG")
    return short, long_


def busy_per_block(busy_instructions: float, audio_s: float) -> float:
    """Mean busy instructions per block over a window of AUDIO_S seconds."""
    if audio_s <= 0:
        raise ValueError("the window had no audio")
    return busy_instructions / (audio_s * BLOCK_RATE)


def cycles_per_block(short: dict[str, Any], long_: dict[str, Any]) -> list[float]:
    """Real cycles per block, [low, nominal, high]: (long - short) / the blocks between the windows. Each run dict
    has window_audio_s and total_cycles (the cycle model's totals over the whole run: boot and panel steps cancel)."""
    blocks = (long_["window_audio_s"] - short["window_audio_s"]) * BLOCK_RATE
    if blocks <= 0:
        raise ValueError("the long window must hold more audio than the short one")
    return [
        (b - a) / blocks
        for a, b in zip(short["total_cycles"], long_["total_cycles"], strict=True)
    ]


def load_pct(cycles: float) -> float:
    """Share of a block's 666,667 cycles."""
    return 100.0 * cycles / BLOCK_CYCLES


# Busy instructions per block of the stock mix with a PLAITS track that sounds nothing (the RINGS and drum runs of
# finding 06: 209,891), and the margin over it below which a run is taken as not having measured PLAITS at all.
# A cycle figure is plausible only when cycles per busy instruction are between CPI_MIN (low bound) and CPI_MAX (high
# bound): stock code runs at 1.3 to 2.9 (finding 06). Outside it the cycle model's subtraction is broken (see
# cycle_model_ok) and the run FAILS: never a pass.
CPI_MIN = 1.0
CPI_MAX = 6.0
STOCK_BUSY = 209_891
MIN_DELTA = 20_000


def cycle_model_ok(cyc: list[float], busy: float) -> bool:
    """Whether the cycle figures can be a measurement: all positive, and between CPI_MIN and CPI_MAX cycles per busy
    instruction. OPEN: the figures are (long - short) of two runs' totals, which assumes boot and the panel steps cost
    the same in both; they do not always (model 9 at MORPH 0x173c: the 100 ms run's total was larger than the 500 ms
    run's, -1.4 M cycles per block; at MORPH 0x7f00 it was fine). The likely cause is the panel steps settling
    differently for that knob value; not established."""
    return (
        busy > 0
        and all(c > 0 for c in cyc)
        and cyc[0] >= CPI_MIN * busy
        and cyc[2] <= CPI_MAX * busy
    )


def report(short: dict[str, Any], long_: dict[str, Any] | None) -> dict[str, Any]:
    """The result: instruction counters of the long window and, with the cycle model, the loads and the verdict."""
    ref = long_ or short
    busy = busy_per_block(ref["window_busy_instructions"], ref["window_audio_s"])
    res: dict[str, Any] = {
        "busy_per_block": busy,
        "stock_mix_load_nominal_pct": pk_dsp_budget.load_pct(busy, "dt2", "nominal"),
        "stock_mix_load_high_pct": pk_dsp_budget.load_pct(busy, "dt2", "high"),
        "budget_pct": BUDGET_PCT,
        "delta_over_stock": busy - STOCK_BUSY,
    }
    if long_ is not None and "total_cycles" in short and "total_cycles" in long_:
        cyc = cycles_per_block(short, long_)
        pct = [load_pct(c) for c in cyc]
        res["cycles_per_block"] = cyc
        res["load_pct"] = {"low": pct[0], "nominal": pct[1], "high": pct[2]}
        res["cycle_model_invalid"] = not cycle_model_ok(cyc, busy)
        res["ok"] = pct[2] <= BUDGET_PCT and not res["cycle_model_invalid"]
    else:
        res["ok"] = res["stock_mix_load_high_pct"] <= BUDGET_PCT
    # a window past the voice's life reads the stock baseline: a PASS-looking figure that measured nothing
    res["no_signal"] = res["delta_over_stock"] < MIN_DELTA
    if res["no_signal"]:
        res["ok"] = False
    return res


def summary(model: int, tracks: int, knobs: dict[str, int], res: dict[str, Any]) -> str:
    k = ",".join(f"{n}={v:#x}" for n, v in knobs.items()) or "machine defaults"
    lines = [
        f"PLAITS model {model}, {tracks} track(s), knobs {k}",
        f"  busy {res['busy_per_block']:,.0f} instructions per block (stock-mix projection "
        f"{res['stock_mix_load_nominal_pct']:.1f} % nominal / {res['stock_mix_load_high_pct']:.1f} % high)",
    ]
    if "load_pct" in res:
        p = res["load_pct"]
        c = res["cycles_per_block"]
        lines.append(
            f"  cycle model: {c[2]:,.0f} cycles per block at the high bound; load {p['low']:.1f} / "
            f"{p['nominal']:.1f} / {p['high']:.1f} % (low / nominal / high)"
        )
    if res.get("cycle_model_invalid"):
        lines.append(
            "  CYCLE MODEL INVALID: the cycle figures are negative or not within "
            f"{CPI_MIN:g} to {CPI_MAX:g} cycles per busy instruction (the two windows' boot and panel steps did not "
            "cancel); the load above is not a measurement. Rerun, or change the knobs (MORPH 0x7f00)."
        )
    if res.get("no_signal"):
        lines.append(
            f"  NO SIGNAL: only {res['delta_over_stock']:+,.0f} busy instructions per block over the stock baseline "
            f"({STOCK_BUSY:,}; margin {MIN_DELTA:,}): the window did not contain PLAITS rendering (the stock voice's "
            "ACTIVE had ended: finding 06; a longer decay, MORPH 0x7f00, kept it going for RINGS and the drums). "
            "The load above is not the model's."
        )
    which = "high bound" if "load_pct" in res else "stock-mix high projection"
    lines.append(
        f"  {'within' if res['ok'] else 'OVER'} the {res['budget_pct']:g} % budget ({which})"
    )
    return "\n".join(lines)


# -- the emulator run


def _alarm(_sig: int, _frame: FrameType | None) -> None:
    raise Timeout("--max-seconds ran out")


def _term(_sig: int, _frame: FrameType | None) -> None:
    raise SystemExit(143)


def tiers(emu: Any) -> dict[str, Any]:
    return (emu.status().get("audio") or {}).get("dsp_tiers") or {}


def select_plaits(s: Any) -> None:
    s.do_tap(["FUNC", "SRC"])
    s.run_ms(200)
    for _ in range(FUNC_SRC_DOWNS):
        s.do_tap("DOWN")
        s.run_ms(80)
    s.do_tap("YES")
    s.run_ms(300)
    s.do_tap("YES")
    s.run_ms(300)


def set_model(s: Any, n: int, track: int) -> None:
    """The MODEL walk of the emulator proof (pk_proof_plaits.PlaitsProof.set_model: bursts while far, single detents
    near): 23 single detents of 160 ms each took longer than --max-seconds for models 16 and up."""
    from tools.patchkit import pk_proof_plaits

    s.do_tap("SRC")
    s.run_ms(200)
    got = pk_proof_plaits.PlaitsProof.set_model(None, s, n, track)  # type: ignore[arg-type]
    if got != n << 8:
        raise RuntimeError(f"MODEL stuck at {got:#x}, wanted {n << 8:#x}")


def run_once(
    image: Path,
    mods: list[str],
    model: int | list[int],
    tracks: int,
    knobs: dict[str, int] | list[dict[str, int]],
    window_ms: int,
    cycles: bool,
    stock: bool = False,
) -> dict[str, Any]:
    """One emulator run with TRACKS tracks of MODEL, measured over WINDOW_MS of audio; or, with MODEL a list (and
    KNOBS a list beside it), track t plays MODEL[t] at KNOBS[t]. The emulator is stopped on every path out."""
    models = model if isinstance(model, list) else [model] * tracks
    track_knobs = knobs if isinstance(knobs, list) else [knobs] * tracks
    from tools.patchkit import pk_harness as H
    from tools.sharc_dev import sharc_cpi

    OUT.mkdir(parents=True, exist_ok=True)
    # --stock: the stock image's own program database, and TRIG 1 on its sample (finding 06's stock row)
    db = STOCK_DB if stock else image.name.removesuffix(".syx")
    tag = f"{run_tag(models) if models else 'stock' + ('-' + TAG if TAG else '')}-{window_ms}"
    tsv = OUT / f"{tag}.tsv"
    env: dict[str, str] = {}
    if cycles:
        memspec = MEMSPEC or sharc_cpi.memspec_path(db)
        if not memspec.is_file():
            sharc_cpi.cmd_memspec(argparse.Namespace(image=db, out=None))
        tsv.unlink(missing_ok=True)
        env = {
            "SHARC_CYCMIX": "1",
            "SHARC_CYCMIX_OUT": str(tsv),
            "SHARC_CYCMIX_MEM": str(memspec),
            "SHARC_FAST_REGIONS": "",
            "SHARC_BLOCKTRANS": "",
            "SHARC_AOT": "0",
        }
    started = time.monotonic()
    s = H.Session(
        H.Cell("dt2", "1.17", mods), H.Goldens(), run_name=f"plaits-cpu-{tag}", env=env
    )
    res: dict[str, Any] = {
        "model": model,
        "tracks": len(models),
        "window_ms": window_ms,
    }
    try:
        s.open(image=image)
        emu = s.emu
        assert emu is not None
        s.reset()
        s.run_ms(1500)
        if stock:
            s.do_hold("TRIG 1", 40)
            s.run_ms(200)
        for t in range(len(models)):
            if t:
                s.do_tap(["TRK", f"TRIG {t + 1}"])
                s.run_ms(200)
            select_plaits(s)
            s.do_hold(
                f"TRIG {t + 1}", 40
            )  # the first trig loads the machine's defaults into the slots
            s.run_ms(200)
            set_model(s, models[t], t)
            for name, val in track_knobs[t].items():
                idx, enc = KNOBS[name]
                s.turn_to(enc, idx, val, track=t, tol=0x100)
            s.do_hold(
                f"TRIG {t + 1}", 40
            )  # PLAITS renders from its first trig while the stock voice is active (finding 06)
            s.run_ms(200)
        res["slots"] = {
            str(t): {
                n: hex(s.mirror(i, t))
                for n, (i, _e) in {"MODEL": (MODEL_SLOT, ""), **KNOBS}.items()
            }
            for t in range(len(models))
        }
        t0 = tiers(emu)
        s.run_ms(window_ms)
        t1 = tiers(emu)
        s.health("window")
        audio = t1["audio_s"] - t0["audio_s"]
        busy = (t1["instructions"] - t0["instructions"]) - (
            t1["idle_skip"] - t0["idle_skip"]
        )
        res.update(window_audio_s=audio, window_busy_instructions=busy)
    finally:
        s.close()  # the profile file is written when the emulator exits
    if cycles:
        for _ in range(100):
            if tsv.is_file() and tsv.stat().st_size:
                break
            time.sleep(0.2)
        cost = sharc_cpi.cost_profile(
            sharc_cpi.Profile(tsv), db, 1
        )  # blocks = 1: totals over the whole run
        res["total_cycles"] = cost["cycles_per_block"]
        res["undecoded_instructions"] = cost["undecoded_instructions"]
    res["wall_s"] = time.monotonic() - started
    return res


# --tag: a suffix of every file and run directory name, so runs of the same models side by side (two lanes share
# out/) do not overwrite each other's profile.
TAG = ""
STOCK_IMAGE = ROOT / "firmware" / "Digitakt_II_OS1.17.syx"
STOCK_DB = "dt2-1.17"
# --memspec: the engine's per-PC file given outright (an A/B against an engine built before the memspec-ea slots,
# which needs the old bare-I format)
MEMSPEC: Path | None = None


def run_tag(models: list[int]) -> str:
    """`mN-tK` for K tracks of one model, else `mix-A_B_..` (the models in track order), plus `-TAG` with --tag."""
    if len(set(models)) == 1:
        base = f"m{models[0]}-t{len(models)}"
    else:
        base = "mix-" + "_".join(str(m) for m in models)
    return f"{base}-{TAG}" if TAG else base


def parse_models(spec: str) -> list[int]:
    models = [int(x) for x in spec.split(",") if x.strip()]
    if not 1 <= len(models) <= 16 or not all(0 <= m <= 24 for m in models):
        raise ValueError("--models: 1..16 model numbers 0..24")
    return models


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None
    )
    ap.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="the built .syx (default: the owner image)",
    )
    ap.add_argument("--model", type=int, help="PLAITS model number 0..24")
    ap.add_argument(
        "--models",
        default=None,
        help="A,B,...: one PLAITS track per entry, track t on model A.. (a mixed run; each model at --knobs, "
        "`worst` per model); replaces --model and --tracks",
    )
    ap.add_argument(
        "--tracks", type=int, default=1, help="PLAITS tracks, all on MODEL (default 1)"
    )
    ap.add_argument(
        "--knobs",
        default="default",
        help="worst | default | NAME=VALUE,... (TUNE HARM TIMB MORPH LEV)",
    )
    ap.add_argument(
        "--windows",
        default="500,2500",
        help="SHORT,LONG audio ms of the cycle model (default 500,2500)",
    )
    ap.add_argument(
        "--no-cycles",
        action="store_true",
        help="instruction counters only, one window (the LONG one)",
    )
    ap.add_argument(
        "--mods",
        default=",".join(DEFAULT_MODS),
        help="harness cell modules (labels; the image is loaded as is)",
    )
    ap.add_argument(
        "--max-seconds",
        type=int,
        default=3600,
        help="wall-clock bound of the whole command (default 3600)",
    )
    ap.add_argument(
        "--fuzz",
        action="store_true",
        help="search the worst block on the isolated compiled routine first (pk_plaits_fuzz)",
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="with --fuzz: run the worst sequence's static knobs through the emulator path below",
    )
    ap.add_argument(
        "--memspec",
        type=Path,
        default=None,
        help="the cycle profile's per-PC file (default: sharc_cpi's memspec-ea file of the image)",
    )
    ap.add_argument(
        "--stock",
        action="store_true",
        help="the stock baseline: stock 1.17 with TRIG 1 on its sample, no PLAITS (cycle model as for the models)",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="suffix of the run's file and run directory names (runs side by side)",
    )
    pk_plaits_fuzz.add_arguments(ap)
    args = ap.parse_args(argv)
    global TAG, MEMSPEC
    TAG = args.tag
    MEMSPEC = args.memspec
    if args.confirm and not args.fuzz:
        ap.error("--confirm needs --fuzz")
    if args.stock:
        return main_stock(args)
    if (args.model is None) == (args.models is None):
        ap.error("give exactly one of --model N and --models A,B,..")
    if args.models is not None:
        if args.fuzz:
            ap.error("--fuzz takes --model")
        try:
            models = parse_models(args.models)
        except ValueError as e:
            ap.error(str(e))
        return main_mixed(args, models)
    fuzz_res: dict[str, Any] | None = None
    fuzz_path: Path | None = None
    if args.fuzz:
        fuzz_res = pk_plaits_fuzz.fuzz(
            args.model,
            args.seed,
            args.budget_seconds,
            args.jobs,
            args.sequences,
            args.fuzz_mode,
        )
        fuzz_path = pk_plaits_fuzz.save(fuzz_res)
        print(pk_plaits_fuzz.summary(fuzz_res))
        print(f"  saved {fuzz_path}")
        if not args.confirm:
            return 0
        args.knobs = ",".join(
            f"{n}={v:#x}" for n, v in fuzz_res["worst"]["confirm_knobs"].items()
        )
    if not 0 <= args.model <= 24:
        ap.error("--model: 0..24")
    if not 1 <= args.tracks <= 16:
        ap.error("--tracks: 1..16")
    try:
        knobs = parse_knobs(args.knobs, args.model)
        short_ms, long_ms = parse_windows(args.windows)
    except ValueError as e:
        ap.error(str(e))
    if args.knobs == "worst" and not knobs:
        print(
            f"note: no WORST line for model {args.model}; running at the machine's defaults",
            file=sys.stderr,
        )
    image = args.image if args.image.is_absolute() else ROOT / args.image
    if not image.is_file():
        ap.error(f"image {image} not found")
    mods = [m for m in args.mods.split(",") if m]
    cycles = not args.no_cycles

    signal.signal(signal.SIGALRM, _alarm)
    signal.signal(signal.SIGTERM, _term)
    signal.alarm(args.max_seconds)
    try:
        runs = (
            [run_once(image, mods, args.model, args.tracks, knobs, short_ms, cycles)]
            if cycles
            else []
        )
        runs.append(
            run_once(image, mods, args.model, args.tracks, knobs, long_ms, cycles)
        )
        res = report(runs[0], runs[-1] if cycles else None)
    except (Timeout, KeyboardInterrupt) as e:
        print(
            f"stopped: {e or 'interrupted'}; the emulator was stopped", file=sys.stderr
        )
        return 2
    finally:
        signal.alarm(0)
    res["runs"] = runs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run_tag([args.model] * args.tracks)}.json").write_text(
        json.dumps(res, indent=1)
    )
    print(summary(args.model, args.tracks, knobs, res))
    if fuzz_res is not None and fuzz_path is not None:
        fuzz_res["confirm"] = {
            "knobs": knobs,
            "busy_per_block": res["busy_per_block"],
            "load_pct": res.get("load_pct"),
            "no_signal": res["no_signal"],
            "cycle_model_invalid": res.get("cycle_model_invalid", False),
            "ok": res["ok"],
        }
        fuzz_path.write_text(json.dumps(fuzz_res, indent=1, sort_keys=True))
        if "load_pct" in res:
            print(
                f"  confirmed: {res['load_pct']['high']:.1f} % of the DSP budget at the high bound "
                f"({res['load_pct']['nominal']:.1f} % nominal)"
            )
    return 0 if res["ok"] else 1


def main_stock(args: argparse.Namespace) -> int:
    """The stock baseline through the same harness and cycle model (two windows, (long - short) per block)."""
    short_ms, long_ms = parse_windows(args.windows)
    cycles = not args.no_cycles
    signal.signal(signal.SIGALRM, _alarm)
    signal.signal(signal.SIGTERM, _term)
    signal.alarm(args.max_seconds)
    try:
        runs = (
            [run_once(STOCK_IMAGE, [], [], 0, [], short_ms, cycles, stock=True)]
            if cycles
            else []
        )
        runs.append(run_once(STOCK_IMAGE, [], [], 0, [], long_ms, cycles, stock=True))
        res = report(runs[0], runs[-1] if cycles else None)
    except (Timeout, KeyboardInterrupt) as e:
        print(
            f"stopped: {e or 'interrupted'}; the emulator was stopped", file=sys.stderr
        )
        return 2
    finally:
        signal.alarm(0)
    res["runs"] = runs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"stock{'-' + TAG if TAG else ''}.json").write_text(
        json.dumps(res, indent=1)
    )
    print("stock 1.17, TRIG 1 on its sample (no PLAITS)")
    print(f"  busy {res['busy_per_block']:,.0f} instructions per block")
    if "load_pct" in res:
        lp = res["load_pct"]
        print(
            f"  cycle model: {res['cycles_per_block'][2]:,.0f} cycles per block at the high bound; "
            f"load {lp['low']:.1f} / {lp['nominal']:.1f} / {lp['high']:.1f} % (low / nominal / high)"
        )
    return 0


def main_mixed(args: argparse.Namespace, models: list[int]) -> int:
    """A mixed run: track t on models[t], each at its own --knobs (worst knobs per model)."""
    knobs = [parse_knobs(args.knobs, m) for m in models]
    short_ms, long_ms = parse_windows(args.windows)
    image = args.image if args.image.is_absolute() else ROOT / args.image
    if not image.is_file():
        raise SystemExit(f"image {image} not found")
    mods = [m for m in args.mods.split(",") if m]
    cycles = not args.no_cycles
    signal.signal(signal.SIGALRM, _alarm)
    signal.signal(signal.SIGTERM, _term)
    signal.alarm(args.max_seconds)
    try:
        runs = (
            [run_once(image, mods, models, len(models), knobs, short_ms, cycles)]
            if cycles
            else []
        )
        runs.append(run_once(image, mods, models, len(models), knobs, long_ms, cycles))
        res = report(runs[0], runs[-1] if cycles else None)
    except (Timeout, KeyboardInterrupt) as e:
        print(
            f"stopped: {e or 'interrupted'}; the emulator was stopped", file=sys.stderr
        )
        return 2
    finally:
        signal.alarm(0)
    res["runs"] = runs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{run_tag(models)}.json").write_text(json.dumps(res, indent=1))
    print(f"PLAITS models {','.join(map(str, models))} (one track each)")
    print("\n".join(summary(models[0], len(models), knobs[0], res).splitlines()[1:]))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
