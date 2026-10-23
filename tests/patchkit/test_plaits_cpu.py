"""tools/patchkit/pk_plaits_cpu.py without the emulator: the argument parsing and the result math."""

from __future__ import annotations

from typing import Any

import pytest

from tools.patchkit import pk_plaits_cpu as C


def test_parse_knobs() -> None:
    assert C.parse_knobs("default", 12) == {}
    assert C.parse_knobs("", 12) == {}
    assert C.parse_knobs("worst", 12) == {"MORPH": 0x7F00}
    assert C.parse_knobs("worst", 99) == {}
    assert C.parse_knobs("harm=0x7f00, timb=0", 0) == {"HARM": 0x7F00, "TIMB": 0}
    for bad in ("HARM", "FOO=1", "HARM=0x8000", "HARM=-1"):
        with pytest.raises(ValueError):
            C.parse_knobs(bad, 0)


def test_worst_knobs_are_known_names() -> None:
    for knobs in C.WORST.values():
        assert set(knobs) <= set(C.KNOBS)


def test_parse_windows() -> None:
    assert C.parse_windows("500,2500") == (500, 2500)
    for bad in ("2500,500", "0,5", "5,5", "5"):
        with pytest.raises(ValueError):
            C.parse_windows(bad)


def run(
    audio_s: float, busy: float, cycles: list[float] | None = None
) -> dict[str, Any]:
    r: dict[str, Any] = {"window_audio_s": audio_s, "window_busy_instructions": busy}
    if cycles is not None:
        r["total_cycles"] = cycles
    return r


def test_busy_per_block() -> None:
    assert C.busy_per_block(1500 * 200_000, 1.0) == 200_000
    with pytest.raises(ValueError):
        C.busy_per_block(1, 0.0)


def test_cycle_model_difference_and_verdict() -> None:
    # 0.5 s and 2.5 s windows: 3,000 blocks between them; 666,666.67 cycles per block is 100 %
    short = run(0.5, 1e8, [1.0e9, 2.0e9, 3.0e9])
    long_ = run(
        2.5,
        1e9,
        [1.0e9 + 3000 * 400_000, 2.0e9 + 3000 * 500_000, 3.0e9 + 3000 * 600_000],
    )
    assert C.cycles_per_block(short, long_) == pytest.approx(
        [400_000, 500_000, 600_000]
    )
    res = C.report(short, long_)
    assert res["load_pct"]["high"] == pytest.approx(90.0)
    assert res["load_pct"]["low"] == pytest.approx(60.0)
    assert res["ok"] is False
    assert res["busy_per_block"] == pytest.approx(1e9 / (2.5 * 1500))
    assert "OVER" in C.summary(12, 1, {"MORPH": 0x7F00}, res)
    long_["total_cycles"] = [
        1.0e9 + 3000 * 300_000,
        2.0e9 + 3000 * 400_000,
        3.0e9 + 3000 * 500_000,
    ]
    res = C.report(short, long_)
    assert res["load_pct"]["high"] == pytest.approx(75.0)
    assert res["ok"] is True
    assert "within" in C.summary(12, 1, {}, res)


def test_counters_only_report() -> None:
    res = C.report(run(2.5, 1.0e6 * 3750), None)  # 1e6 busy instructions per block
    assert "load_pct" not in res
    assert res["busy_per_block"] == pytest.approx(1.0e6)
    assert res["ok"] is False  # over any budget
    with pytest.raises(ValueError):
        C.cycles_per_block(run(1.0, 0, [1, 2, 3]), run(1.0, 0, [1, 2, 3]))


def test_main_rejects_bad_arguments() -> None:
    with pytest.raises(SystemExit):
        C.main(["--model", "99"])
    with pytest.raises(SystemExit):
        C.main(["--model", "1", "--knobs", "FOO=1"])
    with pytest.raises(SystemExit):
        C.main(["--model", "1", "--image", "/nonexistent.syx"])


def test_baseline_guard() -> None:
    """A run at the stock baseline (the window past the voice's life) is not a PLAITS measurement: not ok, with a
    message; PLAITS's own increment passes."""
    base = C.STOCK_BUSY * 3750 / 2.5 * 2.5 / 3750  # busy per block == STOCK_BUSY
    res = C.report(run(2.5, base * 3750), None)
    assert res["no_signal"] is True and res["ok"] is False
    assert "NO SIGNAL" in C.summary(13, 1, {}, res)
    near = C.report(run(2.5, (C.STOCK_BUSY + C.MIN_DELTA - 1) * 3750), None)
    assert near["no_signal"] is True
    ok = C.report(run(2.5, (C.STOCK_BUSY + C.MIN_DELTA + 1) * 3750), None)
    assert ok["no_signal"] is False and ok["ok"] is True
    assert "NO SIGNAL" not in C.summary(13, 1, {}, ok)


def test_negative_or_implausible_cycles_fail_closed() -> None:
    """The two runs of NOISE at MORPH 0x173c (the long run's cycle total was below the short run's): a load of -215 %
    is not a pass: cycle_model_invalid, ok False, the summary says so (the exit code of main is 1 for ok False)."""
    short = run(0.1, 41592330, [9.43e9, 10.48e9, 12.94e9])
    long_ = run(0.5, 208076634, [8.81e9, 9.79e9, 12.08e9])
    res = C.report(short, long_)
    assert res["load_pct"]["high"] < 0
    assert res["cycle_model_invalid"] is True and res["ok"] is False
    assert "CYCLE MODEL INVALID" in C.summary(9, 1, {}, res)
    # the valid pair of the same model at MORPH 0x7f00
    good = C.report(
        run(0.1, 41492154, [8.0e9, 8.88e9, 10.96e9]),
        run(0.5, 207446370, [8.21e9, 9.12e9, 11.26e9]),
    )
    assert good["cycle_model_invalid"] is False and good["ok"] is True
    # too many cycles per instruction is as impossible as too few
    slow = C.report(run(0.1, 4e7, [1e9, 1e9, 1e9]), run(0.5, 2e8, [1e9 + 1.2e9] * 3))
    assert slow["cycle_model_invalid"] is True and slow["ok"] is False


def test_mixed_run_tag_and_models() -> None:
    from tools.patchkit import pk_plaits_cpu as C

    assert C.run_tag([6]) == "m6-t1"
    assert C.run_tag([13, 13]) == "m13-t2"
    assert C.run_tag([13, 13, 12, 6]) == "mix-13_13_12_6"
    assert C.parse_models("13,13,13,13,12,12,6") == [13, 13, 13, 13, 12, 12, 6]
    for bad in ("", "25", ",".join(["0"] * 17)):
        try:
            C.parse_models(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
