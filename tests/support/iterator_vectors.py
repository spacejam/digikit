"""Reference vectors of the ITERATOR engine: the output of the reference voice as (int32 8.24 sample, uint32 period)
streams, recorded bit-exact for a set of control sequences (tests/iterator/vectors/manifest.json and streams.npz).

Each case is a control timeline in module milliseconds. The voice is driven like this (Manifest.drive_rule): the module
clock is the sum of the periods rendered so far (240 MHz ticks, uint32). Before a sample is rendered, if
clock >= next_call_tick (initially 0), control call k is made with the timeline state in effect at k ms, and
next_call_tick = (k + 1) * 240000. Rendering stops when the clock reaches duration_ms * 240000. A call whose gate is high
while the previous call's gate was low is a trigger, with that call's controls. `calls(case)` expands the timeline into
the controls of every call, so an engine under test is driven by `controls_at(case, k)` and compared on
(sample, period) with `compare`. Long cases store only the first `stored_samples` samples; `digests_match` checks the
whole stream by SHA-256 either way."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
VECTOR_DIR = ROOT / "tests" / "iterator" / "vectors"
TICKS_PER_MS = 240000
MODE_NAMES = (
    "SKN-L",
    "MTL-L",
    "LIQ-L",
    "SKN-M",
    "MTL-M",
    "LIQ-M",
    "SKN-H",
    "MTL-H",
    "LIQ-H",
)

I32: TypeAlias = NDArray[np.int32]
U32: TypeAlias = NDArray[np.uint32]


@dataclass(frozen=True)
class Controls:
    """One control call. mode is 0..8 (MODE_NAMES); range = mode // 3; engine mode word = mode % 3 (0 skin, 1 metal,
    2 liquid). Knobs are the control floats as float32 values."""

    ms: int
    pitch: float
    attack: float
    decay: float
    morph: float
    fold: float
    harmonic: float
    spread: float
    mode: int
    gate: bool

    @property
    def range(self) -> int:
        return self.mode // 3

    @property
    def mode_word(self) -> int:
        return self.mode % 3


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    description: str
    duration_ms: int
    timeline: tuple[Controls, ...]
    n_samples: int
    n_calls: int
    trigger_samples: tuple[int, ...]
    stored_samples: int
    sha256_samples: str
    sha256_periods: str
    sha256_pairs: str
    samples: I32
    periods: U32
    call_samples: U32

    @property
    def complete(self) -> bool:
        return self.stored_samples == self.n_samples

    @property
    def duration_ticks(self) -> int:
        return self.duration_ms * TICKS_PER_MS


@lru_cache(maxsize=1)
def _raw() -> tuple[dict[str, Any], Any]:
    manifest = json.loads((VECTOR_DIR / "manifest.json").read_text())
    return manifest, np.load(VECTOR_DIR / "streams.npz")


@lru_cache(maxsize=1)
def _all() -> dict[str, Case]:
    manifest, data = _raw()
    out: dict[str, Case] = {}
    for m in manifest["cases"]:
        cid: str = m["id"]
        timeline = tuple(
            Controls(
                ms=s["ms"],
                pitch=s["pitch"],
                attack=s["attack"],
                decay=s["decay"],
                morph=s["morph"],
                fold=s["fold"],
                harmonic=s["harmonic"],
                spread=s["spread"],
                mode=s["mode"],
                gate=bool(s["gate"]),
            )
            for s in m["timeline"]
        )
        out[cid] = Case(
            id=cid,
            family=m["family"],
            description=m["description"],
            duration_ms=m["duration_ms"],
            timeline=timeline,
            n_samples=m["n_samples"],
            n_calls=m["n_calls"],
            trigger_samples=tuple(m["trigger_samples"]),
            stored_samples=m["stored_samples"],
            sha256_samples=m["sha256_samples"],
            sha256_periods=m["sha256_periods"],
            sha256_pairs=m["sha256_pairs"],
            samples=data[cid + ".sample"],
            periods=data[cid + ".period"],
            call_samples=data[cid + ".calls"],
        )
    return out


def drive_rule() -> str:
    """The text of the driving rule recorded in the manifest."""
    manifest, _ = _raw()
    return str(manifest["drive"]["rule"])


def cases(family: str | None = None) -> list[Case]:
    """All cases (or one family: mode, sweep, pitch, trigger, long, edge, sequence, fuzz), in manifest order."""
    return [c for c in _all().values() if family is None or c.family == family]


def case(case_id: str) -> Case:
    return _all()[case_id]


def families() -> list[str]:
    return sorted({c.family for c in _all().values()})


def controls_at(c: Case, k: int) -> Controls:
    """Controls of control call k: the last timeline segment with ms <= k."""
    cur = c.timeline[0]
    for s in c.timeline:
        if s.ms <= k:
            cur = s
        else:
            break
    return cur


def calls(c: Case) -> list[Controls]:
    """The controls of every control call, k = 0 .. n_calls - 1."""
    return [controls_at(c, k) for k in range(c.n_calls)]


def digests_of(samples: I32, periods: U32) -> tuple[str, str, str]:
    """(sha256_samples, sha256_periods, sha256_pairs) of whole streams."""
    s = np.asarray(samples, dtype="<i4")
    p = np.asarray(periods, dtype="<u4")
    pairs = np.empty(2 * len(s), dtype="<u4")
    pairs[0::2] = s.view("<u4")
    pairs[1::2] = p
    return (
        hashlib.sha256(s.tobytes()).hexdigest(),
        hashlib.sha256(p.tobytes()).hexdigest(),
        hashlib.sha256(pairs.tobytes()).hexdigest(),
    )


def digests_match(c: Case, samples: I32, periods: U32) -> bool:
    """True when whole (sample, period) streams equal the case's, by length and digests."""
    if len(samples) != c.n_samples or len(periods) != c.n_samples:
        return False
    return digests_of(samples, periods) == (
        c.sha256_samples,
        c.sha256_periods,
        c.sha256_pairs,
    )


def first_mismatch(c: Case, samples: I32, periods: U32) -> int | None:
    """Index of the first stored sample where (sample, period) differ, or None (compares the stored prefix, and a
    length shorter than it counts as a mismatch at len)."""
    n = c.stored_samples
    m = min(n, len(samples), len(periods))
    bad = np.nonzero(
        (np.asarray(samples[:m]) != c.samples[:m])
        | (np.asarray(periods[:m]) != c.periods[:m])
    )[0]
    if len(bad):
        return int(bad[0])
    return None if m == n else m
