"""pk_proof_plaits.PlaitsProof.set_model (the proof's MODEL walk) against a simulated encoder: it converges from any start to
any model for single-detent steps that sometimes drop, and for bursts of unknown step size (1 to 4 times the detents sent);
a model up to 7 from 0 never uses a burst (the old single-detent walk, so models 3, 4 and 5 behave as before); after an
overshoot it goes back to single detents."""

from __future__ import annotations

from typing import Any

import pytest

from tools.patchkit import pk_proof_plaits as P


class FakeSession:
    def __init__(self, start: int, gain: int, drop_every: int) -> None:
        self.v = start << 8
        self.gain = gain
        self.drop_every = drop_every
        self.bursts = 0
        self.singles = 0
        self.n = 0

    def mirror(self, idx: int, track: int = 0) -> int:
        return self.v

    def turn_burst(
        self, enc: str, detents: int, events: int, gap_ms: float = 2
    ) -> None:
        if events > 1:
            self.bursts += 1
            self.v += self.gain * detents * events * 0x100
        else:
            self.singles += 1
            self.n += 1
            if self.drop_every and self.n % self.drop_every == 0:
                return  # a dropped turn
            self.v += detents * 0x100
        self.v = max(0, min(0x1800, self.v))


def walk(start: int, n: int, gain: int, drop_every: int) -> FakeSession:
    s = FakeSession(start, gain, drop_every)
    got = P.PlaitsProof.set_model(None, s, n)  # type: ignore[arg-type]
    assert got == n << 8, (start, n, gain, drop_every, hex(got))
    return s


@pytest.mark.parametrize("gain", [1, 2, 3, 4])
@pytest.mark.parametrize("drop_every", [0, 2, 7])
def test_converges_from_any_start_to_any_model(gain: int, drop_every: int) -> None:
    for start in (0, 3, 9, 16, 24):
        for n in range(25):
            walk(start, n, gain, drop_every)


def test_near_models_never_burst() -> None:
    cases: list[Any] = [(n, g) for n in range(8) for g in (1, 4)]
    for n, gain in cases:
        assert walk(0, n, gain, 0).bursts == 0


def test_cpu_tool_set_model_reaches_every_model() -> None:
    """tools.patchkit.pk_plaits_cpu.set_model (the CPU tool's MODEL walk) uses the proof's walk: every model from 0."""
    from tools.patchkit import pk_plaits_cpu as C

    class Tapper(FakeSession):
        def do_tap(self, name: str) -> None:
            pass

        def run_ms(self, ms: float) -> None:
            pass

    for n in range(25):
        for gain in (1, 3):
            s = Tapper(0, gain, 0)
            C.set_model(s, n, 0)
            assert s.v == n << 8
