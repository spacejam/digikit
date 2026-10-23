"""machines/plaits on the concrete SHARC+ core. The rig itself lives in tools/patchkit/plaits_rig.py (the fuzz tool
shares it) and is re-exported here; ImageRig (a patched image's own DSP code) and the model-switch helpers stay here.
Used by tests/patchkit/test_plaits_module.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.patchkit.plaits_rig import (  # noqa: F401  (re-exported for the tests)
    AFTER,
    CALLER_RET,
    DECIMATE,
    RESUME,
    PlaitsRig,
    f32,
    plaits_type,
)


class ImageRig:
    """A patched 1.17 image's own DSP code on the concrete core (tests/support/dsp_inline_image.py), the memory kept
    from block to block: each block calls the stock renderer (SW 0x1c4ecf) for one voice the way the voice loop of
    FUN_1c642a does and runs to the caller's return point."""

    def __init__(self, syx: Path, plan: dict) -> None:
        from support.dsp_inline_image import IMG_STACK, image_memory
        from tools.sharc_dev import sharc_run as sr
        from tools.sharc_dev import sharc_trace as st

        self.st = st
        self.stack = IMG_STACK
        self.symbols = {k.split(":", 1)[1]: v for k, v in plan["symbols"].items()}
        self.runner = sr.Runner(
            image_memory(syx), 0x1C4ECF, breakpoints=[CALLER_RET], approx_recips=True
        )
        self.runner.state.explicit_memory_model = True

    def poke(self, addr: int, word: int) -> None:
        self.st._dm_write(self.runner.state, addr, 4, self.st.Const(word & 0xFFFFFFFF))

    def poke_byte(self, addr: int, byte: int) -> None:
        self.st._dm_write(self.runner.state, addr, 1, self.st.Const(byte & 0xFF))

    def peek(self, addr: int) -> int:
        x = self.st._dm_read(self.runner.state, addr, 4)
        return 0 if x is None else x.value & 0xFFFFFFFF

    def words(self, addr: int, n: int) -> list[int]:
        return [self.peek(addr + 4 * k) for k in range(n)]

    def set_half(self, addr: int, half: int, value: int) -> None:
        sh = 16 * half
        self.poke(addr, (self.peek(addr) & ~(0xFFFF << sh)) | ((value & 0xFFFF) << sh))

    def block(self, v: int) -> dict:
        from support.dsp_inline_image import CALLEE_SAVED, IMG_OUT, IMG_RECORDS, REC

        rec, outp = IMG_RECORDS + v * REC, IMG_OUT + v * 0x80
        regs: dict[str | int, int | str] = {
            "R4": rec,
            "R8": outp,
            "R12": 32,
            "R14": 32 - v,
            "M5": 0,
            "M6": 1,
            "M7": -1,
            "M13": 0,
            "M14": 1,
            "M15": -1,
            "I6": self.stack,
            "I7": self.stack - 8,
            "B6": self.stack - 0x400,
            "B7": self.stack - 0x400,
            "L6": 0x800,
            "L7": 0x800,
            "MODE1": 0x1011800,
        }
        regs.update(CALLEE_SAVED)
        self.poke(self.stack - 4, CALLER_RET - 1)
        self.poke(self.stack, self.stack + 0x10000)
        for k in range(32):
            self.poke(outp + 4 * k, 0x7F7F7F7F)
        r = self.runner.fresh_call(0x1C4ECF, regs=regs, return_address=CALLER_RET)
        r.state.explicit_memory_model = True
        res = r.run(3_000_000)
        self.runner = r
        uregs = r.state.uregs
        return {
            "halt": res.halt.to_json(),
            "final_pc_sw": res.final_pc_sw,
            "instructions": res.instructions,
            "out": self.words(outp, 32),
            "samples": [f32(w) for w in self.words(outp, 32)],
            "work": self.words(rec + 4, 64),
            "active": self.peek(rec + 0x1B8) & 0xFF,
            "regs": {
                n: getattr(uregs.get(self.st.UREG_CODES[n]), "value", None)
                for n in (*CALLEE_SAVED, "I6", "I7")
            },
        }


def model_blocks(
    rig: PlaitsRig, steps: list[tuple[int, int]], note: int = 57, **slots: int
) -> list[Any]:
    """Voice 0 (track 0's even voice) of `rig`, one block per (model, trig) step: MODEL set to the model's row value
    (model << 8) and the trigger latch to trig; the other rows from `slots` (defaults as `controls`). Keeps
    `rig.since_init`, the steps played since the voice's engine was last initialised (its first note-on, or a note-on
    that changed the model: plaits.c note_on), which determine its state."""
    import numpy as np

    out = []
    for model, trig in steps:
        if trig and rig.latched != model:
            rig.latched = model
            rig.since_init = []
        rig.since_init.append((model, trig, note, dict(slots)))
        rig.controls(0, note, model=model << 8, **slots)
        rig.trig(0, bool(trig))
        out.append(np.array(rig.block(0)["samples"], dtype=np.float32))
    return out


def assert_model_switch(
    rig: PlaitsRig, a: int, b: int, note: int = 57, **slots: int
) -> list[tuple[int, int]]:
    """A model change on one voice of `rig` (whatever it played before, through `model_blocks`): a note-on of model a
    and 4 blocks; MODEL turned to b without a note-on (2 blocks: a runs on, bit for bit as a fresh rig's voice that
    replays the steps since the voice's last initialisation and then plays a without the change: a note-on of the
    model the voice already runs leaves its engine running); a note-on of b and 5 blocks (bit for bit a fresh rig's
    voice of b: nothing of the previous engine's state survives in the shared union). The new model must sound.
    Returns the (model, trig) steps it played, for a comparison with the host. Used by
    tests/patchkit/test_plaits_module.py for every enabled model."""
    import numpy as np

    rig.active(0)
    history = list(rig.since_init)
    steps = [(a, int(k == 0)) for k in range(4)] + [(b, 0)] * 2
    steps += [(b, int(k == 0)) for k in range(5)]
    got = model_blocks(rig, steps, note, **slots)
    ref = PlaitsRig(rig.mode)
    ref.active(0)
    for m, t, n, sl in history:
        model_blocks(ref, [(m, t)], n, **sl)
    want_a = model_blocks(ref, [(a, int(k == 0)) for k in range(6)], note, **slots)
    assert all(np.array_equal(x, y) for x, y in zip(got[:6], want_a, strict=True)), (
        f"model {a} did not run on through MODEL {b} without a note-on"
    )
    fresh = PlaitsRig(rig.mode)
    fresh.active(0)
    want_b = model_blocks(fresh, [(b, int(k == 0)) for k in range(5)], note, **slots)
    assert all(np.array_equal(x, y) for x, y in zip(got[6:], want_b, strict=True)), (
        f"model {b} after model {a} differs from a fresh voice of model {b}"
    )
    assert any(x.any() for x in want_b), f"model {b} is silent"
    return steps
