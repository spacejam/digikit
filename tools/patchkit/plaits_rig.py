"""The isolated compiled PLAITS routine on the concrete SHARC+ core: core/dsp-inline linked at test addresses (firmware
addresses stood in by L2 test pages, as in tests/patchkit/test_inline_tone.py) and the PLAITS object of
machines/plaits-48k (mode 1) or -96k (mode 0) built as pk_build builds it, its data sections loaded at test
addresses. The machine state (the C voice array in L2) persists from block to block: each block is a fresh call of
`_di_entry` on the same memory. Shared by tests/support/plaits_rig.py (the tests) and tools/patchkit/pk_plaits_fuzz.py
(the worst-case CPU search)."""

from __future__ import annotations

import struct
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tools.patchkit import pk_build, pk_sharc, pk_toolchains

ROOT = pk_toolchains.ROOT
MODULES = ROOT / "modules"
HOOK_SRC = MODULES / "core" / "dsp-inline" / "src" / "dsp" / "dsp-inline.s"
TYPES = MODULES / "core" / "machines" / "types.toml"
VARIANT = {0: "machines/plaits-96k", 1: "machines/plaits-48k"}

# TAIL_BASE moved from 0xBA8000 (the module code, 91,621 words with DUST, outgrew the 90,112 words below it and the
# tail overwrote its end: HIHAT jumped into the tail); __init__ asserts that the code stays below it. O6 (the kernels,
# 106,496 words full) moved it again to 0xBC0000, RETURN_SW 0xBAF100 -> 0xBC3100.
CORE_BASE, PL_BASE, TAIL_BASE = 0xB90000, 0xB92000, 0xBC0000
RESUME, DECIMATE, AFTER = 0x1C4F18, 0x1C524A, 0x1C5264
RETURN_SW = 0xBC3100
CALLER_RET = (
    0x1C6B06  # after `CJUMP 0x1c4ecf` in the voice loop of FUN_1c642a (the image rig)
)
DISPLACED = bytes.fromhex("5499a801408c")
# the release flag array (core/dsp-inline `_fw_release_latch`) stands 0x100 past the trigger latch in its test page
TYPE_CACHE, FRAME, RECORDS, LATCH, OUT, TABLE, STACK = (
    0x20091000,
    0x20092000,
    0x20093000,
    0x20097000,
    0x20098000,
    0x2009A000,
    0x2009C000,
)
RELEASE = LATCH + 0x100
DATA_BASE = 0x200A0000
REC = 0x1D8
# sizeof(PlaitsVoice) (plaits.h; every member 4-byte, equal on the host: tests/support/plaits_host.py Port.voice_size)
VOICE_BYTES = 17124
TUNE_REST = 0x4000
VOICE_POISON = 0x5A5A5A5A  # a finite, large float: an engine that read its union before init would show it


def plaits_type() -> int:
    return int(
        tomllib.loads(TYPES.read_text())["dt2"]["alloc"]["machines/plaits:plaits"]
    )


def f32(w: int) -> float:
    return float(struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0])


def sharc_jump(target: int) -> bytes:
    return struct.pack("<HHH", 0x063E, (target >> 16) & 0xFF, target & 0xFFFF)


def variant_sources(mode: int) -> list[Path]:
    moddir = MODULES / VARIANT[mode]
    manifest = tomllib.loads((moddir / "module.toml").read_text())
    (obj,) = manifest["object"]
    return [moddir / s for s in obj["sources"]]


def tune_hw(t: int) -> int:
    """Frame halfword of track t's machine slot 0 (TUNE): byte 0xda + 0x60 t."""
    return (0xDA + 0x60 * t) // 2


def knob_of_engine(engine: int) -> int:
    """The MODEL knob position of an engine number (plaits.c plaits_knob_engine: knob 21 is DX7-4 = engine 24, knobs 22..24 are
    the engines 21..23)."""
    if engine == 24:
        return 21
    return engine + 1 if 21 <= engine <= 23 else engine


SLOT_HW = {"tune": 0, "model": 1, "harmonics": 2, "timbre": 6, "morph": 7, "mix": 8}


class PlaitsRig:
    def __init__(self, mode: int) -> None:
        from tools.sharc_dev import sharc_run as sr
        from tools.sharc_dev import sharc_trace as st
        from tools.sharc_dev.sharcldr import LoadedMemory

        self.mode = mode
        self.st = st
        self.core = pk_sharc.link(
            pk_sharc.assemble(HOOK_SRC.read_text()),
            CORE_BASE,
            {
                "_di_table": TABLE,
                "_fw_records": RECORDS,
                "_fw_type_cache": TYPE_CACHE,
                "_fw_frame": FRAME,
                "_fw_trig_latch": LATCH,
                "_fw_release_latch": RELEASE,
                "_fw_decimate": DECIMATE,
                "_fw_after_decimate": AFTER,
                "_fw_stock": TAIL_BASE,
            },
        )
        self.obj = pk_build.dsp_object(variant_sources(mode))
        at = DATA_BASE
        self.data_at: dict[str, int] = {}
        for sec, (blob, align, _bss) in self.obj.data.items():
            at = -(-at // max(align, 8)) * max(align, 8)
            self.data_at[sec] = at
            at += len(blob)
        externs = {
            name: self.data_at[sec] + off
            for name, (sec, off, _g) in self.obj.data_symbols.items()
        }
        for name in (
            "_di_frame_hw",
            "_di_trig",
            "_di_released",
            "_di_return_work",
            "_di_return_out",
        ):
            externs[name] = self.core.symbols[name]
        self.symbols = dict(externs)
        self.pl = pk_sharc.link(self.obj, PL_BASE, externs)
        assert len(self.obj.data["pkbss_plaits_voice_mem."][0]) == 16 * VOICE_BYTES, (
            "VOICE_BYTES"
        )
        assert PL_BASE + len(self.pl.code_boot) // 2 <= TAIL_BASE, (
            "module code overlaps the rig's tail block"
        )
        tail = DISPLACED + sharc_jump(RESUME)
        payload = b""
        blocks: list[dict[str, int | bool]] = []
        for base_sw, code in (
            (CORE_BASE, self.core.code_boot),
            (PL_BASE, self.pl.code_boot),
            (TAIL_BASE, tail),
        ):
            blocks.append(
                {
                    "target_address": pk_sharc.SW_ALIAS_BASE + 2 * base_sw,
                    "byte_count": len(code),
                    "fill": False,
                    "payload_offset": len(payload),
                    "payload_len": len(code),
                    "index": len(blocks),
                }
            )
            payload += code
        for sec, (blob, _align, bss) in self.obj.data.items():
            data = bytes(len(blob)) if bss else blob
            blocks.append(
                {
                    "target_address": self.data_at[sec],
                    "byte_count": len(data),
                    "fill": False,
                    "payload_offset": len(payload),
                    "payload_len": len(data),
                    "index": len(blocks),
                }
            )
            payload += data
        self.mem = LoadedMemory(payload, blocks)
        self.runner = sr.Runner(
            self.mem,
            self.core.symbols["_di_entry"],
            breakpoints=[RESUME, DECIMATE, AFTER, RETURN_SW],
            approx_recips=True,
        )
        self.type = plaits_type()
        for base, n in (
            (TYPE_CACHE, 8),
            (FRAME, 256),
            (RECORDS, 32 * REC // 4),
            (LATCH, 8),
            (RELEASE, 8),
            (OUT, 32 * 32),
            (TABLE, 34),
            (STACK - 0x200, 0x90),
        ):
            for k in range(n):
                self.poke(base + 4 * k, 0)
        self.poke(TABLE, self.type)
        self.poke(TABLE + 4, self.pl.symbols["_pl_render"])
        self.instructions: list[int] = []
        # kept by model_blocks: voice 0's latched model and the steps since its engine was last initialised
        self.latched: int | None = None
        self.since_init: list[tuple[int, int, int, dict[str, int]]] = []

    # ---- memory
    def poke(self, addr: int, word: int) -> None:
        self.st._dm_write(self.runner.state, addr, 4, self.st.Const(word & 0xFFFFFFFF))

    def peek(self, addr: int) -> int:
        x = self.st._dm_read(self.runner.state, addr, 4)
        assert x is not None, hex(addr)
        return x.value & 0xFFFFFFFF

    def words(self, addr: int, n: int) -> list[int]:
        return [self.peek(addr + 4 * k) for k in range(n)]

    def _set_half(self, addr: int, half: int, value: int) -> None:
        sh = 16 * half
        w = self.peek(addr)
        self.poke(addr, (w & ~(0xFFFF << sh)) | ((value & 0xFFFF) << sh))

    # ---- the block's inputs
    def set_type(self, track: int, t: int) -> None:
        self._set_half(TYPE_CACHE + 4 * (track // 2), track % 2, t)

    def set_hw(self, k: int, value: int) -> None:
        self._set_half(FRAME + 4 * (k // 2), k % 2, value)

    def controls(
        self,
        track: int,
        note: int,
        tune: int = TUNE_REST,
        velocity: int = 100,
        **slots: int,
    ) -> None:
        """The track's type = PLAITS, its note, velocity (0..127; the frame word is velocity << 8 at halfword
        0x1a + track, finding 06) and TUNE, and any of model/harmonics/timbre/morph/mix (row values)."""
        self.set_type(track, self.type)
        self.set_hw(1 + track, note << 8)
        self.set_hw(0x1A + track, velocity << 8)
        self.set_hw(tune_hw(track), tune)
        defaults = {
            "model": 0,
            "harmonics": 0x4000,
            "timbre": 0x4000,
            "morph": 0x4000,
            "mix": 0,
        }
        for name, value in {**defaults, **slots}.items():
            if (
                name == "model"
            ):  # callers name the engine; the row holds the MODEL knob position
                value = knob_of_engine(value >> 8) << 8 | (value & 0xFF)
            self.set_hw(tune_hw(track) + SLOT_HW[name], value)

    def trig(self, v: int, on: bool = True) -> None:
        a = LATCH + 4 * (v // 4)
        sh = 8 * (v % 4)
        w = self.peek(a)
        self.poke(a, (w & ~(0xFF << sh)) | ((1 if on else 0) << sh))

    def release(self, v: int, on: bool = True) -> None:
        """The voice's release flag (the DSP's byte at 0x24f0ec + v: 1 from the note-off, 0 from the note-on)."""
        a = RELEASE + 4 * (v // 4)
        sh = 8 * (v % 4)
        w = self.peek(a)
        self.poke(a, (w & ~(0xFF << sh)) | ((1 if on else 0) << sh))

    def active(self, v: int, on: bool = True) -> None:
        a = RECORDS + v * REC + 0x1B8
        w = self.peek(a)
        self.poke(a, (w & ~0xFF) | (1 if on else 0))

    def clear_voice(self, track: int) -> None:
        """Put track's voice state (plaits_voice_mem[track], PlaitsVoice, VOICE_BYTES each) as at load: reserved memory
        (.PKBSS, not zero: a poison pattern here) with the voice's first-use flag clear, so plaits_track zeroes its
        non-engine fields and the engine is initialised by the first note-on. Never triggered."""
        base = self.symbols["plaits_voice_mem."] + track * VOICE_BYTES
        for k in range(VOICE_BYTES // 4):
            self.poke(base + 4 * k, VOICE_POISON)
        self.poke(self.symbols["plaits_voice_ready."] + 4 * track, 0)

    # ---- one block
    def block(
        self,
        v: int,
        max_steps: int = 2_000_000,
        want: int | None = None,
        on_step: Callable[[Any], None] | None = None,
    ) -> dict:
        """One render block of voice v through the hook (as the renderer calls it at SW 0x1c4f15), run to the
        continuation `want` (default: the variant's return, AFTER for mode 1, DECIMATE for mode 0; RESUME is the
        stock path of a type without a row). `on_step(state)` sees the core state before every instruction (an
        interrupt could be taken at any of those points)."""
        rec, outp = RECORDS + v * REC, OUT + v * 0x80
        for off, val in {-4: 32, -6: rec, -7: outp, -5: 0xDEAD}.items():
            self.poke(STACK + 4 * off, val)
        for k in range(32):
            self.poke(outp + 4 * k, 0x7F7F7F7F)
        regs: dict[str | int, int | str] = {
            "R0": 64,
            "R4": rec,
            "I4": rec,
            "I6": STACK,
            "I7": STACK - 104,
            # the firmware's stack is a circular buffer (B6 = B7, L6 = L7 set by its startup, SW 0x1c0f64)
            "B6": STACK - 0x400,
            "B7": STACK - 0x400,
            "L6": 0x800,
            "L7": 0x800,
            "MODE1": 0x1011800,  # the firmware's MODE1 (startup, finding 06): CBUFEN (bit 24) on
            "M5": 0,
            "M6": 1,
            "M7": -1,
            "M13": 0,
            "M14": 1,
            "M15": -1,
        }
        r = self.runner.fresh_call(
            self.core.symbols["_di_entry"], regs=regs, return_address=RETURN_SW
        )
        if on_step is not None:
            step = r.step

            def stepped() -> None:
                on_step(r.state)
                step()

            r.step = stepped  # type: ignore[method-assign]
        res = r.run(max_steps)
        self.runner = r
        if want is None:
            want = AFTER if self.mode else DECIMATE
        assert res.halt.reason == "breakpoint", res.halt.to_json()
        assert res.final_pc_sw == want, hex(res.final_pc_sw)
        self.instructions.append(res.instructions)
        uregs = r.state.uregs
        regs_out = {
            n: getattr(uregs.get(self.st.UREG_CODES[n]), "value", None)
            for n in (
                "I6",
                "I7",
                "M5",
                "M6",
                "M7",
                "M13",
                "M14",
                "M15",
                "I8",
                "I13",
                "M0",
                "M8",
                "L6",
                "L7",
                "B6",
                "B7",
            )
        }
        work = self.words(rec + 4, 64)
        out = self.words(outp, 32)
        return {
            "final": res.final_pc_sw,
            "work": work,
            "out": out,
            "samples": [f32(w) for w in (out if self.mode else work)],
            "regs": regs_out,
            "instructions": res.instructions,
        }


C_STACK = 0x200F0000  # a C frame for direct calls: [C_STACK] = 0 (caller frame), [C_STACK - 4] = return address - 1


def ccall(
    rig: PlaitsRig, name: str, args: list[int], max_steps: int = 2_000_000
) -> int:
    """Call the compiled C function `name` (its symbol, e.g. `va_render.`) with integer/pointer/raw-float `args`
    (R4, R8, R12, then the stack) on the rig's memory; return R0."""
    rig.poke(C_STACK, 0)
    rig.poke(C_STACK - 4, RETURN_SW - 1)
    regs: dict[str | int, int | str] = {
        "I6": C_STACK,
        "M5": 0,
        "M6": 1,
        "M7": -1,
        "M13": 0,
        "M14": 1,
        "M15": -1,
    }
    for reg, a in zip(("R4", "R8", "R12"), args[:3], strict=False):
        regs[reg] = a & 0xFFFFFFFF
    # stack arguments: the 4th at [I6 + 4], the 5th at [I6 + 8], ...
    for k, a in enumerate(args[3:]):
        rig.poke(C_STACK + 4 * (k + 1), a)
    regs["I7"] = C_STACK - 8
    r = rig.runner.fresh_call(rig.pl.symbols[name], regs=regs, return_address=RETURN_SW)
    res = r.run(max_steps)
    rig.runner = r
    assert res.halt.reason == "breakpoint" and res.final_pc_sw == RETURN_SW, (
        res.halt.to_json()
    )
    v = r.state.uregs.get(rig.st.UREG_CODES["R0"])
    return int(getattr(v, "value", 0) or 0)
