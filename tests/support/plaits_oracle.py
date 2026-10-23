"""The continuous host oracle (tests/plaits/oracle/run_oracle.cc) built and driven through ctypes: upstream Plaits'
engines of the ported models (0..7, 12 RINGS, 21 TERRAIN) with a uniform select/render interface, and upstream's
output-stage limiters (run_oracle_post*). Used by tests/plaits/test_continuous_host.py (the wrapper rendered
continuously), test_modal_host.py and test_terrain_host.py (the engines on their own)."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from support.plaits_host import (
    DC_R,
    EURORACK,
    FP,
    GAIN,
    INV_FULL,
    build_oracle,
    owner_bank,
    stmlib_root,
    upstream_gains,
)

ORACLE = Path(__file__).resolve().parents[1] / "plaits" / "oracle"
Arr = Any
# a step: (note, harmonics, timbre, morph, trigger, accent) for one sub-block
Step = tuple[float, float, float, float, int, float]
# a block of the wrapper's controls (tests/support/plaits_streams.py): row values, the trigger latch, velocity
Block = dict[str, int]
INV_256 = np.float32(0.00390625)
RISING_HIGH, HIGH = 1 | 4, 4
SUB = 16
# the six-op models: 18, 19, 20 (upstream's banks) and 24 (the owner's bank)
DX7_MODELS = (18, 19, 20, 24)


class Oracle:
    def __init__(self, so: Path) -> None:
        self.lib = ctypes.CDLL(str(so))
        self.lib.run_oracle_select.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.run_oracle_select_bank.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        self.owner_bank = owner_bank()
        self.lib.run_oracle_render.argtypes = [
            ctypes.c_void_p,
            *[ctypes.c_float] * 4,
            ctypes.c_int,
            ctypes.c_float,
            FP,
            FP,
            ctypes.c_int,
        ]
        self.size = self.lib.run_oracle_size()
        self.post_size = self.lib.run_oracle_post_size()
        for name in ("run_oracle_post_init", "run_oracle_post_reset"):
            getattr(self.lib, name).argtypes = [ctypes.c_void_p]
        self.lib.run_oracle_post.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            FP,
            FP,
            ctypes.c_int,
        ]

    def select(self, st: Any, model: int, bank: bytes | None = None) -> None:
        """Plaits' Voice at an engine change: Init, LoadUserData (the DX7 models: upstream's own bank 18..20, the owner's
        bank for model 24, or `bank`), Reset."""
        if model == 24 and bank is None:
            bank = self.owner_bank
        if bank is None:
            self.lib.run_oracle_select(st, model)
        else:
            self.lib.run_oracle_select_bank(st, model, bank)

    def engine(self, model: int, steps: list[Step], size: int = 16) -> tuple[Arr, Arr]:
        """Upstream's engine of `model` selected as Plaits' Voice does (Init, LoadUserData(NULL), Reset), then one
        render of `size` samples per step: OUT and AUX."""
        st = ctypes.create_string_buffer(self.size)
        self.select(st, model)
        out = np.zeros(len(steps) * size, dtype=np.float32)
        aux = np.zeros_like(out)
        for k, (note, h, t, m, trig, accent) in enumerate(steps):
            self.lib.run_oracle_render(
                st,
                note,
                h,
                t,
                m,
                trig,
                accent,
                out[k * size :].ctypes.data_as(FP),
                aux[k * size :].ctypes.data_as(FP),
                size,
            )
        return out, aux


# Upstream's WaveTerrainEngine reads outside its tables when its path leaves the unit square (undefined behaviour:
# below lut_sine for a negative Sine phase, before and past wav_integrated_waves). The port guards exactly those inputs
# (engines/terrain.c); the oracle applies the same guard to upstream's own source, only there, and is otherwise
# upstream unchanged. (pattern, replacement), each pattern exactly once.
TERRAIN_GUARD = [
    (
        "const float sample = (y + 1.0f) * 0.5f * float(table_size);",
        "float sample = (y + 1.0f) * 0.5f * float(table_size);\n"
        "  while (sample < 0.0f) sample += 128.0f;\n"
        "  while (sample >= 130.0f) sample -= 128.0f;",
    ),
    (
        "const float wt = (x + 1.0f) * 0.5f * float(num_waves - 1);",
        "float wt = (x + 1.0f) * 0.5f * float(num_waves - 1);\n"
        "  if (wt < 0.0f) wt = 0.0f;\n"
        "  if (wt > 63.0f) wt = 63.0f;",
    ),
    (
        "xy[1] = InterpolateIntegratedWave(waves, sample_integral, sample_fractional);",
        "xy[1] = wt_integral >= 63 ? xy[0] : InterpolateIntegratedWave(waves, sample_integral, sample_fractional);",
    ),
    (
        '#include "plaits/dsp/oscillator/wavetable_oscillator.h"',
        '#include "plaits/dsp/oscillator/wavetable_oscillator.h"\n'
        "namespace plaits {\n"
        "static inline float GuardedSine(float phase) {\n"
        "  if (phase < 0.0f) {\n"
        "    phase = phase - static_cast<float>(static_cast<int32_t>(phase)) + 1.0f;\n"
        "    if (phase >= 1.0f) phase -= 1.0f;\n"
        "  }\n"
        "  return Sine(phase);\n"
        "}\n"
        "}  // namespace plaits",
    ),
]


def memoryless_six_op(d: Path) -> Path:
    """Upstream's six_op_engine.cc with the hysteresis of the patch quantiser 0 (the owner's decision, 2026-10-23: the DX7
    patch is a fixed function of HARMONICS so the display can show its name; PORTING drift 28). Fails closed if upstream's
    line is not exactly the expected one."""
    text = (EURORACK / "plaits" / "dsp" / "engine2" / "six_op_engine.cc").read_text()
    pat = "patch_index_quantizer_.Init(32, 0.005f, false);"
    assert text.count(pat) == 1, pat
    out = d / "six_op_engine_memoryless.cc"
    out.write_text(text.replace(pat, "patch_index_quantizer_.Init(32, 0.0f, false);"))
    return out


def guarded_wave_terrain(d: Path) -> Path:
    """Upstream's wave_terrain_engine.cc with TERRAIN_GUARD applied and every Sine( call made GuardedSine(."""
    import re

    text = (
        EURORACK / "plaits" / "dsp" / "engine2" / "wave_terrain_engine.cc"
    ).read_text()
    for pat, rep in TERRAIN_GUARD[:3]:
        assert text.count(pat) == 1, pat
        text = text.replace(pat, rep)
    text, n = re.subn(r"(?<![A-Za-z_])Sine\(", "GuardedSine(", text)
    assert n == 9, n
    pat, rep = TERRAIN_GUARD[3]
    assert text.count(pat) == 1, pat
    text = text.replace(pat, rep)
    out = d / "wave_terrain_engine_guarded.cc"
    out.write_text(text)
    return out


SQRT_DEF = (
    "#include <cstdint>\n"
    "namespace plaits {\n"
    "static inline float PlOracleSqrt(float x) {\n"
    "  if (!(x > 0.0f)) return 0.0f;\n"
    "  union { int32_t i; float f; } u;\n"
    "  u.f = x;\n"
    "  u.i = (u.i >> 1) + 0x1fbd1df5;\n"
    "  float y = u.f;\n"
    "  y = 0.5f * (y + x / y);\n"
    "  y = 0.5f * (y + x / y);\n"
    "  y = 0.5f * (y + x / y);\n"
    "  y = 0.5f * (y + x / y);\n"
    "  return y;\n"
    "}\n"
    "}  // namespace plaits\n"
)


def sqrt_override(d: Path) -> Path:
    """An include directory holding upstream's synthetic_snare_drum.h with its two stmlib::Sqrt calls (the hardware
    vsqrt, a correctly rounded square root) replaced by the port's Newton iteration (lib/drums.h pl_sqrt): the port has
    no square root instruction. Fails closed when upstream's text changes."""
    src = (EURORACK / "plaits" / "dsp" / "drums" / "synthetic_snare_drum.h").read_text()
    assert src.count("stmlib::Sqrt(") == 2
    head = '#include "plaits/dsp/dsp.h"'
    assert src.count(head) == 1
    text = src.replace("stmlib::Sqrt(", "PlOracleSqrt(").replace(
        head, head + "\n" + SQRT_DEF
    )
    out = d / "sqrt-ov" / "plaits" / "dsp" / "drums"
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_snare_drum.h").write_text(text)
    return d / "sqrt-ov"


RSQRT_CALL = "1.0f / Sqrt((0.5f + q) * 40.0f * f0)"
RSQRT_DEF = (
    "namespace plaits {\n"
    "static inline float PlOracleRsqrt(float x) {\n"
    "  if (!(x > 0.0f)) return 0.0f;\n"
    "  float h = 0.5f * x;\n"
    "  union { int32_t i; float f; } u;\n"
    "  u.f = x;\n"
    "  u.i = 0x5f3759df - (u.i >> 1);\n"
    "  float y = u.f;\n"
    "  y = y * (1.5f - h * y * y);\n"
    "  y = y * (1.5f - h * y * y);\n"
    "  y = y * (1.5f - h * y * y);\n"
    "  return y;\n"
    "}\n"
    "}  // namespace plaits\n"
)


def rsqrt_noise_engine(d: Path) -> Path:
    """Upstream's noise_engine.cc with its one per-sample `1.0f / Sqrt(x)` replaced by the port's Newton reciprocal
    square root (engines/noise.c pl_rsqrt): the port has no square root instruction. Fails closed when upstream's text
    changes."""
    text = (EURORACK / "plaits" / "dsp" / "engine" / "noise_engine.cc").read_text()
    assert text.count(RSQRT_CALL) == 1
    head = "namespace plaits {\n\nusing namespace std;"
    assert text.count(head) == 1
    text = text.replace(RSQRT_CALL, "PlOracleRsqrt((0.5f + q) * 40.0f * f0)")
    text = text.replace(head, "#include <cstdint>\n" + RSQRT_DEF + "\n" + head)
    out = d / "noise_engine_rsqrt.cc"
    out.write_text(text)
    return out


DUST_RSQRT_CALL = "0.5f / stmlib::Sqrt(q * f * stmlib::Sqrt(density))"
DUST_RSQRT_NEW = "0.5f * PlOracleRsqrt(q * f * (density * PlOracleRsqrt(density)))"


def dust_override(d: Path) -> Path:
    """An include directory holding upstream's particle.h with its two stmlib::Sqrt calls (the hardware vsqrt) replaced
    by the port's Newton reciprocal square root (engines/noise.c pl_rsqrt, as engines/dust.c uses it: 0.5 * rsqrt(q * f *
    (density * rsqrt(density)))). Fails closed when upstream's text changes."""
    src = (EURORACK / "plaits" / "dsp" / "noise" / "particle.h").read_text()
    assert src.count(DUST_RSQRT_CALL) == 1
    assert src.count("Sqrt(") == 2
    head = "namespace plaits {"
    assert src.count(head) == 1
    text = src.replace(DUST_RSQRT_CALL, DUST_RSQRT_NEW).replace(
        head, "#include <cstdint>\n" + RSQRT_DEF + "\n" + head
    )
    out = d / "dust-ov" / "plaits" / "dsp" / "noise"
    out.mkdir(parents=True, exist_ok=True)
    (out / "particle.h").write_text(text)
    # the FxEngine's 12-bit Compress calls stmlib's Clip16, which is an ARM `ssat` unless TEST is defined (TEST would
    # change other upstream code): the override takes the portable branch of Clip16 only (its first `#ifdef TEST`)
    dsp = (stmlib_root() / "stmlib" / "dsp" / "dsp.h").read_text()  # type: ignore[operator]
    assert dsp.count("#ifdef TEST\n  inline int32_t Clip16(int32_t x) {") == 1
    stm = d / "dust-ov" / "stmlib" / "dsp"
    stm.mkdir(parents=True, exist_ok=True)
    (stm / "dsp.h").write_text(
        dsp.replace(
            "#ifdef TEST\n  inline int32_t Clip16(int32_t x) {",
            "#if 1\n  inline int32_t Clip16(int32_t x) {",
        )
    )
    return d / "dust-ov"


def build(d: Path) -> Oracle:
    plaits = EURORACK / "plaits"
    engine = plaits / "dsp" / "engine"
    sources = [
        ORACLE / "run_oracle.cc",
        engine / "virtual_analog_engine.cc",
        engine / "waveshaping_engine.cc",
        engine / "fm_engine.cc",
        engine / "grain_engine.cc",
        engine / "additive_engine.cc",
        engine / "wavetable_engine.cc",
        engine / "chord_engine.cc",
        engine / "speech_engine.cc",
        engine / "modal_engine.cc",
        rsqrt_noise_engine(d),
        engine / "bass_drum_engine.cc",
        engine / "snare_drum_engine.cc",
        engine / "hi_hat_engine.cc",
        plaits / "dsp" / "engine2" / "phase_distortion_engine.cc",
        plaits / "dsp" / "engine2" / "virtual_analog_vcf_engine.cc",
        plaits / "dsp" / "engine2" / "chiptune_engine.cc",
        plaits / "dsp" / "engine2" / "string_machine_engine.cc",
        memoryless_six_op(d),
        plaits / "dsp" / "fm" / "algorithms.cc",
        plaits / "dsp" / "fm" / "dx_units.cc",
        engine / "particle_engine.cc",
        engine / "swarm_engine.cc",
        engine / "string_engine.cc",
        plaits / "dsp" / "physical_modelling" / "string_voice.cc",
        plaits / "dsp" / "physical_modelling" / "string.cc",
        guarded_wave_terrain(d),
        plaits / "dsp" / "chords" / "chord_bank.cc",
        plaits / "dsp" / "physical_modelling" / "modal_voice.cc",
        plaits / "dsp" / "physical_modelling" / "resonator.cc",
        plaits / "resources.cc",
        *(
            plaits / "dsp" / "speech" / f"{n}.cc"
            for n in (
                "lpc_speech_synth_words",
                "naive_speech_synth",
                "sam_speech_synth",
                "lpc_speech_synth",
                "lpc_speech_synth_controller",
                "lpc_speech_synth_phonemes",
            )
        ),
    ]
    stm = stmlib_root()
    if stm is not None:
        sources.append(stm / "stmlib" / "utils" / "random.cc")
    return Oracle(
        build_oracle(d, sources, "run_oracle", (sqrt_override(d), dust_override(d)))
    )


def port_engine(
    port: Any, name: str, steps: list[Step], size: int = 16
) -> tuple[Arr, Arr]:
    """The port's engine `name` from <name>_init, one render per step (Params with trigger and accent)."""
    return port.engine(name, [tuple(s) for s in steps], size)


def same(a: Arr, b: Arr) -> bool:
    return bool(
        np.array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))
    )


# ---- the wrapper rendered from upstream (tests/plaits/test_continuous_host.py, the digests)


def _unit(word: int) -> np.float32:
    return np.float32(
        min(
            max(np.float32(np.float32(word) * INV_FULL), np.float32(0.0)),
            np.float32(1.0),
        )
    )


def reference(
    oracle: Oracle,
    blocks: list[Block],
    gains: list[tuple[np.float32, np.float32]] | None = None,
) -> Arr:
    """Upstream's engine, selected at the first note-on and at each note-on that changes the model, rendered
    continuously with the accent of the last note-on's velocity; upstream's output stage (gains from voice.cc, the
    limiters); then OUT + MIX (AUX - OUT), the DC blocker (cleared at each selection) and the gain, as plaits.c."""
    st = ctypes.create_string_buffer(oracle.size)
    post = ctypes.create_string_buffer(oracle.post_size)
    gains = gains or upstream_gains()
    out16 = np.zeros(SUB, dtype=np.float32)
    aux16 = np.zeros(SUB, dtype=np.float32)
    latched = -1
    accent = np.float32(0.0)
    x1 = y1 = np.float32(0.0)
    res = []
    for s in blocks:
        if s["trig"]:
            accent = _unit(s.get("velocity", 100) << 8)
        if s["trig"] and s["model"] != latched:
            if latched < 0:
                oracle.lib.run_oracle_post_init(post)
            else:
                oracle.lib.run_oracle_post_reset(post)
            latched = s["model"]
            oracle.select(st, latched)
            x1 = y1 = np.float32(0.0)
        assert latched >= 0
        note = np.float32(np.float32(s["note"] << 8) * INV_256) + np.float32(
            np.float32(np.float32(s["tune"]) * INV_256) - np.float32(64.0)
        )
        h, t, m, mix = (_unit(s[k]) for k in ("harmonics", "timbre", "morph", "mix"))
        blk = np.zeros(2 * SUB, dtype=np.float32)
        for sub in range(2):
            trig = RISING_HIGH if (s["trig"] and sub == 0) else HIGH
            if latched == 8:
                trig = 2  # SWARM: TRIGGER_UNPATCHED on every block (plaits.c TRIG_UNPATCHED, the owner's decision)
            elif latched in DX7_MODELS and not (s["trig"] and sub == 0):
                # the DX7 exception (plaits.c): the gate is LOW from the Digitakt's note-off to the next note-on
                trig = 0 if s.get("released", 0) else HIGH
            oracle.lib.run_oracle_render(
                st,
                float(note),
                float(h),
                float(t),
                float(m),
                trig,
                float(accent),
                out16.ctypes.data_as(FP),
                aux16.ctypes.data_as(FP),
                SUB,
            )
            go, ga = gains[latched]
            oracle.lib.run_oracle_post(
                post,
                float(go),
                float(ga),
                out16.ctypes.data_as(FP),
                aux16.ctypes.data_as(FP),
                SUB,
            )
            blk[sub * SUB : (sub + 1) * SUB] = out16 + mix * (aux16 - out16)
        for i in range(2 * SUB):
            x = blk[i]
            y1 = np.float32(np.float32(x - x1) + np.float32(DC_R * y1))
            x1 = x
            blk[i] = np.float32(y1 * GAIN)
        res.append(blk)
    return np.concatenate(res)


def up2(port: Any, x48: Arr, blocks: list[Block]) -> Arr:
    """The 96 kHz output from the 48 kHz one: hb_up2 per block, the interpolator cleared at each engine selection."""
    hb = ctypes.create_string_buffer(4096)
    out = []
    latched = -1
    for k, s in enumerate(blocks):
        if s["trig"] and s["model"] != latched:
            latched = s["model"]
            port.lib.hb_init(hb)
        src = np.ascontiguousarray(x48[32 * k : 32 * (k + 1)], dtype=np.float32)
        dst = np.zeros(64, dtype=np.float32)
        port.lib.hb_up2(hb, src.ctypes.data_as(FP), dst.ctypes.data_as(FP), 32)
        out.append(dst)
    return np.concatenate(out)
