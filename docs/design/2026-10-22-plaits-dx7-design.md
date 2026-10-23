# PLAITS DX7 models: design

Status: approved by the owner (2026-10-22), with HARMONICS showing patch names.

## Intent

Add Plaits' six-operator FM engine (its DX7 mode) to the PLAITS machine as four models. Three of them carry Plaits'
built-in patch banks; the fourth carries the owner's own favourite bank. There is no new UI: no file browser, no
selector, no user-uploadable banks. That idea was explored (research-dx7.md) and dropped by the owner in favour of
fixed banks.

## Decisions (owner)

- **Models:** `DX7-1`, `DX7-2` and `DX7-3` are models 18, 19 and 20, with upstream's three built-in banks in upstream
  order. `DX7-4` is a new model 24 holding the owner's bank,
  `~/src/spacejam.github.io/tv7/examples/tyler-favorites.syx`: a 4,104-byte DX7 32-voice bulk dump with a valid
  checksum, starting with `*PPG*Vol.1`. MODEL's range grows from 0..23 to 0..24, and the name formatter shows
  `DX7-1`..`DX7-4`.
- **No new UI.** The top-right SRC slot stays unused.
- **HARMONICS shows the patch name** on DX7 models (e.g. `*Synclav.1`), taken from the bank's 10-character voice
  names. On every other model, HARMONICS displays as today.
- **Note-off reaches DX7.** DX7 models get a real key-up, so the patches' operator and pitch envelopes reach their
  release stages as on a DX7. This is a documented DX7-only exception to the note-on-only trigger convention, like
  SWARM's continuous mode. The DT2 amp envelope still shapes the overall level on top.

## Behaviour (upstream unless stated)

- **Controls:** HARMONICS picks the patch (32 steps across the bank), TIMBRE is brightness (modulator level), MORPH
  scales the envelope times, all exactly as upstream's six-op engine maps them. TUNE, MIX and LEV are as for every
  model.
- **Triggers:** each note-on is `TRIGGER_RISING_EDGE` and starts a voice (upstream round-robins its two internal
  voices, so a new note doesn't cut the previous one's tail). Gate HIGH is held until the DT2 note-off, then LOW: the
  DX7 exception above. How the DSP learns of the note-off (the frame's release mask at 0x24, per research-dx7.md
  section 5) is settled during the port with one bounded emulator check, and documented.
- **Envelopes:** upstream registers the engine as already enveloped (no LPG), which matches the earlier owner decision
  (no Plaits LPG). The patch's own envelopes are part of the sound and stay.
- **Banks:** stored in the DSP image as tables, in upstream's packed 128-byte voice format. The `.syx` is converted at
  build time by a generator (header, checksum and 4,096-byte body checked; the build fails on a bad file). Unpacking to
  the engine's patch structure follows upstream's code exactly.

## Constraints

- Bit-exact against upstream's six-op engine on the host oracle, including the fourth bank: the oracle loads the
  owner's bank through upstream's own patch-loading path.
- The PORTING.md recipe and section 5: SIMD-friendly layout (6 operators); per-voice state in the `.PKBSS` voice array;
  every drift-catalogue rule.
- CPU: fuzz-first measurement and the 85 % one-track rule. The research estimate is about 10-20k instructions per block
  (unmeasured). A kernel step follows in the optimisation phase if the hot loop is above 40 %.
- Memory: the four banks are 16 KB of tables; per-voice state is estimated at about 2.5 KB. Both fit comfortably
  (section 7 had about 386 KB free at eff475a).

## Patch-name display

The MODEL name formatter (`plaits_model_fmt`, `core/params` formatter field) already proves custom display text works.
HARMONICS needs a formatter whose text depends on the same track's MODEL, but the research found that a formatter
receives only its own raw value. The builder must find how the formatter can read the current track's MODEL parameter
(the kit or track parameter storage the SRC page reads; static research first, then one bounded emulator check), and
map HARMONICS' value to the patch index exactly as the DSP does (upstream's 32-step quantisation), so the name shown is
the patch that plays. The 4 × 32 names (10 characters each, about 1.3 KB) live in the ColdFire expansion, generated
from the same bank tables as the DSP data, so they can't disagree. If reading MODEL from the formatter proves
infeasible, report it before shipping: do not show possibly wrong names.

**Owner decision (2026-10-23): exact patch boundaries.** Upstream picks the patch with a hysteresis quantiser, so near a
boundary the patch depends on the direction HARMONICS was turned, and a display that sees only the knob value can't
know it. DX7 models therefore use a memoryless quantiser: patch = a fixed function of HARMONICS, with the boundaries
at upstream's nominal thresholds. This is a deliberate, documented deviation from upstream. The DX7 oracle applies the
same quantiser, and the DX7 digests are re-recorded in a dedicated commit stating why. The display uses the identical
function (one generated table shared by the DSP and the ColdFire names), so the name always matches the patch.

## Left to the builder

The exact note-off signal source and its test; table layout; how DX7-4 shares code with models 18-20 (one engine,
bank pointer per model); the HARDWARE-TEST checklist (patch walk with HARMONICS, held notes releasing on note-off, fast
retriggers, the track rule); and the image.

## Out of scope

User-loadable banks and a bank or patch browser.
