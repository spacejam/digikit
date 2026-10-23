# ITERATOR machine: design

Status: approved by the owner (2026-10-23), with randomize-to-p-locks added.

## Intent

A second custom machine for the Digitakt II 1.17, side by side with PLAITS: a resonant percussion and tone voice with
six morphing partials, a wavefolder and a noise burst. It is a clean-room implementation from a written behavioural
specification of an existing Eurorack voice. Like PLAITS it follows the "reference quality" rule: the sound engine
reproduces the specification's arithmetic exactly, and the only adaptation is to the DT2's fixed 48 kHz output.

## Decisions (owner)

- **Name:** ITERATOR. No vendor or product names anywhere in the build: code, strings, symbols, docs shipped in the
  image, the UI.
- **Clean room:** implementer agents read only the specification files (the reference voice's `NOTES.md`,
  and the platform `FRONTEND.md`, kept outside this repo). A separate verifier may use the bit-exact reference C (kept outside this repo) and the firmware
  harness to produce reference audio and vectors. Implementers never read the reference C, the decompiles or the
  firmware.
- **One SRC page,** using the same one-page machine rule PLAITS uses.
- **MODE** merges the engine's 3-way mode switch and its 3-way range switch into one 9-position knob. RANGE is not just
  pitch: it also multiplies the overtone spacing (×4, ×16) and changes the noise burst's grain. Order and display names,
  exactly as given:
  `SKN-L, MTL-L, LIQ-L, SKN-M, MTL-M, LIQ-M, SKN-H, MTL-H, LIQ-H`. SKN, MTL and LIQ are the engine's three modes; L, M
  and H are range positions 0, 1 and 2. The engine's internal mode numbering (the spec's switch-order remap) is
  applied inside, so the names match the sounds.
- **Sample rate: faithful, then resampled.** The engine runs at its own pitch-locked variable rate (about
  24.4-48.8 kHz, the period computed per sample exactly as specified), and its (sample, duration) stream is converted
  to 48 kHz by a high-quality band-limited fractional resampler. That keeps the character of aliasing landing on
  harmonics.

## Coordinator's calls (owner may override)

- **Page layout:** TUNE, MODE, ATTACK, DECAY / SPREAD, FOLD, MORPH, HARMONIC. LEV is off the page; the AMP page volume
  covers it. All eight are p-lockable and LFO targets.
- **Knob ranges:** each knob covers the spec's 0..1 control range at full resolution. ATTACK keeps the spec's bipolar
  centre dead zone, so the centre is the "click" setting. TUNE behaves as on PLAITS. Pitch maps the DT2 note into the
  spec's 0..8-octave pitch input, with the integer and fraction split as specified. The exact note → pitch offset is
  chosen so that MODE ×-L at the default TUNE sounds the trig's note, and tested.
- **Triggers:** every DT2 note-on is a trigger (the engine is trigger-only, with no gate). The spec's 50 ms retrigger
  rule, which flips output polarity on fast retriggers, is kept exactly. P-locks on a trig apply before its trigger, in
  the same update, as the spec asks.
- **Envelopes:** the engine's own attack and decay envelopes are the sound (two of its six knobs), so they stay. The DT2
  amp envelope, filter, FX and LFOs act on top, as for any machine. Velocity is left to the DT2's normal velocity
  handling (the engine has no accent input).
- **SRC+YES** randomizes MODE and the six knobs, not TUNE, matching PLAITS.
- **Randomize into p-locks (owner requirement):** with one or more trigs held, SRC+YES writes the randomized values as
  **p-locks on the held trig(s)**, exactly as stock machines do for their randomizable parameters, instead of changing
  the track's base values. Without a held trig it changes the base values. This is required for intuitive use. **Deferred by the owner (2026-10-23):** stock machines don't do this either (no stock gesture randomizes into p-locks), so it is a new feature for later. The
  same requirement is verified, and fixed if needed, for PLAITS, whose randomize already goes through the stock
  parameter write path.
- **Level:** the engine's output (±2.0 in 8.24 at full scale) is normalised to the same loudness reference PLAITS uses.

## Constraints

- **Bit-exact core:** the engine's 8.24 fixed-point arithmetic (truncating `(a*b) >> 24` products, the LCG noise,
  phase wrap) reproduced exactly on the SHARC's integer ALU. It is tested against the verifier's reference vectors as
  (sample, period) streams before resampling.
- **Resampler:** specified and tested on its own (passband flatness, stopband rejection, group delay); its output is
  pinned by digests.
- **Same infrastructure as PLAITS:** `core/dsp-inline` hook, `core/machines` / `core/params` rows, formatter for MODE
  names, `.PKBSS` voice memory, encoding gate, full-state kernel tests, fuzz-first CPU, the 85 % one-track rule, SIMD-
  friendly layout (six partials pair naturally).
- **The hardware rules learned on PLAITS** (no same-DAG stores, no SIMD-mode transfers until a probe passes, restore
  every mode and DAG register).

## Left to the builder

Resampler design (filter length, phase count, interpolation) within a CPU budget; per-voice state layout; table
generation from the spec; the HARDWARE-TEST checklist; and the image.

## Order of work

After PLAITS is finished: the UI batch, DX7 names, the final optimisation round and integration. The design can be
approved now; the build starts when PLAITS is done.
