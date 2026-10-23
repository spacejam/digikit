# Task: PLAITS fuzz worst-case CPU search

Status: done. Commits: 6ded02b (rig moved to tools/patchkit/plaits_rig.py, tests re-export), 2f9d05d (`pk_plaits_cpu --fuzz/--confirm`,
tools/patchkit/pk_plaits_fuzz.py, tests/patchkit/test_plaits_fuzz.py, TYPED), 7111dc3 (PORTING.md CPU step).

## Tool
`uv run python -m tools.patchkit.pk_plaits_cpu --fuzz --model N [--seed S] [--budget-seconds T] [--jobs J] [--sequences K] [--confirm]`.
Compiled 48k object on the concrete core; sequences of 6..40 blocks (knobs incl. extremes and notable values, notes with
jumps, velocity, TUNE, MIX, note-on spacing 1..48 blocks, start on another model then switch into N by note-on or by
turning MODEL first; plaits.c latch/init/strike conventions: RISING_EDGE only in the note-on block). Only blocks with N
latched count (the switch's init block included). 35 % of sequences mutate the worker's best. JSON per (model, seed) in
out/runs/plaits-cpu/fuzz/ (not committed). `--confirm` carries HARM/TIMB/MORPH and the note (via TUNE; emulator plays note
60); velocity and MIX are not carried; the baseline guard (no_signal) is unchanged.

## Rate
About 100 blocks/min per process (0.65 s per block, ~100k interpreted instructions/s, CPython; no PyPy installed; the
interpreter is pure Python and changing it would change CORE_FINGERPRINT). 3 workers: 150..550 blocks/min (blocks are
shorter for light models), 200 s per model incl. 8 s rig build per worker. The "thousands per minute" aim is NOT met.

## Results (seed 1, 180 s budget, --jobs 3, mode 1; recorded = finding 06 / test bound comments)
| model | recorded worst | fuzz worst | delta | needs confirm | blocks/min |
|---|---|---|---|---|---|
| 0 | 76,249 | 80,461 | +5.52 % | YES | 244 |
| 1 | 42,307 | 46,374 | +9.61 % | YES | 345 |
| 2 | 57,442 | 61,498 | +7.06 % | YES | 273 |
| 3 | 80,917 | 89,835 | +11.02 % | YES | 180 |
| 4 | 88,288 | 92,338 | +4.59 % | YES | 172 |
| 5 | 41,168 | 45,210 | +9.82 % | YES | 368 |
| 6 | 55,296 | 56,723 | +2.58 % | YES | 365 |
| 7 | 34,083 | 39,530 | +15.98 % | YES | 546 |
| 12 | 124,173 | 125,766 | +1.28 % | no | 152 |
| 13 | 67,683 | 67,911 | +0.34 % | no | 237 |
| 14 | 73,269 | 74,168 | +1.23 % | no | 233 |
| 15 | 71,381 | 72,783 | +1.96 % | no | 249 |
| 21 | 71,204 | 72,440 | +1.74 % | no | 204 |

Over +2 %: models 0, 1, 2, 3, 4, 5, 6, 7 (grid-style recorded worsts missed interior knob values, MIX (model 0 worst has
MIX 0x7afc, which no earlier grid varied), velocity and run-on after another model). Within 2 %: 12, 13, 14, 15, 21.
The test bounds (15 % margin) still hold except model 7 (+16 %: 39,530 vs bound 39,500; mode-1 bound exceeded) and model 3 is
at 89,835 vs bound 93,000. Model 7's bound should be raised (not done here; no behaviour change).

Emulator confirmations needed (not run): models 0, 1, 2, 3, 4, 5, 6, 7 (knobs in each JSON "confirm_knobs"); note
model 0 needs MIX, which the emulator tool does not set. Short budgets: a longer fuzz may find more.
