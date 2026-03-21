# AGENTS.md - Sekiro Gaming Session Highlight Extractor

This file is important to the project. Read it before working, and keep it up to date whenever code changes alter the workflow, defaults, outputs, or verification process.

## Project Overview

This project automates post-processing for gaming session recordings made while playing **Sekiro: Shadows Die Twice** with Amelia (Ken's daughter). Each session is typically a 30-90 minute MP4 screen recording with a facecam overlay showing both player and child reactions.

The goal is to make it easy to produce a curated set of highlight clips from each session, capturing heartwarming, funny, and exciting moments without manually scrubbing through the full video every time.

The human will manually cherry-pick and edit the output highlight clips, so **err on the side of too many clips rather than missing a funny moment**.

---

## Pipeline Architecture

The project supports two workflows:

### A. Three-Step Pipeline (recommended)

```text
Step 1: python preprocess.py data/DayN/DayN.mp4
           -> DayN.srt + DayN_candidates/ (thumbnails + candidates.md)

Step 2: Claude AI reviews candidates.md + images + optional SRT
           -> highlights.md

Step 3: python postprocess.py data/DayN/DayN.mp4 data/DayN/highlights.md --srt data/DayN/DayN.srt
           -> highlight/ (final clips with burned-in subtitles)
```

Why three steps:
- The AI sees actual thumbnails plus nearby subtitle context.
- Candidate generation is intentionally generous.
- The final editorial decision stays in human hands.

### B. Monolithic Script (legacy, still functional)

```bash
python3 highlight_extractor.py data/DayN/DayN.mp4
```

This runs a full automated pass in one script without AI review.

---

## Project Context

- Game: `Sekiro: Shadows Die Twice`
- Players: Ken and Amelia
- Language: mostly Mandarin Chinese, sometimes English praise/reactions
- Facecam: top-right overlay with visible reactions
- Session naming: `Day1.mp4`, `Day2.mp4`, etc.
- Recording length: usually 30-90 minutes
- Typical source size: around 4 GB at 1080p60

### What makes a good highlight

Priority order:
1. Amelia's verbal reactions, laughter, surprise, commentary
2. Dad and daughter talking to each other
3. Funny or memorable real-world interruptions
4. Boss fights and named encounters
5. Lore talk or funny non-canonical explanations

Clip guidance:
- Aim for 10-30 clips per session
- Each clip should usually be 3-30 seconds
- Total highlight runtime should stay under 5 minutes
- Avoid leaving long silent stretches inside a clip

---

## Current Repository Layout

```text
highlightExtractor/
├── AGENTS.md
├── highlight_extractor.py
├── postprocess.py
├── prepare_enroll.py
├── preprocess.py
├── utils.py
├── verify_subtitles.py
├── data/
├── tests/
└── verification/
```

Important notes:
- `preprocess.py`, `postprocess.py`, and `highlight_extractor.py` are the active pipeline scripts.
- `utils.py` holds shared transcription, scoring, and clip helpers.
- `verify_subtitles.py` computes subtitle-quality metrics between a predicted SRT and a ground-truth SRT.
- `prepare_enroll.py` still exists in the repo, but speaker identification is no longer part of the active pipeline.
- The active automated test suite currently lives in `tests/test_verify_subtitles.py`.

---

## Setup

### System Requirements

- Python 3.8+
- `ffmpeg` installed and on `PATH`
- `libass` available in FFmpeg if subtitle burning is needed

### Python Dependencies

Install once:

```bash
pip install openai-whisper numpy opencc-python-reimplemented --break-system-packages
```

Notes:
- `openai-whisper` downloads model weights on first use.
- The preprocess pipeline currently defaults to the Whisper `large` model.

### Whisper Model Tradeoffs

| Model | Size | Speed (60 min video) | Accuracy |
|---|---:|---:|---|
| tiny | 75 MB | ~5 min | Basic |
| small | 460 MB | ~20 min | Good |
| medium | 1.4 GB | ~50 min | Better |
| large | larger | slowest | Best of the currently used options |

Practical guidance:
- `large` is the default for `preprocess.py`
- `medium` is still useful when runtime matters
- `small` is the fallback for memory-constrained runs

---

## Step 1: preprocess.py

Purpose:
- Transcribe the full video with Whisper
- Convert subtitles to Traditional Chinese
- Score each second of the session
- Select generous candidate highlight timestamps
- Extract thumbnails
- Write `candidates.md` for AI review

### Current CLI

```bash
python preprocess.py <video.mp4> [options]

  --candidates N
  --min-gap N
  --model NAME
  --beam-size N
  --best-of N
  --temperatures S
  --condition-on-previous-text
  --no-condition-on-previous-text
  --initial-prompt TEXT
  --frame-size WxH
```

Current defaults:
- `--model large`
- `--beam-size 5`
- `--best-of 5`
- `--temperatures 0.0,0.2,0.4,0.6`
- `--no-condition-on-previous-text`
- `--frame-size 640x360`

### Current Output

For input `data/Day5/Day5.mp4`, the script writes:
- `data/Day5/Day5.srt`
- `data/Day5/Day5_candidates/candidate_*.jpg`
- `data/Day5/Day5_candidates/candidates.md`

---

## Step 2: AI Review

Give Claude:
1. `DayN_candidates/candidates.md`
2. All candidate JPEGs in that folder
3. Optionally, `DayN.srt`

Expected output format:

```markdown
## highlight_01
* start: 02:34
* end: 02:58
* reason: Amelia shouts and laughs at the boss explosion
* confidence: 0.95
```

`postprocess.py` expects:
- Header: `## highlight_N`
- `start` and `end`: `MM:SS` or `HH:MM:SS`
- `reason`: single-line text
- `confidence`: float in `[0.0, 1.0]`

Blocks missing `start` or `end` are skipped.

---

## Step 3: postprocess.py

Purpose:
- Parse the AI-authored markdown
- Cut approved clips from the source video
- Optionally burn in subtitles

### Current CLI

```bash
python postprocess.py <video.mp4> <highlights.md> [options]

  --srt PATH
  --min-confidence F
  --no-burn
  --out-dir PATH
  --min-dur N
  --max-dur N
```

Current defaults:
- `--min-confidence 0.0`
- subtitles burned in unless `--no-burn` is used
- output goes to `highlight/` beside the source video

---

## Legacy: highlight_extractor.py

This script still exists and still works as a one-pass automatic pipeline.

Current CLI:

```bash
python3 highlight_extractor.py <video.mp4> [options]

  --clips N
  --min-gap N
  --min-dur N
  --max-dur N
  --model NAME
  --no-burn
```

Current default model for the legacy script:
- `medium`

---

## Scoring System

Used by both `preprocess.py` and `highlight_extractor.py`:

```text
combined_score = 0.45 * audio_score + 0.55 * subtitle_score
```

### Audio Score

- 1-second RMS windows
- Compared against a 60-second rolling median baseline
- Louder-than-local-baseline moments score higher

### Subtitle Score

- Keyword weighting using `REACTION_KEYWORDS` in `utils.py`
- Dialogue density bonus
- `MUST_INCLUDE_KEYWORDS` force maximum score

If scoring behavior changes, update this file and the verification expectations.

---

## Verification For preprocess.py

Any change that touches `preprocess.py` must run the verification flow below before it is considered verified.

The goal is not only to confirm that the script still runs, but also to make sure subtitle quality stays reasonably close to the current baseline.

### Required verification workflow

```bash
# 1. Keep the test suite green first
python -m unittest discover -s tests -p "test_*.py"

# 2. Build or refresh the standard verification clip
ffmpeg -y -ss 00:00:00 -i data/Day1/Day1.mp4 -t 00:10:00 -c:v libx264 -preset veryfast -crf 23 -c:a aac verification/verification.mp4

# 3. Run the real preprocess pipeline on that clip
#    Use preprocess.py defaults unless testing a specific non-default flag
python preprocess.py verification/verification.mp4 --candidates 5

# 4. Compare the generated subtitles to ground truth
python verify_subtitles.py --pred verification/verification.srt --gt verification/groundtrue.srt
```

### Expected verification outputs

The run should create:
- `verification/verification.srt`
- `verification/verification_candidates/candidates.md`
- candidate JPEG thumbnails under `verification/verification_candidates/`

The run should complete without crashing, and the generated `.srt` should contain real subtitle entries with timestamps.

### Current baseline

Baseline pair:
- Predicted: `verification/verification.srt`
- Ground truth: `verification/groundtrue.srt`

Current baseline metrics:
- Pred blocks: `157`
- GT blocks: `206`
- Global CER: `0.3920`
- Matched pairs: `104`
- Start error: mean `908.7 ms`, median `318.0 ms`
- End error: mean `384.6 ms`, median `280.0 ms`
- Match F1: `P=0.2803 R=0.2136 F1=0.2424`
- Correct matches: `44`

### Regression guardrails

Treat the change as a regression and call it out clearly if any of these happen relative to baseline:
- `CER` increases by more than `0.03`
- `Match F1` drops by more than `0.03`
- Mean start error increases by more than `250 ms`
- Mean end error increases by more than `150 ms`

Small movement is acceptable, but material regressions must be reported explicitly in the final response.

If a change intentionally trades one metric for another, say so clearly and include both the old and new values.

---

## Known Limitations

- Whisper can still struggle with overlapping speech, child speech, and heavy game audio.
- Subtitle burning depends on FFmpeg `libass`.
- The pipeline is currently forced to Chinese transcription with `language="zh"`.
- Thumbnail extraction uses fast seek, so thumbnails may be a little off from the exact subtitle peak.

---

## Working Norm

Whenever code changes alter:
- CLI flags
- defaults
- generated files
- verification files
- baseline metrics
- workflow steps

update this file in the same task so it stays aligned with the actual project.
