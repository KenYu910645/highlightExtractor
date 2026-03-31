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
Step 0: python build_amelia_prototypes.py
           -> data/enroll/amelia_event_prototypes.json

Step 1: python preprocess.py data/DayN/DayN.mp4
           -> DayN.srt + DayN_candidates/ (thumbnails + candidates.md)
           -> DayN_amelia_events.json (if prototypes are available)
           -> DayN_amelia_ranked_review.mp4 + DayN_amelia_ranked_review_windows.json

Step 2: Claude AI reviews candidates.md + images + optional SRT
           -> highlights.md

Step 3: python postprocess.py data/DayN/DayN.mp4 data/DayN/highlights.md --srt data/DayN/DayN.srt
           -> highlight/ (final clips with burned-in subtitles)
```

Why three steps:
- The AI sees actual thumbnails plus nearby subtitle context.
- Candidate generation is intentionally generous.
- Amelia prototype matching can lift child-reaction moments even when subtitles are weak.
- The final editorial decision stays in human hands.

### B. Monolithic Script (legacy, still functional)

```bash
python3 highlight_extractor.py data/DayN/DayN.mp4
```

This runs a full automated pass in one script without AI review.

### C. Manual Single-Image Vision Tool (V1 side tool)

```text
image.png

   ↓
python analyze_image.py image.png --prompt "Describe this image in detail."
   ↓
stdout or answer.txt / answer.json
```

Purpose:
- run `Qwen/Qwen2.5-VL-7B-Instruct` locally on one image
- inspect a single frame or exported screenshot on demand
- support best-effort JSON output for debugging or structured notes

Important:
- this tool is manual only in V1
- it is not part of `preprocess.py`, `postprocess.py`, or `highlight_extractor.py`
- it does not batch frames, analyze full videos, or affect candidate ranking

### D. Pass A Scene Timeline Builder (manual semantic logging)

```text
video.mp4
   ->
python build_scene_log.py video.mp4
   ->
video_scene_frames/ + scene log JSON artifacts
```

Purpose:
- sample frames from a video at low FPS
- run `Qwen/Qwen2.5-VL-7B-Instruct` on each frame with a fixed JSON schema
- build a coarse semantic timeline for later highlight work

Important:
- this is still not highlight selection
- it is currently a manual side workflow for debugging and timeline inspection
- it uses single-frame analysis only in V1

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
├── amelia_event.py
├── analyze_image.py
├── build_scene_log.py
├── build_amelia_review_video.py
├── build_amelia_prototypes.py
├── highlight_extractor.py
├── postprocess.py
├── prepare_enroll.py
├── preprocess.py
├── qwen_vl.py
├── utils.py
├── verify_subtitles.py
├── data/
├── tests/
└── verification/
```

Important notes:
- `preprocess.py`, `postprocess.py`, and `highlight_extractor.py` are the active pipeline scripts.
- `analyze_image.py` is a separate manual side tool for one-image vision analysis with Qwen2.5-VL.
- `build_scene_log.py` is a separate manual side tool for coarse semantic scene logging across a full video.
- `qwen_vl.py` holds shared Qwen model-loading and image-inference helpers used by both vision scripts.
- `amelia_event.py` implements few-shot Amelia event scoring from enrollment clips.
- `build_amelia_review_video.py` creates a ranked Amelia review reel from raw detector windows.
- `build_amelia_prototypes.py` builds the reusable Amelia prototype artifact.
- `utils.py` holds shared transcription, scoring, and clip helpers.
- `verify_subtitles.py` computes subtitle-quality metrics between a predicted SRT and a ground-truth SRT.
- `prepare_enroll.py` still exists in the repo, but speaker identification is no longer part of the active pipeline.
- The active automated test suite currently lives in `tests/`, with coverage in `test_verify_subtitles.py`, `test_amelia_event.py`, `test_analyze_image.py`, and `test_build_scene_log.py`.

---

## Setup

### System Requirements

- Python 3.8+
- `ffmpeg` installed and on `PATH`
- `libass` available in FFmpeg if subtitle burning is needed

### Python Dependencies

Install once:

```bash
pip install openai-whisper numpy opencc-python-reimplemented speechbrain scipy accelerate pillow scikit-image --break-system-packages
```

Notes:
- `openai-whisper` downloads model weights on first use.
- The preprocess pipeline currently defaults to the Whisper `large` model.
- The Amelia detector uses SpeechBrain ECAPA embeddings from `pretrained_models/speechbrain/spkrec-ecapa-voxceleb` by default.
- The Qwen side tool also needs a recent `transformers` build with Qwen2.5-VL support. If your installed version is too old, follow upstream guidance and upgrade `transformers` before using `analyze_image.py`.
- `qwen-vl-utils` is not required for V1 because the tool only handles one local image at a time.

### Manual image-analysis tool

Current CLI:

```bash
python analyze_image.py <image> [options]

  --prompt TEXT
  --json
  --out PATH
  --max-new-tokens N
  --max-pixels N
  --model NAME
```

Current defaults:
- `--prompt "Describe this image in detail."`
- plain text output unless `--json` is used
- `--max-new-tokens 256`
- `--model Qwen/Qwen2.5-VL-7B-Instruct`
- GPU-first loading via `device_map="auto"`
- `torch_dtype=torch.bfloat16` when CUDA is available, otherwise CPU-safe fallback

Examples:

```bash
python analyze_image.py frame.png
python analyze_image.py frame.png --prompt "What is happening here?"
python analyze_image.py frame.png --json --out answer.json
```

Behavior:
- accepts one local image file
- prints the model answer to stdout
- writes the same answer to `--out` when requested
- in `--json` mode, asks the model for JSON and pretty-prints valid JSON when possible
- if JSON parsing fails, returns a fallback JSON object containing the raw model output and a warning that it was not validated

### Pass A scene timeline tool

Current CLI:

```bash
python build_scene_log.py <video.mp4> [options]

  --enable-vlm
  --fps F
  --model NAME
  --max-new-tokens N
  --max-pixels N
  --max-frames N
  --ssim-threshold F
  --max-skip-sec N
  --ssim-size WxH
  --disable-ssim-gating
  --force
```

Current defaults:
- VLM scene logging is temporarily disabled unless `--enable-vlm` is passed
- `--fps 0.5`
- `--model Qwen/Qwen2.5-VL-7B-Instruct`
- `--max-new-tokens 192`
- `--ssim-threshold 0.08`
- `--max-skip-sec 8.0`
- `--ssim-size 64x36`
- extracted frames are stored beside the source video
- raw analysis is cached and reused unless `--force` is set

Current outputs for input `data/Day5/Day5.mp4`:
- `data/Day5/Day5_scene_frames/frame_*.jpg`
- `data/Day5/Day5_scene_frame_index.json`
- `data/Day5/Day5_scene_selection.json`
- `data/Day5/Day5_scene_log_raw.json`
- `data/Day5/Day5_scene_log_smoothed.json`
- `data/Day5/Day5_scene_segments.json`
- `data/Day5/Day5_candidate_events.json`

Behavior:
- refuses to run unless `--enable-vlm` is passed, because the VLM timeline workflow is temporarily disabled
- samples frames with FFmpeg at low FPS
- computes whole-frame grayscale SSIM against the last frame accepted for Qwen
- skips Qwen on highly similar frames unless `--max-skip-sec` forces a refresh
- asks Qwen for JSON-only frame descriptions using a fixed schema
- normalizes malformed or partial outputs into fallback records instead of aborting
- propagates the last accepted analysis forward when a frame is skipped by SSIM gating
- smooths isolated frame-level noise
- merges contiguous scene spans
- extracts candidate timeline events from transitions, spikes, and explicit notable changes

Examples:

```bash
python build_scene_log.py verification/verification.mp4 --enable-vlm --max-frames 10
python build_scene_log.py data/Day1/Day1.mp4 --enable-vlm --fps 1.0 --max-frames 120
python build_scene_log.py verification/verification.mp4 --enable-vlm --fps 0.5 --max-pixels 262144
```

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
- Optionally add Amelia-event scores from few-shot prototype matching
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
  --amelia-prototypes PATH
  --disable-amelia-detector
  --amelia-weight F
  --disable-amelia-review-video
  --amelia-review-target-fraction F
  --amelia-review-max-clip-sec N
```

Current defaults:
- `--model large`
- `--beam-size 5`
- `--best-of 5`
- `--temperatures 0.0,0.2,0.4,0.6`
- `--no-condition-on-previous-text`
- `--frame-size 640x360`
- `--amelia-prototypes data/enroll/amelia_event_prototypes.json`
- `--amelia-weight 0.40`
- Amelia detector auto-runs when the prototype artifact exists unless `--disable-amelia-detector` is used
- Amelia review video auto-runs when detector scoring succeeds unless `--disable-amelia-review-video` is used
- `--amelia-review-target-fraction 0.10`
- `--amelia-review-max-clip-sec 5.0`

### Current Output

For input `data/Day5/Day5.mp4`, the script writes:
- `data/Day5/Day5.srt`
- `data/Day5/Day5_candidates/candidate_*.jpg`
- `data/Day5/Day5_candidates/candidates.md`
- `data/Day5/Day5_amelia_events.json` when Amelia prototype scoring is enabled
- `data/Day5/Day5_amelia_ranked_review.mp4` when Amelia review video generation is enabled
- `data/Day5/Day5_amelia_ranked_review_windows.json` with the selected review clips

### Amelia prototype setup

Build the prototype artifact from prepared enrollment clips:

```bash
python build_amelia_prototypes.py
```

Expected source clips:
- `data/enroll/amelia/prepared/*.wav`

Recommended Amelia clips:
- mostly Amelia only
- 1-3 seconds long
- laughs, squeals, shouts, excited exclamations, or playful babble
- minimal game audio and minimal silence padding

Avoid:
- Ken-dominant clips
- mixed dialogue where Amelia is not clearly dominant
- clips buried under loud game SFX/music
- long calm speech turns

### Amelia review video

To create a single review video from raw detector windows without merging:

```bash
python build_amelia_review_video.py verification/verification.mp4 verification/verification_amelia_events.json --target-fraction 0.10 --max-clip-sec 5
```

Behavior:
- sorts raw detector windows by score descending
- dynamically determines the minimum score needed to reach the target output duration
- consolidates overlapping raw windows into one clip so the review reel does not contain duplicates
- caps each output clip to 5 seconds
- reorders the final clips by time so the earliest scene appears first
- concatenates all kept clips into one review video
- intermediate concat files and per-clip folders are treated as temporary files and are not kept
- `preprocess.py` calls this automatically after detector scoring by default

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

Used by `preprocess.py` when the detector is unavailable, and still by `highlight_extractor.py`:

```text
combined_score = 0.45 * audio_score + 0.55 * subtitle_score
```

When Amelia prototype scoring is enabled in `preprocess.py`, candidate ranking becomes:

```text
combined_score = 0.25 * audio_score + 0.35 * subtitle_score + 0.40 * amelia_event_score
```

Notes:
- Amelia scoring is intentionally high recall.
- Subtitle must-include keywords still override the normal rank ordering.

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

# 1b. Refresh Amelia prototypes if enrollment clips changed
python build_amelia_prototypes.py

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
- `verification/verification_amelia_events.json` if the prototype artifact is present

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
