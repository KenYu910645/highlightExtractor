# AGENTS.md — Sekiro Gaming Session Highlight Extractor

## Project Overview

This project automates the post-processing of gaming session recordings made while playing **Sekiro: Shadows Die Twice** with Amelia (Ken's daughter). Each session is a 30–60 minute MP4 screen recording with a facecam overlay showing both player and child reactions.

The goal is to make it easy to produce a curated set of highlight clips from each session — capturing heartwarming, funny, and exciting moments — without manually scrubbing through the full video every time.

The human is going to manually cherry-pick and edit the outputed hightlight clip, so make sure all the worth-mentioned clip is included, I'd rather you put too many clip than missing out some really funny moments.

---

## What the Pipeline Does

Given a raw `.mp4` session recording, the script automatically:

1. **Transcribes** the audio using OpenAI Whisper (runs locally, 100% free, no API fees)
2. **Generates** a full `.srt` subtitle file for the entire video
3. **Scores** every second of the video using two combined signals:
   - 🔊 **Audio amplitude (RMS)** — detects loud/exciting moments (screams, laughter, gasps)
   - 💬 **Subtitle content** — detects Traditional Chinese reaction keywords and dialogue density
4. **Selects** the top N highlight moments, ensuring must-include keywords (e.g. `地震`, `爆米花`) always get a clip
5. **Cuts** each highlight as an individual MP4 clip
6. **Burns** the subtitles directly into each clip
7. **Saves** a clip-specific `.srt` file alongside each clip

---

## Project Context & Content Notes

- **Game:** Sekiro: Shadows Die Twice
- **Players:** Ken (dad, handles controller) + Amelia (young daughter, co-commentator)
- **Language:** Mandarin Chinese (Traditional), with occasional English praise
- **Facecam:** Top-right corner overlay showing both players' reactions throughout
- **Session naming:** Files are named `Day1.mp4`, `Day2.mp4`, etc.
- **Recording length:** Typically 30–60 minutes per session
- **File size:** Approximately 4 GB per session at 1080p 60fps

### What makes a good highlight
1. Amelia's verbal reactions — exclamations, laughter, funny commentary on enemies
2. Dad and daughter talking to each other — teaching moments, baby-talk story explanations
3. Dad say something interseting/or have a different perspective(non-concanical) of the game
4. Boss fight moments — especially named bosses
5. Amelia discovering new game mechanics (swimming, grappling, stealth kills)
6. Unexpected event during the streaming.
7. Each hightlight shoudl be 10 seconds to 60 seconeds, if the build up is longer, you may extended the length , but never cut it shorter

---

## Setup & Requirements

### System Requirements
- Python 3.8+
- `ffmpeg` installed and on PATH
- `libass` (for subtitle burning — usually bundled with ffmpeg on macOS/Linux)

### Python Dependencies

Install once:
```bash
pip install openai-whisper numpy --break-system-packages
```

> **Note:** `openai-whisper` downloads the model weights on first run (~460 MB for `small`, ~1.4 GB for `medium`). After that, no internet is required.

### Whisper Model Tradeoffs

| Model  | Size   | Speed (40 min video) | Accuracy |
|--------|--------|----------------------|----------|
| tiny   | 75 MB  | ~3 min               | Basic    |
| small  | 460 MB | ~15 min              | Good ✅  |
| medium | 1.4 GB | ~35 min              | Better   |

`small` is the recommended default — it handles Mandarin Chinese well and fits in available memory.
`medium` was tested but caused out-of-memory issues on this machine (killed at ~25% model download).

---

## Running the Script

### Basic usage (recommended for daily use)
```bash
python3 highlight_extractor.py Day5.mp4
```

This will:
- Create `Day5.srt` in the same folder as the video
- Create a `highlight/` subfolder with 15 clips (each ~15 sec, subtitles burned in)

### All options
```bash
python3 highlight_extractor.py <video.mp4> [options]

  --clips N        Number of highlight clips to produce   (default: 15)
  --duration N     Duration of each clip in seconds       (default: 15)
  --min-gap N      Minimum seconds between clip centers   (default: 25)
  --model NAME     Whisper model: tiny / small / medium   (default: small)
  --no-burn        Skip subtitle burning (clips still get .srt files)
```

### Examples
```bash
# More clips, longer each
python3 highlight_extractor.py Day5.mp4 --clips 20 --duration 20

# Fast run without subtitle burn (e.g. just want to preview timestamps)
python3 highlight_extractor.py Day5.mp4 --no-burn

# Higher accuracy transcription
python3 highlight_extractor.py Day5.mp4 --model medium
```

---

## Output Structure

```
SEKIRO/
├── Day4.mp4                         ← original session recording
├── Day4.srt                         ← full video subtitles (generated)
├── highlight_extractor.py           ← this script
├── AGENTS.md                        ← this file
└── highlight/
    ├── 01_01m29s_阿嬷在拜拜.mp4     ← clip with burned subtitles
    ├── 01_01m29s_阿嬷在拜拜.srt     ← clip-specific subtitle file
    ├── 02_02m54s_有泳诶.mp4
    ├── 02_02m54s_有泳诶.srt
    ├── ...
    └── 15_36m41s_擦桌子.mp4         ← e.g. "wiping the table mid-boss-fight"
```

Clip filenames follow the pattern: `{index}_{mm}m{ss}s_{subtitle_description}.mp4`

---

## Highlight Scoring System

Every second of the video receives a combined score:

```
combined_score = 0.45 × audio_score + 0.55 × subtitle_score
```

### Audio Score (RMS excitement ratio)
- Computes RMS (root mean square) amplitude in 1-second windows
- Compares each window against a 60-second rolling median baseline
- High ratio = significantly louder than surrounding audio = reaction/excitement

### Subtitle Score (keyword + density)
- Scans each subtitle segment for **Traditional Chinese reaction keywords**
- Each keyword has a weight (see `REACTION_KEYWORDS` dict in the script)
- Adds a **dialogue density bonus** — rapid back-and-forth conversation scores higher

### Keyword Weight Reference

| Category | Keywords | Weight |
|----------|----------|--------|
| Must-include (forced) | 地震, 爆米花, 擦桌子, 哇塞, 你好厲害, 太厲害 | forced |
| Strong exclamation | 哇塞, 哎呀, 哈哈, 嘻嘻 | 4–5 |
| Reaction / praise | 你好厲害, 好聰明, 太棒了 | 4–5 |
| Discovery | 打雷, 下雪, 快看 | 3–4 |
| Common exclamation | 哇, 啊, 诶, 怕怕 | 2–3 |
| Character names | 阿嬤, 哥哥, 格格, 叔叔 | 1–2 |

To tune the criteria, edit the `REACTION_KEYWORDS` dictionary and `MUST_INCLUDE_KEYWORDS` set at the top of `highlight_extractor.py`.

---

## Known Limitations

- **Whisper accuracy with children's voices:** Amelia's voice is occasionally mis-transcribed, especially when she speaks quickly or over game audio. The transcript is good enough for highlight detection but may need light editing for publication.
- **Subtitle burn requires libass:** If `ffmpeg` is built without `libass`, burning will fail gracefully and the raw clip is saved instead (a warning is printed). The `.srt` file is always saved alongside.
- **medium model OOM:** The `medium` Whisper model (~1.4 GB) caused out-of-memory errors on this machine. Use `small` unless running on a machine with more RAM.
- **Language auto-detection:** The script forces `language="zh"` for Chinese. If future sessions include more English, consider switching to `language=None` for auto-detection.

---

## Workflow for Each New Session

1. Record the gaming session → save as `DayN.mp4` in the SEKIRO folder
2. Run:
   ```bash
   python3 highlight_extractor.py DayN.mp4
   ```
3. Wait ~15–20 minutes (Whisper transcription + clip cutting)
4. Review the clips in `highlight/` — they are named descriptively by the subtitle text at that moment
5. Use the `.srt` files or burned subtitles for social media posting (Instagram Reels, YouTube Shorts, etc.)
6. Optionally assemble a 60-second reel by hand-picking the best 5–6 clips
