# Speaker Classification Pipeline Spec
## Project: Whisper subtitles + local speaker classification via pretrained embeddings

## Goal

Build a fully local offline pipeline that:

1. Takes an input gameplay video or audio file.
2. Uses existing Whisper subtitle timestamps as the base segmentation.
3. Assigns each subtitle segment to one of:
   - `Ken`
   - `Amelia`
   - `unknown`
4. Outputs a new subtitle file (`.srt` or `.json`) with speaker labels.

This is **not** a full diarization project.
We only need **binary known-speaker classification** for two recurring speakers.

---

## Chosen stack

### Core choice
Use **SpeechBrain ECAPA-TDNN speaker embeddings** for prototype matching.

### Supporting tools
- **Python 3.10+**
- **PyTorch**
- **SpeechBrain**
- **torchaudio**
- **Silero VAD**
- **ffmpeg**
- **numpy**
- **scipy**
- **pydub** or `ffmpeg-python`
- **pysrt** or custom SRT parser
- optional:
  - **librosa** for debugging / audio inspection
  - **scikit-learn** for later calibration or analysis
  - **matplotlib** for visualization

---

## Why this model

We want the most data-efficient path.

Instead of training a speaker model from scratch, we will:

1. Use a pretrained speaker embedding model.
2. Build one embedding prototype for each known speaker.
3. Compare each subtitle-aligned audio chunk against the two prototypes using cosine similarity.
4. Pick the closest speaker if confidence is high enough, otherwise output `unknown`.

This should work well because:
- there are only 2 known speakers
- voices are very different
- the pipeline is offline
- clean daughter data is limited

---

## Final recommendation

### Use:
- **SpeechBrain**: `speechbrain/spkrec-ecapa-voxceleb`
- **Silero VAD** for speech activity filtering
- **Whisper subtitles** as the primary segmentation source

### Do not do in v1:
- no full diarization
- no custom model training
- no fine-tuning
- no pitch-only heuristic as final logic

---

## Expected input / output

### Input
- gameplay video or audio file
- Whisper subtitle file (`.srt`, preferred)
- enrollment audio for:
  - `Ken`
  - `Amelia`

### Output
One of:
- updated `.srt` with speaker prefixes
- `.json` containing subtitle metadata + speaker label + confidence

Example output text:
- `Ken: Let's go this way`
- `Amelia: No!`
- `unknown: [laughing / unclear]`

---

## Directory layout

```text
speaker_labeler/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ src/
│  ├─ main.py
│  ├─ config.py
│  ├─ pipeline.py
│  ├─ audio/
│  │  ├─ ffmpeg_utils.py
│  │  ├─ vad.py
│  │  ├─ slicing.py
│  │  └─ resample.py
│  ├─ subtitles/
│  │  ├─ srt_parser.py
│  │  ├─ srt_writer.py
│  │  └─ subtitle_types.py
│  ├─ speaker/
│  │  ├─ embedding_model.py
│  │  ├─ prototype_store.py
│  │  ├─ classifier.py
│  │  ├─ smoothing.py
│  │  └─ calibration.py
│  ├─ utils/
│  │  ├─ paths.py
│  │  ├─ logging.py
│  │  └─ math_utils.py
│  └─ tests/
│     ├─ test_srt_parser.py
│     ├─ test_vad.py
│     ├─ test_classifier.py
│     └─ test_pipeline.py
├─ data/
│  ├─ enroll/
│  │  ├─ ken/
│  │  └─ amelia/
│  ├─ inputs/
│  ├─ outputs/
│  └─ cache/
└─ scripts/
   ├─ enroll.py
   ├─ label_subtitles.py
   └─ inspect_embeddings.py