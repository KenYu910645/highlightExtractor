import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from speaker_labeler.audio import estimate_voiced_seconds, slice_audio
from speaker_labeler.pipeline import (
    SpeakerLabelingConfig,
    SpeakerLabelingError,
    SpeakerLabelingPipeline,
    cosine_similarity,
    smooth_labels,
)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000):
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


class AudioHelpersTest(unittest.TestCase):
    def test_slice_audio_bounds(self):
        samples = np.arange(20, dtype=np.float32)
        chunk = slice_audio(samples, 10, 0.5, 1.4)
        np.testing.assert_array_equal(chunk, np.arange(5, 14, dtype=np.float32))

    def test_slice_audio_out_of_range(self):
        samples = np.arange(8, dtype=np.float32)
        chunk = slice_audio(samples, 4, -1.0, 0.5)
        np.testing.assert_array_equal(chunk, np.arange(0, 2, dtype=np.float32))

    def test_estimate_voiced_seconds_detects_energy(self):
        sample_rate = 16000
        silence = np.zeros(sample_rate, dtype=np.float32)
        tone = 0.1 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate).astype(np.float32)
        samples = np.concatenate([silence, tone])
        voiced = estimate_voiced_seconds(samples, sample_rate)
        self.assertGreater(voiced, 0.4)


class PipelineHelpersTest(unittest.TestCase):
    def test_cosine_similarity(self):
        vec_a = np.array([1.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.8, 0.2], dtype=np.float32)
        self.assertGreater(cosine_similarity(vec_a, vec_b), 0.9)

    def test_smooth_labels_replaces_isolated_flip(self):
        segments = [
            {"speaker": "Ken", "speaker_confidence": 0.3},
            {"speaker": "unknown", "speaker_confidence": 0.05},
            {"speaker": "Ken", "speaker_confidence": 0.4},
        ]
        smoothed = smooth_labels(segments, margin=0.08)
        self.assertEqual(smoothed[1]["speaker"], "Ken")
        self.assertTrue(smoothed[1]["speaker_smoothed"])

    def test_missing_enrollment_raises_clear_error(self):
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        config = SpeakerLabelingConfig(enroll_dir=temp_dir)
        pipeline = SpeakerLabelingPipeline(config)
        with self.assertRaises(SpeakerLabelingError):
            pipeline.build_prototypes()

    def test_cached_prototypes_load_without_model(self):
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        for speaker in ("ken", "amelia"):
            folder = temp_dir / speaker
            folder.mkdir(parents=True)
            write_wav(folder / "sample.wav", np.ones(1600, dtype=np.float32) * 0.1)

        config = SpeakerLabelingConfig(enroll_dir=temp_dir)
        pipeline = SpeakerLabelingPipeline(config)
        pipeline._save_cached_prototypes(
            {
                "Ken": [temp_dir / "ken" / "sample.wav"],
                "Amelia": [temp_dir / "amelia" / "sample.wav"],
            },
            {
                "Ken": np.array([1.0, 0.0], dtype=np.float32),
                "Amelia": np.array([0.0, 1.0], dtype=np.float32),
            },
        )

        cached = pipeline._load_cached_prototypes(
            {
                "Ken": [temp_dir / "ken" / "sample.wav"],
                "Amelia": [temp_dir / "amelia" / "sample.wav"],
            }
        )
        self.assertIsNotNone(cached)
        self.assertIn("Ken", cached)

    def test_prepared_files_are_preferred(self):
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        ken_dir = temp_dir / "ken"
        ken_dir.mkdir(parents=True)
        write_wav(ken_dir / "raw.wav", np.ones(1600, dtype=np.float32) * 0.1)
        prepared_dir = ken_dir / "prepared"
        prepared_dir.mkdir()
        write_wav(prepared_dir / "prepared.wav", np.ones(1600, dtype=np.float32) * 0.1)

        config = SpeakerLabelingConfig(enroll_dir=temp_dir)
        pipeline = SpeakerLabelingPipeline(config)
        files = pipeline._collect_files(ken_dir)
        self.assertEqual(files, [prepared_dir / "prepared.wav"])


if __name__ == "__main__":
    unittest.main()
