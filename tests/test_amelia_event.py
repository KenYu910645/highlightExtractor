import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from amelia_event import (
    AmeliaEventDetector,
    merge_scored_windows,
    select_top_windows,
    select_top_windows_for_duration,
    windows_to_timeline,
)
from utils import combine_highlight_scores, pick_highlights


class AmeliaEventMergingTest(unittest.TestCase):
    def test_merge_scored_windows_keeps_peak(self):
        spans = merge_scored_windows(
            [
                {"start_sec": 0.0, "end_sec": 1.5, "center_sec": 0.75, "score": 0.61},
                {"start_sec": 1.6, "end_sec": 3.1, "center_sec": 2.35, "score": 0.72},
                {"start_sec": 5.0, "end_sec": 6.5, "center_sec": 5.75, "score": 0.80},
            ],
            threshold=0.6,
            max_gap_sec=0.75,
        )

        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0]["peak_sec"], 2.35)
        self.assertAlmostEqual(spans[0]["peak_prob"], 0.72, places=4)

    def test_timeline_projection_uses_max_overlap(self):
        timeline = windows_to_timeline(
            [
                {"start_sec": 0.0, "end_sec": 1.5, "center_sec": 0.75, "score": 0.2},
                {"start_sec": 1.0, "end_sec": 2.5, "center_sec": 1.75, "score": 0.9},
            ],
            duration=3.0,
        )

        self.assertGreater(timeline[1], 0.5)
        self.assertGreater(timeline[2], 0.2)


class HighlightFusionTest(unittest.TestCase):
    def test_pick_highlights_accepts_amelia_scores(self):
        audio = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        subtitle = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        amelia = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        peaks = pick_highlights(
            audio,
            subtitle,
            [],
            n_clips=1,
            min_gap=1,
            amelia_scores=amelia,
            weights=(0.25, 0.35, 0.40),
        )

        self.assertIn(peaks[0][0], {0, 1, 2})
        self.assertGreater(peaks[0][1], 0.0)

    def test_combine_highlight_scores_falls_back_without_detector(self):
        audio = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        subtitle = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        combined = combine_highlight_scores(audio, subtitle)

        self.assertEqual(len(combined), 3)
        self.assertGreater(combined[1], 0.0)
        self.assertGreater(combined[2], 0.0)


class RankedWindowSelectionTest(unittest.TestCase):
    def test_select_top_windows_sorts_desc_and_stops_at_threshold(self):
        selected = select_top_windows(
            [
                {"start_sec": 0.0, "end_sec": 1.5, "center_sec": 0.75, "score": 0.81},
                {"start_sec": 1.0, "end_sec": 2.5, "center_sec": 1.75, "score": 0.92},
                {"start_sec": 2.0, "end_sec": 3.5, "center_sec": 2.75, "score": 0.79},
            ],
            threshold=0.8,
            max_clip_sec=5.0,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["center_sec"], 1.75)
        self.assertLessEqual(selected[0]["end_sec"] - selected[0]["start_sec"], 5.0)

    def test_select_top_windows_caps_long_clips(self):
        selected = select_top_windows(
            [{"start_sec": 10.0, "end_sec": 14.8, "center_sec": 12.0, "score": 0.95}],
            threshold=0.8,
            max_clip_sec=3.0,
        )

        self.assertEqual(selected[0]["end_sec"], 13.0)

    def test_select_top_windows_consolidates_overlapping_windows(self):
        selected = select_top_windows(
            [
                {"start_sec": 49.0, "end_sec": 50.5, "center_sec": 49.75, "score": 0.89},
                {"start_sec": 49.5, "end_sec": 51.0, "center_sec": 50.25, "score": 0.87},
                {"start_sec": 69.0, "end_sec": 70.5, "center_sec": 69.75, "score": 0.86},
            ],
            threshold=0.8,
            max_clip_sec=5.0,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["start_sec"], 49.0)
        self.assertEqual(selected[0]["end_sec"], 51.0)

    def test_select_top_windows_for_duration_returns_chronological_clips(self):
        selected, threshold = select_top_windows_for_duration(
            [
                {"start_sec": 10.0, "end_sec": 11.5, "center_sec": 10.75, "score": 0.90},
                {"start_sec": 2.0, "end_sec": 3.5, "center_sec": 2.75, "score": 0.95},
                {"start_sec": 2.5, "end_sec": 4.0, "center_sec": 3.25, "score": 0.92},
            ],
            target_duration_sec=3.0,
            max_clip_sec=5.0,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["start_sec"], 2.0)
        self.assertEqual(selected[1]["start_sec"], 10.0)
        self.assertEqual(threshold, 0.9)


class AmeliaDetectorConfigTest(unittest.TestCase):
    def test_detector_reads_config_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prototypes.json"
            path.write_text(
                json.dumps(
                    {
                        "model_source": "dummy",
                        "config": {"window_sec": 2.0},
                        "prototype_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                    }
                ),
                encoding="utf-8",
            )

            detector = AmeliaEventDetector(path, device="cpu")

            self.assertEqual(detector.config.window_sec, 2.0)
            self.assertEqual(detector.config.hop_sec, 0.5)


if __name__ == "__main__":
    unittest.main()
