import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_scene_log import (
    DEFAULT_FPS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    build_frame_index,
    build_raw_log_with_propagation,
    build_parser,
    DEFAULT_MAX_SKIP_SEC,
    DEFAULT_SSIM_THRESHOLD,
    DEFAULT_SSIM_SIZE,
    extract_candidate_events,
    merge_scene_segments,
    parse_size,
    parse_scene_analysis,
    select_frames_for_qwen,
    smooth_scene_log,
)


class SceneLogCliTest(unittest.TestCase):
    def test_parser_defaults_match_plan(self):
        args = build_parser().parse_args(["video.mp4"])

        self.assertEqual(args.fps, DEFAULT_FPS)
        self.assertEqual(args.max_new_tokens, DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertIsNone(args.max_pixels)
        self.assertIsNone(args.max_frames)
        self.assertEqual(args.ssim_threshold, DEFAULT_SSIM_THRESHOLD)
        self.assertEqual(args.max_skip_sec, DEFAULT_MAX_SKIP_SEC)
        self.assertEqual(args.ssim_size, f"{DEFAULT_SSIM_SIZE[0]}x{DEFAULT_SSIM_SIZE[1]}")
        self.assertFalse(args.disable_ssim_gating)
        self.assertFalse(args.enable_vlm)
        self.assertFalse(args.force)


class FrameIndexTest(unittest.TestCase):
    def test_build_frame_index_uses_sampling_interval(self):
        frames = [Path("frame_000001.jpg"), Path("frame_000002.jpg"), Path("frame_000003.jpg")]

        index = build_frame_index(frames, fps=0.5, video_duration=60.0)

        self.assertEqual(index[0]["timestamp"], 0.0)
        self.assertEqual(index[1]["timestamp"], 2.0)
        self.assertEqual(index[2]["timestamp"], 4.0)


class ParseSceneAnalysisTest(unittest.TestCase):
    def test_parse_scene_analysis_normalizes_valid_payload(self):
        analysis, parse_ok, parse_error = parse_scene_analysis(
            '{"scene_type":"boss_combat","scene_summary":"very big boss fight right now",'
            '"important_entities":["Boss","boss","UI"],"intensity":7,"danger_level":"4",'
            '"ken_expression":"tense","amelia_expression":"excited","notable_change":"yes","confidence":1.4}'
        )

        self.assertTrue(parse_ok)
        self.assertIsNone(parse_error)
        self.assertEqual(analysis["scene_type"], "boss_combat")
        self.assertEqual(analysis["scene_summary"], "very big boss fight right now")
        self.assertEqual(analysis["important_entities"], ["Boss", "UI"])
        self.assertEqual(analysis["intensity"], 5)
        self.assertEqual(analysis["danger_level"], 4)
        self.assertTrue(analysis["notable_change"])
        self.assertEqual(analysis["confidence"], 1.0)

    def test_parse_scene_analysis_falls_back_when_json_missing(self):
        analysis, parse_ok, parse_error = parse_scene_analysis("not json")

        self.assertFalse(parse_ok)
        self.assertEqual(parse_error, "no_json_object_found")
        self.assertEqual(analysis["scene_type"], "other")
        self.assertEqual(analysis["confidence"], 0.0)


class SmoothingTest(unittest.TestCase):
    def test_smoothing_replaces_isolated_scene_label_and_expression_spike(self):
        raw = [
            {
                "timestamp": 0.0,
                "frame": "frame_000001.jpg",
                "parse_ok": True,
                "analysis": {
                    "scene_type": "combat",
                    "scene_summary": "fight",
                    "important_entities": ["player"],
                    "intensity": 1,
                    "danger_level": 1,
                    "ken_expression": "neutral",
                    "amelia_expression": "neutral",
                    "notable_change": False,
                    "confidence": 0.5,
                },
            },
            {
                "timestamp": 2.0,
                "frame": "frame_000002.jpg",
                "parse_ok": True,
                "analysis": {
                    "scene_type": "boss_combat",
                    "scene_summary": "boss",
                    "important_entities": ["boss"],
                    "intensity": 5,
                    "danger_level": 5,
                    "ken_expression": "surprised",
                    "amelia_expression": "excited",
                    "notable_change": False,
                    "confidence": 0.4,
                },
            },
            {
                "timestamp": 4.0,
                "frame": "frame_000003.jpg",
                "parse_ok": True,
                "analysis": {
                    "scene_type": "combat",
                    "scene_summary": "fight",
                    "important_entities": ["player"],
                    "intensity": 1,
                    "danger_level": 1,
                    "ken_expression": "neutral",
                    "amelia_expression": "neutral",
                    "notable_change": False,
                    "confidence": 0.5,
                },
            },
        ]

        smoothed = smooth_scene_log(raw)

        self.assertEqual(smoothed[1]["analysis"]["scene_type"], "combat")
        self.assertEqual(smoothed[1]["analysis"]["ken_expression"], "neutral")
        self.assertEqual(smoothed[1]["analysis"]["amelia_expression"], "neutral")
        self.assertEqual(smoothed[1]["analysis"]["intensity"], 2)
        self.assertEqual(smoothed[1]["analysis"]["danger_level"], 2)


class SsimSelectionTest(unittest.TestCase):
    def _make_frame(self, folder: Path, name: str, color: int) -> Path:
        path = folder / name
        Image.new("L", (32, 32), color=color).save(path)
        return path

    def test_parse_size_reads_width_height(self):
        self.assertEqual(parse_size("64x36"), (64, 36))

    def test_ssim_selection_skips_similar_frames_and_forces_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            frames = [
                self._make_frame(folder, "frame_000001.jpg", 0),
                self._make_frame(folder, "frame_000002.jpg", 0),
                self._make_frame(folder, "frame_000003.jpg", 0),
                self._make_frame(folder, "frame_000004.jpg", 255),
            ]
            frame_index = [
                {"frame": frame.name, "frame_path": str(frame), "timestamp": ts}
                for frame, ts in zip(frames, [0.0, 2.0, 8.0, 10.0])
            ]

            selection = select_frames_for_qwen(
                frame_index,
                ssim_threshold=0.08,
                max_skip_sec=8.0,
                ssim_size=(32, 32),
                disable_ssim_gating=False,
            )

            self.assertTrue(selection[0]["selected_for_qwen"])
            self.assertEqual(selection[0]["selection_reason"], "first_frame")
            self.assertFalse(selection[1]["selected_for_qwen"])
            self.assertEqual(selection[1]["selection_reason"], "propagated")
            self.assertTrue(selection[2]["selected_for_qwen"])
            self.assertEqual(selection[2]["selection_reason"], "max_skip_sec")
            self.assertTrue(selection[3]["selected_for_qwen"])
            self.assertEqual(selection[3]["selection_reason"], "ssim_change")

    def test_build_raw_log_with_propagation_keeps_every_timestamp(self):
        frame_index = [
            {"frame": "frame_000001.jpg", "timestamp": 0.0},
            {"frame": "frame_000002.jpg", "timestamp": 2.0},
            {"frame": "frame_000003.jpg", "timestamp": 4.0},
        ]
        selection = [
            {"frame": "frame_000001.jpg", "timestamp": 0.0, "selected_for_qwen": True, "selection_reason": "first_frame", "ssim": None, "change_score": None},
            {"frame": "frame_000002.jpg", "timestamp": 2.0, "selected_for_qwen": False, "selection_reason": "propagated", "ssim": 0.99, "change_score": 0.01},
            {"frame": "frame_000003.jpg", "timestamp": 4.0, "selected_for_qwen": True, "selection_reason": "ssim_change", "ssim": 0.70, "change_score": 0.30},
        ]
        qwen_entries_by_key = {
            ("frame_000001.jpg", 0.0): {
                "timestamp": 0.0,
                "frame": "frame_000001.jpg",
                "analysis": {"scene_type": "menu", "scene_summary": "menu", "important_entities": [], "intensity": 1, "danger_level": 0, "ken_expression": "unclear", "amelia_expression": "unclear", "notable_change": False, "confidence": 0.9},
                "parse_ok": True,
                "analysis_source": "qwen",
            },
            ("frame_000003.jpg", 4.0): {
                "timestamp": 4.0,
                "frame": "frame_000003.jpg",
                "analysis": {"scene_type": "combat", "scene_summary": "fight", "important_entities": [], "intensity": 4, "danger_level": 3, "ken_expression": "tense", "amelia_expression": "excited", "notable_change": True, "confidence": 0.8},
                "parse_ok": True,
                "analysis_source": "qwen",
            },
        }

        raw_log = build_raw_log_with_propagation(frame_index, selection, qwen_entries_by_key)

        self.assertEqual(len(raw_log), 3)
        self.assertEqual(raw_log[1]["analysis_source"], "propagated")
        self.assertEqual(raw_log[1]["analysis"]["scene_type"], "menu")
        self.assertEqual(raw_log[1]["propagated_from_timestamp"], 0.0)
        self.assertEqual(raw_log[2]["analysis_source"], "qwen")


class SegmentAndEventTest(unittest.TestCase):
    def test_merge_scene_segments_groups_matching_scene_types(self):
        smoothed = [
            {
                "timestamp": 0.0,
                "frame": "frame_000001.jpg",
                "parse_ok": True,
                "analysis": {
                    "scene_type": "combat",
                    "scene_summary": "fight one",
                    "important_entities": ["player"],
                    "intensity": 4,
                    "danger_level": 3,
                    "ken_expression": "tense",
                    "amelia_expression": "neutral",
                    "notable_change": False,
                    "confidence": 0.7,
                },
                "raw_analysis": {},
            },
            {
                "timestamp": 2.0,
                "frame": "frame_000002.jpg",
                "parse_ok": True,
                "analysis": {
                    "scene_type": "combat",
                    "scene_summary": "fight two",
                    "important_entities": ["player", "ui"],
                    "intensity": 5,
                    "danger_level": 4,
                    "ken_expression": "tense",
                    "amelia_expression": "excited",
                    "notable_change": True,
                    "confidence": 0.9,
                },
                "raw_analysis": {},
            },
            {
                "timestamp": 4.0,
                "frame": "frame_000003.jpg",
                "parse_ok": True,
                "analysis": {
                    "scene_type": "death",
                    "scene_summary": "player died",
                    "important_entities": ["death_screen"],
                    "intensity": 2,
                    "danger_level": 1,
                    "ken_expression": "surprised",
                    "amelia_expression": "tense",
                    "notable_change": False,
                    "confidence": 0.8,
                },
                "raw_analysis": {},
            },
        ]

        segments = merge_scene_segments(smoothed, fps=0.5)
        events = extract_candidate_events(smoothed)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 4.0)
        self.assertEqual(segments[0]["scene_type"], "combat")
        self.assertEqual(segments[0]["scene_summary"], "fight two")

        event_types = [event["event_type"] for event in events]
        self.assertIn("scene_transition", event_types)
        self.assertIn("notable_change", event_types)
        self.assertIn("amelia_expression_change", event_types)


if __name__ == "__main__":
    unittest.main()
