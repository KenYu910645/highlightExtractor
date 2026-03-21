import tempfile
import unittest
from pathlib import Path

from verify_subtitles import (
    build_report,
    evaluate_subtitles,
    parse_srt,
)


class ParseSrtTest(unittest.TestCase):
    def test_parses_utf8_chinese_blocks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.srt"
            path.write_text(
                "1\n00:00:00,100 --> 00:00:01,200\n你好\n\n"
                "2\n00:00:01,500 --> 00:00:02,000\n世界\n",
                encoding="utf-8",
            )

            entries = parse_srt(path)

            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].text, "你好")
            self.assertAlmostEqual(entries[1].start_sec, 1.5)

    def test_skips_invalid_and_empty_blocks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.srt"
            path.write_text(
                "1\nnot a time line\nbad\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n\n\n"
                "3\n00:00:03,000 --> 00:00:04,000\nok\n",
                encoding="utf-8",
            )

            entries = parse_srt(path)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].text, "ok")


class EvaluateSubtitlesTest(unittest.TestCase):
    def _write_pair(self, pred_text: str, gt_text: str):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        pred_path = Path(temp_dir.name) / "pred.srt"
        gt_path = Path(temp_dir.name) / "gt.srt"
        pred_path.write_text(pred_text, encoding="utf-8")
        gt_path.write_text(gt_text, encoding="utf-8")
        return parse_srt(pred_path), parse_srt(gt_path)

    def test_perfect_match_scores_ideal_metrics(self):
        pred_entries, gt_entries = self._write_pair(
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n",
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n",
        )

        metrics = evaluate_subtitles(pred_entries, gt_entries)

        self.assertEqual(metrics["global_cer"], 0.0)
        self.assertEqual(metrics["matched_pairs"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["start_error_mean_sec"], 0.0)
        self.assertEqual(metrics["end_error_mean_sec"], 0.0)

    def test_text_mismatch_increases_cer(self):
        pred_entries, gt_entries = self._write_pair(
            "1\n00:00:00,000 --> 00:00:01,000\n你好啊\n\n",
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n",
        )

        metrics = evaluate_subtitles(pred_entries, gt_entries)

        self.assertGreater(metrics["global_cer"], 0.0)
        self.assertEqual(metrics["matched_pairs"], 1)

    def test_timing_drift_lowers_f1(self):
        pred_entries, gt_entries = self._write_pair(
            "1\n00:00:02,000 --> 00:00:03,000\n你好\n\n",
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n",
        )

        metrics = evaluate_subtitles(pred_entries, gt_entries)

        self.assertEqual(metrics["matched_pairs"], 0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_segmentation_mismatch_still_matches_greedily(self):
        pred_entries, gt_entries = self._write_pair(
            "1\n00:00:00,000 --> 00:00:04,000\n打了啦打了啦來\n\n",
            "1\n00:00:00,000 --> 00:00:02,000\n打了啦\n\n"
            "2\n00:00:02,000 --> 00:00:04,000\n打了啦來\n\n",
        )

        metrics = evaluate_subtitles(pred_entries, gt_entries)

        self.assertEqual(metrics["pred_count"], 1)
        self.assertEqual(metrics["gt_count"], 2)
        self.assertEqual(metrics["matched_pairs"], 1)
        self.assertLess(metrics["precision"], 1.0)
        self.assertLess(metrics["recall"], 1.0)

    def test_unmatched_subtitles_reduce_precision_and_recall(self):
        pred_entries, gt_entries = self._write_pair(
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\n再見\n\n",
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n",
        )

        metrics = evaluate_subtitles(pred_entries, gt_entries)

        self.assertEqual(metrics["matched_pairs"], 1)
        self.assertLess(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_report_contains_headline_metrics(self):
        pred_entries, gt_entries = self._write_pair(
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n",
            "1\n00:00:00,000 --> 00:00:01,000\n你好\n\n",
        )

        metrics = evaluate_subtitles(pred_entries, gt_entries)
        report = build_report(Path("pred.srt"), Path("gt.srt"), metrics)

        self.assertIn("Global CER", report)
        self.assertIn("Start error", report)
        self.assertIn("End error", report)
        self.assertIn("Match F1", report)


if __name__ == "__main__":
    unittest.main()
