import json
import tempfile
import unittest
from pathlib import Path

from analyze_image import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    JSON_FALLBACK_MESSAGE,
    build_messages,
    build_parser,
    format_response,
    path_to_file_uri,
    resolve_image_path,
    write_output,
)


class AnalyzeImagePathTest(unittest.TestCase):
    def test_resolve_image_path_requires_existing_file(self):
        with self.assertRaises(FileNotFoundError):
            resolve_image_path("missing-image.png")

    def test_path_to_file_uri_escapes_spaces(self):
        path = Path(r"C:\temp\face cam frame 01.png")

        uri = path_to_file_uri(path)

        self.assertEqual(uri, "file:///C:/temp/face%20cam%20frame%2001.png")

    def test_resolve_image_path_returns_absolute_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.jpg"
            path.write_bytes(b"fake")

            resolved = resolve_image_path(str(path))

            self.assertTrue(resolved.is_absolute())
            self.assertEqual(resolved.name, "frame.jpg")


class AnalyzeImageFormattingTest(unittest.TestCase):
    def test_build_messages_uses_default_json_instruction(self):
        messages = build_messages("file:///tmp/test.png", DEFAULT_PROMPT, json_mode=True)

        self.assertEqual(messages[0]["content"][0]["image"], "file:///tmp/test.png")
        self.assertIn("Return JSON only", messages[0]["content"][1]["text"])

    def test_format_response_keeps_plain_text(self):
        output, validated = format_response(" A detailed description. ", json_mode=False)

        self.assertEqual(output, "A detailed description.")
        self.assertFalse(validated)

    def test_format_response_pretty_prints_valid_json(self):
        output, validated = format_response('```json\n{"description":"cat"}\n```', json_mode=True)

        self.assertTrue(validated)
        self.assertEqual(json.loads(output), {"description": "cat"})

    def test_format_response_wraps_invalid_json(self):
        output, validated = format_response("not actually json", json_mode=True)

        self.assertFalse(validated)
        payload = json.loads(output)
        self.assertFalse(payload["validated_json"])
        self.assertEqual(payload["warning"], JSON_FALLBACK_MESSAGE)
        self.assertEqual(payload["raw_output"], "not actually json")


class AnalyzeImageCliTest(unittest.TestCase):
    def test_parser_defaults_match_plan(self):
        args = build_parser().parse_args(["image.png"])

        self.assertEqual(args.prompt, DEFAULT_PROMPT)
        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertEqual(args.max_new_tokens, DEFAULT_MAX_NEW_TOKENS)
        self.assertIsNone(args.max_pixels)
        self.assertFalse(args.json)
        self.assertIsNone(args.out)

    def test_write_output_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "nested" / "answer.json"

            written = write_output('{"ok":true}', str(out_path))

            self.assertEqual(written, out_path.resolve())
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.read_text(encoding="utf-8"), '{"ok":true}\n')


if __name__ == "__main__":
    unittest.main()
