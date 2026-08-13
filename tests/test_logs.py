import tempfile
import unittest
from pathlib import Path

from zordon.logs import latest_log, list_logs, read_log_tail, search_log, summarize_events, summarize_last_turn


class LogsTest(unittest.TestCase):
    def test_list_logs_and_read_tail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text("one\ntwo\nthree\n", encoding="utf-8")
            (logs / "ignore.txt").write_text("nope", encoding="utf-8")

            self.assertEqual(list_logs(root), ["zordon-2026-07-21.jsonl"])
            self.assertEqual(read_log_tail(root, "zordon-2026-07-21.jsonl", line_count=2), "two\nthree")
            self.assertEqual(latest_log(root), "zordon-2026-07-21.jsonl")

    def test_summarize_events_counts_event_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text(
                '{"event":"user_turn"}\n{"event":"assistant_reply"}\n{"event":"user_turn"}\nnot json\n',
                encoding="utf-8",
            )

            summary = summarize_events(root, "zordon-2026-07-21.jsonl")

        self.assertIn("Events in zordon-2026-07-21.jsonl:", summary)
        self.assertIn("- user_turn: 2", summary)
        self.assertIn("- assistant_reply: 1", summary)
        self.assertIn("- invalid_lines: 1", summary)

    def test_summarize_events_uses_latest_log_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-20.jsonl").write_text('{"event":"old"}\n', encoding="utf-8")
            (logs / "zordon-2026-07-21.jsonl").write_text('{"event":"new"}\n', encoding="utf-8")

            summary = summarize_events(root)

        self.assertIn("Events in zordon-2026-07-21.jsonl:", summary)
        self.assertIn("- new: 1", summary)

    def test_summarize_last_turn_finds_latest_user_and_assistant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text(
                (
                    '{"event":"user_turn","text":"First"}\n'
                    '{"event":"assistant_reply","text":"Reply one"}\n'
                    '{"event":"user_turn","text":"Second"}\n'
                    '{"event":"assistant_reply","text":"Reply two"}\n'
                ),
                encoding="utf-8",
            )

            summary = summarize_last_turn(root, "zordon-2026-07-21.jsonl")

        self.assertIn("Last turn in zordon-2026-07-21.jsonl:", summary)
        self.assertIn("User: Second", summary)
        self.assertIn("Assistant: Reply two", summary)

    def test_summarize_last_turn_uses_latest_log_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-20.jsonl").write_text('{"event":"user_turn","text":"Old"}\n', encoding="utf-8")
            (logs / "zordon-2026-07-21.jsonl").write_text('{"event":"user_turn","text":"New"}\n', encoding="utf-8")

            summary = summarize_last_turn(root)

        self.assertIn("Last turn in zordon-2026-07-21.jsonl:", summary)
        self.assertIn("User: New", summary)

    def test_search_log_finds_matching_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text(
                '{"event":"user_turn","text":"Find the launch hook"}\n{"event":"assistant_reply","text":"Done"}\n',
                encoding="utf-8",
            )

            output = search_log(root, "launch")

        self.assertIn("Matches for 'launch' in zordon-2026-07-21.jsonl:", output)
        self.assertIn("line 1 user_turn", output)
        self.assertIn("Find the launch hook", output)

    def test_search_log_reports_no_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            logs = root / "logs"
            logs.mkdir()
            (logs / "zordon-2026-07-21.jsonl").write_text('{"event":"user_turn","text":"Hello"}\n', encoding="utf-8")

            output = search_log(root, "missing", "zordon-2026-07-21.jsonl")

        self.assertEqual(output, "No matches for 'missing' in zordon-2026-07-21.jsonl.")

    def test_read_log_rejects_missing_or_unsafe_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaises(FileNotFoundError):
                read_log_tail(root, "missing.jsonl")
            with self.assertRaises(ValueError):
                read_log_tail(root, "..\\outside.jsonl")
            with self.assertRaises(ValueError):
                read_log_tail(root, "audit.txt")


if __name__ == "__main__":
    unittest.main()
