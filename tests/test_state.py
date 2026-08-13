import tempfile
import unittest
from pathlib import Path

from zordon.state import migrate_state, state_text


class StateTest(unittest.TestCase):
    def test_state_text_reports_storage_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            text = state_text(root)

        self.assertIn("Zordon state", text)
        self.assertIn("state root:", text)
        self.assertIn("memory.jsonl", text)
        self.assertIn("drafts folder:", text)
        self.assertIn("images folder:", text)
        self.assertIn("logs folder:", text)

    def test_migrate_state_copies_known_runtime_folders(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            source = Path(source_dir)
            target = Path(target_dir)
            (source / "data").mkdir()
            (source / "drafts").mkdir()
            (source / "images").mkdir()
            (source / "logs").mkdir()
            (source / "data" / "memory.jsonl").write_text("memory\n", encoding="utf-8")
            (source / "drafts" / "idea.txt").write_text("draft\n", encoding="utf-8")
            (source / "images" / "cover.png").write_bytes(b"png")
            (source / "logs" / "zordon.jsonl").write_text("log\n", encoding="utf-8")

            result = migrate_state(source, target)

            self.assertEqual(len(result.copied), 4)
            self.assertTrue((target / "data" / "memory.jsonl").exists())
            self.assertTrue((target / "drafts" / "idea.txt").exists())
            self.assertTrue((target / "images" / "cover.png").exists())
            self.assertTrue((target / "logs" / "zordon.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
