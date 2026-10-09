import tempfile
import unittest
from pathlib import Path

from zordon.drafts import delete_draft, list_drafts, read_draft, save_draft


class DraftsTest(unittest.TestCase):
    def test_save_list_and_read_draft(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            save_draft(root, "b.md", "Second")
            save_draft(root, "a.txt", "First")

            self.assertEqual(list_drafts(root), ["a.txt", "b.md"])
            self.assertEqual(read_draft(root, "a.txt"), "First")

    def test_list_drafts_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            drafts = root / "drafts"
            drafts.mkdir()
            (drafts / "a.md").write_text("Draft", encoding="utf-8")
            (drafts / "ignore.exe").write_text("Nope", encoding="utf-8")

            self.assertEqual(list_drafts(root), ["a.md"])

    def test_read_draft_rejects_missing_or_unsafe_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaises(FileNotFoundError):
                read_draft(root, "missing.md")
            with self.assertRaises(ValueError):
                read_draft(root, "..\\outside.md")

    def test_delete_draft_removes_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_draft(root, "idea.md", "Opening hook.")

            deleted = delete_draft(root, "idea.md")

            self.assertEqual(deleted.name, "idea.md")
            self.assertEqual(list_drafts(root), [])

    def test_delete_draft_rejects_missing_or_unsafe_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaises(FileNotFoundError):
                delete_draft(root, "missing.md")
            with self.assertRaises(ValueError):
                delete_draft(root, "../outside.md")


if __name__ == "__main__":
    unittest.main()
