import tempfile
import unittest
from pathlib import Path

from zordon.audit import AuditLog


class AuditLogTest(unittest.TestCase):
    def test_record_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing" / "audit.jsonl"
            audit = AuditLog(path=path)

            audit.record("test_event", value="ok")

            self.assertTrue(path.exists())
            self.assertIn('"event": "test_event"', path.read_text(encoding="utf-8"))

    def test_record_does_not_raise_when_file_write_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audit_dir"
            path.mkdir()
            audit = AuditLog(path=path, write_errors=[])

            audit.record("test_event")

            self.assertEqual(len(audit.write_errors), 1)


if __name__ == "__main__":
    unittest.main()
