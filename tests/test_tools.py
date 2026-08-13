import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zordon.config import Config
from zordon.projects import create_project
from zordon.tools import default_registry


class ToolsTest(unittest.TestCase):
    def test_save_draft_requires_confirmation_and_writes_after_approval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tools = default_registry(root=root)

            blocked = tools.execute(
                "save_draft",
                {"filename": "idea.md", "content": "Opening hook."},
            )
            saved = tools.execute(
                "save_draft",
                {"filename": "idea.md", "content": "Opening hook."},
                confirmed=True,
            )

            self.assertFalse(blocked.ok)
            self.assertIn("requires confirmation", blocked.output)
            self.assertTrue(saved.ok)
            self.assertEqual((root / "drafts" / "idea.md").read_text(encoding="utf-8"), "Opening hook.")

    def test_save_draft_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = default_registry(root=Path(temp_dir))

            result = tools.execute(
                "save_draft",
                {"filename": "..\\outside.md", "content": "Nope."},
                confirmed=True,
            )

            self.assertFalse(result.ok)
            self.assertIn("path separators", result.output)

    def test_save_draft_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tools = default_registry(root=Path(temp_dir))

            result = tools.execute(
                "save_draft",
                {"filename": "idea.exe", "content": "Nope."},
                confirmed=True,
            )

            self.assertFalse(result.ok)
            self.assertIn(".txt or .md", result.output)

    def test_save_draft_writes_to_active_project_when_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_project(root, "Green Campaign")
            tools = default_registry(root=root)

            saved = tools.execute(
                "save_draft",
                {"filename": "idea.md", "content": "Project hook."},
                confirmed=True,
            )

            self.assertTrue(saved.ok)
            self.assertIn("Project: Green Campaign", saved.output)
            self.assertEqual(
                (root / "projects" / "green-campaign" / "drafts" / "idea.md").read_text(encoding="utf-8"),
                "Project hook.",
            )
            self.assertFalse((root / "drafts" / "idea.md").exists())

    def test_generate_image_writes_to_active_project_when_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_project(root, "Green Campaign")
            tools = default_registry(root=root, config=_config())

            with (
                patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
                patch("urllib.request.urlopen", return_value=_fake_image_response()),
            ):
                saved = tools.execute(
                    "generate_image",
                    {"filename": "cover.png", "prompt": "A green cover"},
                    confirmed=True,
                )

            self.assertTrue(saved.ok)
            self.assertIn("Project: Green Campaign", saved.output)
            self.assertTrue((root / "projects" / "green-campaign" / "images" / "cover.png").exists())


def _config() -> Config:
    return Config(
        assistant_name="Zordon",
        model="test-model",
        transcription_model="test-transcribe",
        tts_model="test-tts",
        tts_voice="verse",
        push_to_talk_key="space",
        audio_sample_rate=16000,
        temperature=0.7,
        request_timeout_seconds=5,
        image_model="test-image",
        image_size="1024x1024",
        image_quality="low",
    )


def _fake_image_response():
    import base64
    import json

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {
                            "b64_json": base64.b64encode(b"png-bytes").decode("ascii"),
                            "revised_prompt": "revised",
                        }
                    ]
                }
            ).encode("utf-8")

    return FakeResponse()


if __name__ == "__main__":
    unittest.main()
