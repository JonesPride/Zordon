import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zordon.config import Config
from zordon.images import generate_image, images_dir, latest_image, list_images


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


class ImagesTest(unittest.TestCase):
    def test_generate_image_saves_png_from_base64_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config()

            with (
                patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
                patch("urllib.request.urlopen", return_value=FakeResponse()),
            ):
                image = generate_image(config, root, "A green command center", "command-center")

            self.assertEqual(image.path, root / "images" / "command-center.png")
            self.assertEqual(image.path.read_bytes(), b"png-bytes")
            self.assertEqual(image.revised_prompt, "revised")

    def test_images_dir_uses_root_for_non_project_roots(self):
        root = Path("C:/Temp/ZordonTest")

        self.assertEqual(images_dir(root), root / "images")

    def test_list_and_latest_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = images_dir(root)
            directory.mkdir()
            first = directory / "first.png"
            second = directory / "second.png"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            os.utime(first, (1, 1))
            os.utime(second, (2, 2))

            self.assertEqual(list_images(root), ["first.png", "second.png"])
            self.assertEqual(latest_image(root).name, "second.png")


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


if __name__ == "__main__":
    unittest.main()
