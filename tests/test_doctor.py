import unittest
from pathlib import Path
from unittest.mock import patch

from zordon.config import Config
from zordon.doctor import doctor_text, run_doctor
from zordon.memory import MemoryStore
from zordon.tools import default_registry


class DoctorTest(unittest.TestCase):
    def test_doctor_warns_when_api_key_is_missing(self):
        memory = MemoryStore(entries=[])
        tools = default_registry(memory)

        with patch.dict("os.environ", {}, clear=True):
            text = doctor_text(_config(), memory, tools, Path("."))

        self.assertIn("Zordon doctor: needs attention", text)
        self.assertIn("WARN OPENAI_API_KEY", text)
        self.assertIn("OK config", text)
        self.assertIn("OK tools", text)
        self.assertTrue(
            "requirements-voice.txt" in text
            or "- OK sounddevice: installed" in text
            or "- OK keyboard: installed" in text
        )

    def test_doctor_reports_api_key_when_set(self):
        memory = MemoryStore(entries=[])
        tools = default_registry(memory)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "set"}, clear=True):
            checks = run_doctor(_config(), memory, tools, Path("."))

        api_check = next(check for check in checks if check.name == "OPENAI_API_KEY")
        self.assertTrue(api_check.ok)
        self.assertEqual(api_check.detail, "set")


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
    )


if __name__ == "__main__":
    unittest.main()
