import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from zordon import voice_cli
from zordon.memory import MemoryStore


class VoiceCliTest(unittest.TestCase):
    def test_typed_fallback_handles_help_command_without_audio_or_model(self):
        memory = MemoryStore(entries=[])

        with (
            patch.object(sys, "argv", ["run_voice.py", "--typed"]),
            patch("zordon.voice_cli.load_dotenv"),
            patch("zordon.voice_cli.load_config", return_value=_config()),
            patch("zordon.voice_cli.default_audit_log", return_value=_audit()),
            patch("zordon.voice_cli.default_memory_store", return_value=memory),
            patch("zordon.voice_cli.OpenAIProvider"),
            patch("zordon.voice_cli.OpenAIAudioProvider"),
            patch("builtins.input", side_effect=["/help", "exit"]),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(voice_cli.main(), 0)

        text = output.getvalue()
        self.assertIn("Typed fallback supports slash commands.", text)
        self.assertIn("Commands:", text)
        self.assertIn("/delete-draft <filename>", text)
        self.assertIn("Goodbye.", text)

    def test_typed_fallback_handles_memory_command(self):
        memory = MemoryStore(entries=[])
        memory.remember("tone", "cool and calm", "manual")

        with (
            patch.object(sys, "argv", ["run_voice.py", "--typed"]),
            patch("zordon.voice_cli.load_dotenv"),
            patch("zordon.voice_cli.load_config", return_value=_config()),
            patch("zordon.voice_cli.default_audit_log", return_value=_audit()),
            patch("zordon.voice_cli.default_memory_store", return_value=memory),
            patch("zordon.voice_cli.OpenAIProvider"),
            patch("zordon.voice_cli.OpenAIAudioProvider"),
            patch("builtins.input", side_effect=["/memory", "exit"]),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(voice_cli.main(), 0)

        self.assertIn("tone: cool and calm", output.getvalue())

    def test_typed_fallback_exits_cleanly_on_eof(self):
        memory = MemoryStore(entries=[])

        with (
            patch.object(sys, "argv", ["run_voice.py", "--typed"]),
            patch("zordon.voice_cli.load_dotenv"),
            patch("zordon.voice_cli.load_config", return_value=_config()),
            patch("zordon.voice_cli.default_audit_log", return_value=_audit()),
            patch("zordon.voice_cli.default_memory_store", return_value=memory),
            patch("zordon.voice_cli.OpenAIProvider"),
            patch("zordon.voice_cli.OpenAIAudioProvider"),
            patch("builtins.input", side_effect=EOFError),
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(voice_cli.main(), 0)

        self.assertIn("Goodbye.", output.getvalue())


def _config():
    return type(
        "ConfigStub",
        (),
        {
            "assistant_name": "Zordon",
            "model": "test-model",
            "transcription_model": "test-transcribe",
            "tts_model": "test-tts",
            "tts_voice": "verse",
            "push_to_talk_key": "space",
            "audio_sample_rate": 16000,
            "temperature": 0.7,
            "request_timeout_seconds": 5,
        },
    )()


def _audit():
    return type(
        "AuditStub", (), {"path": Path("test.log"), "record": lambda *args, **kwargs: None}
    )()


if __name__ == "__main__":
    unittest.main()
