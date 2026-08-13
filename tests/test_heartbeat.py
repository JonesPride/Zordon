import unittest
from pathlib import Path

from zordon.config import Config
from zordon.heartbeat import build_pulse
from zordon.memory import MemoryStore
from zordon.tools import default_registry


class HeartbeatTest(unittest.TestCase):
    def test_build_pulse_reports_local_state(self):
        memory = MemoryStore(entries=[])
        memory.remember("tone", "cool and calm", "manual")
        tools = default_registry(memory)

        pulse = build_pulse(_config(), memory, tools, Path("audit.jsonl"))

        text = pulse.as_text()
        self.assertIn("Zordon pulse", text)
        self.assertIn("model: test-model", text)
        self.assertIn("memories: 1", text)
        self.assertIn("remember_preference", text)
        self.assertIn("audit.jsonl", text)


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
