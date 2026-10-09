import tempfile
import unittest
from pathlib import Path

from zordon.agent import Agent
from zordon.audit import AuditLog
from zordon.config import Config
from zordon.memory import MemoryStore
from zordon.openai_provider import ToolCall
from zordon.tools import default_registry


class CaptureProvider:
    def __init__(self):
        self.inputs = []

    def stream_step(self, system_prompt, input_text, tools):
        self.inputs.append(input_text)
        yield "Done."
        return []


class RememberingProvider:
    def __init__(self):
        self.inputs = []

    def stream_step(self, system_prompt, input_text, tools):
        self.inputs.append(input_text)
        if len(self.inputs) == 1:
            return [
                ToolCall(
                    name="remember_preference",
                    arguments={
                        "key": "preferred_format",
                        "value": "short scripts",
                        "category": "preference",
                    },
                    call_id="call_memory",
                )
            ]
        yield "I will remember that."
        return []


class MemoryTest(unittest.TestCase):
    def test_memory_store_persists_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "memory.jsonl"
            store = MemoryStore(path=path)
            store.remember("preferred_format", "short scripts", "preference")

            reloaded = MemoryStore(path=path)

        self.assertEqual(len(reloaded.all()), 1)
        self.assertEqual(reloaded.all()[0].key, "preferred_format")
        self.assertIn("short scripts", reloaded.summary())

    def test_memory_store_forgets_entries_by_key(self):
        store = MemoryStore(entries=[])
        store.remember("preferred_format", "short scripts", "preference")
        store.remember("tone", "cool and calm", "preference")

        self.assertTrue(store.forget("PREFERRED_FORMAT"))

        self.assertNotIn("short scripts", store.summary())
        self.assertIn("cool and calm", store.summary())
        self.assertFalse(store.forget("missing"))

    def test_memory_store_returns_raw_json_lines(self):
        store = MemoryStore(entries=[])
        store.remember("tone", "cool and calm", "manual")

        raw_lines = store.raw_lines()

        self.assertEqual(len(raw_lines), 1)
        self.assertIn('"key": "tone"', raw_lines[0])
        self.assertIn('"category": "manual"', raw_lines[0])

    def test_default_registry_can_save_and_list_memories(self):
        store = MemoryStore(entries=[])
        tools = default_registry(store)

        result = tools.execute(
            "remember_preference",
            {
                "key": "tone",
                "value": "cool and calm",
                "category": "preference",
            },
        )
        listed = tools.execute("list_memories", {})

        self.assertTrue(result.ok)
        self.assertIn("Remembered tone", result.output)
        self.assertTrue(listed.ok)
        self.assertIn("cool and calm", listed.output)

    def test_agent_includes_durable_memory_in_model_input(self):
        config = _config()
        provider = CaptureProvider()
        store = MemoryStore(entries=[])
        store.remember("preferred_format", "short scripts", "preference")
        agent = Agent(config=config, provider=provider, audit=AuditLog(events=[]), memory=store)

        self.assertEqual("".join(agent.respond("What should we make?")), "Done.")

        self.assertIn("Durable memories", provider.inputs[0])
        self.assertIn("preferred_format: short scripts", provider.inputs[0])

    def test_agent_memory_tool_persists_before_final_reply(self):
        config = _config()
        provider = RememberingProvider()
        store = MemoryStore(entries=[])
        audit = AuditLog(events=[])
        agent = Agent(config=config, provider=provider, audit=audit, memory=store)

        self.assertEqual(
            "".join(agent.respond("Remember that I like short scripts.")),
            "[tool: remember_preference]\nI will remember that.",
        )

        self.assertIn("short scripts", store.summary())
        self.assertIn("remember_preference (ok)", provider.inputs[-1])
        self.assertEqual(audit.events[1]["name"], "remember_preference")


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
