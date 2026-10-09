import unittest

from zordon.audio_provider import _multipart_body


class AudioProviderTest(unittest.TestCase):
    def test_multipart_body_contains_file_and_model(self):
        body = _multipart_body(
            "boundary",
            fields={"model": "gpt-4o-transcribe"},
            files={"file": ("sample.wav", "audio/wav", b"RIFF")},
        )

        self.assertIn(b'name="model"', body)
        self.assertIn(b"gpt-4o-transcribe", body)
        self.assertIn(b'filename="sample.wav"', body)
        self.assertIn(b"Content-Type: audio/wav", body)
        self.assertIn(b"RIFF", body)


if __name__ == "__main__":
    unittest.main()
