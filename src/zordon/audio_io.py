from __future__ import annotations

import tempfile
import time
import wave
from pathlib import Path

import numpy as np


class AudioError(RuntimeError):
    pass


def cue_sound(kind: str) -> None:
    try:
        import winsound
    except ImportError:
        return

    tones = {
        "listening": (880, 90),
        "thinking": (660, 90),
        "done": (520, 70),
        "error": (220, 140),
    }
    frequency, duration_ms = tones.get(kind, tones["done"])
    winsound.Beep(frequency, duration_ms)


def record_while_key_held(key: str, sample_rate: int) -> Path:
    try:
        import keyboard
        import sounddevice as sd
    except ImportError as exc:
        raise AudioError("Install sounddevice and keyboard to use push-to-talk.") from exc

    print(f"Hold {key.upper()} to talk. Release it to send.")
    keyboard.wait(key)
    cue_sound("listening")
    print("Listening...")

    frames: list[np.ndarray] = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f"Audio status: {status}")
        frames.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            while keyboard.is_pressed(key):
                time.sleep(0.02)
    except Exception as exc:
        raise AudioError(f"Could not record audio: {exc}") from exc

    if not frames:
        raise AudioError("No audio was captured.")

    audio = np.concatenate(frames, axis=0)
    path = Path(tempfile.gettempdir()) / "zordon-push-to-talk.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())
    return path


def play_wav(path: Path, interrupt_key: str | None = None) -> bool:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AudioError("Install sounddevice to play speech audio.") from exc

    # Stream audio in small chunks instead of loading entire file
    chunk_size = 1024  # frames per chunk

    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()

        interrupted = False

        if interrupt_key is None:
            # Non-interruptible playback - stream and play
            with sd.OutputStream(
                samplerate=sample_rate, channels=channels, dtype="int16", blocksize=chunk_size
            ) as stream:
                while True:
                    data = handle.readframes(chunk_size)
                    if not data:
                        break
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if channels > 1:
                        audio_chunk = audio_chunk.reshape(-1, channels)
                    stream.write(audio_chunk)
            return False
        else:
            # Interruptible playback
            import keyboard

            # First, get total duration
            total_frames = handle.getnframes()
            total_frames / sample_rate

            time.perf_counter()

            with sd.OutputStream(
                samplerate=sample_rate, channels=channels, dtype="int16", blocksize=chunk_size
            ) as stream:
                while True:
                    data = handle.readframes(chunk_size)
                    if not data:
                        break

                    # Check for interrupt
                    if keyboard.is_pressed(interrupt_key):
                        sd.stop()
                        interrupted = True
                        break

                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if channels > 1:
                        audio_chunk = audio_chunk.reshape(-1, channels)
                    stream.write(audio_chunk)

            return interrupted
