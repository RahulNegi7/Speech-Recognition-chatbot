"""
Generates a sample audio WAV file in assets/ for testing audio upload and ASR.
"""
import io
import wave
import struct
import math
from pathlib import Path

assets_dir = Path(__file__).parent / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
sample_file = assets_dir / "sample_audio.wav"

sample_rate = 16000
duration = 2.0  # seconds
num_samples = int(duration * sample_rate)

with wave.open(str(sample_file), "wb") as wf:
    wf.setnchannels(1)      # mono
    wf.setsampwidth(2)      # 16-bit PCM
    wf.setframerate(sample_rate)
    # Generate a soft combination of 440Hz and 880Hz tones
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        val = 0.3 * math.sin(2.0 * math.pi * 440.0 * t) + 0.1 * math.sin(2.0 * math.pi * 880.0 * t)
        samples.append(int(32767.0 * val))
    wf.writeframes(struct.pack(f"<{num_samples}h", *samples))

print(f"Sample WAV generated at: {sample_file}")

