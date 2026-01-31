#!/usr/bin/env python3

import os
import time
import random
import warnings
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

from openwakeword.model import Model
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs.client import ElevenLabs

# -------------------- ENV / WARNINGS --------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ORT_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# -------------------- CONFIG --------------------

WAKE_WORD = "models/hey_jarvis.onnx"        # must exist as .onnx or builtin
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1.2  # seconds

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

VOICE_ID = "2vubyVoGjNJ5HPga4SkV"  # Boston-style voice
CACHE_DIR = "cache"

# -------------------- COLORS --------------------

G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
B = "\033[94m"
RESET = "\033[0m"

# -------------------- STARTUP --------------------

print(f"{B}🧸 Ted is booting… and he's already annoyed.{RESET}")

try:
    wake_model = Model(wakeword_model_paths=[WAKE_WORD])
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    eleven = ElevenLabs(api_key=ELEVEN_API_KEY)

    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"{G}✅ All systems online. Ted is ready to judge you.{RESET}")

except Exception as e:
    print(f"{R}❌ Startup failed: {e}{RESET}")
    exit(1)

# -------------------- TTS --------------------

def speak(text: str):
    print(f"{B}🧸 TED:{RESET} {text}")

    safe = "".join(c for c in text if c.isalnum()).lower()[:40]
    filename = f"{CACHE_DIR}/{safe}.wav"

    if os.path.exists(filename):
        os.system(f"aplay {filename}")
        return

    if not ELEVEN_API_KEY:
        print(f"{Y}⚠ No ElevenLabs key set.{RESET}")
        return

    try:
        audio = eleven.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="wav_22050"
        )

        with open(filename, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        os.system(f"aplay {filename}")

    except Exception as e:
        print(f"{R}❌ ElevenLabs error: {e}{RESET}")

# -------------------- WAKE WORD --------------------

def listen_for_wake():
    print(f"\n{B}🎧 Listening for '{WAKE_WORD}'…{RESET}")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE
    ) as stream:

        while True:
            audio, _ = stream.read(CHUNK_SIZE)
            frame = np.frombuffer(audio, dtype=np.int16)

            prediction = wake_model.predict(frame)

            for _, score in prediction.items():
                if score > 0.5:
                    print(f"{G}✨ Wake word detected! ({score:.2f}){RESET}")
                    speak(random.choice([
                        "Yeah? Whaddya want?",
                        "This better be good.",
                        "Wicked. What?",
                        "I'm listenin'."
                    ]))
                    return

# -------------------- RECORDING --------------------

def record_until_silence():
    print(f"{Y}🎤 Recording…{RESET}")

    frames = []
    silent_chunks = 0
    max_silent = int((SAMPLE_RATE / CHUNK_SIZE) * SILENCE_DURATION)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE
    ) as stream:

        while True:
            data, _ = stream.read(CHUNK_SIZE)
            frames.append(data.copy())

            amplitude = np.abs(np.frombuffer(data, np.int16)).mean()

            if amplitude < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks > max_silent:
                print(f"{B}⏹ Silence detected. Stopping.{RESET}")
                break

    audio = np.concatenate(frames)
    wav.write("command.wav", SAMPLE_RATE, audio)

# -------------------- TRANSCRIBE --------------------

def transcribe():
    segments, _ = whisper_model.transcribe("command.wav")
    text = " ".join(seg.text for seg in segments).strip()

    if text:
        print(f"{G}🗣 YOU SAID:{RESET} {text}")

    return text

# -------------------- AI --------------------

def ask_ted(prompt: str) -> str:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.85,
            messages=[
                {
                    "role": "system",
                    "content": "You are Ted, a sarcastic teddy bear with a thick Boston accent. Be brief."
                },
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI's havin' a stroke. {e}"

# -------------------- MAIN LOOP --------------------

if __name__ == "__main__":
    try:
        speak("Ted is back, kid. Try not to break anything.")

        while True:
            listen_for_wake()
            record_until_silence()
            text = transcribe()

            if len(text) > 2:
                speak(ask_ted(text))
            else:
                speak("Speak up, I can't hear ya.")

    except KeyboardInterrupt:
        print(f"\n{R}🛑 Shutting down Ted… finally.{RESET}")
