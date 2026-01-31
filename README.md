# Ted AI - Sarcastic Talking Teddy Bear

🧸 **Ted AI** is a Raspberry Pi-based interactive teddy bear that listens, thinks, and talks with personality. Say the wake word and Ted responds with a sarcastic Boston accent, powered by AI.

---

## Features

- **Wake-word detection**: Say "Hey Jarvis" and Ted wakes up.
- **Speech-to-text**: Converts your speech to text using `faster-whisper`.
- **AI responses**: Generates sarcastic, personality-driven replies via OpenAI.
- **Text-to-speech**: Speaks with a Boston-accented voice using ElevenLabs.
- **Bluetooth audio support**: Works with external Bluetooth speakers.
- **Offline-first design**: Core audio processing runs on the Pi.
- **TTS caching**: Saves audio locally to minimize API usage.

---

## Requirements

- Raspberry Pi 4 or newer
- Python 3.13
- Bluetooth speaker (optional, for audio output)
- ElevenLabs API key (for TTS)
- OpenAI API key (for AI responses)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ted-ai.git
cd ted

Create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Set your API keys:

export ELEVEN_API_KEY="your_elevenlabs_api_key"
export OPENAI_API_KEY="your_openai_api_key"

Usage

Start Ted:

python ted_listener.py


Say the wake word: "Hey Jarvis"

Speak your command

Ted responds with AI-generated speech

Optional: Bluetooth Speaker Setup

Make sure the adapter is powered:

sudo hciconfig hci0 up


Pair and connect via bluetoothctl:

power on
agent on
default-agent
scan on
pair <MAC_ADDRESS>
trust <MAC_ADDRESS>
connect <MAC_ADDRESS>
exit


Set as default audio sink:

pactl set-default-sink <SINK_NAME>


Test audio:

aplay /usr/share/sounds/alsa/Front_Center.wav

Tips

Keep replies short to save ElevenLabs TTS credits

Cached TTS audio is stored in cache/

Use Ctrl+C to safely stop Ted

License

This project is licensed under the MIT License — see the LICENSE
 file for details.

Made with 🧸, ☕, and a little bit of sarcasm.
