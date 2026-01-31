# Ted AI — Sarcastic Talking Teddy Bear (Raspberry Pi)

Ted AI is a voice-driven teddy bear assistant for a :contentReference[oaicite:4]{index=4}. It listens for a wake word, transcribes speech, generates a snarky response, and speaks back using TTS.

## Features
- Wake word → listen loop
- Speech-to-text (Whisper / faster-whisper style pipeline)
- AI responses via :contentReference[oaicite:5]{index=5}
- Text-to-speech via :contentReference[oaicite:6]{index=6}
- TTS caching (reduces repeat API usage)
- Optional Bluetooth speaker output

## Requirements
- Raspberry Pi 4+ (recommended)
- Python 3.10+ (3.11+ recommended)
- Microphone + speaker (Bluetooth optional)
- API keys:
  - OpenAI
  - ElevenLabs

## Quick Start (Recommended)
```bash
git clone https://github.com/Tyler-ryan-code/ted-ai.git
cd ted-ai

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your keys

bash run_ted.sh

Run Manually (Dev)

source venv/bin/activate
python3 ted_listener.py

Environment Variables

Create a .env file (never commit it) using .env.example as a template.

Example keys:

OPENAI_API_KEY=...

ELEVENLABS_API_KEY=...

Bluetooth Audio (Optional)

bluetoothctl
power on
agent on
default-agent
scan on
pair <MAC_ADDRESS>
trust <MAC_ADDRESS>
connect <MAC_ADDRESS>
exit

Set default sink:

pactl set-default-sink <SINK_NAME>

Test audio:

aplay /usr/share/sounds/alsa/Front_Center.wav

Repo Notes

Generated audio, logs, caches, venv, and secrets are ignored via .gitignore.

License

MIT — see LICENSE


---








