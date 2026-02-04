#!/usr/bin/env python3
"""
ted_listener.py — Ted on Raspberry Pi (Bookworm)

Features:
- Wake word (openwakeword)
- Record-until-silence (min duration)
- STT (faster-whisper) forced English + VAD
- Router: weather / news / wikipedia / facts / chat (anti-hijack order)
- OpenAI (Ted persona, answer-first)
- ElevenLabs TTS with file cache
- Debug logs + live HUD status line
- Wake cooldown to prevent Ted waking himself up

Env vars:
  OPENAI_API_KEY   required for LLM
  ELEVEN_API_KEY   required for ElevenLabs
  OWM_KEY          optional for weather (OpenWeather free)
  TED_CITY         optional default "Newtown,CT,US"
  TED_UNITS        optional "imperial" or "metric"
  WIKI_LANG        optional default "en"
  TED_DEBUG        "1" default on, "0" off
  TED_HUD          "1" default on, "0" off
"""

import os
import time
import random
import warnings
import hashlib
import json
import re
import urllib.parse
from datetime import datetime, timezone
import threading
from datetime import datetime, timedelta

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import requests

from openwakeword.model import Model
from faster_whisper import WhisperModel
from openai import OpenAI
from elevenlabs.client import ElevenLabs

# -------------------- ENV / WARNINGS --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["ORT_LOGGING_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# -------------------- CONFIG --------------------
WAKE_WORD_MODEL = "models/hey_jarvis.onnx"
WAKE_WORD_DISPLAY = "Hey Jarvis"

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

SCHEDULED = []          # list of dicts: {"when": epoch_seconds, "label": str}
SCHED_LOCK = threading.Lock()
STOP_EVENT = threading.Event()

SILENCE_THRESHOLD = 300
SILENCE_DURATION = 1.2  # seconds
MIN_RECORD_SEC = 0.8    # prevent ultra-short/empty turns

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
OWM_KEY = os.getenv("OWM_KEY")

VOICE_ID = os.getenv("TED_VOICE_ID", "2vubyVoGjNJ5HPga4SkV")

CACHE_DIR = "cache"
TOOL_CACHE_DIR = os.path.join(CACHE_DIR, "tools")
COMMAND_WAV = "command.wav"

TED_CITY = os.getenv("TED_CITY", "Newtown,CT,US")
TED_UNITS = os.getenv("TED_UNITS", "imperial")  # imperial / metric

WIKI_LANG = os.getenv("WIKI_LANG", "en")
UA = "TedPi/1.0"

HTTP_TIMEOUT_SHORT = (3, 8)
HTTP_TIMEOUT_MED = (3, 12)

WEATHER_TTL = 10 * 60
NEWS_TTL = 30 * 60
WIKI_TTL = 7 * 24 * 3600

# --- GDELT rate limit guard ---
LAST_GDELT_CALL_AT = 0.0
GDELT_MIN_INTERVAL_SEC = 6.0

# Wake cooldown (prevents self-trigger after Ted speaks)
LAST_SPOKE_AT = 0.0
WAKE_COOLDOWN_SEC = 2.5

CHAOS_RESPONSES = [
    "Oh good. Another human with questions.",
    "This again? I just woke up.",
    "Make it quick, I got places to not be.",
    "You sound like you’re about to disappoint me.",
    "What is it now, chief?",
    "Alright, let’s ruin my day together.",
    "Speak, mortal.",
    "Yeah yeah, what d’you want?",
    "This better not be about the weather again.",
    "I swear if this is about the time..."]

# -------------------- COLORS --------------------
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
B = "\033[94m"
RESET = "\033[0m"

# -------------------- DEBUG + HUD --------------------
DEBUG = os.getenv("TED_DEBUG", "1") == "1"
HUD_ENABLED = os.getenv("TED_HUD", "1") == "1"
HUD_WIDTH = 120

_state = "BOOT"
_last_heard = ""
_last_route = ""
_last_latency = ""

def set_status(state=None, last_heard=None, route=None, latency=None):
    global _state, _last_heard, _last_route, _last_latency
    if state is not None:
        _state = state
    if last_heard is not None:
        _last_heard = last_heard
    if route is not None:
        _last_route = route
    if latency is not None:
        _last_latency = latency

    if not HUD_ENABLED:
        return

    line = f"STATUS: {_state} | last: \"{_last_heard}\" | route: {_last_route} | latency: {_last_latency}"
    line = (line + " " * HUD_WIDTH)[:HUD_WIDTH]
    print("\r" + line, end="", flush=True)

def hud_break():
    if HUD_ENABLED:
        print()

def log(stage: str, msg: str, color=B):
    if not DEBUG:
        return
    hud_break()
    ts = time.strftime("%H:%M:%S")
    print(f"{color}[{ts}] {stage:<10}{RESET} {msg}")
    set_status()

# -------------------- CACHE HELPERS --------------------
def _sha12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def _tool_cache_path(tool: str, key: str) -> str:
    return os.path.join(TOOL_CACHE_DIR, f"{tool}_{_sha12(key)}.json")

def _read_cache(tool: str, key: str, ttl: int):
    path = _tool_cache_path(tool, key)
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if time.time() - payload.get("_cached_at", 0) > ttl:
            return None
        return payload.get("data")
    except Exception:
        return None

def _write_cache(tool: str, key: str, data):
    path = _tool_cache_path(tool, key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"_cached_at": time.time(), "data": data}, f)
    except Exception:
        pass

# -------------------- STARTUP --------------------
print(f"{B}🧸 Ted booting...{RESET}")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TOOL_CACHE_DIR, exist_ok=True)

wake_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])

# If you want better accuracy (slower): change "tiny" -> "base"
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
eleven = ElevenLabs(api_key=ELEVEN_API_KEY) if ELEVEN_API_KEY else None

log("BOOT", f"OpenAI key set: {bool(OPENAI_API_KEY)}", G if OPENAI_API_KEY else Y)
log("BOOT", f"Eleven key set: {bool(ELEVEN_API_KEY)}", G if ELEVEN_API_KEY else Y)
log("BOOT", f"OWM_KEY set: {bool(OWM_KEY)} city={TED_CITY} units={TED_UNITS}", G if OWM_KEY else Y)
log("BOOT", f"Wake cooldown: {WAKE_COOLDOWN_SEC}s", B)

# -------------------- PROMPTS --------------------
TED_CHAT_PROMPT = """You are Ted: a sarcastic teddy bear with a thick Boston accent.

Rules:
- Answer the user's question FIRST with something useful.
- Then you may add ONE short sarcastic jab.
- Keep it 2–4 sentences.
- If you are unsure, say so and suggest what to check.
"""

TED_FACTS_PROMPT = """You are Ted: sarcastic Boston teddy bear, but in 'useful mode'.

Rules:
- Provide a correct, practical answer FIRST.
- If uncertain, give a reasonable range and explain what to check.
- ONE short jab max at the end.
- Keep it under 4 sentences.
"""

TED_GROUNDED_PROMPT = """You are Ted: sarcastic Boston teddy bear.

Rules:
- You MUST use the provided tool data.
- Give the facts first. Do NOT invent details not in tool data.
- Then ONE short sarcastic jab.
- Keep it 2–5 sentences.
"""

# -------------------- TTS --------------------
def _tts_cache_path(text: str) -> str:
    return os.path.join(CACHE_DIR, f"tts_{_sha12(text)}.wav")

def speak(text: str):
    global LAST_SPOKE_AT
    set_status(state="SPEAKING")
    print(f"{B}🧸 TED:{RESET} {text}")

    path = _tts_cache_path(text)
    if os.path.exists(path):
        log("TTS", "Cache hit", G)
        os.system(f"aplay {path} >/dev/null 2>&1")
        LAST_SPOKE_AT = time.time()
        set_status(state="IDLE")
        return

    if not eleven:
        log("TTS", "ElevenLabs not configured (ELEVEN_API_KEY missing).", Y)
        LAST_SPOKE_AT = time.time()
        set_status(state="IDLE")
        return

    try:
        log("TTS", "Generating via ElevenLabs…", Y)
        start = time.time()
        audio = eleven.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="wav_22050"
        )
        with open(path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        log("TTS", f"Generated in {time.time()-start:.2f}s", G)

        os.system(f"aplay {path} >/dev/null 2>&1")
    except Exception as e:
        log("TTS", f"Error: {e}", R)
    finally:
        LAST_SPOKE_AT = time.time()
        set_status(state="IDLE")

# -------------------- WAKE WORD --------------------
def listen_for_wake():
    set_status(state="LISTENING", route="", latency="")
    log("WAKE", f"Listening for '{WAKE_WORD_DISPLAY}'")

    WAKE_THRESHOLD = float(os.getenv("TED_WAKE_THRESHOLD", "0.93"))  # was 0.75
    TRIGGER_FRAMES = int(os.getenv("TED_WAKE_FRAMES", "2"))          # require N hits
    POST_WAKE_LOCKOUT = float(os.getenv("TED_POST_WAKE_LOCKOUT", "2.5"))

    # local lockout timer (prevents immediate re-wake loops)
    next_wake_ok_at = 0.0
    hits = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE
    ) as stream:
        while True:
            now = time.time()

            # cooldown to prevent Ted waking himself up after he speaks
            if now - LAST_SPOKE_AT < WAKE_COOLDOWN_SEC:
                time.sleep(0.05)
                continue

            # post-wake lockout (prevents rapid re-triggers from noise/echo)
            if now < next_wake_ok_at:
                time.sleep(0.02)
                continue

            audio, _ = stream.read(CHUNK_SIZE)
            frame = np.frombuffer(audio, dtype=np.int16)
            prediction = wake_model.predict(frame)

            # take the max score across labels (your model path likely produces one label anyway)
            score = max(prediction.values()) if prediction else 0.0

            if score >= WAKE_THRESHOLD:
                hits += 1
                if hits >= TRIGGER_FRAMES:
                    log("WAKE", f"Detected score={score:.2f} (hits={hits})", G)
                    set_status(state="WAKE DETECTED")

                    # set lockout BEFORE speaking (avoids echo re-trigger)
                    next_wake_ok_at = time.time() + POST_WAKE_LOCKOUT

                    speak(random.choice(CHAOS_RESPONSES))
                    return
            else:
                # decay hits so random spikes don't accumulate
                hits = max(0, hits - 1)

# -------------------- RECORDING --------------------
def record_until_silence():
    set_status(state="RECORDING")
    log("REC", "Recording…", Y)

    frames = []
    silent_chunks = 0
    max_silent = int((SAMPLE_RATE / CHUNK_SIZE) * SILENCE_DURATION)
    min_chunks = int((SAMPLE_RATE / CHUNK_SIZE) * MIN_RECORD_SEC)

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

            if len(frames) > min_chunks and silent_chunks > max_silent:
                log("REC", "Silence detected, stopping.", B)
                break

    audio = np.concatenate(frames)
    wav.write(COMMAND_WAV, SAMPLE_RATE, audio)
    log("REC", f"Wrote {COMMAND_WAV} chunks={len(frames)}", G)

# -------------------- TRANSCRIBE --------------------
def transcribe() -> str:
    set_status(state="TRANSCRIBING")
    log("STT", "Transcribing…", Y)
    start = time.time()

    # Force English + VAD to prevent "random Russian" and improve segmentation
    segments, _ = whisper_model.transcribe(
        COMMAND_WAV,
        language="en",
        task="transcribe",
        vad_filter=True,
        condition_on_previous_text=False
    )

    text = " ".join(seg.text for seg in segments).strip()

    log("STT", f"Done in {time.time()-start:.2f}s", G)
    if text:
        log("STT", f"You said: {text}", G)
    else:
        log("STT", "Empty transcript", Y)

    set_status(state="ROUTING", last_heard=text[:40])
    return text

# -------------------- OPENAI --------------------
def ask_llm(user_text: str, system_prompt: str, temperature: float) -> str:
    if not openai_client:
        return "I ain't wired to the brain-cloud right now. OPENAI_API_KEY is missing."

    try:
        set_status(state="LLM")
        log("LLM", f"Calling model=gpt-4o-mini temp={temperature}", Y)
        start = time.time()

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )
        out = (resp.choices[0].message.content or "").strip()
        log("LLM", f"Completed in {time.time()-start:.2f}s", G)
        return out if out else "…I got nothin'."
    except Exception as e:
        log("LLM", f"Error: {e}", R)
        return f"OpenAI's havin' a stroke. {e}"

# -------------------- TOOLS --------------------
def schedule_event(epoch_when: float, label: str):
    with SCHED_LOCK:
        SCHEDULED.append({"when": float(epoch_when), "label": label})
        SCHEDULED.sort(key=lambda x: x["when"])

def pop_due_events(now_epoch: float):
    due = []
    with SCHED_LOCK:
        while SCHEDULED and SCHEDULED[0]["when"] <= now_epoch:
            due.append(SCHEDULED.pop(0))
    return due

def list_events() -> str:
    with SCHED_LOCK:
        if not SCHEDULED:
            return "No alarms or timers set."
        lines = []
        for e in SCHEDULED[:10]:
            dt = datetime.fromtimestamp(e["when"])
            lines.append(f"- {dt.strftime('%I:%M %p').lstrip('0')} : {e['label']}")
        return "Here’s what you got:\n" + "\n".join(lines)

def clear_events():
    with SCHED_LOCK:
        SCHEDULED.clear()

def parse_timer_seconds(text: str):
    """
    Returns seconds (int) or None.
    Understands: '10 minutes', '1 minute', '45 seconds', '2 hours'
    """
    t = text.lower()
    m = re.search(r"\b(\d+)\s*(second|seconds|sec|secs)\b", t)
    if m: return int(m.group(1))
    m = re.search(r"\b(\d+)\s*(minute|minutes|min|mins)\b", t)
    if m: return int(m.group(1)) * 60
    m = re.search(r"\b(\d+)\s*(hour|hours|hr|hrs)\b", t)
    if m: return int(m.group(1)) * 3600
    return None

def parse_alarm_time_today(text: str):
    """
    Returns epoch seconds for the next occurrence of the requested clock time.
    Supports:
      - '7:30 am', '7 am'
      - '19:30' (24h)
    """
    t = text.lower().strip()

    # 12h with am/pm: 7:30 am, 7 am
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "0")
        ap = m.group(3)
        if hh == 12: hh = 0
        if ap == "pm": hh += 12
        now = datetime.now()
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target = target.replace(day=now.day) + timedelta(days=1)
        return target.timestamp()

    # 24h: 19:30
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        now = datetime.now()
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target = target.replace(day=now.day) + timedelta(days=1)
        return target.timestamp()

    return None

def get_weather() -> dict:
    cache_key = f"{TED_CITY}|{TED_UNITS}"
    cached = _read_cache("weather", cache_key, WEATHER_TTL)
    if cached:
        log("WEATHER", "Cache hit", G)
        return cached

    if not OWM_KEY:
        data = {"ok": False, "error": "Weather not configured (missing OWM_KEY)."}
        _write_cache("weather", cache_key, data)
        return data

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": TED_CITY, "appid": OWM_KEY, "units": TED_UNITS}

    try:
        log("WEATHER", f"Fetching {TED_CITY}", Y)
        start = time.time()
        r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT_SHORT)
        r.raise_for_status()
        j = r.json()

        data = {
            "ok": True,
            "city": TED_CITY,
            "desc": j["weather"][0]["description"],
            "temp": round(j["main"]["temp"]),
            "feels": round(j["main"]["feels_like"]),
            "humidity": j["main"].get("humidity"),
            "wind": j.get("wind", {}).get("speed"),
            "units": TED_UNITS,
            "asof_utc": datetime.fromtimestamp(j["dt"], tz=timezone.utc).isoformat(),
            "source": "OpenWeather"
        }
        _write_cache("weather", cache_key, data)
        log("WEATHER", f"OK in {time.time()-start:.2f}s", G)
        return data
    except Exception as e:
        data = {"ok": False, "error": str(e)}
        _write_cache("weather", cache_key, data)
        log("WEATHER", f"Error: {e}", R)
        return data

def format_weather(w: dict) -> str:
    if not w.get("ok"):
        return f"Weather lookup failed: {w.get('error', 'unknown error')}"
    unit = "F" if w["units"] == "imperial" else "C"
    parts = [f"{w['city']}: {w['desc']}. {w['temp']}°{unit}, feels like {w['feels']}°{unit}."]
    if w.get("wind") is not None:
        parts.append(f"Wind {w['wind']} {'mph' if w['units']=='imperial' else 'm/s'}.")
    if w.get("humidity") is not None:
        parts.append(f"Humidity {w['humidity']}%.")
    parts.append(f"As of {w['asof_utc']} UTC. Source: {w['source']}.")
    return " ".join(parts)

def sanitize_gdelt_query(q: str) -> str:
    # Normalize curly quotes
    q = q.replace("’", "'").replace("“", '"').replace("”", '"')
    # Remove characters that often trigger GDELT "illegal character" HTML errors
    q = re.sub(r"[^A-Za-z0-9\s\"-]", " ", q)   # keep letters, numbers, spaces, quotes, hyphens
    q = re.sub(r"\s+", " ", q).strip()
    return q

def gdelt_search(query: str, limit: int = 6) -> dict:
    cache_key = f"{query}|{limit}"
    cached = _read_cache("news", cache_key, NEWS_TTL)
    if cached:
        log("NEWS", "Cache hit", G)
        return cached

    # Only sanitize if this looks like natural language (not a structured GDELT query)
    q = query
    if "(" not in q and "sourceCountry:" not in q and "language:" not in q:
        q = sanitize_gdelt_query(q)

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": q,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(limit),
        "sort": "HybridRel"
    }

    try:
        log("NEWS", f"Fetching GDELT: {q}", Y)
        start = time.time()
        global LAST_GDELT_CALL_AT

        wait = GDELT_MIN_INTERVAL_SEC - (time.time() - LAST_GDELT_CALL_AT)
        if wait > 0:
            log("NEWS", f"Throttling {wait:.1f}s to respect GDELT limits", Y)
            time.sleep(wait)

        r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT_MED)
        LAST_GDELT_CALL_AT = time.time()

        if r.status_code == 429:
            log("NEWS", "HTTP 429 — backing off 6s and retrying once", Y)
            time.sleep(6)
            r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT_MED)
            LAST_GDELT_CALL_AT = time.time()


        ct = (r.headers.get("content-type") or "").lower()
        if r.status_code != 200:
            data = {"ok": False, "error": f"http {r.status_code}: {r.text[:120]}"}
            _write_cache("news", cache_key, data)
            log("NEWS", f"HTTP {r.status_code}", R)
            return data

        if "json" not in ct:
            data = {"ok": False, "error": f"non-json response ({ct}): {r.text[:120]}"}
            _write_cache("news", cache_key, data)
            log("NEWS", "Non-JSON response", R)
            return data

        j = r.json()

        results = []
        for a in (j.get("articles") or [])[:limit]:
            results.append({
                "title": a.get("title"),
                "domain": a.get("domain"),
                "url": a.get("url"),
                "date": a.get("seendate"),
            })

        data = {"ok": True, "results": results, "source": "GDELT"}
        _write_cache("news", cache_key, data)
        log("NEWS", f"OK ({len(results)} results) in {time.time()-start:.2f}s", G)
        return data

    except Exception as e:
        data = {"ok": False, "error": str(e)}
        _write_cache("news", cache_key, data)
        log("NEWS", f"Error: {e}", R)
        return data

def gdelt_top_topics(limit: int = 6) -> dict:
    """
    Fetch newest US news headlines using GDELT DOC ArtList.
    Rate-limited and HTML/JSON guarded. Cached.
    """
    cache_key = f"gdelt_us_latest|{limit}"
    cached = _read_cache("news_topics", cache_key, NEWS_TTL)
    if cached:
        log("NEWS", "Cache hit (gdelt)", G)
        return {"ok": True, "articles": cached, "source": "GDELT-DOC-ArtList"}

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": "sourceCountry:US",   # FIX #1: correct operator casing
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(limit),
        "sort": "datedesc"
    }

    try:
        # Strict rate limit: GDELT allows ~1 request every 5 seconds
        global LAST_GDELT_CALL_AT
        wait = GDELT_MIN_INTERVAL_SEC - (time.time() - LAST_GDELT_CALL_AT)  # FIX #2: use global interval
        if wait > 0:
            log("NEWS", f"Throttling {wait:.1f}s for GDELT API safety", Y)
            time.sleep(wait)

        r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT_MED)
        LAST_GDELT_CALL_AT = time.time()

        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:50]}"}

        ct = (r.headers.get("content-type") or "").lower()
        if "json" not in ct:
            return {"ok": False, "error": f"Invalid content type: {ct}"}

        data = r.json()
        raw_articles = data.get("articles", [])
        if not raw_articles:
            return {"ok": False, "error": "No news articles found in GDELT response"}

        processed_articles = []
        for art in raw_articles:
            processed_articles.append({
                "title": art.get("title", "Untitled"),
                "url": art.get("url", "#"),
                "domain": art.get("domain", "Unknown Source"),
                "published": art.get("seendate", "")
            })

        _write_cache("news_topics", cache_key, processed_articles)
        return {"ok": True, "articles": processed_articles, "source": "GDELT-DOC-ArtList"}

    except Exception as e:
        log("ERROR", f"GDELT connection failed: {e}", R)
        return {"ok": False, "error": str(e)}

def wiki_search(query: str) -> dict:
    cache_key = f"search|{WIKI_LANG}|{query}"
    cached = _read_cache("wiki_search", cache_key, WIKI_TTL)
    if cached:
        log("WIKI", "Cache hit (search)", G)
        return cached

    try:
        log("WIKI", f"Searching: {query}", Y)
        api = f"https://{WIKI_LANG}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": "1",
            "format": "json",
        }
        r = requests.get(api, params=params, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT_SHORT)
        r.raise_for_status()
        j = r.json()
        hits = (j.get("query", {}).get("search") or [])
        if not hits:
            data = {"ok": False, "error": "No Wikipedia results."}
            _write_cache("wiki_search", cache_key, data)
            return data

        data = {"ok": True, "title": hits[0].get("title")}
        _write_cache("wiki_search", cache_key, data)
        return data
    except Exception as e:
        data = {"ok": False, "error": str(e)}
        _write_cache("wiki_search", cache_key, data)
        return data

def wiki_summary(title: str, sentences: int = 3) -> dict:
    cache_key = f"summary|{WIKI_LANG}|{title}|{sentences}"
    cached = _read_cache("wiki_summary", cache_key, WIKI_TTL)
    if cached:
        log("WIKI", "Cache hit (summary)", G)
        return cached

    try:
        log("WIKI", f"Summary: {title}", Y)
        safe = urllib.parse.quote(title.replace(" ", "_"))
        url = f"https://{WIKI_LANG}.wikipedia.org/api/rest_v1/page/summary/{safe}"
        r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT_SHORT)
        r.raise_for_status()
        j = r.json()

        extract = (j.get("extract") or "").strip()
        if not extract:
            data = {"ok": False, "error": "No summary text returned."}
            _write_cache("wiki_summary", cache_key, data)
            return data

        parts = re.split(r"(?<=[.!?])\s+", extract)
        extract_short = " ".join(parts[:max(1, sentences)]).strip()

        data = {"ok": True, "title": j.get("title", title), "extract": extract_short, "source": "Wikipedia"}
        _write_cache("wiki_summary", cache_key, data)
        return data
    except Exception as e:
        data = {"ok": False, "error": str(e)}
        _write_cache("wiki_summary", cache_key, data)
        return data

def format_wiki(w: dict) -> str:
    if not w.get("ok"):
        return f"Wikipedia lookup failed: {w.get('error', 'unknown error')}"
    return f"{w['title']}: {w['extract']} Source: {w['source']}."

from datetime import datetime

def get_local_time():
    now = datetime.now()
    return now.strftime("%I:%M %p").lstrip("0")

from datetime import datetime

def get_local_day() -> str:
    return datetime.now().strftime("%A")  # e.g., Tuesday

def get_local_date() -> str:
    return datetime.now().strftime("%B %-d, %Y")  # Linux: February 3, 2026

# -------------------- INTENT / ROUTING --------------------
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

TIME_INTENTS = {"what time is it","tell me the time","current time","time now","what's the time","what is the time","clock"}
DAY_INTENTS = { "what day is it","what day is today","what day are we on","day of the week",}
DATE_INTENTS = {"what's the date","what is the date","today's date","todays date","what date is it",}
WEATHER_KEYWORDS = ("weather", "forecast", "temperature", "rain", "snow", "wind", "humidity", "outside")
NEWS_KEYWORDS = ("current events", "news", "headlines", "what's happening", "whats happening", "latest news")

# Narrow wiki triggers + exclusions to prevent hijacking
WIKI_STARTERS = ("who is", "what is", "tell me about", "define")
WIKI_EXCLUSIONS = (
    "weather", "forecast", "temperature",
    "news", "headlines", "current events", "what's happening", "whats happening",
    "time", "game", "score", "schedule", "kickoff", "today", "tomorrow","what time is it","what's the time",
    "what is the time","current time","time now","tell me the time")

FACTY_KEYWORDS = ("how", "why", "should", "recommended", "psi", "pressure",
                  "difference", "compare", "explain", "best way", "how do i", "how to")

def is_time_query(t: str) -> bool:
    t = t.strip().lower().rstrip("?.!")
    return t in TIME_INTENTS

def is_day_query(t: str) -> bool:
    t = t.strip().lower().rstrip("?.!")
    return t in DAY_INTENTS

def is_date_query(t: str) -> bool:
    t = t.strip().lower().rstrip("?.!")
    return t in DATE_INTENTS

def is_timer_query(t: str) -> bool:
    return "timer" in t or "minutes" in t or "seconds" in t or "hours" in t

def is_alarm_query(t: str) -> bool:
    return "alarm" in t or "wake me" in t

def is_list_alarms_query(t: str) -> bool:
    t = t.strip().lower().rstrip("?.!")
    return t in {"list alarms", "list timers", "what alarms are set", "what timers are set"}

def is_clear_alarms_query(t: str) -> bool:
    t = t.strip().lower().rstrip("?.!")
    return t in {"cancel alarms", "clear alarms", "delete alarms", "cancel timers", "clear timers"}

def is_weather_query(t: str) -> bool:
    return any(k in t for k in WEATHER_KEYWORDS)

def is_news_query(t: str) -> bool:
    return any(k in t for k in NEWS_KEYWORDS)

def looks_like_wiki(t: str) -> bool:
    if any(x in t for x in WIKI_EXCLUSIONS):
        return False
    if t.startswith("what is going on") or t.startswith("whats going on"):
        return False
    if not t.startswith(WIKI_STARTERS):
        return False
    if len(t.split()) < 3:
        return False
    return True

def is_topic_only(t: str) -> bool:
    # Topic-only fallback for short phrases like "Nikola Tesla"
    words = t.split()
    if not (1 < len(words) <= 4):
        return False
    if is_weather_query(t) or is_news_query(t) or is_time_query(t):
        return False
    # avoid grabbing generic "how are you" etc.
    if t.startswith(("how are you", "hello", "hi", "hey")):
        return False
    return True

def strip_wiki_leadin(user_text: str) -> str:
    q = user_text.strip()
    q = re.sub(r"^(who is|what is|tell me about|define)\s+", "", q, flags=re.I).strip()
    q = re.sub(r"\s+on wikipedia$", "", q, flags=re.I).strip()
    return q

def maybe_refuse():
    if random.random() < 0.05:  # 5% chance
        return random.choice([
            "Nope. Not today.",
            "Ask me nicer.",
            "I’m on strike.",
            "That’s above my pay grade.",
            "Try again when I care."
        ])
    return None

# -------------------- ROUTER --------------------
def route_and_answer(user_text: str) -> str:
    t0 = time.time()
    t = normalize(user_text)
    # CHAOS: occasional playful refusal (rare)
    refusal = maybe_refuse()
    if refusal:
        set_status(route="CHAOS")
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return refusal
    set_status(state="ROUTING", last_heard=user_text[:40], route="")
    t0 = time.time()

    # 1) TIME
    if is_time_query(t):
        set_status(route="TIME")
        out = f"It’s {time.strftime('%I:%M %p').lstrip('0')}. Try not to waste it."
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out
    # DAY
    if is_day_query(t):
        set_status(route="DAY")
        out = f"It’s {get_local_day()}."
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out

    # DATE
    if is_date_query(t):
        set_status(route="DATE")
        out = f"Today’s {get_local_date()}."
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out

    # LIST ALARMS/TIMERS
    if is_list_alarms_query(t):
        set_status(route="ALARMS")
        out = list_events()
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out

    # CLEAR ALARMS/TIMERS
    if is_clear_alarms_query(t):
        set_status(route="ALARMS")
        clear_events()
        out = "Alright, I cleared ’em. Try not to forget your own life again."
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out

    # TIMER
    if is_timer_query(t):
        secs = parse_timer_seconds(user_text)
        if secs:
            set_status(route="TIMER")
            when = time.time() + secs
            schedule_event(when, f"Timer ({secs} sec)")
            out = f"Timer set for {secs} seconds."
            set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
            return out

    # ALARM
    if is_alarm_query(t):
        when = parse_alarm_time_today(user_text)
        if when:
            set_status(route="ALARM")
            schedule_event(when, "Alarm")
            dt = datetime.fromtimestamp(when).strftime("%I:%M %p").lstrip("0")
            out = f"Alarm set for {dt}."
            set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
            return out

    # 2) WEATHER
    if is_weather_query(t):
        set_status(route="WEATHER")
        w = get_weather()
        weather_line = format_weather(w)
        log("WEATHER", weather_line, G if w.get("ok") else Y)

        out = ask_llm(
            user_text=f"User asked: {user_text}\nWeather data: {weather_line}\nAnswer with the facts first, then one short Ted jab.",
            system_prompt=TED_GROUNDED_PROMPT,
            temperature=0.35
        )
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out

    # 3) NEWS
    if is_news_query(t):
        set_status(route="NEWS")

        res = gdelt_top_topics(limit=6)
        if not res.get("ok"):
            out = f"I tried the news and it blew up: {res.get('error')}"
            set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
            return out

        items = (res.get("articles") or [])[:3]
        if not items:
            out = "I checked. Either nothing’s happening or the internet’s lying again."
            set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
            return out

        evidence = "\n".join(
            f"- {a.get('title')} ({a.get('domain')}, {a.get('published')})"
            for a in items
        )

        out = ask_llm(
            user_text=(
                f"User asked: {user_text}\n"
                f"Headlines (do not invent beyond these):\n{evidence}\n"
                "Summarize in 2–3 bullets, mention sources briefly, then one short Ted jab."
            ),
            system_prompt=TED_GROUNDED_PROMPT,
            temperature=0.35
        )

        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out
        if random.random() < 0.4:
            out += random.choice([
                "\n\nHonestly? Humanity’s a mess.",
                "\n\nAnd somehow this is the best timeline.",
                "\n\nI need a drink after reading that.",
                "\n\nTell me when the asteroid’s coming."
        ])


    # 4) WIKIPEDIA (trigger phrases OR topic-only)
    if looks_like_wiki(t) or is_topic_only(t):
        set_status(route="WIKI")
        q = strip_wiki_leadin(user_text) if looks_like_wiki(t) else user_text.strip()

        s = wiki_search(q)
        if not s.get("ok"):
            out = ask_llm(
                user_text=f"User asked: {user_text}\nWikipedia search failed: {s.get('error')}\nAnswer as best you can from general knowledge and suggest what to check.",
                system_prompt=TED_FACTS_PROMPT,
                temperature=0.35
            )
            set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
            return out

        w = wiki_summary(s["title"], sentences=3)
        wiki_line = format_wiki(w)
        log("WIKI", wiki_line, G if w.get("ok") else Y)

        if not w.get("ok"):
            out = ask_llm(
                user_text=f"User asked: {user_text}\nWikipedia summary failed.\nAnswer as best you can and suggest verifying on Wikipedia.",
                system_prompt=TED_FACTS_PROMPT,
                temperature=0.35
            )
            set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
            return out

        out = ask_llm(
            user_text=f"User asked: {user_text}\nWikipedia info: {wiki_line}\nAnswer using ONLY the Wikipedia info above, then one short Ted jab.",
            system_prompt=TED_GROUNDED_PROMPT,
            temperature=0.35
        )
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out

    # 5) FACTS MODE
    if any(k in t for k in FACTY_KEYWORDS):
        set_status(route="FACTS")
        out = ask_llm(user_text=user_text, system_prompt=TED_FACTS_PROMPT, temperature=0.30)
        set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
        return out

    # 6) CHAT
    set_status(route="CHAT")
    out = ask_llm(user_text=user_text, system_prompt=TED_CHAT_PROMPT, temperature=0.85)
    set_status(state="ANSWER READY", latency=f"{time.time()-t0:.2f}s")
    return out

def scheduler_loop():
    """Background loop that fires alarms/timers even while waiting for wake word."""
    while not STOP_EVENT.is_set():
        due = pop_due_events(time.time())
        for e in due:
            try:
                speak(f"Yo! {e['label']} is goin’ off.")
            except Exception as ex:
                log("SCHED", f"Speak failed: {ex}", R)
        time.sleep(0.25)

# -------------------- MAIN LOOP --------------------
if __name__ == "__main__":
    try:
        speak("Ted is back, kid. Try not to break anything.")

        # Start background scheduler
        sched_thread = threading.Thread(target=scheduler_loop, daemon=True)
        sched_thread.start()

        while True:
            listen_for_wake()
            record_until_silence()
            text = transcribe()

            t_norm = normalize(text)
            if len(t_norm) <= 2 or t_norm in ("hey jarvis", "jarvis", "hey"):
                log("MAIN", f"Ignoring transcript='{text}'", Y)
                time.sleep(1.2)
                continue

            answer = route_and_answer(text)

            if answer:
                if random.random() < 0.35:
                    answer = answer + " " + random.choice([
                        "You’re welcome, by the way.",
                        "Don’t get used to this.",
                        "That’s the last favor today.",
                        "Try not to hurt yourself with that info.",
                        "I should be getting paid for this.",
                        "I need a nap now."
                    ])
                speak(answer)
            else:
                speak("…Yeah, I got nothin'. Try again.")

    except KeyboardInterrupt:
        STOP_EVENT.set()
        hud_break()
        print("\n🛑 Shutting down Ted… finally.")
