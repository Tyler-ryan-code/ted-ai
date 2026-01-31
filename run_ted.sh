#!/bin/bash

cd /home/tyler/ted

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Load .bashrc as backup (optional)
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# Check that keys exist
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set. Add it to .env or ~/.bashrc"
    read -p "Press ENTER to exit."
    exit 1
fi

if [ -z "$ELEVEN_API_KEY" ]; then
    echo "❌ ELEVEN_API_KEY not set. Add it to .env or ~/.bashrc"
    read -p "Press ENTER to exit."
    exit 1
fi

# Activate Python environment
source venv/bin/activate

# Run Ted and log all output
mkdir -p logs
LOGFILE="logs/ted_$(date +%Y%m%d_%H%M%S).log"
python ted_listener.py 2>&1 | tee "$LOGFILE"

echo ""
echo "🛑 Ted has stopped. Press ENTER to close this window."
read
