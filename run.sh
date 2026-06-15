#!/bin/bash
# DreamVoice - keeps app running until Ctrl+C
cd "$(dirname "$0")"

cleanup() {
    echo ""
    echo "Stopping DreamVoice..."
    kill $APP_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    echo "========================================="
    echo "  DreamVoice running at http://localhost:7870"
    echo "  Press Ctrl+C to stop"
    echo "========================================="
    python3 app.py
    echo "App crashed. Restarting in 3 seconds..."
    sleep 3
done
