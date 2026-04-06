#!/usr/bin/env bash
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-99}"
SCREEN_RES="${SCREEN_RES:-1920x1080x24}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
SERVER_PORT="${SERVER_PORT:-4999}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"

export DISPLAY=":${DISPLAY_NUM}"
export XDG_SESSION_TYPE="x11"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

cleanup() {
    echo "Shutting down desktop stack..."
    kill $(jobs -p) 2>/dev/null || true
    wait
}
trap cleanup EXIT

if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "Display $DISPLAY already active, skipping Xvfb"
else
    echo "Starting Xvfb on $DISPLAY ($SCREEN_RES)"
    Xvfb "$DISPLAY" -screen 0 "$SCREEN_RES" &
    sleep 2
fi

echo "Starting Xfce4 desktop"
dbus-launch --exit-with-session startxfce4 &
sleep 3

echo "Starting x11vnc on port $VNC_PORT"
x11vnc -display "$DISPLAY" -forever -nopw -rfbport "$VNC_PORT" -shared &
sleep 1

NOVNC_WEB="/usr/share/novnc"
if [ ! -d "$NOVNC_WEB" ]; then
    NOVNC_WEB="/usr/share/noVNC"
fi

echo "Starting noVNC websockify on port $NOVNC_PORT"
websockify --web "$NOVNC_WEB" "$NOVNC_PORT" "localhost:${VNC_PORT}" &
sleep 1

echo "Starting local_desktop_server on port $SERVER_PORT"
DISPLAY="$DISPLAY" NOVNC_PORT="$NOVNC_PORT" PORT="$SERVER_PORT" \
    "$PYTHON_BIN" "$SCRIPT_DIR/local_desktop_server.py" &
sleep 1

echo "Desktop stack ready."
echo "  Open http://localhost:${SERVER_PORT}/ to view the desktop"
echo "  API:     http://localhost:${SERVER_PORT}/server/*"
echo "  DISPLAY=$DISPLAY"

wait
