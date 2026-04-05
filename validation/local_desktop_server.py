"""
Compatibility server that implements the same HTTP API as the remote
OSWorld orchestrator, but drives a local Xvfb desktop via pyautogui
and (optionally) the guest Flask server from GUI-Docker-Env.
"""

import os
os.environ.setdefault("XDG_SESSION_TYPE", "x11")

import io
import json
import logging
import subprocess
import sys
import threading
import time
import uuid

import requests
from flask import Flask, jsonify, redirect, request, send_file, send_from_directory

NOVNC_DIR = (
    "/usr/share/novnc" if os.path.isdir("/usr/share/novnc") else "/usr/share/noVNC"
)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("local_desktop_server")

NOVNC_WS_PORT = int(os.environ.get("NOVNC_PORT", 6080))


@app.route("/")
def index():
    return redirect("/vnc")


@app.route("/vnc")
def vnc_viewer():
    host = request.host.split(":")[0]
    return redirect(
        f"/novnc/vnc_lite.html?host={host}&port={NOVNC_WS_PORT}"
        "&path=websockify&autoconnect=true&resize=scale"
    )


@app.route("/novnc/<path:filename>")
def novnc_static(filename):
    return send_from_directory(NOVNC_DIR, filename)

DISPLAY = os.environ.get("DISPLAY", ":99")
GUEST_SERVER = os.environ.get("GUEST_SERVER", "")  # e.g. http://localhost:5000

sessions: dict[str, dict] = {}
sessions_lock = threading.Lock()

EVALUATOR_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "GUI-Docker-Env", "desktop_env"
)
sys.path.insert(0, os.path.join(EVALUATOR_ROOT, ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_setup_steps(config_steps: list[dict], service_id: str):
    """Run task config/setup steps either via guest server or locally."""
    for step in config_steps:
        step_type = step.get("type", "")
        params = step.get("parameters", {})
        log.info(f"[{service_id}] setup step: {step_type}")

        if step_type == "execute":
            _run_execute(params)
        elif step_type == "launch":
            _run_launch(params)
        elif step_type == "sleep":
            duration = params.get("seconds", params.get("duration", 5))
            time.sleep(float(duration))
        elif GUEST_SERVER:
            resp = requests.post(
                f"{GUEST_SERVER}/setup/{step_type}",
                json=params,
                timeout=300,
            )
            if resp.status_code != 200:
                log.warning(f"[{service_id}] guest setup/{step_type} failed: {resp.text}")
        else:
            log.warning(f"[{service_id}] unhandled setup type: {step_type}")


def _run_execute(params: dict):
    command = params.get("command", "")
    shell = params.get("shell", isinstance(command, str))
    env = {**os.environ, "DISPLAY": DISPLAY}
    try:
        subprocess.run(
            command, shell=shell, env=env, timeout=120,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except Exception as e:
        log.error(f"execute error: {e}")


def _run_launch(params: dict):
    command = params.get("command", "")
    shell = params.get("shell", isinstance(command, str))
    env = {**os.environ, "DISPLAY": DISPLAY}
    try:
        subprocess.Popen(command, shell=shell, env=env)
    except Exception as e:
        log.error(f"launch error: {e}")


def _take_screenshot() -> bytes:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    os.unlink(path)
    try:
        subprocess.run(
            ["scrot", "-z", path],
            env={**os.environ, "DISPLAY": DISPLAY},
            timeout=10, check=True,
        )
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _run_evaluation(evaluator: dict) -> float:
    """Run real evaluation using GUI-Docker-Env evaluator logic or return dummy."""
    try:
        from desktop_env.desktop_env import DesktopEnv
    except Exception:
        log.warning("Could not import DesktopEnv evaluator, returning 0.0")
        return 0.0

    try:
        env = DesktopEnv(
            action_space="pyautogui",
            headless=True,
        )
        env.evaluator = evaluator
        env.action_history = []
        result = env.evaluate()
        return float(result)
    except Exception as e:
        log.error(f"Evaluation error: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# API Endpoints  (contract expected by env_k8s.RemoteDesktopEnv)
# ---------------------------------------------------------------------------

@app.route("/server/getAvailableAndLock", methods=["POST"])
def get_available_and_lock():
    data = request.get_json(force=True, silent=True) or {}
    service_id = str(uuid.uuid4())[:12]
    task_id = data.get("task_id", "")
    config_steps = data.get("config", [])

    with sessions_lock:
        sessions[service_id] = {
            "task_id": task_id,
            "created": time.time(),
            "evaluator": None,
        }

    try:
        _run_setup_steps(config_steps, service_id)
    except Exception as e:
        log.error(f"[{service_id}] setup failed: {e}")

    return jsonify({"data": {"service_id": service_id, "server_id": service_id}})


@app.route("/server/create", methods=["POST"])
def create_env():
    service_id = str(uuid.uuid4())[:12]
    with sessions_lock:
        sessions[service_id] = {
            "task_id": "",
            "created": time.time(),
            "evaluator": None,
        }
    return jsonify({"data": {"service_id": service_id, "server_id": service_id}})


@app.route("/server/status/<service_id>", methods=["GET"])
def status(service_id):
    with sessions_lock:
        if service_id in sessions:
            return jsonify({"status": "ok", "service_id": service_id})
    return jsonify({"error": "not found"}), 404


@app.route("/server/list", methods=["GET"])
def list_envs():
    with sessions_lock:
        envs = [
            {"service_id": sid, "server_id": sid, "task_id": s.get("task_id", "")}
            for sid, s in sessions.items()
        ]
    return jsonify({"data": envs})


@app.route("/server/execute/<service_id>", methods=["POST"])
def execute(service_id):
    with sessions_lock:
        if service_id not in sessions:
            return jsonify({"error": "unknown session"}), 404

    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action", "")

    if action == "screenshot":
        try:
            png_bytes = _take_screenshot()
            return send_file(io.BytesIO(png_bytes), mimetype="image/png")
        except Exception as e:
            log.error(f"screenshot error: {e}")
            return jsonify({"error": str(e)}), 500

    if action in ("WAIT", "FAIL", "DONE"):
        return jsonify({"status": "ok", "action": action})

    if GUEST_SERVER:
        try:
            resp = requests.post(
                f"{GUEST_SERVER}/setup/execute",
                json={"command": ["python3", "-c", action], "shell": False},
                timeout=15,
            )
            return jsonify(resp.json()), resp.status_code
        except Exception as e:
            log.error(f"guest execute proxy error: {e}")

    env = {**os.environ, "DISPLAY": DISPLAY}
    try:
        subprocess.run(
            ["python3", "-c", action],
            env=env, timeout=15,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        log.error(f"execute error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/server/evaluate/<service_id>", methods=["POST"])
def evaluate(service_id):
    with sessions_lock:
        if service_id not in sessions:
            return jsonify({"error": "unknown session"}), 404

    evaluator = request.get_json(force=True, silent=True)
    if isinstance(evaluator, str):
        evaluator = json.loads(evaluator)

    result = _run_evaluation(evaluator)
    return jsonify({"data": {"result": result}})


@app.route("/server/release/<service_id>", methods=["POST"])
def release(service_id):
    with sessions_lock:
        removed = sessions.pop(service_id, None)
    if removed is not None:
        log.info(f"Released session {service_id}")
        return jsonify({"status": "ok", "service_id": service_id})
    return jsonify({"status": "ok", "service_id": service_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4999))
    log.info(f"Starting local desktop server on :{port} (DISPLAY={DISPLAY})")
    app.run(host="0.0.0.0", port=port, threaded=True)
