"""Optional API + named voice-worker supervisor for the immutable VPS image."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

REGISTRATION_FILE = Path("/tmp/dentnode-voice-registered.json")
SHUTDOWN_SECONDS = 350


def voice_enabled() -> bool:
    value = os.environ.get("VOICE_WORKER_ENABLED", "false").strip().lower()
    if value not in ("true", "false"):
        raise ValueError("VOICE_WORKER_ENABLED must be true or false")
    return value == "true"


def validate_voice_config() -> None:
    port = int(os.environ.get("PORT", "8080"))
    if not 1 <= port <= 65535 or port == 8081:
        raise ValueError("API PORT must be valid and distinct from voice health port 8081")
    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "CALLING_SERVICE_URL"):
        if not os.environ.get(key, "").strip():
            raise ValueError(f"{key} is required when VOICE_WORKER_ENABLED=true")
    livekit = urlsplit(os.environ["LIVEKIT_URL"])
    callback = urlsplit(os.environ["CALLING_SERVICE_URL"])
    if livekit.scheme != "wss" or not livekit.hostname or livekit.username or livekit.password:
        raise ValueError("Supervised voice worker requires a wss LIVEKIT_URL")
    if (callback.scheme != "https" or not callback.hostname or callback.username
            or callback.password or callback.query or callback.fragment or callback.path not in ("", "/")):
        raise ValueError("CALLING_SERVICE_URL must be an HTTPS origin")
    if os.environ.get("LIVEKIT_AGENT_NAME") != "dentnode-receptionist":
        raise ValueError("Supervised worker requires LIVEKIT_AGENT_NAME=dentnode-receptionist")
    warm = int(os.environ.get("VOICE_WORKER_IDLE_PROCESSES", "1"))
    if not 0 <= warm <= 2:
        raise ValueError("VOICE_WORKER_IDLE_PROCESSES must be between 0 and 2")


def mark_voice_registered(*_args) -> None:
    # Called only by the SDK's public worker_registered event, never by /health.
    temporary = REGISTRATION_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "agent": "dentnode-receptionist"}))
    temporary.replace(REGISTRATION_FILE)


def health() -> bool:
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{int(os.environ.get('PORT', '8080'))}/health", timeout=2) as response:
            if response.status != 200:
                return False
        if voice_enabled():
            marker = json.loads(REGISTRATION_FILE.read_text())
            if marker.get("agent") != "dentnode-receptionist":
                return False
            os.kill(int(marker["pid"]), 0)
            with opener.open("http://127.0.0.1:8081/", timeout=1) as response:
                return response.status == 200
        return True
    except Exception:
        return False


def _signal_group(process, sig) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _wait_until(process, deadline) -> None:
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)


def main() -> int:
    if sys.argv[1:] == ["--health"]:
        return 0 if health() else 1
    api_command = [sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", os.environ.get("PORT", "8080")]
    if not voice_enabled():
        os.execv(sys.executable, api_command)
    validate_voice_config()
    REGISTRATION_FILE.unlink(missing_ok=True)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    api = worker = None
    failed = False
    try:
        api = subprocess.Popen(api_command, start_new_session=True)
        worker = subprocess.Popen([sys.executable, "-m", "livekit.agents", "start", "voice_call/worker.py"], start_new_session=True)
        startup_deadline = time.monotonic() + 90
        ready = False
        health_failures = 0
        while not stopping:
            if api.poll() is not None or worker.poll() is not None:
                failed = True
                print("[runtime] A required process exited; restarting the container.", flush=True)
                break
            healthy = health()
            if healthy:
                ready = True
                health_failures = 0
            elif ready:
                health_failures += 1
            if health_failures >= 3 or (not ready and time.monotonic() >= startup_deadline):
                failed = True
                print("[runtime] API/registered worker health failed; restarting the container.", flush=True)
                break
            time.sleep(1)
    finally:
        REGISTRATION_FILE.unlink(missing_ok=True)
        deadline = time.monotonic() + SHUTDOWN_SECONDS
        # Keep the API alive while active voice jobs flush their metering.
        if worker:
            # The SDK owns its job processes and drains them before stopping.
            # Signalling the whole group here would kill active calls immediately.
            if worker.poll() is None:
                worker.send_signal(signal.SIGTERM)
            _wait_until(worker, min(deadline - 10, time.monotonic() + 340))
            _signal_group(worker, signal.SIGKILL)
            worker.wait(timeout=2)
        if api:
            _signal_group(api, signal.SIGTERM)
            _wait_until(api, deadline - 1)
            _signal_group(api, signal.SIGKILL)
            api.wait(timeout=1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
