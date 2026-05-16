"""
Quadtrix web backend.

Run from the repository root:
    python server.py

Or directly:
    python backend/server.py
"""

from __future__ import annotations

import codecs
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Generator

USER_SITE = (
    Path.home()
    / "AppData"
    / "Roaming"
    / "Python"
    / f"Python{sys.version_info.major}{sys.version_info.minor}"
    / "site-packages"
)
if str(USER_SITE) not in sys.path:
    sys.path.append(str(USER_SITE))

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS


BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

EXE_PATH = Path(os.environ.get("QUADTRIX_EXE", ROOT_DIR / "Quadtrix.exe")).resolve()
DATA_PATH = Path(os.environ.get("QUADTRIX_DATA", ROOT_DIR / "data" / "input.txt")).resolve()
DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT_DIR / "best_model.bin")).resolve()

MODEL_MARKER = "Quadtrix>"
PROMPT_MARKERS = ("\r\n\r\nYou>", "\n\nYou>", "\r\rYou>")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
SAFE_MODEL_ROOTS = (ROOT_DIR, BACKEND_DIR)
MODEL_LOAD_TIMEOUT_SECONDS = 180
GENERATION_IDLE_TIMEOUT_SECONDS = 180

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

procs: dict[str, subprocess.Popen] = {}
procs_lock = threading.Lock()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _public_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)


def _discover_models() -> list[dict[str, str | bool]]:
    """Find practical .bin model locations without walking the whole repo."""
    search_dirs = [ROOT_DIR, ROOT_DIR / "models", BACKEND_DIR / "models"]
    seen: set[Path] = set()
    models: list[dict[str, str | bool]] = []

    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for model in sorted(directory.glob("*.bin")):
            resolved = model.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            models.append(
                {
                    "name": model.stem,
                    "path": _public_path(resolved),
                    "exists": resolved.exists(),
                }
            )

    if DEFAULT_MODEL_PATH not in seen:
        models.insert(
            0,
            {
                "name": DEFAULT_MODEL_PATH.stem or "best_model",
                "path": _public_path(DEFAULT_MODEL_PATH),
                "exists": DEFAULT_MODEL_PATH.exists(),
            },
        )

    return models


def _resolve_model(selection: str | None) -> tuple[Path | None, str | None]:
    raw = (selection or "").strip()
    candidate = Path(raw) if raw else DEFAULT_MODEL_PATH

    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate

    candidate = candidate.resolve()

    if candidate.suffix.lower() != ".bin":
        return None, "selected model must be a .bin file"

    if not any(_is_inside(candidate, root) for root in SAFE_MODEL_ROOTS):
        return None, "selected model must be inside this project"

    if not candidate.exists():
        return None, f"model not found: {candidate}"

    return candidate, None


def _drain_pipe(pipe) -> None:
    try:
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _split_ansi(text: str) -> tuple[str, str]:
    out: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        char = text[i]
        if char == "\x1b":
            match = ANSI_RE.match(text, i)
            if match:
                i = match.end()
                continue
            return "".join(out), text[i:]
        out.append(char)
        i += 1

    return "".join(out), ""


def _sse_char(char: str) -> str | None:
    if char == "\n":
        return "data: __NL__\n\n"
    if char == "\r":
        return None
    return f"data: {char}\n\n"


def _sse_event(name: str, data: str) -> str:
    return f"event: {name}\ndata: {data}\n\n"


def _ends_like_prompt_marker(text: str) -> bool:
    tail = text[-8:]
    return any(marker.startswith(tail) for marker in PROMPT_MARKERS if tail)


def stream_exe(
    prompt: str,
    max_tokens: int,
    session_id: str,
    model_path: Path,
) -> Generator[str, None, None]:
    """Run Quadtrix.exe and stream only the model reply as SSE events."""
    env = os.environ.copy()
    env["MODEL_PATH"] = str(model_path)

    cmd = [
        str(EXE_PATH),
        "--chat",
        "--chat-tokens",
        str(max_tokens),
        str(DATA_PATH),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(ROOT_DIR),
            bufsize=0,
        )
    except FileNotFoundError:
        yield f"data: [ERROR] exe not found: {EXE_PATH}\n\n"
        yield "data: [DONE]\n\n"
        return
    except Exception as exc:
        yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"
        return

    with procs_lock:
        procs[session_id] = proc

    threading.Thread(target=_drain_pipe, args=(proc.stderr,), daemon=True).start()

    try:
        assert proc.stdin is not None
        proc.stdin.write((prompt + "\n").encode("utf-8"))
        proc.stdin.flush()
    except Exception as exc:
        yield f"data: [ERROR] stdin write failed: {exc}\n\n"
        try:
            proc.kill()
        except Exception:
            pass
        with procs_lock:
            procs.pop(session_id, None)
        yield "data: [DONE]\n\n"
        return

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    pre_buffer = ""
    output_hold = ""
    started = False
    first_emit = True
    stopped_by_prompt = False
    start_time = time.monotonic()
    last_byte_time = start_time
    last_status_time = start_time
    stdout_queue: queue.Queue[bytes | None] = queue.Queue()

    def read_stdout() -> None:
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(1)
                if not chunk:
                    break
                stdout_queue.put(chunk)
        finally:
            stdout_queue.put(None)

    threading.Thread(target=read_stdout, daemon=True).start()
    yield _sse_event("status", "Starting Quadtrix and loading the model...")

    def drain(final: bool) -> Generator[str, None, None]:
        nonlocal pending, pre_buffer, output_hold, started, first_emit, stopped_by_prompt

        clean, remainder = _split_ansi(pending)
        pending = "" if final else remainder

        if not started:
            pre_buffer += clean
            marker_index = pre_buffer.find(MODEL_MARKER)
            if marker_index == -1:
                # Some builds do not print the exact marker, or print it very late.
                # After a short wait, stream from the latest known prompt boundary
                # instead of leaving the browser blank forever.
                if time.monotonic() - start_time > 5 and len(pre_buffer) > 0:
                    fallback_index = max(pre_buffer.rfind("\n"), pre_buffer.rfind("\r"))
                    clean = pre_buffer[fallback_index + 1 :] if fallback_index != -1 else pre_buffer
                    pre_buffer = ""
                    started = True
                else:
                    if len(pre_buffer) > 16384:
                        pre_buffer = pre_buffer[-4096:]
                    return
            else:
                clean = pre_buffer[marker_index + len(MODEL_MARKER) :]
                pre_buffer = ""
                started = True

        def emit_text(text: str) -> Generator[str, None, None]:
            nonlocal first_emit
            for item in text:
                if first_emit and item in (" ", "\t", "\n", "\r"):
                    continue
                first_emit = False
                event = _sse_char(item)
                if event:
                    yield event

        max_marker_length = max(len(marker) for marker in PROMPT_MARKERS)

        for char in clean:
            output_hold += char

            for marker in PROMPT_MARKERS:
                marker_index = output_hold.find(marker)
                if marker_index != -1:
                    yield from emit_text(output_hold[:marker_index].rstrip())
                    output_hold = ""
                    stopped_by_prompt = True
                    return

            while len(output_hold) > max_marker_length - 1:
                if _ends_like_prompt_marker(output_hold):
                    break
                yield from emit_text(output_hold[0])
                output_hold = output_hold[1:]

        if final and output_hold:
            yield from emit_text(output_hold)
            output_hold = ""

    try:
        while True:
            try:
                byte = stdout_queue.get(timeout=0.25)
            except queue.Empty:
                now = time.monotonic()
                if not started and now - start_time > MODEL_LOAD_TIMEOUT_SECONDS:
                    yield "data: [ERROR] model did not become ready within 180 seconds\n\n"
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    break

                if started and now - last_byte_time > GENERATION_IDLE_TIMEOUT_SECONDS:
                    yield "data: [ERROR] generation stalled for 180 seconds\n\n"
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    break

                if now - last_status_time >= 3 and proc.poll() is None:
                    last_status_time = now
                    if not started:
                        yield _sse_event("status", "Still loading the model. First token will appear automatically...")
                    else:
                        yield _sse_event("status", "Generating tokens...")
                continue

            if byte is None:
                tail = decoder.decode(b"", final=True)
                if tail:
                    pending += tail
                yield from drain(final=True)
                break

            last_byte_time = time.monotonic()
            text = decoder.decode(byte)
            if not text:
                continue

            pending += text
            yield from drain(final=False)
            if stopped_by_prompt:
                break
    except GeneratorExit:
        try:
            proc.kill()
        except Exception:
            pass
        with procs_lock:
            procs.pop(session_id, None)
        return

    try:
        assert proc.stdin is not None
        proc.stdin.write(b"quit\n")
        proc.stdin.flush()
    except Exception:
        pass

    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()

    with procs_lock:
        procs.pop(session_id, None)

    yield "data: [DONE]\n\n"


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def frontend_file(path: str):
    return send_from_directory(FRONTEND_DIR, path)


@app.route("/models")
def models():
    return jsonify({"default": _public_path(DEFAULT_MODEL_PATH), "models": _discover_models()})


@app.route("/status")
def status():
    selected_model, model_error = _resolve_model(request.args.get("model"))
    model_path = selected_model or DEFAULT_MODEL_PATH
    exe_ok = EXE_PATH.exists()
    data_ok = DATA_PATH.exists()
    model_ok = selected_model is not None

    error = None
    if not exe_ok:
        error = f"Quadtrix.exe not found: {EXE_PATH}"
    elif not data_ok:
        error = f"data input not found: {DATA_PATH}"
    elif model_error:
        error = model_error

    return jsonify(
        {
            "exe": exe_ok,
            "data": data_ok,
            "model": model_ok,
            "ready": exe_ok and data_ok and model_ok,
            "error": error,
            "exe_path": str(EXE_PATH),
            "data_path": str(DATA_PATH),
            "model_path": str(model_path),
            "selected_model": _public_path(model_path),
        }
    )


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/generate")
def generate():
    prompt = request.args.get("prompt", "").strip()
    session_id = request.args.get("sid", "default").strip() or "default"

    try:
        max_tokens = int(request.args.get("max_tokens", 200))
    except ValueError:
        return jsonify({"error": "max_tokens must be a number"}), 400

    max_tokens = max(1, min(max_tokens, 2000))
    model_path, model_error = _resolve_model(request.args.get("model"))

    if not prompt:
        return jsonify({"error": "empty prompt"}), 400
    if not EXE_PATH.exists():
        return jsonify({"error": f"exe not found: {EXE_PATH}"}), 500
    if not DATA_PATH.exists():
        return jsonify({"error": f"data input not found: {DATA_PATH}"}), 500
    if model_error or model_path is None:
        return jsonify({"error": model_error or "model not found"}), 500

    return Response(
        stream_exe(prompt, max_tokens, session_id, model_path),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/stop", methods=["POST"])
def stop():
    session_id = "default"
    if request.is_json:
        body = request.get_json(silent=True) or {}
        session_id = str(body.get("sid") or "default")

    with procs_lock:
        proc = procs.pop(session_id, None)

    if proc and proc.poll() is None:
        proc.kill()
        return jsonify({"status": "stopped"})
    return jsonify({"status": "idle"})


def main() -> None:
    print()
    print("=" * 56)
    print("  Quadtrix Web Interface")
    print("=" * 56)
    print(f"  exe    {EXE_PATH}")
    print(f"         {'found' if EXE_PATH.exists() else 'NOT FOUND'}")
    print(f"  data   {DATA_PATH}")
    print(f"         {'found' if DATA_PATH.exists() else 'NOT FOUND'}")
    print(f"  model  {DEFAULT_MODEL_PATH}")
    print(f"         {'found' if DEFAULT_MODEL_PATH.exists() else 'NOT FOUND'}")
    print("  open   http://localhost:5000")
    print("=" * 56)
    print()
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
