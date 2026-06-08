from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
DEFAULT_CHECKPOINT = ROOT / "engine" / "best_model.pt"


def npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def python_command() -> str:
    venv_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(venv_python) if venv_python.exists() else sys.executable


def start_process(name: str, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    print(f"[start] {name}: {' '.join(command)}")
    return subprocess.Popen(command, cwd=str(cwd), env=env)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    api_port = os.environ.get("API_PORT", "3001")
    frontend_port = os.environ.get("FRONTEND_PORT", "5173")
    checkpoint = Path(os.environ.get("TORCH_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT))).resolve()

    if not checkpoint.exists():
        print(f"[error] .pt checkpoint not found: {checkpoint}")
        print("        Set TORCH_CHECKPOINT_PATH to your best_model.pt file.")
        return 1

    backend_env = os.environ.copy()
    backend_env.update(
        {
            "API_PORT": api_port,
            "CORS_ORIGINS": f"http://localhost:{frontend_port},http://127.0.0.1:{frontend_port}",
            "TORCH_CHECKPOINT_PATH": str(checkpoint),
        }
    )

    frontend_env = os.environ.copy()
    frontend_env.update(
        {
            "VITE_API_BASE_URL": f"http://localhost:{api_port}",
            "VITE_TORCH_ONLY": "1",
        }
    )

    backend = start_process(
        "backend (.pt)",
        [python_command(), "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", api_port, "--reload"],
        BACKEND,
        backend_env,
    )
    frontend = start_process(
        "frontend",
        [npm_command(), "run", "dev", "--", "--port", frontend_port],
        FRONTEND,
        frontend_env,
    )

    url = f"http://localhost:{frontend_port}"
    print(f"[ready] frontend: {url}")
    print(f"[ready] backend : http://localhost:{api_port}")
    print("[mode]  PyTorch .pt only")
    print("[stop]  Press Ctrl+C to stop both servers.")

    if os.environ.get("NO_BROWSER") != "1":
        time.sleep(2)
        webbrowser.open(url)

    try:
        while True:
            if backend.poll() is not None:
                print(f"[exit] backend stopped with code {backend.returncode}")
                return backend.returncode or 1
            if frontend.poll() is not None:
                print(f"[exit] frontend stopped with code {frontend.returncode}")
                return frontend.returncode or 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[stop] stopping servers...")
        return 0
    finally:
        stop_process(frontend)
        stop_process(backend)


if __name__ == "__main__":
    raise SystemExit(main())
