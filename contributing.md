# Contributing to Quadtrix.cpp

Thanks for helping improve Quadtrix.cpp. This project is a transformer learning lab with several execution paths: native C++ training and inference, PyTorch experiments, a FastAPI backend, and a React + TypeScript chat UI. Contributions are easiest to review when they keep those paths clear and testable.

## Good First Contributions

Useful contributions include:

- Fixing correctness bugs in the C++ transformer implementation.
- Improving training, inference, checkpoint loading, or export scripts.
- Making the FastAPI backend more reliable and easier to run locally.
- Improving the React chat UI without hiding the model/backend behavior.
- Adding focused documentation for setup, model files, datasets, or run commands.
- Tightening CI, dependency versions, packaging, or release steps.

For larger model architecture changes, open an issue first so the design can be discussed before a big patch lands.

## Repository Layout

```text
Quadtrix.cpp/
  main.cpp              Native C++ entry point
  include/              C++ headers
  src/                  C++ source files
  engine/               PyTorch training, inference, export, and model files
  backend/              FastAPI backend and session handling
  frontend/             React + TypeScript chat UI
  iGPU/                 Integrated GPU experiments
  config/               Runtime configuration
  data/                 Local datasets and helpers
  .github/workflows/    CI and release workflows
```

## Development Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm.cmd install
```

Build the native C++ runtime from the repository root:

```bash
g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix main.cpp
```

## Running Locally

Run the C++ model:

```powershell
.\Quadtrix.exe data\input.txt --chat
```

Run the C++ HTTP server:

```powershell
.\Quadtrix.exe data\input.txt --server --port 8080
```

Run the FastAPI backend:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

Run the frontend:

```powershell
cd frontend
npm.cmd run dev
```

Open the app at:

```text
http://localhost:5173
```

## Checks Before Opening a Pull Request

Run the checks that match your change.

C++:

```bash
g++ -std=c++17 -O2 -I. -Iinclude -o quadtrix main.cpp
```

Python:

```powershell
.\.venv\Scripts\python.exe -m compileall backend engine iGPU
cd backend
..\.venv\Scripts\python.exe -c "from main import app; print(app.title)"
```

Frontend:

```powershell
cd frontend
npm.cmd run build
```

If you cannot run a relevant check, mention that in the pull request and explain why.

## Pull Request Guidelines

- Keep changes focused on one problem or feature.
- Use the existing style of the file you are editing.
- Avoid committing generated artifacts unless the project already expects them.
- Do not commit `.env` files, secrets, private datasets, or personal checkpoints.
- Update `README.md`, `run.md`, or related docs when commands or behavior change.
- Include screenshots or short notes for UI changes.
- Mention any change that affects model files, ports, CORS, service workers, or packaging.

The pull request template asks for:

- Summary and user-facing impact.
- C++ build status.
- Backend smoke-test status.
- Frontend build status.
- Documentation or screenshot updates when needed.

## Coding Notes

For C++ changes:

- Prefer clear, debuggable code over clever abstractions.
- Keep the educational value of the implementation visible.
- Be careful with tensor shapes, bounds, and ownership.
- Add comments only where the math or control flow is not obvious.

For Python changes:

- Keep backend behavior explicit and local-development friendly.
- Avoid broad exception swallowing around model loading or inference.
- Treat model paths, datasets, and request payloads as untrusted inputs.

For frontend changes:

- Keep the chat UI practical and responsive.
- Preserve the ability to switch between the C++ backend and the `.pt` path.
- Make loading, error, and disconnected states clear.

## Documentation Style

Use concrete commands and paths. Quadtrix.cpp has multiple runtime paths, so say exactly which path a command belongs to: C++, PyTorch, FastAPI backend, frontend, iGPU, or packaging.

When documenting training results, include the hardware, dataset, iteration count, elapsed time, and validation metric so results can be compared fairly.
