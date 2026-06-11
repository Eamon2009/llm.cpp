# Quadtrix.cpp

Quadtrix.cpp is a local GPT-style language model project with multiple runtime paths:

- Native C++ inference and training through `Quadtrix.exe` / `main.cpp`
- PyTorch checkpoint inference through `engine/inference.py` and `engine/best_model .pt`
- FastAPI middleware in `backend/`
- React + TypeScript chat UI in `frontend/`

The web interface can chat with both model backends:

- `C++`: calls the C++ HTTP server on port `8080`
- `.pt`: loads the PyTorch checkpoint directly from `engine/best_model .pt`

## Project Layout

```text
Quadtrix.cpp/
  Quadtrix.exe
  main.cpp
  config/
  include/
  data/
  engine/
    inference.py
    main.py
    fine-tune/main.py
    best_model .pt
    fineweb_30mb.txt
  backend/
    main.py
    inference.py
    requirements.txt
  frontend/
    package.json
    src/
```

## Requirements

- Python 3.10+
- Node.js 18+
- npm
- C++17 compiler if you want to rebuild the C++ executable

## 1. Python Setup

From the repo root:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install backend and PyTorch inference dependencies:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Frontend Setup

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd install
npm.cmd run build
```

Run the frontend:

```powershell
npm.cmd run dev
```

Frontend URL:

```text
http://localhost:5173
```

## Install as a Web App

The frontend is configured as an installable PWA. It includes:

- `frontend/manifest.webmanifest`
- `frontend/sw.js`
- `frontend/public/manifest.webmanifest`
- `frontend/public/sw.js`
- service worker registration in `frontend/src/registerServiceWorker.ts`

For the clean installable version, build and preview the frontend:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd run build
npm.cmd run preview
```

Open the preview URL, usually:

```text
http://localhost:4173
```

Then install from the browser:

- Chrome / Edge: click the install icon in the address bar
- Or open browser menu -> Apps -> Install this site as an app

The installed app still talks to the backend at:

```text
http://localhost:3001
```

So keep the FastAPI backend running when chatting.

## 3. Run the PyTorch `.pt` Model in the Web UI

The `.pt` model does not need a separate model server. The FastAPI backend loads it directly from:

```text
engine/best_model .pt
```

Start the backend:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

Start the frontend in another terminal:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd run dev
```

Open:

```text
http://localhost:5173
```

Select `.pt` in the top bar.

## 4. Run the C++ Model in the Web UI

Start the C++ inference server:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\Quadtrix.exe data\input.txt --server --port 8080
```

Start the backend:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

Start the frontend:

```powershell
cd ..\frontend
npm.cmd run dev
```

Open:

```text
http://localhost:5173
```

Select `C++` in the top bar.

## 5. Run Both Backends Together

Use three terminals.

Terminal 1:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\Quadtrix.exe data\input.txt --server --port 8080
```

Terminal 2:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

Terminal 3:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd run dev
```

Open:

```text
http://localhost:5173
```

Switch between `C++` and `.pt` from the model selector.

## 6. Backend API

Base URL:

```text
http://localhost:3001
```

Routes:

```text
GET    /api/health
GET    /api/stats
POST   /api/chat
GET    /api/sessions
POST   /api/sessions
DELETE /api/sessions/{id}
GET    /api/sessions/{id}/messages
POST   /api/feedback
```

Example `.pt` chat request:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:3001/api/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "session_id": null,
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 1.0,
    "stream": false,
    "model_backend": "torch"
  }'
```

Example C++ chat request:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:3001/api/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{
    "session_id": null,
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 1.0,
    "stream": false,
    "model_backend": "cpp"
  }'
```

## 7. Environment Variables

Backend defaults are in `backend/.env.example`:

```text
API_PORT=3001
CORS_ORIGINS=http://localhost:5173
REDIS_URL=
LOG_LEVEL=INFO
MAX_SESSIONS=1000
SESSION_TTL_HOURS=24
CPP_SERVER_URL=http://localhost:8080
TORCH_CHECKPOINT_PATH=../engine/best_model .pt
REQUEST_TIMEOUT_SECONDS=60
```

Create `backend/.env` if you want overrides.

Frontend defaults are in `frontend/.env.example`:

```text
VITE_API_BASE_URL=http://localhost:3001
```

## 8. PyTorch CLI Inference

Interactive chat:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\.venv\Scripts\python.exe engine\inference.py --checkpoint "engine\best_model .pt"
```

Generate once:

```powershell
.\.venv\Scripts\python.exe engine\inference.py --checkpoint "engine\best_model .pt" --prompt "Hello" --max-new-tokens 100 --temperature 1.0
```

## 9. PyTorch Training

Main training:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\.venv\Scripts\python.exe engine\main.py
```

Fine-tuning:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\.venv\Scripts\python.exe engine\fine-tune\main.py
```

## 10. C++ Build and Run

Build manually:

```powershell
g++ -std=c++17 -O2 -I. -Iinclude -o Quadtrix.exe main.cpp
```

Train from scratch:

```powershell
.\Quadtrix.exe data\input.txt
```

Terminal chat:

```powershell
.\Quadtrix.exe data\input.txt --chat
```

Raw generation:

```powershell
.\Quadtrix.exe data\input.txt --generate
```

HTTP server:

```powershell
.\Quadtrix.exe data\input.txt --server --port 8080
```

## 11. Health Checks

Backend:

```powershell
Invoke-RestMethod http://localhost:3001/api/health
```

C++ server:

```powershell
Invoke-RestMethod http://localhost:8080/health
```

Frontend:

```text
http://localhost:5173
```

When only `.pt` is available, backend health should show:

```json
{
  "status": "degraded",
  "api": "ok",
  "cpp_server": "unreachable",
  "torch_model": "ok"
}
```

When both are available, backend health should show:

```json
{
  "status": "ok",
  "api": "ok",
  "cpp_server": "ok",
  "torch_model": "ok"
}
```

## 12. Troubleshooting

### PowerShell blocks `npm`

Use `npm.cmd`:

```powershell
npm.cmd run dev
npm.cmd run build
```

### `.pt` model is unavailable

Check that this file exists:

```text
engine/best_model .pt
```

Then check Python dependencies:

```powershell
cd backend
..\.venv\Scripts\python.exe -c "import torch, tiktoken; print(torch.__version__)"
```

### Backend cannot import FastAPI

Install dependencies into the repo venv:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### C++ option is offline

Start the C++ server:

```powershell
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\Quadtrix.exe data\input.txt --server --port 8080
```

### Frontend cannot reach backend

Check:

```text
http://localhost:3001/api/health
```

Make sure frontend config points to:

```text
VITE_API_BASE_URL=http://localhost:3001
```

### Port already in use

```powershell
Get-NetTCPConnection -LocalPort 3001
Get-NetTCPConnection -LocalPort 5173
Get-NetTCPConnection -LocalPort 8080
```

## Recommended Daily Run

```powershell
# Terminal 1
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp
.\Quadtrix.exe data\input.txt --server --port 8080
```

```powershell
# Terminal 2
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 3001
```

```powershell
# Terminal 3
cd C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\frontend
npm.cmd run dev
```

Open:

```text
http://localhost:5173
```

## License

MIT
