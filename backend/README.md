# Quadtrix Middleware

FastAPI middleware for the Quadtrix.cpp C++ inference server.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --port 3001 --reload
```

Start the C++ server first:

```bash
./Quadtrix data/input.txt --server --port 8080
```
