import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from inference import InferenceClient
from middleware.error_handler import register_error_handlers
from middleware.logging import RequestLoggingMiddleware, configure_logging
from router import chat, feedback, health, sessions
from session_store import build_session_store


settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(title="Quadtrix API", version="1.0.0")
app.state.session_store = build_session_store(
    max_sessions=settings.max_sessions,
    ttl_hours=settings.session_ttl_hours,
    redis_url=settings.redis_url,
)
app.state.inference_client = InferenceClient(settings)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(feedback.router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "quadtrix-api"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.api_port, reload=True)
