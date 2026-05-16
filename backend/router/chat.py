import logging
import time

from fastapi import APIRouter, HTTPException, Request

from inference import InferenceClient, InferenceUnavailableError
from models import ChatRequest, ChatResponse, Message, Role, new_id, utc_now
from session_store import SessionStore

router = APIRouter()
logger = logging.getLogger("quadtrix.api")


@router.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    started = time.monotonic()
    store: SessionStore = request.app.state.session_store
    client: InferenceClient = request.app.state.inference_client
    title = payload.prompt[:40]
    session = store.get_or_create_session(payload.session_id, title=title)
    user_message = Message(session_id=session.id, role=Role.user, text=payload.prompt)
    store.add_message(user_message)

    try:
        generated = await client.generate(
            prompt=payload.prompt,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            model_backend=payload.model_backend,
        )
    except InferenceUnavailableError as exc:
        error_message = Message(
            session_id=session.id,
            role=Role.assistant,
            text="Could not reach the selected model. Check the C++ server or engine checkpoint.",
            error="model_unavailable",
        )
        store.add_message(error_message)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_unavailable",
                "message": exc.message,
                "code": 503,
            },
        ) from exc

    assistant_message = Message(
        id=new_id("msg"),
        session_id=session.id,
        role=Role.assistant,
        text=generated.text,
        prompt=payload.prompt,
        chars=generated.chars,
        seconds=generated.seconds,
    )
    store.add_message(assistant_message)
    latency = round(time.monotonic() - started, 3)
    logger.info(
        "chat_request",
        extra={
            "session_id": session.id,
            "prompt_length": len(payload.prompt),
            "latency": latency,
            "chars": generated.chars,
            "model_backend": payload.model_backend,
        },
    )
    return ChatResponse(
        id=assistant_message.id,
        session_id=session.id,
        prompt=payload.prompt,
        text=generated.text,
        chars=generated.chars,
        seconds=generated.seconds,
        model="quadtrix-v1.0-pt" if payload.model_backend == "torch" else "quadtrix-v1.0",
        model_backend=payload.model_backend,
        created_at=assistant_message.created_at,
    )
