from typing import List

from fastapi import APIRouter, HTTPException, Request, status

from models import AddMessageRequest, CreateSessionRequest, Message, Session
from session_store import SessionStore

router = APIRouter()


def store_from_request(request: Request) -> SessionStore:
    return request.app.state.session_store


@router.get("/api/sessions", response_model=List[Session])
async def list_sessions(request: Request) -> List[Session]:
    return store_from_request(request).list_sessions()[:50]


@router.post("/api/sessions", response_model=Session, status_code=status.HTTP_201_CREATED)
async def create_session(payload: CreateSessionRequest, request: Request) -> Session:
    return store_from_request(request).create_session(title=payload.title)


@router.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, request: Request) -> None:
    deleted = store_from_request(request).delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/api/sessions/{session_id}/messages", response_model=List[Message])
async def get_messages(session_id: str, request: Request) -> List[Message]:
    store = store_from_request(request)
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return store.get_messages(session_id)


@router.post("/api/sessions/{session_id}/messages", response_model=Message, status_code=status.HTTP_201_CREATED)
async def add_message(session_id: str, payload: AddMessageRequest, request: Request) -> Message:
    store = store_from_request(request)
    store.get_or_create_session(session_id)
    return store.add_message(Message(session_id=session_id, role=payload.role, text=payload.text))
