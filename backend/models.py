from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ErrorResponse(BaseModel):
    error: str
    message: str
    code: int


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    prompt: str = Field(min_length=1, max_length=500)
    max_tokens: int = Field(default=200, ge=1, le=500)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    stream: bool = False
    model_backend: Literal["cpp", "torch"] = "cpp"


class ChatResponse(BaseModel):
    id: str
    session_id: str
    prompt: str
    text: str
    chars: int
    seconds: float
    model: str = "quadtrix-v1.0"
    model_backend: Literal["cpp", "torch"] = "cpp"
    created_at: datetime


class Message(BaseModel):
    id: str = Field(default_factory=lambda: new_id("msg"))
    session_id: str
    role: Role
    text: str
    prompt: Optional[str] = None
    chars: int = 0
    seconds: float = 0.0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "New conversation"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    message_count: int = 0


class CreateSessionRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=80)


class AddMessageRequest(BaseModel):
    role: Role
    text: str = Field(min_length=1)


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: Optional[str] = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    ok: bool
    id: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: Literal["ok"]
    cpp_server: Literal["ok", "unreachable"]
    torch_model: Literal["ok", "unavailable"]
    model: str = "quadtrix-v1.0"
    vocab: int = 105
    params: int = 826985
    uptime_seconds: float


class StatsResponse(BaseModel):
    model: str = "quadtrix-v1.0"
    architecture: str = "4L x 4H x 200d"
    parameters: int = 826985
    vocabulary: int = 105
    val_loss: float = 1.6371
    context: int = 128
    training: str = "76.2 min CPU"
    backend: str
    backend_online: bool
    torch_checkpoint: str
    torch_online: bool
    uptime_seconds: float


class CppGenerateRequest(BaseModel):
    prompt: str
    max_tokens: int


class CppGenerateResponse(BaseModel):
    text: str
    chars: int
    seconds: float

    model_config = ConfigDict(extra="ignore")
