from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_port: int = Field(default=3001, alias="API_PORT")
    cors_origins: str = Field(default="http://localhost:5173", alias="CORS_ORIGINS")
    redis_url: str = Field(default="", alias="REDIS_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_sessions: int = Field(default=1000, alias="MAX_SESSIONS")
    session_ttl_hours: int = Field(default=24, alias="SESSION_TTL_HOURS")
    cpp_server_url: str = Field(default="http://localhost:8080", alias="CPP_SERVER_URL")
    torch_checkpoint_path: str = Field(default="../engine/best_model.pt", alias="TORCH_CHECKPOINT_PATH")
    request_timeout_seconds: float = Field(default=60.0, alias="REQUEST_TIMEOUT_SECONDS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
