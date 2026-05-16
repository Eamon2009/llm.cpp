import time

from fastapi import APIRouter, Request

from inference import InferenceClient, InferenceUnavailableError
from models import HealthResponse, StatsResponse

router = APIRouter()
START_TIME = time.monotonic()


def uptime_seconds() -> float:
    return round(time.monotonic() - START_TIME, 3)


@router.get("/api/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    client: InferenceClient = request.app.state.inference_client
    torch_status = "ok" if client.torch_health() else "unavailable"
    try:
        data = await client.health()
        return HealthResponse(
            status="ok",
            api="ok",
            cpp_server="ok",
            torch_model=torch_status,
            model=str(data.get("model", "quadtrix-v1.0")),
            vocab=int(data.get("vocab", 105)),
            params=int(data.get("params", 826985)),
            uptime_seconds=uptime_seconds(),
        )
    except InferenceUnavailableError:
        return HealthResponse(
            status="degraded",
            api="ok",
            cpp_server="unreachable",
            torch_model=torch_status,
            uptime_seconds=uptime_seconds(),
        )


@router.get("/api/stats", response_model=StatsResponse)
async def stats(request: Request) -> StatsResponse:
    client: InferenceClient = request.app.state.inference_client
    online = True
    try:
        await client.health()
    except InferenceUnavailableError:
        online = False
    return StatsResponse(
        backend=client.settings.cpp_server_url,
        backend_online=online,
        torch_checkpoint=str(client.torch_runner.checkpoint_path()),
        torch_online=client.torch_health(),
        uptime_seconds=uptime_seconds(),
    )
