import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

import httpx

from config import Settings
from models import CppGenerateResponse


class InferenceUnavailableError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InferenceClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.torch_runner = TorchInferenceRunner(settings)

    async def generate(self, prompt: str, max_tokens: int, temperature: float, model_backend: str) -> CppGenerateResponse:
        if model_backend == "torch":
            return await self.torch_runner.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        return await self.generate_cpp(prompt=prompt, max_tokens=max_tokens)

    async def generate_cpp(self, prompt: str, max_tokens: int) -> CppGenerateResponse:
        url = f"{self.settings.cpp_server_url.rstrip('/')}/generate"
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.post(url, json={"prompt": prompt, "max_tokens": max_tokens})
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise InferenceUnavailableError("The C++ inference server timed out") from exc
        except httpx.HTTPError as exc:
            raise InferenceUnavailableError(
                f"The C++ inference server is not reachable at {self.settings.cpp_server_url}"
            ) from exc

        try:
            return CppGenerateResponse.model_validate(response.json())
        except ValueError as exc:
            raise InferenceUnavailableError("The C++ inference server returned invalid JSON") from exc

    async def health(self) -> Dict[str, Any]:
        url = f"{self.settings.cpp_server_url.rstrip('/')}/health"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise InferenceUnavailableError(
                f"The C++ inference server is not reachable at {self.settings.cpp_server_url}"
            ) from exc

    def torch_health(self) -> bool:
        return self.torch_runner.is_available()


class TorchInferenceRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._module: Optional[ModuleType] = None
        self._model: Optional[Any] = None

    def checkpoint_path(self) -> Path:
        path = Path(self.settings.torch_checkpoint_path)
        if path.is_absolute():
            return path
        return (Path(__file__).resolve().parent / path).resolve()

    def engine_inference_path(self) -> Path:
        return (Path(__file__).resolve().parents[1] / "engine" / "inference.py").resolve()

    def is_available(self) -> bool:
        return self.checkpoint_path().exists() and self.engine_inference_path().exists()

    def _load_module(self) -> ModuleType:
        if self._module is not None:
            return self._module
        module_path = self.engine_inference_path()
        if not module_path.exists():
            raise InferenceUnavailableError(f"PyTorch inference file not found at {module_path}")
        spec = importlib.util.spec_from_file_location("quadtrix_engine_inference", module_path)
        if spec is None or spec.loader is None:
            raise InferenceUnavailableError("Could not load engine/inference.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["quadtrix_engine_inference"] = module
        spec.loader.exec_module(module)
        self._module = module
        return module

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        checkpoint = self.checkpoint_path()
        if not checkpoint.exists():
            raise InferenceUnavailableError(f"PyTorch checkpoint not found at {checkpoint}")
        module = self._load_module()
        self._model = module.load_model(checkpoint)
        return self._model

    def _generate_sync(self, prompt: str, max_tokens: int, temperature: float) -> CppGenerateResponse:
        started = time.monotonic()
        module = self._load_module()
        model = self._load_model()
        text = module.generate_response(
            model=model,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=None,
        )
        seconds = round(time.monotonic() - started, 3)
        return CppGenerateResponse(text=text, chars=len(text), seconds=seconds)

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> CppGenerateResponse:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._generate_sync, prompt, max_tokens, temperature),
                timeout=self.settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise InferenceUnavailableError("The PyTorch model timed out") from exc
        except (RuntimeError, FileNotFoundError, ImportError, AttributeError) as exc:
            raise InferenceUnavailableError(f"The PyTorch model is unavailable: {exc}") from exc
