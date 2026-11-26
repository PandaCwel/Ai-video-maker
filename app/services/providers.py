"""Explicit external-provider adapters. Missing credentials produce actionable errors, never fake media."""
import os
import time
from pathlib import Path
from typing import Any
import httpx


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY")
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def complete(self, system: str, user: str) -> str:
        if not self.key:
            raise ProviderError("Brak OPENAI_API_KEY. Dodaj klucz w .env, aby uruchomić analizę LLM.")
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"model": self.model, "temperature": 0.3, "messages": [
                    {"role": "system", "content": system}, {"role": "user", "content": user}
                ]},
            )
            response.raise_for_status()
            data = response.json()
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError("LLM zwrócił nieoczekiwaną odpowiedź.") from exc


class ReplicateProvider:
    def __init__(self):
        self.token = os.getenv("REPLICATE_API_TOKEN")
        self.base = "https://api.replicate.com/v1"

    def _headers(self):
        if not self.token:
            raise ProviderError("Brak REPLICATE_API_TOKEN. Ustaw token Replicate w .env.")
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    async def run(self, model: str, inputs: dict[str, Any], timeout: int = 1800) -> dict[str, Any]:
        if not model or "/" not in model:
            raise ProviderError("Ustaw identyfikator modelu Replicate w konfiguracji, np. owner/model.")
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.base}/models/{model}/predictions",
                headers=self._headers(), json={"input": inputs})
            if response.status_code == 404:
                raise ProviderError(f"Model Replicate {model} nie istnieje lub nie jest dostępny.")
            response.raise_for_status()
            prediction = response.json()
            poll_url = prediction.get("urls", {}).get("get")
            if not poll_url:
                return prediction
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                await __import__("asyncio").sleep(3)
                poll = await client.get(poll_url, headers=self._headers())
                poll.raise_for_status()
                prediction = poll.json()
                if prediction.get("status") in {"succeeded", "failed", "canceled"}:
                    if prediction["status"] != "succeeded":
                        raise ProviderError(prediction.get("error") or f"Generowanie zakończone: {prediction['status']}")
                    return prediction
            raise ProviderError("Przekroczono limit czasu oczekiwania na model.")

class ElevenLabsProvider:
    def __init__(self):
        self.key = os.getenv("ELEVENLABS_API_KEY")
        self.base = "https://api.elevenlabs.io/v1"

    async def clone_voice(self, name: str, sample_name: str, sample_bytes: bytes) -> dict:
        if not self.key:
            raise ProviderError("Brak ELEVENLABS_API_KEY.")
        files = {"files": (sample_name, sample_bytes, "application/octet-stream")}
        data = {"name": name, "description": "Voice cloned from a user-provided sample with confirmed permission."}
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(f"{self.base}/voices/add",
                headers={"xi-api-key": self.key}, data=data, files=files)
            response.raise_for_status()
            result = response.json()
            if not result.get("voice_id"):
                raise ProviderError("ElevenLabs nie zwrócił voice_id.")
            return result

    async def synthesize(self, voice_id: str, text: str, output: Path) -> Path:
        if not self.key:
            raise ProviderError("Brak ELEVENLABS_API_KEY.")
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(f"{self.base}/text-to-speech/{voice_id}",
                headers={"xi-api-key": self.key, "Accept": "audio/mpeg", "Content-Type": "application/json"},
                json={"text": text, "model_id": "eleven_multilingual_v2"})
            response.raise_for_status()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(response.content)
        return output
