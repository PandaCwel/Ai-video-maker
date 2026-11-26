import json
from typing import Any
from app.services.providers import OpenAICompatibleLLM, ProviderError


async def build_shot_plan(script: str) -> dict[str, Any]:
    llm = OpenAICompatibleLLM()
    prompt = """Analyze the screenplay and return ONLY valid JSON with keys: logline, tone, characters (array of {name, description, visual_traits, voice_notes}), scenes (array of {number, heading, summary, location, time_of_day, mood, shots (array of {shot_number, framing, camera_motion, visual_prompt, duration_seconds, characters, dialogue, sound_notes})}). Keep continuity consistent. Each scene should have 2-5 practical shots. Do not invent dialogue; quote only supplied script dialogue."""
    try:
        raw = await llm.complete(prompt, script[:90000])
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
            raise ProviderError("LLM JSON nie zawiera tablicy scenes.")
        data["engine"] = "llm"
        return data
    except json.JSONDecodeError as exc:
        raise ProviderError("LLM nie zwrócił poprawnego JSON. Spróbuj ponownie lub skróć scenariusz.") from exc
