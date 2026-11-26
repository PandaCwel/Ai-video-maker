import json
from typing import Any
from app.services.local_models import local_llm_json, LocalModelError

def build_shot_plan(script: str) -> dict[str, Any]:
    system = """Analyze the screenplay. Return JSON only with logline, tone, characters (array of {name,description,visual_traits,voice_notes}), scenes (array of {number,heading,summary,location,time_of_day,mood,shots (array of {shot_number,framing,camera_motion,visual_prompt,duration_seconds,characters,dialogue,sound_notes})}). Create 2-4 filmable shots per scene. Preserve story facts and supplied dialogue. Never invent dialogue."""
    data=local_llm_json(system,script[:50000],max_new_tokens=3500)
    if not isinstance(data.get("scenes"),list): raise LocalModelError("AI Director nie zwrócił listy scen.")
    data["engine"]="local-llm"
    return data
