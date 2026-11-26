from __future__ import annotations

import re
from typing import Any

from app.services.local_models import local_llm_json, LocalModelError


SYSTEM = """You are a professional film director and screenplay analyst.
Return valid JSON only with:
logline, genre, tone,
characters: [{name, description, visual_traits, voice_notes}],
scenes: [{number, heading, summary, location, time_of_day, mood,
shots: [{shot_number, framing, camera_motion, visual_prompt, duration_seconds,
characters: [character names], dialogue: [{character, text}], sound_notes}]}].
Create 2-4 filmable shots per scene. Preserve story facts and exact supplied dialogue.
Never invent dialogue. Every spoken line must be assigned to its screenplay character.
Do not omit dialogue. Keep character names stable across scenes."""


def _chunks(script: str, limit: int = 14000) -> list[str]:
    """Split at screenplay scene headings where possible, without silently truncating."""
    normalized = script.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    starts = [match.start() for match in re.finditer(r"(?im)^(?:INT\.|EXT\.|INT/EXT\.|I/E\.)", normalized)]
    if not starts:
        return [normalized[i:i + limit] for i in range(0, len(normalized), limit)]
    preamble = normalized[:starts[0]].strip()
    sections = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(normalized)
        sections.append(normalized[start:end].strip())
    chunks, current = [], preamble
    for section in sections:
        if current and len(current) + len(section) + 2 > limit:
            chunks.append(current)
            current = ""
        if len(section) > limit:
            if current:
                chunks.append(current)
                current = ""
            # Preserve every character, splitting oversized individual scenes at line boundaries.
            lines, part = section.splitlines(keepends=True), ""
            for line in lines:
                if len(part) + len(line) > limit and part:
                    chunks.append(part)
                    part = ""
                part += line
            current = part
        else:
            current = (current + "\n\n" + section).strip() if current else section
    if current:
        chunks.append(current)
    return chunks


def build_shot_plan(script: str) -> dict[str, Any]:
    chunks = _chunks(script)
    if not chunks:
        raise LocalModelError("Scenariusz jest pusty.")
    merged_scenes = []
    merged_characters: dict[str, dict] = {}
    loglines, genres, tones = [], [], []
    for index, chunk in enumerate(chunks, 1):
        data = local_llm_json(
            SYSTEM,
            f"SCREENPLAY PART {index}/{len(chunks)}. Analyze only this part; retain its dialogue and scene order.\n\n{chunk}",
            max_new_tokens=3500,
        )
        if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
            raise LocalModelError(f"AI Director nie zwrócił listy scen dla części {index}/{len(chunks)}.")
        for character in data.get("characters", []):
            if isinstance(character, dict) and character.get("name"):
                key = str(character["name"]).strip().casefold()
                merged_characters[key] = {**merged_characters.get(key, {}), **character}
        merged_scenes.extend(data["scenes"])
        if data.get("logline"):
            loglines.append(str(data["logline"]))
        if data.get("genre"):
            genres.append(str(data["genre"]))
        if data.get("tone"):
            tones.append(str(data["tone"]))
    for number, scene in enumerate(merged_scenes, 1):
        scene["number"] = number
        for shot_number, shot in enumerate(scene.get("shots", []), 1):
            shot["shot_number"] = shot_number
            shot.setdefault("dialogue", [])
    if not merged_scenes:
        raise LocalModelError("AI Director nie wykrył żadnych scen.")
    return {
        "engine": "local-llm",
        "parts_analyzed": len(chunks),
        "logline": " ".join(loglines)[:2000],
        "genre": genres[0] if genres else "",
        "tone": tones[0] if tones else "",
        "characters": list(merged_characters.values()),
        "scenes": merged_scenes,
        "scene_count": len(merged_scenes),
    }
