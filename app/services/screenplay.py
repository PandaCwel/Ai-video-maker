import re
from typing import Any


def analyze_screenplay(script: str) -> dict[str, Any]:
    """A deterministic baseline parser. This is not an LLM analysis."""
    normalized = script.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    scenes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for block in blocks:
        first = block.splitlines()[0].strip()
        if re.match(r"^(INT\.|EXT\.|INT/EXT\.|I/E\.)", first, re.IGNORECASE):
            if current:
                scenes.append(current)
            current = {"heading": first, "content": block, "characters": [], "dialogue": []}
        elif current is None:
            current = {"heading": "OPENING", "content": block, "characters": [], "dialogue": []}
        else:
            current["content"] += "\n\n" + block

    if current:
        scenes.append(current)

    for scene in scenes:
        lines = scene["content"].splitlines()
        names: list[str] = []
        for index, line in enumerate(lines[:-1]):
            candidate = line.strip()
            if (2 <= len(candidate) <= 40 and candidate.upper() == candidate
                    and re.fullmatch(r"[A-Z0-9 ._'()-]+", candidate)
                    and not candidate.startswith(("INT.", "EXT."))):
                names.append(candidate)
                following = lines[index + 1].strip()
                if following:
                    scene["dialogue"].append({"character": candidate, "text": following})
        scene["characters"] = sorted(set(names))

    return {
        "engine": "baseline-parser",
        "notice": "To jest wstępny podział heurystyczny, nie analiza LLM. Wymaga ręcznej weryfikacji.",
        "scene_count": len(scenes),
        "scenes": scenes,
    }
