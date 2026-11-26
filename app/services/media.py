from pathlib import Path
from uuid import uuid4
import re


ALLOWED = {
    "image": {".png", ".jpg", ".jpeg", ".webp"},
    "audio": {".wav", ".mp3", ".m4a", ".ogg", ".flac"},
    "video": {".mp4", ".mov", ".webm"},
}


def save_upload(project_dir: Path, category: str, filename: str, content: bytes) -> dict:
    if category not in ALLOWED:
        raise ValueError("Nieznana kategoria pliku.")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED[category]:
        raise ValueError(f"Niedozwolone rozszerzenie {suffix or '(brak)'}.")
    if len(content) > 100 * 1024 * 1024:
        raise ValueError("Plik przekracza limit 100 MB.")
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(filename).name)[:100]
    asset_id = str(uuid4())
    folder = project_dir / "assets" / category
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{asset_id}_{safe_name}"
    target.write_bytes(content)
    return {"id": asset_id, "name": safe_name, "category": category, "path": str(target), "size": len(content)}
