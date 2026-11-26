import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

_lock = Lock()
_memory: dict[str, dict] = {}
_root = Path(os.getenv("STUDIO_DATA_DIR", "./data")) / "jobs"


def _persist(job: dict) -> None:
    _root.mkdir(parents=True, exist_ok=True)
    target = _root / f"{job['id']}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


def create_job(project_id: str, kind: str, payload: dict) -> dict:
    job = {"id": str(uuid4()), "project_id": project_id, "kind": kind, "payload": payload,
           "status": "queued", "progress": 0, "message": "Queued",
           "created_at": datetime.now(timezone.utc).isoformat()}
    with _lock:
        _memory[job["id"]] = job
        _persist(job)
    return job.copy()


def update_job(job_id: str, **changes) -> None:
    with _lock:
        job = _memory.get(job_id)
        if job is None:
            path = _root / f"{job_id}.json"
            if path.is_file():
                job = json.loads(path.read_text(encoding="utf-8"))
                _memory[job_id] = job
        if job is not None:
            job.update(changes, updated_at=datetime.now(timezone.utc).isoformat())
            _persist(job)


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _memory.get(job_id)
        if job is None:
            path = _root / f"{job_id}.json"
            if path.is_file():
                job = json.loads(path.read_text(encoding="utf-8"))
                _memory[job_id] = job
        return job.copy() if job else None


def list_jobs(project_id: str) -> list[dict]:
    with _lock:
        if _root.is_dir():
            for path in _root.glob("*.json"):
                if path.stem not in _memory:
                    try:
                        _memory[path.stem] = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
        return [j.copy() for j in _memory.values() if j["project_id"] == project_id]
