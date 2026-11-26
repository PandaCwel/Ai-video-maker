from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

_lock = Lock()
_jobs: dict[str, dict] = {}


def create_job(project_id: str, kind: str, payload: dict) -> dict:
    job = {"id": str(uuid4()), "project_id": project_id, "kind": kind, "payload": payload,
           "status": "queued", "progress": 0, "message": "Queued", "created_at": datetime.now(timezone.utc).isoformat()}
    with _lock:
        _jobs[job["id"]] = job
    return job.copy()


def update_job(job_id: str, **changes) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes, updated_at=datetime.now(timezone.utc).isoformat())


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return job.copy() if job else None


def list_jobs(project_id: str) -> list[dict]:
    with _lock:
        return [j.copy() for j in _jobs.values() if j["project_id"] == project_id]
