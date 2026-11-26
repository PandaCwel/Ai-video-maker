import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class ProjectNotFoundError(Exception):
    pass


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9-]{36}", project_id):
            raise ProjectNotFoundError(project_id)
        return self.root / project_id / "project.json"

    def _read(self, project_id: str) -> dict[str, Any]:
        path = self._path(project_id)
        if not path.is_file():
            raise ProjectNotFoundError(project_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, project: dict[str, Any]) -> dict[str, Any]:
        path = self._path(project["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return project

    def list(self) -> list[dict[str, Any]]:
        projects = []
        for path in self.root.glob("*/project.json"):
            try:
                projects.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)

    def create(self, title: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        project = {
            "id": str(uuid4()), "title": title.strip(), "created_at": now,
            "updated_at": now, "script": "", "analysis": None,
            "characters": [], "scenes": [], "status": "draft"
        }
        return self._write(project)

    def get(self, project_id: str) -> dict[str, Any]:
        return self._read(project_id)

    def update(self, project_id: str, **changes: Any) -> dict[str, Any]:
        project = self._read(project_id)
        for key, value in changes.items():
            if key in {"title", "script", "status"}:
                project[key] = value
        project["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self._write(project)

    def delete(self, project_id: str) -> None:
        path = self._path(project_id)
        if not path.is_file():
            raise ProjectNotFoundError(project_id)
        import shutil
        shutil.rmtree(path.parent)
