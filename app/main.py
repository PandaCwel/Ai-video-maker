from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.project_store import ProjectNotFoundError, ProjectStore
from app.services.screenplay import analyze_screenplay

store = ProjectStore(settings.projects_dir)
app = FastAPI(title=settings.app_name, version="0.1.0", description="AI film production workspace API")
ROOT_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=ROOT_DIR / "frontend"), name="static")

@app.get("/", include_in_schema=False)
def studio_home():
    return FileResponse(ROOT_DIR / "frontend" / "index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    script: str | None = None
    status: str | None = Field(default=None, pattern="^(draft|planning|producing|completed|failed)$")


class ScriptUpdate(BaseModel):
    script: str = Field(max_length=1_000_000)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name, "version": app.version}


@app.get("/api/projects")
def list_projects():
    return {"projects": store.list()}


@app.post("/api/projects", status_code=201)
def create_project(payload: ProjectCreate):
    return store.create(payload.title)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    try:
        return store.get(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdate):
    try:
        return store.update(project_id, **payload.model_dump(exclude_unset=True))
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: str):
    try:
        store.delete(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    return None


@app.get("/api/projects/{project_id}/script")
def get_script(project_id: str):
    try:
        project = store.get(project_id)
        return {"script": project["script"]}
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")


@app.put("/api/projects/{project_id}/script")
def save_script(project_id: str, payload: ScriptUpdate):
    try:
        project = store.update(project_id, script=payload.script)
        return {"script": project["script"], "updated_at": project["updated_at"]}
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")


@app.post("/api/projects/{project_id}/script/analyze")
def analyze_script(project_id: str):
    try:
        project = store.get(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project["script"].strip():
        raise HTTPException(status_code=400, detail="Script is empty")
    result = analyze_screenplay(project["script"])
    store.update(project_id, status="planning")
    project_dir = settings.projects_dir / project_id
    (project_dir / "analysis.json").write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
