import json
import os
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.director import build_shot_plan
from app.services.jobs import create_job, get_job, list_jobs
from app.services.media import save_upload
from app.services.production import run_full_production, run_generation, run_tts
from app.services.project_store import ProjectNotFoundError, ProjectStore
from app.services.providers import ElevenLabsProvider, OpenAICompatibleLLM, ProviderError
from app.services.render import RenderError, render_video
from app.services.screenplay import analyze_screenplay

store = ProjectStore(settings.projects_dir)
app = FastAPI(title=settings.app_name, version="0.2.0", description="AI film production workspace API")
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


class CharacterCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=4000)
    visual_traits: str = Field(default="", max_length=4000)
    voice_notes: str = Field(default="", max_length=2000)
    reference_asset_id: str | None = None


class VoiceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    provider: str = Field(default="elevenlabs")
    provider_voice_id: str = Field(min_length=1, max_length=200)
    character_id: str | None = None
    consent_confirmed: bool = False


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=8000)
    image_url: str | None = None
    reference_asset_id: str | None = None
    inputs: dict[str, Any] = {}


class SpeechRequest(BaseModel):
    voice_id: str
    text: str = Field(min_length=1, max_length=5000)


class RenderRequest(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=200)


def get_project_or_404(project_id: str):
    try:
        return store.get(project_id)
    except ProjectNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name, "version": app.version,
            "providers": {"llm": bool(os.getenv("OPENAI_API_KEY")), "replicate": bool(os.getenv("REPLICATE_API_TOKEN")),
                          "speech": bool(os.getenv("ELEVENLABS_API_KEY"))}}


@app.get("/api/projects")
def list_projects():
    return {"projects": store.list()}


@app.post("/api/projects", status_code=201)
def create_project(payload: ProjectCreate):
    return store.create(payload.title)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    return get_project_or_404(project_id)


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, payload: ProjectUpdate):
    get_project_or_404(project_id)
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
    return {"script": get_project_or_404(project_id)["script"]}


@app.put("/api/projects/{project_id}/script")
def save_script(project_id: str, payload: ScriptUpdate):
    get_project_or_404(project_id)
    project = store.update(project_id, script=payload.script)
    return {"script": project["script"], "updated_at": project["updated_at"]}


@app.post("/api/projects/{project_id}/script/analyze")
async def analyze_script(project_id: str):
    project = get_project_or_404(project_id)
    if not project["script"].strip():
        raise HTTPException(status_code=400, detail="Script is empty")
    try:
        llm = OpenAICompatibleLLM()
        raw = await llm.complete("Analyze this screenplay. Return JSON only with logline, genre, tone, characters, scenes. Preserve story facts and do not invent dialogue.", project["script"][:90000])
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        analysis = json.loads(cleaned)
        analysis["engine"] = "llm"
    except (ProviderError, json.JSONDecodeError) as exc:
        analysis = analyze_screenplay(project["script"])
        analysis["warning"] = str(exc)
    store.update(project_id, analysis=analysis, scenes=analysis.get("scenes", []), status="planning")
    return analysis


@app.post("/api/projects/{project_id}/director/plan")
async def director_plan(project_id: str):
    project = get_project_or_404(project_id)
    if not project["script"].strip():
        raise HTTPException(status_code=400, detail="Add a screenplay first.")
    try:
        plan = await build_shot_plan(project["script"])
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    store.update(project_id, analysis=plan, scenes=plan["scenes"], status="planning")
    return plan


@app.post("/api/projects/{project_id}/characters", status_code=201)
def add_character(project_id: str, payload: CharacterCreate):
    project = get_project_or_404(project_id)
    characters = project.get("characters", [])
    character = {"id": __import__("uuid").uuid4().hex, **payload.model_dump()}
    characters.append(character)
    store.update(project_id, characters=characters)
    return character


@app.delete("/api/projects/{project_id}/characters/{character_id}", status_code=204)
def delete_character(project_id: str, character_id: str):
    project = get_project_or_404(project_id)
    characters = [item for item in project.get("characters", []) if item["id"] != character_id]
    if len(characters) == len(project.get("characters", [])):
        raise HTTPException(status_code=404, detail="Character not found")
    store.update(project_id, characters=characters)
    return None


@app.post("/api/projects/{project_id}/assets", status_code=201)
async def upload_asset(project_id: str, category: str = Form(...), file: UploadFile = File(...)):
    get_project_or_404(project_id)
    content = await file.read()
    try:
        asset = save_upload(store.project_dir(project_id), category, file.filename or "upload", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    project = store.get(project_id)
    assets = project.get("assets", [])
    assets.append(asset)
    store.update(project_id, assets=assets)
    return {key: value for key, value in asset.items() if key != "path"}


@app.get("/api/projects/{project_id}/assets")
def list_assets(project_id: str):
    project = get_project_or_404(project_id)
    return {"assets": [{key: value for key, value in item.items() if key != "path"} for item in project.get("assets", [])]}


@app.post("/api/projects/{project_id}/voices", status_code=201)
def add_voice(project_id: str, payload: VoiceCreate):
    project = get_project_or_404(project_id)
    if not payload.consent_confirmed:
        raise HTTPException(status_code=400, detail="Potwierdź zgodę na używanie tego głosu.")
    voices = project.get("voices", [])
    voice = {"id": __import__("uuid").uuid4().hex, **payload.model_dump()}
    voices.append(voice)
    store.update(project_id, voices=voices)
    return voice



@app.post("/api/projects/{project_id}/voices/clone", status_code=201)
async def clone_voice(project_id: str, name: str = Form(...), consent_confirmed: bool = Form(False), sample: UploadFile = File(...)):
    project = get_project_or_404(project_id)
    if not consent_confirmed:
        raise HTTPException(status_code=400, detail="Potwierdź, że masz prawo użyć tej próbki głosu.")
    suffix = Path(sample.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
        raise HTTPException(status_code=400, detail="Prześlij próbkę audio WAV, MP3, M4A, OGG lub FLAC.")
    content = await sample.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Próbka głosu przekracza limit 25 MB.")
    try:
        cloned = await ElevenLabsProvider().clone_voice(name, sample.filename or "voice_sample.wav", content)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    voice = {"id": __import__("uuid").uuid4().hex, "name": name, "provider": "elevenlabs",
             "provider_voice_id": cloned["voice_id"], "consent_confirmed": True}
    voices = project.get("voices", [])
    voices.append(voice)
    store.update(project_id, voices=voices)
    return voice

@app.post("/api/projects/{project_id}/speech", status_code=202)
def create_speech(project_id: str, payload: SpeechRequest, background: BackgroundTasks):
    get_project_or_404(project_id)
    job = create_job(project_id, "speech", payload.model_dump())
    background.add_task(run_tts, job["id"], store.project_dir(project_id), payload.model_dump())
    return job


@app.post("/api/projects/{project_id}/generate/{kind}", status_code=202)
def generate(project_id: str, kind: str, payload: GenerateRequest, background: BackgroundTasks):
    get_project_or_404(project_id)
    if kind not in {"image", "video", "music", "lipsync"}:
        raise HTTPException(status_code=404, detail="Unknown generation type")
    data = {**payload.inputs, "prompt": payload.prompt, "image_url": payload.image_url, "reference_asset_id": payload.reference_asset_id, "project_id": project_id}
    job = create_job(project_id, kind, data)
    background.add_task(run_generation, job["id"], store.project_dir(project_id), kind, data)
    return job


@app.get("/api/projects/{project_id}/jobs")
def project_jobs(project_id: str):
    get_project_or_404(project_id)
    return {"jobs": list_jobs(project_id)}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/projects/{project_id}/render")
def render_project(project_id: str, payload: RenderRequest):
    project = get_project_or_404(project_id)
    assets = {item["id"]: item for item in project.get("assets", [])}
    clips = []
    for asset_id in payload.asset_ids:
        asset = assets.get(asset_id)
        if not asset or asset.get("category") != "video":
            raise HTTPException(status_code=400, detail=f"Video asset not found: {asset_id}")
        clips.append(asset["path"])
    try:
        output = render_video(store.project_dir(project_id), clips)
    except RenderError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return FileResponse(output, media_type="video/mp4", filename="AI-Movie-Studio.mp4")



class ProductionStart(BaseModel):
    reference_asset_id: str | None = None


@app.post("/api/projects/{project_id}/production/start", status_code=202)
def start_full_production(project_id: str, payload: ProductionStart, background: BackgroundTasks):
    project = get_project_or_404(project_id)
    scenes = project.get("scenes") or []
    if not scenes:
        raise HTTPException(status_code=400, detail="Najpierw zapisz scenariusz i utwórz plan ujęć AI Director.")
    job = create_job(project_id, "full_production", {"reference_asset_id": payload.reference_asset_id})
    background.add_task(run_full_production, job["id"], store.project_dir(project_id), project_id, scenes, payload.reference_asset_id)
    store.update(project_id, status="producing")
    return job


@app.get("/api/jobs/{job_id}/download")
def download_job_output(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = job.get("result") or {}
    output = result.get("final_video") or result.get("path")
    if not output:
        raise HTTPException(status_code=404, detail="This job has no downloadable output.")
    path = Path(output).resolve()
    project_dir = store.project_dir(job["project_id"]).resolve()
    if not path.is_relative_to(project_dir) or not path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(path, media_type="video/mp4" if path.suffix == ".mp4" else "application/octet-stream", filename=path.name)
