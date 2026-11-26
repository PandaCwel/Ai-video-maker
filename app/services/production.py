import json
import os
from pathlib import Path
import httpx
from app.services.jobs import update_job
from app.services.providers import ReplicateProvider, ElevenLabsProvider


def _register_generated_assets(project_dir: Path, project_id: str, files: list[str], category: str, job_id: str):
    metadata = project_dir / "project.json"
    if not metadata.is_file() or not files:
        return
    project = json.loads(metadata.read_text(encoding="utf-8"))
    assets = project.setdefault("assets", [])
    for raw in files:
        path = Path(raw)
        assets.append({"id": f"generated-{job_id}-{len(assets)}", "name": path.name,
                       "category": category, "path": str(path), "size": path.stat().st_size,
                       "generated": True})
    project["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    tmp = metadata.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(metadata)


async def run_generation(job_id: str, project_dir: Path, kind: str, payload: dict):
    update_job(job_id, status="running", progress=5, message="Submitting request to provider")
    try:
        provider = ReplicateProvider()
        if kind == "video":
            model = os.getenv("REPLICATE_VIDEO_MODEL", "")
            inputs = {"prompt": payload["prompt"]}
            if payload.get("image_url"): inputs["image"] = payload["image_url"]
        elif kind == "image":
            model = os.getenv("REPLICATE_IMAGE_MODEL", "")
            inputs = {"prompt": payload["prompt"]}
        elif kind == "music":
            model = os.getenv("REPLICATE_MUSIC_MODEL", "")
            inputs = {"prompt": payload["prompt"]}
        elif kind == "lipsync":
            model = os.getenv("REPLICATE_LIPSYNC_MODEL", "")
            inputs = {k: v for k, v in payload.items() if k != "prompt"}
            if payload.get("prompt"): inputs["prompt"] = payload["prompt"]
        else:
            raise ValueError("Unsupported generation kind")
        result = await provider.run(model, inputs)
        outputs = result.get("output")
        urls = outputs if isinstance(outputs, list) else ([outputs] if isinstance(outputs, str) else [])
        saved = []
        category = "audio" if kind == "music" else "image" if kind == "image" else "video"
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
            for index, url in enumerate(urls):
                if not isinstance(url, str) or not url.startswith("https://"):
                    continue
                response = await client.get(url)
                response.raise_for_status()
                suffix = Path(url.split("?", 1)[0]).suffix
                if suffix.lower() not in {".mp4", ".webm", ".mov", ".png", ".jpg", ".jpeg", ".webp", ".mp3", ".wav", ".m4a"}:
                    suffix = { "video": ".mp4", "image": ".png", "music": ".mp3", "lipsync": ".mp4" }[kind]
                folder = project_dir / "assets" / category
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{kind}_{job_id}_{index}{suffix}"
                target.write_bytes(response.content)
                saved.append(str(target))
        _register_generated_assets(project_dir, payload.get("project_id", ""), saved, category, job_id)
        update_job(job_id, status="succeeded", progress=100, message="Provider finished",
                   result={"provider_output": outputs, "saved_files": saved})
    except Exception as exc:
        update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))


async def run_tts(job_id: str, project_dir: Path, payload: dict):
    update_job(job_id, status="running", progress=10, message="Generating speech")
    try:
        output = project_dir / "assets" / "audio" / f"speech_{job_id}.mp3"
        await ElevenLabsProvider().synthesize(payload["voice_id"], payload["text"], output)
        update_job(job_id, status="succeeded", progress=100, message="Speech generated", result={"path": str(output)})
    except Exception as exc:
        update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))
