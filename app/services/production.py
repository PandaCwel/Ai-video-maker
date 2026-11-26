import base64
import json
import mimetypes
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
            if payload.get("image_url"):
                inputs["image"] = payload["image_url"]
            elif payload.get("reference_asset_id"):
                project_meta = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
                asset = next((a for a in project_meta.get("assets", []) if a.get("id") == payload["reference_asset_id"] and a.get("category") == "image"), None)
                if not asset:
                    raise ValueError("Wybrane zdjęcie referencyjne nie istnieje w bibliotece projektu.")
                image_path = Path(asset["path"])
                if image_path.stat().st_size > 10 * 1024 * 1024:
                    raise ValueError("Zdjęcie referencyjne przekracza limit 10 MB dla data URI.")
                mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
                inputs["image"] = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
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
        project_file = project_dir / "project.json"
        if project_file.is_file():
            project = json.loads(project_file.read_text(encoding="utf-8"))
            assets = project.setdefault("assets", [])
            assets.append({"id": f"speech-{job_id}", "name": output.name, "category": "audio",
                           "path": str(output), "size": output.stat().st_size, "generated": True})
            project["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
            tmp = project_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(project_file)
        update_job(job_id, status="succeeded", progress=100, message="Speech generated", result={"path": str(output)})
    except Exception as exc:
        update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))



async def run_full_production(job_id: str, project_dir: Path, project_id: str, scenes: list[dict], reference_asset_id: str | None = None):
    """Generate each planned shot sequentially and assemble available video outputs."""
    update_job(job_id, status="running", progress=1, message="Preparing shot list")
    try:
        model = os.getenv("REPLICATE_VIDEO_MODEL", "")
        provider = ReplicateProvider()
        if not scenes:
            raise ValueError("Brak planu scen. Najpierw uruchom AI Director.")
        project_meta = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        reference = None
        if reference_asset_id:
            reference = next((a for a in project_meta.get("assets", []) if a.get("id") == reference_asset_id and a.get("category") == "image"), None)
            if not reference:
                raise ValueError("Zdjęcie referencyjne nie istnieje.")
        reference_data = None
        if reference:
            ref_path = Path(reference["path"])
            if ref_path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("Zdjęcie referencyjne przekracza limit 10 MB.")
            mime = __import__("mimetypes").guess_type(ref_path.name)[0] or "image/jpeg"
            reference_data = f"data:{mime};base64," + __import__("base64").b64encode(ref_path.read_bytes()).decode("ascii")
        shots = []
        for scene in scenes:
            shots.extend(scene.get("shots", []) if isinstance(scene, dict) else [])
        if not shots:
            raise ValueError("Plan AI Director nie zawiera ujęć.")
        clips = []
        total = len(shots)
        for index, shot in enumerate(shots, start=1):
            prompt = shot.get("visual_prompt") or shot.get("prompt") or shot.get("summary")
            if not prompt:
                continue
            inputs = {"prompt": prompt}
            if reference_data:
                inputs["image"] = reference_data
            result = await provider.run(model, inputs)
            output = result.get("output")
            urls = output if isinstance(output, list) else ([output] if isinstance(output, str) else [])
            if not urls:
                raise ValueError(f"Model nie zwrócił pliku dla ujęcia {index}.")
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                response = await client.get(urls[0])
                response.raise_for_status()
            folder = project_dir / "assets" / "video"
            folder.mkdir(parents=True, exist_ok=True)
            clip_path = folder / f"shot_{index:04d}_{job_id[:8]}.mp4"
            clip_path.write_bytes(response.content)
            clips.append(str(clip_path))
            _register_generated_assets(project_dir, project_id, [str(clip_path)], "video", f"{job_id}-{index}")
            progress = min(90, int(index / total * 90))
            update_job(job_id, progress=progress, message=f"Generated shot {index}/{total}")
        if not clips:
            raise ValueError("Żadne ujęcie nie zwróciło klipu wideo.")
        from app.services.render import render_video
        final_path = render_video(project_dir, clips, output_name=f"episode_{job_id[:8]}.mp4")
        update_job(job_id, status="succeeded", progress=100, message=f"Episode assembled from {len(clips)} generated shots",
                   result={"clip_count": len(clips), "clips": clips, "final_video": str(final_path)})
    except Exception as exc:
        update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))
