import os
import httpx
from pathlib import Path
from app.services.jobs import update_job
from app.services.providers import ReplicateProvider, ElevenLabsProvider


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
            inputs = payload
        else:
            raise ValueError("Unsupported generation kind")
        result = await provider.run(model, inputs)
        outputs = result.get("output")
        urls = outputs if isinstance(outputs, list) else ([outputs] if isinstance(outputs, str) else [])
        saved = []
        ext = {"video": ".mp4", "image": ".png", "music": ".mp3", "lipsync": ".mp4"}[kind]
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
            for index, url in enumerate(urls):
                if not isinstance(url, str) or not url.startswith("https://"):
                    continue
                response = await client.get(url)
                response.raise_for_status()
                folder = project_dir / "assets" / ("audio" if kind == "music" else "video" if kind in {"video", "lipsync"} else "image")
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{kind}_{job_id}_{index}{ext}"
                target.write_bytes(response.content)
                saved.append(str(target))
        update_job(job_id, status="succeeded", progress=100, message="Provider finished", result={"provider_output": outputs, "saved_files": saved})
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
