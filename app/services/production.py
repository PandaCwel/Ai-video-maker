import os
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
        update_job(job_id, status="succeeded", progress=100, message="Provider finished", result=result)
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
