import json
import shutil
import subprocess
from pathlib import Path

from app.services.jobs import update_job
from app.services.local_models import (
    LocalModelError, ensure_wav2lip, generate_video, generate_music,
    synthesize_voice, run_lipsync, release_vram,
)
from app.services.render import render_video


def _register(project_dir: Path, file: Path, category: str, job_id: str):
    meta = project_dir / "project.json"
    project = json.loads(meta.read_text(encoding="utf-8"))
    assets = project.setdefault("assets", [])
    asset = {
        "id": f"local-{job_id}-{len(assets)}", "name": file.name, "category": category,
        "path": str(file), "size": file.stat().st_size, "generated": True, "engine": "local",
    }
    assets.append(asset)
    from datetime import datetime, timezone
    project["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = meta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(meta)
    return asset


def _reference(project_dir: Path, asset_id):
    if not asset_id:
        return None
    project = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    asset = next((item for item in project.get("assets", []) if item.get("id") == asset_id and item.get("category") == "image"), None)
    if not asset:
        raise LocalModelError("Nie znaleziono obrazu referencyjnego.")
    path = Path(asset["path"])
    if not path.is_file():
        raise LocalModelError("Plik referencyjny nie istnieje.")
    return path


def _ffmpeg():
    binary = shutil.which("ffmpeg")
    if not binary:
        raise LocalModelError("FFmpeg nie jest zainstalowany. Uruchom komórkę instalacyjną notebooka.")
    return binary


def _dialogue_lines(shot):
    raw = shot.get("dialogue") or []
    if isinstance(raw, str):
        return [{"character": shot.get("speaker") or "", "text": raw}]
    if isinstance(raw, dict):
        raw = [raw]
    lines = []
    for item in raw:
        if isinstance(item, str):
            lines.append({"character": shot.get("speaker") or "", "text": item})
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("line") or "").strip()
            character = str(item.get("character") or item.get("speaker") or "").strip()
            if text:
                lines.append({"character": character, "text": text})
    return lines


def _voice_map(project):
    return {str(voice.get("name", "")).strip().casefold(): voice for voice in project.get("voices", []) if voice.get("sample_path")}


def _make_dialogue_audio(project_dir, job_id, shot_index, shot, voices):
    lines = _dialogue_lines(shot)
    if not lines:
        return None
    ffmpeg = _ffmpeg()
    folder = project_dir / "render" / f"audio_{job_id}_{shot_index:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    generated = []
    for line_index, line in enumerate(lines, 1):
        character = line["character"].strip().casefold()
        voice = voices.get(character)
        if not voice:
            raise LocalModelError(
                f"Brak profilu głosu dla postaci '{line['character']}'. "
                "Dodaj w Voice Studio próbkę głosu o identycznej nazwie jak postać."
            )
        audio = folder / f"line_{line_index:03d}.wav"
        synthesize_voice(line["text"], Path(voice["sample_path"]), audio, voice.get("language", "pl"))
        generated.append(audio)
        update_job(job_id, message=f"XTTS: dialog {line_index}/{len(lines)} w ujęciu {shot_index}")
    listing = folder / "concat.txt"
    listing.write_text("".join("file '" + str(path.resolve()).replace("'", "'\\''") + "'\n" for path in generated), encoding="utf-8")
    output = folder / "dialogue.wav"
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-ar", "22050", "-ac", "1", str(output)],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode or not output.is_file() or output.stat().st_size == 0:
        raise LocalModelError("Nie udało się połączyć dialogów: " + result.stderr[-2000:])
    return output


def run_generation(job_id, project_dir, kind, payload):
    update_job(job_id, status="running", progress=2, message="Uruchamianie lokalnego modelu; pobieranie nastąpi tylko przy pierwszym użyciu.")
    try:
        ref = _reference(project_dir, payload.get("reference_asset_id"))
        category = "audio" if kind == "music" else "image" if kind == "image" else "video"
        folder = project_dir / "assets" / category
        folder.mkdir(parents=True, exist_ok=True)
        if kind == "image":
            from app.services.local_models import generate_image
            out = generate_image(payload["prompt"], folder / f"image_{job_id}.png", ref)
        elif kind == "video":
            out = generate_video(payload["prompt"], folder / f"video_{job_id}.mp4", ref)
        elif kind == "music":
            out = generate_music(payload["prompt"], folder / f"music_{job_id}.wav")
        elif kind == "lipsync":
            project = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
            video = next((Path(item["path"]) for item in project.get("assets", []) if item.get("id") == payload.get("video_asset_id") and item.get("category") == "video"), None)
            audio = next((Path(item["path"]) for item in project.get("assets", []) if item.get("id") == payload.get("audio_asset_id") and item.get("category") == "audio"), None)
            if not video or not audio:
                raise LocalModelError("Wybierz istniejący klip wideo i ścieżkę audio.")
            ensure_wav2lip()
            out = run_lipsync(video, audio, folder / f"lipsync_{job_id}.mp4")
        else:
            raise LocalModelError("Nieznany typ generowania.")
        _register(project_dir, out, category, job_id)
        update_job(job_id, status="succeeded", progress=100, message="Lokalne generowanie zakończone", result={"saved_files": [str(out)], "engine": "local"})
    except Exception as exc:
        update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))
    finally:
        release_vram()


def run_tts(job_id, project_dir, payload):
    update_job(job_id, status="running", progress=5, message="Ładowanie XTTS-v2 na czas generowania mowy")
    try:
        project = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
        voice = next((item for item in project.get("voices", []) if item.get("id") == payload["voice_id"]), None)
        if not voice:
            raise LocalModelError("Nie znaleziono profilu głosu.")
        output = project_dir / "assets" / "audio" / f"speech_{job_id}.wav"
        synthesize_voice(payload["text"], Path(voice["sample_path"]), output, voice.get("language", "pl"))
        _register(project_dir, output, "audio", job_id)
        update_job(job_id, status="succeeded", progress=100, message="Wygenerowano mowę lokalnie", result={"path": str(output), "engine": "XTTS-v2 local"})
    except Exception as exc:
        update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))
    finally:
        release_vram()


def validate_full_production(project_dir: Path, scenes, prepare_lipsync: bool = True):
    """Fail before expensive rendering if required voice/lip-sync assets are missing."""
    project = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    voices = _voice_map(project)
    shots = [shot for scene in scenes if isinstance(scene, dict) for shot in scene.get("shots", []) if isinstance(shot, dict)]
    if not shots:
        raise LocalModelError("Plan AI Director nie zawiera ujęć.")
    all_lines = [line for shot in shots for line in _dialogue_lines(shot)]
    if not all_lines:
        raise LocalModelError("Pełna produkcja wymaga dialogów, aby wykonać obowiązkowy lip-sync.")
    missing = sorted({line["character"] for line in all_lines if line["character"].strip().casefold() not in voices})
    if missing:
        raise LocalModelError("Brak profili głosu dla postaci: " + ", ".join(missing) + ". Dodaj próbki w Voice Studio, używając identycznych nazw.")
    for voice in voices.values():
        sample = Path(voice["sample_path"])
        if not sample.is_file():
            raise LocalModelError(f"Brak pliku próbki głosu: {sample}")
    if prepare_lipsync:
        ensure_wav2lip()
    return shots, voices


def run_full_production(job_id, project_dir, project_id, scenes, reference_asset_id=None):
    update_job(job_id, status="running", progress=1, message="Sprawdzanie wymaganych zasobów: Wav2Lip, głosy i FFmpeg")
    try:
        shots, voices = validate_full_production(project_dir, scenes)
        ffmpeg = _ffmpeg()
        reference = _reference(project_dir, reference_asset_id)
        clips = []
        video_dir = project_dir / "assets" / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        for index, shot in enumerate(shots, 1):
            prompt = shot.get("visual_prompt") or shot.get("prompt") or shot.get("summary")
            if not prompt:
                raise LocalModelError(f"Ujęcie {index} nie ma visual_prompt/prompt/summary.")
            raw_clip = video_dir / f"shot_{index:04d}_{job_id[:8]}_raw.mp4"
            final_clip = video_dir / f"shot_{index:04d}_{job_id[:8]}_lipsync.mp4"
            update_job(job_id, progress=max(2, int((index - 1) / len(shots) * 85)), message=f"AnimateDiff: generowanie ujęcia {index}/{len(shots)}")
            generate_video(prompt, raw_clip, reference)
            _register(project_dir, raw_clip, "video", f"{job_id}-{index}-raw")
            audio = _make_dialogue_audio(project_dir, job_id, index, shot, voices)
            if audio is None:
                # Silent shots still receive a silent audio stream so FFmpeg concat remains deterministic.
                silent_clip = video_dir / f"shot_{index:04d}_{job_id[:8]}_silent.mp4"
                silent = subprocess.run(
                    [ffmpeg, "-y", "-i", str(raw_clip), "-f", "lavfi", "-i",
                     "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(silent_clip)],
                    capture_output=True, text=True, timeout=1800,
                )
                if silent.returncode or not silent_clip.is_file():
                    raise LocalModelError("Nie udało się dodać ciszy do ujęcia: " + silent.stderr[-1500:])
                clips.append(str(silent_clip))
            else:
                update_job(job_id, message=f"Wav2Lip: synchronizacja ust w ujęciu {index}/{len(shots)}")
                run_lipsync(raw_clip, audio, final_clip)
                _register(project_dir, final_clip, "video", f"{job_id}-{index}-lipsync")
                clips.append(str(final_clip))
            update_job(job_id, progress=min(88, int(index / len(shots) * 88)), message=f"Gotowe ujęcie {index}/{len(shots)}")
            release_vram()

        if not clips:
            raise LocalModelError("Nie powstały żadne klipy.")
        # Generate score only after all video/TTS/Wav2Lip work has released the GPU.
        soundtrack = project_dir / "assets" / "audio" / f"score_{job_id[:8]}.wav"
        soundtrack.parent.mkdir(parents=True, exist_ok=True)
        score_prompt = "Cinematic instrumental film score, emotional orchestral underscore, no vocals, matching the mood of: " + str(
            next((scene.get("mood") for scene in scenes if isinstance(scene, dict) and scene.get("mood")), "dramatic science fiction")
        )
        generate_music(score_prompt, soundtrack, seconds=30)
        _register(project_dir, soundtrack, "audio", f"{job_id}-score")
        update_job(job_id, progress=92, message="FFmpeg: montaż obrazu, dialogów i ścieżki muzycznej")
        final = render_video(project_dir, clips, output_name=f"episode_{job_id[:8]}.mp4", soundtrack=soundtrack)
        update_job(job_id, status="succeeded", progress=100, message=f"Ukończono film: {len(clips)} ujęć; dialogi XTTS, lip-sync Wav2Lip, muzyka i MP4", result={
            "clip_count": len(clips), "clips": clips, "soundtrack": str(soundtrack),
            "final_video": str(final), "engine": "local", "lipsync_required": True,
        })
    except Exception as exc:
        update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))
    finally:
        release_vram()
