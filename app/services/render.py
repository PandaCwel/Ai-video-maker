import shutil
import subprocess
from pathlib import Path


class RenderError(RuntimeError):
    pass


def _run(command, timeout=7200):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RenderError(result.stderr[-4000:] or "FFmpeg zakończył się błędem.")
    return result


def render_video(project_dir: Path, clips: list[str], output_name: str = "final.mp4", soundtrack: Path | None = None) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RenderError("FFmpeg nie jest zainstalowany lub nie znajduje się w PATH.")
    valid = []
    for item in clips:
        path = Path(item).resolve()
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".webm"}:
            raise RenderError(f"Nieprawidłowy klip: {item}")
        valid.append(path)
    if not valid:
        raise RenderError("Brak klipów do montażu.")
    if soundtrack is not None:
        soundtrack = Path(soundtrack).resolve()
        if not soundtrack.is_file():
            raise RenderError(f"Nie znaleziono ścieżki muzycznej: {soundtrack}")
    work = project_dir / "render"
    work.mkdir(parents=True, exist_ok=True)
    listing = work / "concat.txt"
    listing.write_text("".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in valid), encoding="utf-8")
    output = work / output_name
    intermediate = work / (Path(output_name).stem + "_picture.mp4") if soundtrack else output
    command = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(intermediate),
    ]
    _run(command)
    if soundtrack:
        # Preserve dialogue from Wav2Lip and duck the score under it.
        mixed = [
            ffmpeg, "-y", "-i", str(intermediate), "-stream_loop", "-1", "-i", str(soundtrack),
            "-filter_complex",
            "[0:a]volume=1.0[dialogue];[1:a]volume=0.18[score];"
            "[dialogue][score]amix=inputs=2:duration=first:dropout_transition=2[a]",
            "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output),
        ]
        _run(mixed)
        try:
            intermediate.unlink()
        except OSError:
            pass
    if not output.is_file() or output.stat().st_size == 0:
        raise RenderError("FFmpeg nie utworzył prawidłowego pliku MP4.")
    return output
