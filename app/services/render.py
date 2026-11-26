import shutil
import subprocess
from pathlib import Path


class RenderError(RuntimeError):
    pass


def render_video(project_dir: Path, clips: list[str], output_name: str = "final.mp4") -> Path:
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
    work = project_dir / "render"
    work.mkdir(parents=True, exist_ok=True)
    listing = work / "concat.txt"
    listing.write_text("".join("file '" + str(p).replace("'", "'\\''") + "'\n" for p in valid), encoding="utf-8")
    output = work / output_name
    command = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    if result.returncode:
        raise RenderError(result.stderr[-3000:])
    return output
