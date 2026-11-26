# AI Movie Studio — local-only edition

A self-hostable AI film workspace with a custom HTML/CSS/JS interface and FastAPI backend. **All AI inference runs locally in the Python runtime. No OpenAI, Replicate, ElevenLabs, hosted inference endpoint or API key is used.**

Public model weights are downloaded from their model repositories on first use and cached locally / on Google Drive. After the download, inference runs on the Colab GPU/CPU.

## Local engines
| Feature | Local engine | Notes |
|---|---|---|
| Screenplay analysis + AI Director | Qwen 2.5 3B Instruct | Transformers, 4-bit quantization when available |
| Still image generation | Stable Diffusion 1.5 | Diffusers; text-to-image and image-to-image |
| Short video generation | AnimateDiff + SD 1.5 + IP-Adapter | Generates short animated clips; uses selected character reference images when supplied |
| Voice cloning + speech | Coqui XTTS-v2 | Uses an uploaded reference voice sample; permission required |
| Music generation | MusicGen-small | Local short music clips |
| Lip-sync | Wav2Lip | Required for full production; code and checkpoint download on first production run |
| Editing/export | FFmpeg | Local MP4 assembly |

## Start in Google Colab
[Open the local-only notebook in Google Colab](https://colab.research.google.com/github/PandaCwel/Ai-video-maker/blob/main/colab/AI_Movie_Studio.ipynb).

1. Select a GPU runtime (Runtime → Change runtime type → GPU).
2. Run the notebook cells from top to bottom.
3. Mount Google Drive when asked. Project files and downloaded Hugging Face model cache are stored there.
4. Open the generated AI Movie Studio link.

No Secrets, API keys or provider accounts are needed. The notebook does download open model weights from public model repositories; this is not a hosted inference API call.

## Local installation
Python 3.10/3.11 and FFmpeg recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-local.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The local API status is available at `/api/runtime`.

## Configuration
Only local model identifiers and local paths are configurable:
- `LOCAL_LLM_MODEL`
- `LOCAL_IMAGE_MODEL`
- `LOCAL_VIDEO_MODEL`
- `LOCAL_MUSIC_MODEL`
- `WAV2LIP_CHECKPOINT` (optional override; default checkpoint is cached in the Drive models folder)
- `WAV2LIP_REPO` (optional source path; cloned on first lip-sync run)

There are no API-key variables in `.env.example`.

## API overview
- `GET /api/health`, `GET /api/runtime`
- `GET/POST /api/projects`, `GET/PUT/DELETE /api/projects/{id}`
- `GET/PUT /api/projects/{id}/script`
- `POST /api/projects/{id}/script/analyze`
- `POST /api/projects/{id}/director/plan`
- `POST /api/projects/{id}/characters`
- `POST /api/projects/{id}/assets`
- `POST /api/projects/{id}/voices/clone` (stores the reference sample locally; consent required)
- `POST /api/projects/{id}/speech` (local XTTS-v2)
- `POST /api/projects/{id}/generate/{image|video|music|lipsync}`
- `POST /api/projects/{id}/production/start`
- `GET /api/projects/{id}/jobs`, `GET /api/jobs/{job_id}`
- `POST /api/projects/{id}/render`

## Hardware and honest limitations
This is local inference, not unlimited compute. Colab Free does not guarantee a GPU, a specific VRAM size, uninterrupted sessions or enough storage. The local image, video, LLM, TTS, music and lip-sync models compete for GPU memory; engines are loaded only when requested, serialized to avoid overlap, and released after each job. Hugging Face weights are cached under `HF_HOME` (Google Drive in Colab). AnimateDiff creates short clips; a long episode requires many sequential shots and substantial runtime. It is not realistic to promise a polished 25–30 minute film from one Colab Free session.

- SD 1.5 / AnimateDiff quality and speed depend on GPU memory. CPU mode is extremely slow.
- Cross-shot identity consistency is not guaranteed by the base AnimateDiff adapter.
- XTTS-v2 supports voice cloning from a local reference recording; clone only voices you own or have explicit permission to use.
- AI Director creates cast cards automatically. Assign each cast member a reference image and a voice sample; video shots use IP-Adapter image conditioning and XTTS language settings. Full production requires Wav2Lip and a voice profile for every speaking character. The Wav2Lip code, GAN checkpoint and face-detection weights are downloaded on the first full-production request and cached on Google Drive. The official Wav2Lip open-source model is restricted to personal/research/non-commercial use; obtain separate permission for commercial use.
- The renderer concatenates normalized clips and mixes the generated score under the dialogue. It is not a full nonlinear editor with animated transitions, captions, grading or multi-track authoring.
- First-run model downloads require internet access and disk space. Model inference itself is local; weights are cached in Drive when using Colab.
- Model licenses and acceptable-use terms still apply.

## Tests
Run `python -m pytest -q`. GitHub Actions runs the backend test suite. The CI suite mocks heavy local model execution; it tests API behavior, frontend asset delivery, screenplay chunking and the required lip-sync preflight, not actual GPU image/video generation. Real inference must be tested in a GPU runtime.

## Safety
Voice cloning is gated by an explicit consent confirmation. Use only voice samples you own or have permission to use. Protect uploaded reference photos, voices and project data.
