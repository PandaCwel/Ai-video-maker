# AI Movie Studio

A modular AI-assisted film production studio designed for Google Colab and a custom web interface.

> **Status:** early development. The current version provides the application shell, project/script APIs, local asset storage, and a browser UI. Actual generative video, voice cloning, lip-sync, and music providers are not yet implemented.

## Goals
- Custom HTML5/CSS3/JavaScript studio UI (no ComfyUI frontend, no Gradio).
- FastAPI backend with a documented REST API.
- Script ingestion and structured screenplay breakdown.
- Character reference and voice sample asset management.
- Provider adapters for LLM, image/video, TTS, music and lip-sync services.
- Resumable production jobs, checkpoints, FFmpeg assembly and MP4 export.
- Google Colab launcher with optional Google Drive persistence.

## Requirements
- Python 3.10+
- FFmpeg (required for final rendering; not required to start the API)
- Optional: API keys for external model providers

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs

## Google Colab
Open `colab/AI_Movie_Studio.ipynb` in Colab. The notebook installs dependencies and starts the same FastAPI application. Runtime GPU availability depends on the Colab session; heavy generation may require a remote provider or a suitable GPU.

## Current API
- `GET /api/health`
- `GET/POST /api/projects`
- `GET/PUT/DELETE /api/projects/{project_id}`
- `GET/PUT /api/projects/{project_id}/script`
- `POST /api/projects/{project_id}/script/analyze` (deterministic screenplay breakdown; LLM integration is a later milestone)

## Data and privacy
Project data is stored under `STUDIO_DATA_DIR` (default: `./data`). Do not commit reference photos, voice samples, API keys, generated media, or private scripts. Only upload voice samples you own or have permission to use.

## Roadmap
1. [x] API/UI foundation and project persistence
2. [ ] LLM screenplay analysis and editable scene breakdown
3. [ ] Character/voice asset studio
4. [ ] AI director and shot planning
5. [ ] Video/image provider adapters
6. [ ] Speech, lip-sync, music and sound design
7. [ ] Timeline editor, FFmpeg rendering and MP4 export
8. [ ] Drive checkpoints, resumable workflow and end-to-end tests

## License
No license has been selected yet. All rights remain with the respective authors until a license is added.
