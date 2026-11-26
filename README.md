# AI Movie Studio

Custom web-based AI film production workspace for Google Colab and local deployments.

**Current version: 0.2.0 — provider-backed production foundation.** This is a working application scaffold with real API integrations, not a claim that one click creates a finished 30-minute movie. Generation depends on your provider accounts, model availability, quotas and configuration.

## Included
- Custom responsive HTML/CSS/JavaScript studio UI. No ComfyUI frontend and no Gradio.
- FastAPI backend with OpenAPI docs.
- Persistent project/script/character/voice/asset metadata stored as JSON.
- LLM screenplay analysis through an OpenAI-compatible Chat Completions endpoint, with an explicit heuristic fallback when no key is configured.
- AI Director shot breakdown via LLM JSON output.
- Character profiles and reference image/audio/video uploads.
- Consent-confirmed voice cloning and speech generation through ElevenLabs.
- Replicate adapters for image, video, music and lip-sync models. Model identifiers are configured in environment variables.
- Background jobs persisted as JSON and visible in the UI, including sequential multi-shot production jobs.
- Generated output downloads into the project asset library when the provider returns HTTPS output URLs.
- FFmpeg MP4 concatenation/export for uploaded/generated video assets, including the multi-shot production output.
- Google Colab notebook with Drive persistence and Colab Secrets support.
- Dockerfile and GitHub Actions pytest workflow.

## Quick start (local)
Python 3.10+ and FFmpeg are recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000. API documentation: http://127.0.0.1:8000/docs

## Google Colab
[Open AI Movie Studio directly in Google Colab](https://colab.research.google.com/github/PandaCwel/Ai-video-maker/blob/main/colab/AI_Movie_Studio.ipynb), then run cells from top to bottom. [View notebook source](https://github.com/PandaCwel/Ai-video-maker/blob/main/colab/AI_Movie_Studio.ipynb). Add provider credentials under Colab **Secrets**. The notebook mounts Drive at `/content/drive/MyDrive/AI-Movie-Studio/data`.

## Provider configuration
Copy `.env.example` to `.env` and configure only the services you use:
- `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_MODEL` — script analysis and director planning.
- `REPLICATE_API_TOKEN` and `REPLICATE_IMAGE_MODEL`, `REPLICATE_VIDEO_MODEL`, `REPLICATE_MUSIC_MODEL`, `REPLICATE_LIPSYNC_MODEL` — model IDs in `owner/name` form. The model must accept the input fields passed by `app/services/production.py`; model schemas are not universal.
- `ELEVENLABS_API_KEY` — speech synthesis and voice cloning.

The UI and API report missing configuration instead of returning fabricated media. Provider pricing, model licenses, voice rights and data handling are the user's responsibility. Clone only voices you own or have explicit permission to use.

## API overview
- `GET /api/health`
- `GET/POST /api/projects`, `GET/PUT/DELETE /api/projects/{id}`
- `GET/PUT /api/projects/{id}/script`
- `POST /api/projects/{id}/script/analyze`
- `POST /api/projects/{id}/director/plan`
- `POST /api/projects/{id}/characters`
- `POST /api/projects/{id}/assets`
- `POST /api/projects/{id}/voices/clone` (requires explicit consent flag)
- `POST /api/projects/{id}/speech`
- `POST /api/projects/{id}/generate/{image|video|music|lipsync}`
- `GET /api/projects/{id}/jobs`, `GET /api/jobs/{job_id}`
- `POST /api/projects/{id}/render`

## Known limitations
- Replicate model input fields vary. The generic adapter sends a prompt for image/video/music; lip-sync accepts additional JSON inputs. Configure compatible models or adapt the provider mapping.
- Generated output files are registered when the provider returns direct HTTPS output URLs in its standard prediction response.
- The current render endpoint concatenates clips; it does not yet provide multi-track timeline editing, transitions, color grading, subtitles, shot-level audio mixing or a complete sound design suite.
- Job metadata survives restart, but a process interrupted mid-generation is not automatically resumed; rerun failed/interrupted tasks.
- Character identity consistency across shots depends on model-specific reference conditioning and is not guaranteed by the generic adapter.
- No user authentication or multi-user isolation is included. Do not expose this development server publicly without a secure auth/proxy layer.
- A 30-minute episode is a production target, not a guaranteed output duration. Generation cost, provider limits, shot count, GPU capacity and render time apply.

## Tests
Run `pytest -q`. GitHub Actions runs the test suite on pushes and pull requests. Do not treat a configured workflow as a passing test; check the Actions tab for the actual run result.

## License
No project license has been selected. All rights remain with their respective authors until a license is added.
