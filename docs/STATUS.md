# Implementation status — v0.2.0

## Implemented
- Custom responsive browser UI and FastAPI API.
- Project CRUD, script persistence, characters, asset uploads and voice profiles.
- OpenAI-compatible screenplay analysis; explicit baseline parser fallback.
- LLM AI Director JSON shot planning.
- Replicate API adapter for image/video/music/lip-sync prediction workflows.
- ElevenLabs voice clone endpoint gated by an explicit user consent confirmation, and speech synthesis.
- Background job state stored as JSON under STUDIO_DATA_DIR/jobs.
- Provider output download and registration in project asset metadata for standard HTTPS output URLs.
- Automated multi-shot workflow: iterates through planned shot prompts, requests video generation, downloads clips and assembles a sequential MP4.
- Colab notebook that clones the repo, installs FFmpeg/dependencies, mounts Drive, reads Colab Secrets and exposes the web UI.
- Dockerfile, configuration example, tests and CI workflow.

## Not fully solved / production caveats
- Model-specific schemas differ; users must configure compatible Replicate models and may need to adapt input mappings.
- Lip-sync requires model-specific input assets/URLs; the UI currently exposes a generic prompt and does not yet upload/host signed input URLs for every model.
- Character reference conditioning and cross-shot identity consistency depend on selected model capabilities and are not guaranteed by the generic adapter.
- Automated episode mode currently generates planned video shots sequentially and concatenates them; it does not yet provide a full nonlinear multitrack timeline, transitions, subtitles, audio mixing or color grading.
- Jobs are persisted, but interrupted external predictions are not automatically resumed.
- No auth or multi-user isolation; not safe for public exposure as-is.
- Full 25–30 minute film generation requires significant provider credits/GPU, multiple shots and production iterations. Colab Free does not guarantee that compute.
- The test suite must be executed in CI/Colab; no claim of passing tests is made until an actual run reports success.

## Safety and rights
Voice cloning endpoint requires an explicit consent checkbox. Only clone a voice you own or have permission to use. Check model/provider terms, privacy and licensing before commercial use.
