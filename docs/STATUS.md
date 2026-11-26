# Development status

## Implemented in v0.1.0
- FastAPI application and health endpoint.
- Custom responsive HTML/CSS/JavaScript dashboard.
- Project create/list/read/update/delete operations.
- Script save/load endpoints.
- Baseline screenplay parser that detects common scene headings and likely character cues.
- Colab notebook that mounts Drive, installs requirements, starts the server and prints a proxy URL.
- Initial API tests and GitHub Actions test workflow.

## Explicitly not implemented yet
- LLM screenplay analysis (the current parser is deterministic and heuristic).
- Character image generation or identity consistency.
- Voice cloning / TTS provider integration.
- Text-to-video or image-to-video generation.
- Lip-sync, generated soundtrack or sound effects.
- Timeline editing, render queue, FFmpeg export, job resume/checkpoint system.
- Authentication, multi-user isolation or secure public deployment.

## Validation
The repository files and notebook JSON were fetched back from GitHub after commits. The notebook JSON parsed successfully. The test suite has **not** been executed in a Python runtime in this session because the execution environment could not resolve github.com to clone the repository. GitHub Actions is configured to run pytest on pushes and pull requests; check the Actions tab for the first CI result.
