# Implementation status — local-only transition

## Local-only architecture committed
- Removed OpenAI-compatible, Replicate and ElevenLabs network inference from application workflows.
- Added local inference module for Qwen 2.5 3B, Stable Diffusion 1.5, AnimateDiff, MusicGen-small and Coqui XTTS-v2.
- Screenplay analysis and AI Director call local Transformers inference.
- Voice sample upload now stores the sample in the project and local XTTS uses it for cloning/speech.
- Image, video, music and lip-sync jobs now dispatch to local engines.
- Runtime diagnostics report CUDA/GPU/VRAM and explicitly identify local-only mode.
- Colab notebook no longer reads Secrets or configures API keys; it mounts Drive and persists Hugging Face model cache.
- Added separate `requirements-local.txt` so basic API tests do not need heavyweight GPU packages.
- FFmpeg remains the local editor/exporter.

## Features requiring user-provided local assets
- Wav2Lip inference requires the Wav2Lip source tree and a checkpoint path. The checkpoint is not redistributed here.
- Voice cloning requires a reference voice sample and explicit rights/consent.

## Limitations
- This is not a guarantee of a complete 30-minute feature film in Colab Free. Video generation uses short AnimateDiff clips and a multi-shot episode may take hours or exceed available VRAM/session limits.
- Base AnimateDiff does not guarantee consistent character identity from shot to shot; the reference-image option currently guides the prompt rather than implementing a dedicated identity adapter.
- MusicGen-small creates short sound/music clips, not a full soundtrack with automatic score composition.
- The renderer currently concatenates clips; it is not a nonlinear multitrack timeline with advanced transitions, subtitles, grading and audio mixing.
- First run downloads public model weights. Those downloads are not API inference requests, but require internet and disk space.
- CI mocks heavy model inference. A successful CI run does not prove a specific Colab GPU can load all models; GPU smoke testing is still required.

## Safety
Voice cloning requires an explicit consent confirmation. Only use voice samples with permission. Respect each model's license and terms.
