"""On-demand local inference. Models are downloaded to HF_HOME on first use and never kept resident between jobs."""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path
from threading import RLock

_MODEL_LOCK = RLock()


class LocalModelError(RuntimeError):
    pass


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise LocalModelError("Brak PyTorch. Uruchom instalator modeli w notebooku.") from exc


def release_vram():
    """Release references held by callers before collecting and emptying CUDA allocator."""
    gc.collect()
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _dtype(torch):
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _model_cache():
    root = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def local_status():
    try:
        torch = _torch()
        cuda = torch.cuda.is_available()
        return {
            "mode": "local-only", "api_keys_required": False, "torch": torch.__version__,
            "cuda_available": cuda,
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if cuda else 0,
            "model_policy": "download-on-first-use; one engine at a time; VRAM released after each job",
            "cache_dir": str(_model_cache()),
            "models": {
                "llm": os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
                "image": os.getenv("LOCAL_IMAGE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
                "video": os.getenv("LOCAL_VIDEO_MODEL", "guoyww/animatediff-motion-adapter-v1-5-2"),
                "music": os.getenv("LOCAL_MUSIC_MODEL", "facebook/musicgen-small"),
                "voice": "tts_models/multilingual/multi-dataset/xtts_v2",
                "ip_adapter": os.getenv("LOCAL_IP_ADAPTER_MODEL", "h94/IP-Adapter"),
                "lipsync": "Wav2Lip (required checkpoint)",
            },
            "lipsync_ready": bool(
                os.getenv("WAV2LIP_CHECKPOINT") and Path(os.getenv("WAV2LIP_CHECKPOINT", "")).is_file()
                and (Path(os.getenv("WAV2LIP_REPO", "vendor/Wav2Lip")) / "inference.py").is_file()
                and (Path(os.getenv("WAV2LIP_REPO", "vendor/Wav2Lip")) / "face_detection/detection/sfd/s3fd.pth").is_file()
            ),
        }
    except Exception as exc:
        return {"mode": "local-only", "api_keys_required": False, "error": str(exc)}


def local_llm_json(system: str, prompt: str, max_new_tokens=2500):
    torch = _torch()
    model = tokenizer = inputs = output = None
    with _MODEL_LOCK:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            model_id = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(_model_cache()))
            kwargs = {"torch_dtype": _dtype(torch), "low_cpu_mem_usage": True}
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
                try:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                    )
                except Exception:
                    pass
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=str(_model_cache()), **kwargs)
            if not torch.cuda.is_available():
                model.to("cpu")
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
            answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            match = re.search(r"\{.*\}", answer, re.S)
            if not match:
                raise LocalModelError("Lokalny LLM nie zwrócił poprawnego JSON. Spróbuj podzielić scenariusz na krótsze części.")
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LocalModelError(f"LLM zwrócił niepoprawny JSON: {exc}") from exc
        except LocalModelError:
            raise
        except Exception as exc:
            raise LocalModelError(f"Błąd lokalnego LLM: {exc}") from exc
        finally:
            del output, inputs, model, tokenizer
            release_vram()


def generate_image(prompt: str, output: Path, reference: Path | None = None):
    torch = _torch()
    pipe = image = None
    with _MODEL_LOCK:
        try:
            from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
            model_id = os.getenv("LOCAL_IMAGE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
            common = {"torch_dtype": _dtype(torch), "cache_dir": str(_model_cache()), "safety_checker": None}
            if reference:
                from PIL import Image
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, **common)
                image = Image.open(reference).convert("RGB").resize((512, 512))
                call = lambda: pipe(prompt=prompt, image=image, strength=.65, guidance_scale=7, num_inference_steps=22).images[0]
            else:
                pipe = StableDiffusionPipeline.from_pretrained(model_id, **common)
                call = lambda: pipe(prompt=prompt, guidance_scale=7, num_inference_steps=22, height=512, width=512).images[0]
            pipe.enable_attention_slicing()
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cpu")
            output.parent.mkdir(parents=True, exist_ok=True)
            call().save(output)
            return output
        except Exception as exc:
            raise LocalModelError(f"Lokalne generowanie obrazu: {exc}") from exc
        finally:
            del pipe, image
            release_vram()


def generate_video(prompt: str, output: Path, reference: Path | None = None, frames=16, fps=8):
    torch = _torch()
    pipe = adapter = None
    with _MODEL_LOCK:
        try:
            import imageio.v2 as imageio
            from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
            base = os.getenv("LOCAL_IMAGE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
            motion = os.getenv("LOCAL_VIDEO_MODEL", "guoyww/animatediff-motion-adapter-v1-5-2")
            adapter = MotionAdapter.from_pretrained(motion, torch_dtype=_dtype(torch), cache_dir=str(_model_cache()))
            pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=_dtype(torch), cache_dir=str(_model_cache()))
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
            pipe.enable_vae_slicing()
            pipe.enable_attention_slicing()
            call_kwargs = {}
            if reference:
                from PIL import Image
                if not hasattr(pipe, "load_ip_adapter"):
                    raise LocalModelError("Ta wersja Diffusers nie obsługuje IP-Adapter dla AnimateDiff. Zaktualizuj requirements-local.txt.")
                adapter_id = os.getenv("LOCAL_IP_ADAPTER_MODEL", "h94/IP-Adapter")
                pipe.load_ip_adapter(adapter_id, subfolder="models", weight_name="ip-adapter_sd15.bin")
                pipe.set_ip_adapter_scale(float(os.getenv("LOCAL_IP_ADAPTER_SCALE", "0.65")))
                with Image.open(reference) as ref_image:
                    call_kwargs["ip_adapter_image"] = ref_image.convert("RGB")
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cpu")
            result = pipe(prompt=prompt, num_frames=max(8, min(int(frames), 24)), guidance_scale=7,
                           num_inference_steps=18, **call_kwargs).frames[0]
            output.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(output, result, fps=fps, macro_block_size=1)
            return output
        except Exception as exc:
            raise LocalModelError(f"Lokalne generowanie wideo AnimateDiff: {exc}") from exc
        finally:
            del pipe, adapter
            release_vram()


def generate_music(prompt: str, output: Path, seconds=12):
    torch = _torch()
    model = processor = inputs = audio = None
    with _MODEL_LOCK:
        try:
            import scipy.io.wavfile
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            model_id = os.getenv("LOCAL_MUSIC_MODEL", "facebook/musicgen-small")
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(_model_cache()))
            model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=_dtype(torch), cache_dir=str(_model_cache()))
            if torch.cuda.is_available():
                model.to("cuda")
            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
            token_count = min(max(int(seconds), 5), 30) * 50
            with torch.inference_mode():
                audio = model.generate(**inputs, max_new_tokens=token_count)
            wav = audio[0].detach().float().cpu().numpy()
            if wav.ndim == 2:
                wav = wav.T
            output.parent.mkdir(parents=True, exist_ok=True)
            scipy.io.wavfile.write(str(output), model.config.audio_encoder.sampling_rate, wav)
            return output
        except Exception as exc:
            raise LocalModelError(f"Lokalne generowanie muzyki MusicGen: {exc}") from exc
        finally:
            del audio, inputs, model, processor
            release_vram()


def synthesize_voice(text: str, sample: Path, output: Path, language="pl"):
    torch = _torch()
    engine = None
    with _MODEL_LOCK:
        try:
            from TTS.api import TTS
            if not sample.is_file():
                raise LocalModelError(f"Nie znaleziono próbki głosu: {sample}")
            os.environ.setdefault("TTS_HOME", str(Path.home() / ".local" / "share" / "tts"))
            engine = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=torch.cuda.is_available())
            output.parent.mkdir(parents=True, exist_ok=True)
            engine.tts_to_file(text=text, speaker_wav=str(sample), language=language, file_path=str(output))
            if not output.is_file() or output.stat().st_size == 0:
                raise LocalModelError("XTTS nie utworzył pliku audio.")
            return output
        except LocalModelError:
            raise
        except Exception as exc:
            raise LocalModelError(f"Lokalne klonowanie głosu XTTS-v2: {exc}") from exc
        finally:
            del engine
            release_vram()


def ensure_wav2lip():
    """Download Wav2Lip code and weights only when the first lip-sync job is requested."""
    checkpoint = Path(os.getenv("WAV2LIP_CHECKPOINT", str(Path(os.getenv("STUDIO_DATA_DIR", "./data")).parent / "models" / "wav2lip_gan.pth")))
    repo = Path(os.getenv("WAV2LIP_REPO", "vendor/Wav2Lip"))
    script = repo / "inference.py"
    try:
        if not script.is_file():
            if repo.exists():
                shutil.rmtree(repo)
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Rudrabha/Wav2Lip.git", str(repo)],
                           check=True, capture_output=True, text=True, timeout=600)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            subprocess.run([sys.executable, "-m", "gdown",
                            "https://drive.google.com/uc?id=15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk",
                            "-O", str(checkpoint)],
                           check=True, capture_output=True, text=True, timeout=1800)
        detector = repo / "face_detection" / "detection" / "sfd" / "s3fd.pth"
        if not detector.is_file():
            detector.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth", detector
            )
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1024 * 1024:
            raise LocalModelError("Pobrany checkpoint Wav2Lip jest niekompletny.")
        if not detector.is_file() or detector.stat().st_size < 1024 * 1024:
            raise LocalModelError("Nie udało się pobrać wymaganego modelu detekcji twarzy S3FD.")
        return checkpoint.resolve(), script.resolve()
    except LocalModelError:
        raise
    except Exception as exc:
        raise LocalModelError(
            "Nie udało się przygotować wymaganego Wav2Lip. Sprawdź internet, gdown i miejsce na Google Drive: " + str(exc)
        ) from exc


def run_lipsync(video: Path, audio: Path, output: Path):
    checkpoint, script = ensure_wav2lip()
    if not video.is_file() or not audio.is_file():
        raise LocalModelError("Lip-sync wymaga istniejącego pliku wideo i audio.")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script.resolve()), "--checkpoint_path", str(checkpoint.resolve()), "--face", str(video.resolve()), "--audio", str(audio.resolve()), "--outfile", str(output.resolve()), "--nosmooth"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=7200, cwd=str(script.parent))
        if proc.returncode:
            raise LocalModelError("Wav2Lip zakończył się błędem: " + proc.stderr[-3000:])
        if not output.is_file() or output.stat().st_size == 0:
            raise LocalModelError("Wav2Lip nie utworzył pliku wynikowego.")
        return output
    finally:
        release_vram()
