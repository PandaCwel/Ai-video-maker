"""Local-only inference. Model weights download from public model repositories, then run locally."""
from __future__ import annotations
import gc, json, os, re, subprocess
from pathlib import Path


class LocalModelError(RuntimeError): pass


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise LocalModelError("Brak PyTorch. Uruchom instalator lokalnych modeli w notebooku.") from exc


def _cleanup():
    gc.collect()
    try:
        t = _torch()
        if t.cuda.is_available(): t.cuda.empty_cache()
    except Exception: pass


def local_status():
    try:
        t = _torch(); cuda = t.cuda.is_available()
        return {"mode":"local-only","api_keys_required":False,"torch":t.__version__,
                "cuda_available":cuda,"gpu":t.cuda.get_device_name(0) if cuda else None,
                "vram_gb":round(t.cuda.get_device_properties(0).total_memory/1024**3,1) if cuda else 0,
                "models":{"llm":os.getenv("LOCAL_LLM_MODEL","Qwen/Qwen2.5-3B-Instruct"),
                          "image":os.getenv("LOCAL_IMAGE_MODEL","stable-diffusion-v1-5/stable-diffusion-v1-5"),
                          "video":"AnimateDiff + Stable Diffusion 1.5",
                          "music":os.getenv("LOCAL_MUSIC_MODEL","facebook/musicgen-small"),
                          "voice":"Coqui XTTS-v2","lipsync":"Wav2Lip (local checkpoint)"}}
    except Exception as e: return {"mode":"local-only","api_keys_required":False,"error":str(e)}


def local_llm_json(system: str, prompt: str, max_new_tokens=2500):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        mid=os.getenv("LOCAL_LLM_MODEL","Qwen/Qwen2.5-3B-Instruct")
        tok=AutoTokenizer.from_pretrained(mid)
        kw={"device_map":"auto","torch_dtype":torch.float16 if torch.cuda.is_available() else torch.float32}
        if torch.cuda.is_available():
            try: kw["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
            except Exception: pass
        model=AutoModelForCausalLM.from_pretrained(mid,**kw)
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}]
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=tok(rendered,return_tensors="pt").to(model.device)
        with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,repetition_penalty=1.05)
        answer=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        match=re.search(r"\{.*\}",answer,re.S)
        if not match: raise LocalModelError("Lokalny LLM nie zwrócił JSON. Skróć tekst i spróbuj ponownie.")
        return json.loads(match.group(0))
    except LocalModelError: raise
    except Exception as e: raise LocalModelError(f"Błąd lokalnego LLM: {e}") from e
    finally: _cleanup()


def generate_image(prompt: str, output: Path, reference: Path|None=None):
    try:
        import torch
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        mid=os.getenv("LOCAL_IMAGE_MODEL","stable-diffusion-v1-5/stable-diffusion-v1-5")
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        if reference:
            from PIL import Image
            pipe=StableDiffusionImg2ImgPipeline.from_pretrained(mid,torch_dtype=dtype,safety_checker=None)
            image=Image.open(reference).convert("RGB").resize((512,512))
            fn=lambda: pipe(prompt=prompt,image=image,strength=.65,guidance_scale=7,num_inference_steps=22).images[0]
        else:
            pipe=StableDiffusionPipeline.from_pretrained(mid,torch_dtype=dtype,safety_checker=None)
            fn=lambda: pipe(prompt=prompt,guidance_scale=7,num_inference_steps=22,height=512,width=512).images[0]
        pipe.enable_attention_slicing()
        if torch.cuda.is_available(): pipe.enable_model_cpu_offload()
        else: pipe.to("cpu")
        output.parent.mkdir(parents=True,exist_ok=True); fn().save(output)
        del pipe; _cleanup(); return output
    except Exception as e: raise LocalModelError(f"Lokalne generowanie obrazu: {e}") from e


def generate_video(prompt: str, output: Path, reference: Path|None=None, frames=16, fps=8):
    try:
        import torch, imageio.v2 as imageio
        from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        base=os.getenv("LOCAL_IMAGE_MODEL","stable-diffusion-v1-5/stable-diffusion-v1-5")
        motion=os.getenv("LOCAL_VIDEO_MODEL","guoyww/animatediff-motion-adapter-v1-5-2")
        adapter=MotionAdapter.from_pretrained(motion,torch_dtype=dtype)
        pipe=AnimateDiffPipeline.from_pretrained(base,motion_adapter=adapter,torch_dtype=dtype)
        pipe.scheduler=EulerDiscreteScheduler.from_config(pipe.scheduler.config,beta_schedule="linear")
        pipe.enable_vae_slicing(); pipe.enable_attention_slicing()
        if torch.cuda.is_available(): pipe.enable_model_cpu_offload()
        else: pipe.to("cpu")
        if reference: prompt += ", maintain appearance and costume from the reference image"
        result=pipe(prompt=prompt,num_frames=max(8,min(frames,24)),guidance_scale=7,num_inference_steps=18).frames[0]
        output.parent.mkdir(parents=True,exist_ok=True); imageio.mimsave(output,result,fps=fps,macro_block_size=1)
        del pipe,adapter; _cleanup(); return output
    except Exception as e: raise LocalModelError(f"Lokalne generowanie wideo AnimateDiff: {e}") from e


def generate_music(prompt: str, output: Path, seconds=12):
    try:
        import torch, scipy.io.wavfile
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        mid=os.getenv("LOCAL_MUSIC_MODEL","facebook/musicgen-small")
        processor=AutoProcessor.from_pretrained(mid)
        model=MusicgenForConditionalGeneration.from_pretrained(mid,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available(): model.to("cuda")
        inputs=processor(text=[prompt],padding=True,return_tensors="pt")
        if torch.cuda.is_available(): inputs={k:v.to("cuda") for k,v in inputs.items()}
        with torch.inference_mode(): audio=model.generate(**inputs,max_new_tokens=min(max(seconds,5),20)*50)
        wav=audio[0,0].detach().cpu().float().numpy()
        output.parent.mkdir(parents=True,exist_ok=True)
        scipy.io.wavfile.write(str(output),model.config.audio_encoder.sampling_rate,wav)
        del model,processor; _cleanup(); return output
    except Exception as e: raise LocalModelError(f"Lokalne generowanie muzyki MusicGen: {e}") from e


def synthesize_voice(text: str, sample: Path, output: Path, language="pl"):
    try:
        import torch
        from TTS.api import TTS
        if not sample.is_file(): raise LocalModelError("Nie znaleziono próbki głosu.")
        engine=TTS("tts_models/multilingual/multi-dataset/xtts_v2",progress_bar=False,gpu=torch.cuda.is_available())
        output.parent.mkdir(parents=True,exist_ok=True)
        engine.tts_to_file(text=text,speaker_wav=str(sample),language=language,file_path=str(output))
        del engine; _cleanup(); return output
    except LocalModelError: raise
    except Exception as e: raise LocalModelError(f"Lokalne klonowanie głosu XTTS: {e}") from e


def run_lipsync(video: Path,audio: Path,output: Path):
    checkpoint=os.getenv("WAV2LIP_CHECKPOINT","")
    script=Path(os.getenv("WAV2LIP_REPO","./vendor/Wav2Lip/inference.py"))
    if not checkpoint or not Path(checkpoint).is_file(): raise LocalModelError("Lip-sync wymaga lokalnego checkpointu Wav2Lip ustawionego w WAV2LIP_CHECKPOINT.")
    if not script.is_file(): raise LocalModelError("Brak lokalnego kodu Wav2Lip w vendor/Wav2Lip.")
    output.parent.mkdir(parents=True,exist_ok=True)
    p=subprocess.run(["python",str(script),"--checkpoint_path",checkpoint,"--face",str(video),"--audio",str(audio),"--outfile",str(output)],capture_output=True,text=True,timeout=7200)
    if p.returncode: raise LocalModelError(p.stderr[-2500:])
    return output
