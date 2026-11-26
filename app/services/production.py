import json
from pathlib import Path
from app.services.jobs import update_job
from app.services.local_models import LocalModelError,generate_image,generate_video,generate_music,synthesize_voice,run_lipsync
from app.services.render import render_video

def _register(project_dir: Path,file: Path,category: str,job_id: str):
    meta=project_dir/"project.json"; project=json.loads(meta.read_text(encoding="utf-8"))
    assets=project.setdefault("assets",[])
    asset={"id":f"local-{job_id}-{len(assets)}","name":file.name,"category":category,"path":str(file),"size":file.stat().st_size,"generated":True,"engine":"local"}
    assets.append(asset); project["updated_at"]=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    tmp=meta.with_suffix(".json.tmp"); tmp.write_text(json.dumps(project,ensure_ascii=False,indent=2),encoding="utf-8"); tmp.replace(meta)
    return asset

def _reference(project_dir,payload):
    aid=payload.get("reference_asset_id")
    if not aid:return None
    project=json.loads((project_dir/"project.json").read_text(encoding="utf-8"))
    a=next((x for x in project.get("assets",[]) if x.get("id")==aid and x.get("category")=="image"),None)
    if not a:raise LocalModelError("Nie znaleziono obrazu referencyjnego.")
    p=Path(a["path"])
    if not p.is_file():raise LocalModelError("Plik referencyjny nie istnieje.")
    return p

def run_generation(job_id,project_dir,kind,payload):
    update_job(job_id,status="running",progress=2,message="Uruchamianie lokalnego modelu; pierwsze pobranie może potrwać.")
    try:
        ref=_reference(project_dir,payload)
        category="audio" if kind=="music" else "image" if kind=="image" else "video"
        folder=project_dir/"assets"/category; folder.mkdir(parents=True,exist_ok=True)
        if kind=="image": out=generate_image(payload["prompt"],folder/f"image_{job_id}.png",ref)
        elif kind=="video": out=generate_video(payload["prompt"],folder/f"video_{job_id}.mp4",ref)
        elif kind=="music": out=generate_music(payload["prompt"],folder/f"music_{job_id}.wav")
        elif kind=="lipsync":
            project=json.loads((project_dir/"project.json").read_text(encoding="utf-8"))
            v=next((Path(a["path"]) for a in project.get("assets",[]) if a.get("id")==payload.get("video_asset_id") and a.get("category")=="video"),None)
            a=next((Path(a["path"]) for a in project.get("assets",[]) if a.get("id")==payload.get("audio_asset_id") and a.get("category")=="audio"),None)
            if not v or not a:raise LocalModelError("Wybierz klip wideo i audio z biblioteki.")
            out=run_lipsync(v,a,folder/f"lipsync_{job_id}.mp4")
        else:raise LocalModelError("Nieznany typ generowania.")
        _register(project_dir,out,category,job_id)
        update_job(job_id,status="succeeded",progress=100,message="Lokalne generowanie zakończone",result={"saved_files":[str(out)],"engine":"local"})
    except Exception as e:update_job(job_id,status="failed",progress=100,message=str(e),error=str(e))

def run_tts(job_id,project_dir,payload):
    update_job(job_id,status="running",progress=5,message="Ładowanie lokalnego XTTS-v2")
    try:
        project=json.loads((project_dir/"project.json").read_text(encoding="utf-8"))
        voice=next((v for v in project.get("voices",[]) if v.get("id")==payload["voice_id"]),None)
        if not voice:raise LocalModelError("Nie znaleziono profilu głosu.")
        out=project_dir/"assets"/"audio"/f"speech_{job_id}.wav"
        synthesize_voice(payload["text"],Path(voice["sample_path"]),out,voice.get("language","pl"))
        _register(project_dir,out,"audio",job_id)
        update_job(job_id,status="succeeded",progress=100,message="Wygenerowano mowę lokalnie",result={"path":str(out),"engine":"XTTS-v2 local"})
    except Exception as e:update_job(job_id,status="failed",progress=100,message=str(e),error=str(e))

def run_full_production(job_id,project_dir,project_id,scenes,reference_asset_id=None):
    update_job(job_id,status="running",progress=1,message="Lokalna produkcja ujęć")
    try:
        shots=[s for scene in scenes if isinstance(scene,dict) for s in scene.get("shots",[])]
        if not shots:raise LocalModelError("Plan AI Director nie zawiera ujęć.")
        ref=_reference(project_dir,{"reference_asset_id":reference_asset_id}) if reference_asset_id else None
        clips=[]
        for i,shot in enumerate(shots,1):
            prompt=shot.get("visual_prompt") or shot.get("prompt") or shot.get("summary")
            if not prompt:continue
            clip=project_dir/"assets"/"video"/f"shot_{i:04d}_{job_id[:8]}.mp4"
            generate_video(prompt,clip,ref); _register(project_dir,clip,"video",f"{job_id}-{i}"); clips.append(str(clip))
            update_job(job_id,progress=min(90,int(i/len(shots)*90)),message=f"Lokalnie wygenerowano ujęcie {i}/{len(shots)}")
        if not clips:raise LocalModelError("Nie powstały klipy.")
        final=render_video(project_dir,clips,output_name=f"episode_{job_id[:8]}.mp4")
        update_job(job_id,status="succeeded",progress=100,message=f"Zmontowano lokalnie {len(clips)} ujęć",result={"clip_count":len(clips),"clips":clips,"final_video":str(final),"engine":"local"})
    except Exception as e:update_job(job_id,status="failed",progress=100,message=str(e),error=str(e))
