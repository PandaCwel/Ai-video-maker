import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from app import main

client = TestClient(main.app)


def isolated_store():
    return main.ProjectStore(Path(tempfile.mkdtemp()))


def test_health_and_home():
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200


def test_project_crud_script_analysis_and_character():
    original = main.store
    main.store = isolated_store()
    try:
        created = client.post("/api/projects", json={"title": "Test film"})
        assert created.status_code == 201
        project_id = created.json()["id"]
        saved = client.put(f"/api/projects/{project_id}/script", json={"script": "INT. ROOM - NIGHT\n\nA light flickers."})
        assert saved.status_code == 200
        analysis = client.post(f"/api/projects/{project_id}/script/analyze")
        assert analysis.status_code == 200
        assert analysis.json()["scene_count"] == 1 or analysis.json().get("scenes")
        character = client.post(f"/api/projects/{project_id}/characters", json={"name": "Mara", "description": "Pilot"})
        assert character.status_code == 201
        assert len(client.get(f"/api/projects/{project_id}").json()["characters"]) == 1
        assert client.delete(f"/api/projects/{project_id}").status_code == 204
        assert client.get(f"/api/projects/{project_id}").status_code == 404
    finally:
        main.store = original


def test_empty_script_analysis_rejected():
    original = main.store
    main.store = isolated_store()
    try:
        project = client.post("/api/projects", json={"title": "Empty"}).json()
        response = client.post(f"/api/projects/{project['id']}/script/analyze")
        assert response.status_code == 400
    finally:
        main.store = original


def test_voice_clone_requires_consent_and_upload_validation():
    original = main.store
    main.store = isolated_store()
    try:
        project = client.post("/api/projects", json={"title": "Voice test"}).json()
        response = client.post(f"/api/projects/{project['id']}/voices/clone",
            data={"name": "Test", "consent_confirmed": "false"},
            files={"sample": ("voice.wav", b"fake-audio", "audio/wav")})
        assert response.status_code == 400
    finally:
        main.store = original
