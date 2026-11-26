import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app import main
from app.core.config import settings

client = TestClient(main.app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_project_crud_and_script():
    original = main.store
    with tempfile.TemporaryDirectory() as temp_dir:
        main.store = main.ProjectStore(Path(temp_dir))
        try:
            created = client.post("/api/projects", json={"title": "Test film"})
            assert created.status_code == 201
            project_id = created.json()["id"]

            saved = client.put(f"/api/projects/{project_id}/script", json={"script": "INT. ROOM - NIGHT\n\nA light flickers."})
            assert saved.status_code == 200
            assert "A light flickers" in saved.json()["script"]

            analysis = client.post(f"/api/projects/{project_id}/script/analyze")
            assert analysis.status_code == 200
            assert analysis.json()["scene_count"] == 1

            assert client.get(f"/api/projects/{project_id}").status_code == 200
            assert client.delete(f"/api/projects/{project_id}").status_code == 204
            assert client.get(f"/api/projects/{project_id}").status_code == 404
        finally:
            main.store = original


def test_empty_script_analysis_rejected():
    original = main.store
    with tempfile.TemporaryDirectory() as temp_dir:
        main.store = main.ProjectStore(Path(temp_dir))
        try:
            project = client.post("/api/projects", json={"title": "Empty"}).json()
            response = client.post(f"/api/projects/{project['id']}/script/analyze")
            assert response.status_code == 400
        finally:
            main.store = original
