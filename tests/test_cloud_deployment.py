from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from backend.app.core.config import AppSettings, ConfigurationError
from backend.main import app
from backend import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def _load_cloud_build() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
    )


def _deploy_arguments() -> list[str]:
    cloud_build = _load_cloud_build()
    deploy_step = next(
        step for step in cloud_build["steps"] if step["id"] == "deploy-cloud-run"
    )
    return deploy_step["args"]


def test_cloud_run_port_overrides_factory_agent_port():
    settings = AppSettings(factory_agent_port=8000)

    assert run.resolve_server_port(settings, {"PORT": "8080"}) == 8080


def test_factory_agent_port_is_used_without_cloud_run_port():
    settings = AppSettings(factory_agent_port=8100)

    assert run.resolve_server_port(settings, {}) == 8100


def test_server_port_uses_local_default():
    assert run.resolve_server_port(AppSettings(), {}) == 8000


@pytest.mark.parametrize("port", ["invalid", "0", "65536"])
def test_invalid_cloud_run_port_is_rejected(port):
    with pytest.raises(ConfigurationError, match="PORT"):
        run.resolve_server_port(AppSettings(), {"PORT": port})


def test_runner_binds_all_interfaces_with_one_worker(monkeypatch):
    settings = AppSettings(factory_agent_host="0.0.0.0")
    monkeypatch.setattr(run, "get_settings", lambda: settings)
    monkeypatch.setenv("PORT", "8080")

    with patch.object(run.uvicorn, "run") as run_server:
        run.main()

    run_server.assert_called_once_with(
        "backend.main:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
    )


def test_cloud_build_uses_artifact_registry_commit_tag_path():
    cloud_build = _load_cloud_build()
    image = cloud_build["images"][0]

    assert image == (
        "${_REGION}-docker.pkg.dev/$PROJECT_ID/"
        "${_REPOSITORY}/${_IMAGE_NAME}:${_IMAGE_TAG}"
    )
    assert cloud_build["substitutions"]["_REGION"] == "asia-northeast3"
    assert cloud_build["substitutions"]["_IMAGE_TAG"] == "manual"


def test_cloud_build_contains_build_push_and_deploy_steps():
    cloud_build = _load_cloud_build()

    assert [step["id"] for step in cloud_build["steps"]] == [
        "build-image",
        "push-image",
        "deploy-cloud-run",
    ]


def test_cloud_run_deploy_uses_demo_safety_limits():
    arguments = _deploy_arguments()

    assert "--min-instances=0" in arguments
    assert "--max-instances=1" in arguments
    assert "--concurrency=1" in arguments
    assert "--cpu=2" in arguments
    assert "--memory=4Gi" in arguments
    assert "--no-allow-unauthenticated" in arguments


def test_cloud_build_uses_secret_manager_without_plaintext_secret():
    source = (PROJECT_ROOT / "cloudbuild.yaml").read_text(encoding="utf-8")
    arguments = _deploy_arguments()

    assert "AIza" not in source
    assert (
        "--set-secrets=GEMINI_API_KEY=${_SECRET_NAME}:${_SECRET_VERSION}"
        in arguments
    )
    assert not any(argument.startswith("--set-env-vars=GEMINI_API_KEY") for argument in arguments)


def test_cloud_run_does_not_override_reserved_port_environment_variable():
    arguments = _deploy_arguments()

    assert not any("PORT=" in argument for argument in arguments)
    assert "--port=8080" in arguments


def test_gcloudignore_excludes_local_secrets_and_state():
    ignore = (PROJECT_ROOT / ".gcloudignore").read_text(encoding="utf-8")

    assert ".env" in ignore
    assert ".env.*" in ignore
    assert "factoryops.db" in ignore
    assert ".venv" in ignore
    assert "backend/app/data" not in ignore


def test_cloud_health_does_not_require_gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_deployment_helper_checks_prerequisites_and_required_apis():
    source = (PROJECT_ROOT / "scripts/deploy_cloud_run.ps1").read_text(
        encoding="utf-8"
    )

    assert "Get-Command gcloud" in source
    assert "gcloud auth list" in source
    assert "gcloud config get-value project" in source
    assert "run.googleapis.com" in source
    assert "cloudbuild.googleapis.com" in source
    assert "artifactregistry.googleapis.com" in source
    assert "secretmanager.googleapis.com" in source
    assert "roles/secretmanager.secretAccessor" in source
    assert "_SECRET_VERSION=$secretVersion" in source
