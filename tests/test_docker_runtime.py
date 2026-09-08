from pathlib import Path

import yaml

from backend.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_docker_runtime_uses_fixed_non_root_single_worker_configuration():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    runner = (PROJECT_ROOT / "backend/run.py").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.14.3-slim")
    assert "USER appuser" in dockerfile
    assert 'CMD ["python", "-m", "backend.run"]' in dockerfile
    assert "workers=1" in runner
    assert "HEALTHCHECK" in dockerfile
    assert "GEMINI_API_KEY" not in dockerfile


def test_compose_exposes_api_and_healthcheck_without_embedding_secrets():
    compose = yaml.safe_load(
        (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
    )
    service = compose["services"]["factory-agent"]

    assert service["ports"] == [
        "${FACTORY_AGENT_PORT:-8000}:${FACTORY_AGENT_PORT:-8000}"
    ]
    assert service["environment"]["GEMINI_API_KEY"] == "${GEMINI_API_KEY:-}"
    assert service["environment"]["FACTORY_AGENT_API_KEY"] == (
        "${FACTORY_AGENT_API_KEY:-}"
    )
    assert service["environment"]["DATABASE_URL"] == (
        "${DATABASE_URL:-sqlite:///./factoryops.db}"
    )
    assert "healthcheck" in service


def test_health_route_is_registered_without_agent_dependency():
    health_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/health"
    )

    assert health_route.endpoint() == {"status": "ok"}
