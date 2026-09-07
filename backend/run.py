import uvicorn

from backend.app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.factory_agent_host,
        port=settings.factory_agent_port,
        workers=1,
    )


if __name__ == "__main__":
    main()
