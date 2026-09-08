import os

import uvicorn

from backend.app.core.config import ConfigurationError, AppSettings, get_settings


def resolve_server_port(
    settings: AppSettings,
    environment: dict[str, str] | None = None,
) -> int:
    values = environment if environment is not None else os.environ
    cloud_run_port = values.get("PORT")
    if cloud_run_port is None:
        return settings.factory_agent_port

    try:
        port = int(cloud_run_port)
    except ValueError as exc:
        raise ConfigurationError("PORT must be a valid integer") from exc
    if not 1 <= port <= 65535:
        raise ConfigurationError("PORT must be between 1 and 65535")
    return port


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.factory_agent_host,
        port=resolve_server_port(settings),
        workers=1,
    )


if __name__ == "__main__":
    main()
