import json
import logging
from typing import Protocol

from backend.app.observability.models import (
    ObservabilityEvent,
)


class ObservabilitySink(Protocol):
    def emit(
        self,
        event: ObservabilityEvent,
    ) -> None: ...


class NoOpObservabilitySink:
    def emit(
        self,
        event: ObservabilityEvent,
    ) -> None:
        _ = event


class InMemoryObservabilitySink:
    def __init__(
        self,
    ) -> None:
        self.events: list[ObservabilityEvent] = []

    def emit(
        self,
        event: ObservabilityEvent,
    ) -> None:
        self.events.append(event)


class LoggingObservabilitySink:
    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(
            "factoryops.observability"
        )

    def emit(
        self,
        event: ObservabilityEvent,
    ) -> None:
        payload = event.model_dump(mode="json")
        self.logger.info(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
