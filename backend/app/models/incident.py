from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.core.database import Base


class Incident(Base):
    __tablename__ = "incidents"

    incident_id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    equipment_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    process_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    symptom: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    cause: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    action_taken: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    result: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        nullable=False,
    )
