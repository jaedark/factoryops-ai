from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class AlarmSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Equipment(BaseModel):
    model_config = ConfigDict(frozen=True)

    equipment_id: str = Field(min_length=1)
    equipment_name: str = Field(min_length=1)
    equipment_type: str = Field(min_length=1)


class Telemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    equipment_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: int = Field(ge=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: int = Field(ge=0)
    machine_failure: bool


class Alarm(BaseModel):
    model_config = ConfigDict(frozen=True)

    alarm_id: str = Field(min_length=1)
    equipment_id: str = Field(min_length=1)
    alarm_code: str = Field(min_length=1)
    severity: AlarmSeverity
    message: str = Field(min_length=1)


class Maintenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    maintenance_id: str = Field(min_length=1)
    equipment_id: str = Field(min_length=1)
    maintenance_type: str = Field(min_length=1)
    description: str = Field(min_length=1)


class HighRiskEquipment(BaseModel):
    model_config = ConfigDict(frozen=True)

    equipment: Equipment
    latest_telemetry: Telemetry
    risk_reasons: list[str] = Field(min_length=1)