from backend.app.integrations.industrial_data import (
    IndustrialDataAdapter,
    IndustrialDataSource,
)
from backend.app.schemas.industrial_data import (
    Equipment,
    HighRiskEquipment,
    Telemetry,
)


class IndustrialDataService:
    def __init__(
        self,
        data_source: IndustrialDataSource | None = None,
    ) -> None:
        self.data_source = data_source or IndustrialDataAdapter()

    def list_equipment(self) -> list[Equipment]:
        return self.data_source.load_equipment()

    def get_equipment(
        self,
        equipment_id: str,
    ) -> Equipment:
        for equipment in self.list_equipment():
            if equipment.equipment_id == equipment_id:
                return equipment

        raise LookupError(f"Equipment not found: {equipment_id}")

    def get_equipment_telemetry(
        self,
        equipment_id: str,
    ) -> list[Telemetry]:
        self.get_equipment(equipment_id)
        return self.data_source.load_telemetry(equipment_id)

    def get_high_risk_equipment(self) -> list[HighRiskEquipment]:
        high_risk_items: list[HighRiskEquipment] = []

        for equipment in self.list_equipment():
            telemetry = self.get_equipment_telemetry(
                equipment.equipment_id
            )
            if not telemetry:
                continue

            risk_reasons = self._build_risk_reasons(telemetry)
            if not risk_reasons:
                continue

            high_risk_items.append(
                HighRiskEquipment(
                    equipment=equipment,
                    latest_telemetry=telemetry[-1],
                    risk_reasons=risk_reasons,
                )
            )

        return high_risk_items

    @staticmethod
    def _build_risk_reasons(
        telemetry: list[Telemetry],
    ) -> list[str]:
        risk_reasons: list[str] = []

        if any(item.machine_failure for item in telemetry):
            risk_reasons.append("machine_failure")

        if any(item.tool_wear_min >= 200 for item in telemetry):
            risk_reasons.append("tool_wear_threshold")

        if any(
            (item.process_temperature_k - item.air_temperature_k) >= 12.0
            for item in telemetry
        ):
            risk_reasons.append("process_temperature_gap")

        return risk_reasons