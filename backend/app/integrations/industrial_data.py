import csv
from pathlib import Path
from typing import Protocol

from backend.app.schemas.industrial_data import (
    Alarm,
    AlarmSeverity,
    Equipment,
    Maintenance,
    Telemetry,
)


REQUIRED_COLUMNS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}

EQUIPMENT_TYPE_MAP = {
    "L": "light-duty",
    "M": "medium-duty",
    "H": "heavy-duty",
}

FAILURE_CODE_MAP = {
    "TWF": "TOOL_WEAR",
    "HDF": "HEAT_DISSIPATION",
    "PWF": "POWER_FAILURE",
    "OSF": "OVERSTRAIN",
    "RNF": "RANDOM_FAILURE",
}


class IndustrialDataSource(Protocol):
    def load_equipment(self) -> list[Equipment]: ...

    def load_telemetry(
        self,
        equipment_id: str | None = None,
    ) -> list[Telemetry]: ...

    def load_alarms(
        self,
        equipment_id: str | None = None,
    ) -> list[Alarm]: ...

    def load_maintenance(
        self,
        equipment_id: str | None = None,
    ) -> list[Maintenance]: ...


class IndustrialDataAdapter:
    def __init__(
        self,
        source_path: str | Path | None = None,
    ) -> None:
        self.source_path = Path(source_path) if source_path else (
            Path(__file__).resolve().parents[1]
            / "data"
            / "ai4i_sample.csv"
        )
        self._rows: list[tuple[int, dict[str, str]]] | None = None

    def load_equipment(self) -> list[Equipment]:
        equipment_by_id: dict[str, Equipment] = {}

        for row_number, row in self._get_rows():
            product_id = self._get_required_value(
                row,
                "Product ID",
                row_number,
            )

            if product_id in equipment_by_id:
                continue

            equipment_type_code = self._get_required_value(
                row,
                "Type",
                row_number,
            )
            equipment_type = EQUIPMENT_TYPE_MAP.get(
                equipment_type_code
            )
            if equipment_type is None:
                raise ValueError(
                    "Invalid equipment type at row "
                    f"{row_number}: '{equipment_type_code}'"
                )

            equipment_by_id[product_id] = Equipment(
                equipment_id=product_id,
                equipment_name=product_id,
                equipment_type=equipment_type,
            )

        return [
            equipment_by_id[equipment_id]
            for equipment_id in sorted(equipment_by_id)
        ]

    def load_telemetry(
        self,
        equipment_id: str | None = None,
    ) -> list[Telemetry]:
        telemetry: list[Telemetry] = []

        for row_number, row in self._get_rows():
            item = self._build_telemetry(
                row=row,
                row_number=row_number,
            )
            if equipment_id and item.equipment_id != equipment_id:
                continue
            telemetry.append(item)

        return telemetry

    def load_alarms(
        self,
        equipment_id: str | None = None,
    ) -> list[Alarm]:
        alarms: list[Alarm] = []

        for row_number, row in self._get_rows():
            telemetry = self._build_telemetry(
                row=row,
                row_number=row_number,
            )
            if equipment_id and telemetry.equipment_id != equipment_id:
                continue

            failure_codes = self._extract_failure_codes(
                row=row,
                row_number=row_number,
                machine_failure=telemetry.machine_failure,
            )
            if not failure_codes:
                continue

            for failure_code in failure_codes:
                severity = (
                    AlarmSeverity.HIGH
                    if telemetry.machine_failure
                    else AlarmSeverity.MEDIUM
                )
                alarms.append(
                    Alarm(
                        alarm_id=(
                            f"ALARM-{telemetry.equipment_id}"
                            f"-{telemetry.sequence}-{failure_code}"
                        ),
                        equipment_id=telemetry.equipment_id,
                        alarm_code=failure_code,
                        severity=severity,
                        message=(
                            f"{telemetry.equipment_id} reported "
                            f"{failure_code.lower()} at sequence "
                            f"{telemetry.sequence}"
                        ),
                    )
                )

        return alarms

    def load_maintenance(
        self,
        equipment_id: str | None = None,
    ) -> list[Maintenance]:
        maintenance_items: list[Maintenance] = []

        for row_number, row in self._get_rows():
            telemetry = self._build_telemetry(
                row=row,
                row_number=row_number,
            )
            if equipment_id and telemetry.equipment_id != equipment_id:
                continue

            maintenance_type: str | None = None
            description: str | None = None
            if self._parse_binary_flag(
                row=row,
                column="TWF",
                row_number=row_number,
            ) == 1 or telemetry.tool_wear_min >= 200:
                maintenance_type = "tooling_inspection"
                description = (
                    "Inspect tooling wear and prepare replacement."
                )
            elif self._parse_binary_flag(
                row=row,
                column="HDF",
                row_number=row_number,
            ) == 1:
                maintenance_type = "thermal_check"
                description = (
                    "Inspect cooling and heat dissipation condition."
                )
            elif self._parse_binary_flag(
                row=row,
                column="OSF",
                row_number=row_number,
            ) == 1:
                maintenance_type = "load_balance_review"
                description = (
                    "Review overload and process strain condition."
                )
            elif telemetry.machine_failure:
                maintenance_type = "failure_review"
                description = (
                    "Run post-failure inspection for the equipment."
                )

            if maintenance_type is None or description is None:
                continue

            maintenance_items.append(
                Maintenance(
                    maintenance_id=(
                        f"MAINT-{telemetry.equipment_id}"
                        f"-{telemetry.sequence}"
                    ),
                    equipment_id=telemetry.equipment_id,
                    maintenance_type=maintenance_type,
                    description=description,
                )
            )

        return maintenance_items

    def _get_rows(self) -> list[tuple[int, dict[str, str]]]:
        if self._rows is not None:
            return self._rows

        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Industrial data source not found: {self.source_path}"
            )

        with self.source_path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as source_file:
            reader = csv.DictReader(source_file)
            if reader.fieldnames is None:
                raise ValueError(
                    "Invalid industrial data schema: header row is missing"
                )

            missing_columns = sorted(
                REQUIRED_COLUMNS.difference(reader.fieldnames)
            )
            if missing_columns:
                raise ValueError(
                    "Invalid industrial data schema: missing columns "
                    f"{missing_columns}"
                )

            rows: list[tuple[int, dict[str, str]]] = []
            for row_number, row in enumerate(reader, start=2):
                rows.append((row_number, row))

        self._rows = rows
        return rows

    def _build_telemetry(
        self,
        row: dict[str, str],
        row_number: int,
    ) -> Telemetry:
        equipment_id = self._get_required_value(
            row,
            "Product ID",
            row_number,
        )

        return Telemetry(
            equipment_id=equipment_id,
            sequence=self._parse_int(
                row=row,
                column="UDI",
                row_number=row_number,
            ),
            air_temperature_k=self._parse_float(
                row=row,
                column="Air temperature [K]",
                row_number=row_number,
            ),
            process_temperature_k=self._parse_float(
                row=row,
                column="Process temperature [K]",
                row_number=row_number,
            ),
            rotational_speed_rpm=self._parse_int(
                row=row,
                column="Rotational speed [rpm]",
                row_number=row_number,
            ),
            torque_nm=self._parse_float(
                row=row,
                column="Torque [Nm]",
                row_number=row_number,
            ),
            tool_wear_min=self._parse_int(
                row=row,
                column="Tool wear [min]",
                row_number=row_number,
            ),
            machine_failure=bool(
                self._parse_binary_flag(
                    row=row,
                    column="Machine failure",
                    row_number=row_number,
                )
            ),
        )

    def _extract_failure_codes(
        self,
        row: dict[str, str],
        row_number: int,
        machine_failure: bool,
    ) -> list[str]:
        failure_codes: list[str] = []

        for column_name, alarm_code in FAILURE_CODE_MAP.items():
            if self._parse_binary_flag(
                row=row,
                column=column_name,
                row_number=row_number,
            ) == 1:
                failure_codes.append(alarm_code)

        if machine_failure and not failure_codes:
            failure_codes.append("MACHINE_FAILURE")

        return failure_codes

    def _get_required_value(
        self,
        row: dict[str, str],
        column: str,
        row_number: int,
    ) -> str:
        value = row.get(column, "")
        if value is None or value.strip() == "":
            raise ValueError(
                f"Missing required value for column '{column}' at row {row_number}"
            )
        return value.strip()

    def _parse_float(
        self,
        row: dict[str, str],
        column: str,
        row_number: int,
    ) -> float:
        value = self._get_required_value(
            row=row,
            column=column,
            row_number=row_number,
        )
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value for column '{column}' at row {row_number}: '{value}'"
            ) from exc

    def _parse_int(
        self,
        row: dict[str, str],
        column: str,
        row_number: int,
    ) -> int:
        value = self._get_required_value(
            row=row,
            column=column,
            row_number=row_number,
        )
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value for column '{column}' at row {row_number}: '{value}'"
            ) from exc

    def _parse_binary_flag(
        self,
        row: dict[str, str],
        column: str,
        row_number: int,
    ) -> int:
        value = self._parse_int(
            row=row,
            column=column,
            row_number=row_number,
        )
        if value not in (0, 1):
            raise ValueError(
                f"Invalid binary flag for column '{column}' at row {row_number}: '{value}'"
            )
        return value