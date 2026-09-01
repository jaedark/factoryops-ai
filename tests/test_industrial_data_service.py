from pathlib import Path

import pytest

from backend.app.integrations.industrial_data import IndustrialDataAdapter
from backend.app.services.industrial_data_service import (
    IndustrialDataService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SOURCE = (
    PROJECT_ROOT / "backend" / "app" / "data" / "ai4i_sample.csv"
)


def test_industrial_data_adapter_loads_sample_data_successfully():
    adapter = IndustrialDataAdapter(SAMPLE_SOURCE)

    equipment = adapter.load_equipment()
    telemetry = adapter.load_telemetry()

    assert len(equipment) == 5
    assert len(telemetry) == 12


def test_industrial_data_adapter_transforms_equipment_fields():
    adapter = IndustrialDataAdapter(SAMPLE_SOURCE)

    equipment = adapter.load_equipment()[0]

    assert equipment.equipment_id == "Conveyor-01"
    assert equipment.equipment_name == "Conveyor-01"
    assert equipment.equipment_type == "light-duty"


def test_industrial_data_adapter_transforms_telemetry_fields():
    adapter = IndustrialDataAdapter(SAMPLE_SOURCE)

    telemetry = adapter.load_telemetry("Robot-01")[0]

    assert telemetry.equipment_id == "Robot-01"
    assert telemetry.sequence == 1
    assert telemetry.air_temperature_k == pytest.approx(298.1)
    assert telemetry.process_temperature_k == pytest.approx(308.6)
    assert telemetry.rotational_speed_rpm == 1550
    assert telemetry.torque_nm == pytest.approx(42.1)
    assert telemetry.tool_wear_min == 120
    assert telemetry.machine_failure is False


def test_industrial_data_service_lists_equipment():
    service = IndustrialDataService(
        IndustrialDataAdapter(SAMPLE_SOURCE)
    )

    equipment = service.list_equipment()

    assert [item.equipment_id for item in equipment] == [
        "Conveyor-01",
        "Press-01",
        "Pump-07",
        "Robot-01",
        "Vision-02",
    ]


def test_industrial_data_service_gets_equipment_by_id():
    service = IndustrialDataService(
        IndustrialDataAdapter(SAMPLE_SOURCE)
    )

    equipment = service.get_equipment("Vision-02")

    assert equipment.equipment_id == "Vision-02"
    assert equipment.equipment_type == "medium-duty"


def test_industrial_data_service_raises_for_missing_equipment():
    service = IndustrialDataService(
        IndustrialDataAdapter(SAMPLE_SOURCE)
    )

    with pytest.raises(LookupError) as exc:
        service.get_equipment("UNKNOWN-EQUIPMENT")

    assert str(exc.value) == "Equipment not found: UNKNOWN-EQUIPMENT"


def test_industrial_data_service_gets_equipment_telemetry():
    service = IndustrialDataService(
        IndustrialDataAdapter(SAMPLE_SOURCE)
    )

    telemetry = service.get_equipment_telemetry("Vision-02")

    assert len(telemetry) == 2
    assert telemetry[0].sequence == 7
    assert telemetry[1].sequence == 8


def test_industrial_data_service_identifies_high_risk_equipment():
    service = IndustrialDataService(
        IndustrialDataAdapter(SAMPLE_SOURCE)
    )

    high_risk = service.get_high_risk_equipment()

    assert [item.equipment.equipment_id for item in high_risk] == [
        "Conveyor-01",
        "Press-01",
        "Robot-01",
    ]
    assert high_risk[0].risk_reasons == [
        "machine_failure",
        "tool_wear_threshold",
    ]
    assert high_risk[1].risk_reasons == [
        "machine_failure",
        "process_temperature_gap",
    ]
    assert high_risk[2].risk_reasons == [
        "machine_failure",
        "process_temperature_gap",
    ]


def test_industrial_data_adapter_raises_for_missing_source_file(
    tmp_path: Path,
):
    missing_source = tmp_path / "missing.csv"
    adapter = IndustrialDataAdapter(missing_source)

    with pytest.raises(FileNotFoundError) as exc:
        adapter.load_equipment()

    assert str(exc.value) == (
        f"Industrial data source not found: {missing_source}"
    )


def test_industrial_data_adapter_raises_for_invalid_schema(
    tmp_path: Path,
):
    invalid_source = tmp_path / "invalid_schema.csv"
    invalid_source.write_text(
        "UDI,Product ID,Type\n1,Robot-01,M\n",
        encoding="utf-8",
    )
    adapter = IndustrialDataAdapter(invalid_source)

    with pytest.raises(ValueError) as exc:
        adapter.load_equipment()

    assert "Invalid industrial data schema: missing columns" in str(
        exc.value
    )
    assert "Air temperature [K]" in str(exc.value)


def test_industrial_data_adapter_raises_for_invalid_numeric_value(
    tmp_path: Path,
):
    invalid_source = tmp_path / "invalid_numeric.csv"
    invalid_source.write_text(
        "UDI,Product ID,Type,Air temperature [K],Process temperature [K],"
        "Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,"
        "TWF,HDF,PWF,OSF,RNF\n"
        "1,Robot-01,M,bad-value,308.6,1550,42.1,120,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    adapter = IndustrialDataAdapter(invalid_source)

    with pytest.raises(ValueError) as exc:
        adapter.load_telemetry()

    assert str(exc.value) == (
        "Invalid numeric value for column 'Air temperature [K]' "
        "at row 2: 'bad-value'"
    )


def test_industrial_data_adapter_builds_alarm_and_maintenance_views():
    adapter = IndustrialDataAdapter(SAMPLE_SOURCE)

    alarms = adapter.load_alarms("Conveyor-01")
    maintenance = adapter.load_maintenance("Conveyor-01")

    assert len(alarms) == 2
    assert alarms[0].alarm_code == "TOOL_WEAR"
    assert alarms[0].severity.value == "high"
    assert len(maintenance) == 2
    assert maintenance[0].maintenance_type == "tooling_inspection"