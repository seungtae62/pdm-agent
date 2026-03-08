"""Pydantic models for event payloads from Edge systems."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class DefectFrequencies(BaseModel):
    """Bearing defect frequencies in Hz."""

    BPFO: float
    BPFI: float
    BSF: float
    FTF: float


class BearingInfo(BaseModel):
    """Bearing metadata."""

    bearing_id: str
    position: str
    install_date: str
    model: str
    type: str
    rolling_elements_count: int
    ball_diameter_inch: float
    pitch_diameter_inch: float
    contact_angle_deg: float
    defect_frequencies_hz: DefectFrequencies
    last_maintenance_date: str | None = None


class SensorConfig(BaseModel):
    """Sensor configuration."""

    sensor_count: int
    channels: list[str]
    sensor_type: str
    sampling_rate_hz: int
    samples_per_snapshot: int
    snapshot_interval_min: int


class EquipmentMeta(BaseModel):
    """Equipment metadata."""

    equipment_id: str
    equipment_name: str
    location: str
    shaft_rpm: int
    radial_load_lbs: int
    operation_start_date: str
    bearing: BearingInfo
    sensor_config: SensorConfig
    operation_days_elapsed: int
    total_running_hours: float


class TimeDomainFeatures(BaseModel):
    """Time-domain features for a single channel."""

    rms: float
    peak: float
    peak_to_peak: float
    crest_factor: float
    kurtosis: float
    skewness: float
    standard_deviation: float
    mean: float
    shape_factor: float


class FrequencyDomainFeatures(BaseModel):
    """Frequency-domain features for a single channel."""

    bpfo_amplitude: float
    bpfi_amplitude: float
    bsf_amplitude: float
    ftf_amplitude: float
    bpfo_harmonics_2x: float
    bpfi_harmonics_2x: float
    spectral_energy_total: float
    spectral_energy_high_freq_band: float
    dominant_frequency_hz: float
    sideband_presence: bool
    sideband_spacing_hz: float
    sideband_count: int


class CurrentFeatures(BaseModel):
    """Current sensor features snapshot."""

    snapshot_timestamp: str
    time_domain: dict[str, TimeDomainFeatures]
    frequency_domain: dict[str, FrequencyDomainFeatures]


class AnomalyDetectionResult(BaseModel):
    """Anomaly detection result from Edge ML model."""

    model_id: str
    anomaly_detected: bool
    anomaly_score: float
    anomaly_threshold: float
    health_state: str
    confidence: float


class ConfidenceInterval(BaseModel):
    """RUL prediction confidence interval."""

    lower: float | None = None
    upper: float | None = None


class MlRulPrediction(BaseModel):
    """ML-based Remaining Useful Life prediction."""

    model_id: str
    predicted_rul_hours: float | None = None
    confidence_interval_hours: ConfidenceInterval
    prediction_status: str
    reason: str | None = None


class EventPayload(BaseModel):
    """Event payload received from Edge systems.

    Based on the structure of data/payloads/*.json files.
    """

    event_id: str
    timestamp: str
    event_type: str
    edge_node_id: str
    equipment_meta: EquipmentMeta
    anomaly_detection_result: AnomalyDetectionResult
    current_features: CurrentFeatures
    ml_rul_prediction: MlRulPrediction | None = None
