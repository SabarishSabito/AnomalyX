from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TelemetryRecord(BaseModel):
    timestamp: Optional[str] = None
    cpu_usage: Optional[float] = 0.0
    memory_usage: Optional[float] = 0.0
    network_io: Optional[float] = 0.0
    latency_ms: Optional[float] = 0.0
    error_rate: Optional[float] = 0.0

class PredictRequest(BaseModel):
    model_id: str = Field(default="isolation_forest", description="ML Algorithm ID")
    contamination: float = Field(default=0.05, ge=0.01, le=0.30)
    records: List[Dict[str, Any]]

class ModelConfigUpdate(BaseModel):
    model_id: str
    contamination: float = Field(default=0.05, ge=0.01, le=0.30)

class StreamNextRequest(BaseModel):
    model_id: str = "isolation_forest"
    contamination: float = 0.05
    category: str = "server"
    force_anomaly: bool = False

class AlertItem(BaseModel):
    id: str
    timestamp: str
    severity: str
    anomaly_score: float
    confidence_pct: float
    top_contributor: str
    metrics: Dict[str, float]
    explainability: Dict[str, Any]

class MetricsSummary(BaseModel):
    total_analyzed: int
    anomalies_count: int
    anomaly_rate_pct: float
    avg_confidence_pct: float
    active_model: str
    contamination_threshold: float
