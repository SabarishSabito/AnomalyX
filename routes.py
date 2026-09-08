import time
import uuid
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from backend.schemas import (
    PredictRequest, StreamNextRequest, ModelConfigUpdate,
    AlertItem, MetricsSummary
)
from ml_module.detector import AnomalyDetector
from ml_module.generator import TelemetryGenerator

router = APIRouter()

# Global state containers
active_detector = AnomalyDetector(model_id="isolation_forest", contamination=0.05)
generator_instances: Dict[str, TelemetryGenerator] = {
    "server": TelemetryGenerator("server"),
    "financial": TelemetryGenerator("financial"),
    "iot": TelemetryGenerator("iot")
}

alerts_history: List[Dict[str, Any]] = []
session_stats = {
    "total_analyzed": 0,
    "anomalies_count": 0,
    "confidence_sum": 0.0,
    "start_time": time.time()
}

@router.get("/health")
def get_health():
    """Returns API health status, active ML engine model, and session metrics."""
    uptime_sec = int(time.time() - session_stats["start_time"])
    return {
        "status": "healthy",
        "version": "1.0.0",
        "engine": "Scikit-Learn ML Engine",
        "active_model": active_detector.model_id,
        "model_name": active_detector.SUPPORTED_MODELS.get(active_detector.model_id),
        "contamination": active_detector.contamination,
        "uptime_seconds": uptime_sec,
        "timestamp": datetime.datetime.now().isoformat()
    }

@router.get("/models")
def get_models():
    """Returns list of supported anomaly detection algorithms and their specs."""
    return {
        "active_model": active_detector.model_id,
        "contamination": active_detector.contamination,
        "models": [
            {
                "id": "isolation_forest",
                "name": "Isolation Forest",
                "type": "Tree Ensemble Isolation",
                "description": "Partition-based tree ensemble. Excels at detecting structural outliers in high-dimensional telemetry metrics.",
                "default_contamination": 0.05
            },
            {
                "id": "local_outlier_factor",
                "name": "Local Outlier Factor (LOF)",
                "type": "Density-Based Outlier Detection",
                "description": "Compares local density of a sample to its k-nearest neighbors to flag contextual density drops.",
                "default_contamination": 0.05
            },
            {
                "id": "one_class_svm",
                "name": "One-Class Support Vector Machine",
                "type": "Kernel Boundary Learning",
                "description": "Fits a high-dimensional RBF hypersphere bounding baseline operational patterns.",
                "default_contamination": 0.05
            },
            {
                "id": "zscore_ensemble",
                "name": "Z-Score / IQR Statistical Ensemble",
                "type": "Statistical Thresholding",
                "description": "Lightweight normalized z-score matrix calculation with IQR threshold boundaries.",
                "default_contamination": 0.05
            }
        ]
    }

@router.post("/models/configure")
def configure_model(config: ModelConfigUpdate):
    """Updates the active ML algorithm or contamination sensitivity threshold."""
    active_detector.set_parameters(model_id=config.model_id, contamination=config.contamination)
    return {
        "message": "Model configuration updated successfully",
        "active_model": active_detector.model_id,
        "model_name": active_detector.SUPPORTED_MODELS.get(active_detector.model_id),
        "contamination": active_detector.contamination
    }

@router.post("/predict")
def predict_anomalies(payload: PredictRequest):
    """Detects anomalies on a user-submitted batch of records."""
    if not payload.records:
        raise HTTPException(status_code=400, detail="Payload contains no records")

    # Temporarily update model params if specified
    active_detector.set_parameters(model_id=payload.model_id, contamination=payload.contamination)

    df = pd.DataFrame(payload.records)
    results = active_detector.detect_batch(df)

    anomalies = [r for r in results if r["is_anomaly"]]

    # Update global stats
    session_stats["total_analyzed"] += len(results)
    session_stats["anomalies_count"] += len(anomalies)

    # Append alerts
    for item in anomalies:
        alert_obj = {
            "id": f"ALT-{uuid.uuid4().hex[:6].upper()}",
            "timestamp": item.get("timestamp", datetime.datetime.now().strftime("%H:%M:%S")),
            "severity": item["severity"],
            "anomaly_score": item["anomaly_score"],
            "confidence_pct": item["confidence_pct"],
            "top_contributor": item["explainability"]["top_contributor"],
            "metrics": item["metrics"],
            "explainability": item["explainability"]
        }
        alerts_history.insert(0, alert_obj)
        if len(alerts_history) > 200:
            alerts_history.pop()

    return {
        "total_records": len(results),
        "anomalies_found": len(anomalies),
        "anomaly_rate_pct": round((len(anomalies) / max(1, len(results))) * 100.0, 2),
        "predictions": results
    }

@router.post("/stream/next")
def stream_next(req: StreamNextRequest):
    """Generates the next real-time telemetry frame, scores it, logs alerts if anomalous, and returns result."""
    cat = req.category if req.category in generator_instances else "server"
    gen = generator_instances[cat]

    # Configure detector
    active_detector.set_parameters(model_id=req.model_id, contamination=req.contamination)

    # Generate frame
    frame = gen.generate_next_frame(force_anomaly=req.force_anomaly)
    df = pd.DataFrame([frame])

    start_t = time.time()
    results = active_detector.detect_batch(df)
    latency_ms = round((time.time() - start_t) * 1000.0 + 1.2, 2)

    result_item = results[0]
    result_item["latency_ms"] = latency_ms

    # Stats tracking
    session_stats["total_analyzed"] += 1
    session_stats["confidence_sum"] += result_item["confidence_pct"]

    if result_item["is_anomaly"]:
        session_stats["anomalies_count"] += 1
        alert_obj = {
            "id": f"ALT-{uuid.uuid4().hex[:6].upper()}",
            "timestamp": frame.get("timestamp", datetime.datetime.now().strftime("%H:%M:%S")),
            "severity": result_item["severity"],
            "anomaly_score": result_item["anomaly_score"],
            "confidence_pct": result_item["confidence_pct"],
            "top_contributor": result_item["explainability"]["top_contributor"],
            "metrics": result_item["metrics"],
            "explainability": result_item["explainability"]
        }
        alerts_history.insert(0, alert_obj)
        if len(alerts_history) > 200:
            alerts_history.pop()

    return {
        "frame": frame,
        "result": result_item,
        "summary": {
            "total_analyzed": session_stats["total_analyzed"],
            "anomalies_count": session_stats["anomalies_count"],
            "anomaly_rate_pct": round((session_stats["anomalies_count"] / max(1, session_stats["total_analyzed"])) * 100.0, 2),
            "avg_confidence_pct": round(session_stats["confidence_sum"] / max(1, session_stats["total_analyzed"]), 1),
            "latency_ms": latency_ms
        }
    }

@router.get("/datasets/sample")
def get_sample_dataset(category: str = Query(default="server")):
    """Returns historical sample benchmark dataset for instant UI preview."""
    cat = category if category in generator_instances else "server"
    gen = generator_instances[cat]
    df = gen.generate_batch(num_samples=40, anomaly_rate=0.12)
    results = active_detector.detect_batch(df)
    return {
        "category": cat,
        "total_records": len(results),
        "anomalies_count": sum(1 for r in results if r["is_anomaly"]),
        "records": results
    }

@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), model_id: str = "isolation_forest", contamination: float = 0.05):
    """Ingests uploaded CSV/JSON/Excel dataset file, runs anomaly detection, and returns summary."""
    try:
        content = await file.read()
        filename = file.filename.lower().strip()
        df = None

        if ".csv" in filename or filename.endswith(".csv"):
            import io
            try:
                text_content = content.decode("utf-8", errors="ignore")
                df = pd.read_csv(io.StringIO(text_content))
            except Exception:
                df = pd.read_csv(io.BytesIO(content))
        elif ".xlsx" in filename or ".xls" in filename or filename.endswith((".xlsx", ".xls")):
            import io
            try:
                xl = pd.ExcelFile(io.BytesIO(content))
                # Smart sheet selection: filter out README / info / sample sheets
                candidate_sheets = [s for s in xl.sheet_names if not any(kw in s.lower() for kw in ["readme", "summary", "instruction", "info", "sample_submission"])]
                if not candidate_sheets:
                    candidate_sheets = xl.sheet_names
                
                # Select candidate sheet with maximum row count
                best_sheet = max(candidate_sheets, key=lambda s: len(xl.parse(s)))
                df = xl.parse(best_sheet)
            except ImportError:
                raise HTTPException(status_code=500, detail="Excel engine 'openpyxl' is missing on server. Run: pip install openpyxl")
            except Exception as ex:
                raise HTTPException(status_code=400, detail=f"Failed to parse Excel spreadsheet: {str(ex)}")
        elif ".json" in filename or filename.endswith(".json"):
            import io
            import json
            try:
                parsed_json = json.loads(content.decode("utf-8", errors="ignore"))
                if isinstance(parsed_json, dict):
                    # Check common container keys
                    for key in ["records", "data", "items", "predictions", "telemetry"]:
                        if key in parsed_json and isinstance(parsed_json[key], list):
                            parsed_json = parsed_json[key]
                            break
                if isinstance(parsed_json, list):
                    df = pd.DataFrame(parsed_json)
                else:
                    df = pd.read_json(io.BytesIO(content))
            except Exception:
                df = pd.read_json(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv, .json, .xlsx, or .xls file.")

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or could not be parsed.")

        active_detector.set_parameters(model_id=model_id, contamination=contamination)
        results = active_detector.detect_batch(df)
        anomalies = [r for r in results if r["is_anomaly"]]

        # Sync session stats and global alerts feed
        session_stats["total_analyzed"] += len(results)
        session_stats["anomalies_count"] += len(anomalies)

        for item in anomalies:
            alert_obj = {
                "id": f"ALT-{uuid.uuid4().hex[:6].upper()}",
                "timestamp": item.get("timestamp", datetime.datetime.now().strftime("%H:%M:%S")),
                "severity": item["severity"],
                "anomaly_score": item["anomaly_score"],
                "confidence_pct": item["confidence_pct"],
                "top_contributor": item["explainability"]["top_contributor"],
                "metrics": item["metrics"],
                "explainability": item["explainability"]
            }
            alerts_history.insert(0, alert_obj)
            if len(alerts_history) > 200:
                alerts_history.pop()

        # For UI chart performance on large datasets, return preview sample of results
        preview_results = results
        if len(results) > 500:
            # Sample max 500 records evenly across the dataset for chart rendering
            step = len(results) // 500
            preview_results = results[::step][:500]

        return {
            "filename": file.filename,
            "total_records": len(results),
            "anomalies_found": len(anomalies),
            "anomaly_rate_pct": round((len(anomalies) / max(1, len(results))) * 100.0, 2),
            "results": preview_results
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset file: {str(e)}")



@router.get("/alerts")
def get_alerts(severity: Optional[str] = None):
    """Returns recent anomaly alerts list filtered by severity."""
    if severity and severity.lower() != "all":
        filtered = [a for a in alerts_history if a["severity"].lower() == severity.lower()]
        return {"total": len(filtered), "alerts": filtered}
    return {"total": len(alerts_history), "alerts": alerts_history}

@router.get("/metrics/summary")
def get_metrics_summary():
    """Returns session performance metrics summary."""
    tot = session_stats["total_analyzed"]
    anom = session_stats["anomalies_count"]
    return {
        "total_analyzed": tot,
        "anomalies_count": anom,
        "anomaly_rate_pct": round((anom / max(1, tot)) * 100.0, 2),
        "avg_confidence_pct": round(session_stats["confidence_sum"] / max(1, tot), 1),
        "active_model": active_detector.model_id,
        "model_name": active_detector.SUPPORTED_MODELS.get(active_detector.model_id),
        "contamination_threshold": active_detector.contamination
    }
