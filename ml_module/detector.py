import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from ml_module.preprocessor import DataPreprocessor
from ml_module.explainer import AnomalyExplainer

class AnomalyDetector:
    """
    Multi-algorithm ensemble Anomaly Detector.
    Supports Isolation Forest, Local Outlier Factor, One-Class SVM, and Z-Score Ensemble.
    """
    SUPPORTED_MODELS = {
        "isolation_forest": "Isolation Forest",
        "local_outlier_factor": "Local Outlier Factor (LOF)",
        "one_class_svm": "One-Class SVM",
        "zscore_ensemble": "Z-Score / IQR Ensemble"
    }

    def __init__(self, model_id: str = "isolation_forest", contamination: float = 0.05):
        self.model_id = model_id if model_id in self.SUPPORTED_MODELS else "isolation_forest"
        self.contamination = contamination
        self.preprocessor = DataPreprocessor()
        self.explainer = AnomalyExplainer()
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initializes the selected scikit-learn or statistical model."""
        if self.model_id == "isolation_forest":
            self.model = IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_id == "local_outlier_factor":
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1
            )
        elif self.model_id == "one_class_svm":
            self.model = OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=self.contamination
            )
        elif self.model_id == "zscore_ensemble":
            self.model = "zscore_ensemble"

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fits the preprocessor and algorithm model on baseline dataframe."""
        scaled_data, _ = self.preprocessor.fit_transform(df)
        if self.model != "zscore_ensemble":
            self.model.fit(scaled_data)
        return self

    def set_parameters(self, model_id: Optional[str] = None, contamination: Optional[float] = None):
        """Updates parameters and re-initializes model if algorithm changes."""
        changed = False
        if model_id and model_id in self.SUPPORTED_MODELS and model_id != self.model_id:
            self.model_id = model_id
            changed = True
        if contamination is not None and contamination != self.contamination:
            self.contamination = max(0.01, min(0.30, contamination))
            changed = True

        if changed:
            self._init_model()

    def _score_zscore(self, scaled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Statistical Z-Score scoring."""
        # Calculate max absolute z-score per record across features
        max_z = np.max(np.abs(scaled_data), axis=1)
        # Threshold dynamic based on contamination
        threshold = 2.0 + (1.0 - self.contamination * 4)
        preds = np.where(max_z > threshold, -1, 1)

        # Normalize score into [0, 1] interval
        raw_scores = max_z / (threshold * 2.0)
        norm_scores = np.clip(raw_scores, 0.0, 1.0)
        return preds, norm_scores

    def detect_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Runs anomaly detection on a batch DataFrame.
        Returns list of structured prediction records.
        """
        scaled_data, feature_names = self.preprocessor.transform(df)
        num_records = len(df)

        if self.model_id == "zscore_ensemble":
            preds, scores = self._score_zscore(scaled_data)
        else:
            # Fit if not trained yet
            try:
                preds = self.model.predict(scaled_data)
            except Exception:
                self.model.fit(scaled_data)
                preds = self.model.predict(scaled_data)

            # Extract decision function score if available
            if hasattr(self.model, "score_samples"):
                raw_scores = -self.model.score_samples(scaled_data)
            elif hasattr(self.model, "decision_function"):
                raw_scores = -self.model.decision_function(scaled_data)
            else:
                raw_scores = np.max(np.abs(scaled_data), axis=1)

            # Normalize scores to [0.0, 1.0] interval
            s_min, s_max = np.min(raw_scores), np.max(raw_scores)
            if s_max > s_min:
                scores = (raw_scores - s_min) / (s_max - s_min)
            else:
                scores = np.zeros(num_records)

        results = []
        for i in range(num_records):
            is_anomaly = bool(preds[i] == -1)
            score = float(scores[i])
            if is_anomaly and score < 0.5:
                score = round(0.65 + score * 0.35, 3)

            # Calculate severity
            if is_anomaly:
                if score >= 0.85:
                    severity = "critical"
                elif score >= 0.70:
                    severity = "high"
                else:
                    severity = "medium"
            else:
                severity = "low"

            raw_record = df.iloc[i].to_dict()
            explanation = self.explainer.explain(scaled_data[i], feature_names, raw_record)

            res_item = {
                "record_index": i,
                "is_anomaly": is_anomaly,
                "anomaly_score": round(score, 3),
                "confidence_pct": round((score if is_anomaly else 1.0 - score) * 100.0, 1),
                "severity": severity,
                "metrics": {feat: round(float(raw_record.get(feat, 0.0)), 2) for feat in feature_names},
                "explainability": explanation
            }
            if "timestamp" in raw_record:
                res_item["timestamp"] = str(raw_record["timestamp"])
            results.append(res_item)

        return results

    def detect_single(self, record_dict: Dict[str, float]) -> Dict[str, Any]:
        """Detects anomaly for a single record dict."""
        df = pd.DataFrame([record_dict])
        results = self.detect_batch(df)
        return results[0]
