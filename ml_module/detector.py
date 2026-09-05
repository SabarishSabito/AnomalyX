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

    def _check_record_quality(self, record: Dict[str, Any], df: pd.DataFrame, index: int) -> List[str]:
        """Evaluates data quality rules for unreliable records."""
        issues = []
        
        # 1. Corrupted / Outlier Age
        raw_age = record.get("age")
        if raw_age is not None and str(raw_age).strip() != "":
            age_str = str(raw_age).strip()
            try:
                age_val = float(age_str)
                if age_val > 120 or age_val < 0:
                    issues.append(f"Outlier Age ({int(age_val)})")
                elif age_val < 18 and age_val > 0:
                    issues.append(f"Underage Record ({int(age_val)})")
            except ValueError:
                issues.append(f"Corrupted Age Value ('{age_str}')")
        elif raw_age is None or str(raw_age).strip() == "":
            issues.append("Missing Age Value")

        # 2. Salary Defects
        raw_sal = record.get("salary")
        if raw_sal is not None and str(raw_sal).strip() != "":
            try:
                sal_val = float(raw_sal)
                if sal_val < 0:
                    issues.append(f"Negative Salary (${int(sal_val):,})")
                elif sal_val > 1000000:
                    issues.append(f"Extreme Salary Outlier (${int(sal_val):,})")
            except ValueError:
                issues.append(f"Corrupted Salary ('{raw_sal}')")

        # 3. Email Formatting
        raw_email = record.get("email")
        if raw_email is not None:
            email_str = str(raw_email).strip()
            if email_str and ("@" not in email_str or email_str.endswith("@") or email_str.startswith("@") or "." not in email_str.split("@")[-1]):
                issues.append(f"Invalid Email ('{email_str}')")
            elif not email_str:
                issues.append("Missing Email")

        # 4. Phone Number Validation
        raw_phone = record.get("phone")
        if raw_phone is not None:
            phone_str = str(raw_phone).strip()
            if phone_str:
                digits_only = "".join([c for c in phone_str if c.isdigit()])
                if " " in phone_str or len(digits_only) < 10:
                    issues.append(f"Malformed Phone ('{phone_str}')")
            else:
                issues.append("Missing Phone")

        # 5. Future Join Date / Format
        raw_date = record.get("join_date")
        if raw_date is not None and str(raw_date).strip() != "":
            date_str = str(raw_date).strip()
            if "2099" in date_str or "2100" in date_str:
                issues.append(f"Impossible Future Date ('{date_str}')")

        # 6. Duplicate ID Check
        record_id = record.get("record_id") or record.get("id")
        if record_id is not None:
            id_col = "record_id" if "record_id" in df.columns else ("id" if "id" in df.columns else None)
            if id_col and (df[id_col] == record_id).sum() > 1:
                issues.append(f"Duplicate Record ID ({record_id})")

        # 7. Blank Name
        raw_name = record.get("name")
        if raw_name is not None and str(raw_name).strip() == "":
            issues.append("Missing Name")

        return issues

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
            # Fit if not trained yet or if features changed
            if getattr(self.preprocessor, "features_changed", False):
                self.model.fit(scaled_data)

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
            raw_record = df.iloc[i].to_dict()
            quality_issues = self._check_record_quality(raw_record, df, i)

            is_anomaly = bool(preds[i] == -1) or len(quality_issues) > 0
            score = float(scores[i])

            if len(quality_issues) > 0:
                score = max(0.82, score)
                if any("Outlier" in q or "Extreme" in q or "Negative" in q or "Impossible" in q for q in quality_issues):
                    score = max(0.92, score)

            if is_anomaly and score < 0.5:
                score = round(0.65 + score * 0.35, 3)

            # Calculate severity
            if is_anomaly:
                if score >= 0.85 or any("Outlier" in q or "Negative" in q or "Corrupted" in q for q in quality_issues):
                    severity = "critical"
                elif score >= 0.70 or len(quality_issues) > 0:
                    severity = "high"
                else:
                    severity = "medium"
            else:
                severity = "low"

            explanation = self.explainer.explain(scaled_data[i], feature_names, raw_record, quality_issues)

            # Build displayable metrics dict
            display_metrics = {}
            for k, v in raw_record.items():
                if v is None or pd.isna(v):
                    display_metrics[k] = "N/A"
                elif isinstance(v, (int, float, np.number)):
                    val_flt = float(v)
                    if np.isnan(val_flt) or np.isinf(val_flt):
                        display_metrics[k] = "N/A"
                    else:
                        display_metrics[k] = round(val_flt, 2)
                else:
                    display_metrics[k] = str(v)


            res_item = {
                "record_index": i,
                "is_anomaly": is_anomaly,
                "anomaly_score": round(score, 3),
                "confidence_pct": round((score if is_anomaly else 1.0 - score) * 100.0, 1),
                "severity": severity,
                "metrics": display_metrics,
                "explainability": explanation
            }
            if "timestamp" in raw_record and raw_record["timestamp"] and not pd.isna(raw_record["timestamp"]):
                res_item["timestamp"] = str(raw_record["timestamp"])
            elif "join_date" in raw_record and raw_record["join_date"] and not pd.isna(raw_record["join_date"]):
                res_item["timestamp"] = str(raw_record["join_date"])
            results.append(res_item)

        return results

    def detect_single(self, record_dict: Dict[str, float]) -> Dict[str, Any]:
        """Detects anomaly for a single record dict."""
        df = pd.DataFrame([record_dict])
        results = self.detect_batch(df)
        return results[0]

