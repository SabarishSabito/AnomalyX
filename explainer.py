import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class AnomalyExplainer:
    """
    Computes feature attribution and root-cause contribution percentages for detected anomalies
    by measuring individual feature deviations against normal baseline distributions and data quality rules.
    """
    def __init__(self):
        pass

    def explain(
        self,
        scaled_record: np.ndarray,
        feature_names: List[str],
        raw_values: Dict[str, Any],
        quality_issues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculates normalized contribution percentage for each feature and quality defects.
        scaled_record: 1D array of z-scores / scaled values for the single record.
        """
        abs_z_scores = np.abs(scaled_record)
        total_deviation = np.sum(abs_z_scores)

        if total_deviation == 0:
            contributions = {feat: round(100.0 / max(1, len(feature_names)), 2) for feat in feature_names}
            top_feature = feature_names[0] if feature_names else "Data Quality"
        else:
            contributions = {
                feat: round(float((abs_z_scores[i] / total_deviation) * 100.0), 2)
                for i, feat in enumerate(feature_names)
            }
            top_feature = max(contributions, key=contributions.get) if contributions else "Data Quality"

        # Build feature details list
        feature_breakdown = []
        for i, feat in enumerate(feature_names):
            val = raw_values.get(feat, 0.0)
            try:
                numeric_val = float(val)
                if np.isnan(numeric_val) or pd.isna(numeric_val):
                    numeric_val = 0.0
                else:
                    numeric_val = round(numeric_val, 2)
            except (ValueError, TypeError):
                numeric_val = 0.0

            z_val = float(scaled_record[i])
            if np.isnan(z_val) or pd.isna(z_val):
                z_val = 0.0
            else:
                z_val = round(z_val, 2)

            feature_breakdown.append({
                "feature": feat,
                "raw_value": numeric_val,
                "z_score": z_val,
                "contribution_pct": contributions.get(feat, 0.0)
            })


        # Inject specific data quality issues into top_contributor if present
        if quality_issues:
            top_feature = quality_issues[0]
            for q_issue in quality_issues:
                feature_breakdown.append({
                    "feature": f"[DATA QUALITY] {q_issue}",
                    "raw_value": 0.0,
                    "z_score": 3.5,
                    "contribution_pct": round(95.0 / len(quality_issues), 1)
                })

        # Sort breakdown by contribution descending
        feature_breakdown.sort(key=lambda x: x["contribution_pct"], reverse=True)

        return {
            "top_contributor": top_feature,
            "top_contribution_pct": feature_breakdown[0]["contribution_pct"] if feature_breakdown else 100.0,
            "feature_attributions": contributions,
            "feature_breakdown": feature_breakdown
        }

