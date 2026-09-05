import numpy as np
import pandas as pd
from typing import Dict, List, Any

class AnomalyExplainer:
    """
    Computes feature attribution and root-cause contribution percentages for detected anomalies
    by measuring individual feature deviations against normal baseline distributions.
    """
    def __init__(self):
        pass

    def explain(self, scaled_record: np.ndarray, feature_names: List[str], raw_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculates normalized contribution percentage for each feature.
        scaled_record: 1D array of z-scores / scaled values for the single record.
        """
        abs_z_scores = np.abs(scaled_record)
        total_deviation = np.sum(abs_z_scores)

        if total_deviation == 0:
            contributions = {feat: round(100.0 / len(feature_names), 2) for feat in feature_names}
            top_feature = feature_names[0]
        else:
            contributions = {
                feat: round(float((abs_z_scores[i] / total_deviation) * 100.0), 2)
                for i, feat in enumerate(feature_names)
            }
            top_feature = max(contributions, key=contributions.get)

        # Build feature details list
        feature_breakdown = []
        for i, feat in enumerate(feature_names):
            feature_breakdown.append({
                "feature": feat,
                "raw_value": round(float(raw_values.get(feat, 0.0)), 2),
                "z_score": round(float(scaled_record[i]), 2),
                "contribution_pct": contributions[feat]
            })

        # Sort breakdown by contribution descending
        feature_breakdown.sort(key=lambda x: x["contribution_pct"], reverse=True)

        return {
            "top_contributor": top_feature,
            "top_contribution_pct": contributions[top_feature],
            "feature_attributions": contributions,
            "feature_breakdown": feature_breakdown
        }
