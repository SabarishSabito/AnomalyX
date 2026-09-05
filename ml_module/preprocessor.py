import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ID_COLUMN_PATTERNS = ["record_id", "id", "_id", "index", "uuid", "user_id", "patient_id", "employee_id"]

class DataPreprocessor:
    """
    Handles feature extraction, missing value imputation, and standard scaling
    for multi-metric telemetry streams and batch datasets.
    """
    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.default_features = ["cpu_usage", "memory_usage", "network_io", "latency_ms", "error_rate"]
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.is_fitted = False

    def is_id_column(self, col_name: str) -> bool:
        """Determines if a column name matches standard identifier naming patterns."""
        name_lower = col_name.lower().strip()
        return name_lower in ID_COLUMN_PATTERNS or name_lower.endswith("_id") or name_lower.startswith("id_")

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts available numerical columns or matches default feature columns."""
        # 1. Check if explicit feature_columns are present
        if self.feature_columns:
            existing_cols = [col for col in self.feature_columns if col in df.columns]
            if existing_cols:
                sub_df = pd.DataFrame()
                for col in existing_cols:
                    sub_df[col] = pd.to_numeric(df[col], errors="coerce")
                return sub_df

        # 2. Check if default telemetry columns exist
        default_existing = [col for col in self.default_features if col in df.columns]
        if default_existing:
            sub_df = pd.DataFrame()
            for col in default_existing:
                sub_df[col] = pd.to_numeric(df[col], errors="coerce")
            self.feature_columns = default_existing
            return sub_df

        # 3. Dynamic numeric feature extraction (excluding ID columns)
        candidate_cols = []
        sub_df = pd.DataFrame()

        for col in df.columns:
            if self.is_id_column(col):
                continue
            # Try coercing to numeric
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_null_count = coerced.notna().sum()
            # If at least 20% of rows (or > 0 for small datasets) are numeric, include it
            if non_null_count > 0 and (non_null_count / max(1, len(df))) >= 0.2:
                candidate_cols.append(col)
                sub_df[col] = coerced

        if candidate_cols:
            self.feature_columns = candidate_cols
            return sub_df

        # 4. Fallback if no non-ID numeric columns found: take all numeric non-ID or all numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_id_numeric = [c for c in numeric_cols if not self.is_id_column(c)]
        cols_to_use = non_id_numeric if non_id_numeric else numeric_cols

        if not cols_to_use:
            raise ValueError("No numerical features found in dataset")

        self.feature_columns = cols_to_use
        return df[cols_to_use].copy()

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fits the imputer and standard scaler on training baseline data."""
        features_df = self.extract_features(df)
        imputed_data = self.imputer.fit_transform(features_df)
        self.scaler.fit(imputed_data)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Transforms input dataframe into scaled numpy feature matrix."""
        features_df = self.extract_features(df)
        cols = list(features_df.columns)
        
        # Check if feature columns changed from previous fit
        cols_changed = False
        if hasattr(self.imputer, "feature_names_in_"):
            imputer_cols = list(getattr(self.imputer, "feature_names_in_"))
            if imputer_cols != cols:
                cols_changed = True
        elif hasattr(self.scaler, "feature_names_in_"):
            scaler_cols = list(getattr(self.scaler, "feature_names_in_"))
            if scaler_cols != cols:
                cols_changed = True

        if not self.is_fitted or cols_changed:
            imputed_data = self.imputer.fit_transform(features_df)
            scaled_data = self.scaler.fit_transform(imputed_data)
            self.is_fitted = True
            self.features_changed = True
        else:
            imputed_data = self.imputer.transform(features_df)
            scaled_data = self.scaler.transform(imputed_data)
            self.features_changed = False

        return scaled_data, cols



    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fits and transforms input dataframe."""
        features_df = self.extract_features(df)
        cols = list(features_df.columns)
        imputed_data = self.imputer.fit_transform(features_df)
        scaled_data = self.scaler.fit_transform(imputed_data)
        self.is_fitted = True
        return scaled_data, cols

