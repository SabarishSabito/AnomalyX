import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """
    Handles feature extraction, missing value imputation, and standard scaling
    for multi-metric telemetry streams and batch datasets.
    """
    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or ["cpu_usage", "memory_usage", "network_io", "latency_ms", "error_rate"]
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.is_fitted = False

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts available numerical columns or matches default feature columns."""
        existing_cols = [col for col in self.feature_columns if col in df.columns]
        if not existing_cols:
            # Fall back to all numeric columns in dataframe
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numerical features found in dataset")
            existing_cols = numeric_cols
            self.feature_columns = existing_cols

        return df[existing_cols].copy()

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
        if not self.is_fitted:
            # Auto-fit if not fitted yet
            imputed_data = self.imputer.fit_transform(features_df)
            scaled_data = self.scaler.fit_transform(imputed_data)
            self.is_fitted = True
        else:
            imputed_data = self.imputer.transform(features_df)
            scaled_data = self.scaler.transform(imputed_data)
        return scaled_data, cols

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fits and transforms input dataframe."""
        features_df = self.extract_features(df)
        cols = list(features_df.columns)
        imputed_data = self.imputer.fit_transform(features_df)
        scaled_data = self.scaler.fit_transform(imputed_data)
        self.is_fitted = True
        return scaled_data, cols
