from .strategy import OutlierDetectionStrategy
import pandas as pd
import numpy as np

class ZScoreStrategy(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        threshold = 3
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        return df.loc[z_scores <= threshold]
