from .strategy import OutlierDetectionStrategy
import pandas as pd

class MADStrategy(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        threshold = 3.5
        median = df[col].median()
        mad = df[col].mad()
        lower_fence = median - threshold * mad
        upper_fence = median + threshold * mad
        return df.loc[(df[col] >= lower_fence) & (df[col] <= upper_fence)]
