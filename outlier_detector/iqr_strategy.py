from .strategy import OutlierDetectionStrategy
import pandas as pd

class IQRStrategy(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        return df.loc[(df[col] >= lower_fence) & (df[col] <= upper_fence)]
