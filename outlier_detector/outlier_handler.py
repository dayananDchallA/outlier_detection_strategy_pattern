import pandas as pd
from .strategy import OutlierDetectionStrategy

class OutlierHandler:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def handle_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        return self.strategy.detect_outliers(df, col)
