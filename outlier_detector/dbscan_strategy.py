from .strategy import OutlierDetectionStrategy
import pandas as pd
from sklearn.cluster import DBSCAN

class DBScanStrategy(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        eps = 0.5
        min_samples = 5
        X = df[[col]].values
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        return df[dbscan.labels_ != -1]
