from abc import ABC, abstractmethod
import pandas as pd

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        pass
