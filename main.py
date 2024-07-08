from outlier_package.outlier_handler import OutlierHandler
from outlier_package.iqr_strategy import IQRStrategy
from outlier_package.zscore_strategy import ZScoreStrategy
import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'value': np.random.randn(100)
}
df = pd.DataFrame(data)

# Example usage with IQR strategy
handler = OutlierHandler(IQRStrategy())
cleaned_df = handler.handle_outliers(df, 'value')

# Example usage with Z-score strategy
handler = OutlierHandler(ZScoreStrategy())
cleaned_df = handler.handle_outliers(df, 'value')
