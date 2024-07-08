import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

df=pd.read_csv("data/dataset_with_outliers.csv")
print(df.head()) 
print('shape of data:',df.shape)

print(df.columns)

# Inter Quartile Range (IQR)

def handle_outliers_with_iqr(df, col, remove_or_fill_with_quartile=None, impute_logic='median'):
    """
    Detect outliers using IQR (Interquartile Range) and handle them based on specified logic.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data.
    - col (str): Name of the column in the DataFrame to detect outliers and handle.
    - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). 
      If 'fill', impute with quartile-based values.
    - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').

    Returns:
    - df (DataFrame): DataFrame with outliers handled based on specified logic.
    """
    # Calculate quartiles and IQR
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    
    # Calculate fences for outlier detection
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # Print information about outliers
    print('Lower Fence:', lower_fence)
    print('Upper Fence:', upper_fence)
    print('Total number of outliers left:', df[(df[col] < lower_fence) | (df[col] > upper_fence)].shape[0])
    
    # Handle outliers based on specified logic
    if remove_or_fill_with_quartile == "drop":
        df = df.loc[(df[col] >= lower_fence) & (df[col] <= upper_fence)]
    
    elif remove_or_fill_with_quartile == "fill":
        if impute_logic == 'median':
            imputed_value = df[col].median()
        elif impute_logic == 'mean':
            imputed_value = df[col].mean()
        else:
            raise ValueError("Unsupported impute_logic. Please choose 'median' or 'mean' for fill imputation.")
        
        df[col] = np.where(df[col] < lower_fence, imputed_value, df[col])
        df[col] = np.where(df[col] > upper_fence, imputed_value, df[col])
    
    return df

def handle_outliers_with_lof(df, column_name, n_neighbors=20, contamination=0.1, impute_logic='median', remove_or_fill_with_quartile=None):
    """
    Detect outliers using LocalOutlierFactor and impute them based on specified imputation logic.
    Impute outliers with mean or median value of the column.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data.
    - column_name (str): Name of the column in the DataFrame to detect outliers and impute.
    - n_neighbors (int): Number of neighbors to consider for LocalOutlierFactor.
    - contamination (float): Proportion of outliers expected in the data.
    - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').
    - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). If 'fill', impute with quartile-based values.

    Returns:
    - df (DataFrame): DataFrame with outliers imputed based on specified logic.
    """
    # Initialize LocalOutlierFactor model
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    
    # Fit model and predict outliers
    outlier_scores = lof.fit_predict(df[[column_name]])
    outlier_mask = outlier_scores == -1  # -1 indicates outlier according to LOF
    
    # Calculate quartiles for imputation logic
    if remove_or_fill_with_quartile == "fill":
        if impute_logic == 'median':
            imputed_value = df[column_name].median()
        elif impute_logic == 'mean':
            imputed_value = df[column_name].mean()
        else:
            raise ValueError("Unsupported impute_logic. Please choose 'median' or 'mean' for fill imputation.")
        
        df[column_name] = np.where(outlier_mask, imputed_value, df[column_name])
    
    elif remove_or_fill_with_quartile == "drop":
        df = df.loc[~outlier_mask]
    
    return df


def handle_outliers_with_zscore(df, column_name, threshold=3, remove_or_fill_with_quartile=None, impute_logic='median'):
    """
    Detect outliers using z-score and handle them based on specified logic.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data.
    - column_name (str): Name of the column in the DataFrame to detect outliers and handle.
    - threshold (float): Z-score threshold to identify outliers.
    - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). If 'fill', impute with quartile-based values.
    - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').

    Returns:
    - df (DataFrame): DataFrame with outliers handled based on specified logic.
    """
    # Calculate z-score for the column
    df['z_score'] = st.zscore(df[column_name])
    
    # Identify outliers based on z-score
    outlier_mask = np.abs(df['z_score']) > threshold
    
    # Calculate quartiles for imputation logic
    if remove_or_fill_with_quartile == "fill":
        if impute_logic == 'median':
            imputed_value = df[column_name].median()
        elif impute_logic == 'mean':
            imputed_value = df[column_name].mean()
        else:
            raise ValueError("Unsupported impute_logic. Please choose 'median' or 'mean' for fill imputation.")
        
        df[column_name] = np.where(outlier_mask, imputed_value, df[column_name])
    
    elif remove_or_fill_with_quartile == "drop":
        df = df.loc[~outlier_mask]
    
    # Drop z_score column
    df = df.drop(columns=['z_score'])
    
    return df

def handle_outliers_with_dbscan(df, col, eps=0.5, min_samples=5, remove_or_fill_with_quartile=None, impute_logic='median'):
    """
    Detect outliers using DBSCAN and handle them based on specified logic.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data.
    - col (str): Name of the column in the DataFrame to detect outliers and handle.
    - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). 
      If 'fill', impute with quartile-based values.
    - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').

    Returns:
    - df (DataFrame): DataFrame with outliers handled based on specified logic.
    """
    # Reshape the data for DBSCAN fitting
    X = df[[col]].values

    # Fit DBSCAN model
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    # Identify outliers
    outlier_mask = dbscan.labels_ == -1  # -1 indicates outliers

    # Print information about outliers
    print('Total number of outliers:', np.sum(outlier_mask))

    # Handle outliers based on specified logic
    if remove_or_fill_with_quartile == "drop":
        df = df.loc[~outlier_mask]
    
    elif remove_or_fill_with_quartile == "fill":
        if impute_logic == 'median':
            imputed_value = df[col].median()
        elif impute_logic == 'mean':
            imputed_value = df[col].mean()
        else:
            raise ValueError("Unsupported impute_logic. Please choose 'median' or 'mean' for fill imputation.")
        
        df[col] = np.where(outlier_mask, imputed_value, df[col])
    
    return df

def mad_based_outlier(df, col, threshold=3.5, remove_or_fill_with_quartile=None, impute_logic='median'):
    """
    Detect outliers using Mean Absolute Deviation (MAD) and handle them based on specified logic.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data.
    - col (str): Name of the column in the DataFrame to detect outliers and handle.
    - threshold (float): Threshold value to determine outliers based on MAD (default is 3.5).
    - remove_or_fill_with_quartile (str or None): Action to take with outliers ('drop', 'fill'). 
      If 'fill', impute with quartile-based values.
    - impute_logic (str): Imputation logic for outliers ('median', 'mean', 'specific_value').

    Returns:
    - df (DataFrame): DataFrame with outliers handled based on specified logic.
    """
    # Calculate MAD
    mad = np.abs(df[col] - df[col].median()).median()
    
    # Calculate lower and upper fences
    lower_fence = df[col].median() - threshold * mad
    upper_fence = df[col].median() + threshold * mad
    
    # Print information about outliers
    print('Lower Fence:', lower_fence)
    print('Upper Fence:', upper_fence)
    
    # Identify outliers
    outlier_mask = (df[col] < lower_fence) | (df[col] > upper_fence)

    # Handle outliers based on specified logic
    if remove_or_fill_with_quartile == "drop":
        df = df.loc[~outlier_mask]
    
    elif remove_or_fill_with_quartile == "fill":
        if impute_logic == 'median':
            imputed_value = df[col].median()
        elif impute_logic == 'mean':
            imputed_value = df[col].mean()
        else:
            raise ValueError("Unsupported impute_logic. Please choose 'median' or 'mean' for fill imputation.")
        
        df[col] = np.where(outlier_mask, imputed_value, df[col])
    
    return df
        
# identifying_treating_outliers(df,'Scrape Sale Qty','drop')
# identifying_treating_outliers(df,'Rate Rs/Kg','drop')

# Z-score Method

# Applying Zscore in Scrap Rate column defining dataframe by dfn

# Calculate z-scores
z_scores = np.abs(st.zscore(df['Rate Rs/Kg']))

# Define threshold for outlier detection (e.g., z-score > 3)
threshold = 3

# Filter rows based on z-score outliers
dfn = df[(z_scores > threshold)]

# Display the outliers
print(dfn)

z_scores = np.abs(st.zscore(dfn['Scrape Sale Qty']))

# Filter rows based on z-score outliers
dfnf = dfn[(z_scores > threshold)]

# Display the outliers
print(dfnf)

print('before df shape', df.shape)

print('After df shape for Observation dropped in Scrap Rate', dfn.shape)

print('After df shape for observation dropped in weight', dfnf.shape)