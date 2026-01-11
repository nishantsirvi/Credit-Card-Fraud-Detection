"""
Data Cleaning Module
Comprehensive data cleaning and quality checks
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def check_data_quality(df):
    """
    Comprehensive data quality check.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Quality report
    """
    print("\nData Quality Assessment")
    print("="*60)
    
    quality_report = {}
    
    missing_counts = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    quality_report['missing_values'] = missing_counts[missing_counts > 0].to_dict()
    
    print("\nMissing Values:")
    if missing_counts.sum() == 0:
        print("No missing values detected")
    else:
        print(f"Found missing values in {len(quality_report['missing_values'])} columns:")
        for col, count in quality_report['missing_values'].items():
            print(f"  - {col}: {count} ({missing_percent[col]:.2f}%)")
    
    duplicates = df.duplicated().sum()
    quality_report['duplicates'] = duplicates
    
    print(f"\nDuplicate Rows:")
    if duplicates == 0:
        print("No duplicate rows detected")
    else:
        print(f"Found {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
    
    print(f"\nData Types:")
    print(df.dtypes.value_counts().to_string())
    quality_report['dtypes'] = df.dtypes.to_dict()
    
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    quality_report['constant_columns'] = constant_cols
    
    print(f"\nConstant Columns:")
    if len(constant_cols) == 0:
        print("No constant columns detected")
    else:
        print(f"Found {len(constant_cols)} constant columns: {constant_cols}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            outlier_info[col] = {
                'count': outliers,
                'percentage': outliers/len(df)*100
            }
    
    quality_report['outliers'] = outlier_info
    
    print(f"\nOutliers (using IQR method):")
    if len(outlier_info) == 0:
        print("No significant outliers detected")
    else:
        print(f"Found outliers in {len(outlier_info)} columns:")
        for col, info in list(outlier_info.items())[:5]:
            print(f"  - {col}: {info['count']} ({info['percentage']:.2f}%)")
    
    print(f"\nColumn Cardinality:")
    for col in df.columns[:10]:
        unique = df[col].nunique()
        print(f"  - {col}: {unique} unique values")
    
    return quality_report


def clean_missing_values(df, strategy='drop', fill_value=None):
    """
    Handle missing values.
    
    Args:
        df (pd.DataFrame): Input dataset
        strategy (str): 'drop', 'mean', 'median', 'mode', or 'fill'
        fill_value: Value to fill if strategy='fill'
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print(f"\nHandling missing values with strategy: {strategy}")
    
    original_shape = df.shape
    
    if strategy == 'drop':
        df_clean = df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_clean = df.copy()
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    elif strategy == 'mode':
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    elif strategy == 'fill':
        df_clean = df.fillna(fill_value)
    else:
        df_clean = df.copy()
    
    removed_rows = original_shape[0] - df_clean.shape[0]
    if removed_rows > 0:
        print(f"   Removed {removed_rows} rows with missing values")
    
    print(f"   Dataset shape: {original_shape} → {df_clean.shape}")
    
    return df_clean


def remove_duplicates(df):
    """
    Remove duplicate rows.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset without duplicates
    """
    print("\nRemoving duplicate rows...")
    
    original_shape = df.shape
    df_clean = df.drop_duplicates()
    removed_rows = original_shape[0] - df_clean.shape[0]
    
    if removed_rows > 0:
        print(f"  Removed {removed_rows} duplicate rows")
    else:
        print("  No duplicates found")
    
    print(f"  Dataset shape: {original_shape} -> {df_clean.shape}")
    
    return df_clean


def handle_outliers(df, columns=None, method='clip', threshold=3):
    """
    Handle outliers in numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        columns (list): List of columns to process (None = all numeric)
        method (str): 'clip', 'remove', or 'cap'
        threshold (float): Number of standard deviations (for method='clip')
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
    """
    print(f"\nHandling outliers with method: {method}")
    
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    original_shape = df_clean.shape
    
    for col in columns:
        if method == 'clip':
            # Clip using IQR method
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
            if before > 0:
                print(f"   - {col}: Clipped {before} outliers")
        
        elif method == 'remove':
            # Remove rows with outliers
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            df_clean = df_clean[
                (df_clean[col] >= Q1 - 1.5 * IQR) & 
                (df_clean[col] <= Q3 + 1.5 * IQR)
            ]
    
    removed_rows = original_shape[0] - df_clean.shape[0]
    if removed_rows > 0:
        print(f"   Removed {removed_rows} rows with outliers")
    
    print(f"   Dataset shape: {original_shape} → {df_clean.shape}")
    
    return df_clean


def validate_fraud_detection_data(df):
    """
    Validate credit card fraud detection dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (is_valid, issues)
    """
    print("\nFraud Detection Data Validation")
    print("="*60)
    
    issues = []
    
    if 'Class' not in df.columns:
        issues.append("'Class' column not found")
        print("'Class' column not found")
    else:
        unique_classes = df['Class'].unique()
        print(f"Class column found with values: {unique_classes}")
        
        if not set(unique_classes).issubset({0, 1}):
            issues.append("Class column should only contain 0 and 1")
            print("Class column should only contain 0 and 1")
        else:
            class_dist = df['Class'].value_counts()
            print(f"\nClass Distribution:")
            print(f"  - Legitimate (0): {class_dist[0]} ({class_dist[0]/len(df)*100:.2f}%)")
            print(f"  - Fraud (1): {class_dist[1]} ({class_dist[1]/len(df)*100:.2f}%)")
            
            fraud_ratio = class_dist[1] / len(df) * 100
            if fraud_ratio < 0.1:
                print(f"Severe class imbalance detected ({fraud_ratio:.3f}% fraud)")
    
    if 'Amount' in df.columns:
        print(f"\nAmount column found")
        print(f"  - Range: ${df['Amount'].min():.2f} to ${df['Amount'].max():.2f}")
        print(f"  - Mean: ${df['Amount'].mean():.2f}")
    else:
        issues.append("'Amount' column not found")
        print("'Amount' column not found")
    
    if 'Time' in df.columns:
        print(f"\nTime column found")
        print(f"  - Range: {df['Time'].min():.0f} to {df['Time'].max():.0f} seconds")
    else:
        print("'Time' column not found (optional)")
    
    v_columns = [col for col in df.columns if col.startswith('V')]
    print(f"\nFound {len(v_columns)} PCA feature columns (V1-V{len(v_columns)})")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        print("\n" + "="*60)
        print("Validation Passed - Dataset is ready for modeling")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Validation Failed")
        print("="*60)
        for issue in issues:
            print(f"  {issue}")
    
    return is_valid, issues


def clean_fraud_dataset(df, remove_duplicates_flag=True, handle_missing='drop'):
    """
    Complete cleaning pipeline for fraud detection dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
        remove_duplicates_flag (bool): Whether to remove duplicates
        handle_missing (str): Strategy for missing values
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("\nStarting Data Cleaning Pipeline")
    print("="*60)
    
    print(f"\nInput dataset shape: {df.shape}")
    
    quality_report = check_data_quality(df)
    
    if df.isnull().sum().sum() > 0:
        df = clean_missing_values(df, strategy=handle_missing)
    else:
        print("\nNo missing values to handle")
    
    if remove_duplicates_flag:
        df = remove_duplicates(df)
    
    is_valid, issues = validate_fraud_detection_data(df)
    
    print("\n" + "="*60)
    print("Data Cleaning Completed")
    print("="*60)
    print(f"Output dataset shape: {df.shape}")
    
    return df, quality_report
