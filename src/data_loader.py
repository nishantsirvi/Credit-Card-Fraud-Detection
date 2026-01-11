"""Data Loading & Understanding Module"""

import pandas as pd
import numpy as np


def load_dataset(file_path):
    """Load the credit card fraud dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully")
    return df


def understand_data(df):
    """Perform initial data understanding"""
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found")
    else:
        print(missing[missing > 0])
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def analyze_class_distribution(df, target_col='Class'):
    """Analyze the distribution of fraud vs legitimate transactions"""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    class_counts = df[target_col].value_counts()
    class_percentages = df[target_col].value_counts(normalize=True) * 100
    
    print(f"\nClass Distribution:")
    print(f"  Legitimate (0): {class_counts[0]:,} transactions ({class_percentages[0]:.2f}%)")
    print(f"  Fraud (1):      {class_counts[1]:,} transactions ({class_percentages[1]:.2f}%)")
    
    imbalance_ratio = class_counts[0] / class_counts[1]
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    
    return class_counts, class_percentages


def get_feature_info(df):
    """Get information about features in the dataset"""
    print("\n" + "="*60)
    print("FEATURE INFORMATION")
    print("="*60)
    
    pca_features = [col for col in df.columns if col.startswith('V')]
    print(f"\nPCA Features (V1-V28): {len(pca_features)} features")
    
    # Amount feature
    if 'Amount' in df.columns:
        print(f"\n💰 Amount Feature:")
        print(f"   Min:    ${df['Amount'].min():.2f}")
        print(f"   Max:    ${df['Amount'].max():.2f}")
        print(f"   Mean:   ${df['Amount'].mean():.2f}")
        print(f"   Median: ${df['Amount'].median():.2f}")
    
    # Time feature
    if 'Time' in df.columns:
        print(f"\n⏰ Time Feature:")
        print(f"   Represents seconds elapsed between transactions")
        print(f"   Range: {df['Time'].min():.0f}s to {df['Time'].max():.0f}s")
        print(f"   Duration: {df['Time'].max() / 3600:.1f} hours")
    
    # Target feature
    if 'Class' in df.columns:
        print(f"\n🎯 Target Feature (Class):")
        print(f"   0 = Legitimate transaction")
        print(f"   1 = Fraudulent transaction")


def get_dataset_summary(file_path):
    """Complete dataset summary combining all analysis functions"""
    df = load_dataset(file_path)
    understand_data(df)
    analyze_class_distribution(df)
    get_feature_info(df)
    
    print("\n" + "="*60)
    print("Data Loading & Understanding Complete")
    print("="*60 + "\n")
    
    return df


if __name__ == "__main__":
    file_path = "../data/creditcard.csv"
    df = get_dataset_summary(file_path)
