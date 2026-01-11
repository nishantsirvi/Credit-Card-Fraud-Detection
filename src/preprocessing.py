"""
Data Preprocessing Module
Data Preprocessing - Scaling, Balancing, Splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter


def scale_features(df, features_to_scale=None):
    """
    Scale specified features using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input dataset
        features_to_scale (list): List of features to scale (default: ['Amount'])
        
    Returns:
        pd.DataFrame: Dataset with scaled features
        StandardScaler: Fitted scaler object
    """
    if features_to_scale is None:
        features_to_scale = ['Amount']
    
    print("\nFeature Scaling")
    
    df_scaled = df.copy()
    scaler = StandardScaler()
    
    available_features = [f for f in features_to_scale if f in df.columns]
    
    if not available_features:
        print("No features to scale found")
        return df_scaled, None
    
    print(f"Scaling features: {', '.join(available_features)}")
    
    if 'Amount' in available_features:
        print(f"Before scaling - Mean: {df['Amount'].mean():.2f}, Std: {df['Amount'].std():.2f}")
    
    df_scaled[available_features] = scaler.fit_transform(df[available_features])
    
    if 'Amount' in available_features:
        print(f"After scaling - Mean: {df_scaled['Amount'].mean():.6f}, Std: {df_scaled['Amount'].std():.6f}")
    
    print("Feature scaling complete.")
    
    return df_scaled, scaler


def split_features_target(df, target_col='Class', drop_cols=None):
    """
    Split dataset into features (X) and target (y).
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        drop_cols (list): Additional columns to drop
        
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target
    """
    if drop_cols is None:
        drop_cols = ['Time']
    
    cols_to_drop = [target_col] + [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    print(f"\nFeatures shape: {X.shape}, Target shape: {y.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    return X, y


def perform_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size (float): Proportion of test set (default: 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\nTrain-Test Split")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Test size: {test_size * 100:.0f}%, Stratified: Yes")
    
    print(f"\nTraining Set: {len(X_train):,} samples")
    print(f"  Fraud: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"  Legit: {len(y_train) - y_train.sum():,} ({(len(y_train)-y_train.sum())/len(y_train)*100:.2f}%)")
    
    print(f"\nTesting Set: {len(X_test):,} samples")
    print(f"  Fraud: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")
    print(f"  Legit: {len(y_test) - y_test.sum():,} ({(len(y_test)-y_test.sum())/len(y_test)*100:.2f}%)")
    
    print("Train-test split complete.")
    
    return X_train, X_test, y_train, y_test


def handle_imbalance_with_smote(X_train, y_train, random_state=42):
    """
    Handle class imbalance using SMOTE (Synthetic Minority Over-sampling).
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state (int): Random seed
        
    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    print("\nHandling Class Imbalance with SMOTE")
    
    print(f"\nBefore SMOTE:")
    print(f"  Class distribution: {Counter(y_train)}")
    print(f"  Imbalance ratio: {(len(y_train) - y_train.sum()) / y_train.sum():.2f}:1")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE:")
    print(f"  Class distribution: {Counter(y_resampled)}")
    print(f"  Imbalance ratio: 1:1 (balanced)")
    
    print(f"\nDataset size: {len(X_train):,} -> {len(X_resampled):,} samples")
    print(f"New samples created: {len(X_resampled) - len(X_train):,}")
    
    print("SMOTE resampling complete.")
    
    return X_resampled, y_resampled


def undersample_majority_class(X_train, y_train, random_state=42):
    """
    Handle class imbalance by undersampling majority class.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state (int): Random seed
        
    Returns:
        X_resampled, y_resampled: Balanced dataset
    """
    from imblearn.under_sampling import RandomUnderSampler
    
    print("\nHandling Class Imbalance with Undersampling")
    
    print(f"\nBefore Undersampling:")
    print(f"  Class distribution: {Counter(y_train)}")
    print(f"  Total samples: {len(X_train):,}")
    
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print(f"\nAfter Undersampling:")
    print(f"  Class distribution: {Counter(y_resampled)}")
    print(f"  Total samples: {len(X_resampled):,}")
    print(f"  Samples removed: {len(X_train) - len(X_resampled):,}")
    
    print("Undersampling complete.")
    
    return X_resampled, y_resampled
    print(f"   Total samples: {len(X_train):,}")
    
    # Apply undersampling
    rus = RandomUnderSampler(random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print(f"\n📈 After Undersampling:")
    print(f"   Class distribution: {Counter(y_resampled)}")
    print(f"   Total samples: {len(X_resampled):,}")
    print(f"   Samples removed: {len(X_train) - len(X_resampled):,}")
    
    print("\n✅ Undersampling complete!")
    print("⚠️  Note: This reduces dataset size significantly")
    
    return X_resampled, y_resampled


def get_class_weights(y_train):
    """
    Calculate class weights for imbalanced data.
    
    Args:
        y_train: Training target
        
    Returns:
        dict: Class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    print("\nClass Weights for Imbalanced Data")
    print(f"  Class 0 (Legitimate): {class_weights[0]:.4f}")
    print(f"  Class 1 (Fraud):      {class_weights[1]:.4f}")
    
    return class_weights


def preprocess_data(df, use_smote=False, use_undersampling=False, 
                   test_size=0.2, random_state=42, target_col='Class'):
    """
    Complete preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Input dataset
        use_smote (bool): Whether to use SMOTE for balancing
        use_undersampling (bool): Whether to use undersampling
        test_size (float): Test set proportion
        random_state (int): Random seed
        target_col (str): Name of target column
        
    Returns:
        dict: Dictionary containing all preprocessed data and objects
    """
    print("\nStarting Data Preprocessing Pipeline")
    
    df_scaled, scaler = scale_features(df, features_to_scale=['Amount'])
    
    X, y = split_features_target(df_scaled, target_col=target_col)
    
    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train_balanced = X_train.copy()
    y_train_balanced = y_train.copy()
    class_weights = None
    
    if use_smote:
        X_train_balanced, y_train_balanced = handle_imbalance_with_smote(
            X_train, y_train, random_state=random_state
        )
    elif use_undersampling:
        X_train_balanced, y_train_balanced = undersample_majority_class(
            X_train, y_train, random_state=random_state
        )
    else:
        class_weights = get_class_weights(y_train)
        print("Using class weights instead of resampling")
    
    print("\nData preprocessing complete.")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_balanced': X_train_balanced,
        'y_train_balanced': y_train_balanced,
        'scaler': scaler,
        'class_weights': class_weights,
        'feature_names': X.columns.tolist()
    }


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from src.data_loader import load_dataset
    
    df = load_dataset("../data/creditcard.csv")
    
    # Preprocess without resampling (using class weights)
    preprocessed_data = preprocess_data(df, use_smote=False, use_undersampling=False)
    
    # Or preprocess with SMOTE
    # preprocessed_data = preprocess_data(df, use_smote=True)
