"""
Machine Learning Models Module
Model Building - Logistic Regression, Random Forest, Isolation Forest
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import time


class FraudDetectionModel:
    """Base class for fraud detection models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.training_time = 0
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the model"""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError(f"{self.model_name} does not support probability predictions")
    
    def get_model_info(self):
        """Get model information"""
        return {
            'name': self.model_name,
            'trained': self.is_trained,
            'training_time': self.training_time
        }


class LogisticRegressionModel(FraudDetectionModel):
    """
    Logistic Regression for Fraud Detection.
    Fast training, interpretable coefficients, good baseline model.
    """
    
    def __init__(self, class_weights=None, max_iter=1000):
        super().__init__("Logistic Regression")
        self.class_weights = class_weights
        self.max_iter = max_iter
    
    def train(self, X_train, y_train):
        """Train Logistic Regression model"""
        print(f"\nTraining {self.model_name}...")
        print(f"Config: max_iter={self.max_iter}, class_weights={'Balanced' if self.class_weights else 'None'}")
        
        start_time = time.time()
        
        self.model = LogisticRegression(
            class_weight=self.class_weights,
            max_iter=self.max_iter,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        if hasattr(X_train, 'columns'):
            self._display_feature_importance(X_train.columns)
        
        return self
    
    def _display_feature_importance(self, feature_names, top_n=10):
        """Display top important features"""
        coefficients = self.model.coef_[0]
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(f"\nTop {top_n} Important Features:")
        for i, row in importance.head(top_n).iterrows():
            print(f"  {row['Feature']:10s}: {row['Coefficient']:+.4f}")


class RandomForestModel(FraudDetectionModel):
    """
    Random Forest for Fraud Detection.
    Handles non-linear relationships, provides feature importance, robust to outliers.
    """
    
    def __init__(self, n_estimators=100, class_weights=None, max_depth=None):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.class_weights = class_weights
        self.max_depth = max_depth
    
    def train(self, X_train, y_train):
        """Train Random Forest model"""
        print(f"\nTraining {self.model_name}...")
        print(f"Config: n_estimators={self.n_estimators}, max_depth={self.max_depth}, class_weights={'Balanced' if self.class_weights else 'None'}")
        
        start_time = time.time()
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight=self.class_weights,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        if hasattr(X_train, 'columns'):
            self._display_feature_importance(X_train.columns)
        
        return self
    
    def _display_feature_importance(self, feature_names, top_n=10):
        """Display top important features"""
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_n} Important Features:")
        for i, row in importance.head(top_n).iterrows():
            print(f"  {row['Feature']:10s}: {row['Importance']:.4f}")


class IsolationForestModel(FraudDetectionModel):
    """
    Isolation Forest for Fraud Detection (Anomaly Detection).
    Unsupervised learning for outlier detection in high-dimensional data.
    """
    
    def __init__(self, contamination=0.001, n_estimators=100):
        super().__init__("Isolation Forest")
        self.contamination = contamination
        self.n_estimators = n_estimators
    
    def train(self, X_train, y_train=None):
        """Train Isolation Forest model (unsupervised)"""
        print(f"\nTraining {self.model_name}...")
        print(f"Config: n_estimators={self.n_estimators}, contamination={self.contamination:.4f}")
        
        start_time = time.time()
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"Training completed in {self.training_time:.2f} seconds")
        
        return self
    
    def predict(self, X):
        """Make predictions (converts -1/1 to 0/1)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Isolation Forest returns -1 for outliers, 1 for inliers
        # Convert to 0 (legitimate) and 1 (fraud)
        predictions = self.model.predict(X)
        return np.where(predictions == -1, 1, 0)
    
    def predict_proba(self, X):
        """Get anomaly scores as probabilities"""
        # Note: This is an approximation
        # Anomaly scores are converted to probability-like scores
        raise NotImplementedError("Isolation Forest uses anomaly scores, not probabilities")


def build_all_models(X_train, y_train, class_weights=None, use_isolation_forest=False):
    """
    Build and train all fraud detection models.
    
    Args:
        X_train: Training features
        y_train: Training target
        class_weights: Class weights for handling imbalance
        use_isolation_forest: Whether to include Isolation Forest
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\nBuilding fraud detection models...")
    
    models = {}
    
    print("\n1. Building Logistic Regression Model...")
    lr_model = LogisticRegressionModel(class_weights=class_weights)
    lr_model.train(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    print("\n2. Building Random Forest Model...")
    rf_model = RandomForestModel(n_estimators=100, class_weights=class_weights)  # 100 trees seemed like good balance between speed and performance
    rf_model.train(X_train, y_train)
    models['Random Forest'] = rf_model
    
    if use_isolation_forest:
        print("\n3. Building Isolation Forest Model...")
        contamination = y_train.sum() / len(y_train)
        if_model = IsolationForestModel(contamination=contamination)
        if_model.train(X_train, y_train)
        models['Isolation Forest'] = if_model
    
    print("\nAll models trained successfully!")
    print("\nTraining Summary:")
    for name, model in models.items():
        info = model.get_model_info()
        print(f"  {name:20s}: {info['training_time']:.2f}s")
    
    return models


def make_predictions(models, X_test):
    """
    Make predictions using all trained models.
    
    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        
    Returns:
        dict: Dictionary of predictions for each model
    """
    print("\nMaking predictions...")
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        print(f"\nPredicting with {name}...")
        
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        try:
            y_proba = model.predict_proba(X_test)
            probabilities[name] = y_proba
            print(f"  Predictions: {len(y_pred):,} samples (probabilities available)")
        except (ValueError, NotImplementedError):
            probabilities[name] = None
            print(f"  Predictions: {len(y_pred):,} samples (probabilities not available)")
    
    print("\nPredictions complete for all models.")
    
    return predictions, probabilities


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from src.data_loader import load_dataset
    from src.preprocessing import preprocess_data
    
    # Load and preprocess data
    df = load_dataset("../data/creditcard.csv")
    preprocessed = preprocess_data(df)
    
    # Build models
    models = build_all_models(
        preprocessed['X_train_balanced'],
        preprocessed['y_train_balanced'],
        class_weights=preprocessed['class_weights']
    )
    
    # Make predictions
    predictions, probabilities = make_predictions(models, preprocessed['X_test'])
