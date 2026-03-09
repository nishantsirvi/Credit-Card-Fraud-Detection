"""
Utility Functions Module
Helper functions for the fraud detection project
"""

import os
import pickle
import json
from datetime import datetime


def create_results_directory():
    """Create directory to store results"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    return results_dir


def save_model(model, model_name, results_dir="results"):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model object
        model_name: Name of the model
        results_dir: Directory to save the model
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.replace(' ', '_')}_{timestamp}.pkl"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved: {filepath}")
    return filepath


def load_model(filepath):
    """Load trained model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded: {filepath}")
    return model


def save_scaler(scaler, results_dir="results"):
    """Save scaler object to disk"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scaler_{timestamp}.pkl"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Scaler saved: {filepath}")
    return filepath


def load_scaler(filepath):
    """Load scaler object from disk"""
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Scaler loaded: {filepath}")
    return scaler


def save_results(results_dict, filename, results_dir="results"):
    """Save results dictionary to JSON file"""
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"Results saved: {filepath}")
    return filepath


def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)


def print_subsection(title):
    print(f"\n{'─'*60}")
    print(f"{title}")
    print(f"{'─'*60}")


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def get_data_summary(df):
    """
    Get quick summary of dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Summary statistics
    """
    summary = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    return summary


def check_dataset_exists(file_path):
    """Check if dataset file exists"""
    if not os.path.exists(file_path):
        print(f"\nERROR: Dataset not found at {file_path}")
        print(f"\nPlease download the dataset from:")
        print(f"  https://www.kaggle.com/mlg-ulb/creditcardfraud")
        print(f"\nAnd place it at: {file_path}")
        return False
    return True


if __name__ == "__main__":
    # Example usage
    results_dir = create_results_directory()
    print(f"Results directory ready: {results_dir}")
