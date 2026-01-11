"""
Model Evaluation Module
Model Evaluation, Comparison, and Threshold Tuning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix for a model
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    
    # Add percentage annotations
    cm_percent = cm / cm.sum() * 100
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_classification_metrics(y_true, y_pred, model_name):
    """
    Print detailed classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    print(f"\nEvaluation Metrics: {model_name}")
    print("="*60)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {tn:,} - Correctly identified legitimate")
    print(f"  False Positives (FP): {fp:,} - Legitimate flagged as fraud")
    print(f"  False Negatives (FN): {fn:,} - Fraud missed (CRITICAL)")
    print(f"  True Positives (TP):  {tp:,} - Correctly identified fraud")
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nKey Performance Metrics:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"    Of all predicted frauds, {precision*100:.1f}% are actually fraud")
    
    print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"    Of all actual frauds, {recall*100:.1f}% are detected")
    
    print(f"  F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"    Harmonic mean of precision and recall")
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Legitimate', 'Fraud'],
                                digits=4))


def calculate_roc_auc(y_true, y_proba, model_name):
    """
    Calculate and print ROC-AUC score.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        model_name: Name of the model
        
    Returns:
        float: ROC-AUC score
    """
    if y_proba is None:
        print(f"\n{model_name}: Probabilities not available for ROC-AUC calculation")
        return None
    
    if len(y_proba.shape) > 1:
        y_proba_fraud = y_proba[:, 1]
    else:
        y_proba_fraud = y_proba
    
    roc_auc = roc_auc_score(y_true, y_proba_fraud)
    
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    if roc_auc >= 0.95:
        print("  Excellent discrimination ability")
    elif roc_auc >= 0.90:
        print("  Very good discrimination ability")
    elif roc_auc >= 0.80:
        print("  Good discrimination ability")
    else:
        print("  Moderate discrimination ability")
    
    return roc_auc


def plot_roc_curve(y_true, y_proba, model_name, save_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if y_proba is None:
        print(f"Cannot plot ROC curve: probabilities not available")
        return
    
    if len(y_proba.shape) > 1:
        y_proba_fraud = y_proba[:, 1]
    else:
        y_proba_fraud = y_proba
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_fraud)
    roc_auc = roc_auc_score(y_true, y_proba_fraud)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr, tpr, color='#e74c3c', lw=2, 
             label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(y_true, y_proba, model_name, save_path=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save the plot
    """
    if y_proba is None:
        print(f"Cannot plot PR curve: probabilities not available")
        return
    
    if len(y_proba.shape) > 1:
        y_proba_fraud = y_proba[:, 1]
    else:
        y_proba_fraud = y_proba
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba_fraud)
    avg_precision = average_precision_score(y_true, y_proba_fraud)
    
    plt.figure(figsize=(10, 8))
    
    plt.plot(recall, precision, color='#3498db', lw=2,
             label=f'{model_name} (AP = {avg_precision:.4f})')
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def evaluate_model(y_true, y_pred, y_proba, model_name):
    """
    Complete evaluation for a single model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        model_name: Name of the model
    """
    print(f"\nEvaluating Model: {model_name}")
    print("="*70)
    
    print_classification_metrics(y_true, y_pred, model_name)
    
    roc_auc = calculate_roc_auc(y_true, y_proba, model_name)
    
    plot_confusion_matrix(y_true, y_pred, model_name)
    
    if y_proba is not None:
        plot_roc_curve(y_true, y_proba, model_name)
        plot_precision_recall_curve(y_true, y_proba, model_name)
    
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc
    }


def compare_models(y_true, predictions, probabilities, models_dict):
    """
    Compare multiple models and select the best one.
    
    Args:
        y_true: True labels
        predictions: Dictionary of predictions from each model
        probabilities: Dictionary of probabilities from each model
        models_dict: Dictionary of trained models
    """
    print("\nModel Comparison")
    print("="*70)
    
    results = []
    
    for model_name in predictions.keys():
        y_pred = predictions[model_name]
        y_proba = probabilities[model_name]
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if y_proba is not None and len(y_proba.shape) > 1:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            roc_auc = None
        
        training_time = models_dict[model_name].training_time
        
        results.append({
            'Model': model_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Training Time (s)': training_time
        })
    
    comparison_df = pd.DataFrame(results)
    
    print("\nPerformance Comparison Table:")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    print("\nRecommendations:")
    print("="*60)
    
    best_recall_idx = comparison_df['Recall'].idxmax()
    best_recall_model = comparison_df.loc[best_recall_idx, 'Model']
    best_recall_score = comparison_df.loc[best_recall_idx, 'Recall']
    print(f"\nBest for Fraud Detection (Recall):")
    print(f"  Model: {best_recall_model}")
    print(f"  Recall: {best_recall_score:.4f} ({best_recall_score*100:.2f}%)")
    
    best_f1_idx = comparison_df['F1-Score'].idxmax()
    best_f1_model = comparison_df.loc[best_f1_idx, 'Model']
    best_f1_score = comparison_df.loc[best_f1_idx, 'F1-Score']
    print(f"\nBest for Balance (F1-Score):")
    print(f"  Model: {best_f1_model}")
    print(f"  F1-Score: {best_f1_score:.4f} ({best_f1_score*100:.2f}%)")
    
    if comparison_df['ROC-AUC'].notna().any():
        best_auc_idx = comparison_df['ROC-AUC'].idxmax()
        best_auc_model = comparison_df.loc[best_auc_idx, 'Model']
        best_auc_score = comparison_df.loc[best_auc_idx, 'ROC-AUC']
        print(f"\nBest Overall Performance (ROC-AUC):")
        print(f"  Model: {best_auc_model}")
        print(f"  ROC-AUC: {best_auc_score:.4f}")
    
    plot_model_comparison(comparison_df)
    
    return comparison_df


def plot_model_comparison(comparison_df):
    """
    Visualize model comparison
    
    Args:
        comparison_df: DataFrame with model comparison results
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    models = comparison_df['Model']
    metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if comparison_df[metric].notna().any():
            values = comparison_df[metric].values
            bars = ax.bar(models, values, color=colors[idx], alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, f'{metric}\nNot Available', 
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()


def tune_threshold(y_true, y_proba, model_name, thresholds=None):
    """
    Tune classification threshold for optimal fraud detection
    
    Args:
        y_true: True labels
        y_proba: Prediction probabilities
        model_name: Name of the model
        thresholds: List of thresholds to try
        
    Returns:
        DataFrame with results for each threshold
    """
    if y_proba is None:
        print(f"⚠️  Threshold tuning not available: probabilities required")
        return None
    
    print("\n" + "="*70)
    print(f"🎚️  THRESHOLD TUNING: {model_name}")
    print("="*70)
    
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Get fraud probabilities
    if len(y_proba.shape) > 1:
        y_proba_fraud = y_proba[:, 1]
    else:
        y_proba_fraud = y_proba
    
    results = []
    
    for threshold in thresholds:
        # Make predictions with custom threshold
        y_pred_custom = (y_proba_fraud >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred_custom, zero_division=0)
        recall = recall_score(y_true, y_pred_custom, zero_division=0)
        f1 = f1_score(y_true, y_pred_custom, zero_division=0)
        
        results.append({
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n📊 Threshold Tuning Results:")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    # Plot threshold effects
    plt.figure(figsize=(12, 6))
    
    plt.plot(results_df['Threshold'], results_df['Precision'], 
             marker='o', label='Precision', linewidth=2)
    plt.plot(results_df['Threshold'], results_df['Recall'], 
             marker='s', label='Recall', linewidth=2)
    plt.plot(results_df['Threshold'], results_df['F1-Score'], 
             marker='^', label='F1-Score', linewidth=2)
    
    plt.xlabel('Classification Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'Threshold Tuning - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(thresholds)
    
    plt.tight_layout()
    plt.show()
    
    # Recommend optimal threshold
    best_f1_idx = results_df['F1-Score'].idxmax()
    best_threshold = results_df.loc[best_f1_idx, 'Threshold']
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Optimal Threshold: {best_threshold}")
    print(f"   → Maximizes F1-Score")
    print(f"   → Recall: {results_df.loc[best_f1_idx, 'Recall']:.4f}")
    print(f"   → Precision: {results_df.loc[best_f1_idx, 'Precision']:.4f}")
    
    return results_df


if __name__ == "__main__":
    # Example usage
    print("This module provides comprehensive model evaluation functions")
    print("Import and use with trained models and predictions")
