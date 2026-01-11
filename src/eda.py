"""
Exploratory Data Analysis (EDA) Module
EDA - Visualizations and Pattern Discovery
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def set_plot_style():
    """Set consistent plot style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_class_distribution(df, target_col='Class', save_path=None):
    """
    Plot the distribution of fraud vs legitimate transactions
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        save_path (str): Path to save the plot
    """
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count plot
    class_counts = df[target_col].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0].bar(['Legitimate (0)', 'Fraud (1)'], class_counts.values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, class_counts.max() * 1.1)
    
    # Add value labels
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5000, f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Percentage plot
    class_percentages = df[target_col].value_counts(normalize=True) * 100
    axes[1].bar(['Legitimate (0)', 'Fraud (1)'], class_percentages.values, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 105)
    
    # Add percentage labels
    for i, v in enumerate(class_percentages.values):
        axes[1].text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("Class Distribution Plot Generated")


def plot_transaction_amounts(df, target_col='Class', save_path=None):
    """
    Compare transaction amounts between fraud and legitimate transactions
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        save_path (str): Path to save the plot
    """
    set_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Split data
    legit = df[df[target_col] == 0]['Amount']
    fraud = df[df[target_col] == 1]['Amount']
    
    # 1. Box plot comparison
    data_to_plot = [legit, fraud]
    box = axes[0, 0].boxplot(data_to_plot, labels=['Legitimate', 'Fraud'], patch_artist=True,
                              medianprops=dict(color='red', linewidth=2))
    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0, 0].set_ylabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Transaction Amount Comparison (Box Plot)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Histogram comparison
    axes[0, 1].hist(legit, bins=50, alpha=0.6, label='Legitimate', color='#2ecc71', edgecolor='black')
    axes[0, 1].hist(fraud, bins=50, alpha=0.6, label='Fraud', color='#e74c3c', edgecolor='black')
    axes[0, 1].set_xlabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Transaction Amount Distribution', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].set_xlim(0, 1000)  # Focus on majority of data
    
    # 3. Statistics comparison
    stats_data = {
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
        'Legitimate': [f'${legit.mean():.2f}', f'${legit.median():.2f}', 
                      f'${legit.std():.2f}', f'${legit.min():.2f}', f'${legit.max():.2f}'],
        'Fraud': [f'${fraud.mean():.2f}', f'${fraud.median():.2f}', 
                 f'${fraud.std():.2f}', f'${fraud.min():.2f}', f'${fraud.max():.2f}']
    }
    
    axes[1, 0].axis('tight')
    axes[1, 0].axis('off')
    table = axes[1, 0].table(cellText=[[stats_data['Metric'][i], 
                                       stats_data['Legitimate'][i], 
                                       stats_data['Fraud'][i]] for i in range(5)],
                            colLabels=['Metric', 'Legitimate', 'Fraud'],
                            cellLoc='center',
                            loc='center',
                            colColours=['#f0f0f0', '#2ecc71', '#e74c3c'])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 0].set_title('Transaction Amount Statistics', fontsize=13, fontweight='bold', pad=20)
    
    # 4. Violin plot
    plot_data = pd.DataFrame({
        'Amount': list(legit[:5000]) + list(fraud),  # Sample legit for visibility
        'Class': ['Legitimate']*5000 + ['Fraud']*len(fraud)
    })
    
    sns.violinplot(data=plot_data, x='Class', y='Amount', ax=axes[1, 1], palette=['#2ecc71', '#e74c3c'])
    axes[1, 1].set_ylabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('')
    axes[1, 1].set_title('Transaction Amount Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim(0, 500)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("Transaction Amount Analysis Complete")


def plot_time_distribution(df, target_col='Class', save_path=None):
    """
    Analyze transaction timing patterns
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        save_path (str): Path to save the plot
    """
    if 'Time' not in df.columns:
        print("Time column not found in dataset")
        return
    
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Convert time to hours
    df_temp = df.copy()
    df_temp['Hour'] = (df_temp['Time'] / 3600) % 24
    
    # 1. Fraud vs Legitimate over time
    legit_time = df_temp[df_temp[target_col] == 0]['Hour']
    fraud_time = df_temp[df_temp[target_col] == 1]['Hour']
    
    axes[0].hist(legit_time, bins=24, alpha=0.6, label='Legitimate', color='#2ecc71', edgecolor='black')
    axes[0].hist(fraud_time, bins=24, alpha=0.6, label='Fraud', color='#e74c3c', edgecolor='black')
    axes[0].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Number of Transactions', fontsize=11, fontweight='bold')
    axes[0].set_title('Transaction Distribution by Hour', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Fraud rate by hour
    fraud_rate_by_hour = df_temp.groupby(df_temp['Hour'].astype(int))[target_col].mean() * 100
    
    axes[1].bar(fraud_rate_by_hour.index, fraud_rate_by_hour.values, 
                color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Fraud Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Fraud Rate by Hour of Day', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("Time Distribution Analysis Complete")


def plot_correlation_heatmap(df, target_col='Class', save_path=None):
    """
    Plot correlation heatmap for key features
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
        save_path (str): Path to save the plot
    """
    set_plot_style()
    
    # Select features to analyze (Amount + some V features + Class)
    features_to_plot = ['Amount'] + [f'V{i}' for i in [1, 2, 3, 4, 9, 10, 11, 12, 14, 16, 17, 18]] + [target_col]
    features_to_plot = [f for f in features_to_plot if f in df.columns]
    
    # Calculate correlation
    correlation_matrix = df[features_to_plot].corr()
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap (Selected Features)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("Correlation Heatmap Generated")


def analyze_feature_importance_preview(df, target_col='Class'):
    """
    Quick analysis of which features might be important.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
    """
    print("\nFeature Correlation with Target")
    print("="*60)
    
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    
    correlations = correlations[correlations.index != target_col]
    
    print("\nTop 10 Features Correlated with Fraud:")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feature:15s} : {corr:.4f}")


def perform_complete_eda(df, target_col='Class'):
    """
    Perform complete exploratory data analysis.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_col (str): Name of target column
    """
    print("\nStarting Exploratory Data Analysis")
    print("="*60)
    
    plot_class_distribution(df, target_col)
    
    plot_transaction_amounts(df, target_col)
    
    plot_time_distribution(df, target_col)
    
    analyze_feature_importance_preview(df, target_col)
    
    plot_correlation_heatmap(df, target_col)
    
    print("\nExploratory Data Analysis Complete")
    print("="*60)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from src.data_loader import load_dataset
    
    df = load_dataset("../data/creditcard.csv")
    perform_complete_eda(df)
