"""Credit Card Fraud Detection Pipeline"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_dataset, get_dataset_summary
from src.eda import perform_complete_eda
from src.preprocessing import preprocess_data
from src.models import build_all_models, make_predictions
from src.evaluation import evaluate_model, compare_models, tune_threshold
from src.utils import (
    create_results_directory, save_model, save_scaler, 
    save_results, check_dataset_exists, format_time
)


def main():
    """Main pipeline for Credit Card Fraud Detection"""
    print("\nCredit Card Fraud Detection Pipeline")
    print("="*70)
    
    start_time = time.time()
    
    DATA_PATH = "data/creditcard.csv"
    USE_SMOTE = False
    USE_UNDERSAMPLING = False
    USE_ISOLATION_FOREST = False
    PERFORM_EDA = True
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    print("\nStep 1: Checking dataset...")
    if not check_dataset_exists(DATA_PATH):
        print("Error: Dataset not found")
        return
    
    print("\nStep 2: Loading and understanding data...")
    df = get_dataset_summary(DATA_PATH)
    
    if PERFORM_EDA:
        print("\nStep 3: Performing exploratory data analysis...")
        perform_complete_eda(df)
    else:
        print("\nSkipping EDA")
    
    print("\nStep 4: Preprocessing data...")
    preprocessed_data = preprocess_data(
        df,
        use_smote=USE_SMOTE,
        use_undersampling=USE_UNDERSAMPLING,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    print("\nStep 5: Building models...")
    if USE_SMOTE or USE_UNDERSAMPLING:
        X_train = preprocessed_data['X_train_balanced']
        y_train = preprocessed_data['y_train_balanced']
        print("Using balanced training data")
    else:
        X_train = preprocessed_data['X_train']
        y_train = preprocessed_data['y_train']
        print("Using original training data with class weights")
    
    models = build_all_models(
        X_train,
        y_train,
        class_weights=preprocessed_data['class_weights'],
        use_isolation_forest=USE_ISOLATION_FOREST
    )
    
    print("\nStep 6: Making predictions...")
    predictions, probabilities = make_predictions(models, preprocessed_data['X_test'])
    
    print("\nStep 7: Evaluating models...")
    evaluation_results = {}
    
    for model_name in models.keys():
        print(f"\nEvaluating: {model_name}")
        metrics = evaluate_model(
            preprocessed_data['y_test'],
            predictions[model_name],
            probabilities[model_name],
            model_name
        )
        evaluation_results[model_name] = metrics
    
    print("\nStep 8: Comparing models...")
    comparison_results = compare_models(
        preprocessed_data['y_test'],
        predictions,
        probabilities,
        models
    )
    
    print("\nStep 9: Threshold tuning...")
    best_model_name = comparison_results.loc[comparison_results['F1-Score'].idxmax(), 'Model']
    
    if probabilities[best_model_name] is not None:
        print(f"Tuning threshold for: {best_model_name}")
        threshold_results = tune_threshold(
            preprocessed_data['y_test'],
            probabilities[best_model_name],
            best_model_name
        )
    else:
        print(f"Threshold tuning not available for {best_model_name}")
    
    print("\nStep 10: Saving results...")
    results_dir = create_results_directory()
    
    print(f"Saving best model: {best_model_name}")
    save_model(models[best_model_name], best_model_name, results_dir)
    
    if preprocessed_data['scaler'] is not None:
        print("Saving scaler...")
        save_scaler(preprocessed_data['scaler'], results_dir)
    
    print("Saving comparison results...")
    comparison_results.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*70)
    print("Pipeline completed successfully")
    print("="*70)
    print(f"\nTotal execution time: {format_time(total_time)}")
    print(f"\nBest Model: {best_model_name}")
    best_idx = comparison_results['Model'] == best_model_name
    print(f"Precision: {comparison_results.loc[best_idx, 'Precision'].values[0]:.4f}")
    print(f"Recall:    {comparison_results.loc[best_idx, 'Recall'].values[0]:.4f}")
    print(f"F1-Score:  {comparison_results.loc[best_idx, 'F1-Score'].values[0]:.4f}")
    
    if comparison_results.loc[best_idx, 'ROC-AUC'].notna().any():
        print(f"ROC-AUC:   {comparison_results.loc[best_idx, 'ROC-AUC'].values[0]:.4f}")
    
    print(f"\nResults saved in: {results_dir}/\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
