import os
import sys
import json
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def run_evaluation_comparison():
    """
    Coordinates overall evaluation and builds a comparative analysis summary table
    comparing Random Forest, SVM, XGBoost, ResNet50, EfficientNet, and QuantumNow Optimized model.
    """
    print("\n==============================================================")
    print("      Quantum-Enhanced Cancer Detection Model Comparison      ")
    print("==============================================================\n")
    
    # 0. Ensure Quality Checks and Split CSV exist before continuing
    if not os.path.exists(config.QUALITY_REPORT_JSON) or not os.path.exists(config.CLEANING_REPORT_JSON):
        print("Quality and cleaning reports not found. Running quality assessment...")
        from utils.quality_check import run_quality_assessment, analyze_mixed_class
        run_quality_assessment()
        analyze_mixed_class()
        
    if not os.path.exists(config.SPLIT_CSV):
        print("Split CSV not found. Generating train/val/test splits...")
        from utils.dataset_split import perform_dataset_split
        perform_dataset_split()
        
    # Check and run baseline script if metrics not present
    baseline_path = os.path.join(config.OUTPUT_REPORTS, "baseline_metrics.json")
    if not os.path.exists(baseline_path):
        print("Classical baseline metrics not found. Running baseline training...")
        from train_baseline import train_and_eval_baselines
        baseline_metrics = train_and_eval_baselines()
    else:
        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)
            
    # Check and run deep learning training if metrics not present
    deep_path = os.path.join(config.OUTPUT_REPORTS, "deep_learning_metrics.json")
    if not os.path.exists(deep_path):
        print("Deep learning metrics not found. Running deep learning training...")
        from train_deep_model import run_deep_training
        deep_metrics = run_deep_training()
    else:
        with open(deep_path, 'r') as f:
            deep_metrics = json.load(f)
            
    # Check and run quantum pipeline if metrics not present
    quantum_path = os.path.join(config.OUTPUT_REPORTS, "quantum_final_metrics.json")
    if not os.path.exists(quantum_path):
        print("Quantum optimization metrics not found. Running quantum optimization...")
        from train_quantum import run_quantum_pipeline
        quantum_metrics = run_quantum_pipeline()
    else:
        with open(quantum_path, 'r') as f:
            quantum_metrics = json.load(f)
            
    # Combine all results into a single dictionary
    # The models are: Random Forest, SVM, XGBoost, ResNet50, EfficientNet, QuantumNow Optimized Model
    comparison_table = {}
    
    # 1. Classical baselines
    for model_name in ["Random Forest", "SVM", "XGBoost"]:
        if model_name in baseline_metrics:
            comparison_table[model_name] = baseline_metrics[model_name]
            
    # 2. Deep learning
    for model_name in ["ResNet50", "EfficientNet", "Hybrid Ensemble"]:
        if model_name in deep_metrics:
            comparison_table[model_name] = deep_metrics[model_name]
            
    # 3. Quantum optimized
    q_key = "QuantumNow Optimized Model"
    if q_key in quantum_metrics:
        comparison_table[q_key] = quantum_metrics[q_key]
        
    # Save combined results
    out_comparison = os.path.join(config.OUTPUT_REPORTS, "comparative_metrics.json")
    with open(out_comparison, 'w') as f:
        json.dump(comparison_table, f, indent=4)
        
    # Display comparison table as a pandas DataFrame for readable output
    df = pd.DataFrame.from_dict(comparison_table, orient='index')
    
    # Rename columns for clarity in printing
    df_display = df.rename(columns={
        "accuracy": "Accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1": "F1 Score",
        "auc": "AUC ROC"
    })
    
    # Print the table nicely formatted
    print("\nMaster Comparative Performance Summary:")
    print(df_display.to_markdown(floatfmt=".4f"))
    print("\nResults successfully saved to:", out_comparison)
    
    return comparison_table

if __name__ == "__main__":
    # Run evaluation
    run_evaluation_comparison()
