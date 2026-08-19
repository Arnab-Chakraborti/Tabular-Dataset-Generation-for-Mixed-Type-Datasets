import os
import pandas as pd
from src.Evaluations.eval_metrics import evaluate_generator_performance

def main():
    # Resolve absolute paths based on root directory location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base_dir, "data", "test.csv")
    synthetic_path = os.path.join(base_dir, "data", "synthetic_data_export.csv")

    print(f"Loading real test dataset from: {test_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"❌ Test dataset not found at: {test_path}")

    print(f"Loading synthetic dataset from: {synthetic_path}")
    if not os.path.exists(synthetic_path):
        raise FileNotFoundError(f"❌ Synthetic export not found at: {synthetic_path}. Run training first!")

    # Load dataframes
    real_df = pd.read_csv(test_path, skipinitialspace=True)
    synthetic_df = pd.read_csv(synthetic_path, skipinitialspace=True)
    synthetic_df= synthetic_df[real_df.columns]

    print("\nRunning performance evaluation against test set...")
    
    # Evaluate performance using your project's metric engine
    gen_metrics = evaluate_generator_performance(
        real_df=real_df, 
        synthetic_df=synthetic_df, 
        k=5
    )

    # Print out clean results summary
    print("\n" + "="*40)
    print("      STANDALONE TEST EVALUATION RESULTS      ")
    print("="*40)
    print(f"   Shape Error     : {gen_metrics['shape_error_pct']:.2f}%")
    print(f"   Trend Error     : {gen_metrics['trend_error_pct']:.2f}%")
    print(f"   Alpha Precision : {gen_metrics['alpha_precision_pct']:.2f}%")
    print(f"   Beta Recall     : {gen_metrics['beta_recall_pct']:.2f}%")
    print(f"   MSE             : {gen_metrics['mse_pct']:.10f}")
    print("="*40)

if __name__ == "__main__":
    main()
