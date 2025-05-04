import pandas as pd
import argparse
from sklearn.metrics import classification_report

# Parse input filename from command line
parser = argparse.ArgumentParser(description="Evaluate or analyze predictions CSV file.")
parser.add_argument("--input_csv", type=str, required=True, help="Path to CSV file containing 'prediction' column, optionally 'clickbait' labels.")
args = parser.parse_args()

# Load the predictions
df = pd.read_csv(args.input_csv)

print("Evaluating:", args.input_csv)
print("======================================")

if "true_label" in df.columns and "predicted_label" in df.columns:
    # Case 1: Ground truth labels are available
    print("Detected ground truth labels. Running full evaluation...\n")
    report = classification_report(df["true_label"], df["predicted_label"], target_names=["Non-Clickbait", "Clickbait"])
    print("Evaluation Metrics:\n")
    print(report)
else:
    # Case 2: No ground truth — just count clickbait vs. non-clickbait
    if "prediction" not in df.columns:
        raise ValueError("CSV must contain a 'prediction' column.")
    
    clickbait_count = (df["prediction"] == 1).sum()
    non_clickbait_count = (df["prediction"] == 0).sum()
    total = len(df)
    clickbait_percent = clickbait_count / total * 100
    non_clickbait_percent = non_clickbait_count / total * 100

    print("No ground truth labels found. Performing detox analysis...\n")
    print(f"Total headlines evaluated: {total}")
    print(f"Still classified as clickbait: {clickbait_count} ({clickbait_percent:.1f}%)")
    print(f"Classified as non-clickbait: {non_clickbait_count} ({non_clickbait_percent:.1f}%)")
