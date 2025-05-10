import argparse
import pandas as pd
from sklearn.metrics import classification_report

def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument("--input_csv", required=True, help="Path to the CSV file containing predictions.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    print(f"Loaded CSV with columns: {list(df.columns)}\n")

    # Case 1: Evaluate with ground truth if available
    if "true_label" in df.columns and "predicted_label" in df.columns:
        true_vals = df["true_label"]
        pred_vals = df["predicted_label"]

        print("Detected ground truth labels. Running full evaluation...\n")

        # Try label type inference
        labels = sorted(set(true_vals) | set(pred_vals))
        if labels == [0, 1]:
            target_names = ["Non-Clickbait", "Clickbait"]
        elif sorted(labels) == ["Clickbait", "Non-Clickbait"]:
            target_names = labels
        else:
            target_names = [str(l) for l in labels]

        print(classification_report(true_vals, pred_vals, target_names=target_names))

    # Case 2: Only predictions available — report simple stats
    elif "prediction" in df.columns:
        print("No ground truth found. Showing prediction counts only...\n")
        print(df["prediction"].value_counts())

    else:
        print("Could not find required columns: 'true_label' & 'predicted_label' or 'prediction'.")
        print("Please ensure your CSV includes at least one of those column sets.")

if __name__ == "__main__":
    main()
