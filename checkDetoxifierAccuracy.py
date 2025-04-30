import pandas as pd

# Load dataset with existing BERT predictions
df = pd.read_csv("kaggle_dataset_results.csv")

# Ensure column exists
if "detox_predicted_label" not in df.columns:
    raise ValueError("Column 'detox_predicted_label' not found in the dataset.")

# Count labels
num_total = len(df)
num_clickbait = (df["detox_predicted_label"] == 1).sum()
num_non_clickbait = num_total - num_clickbait
pct_clickbait = num_clickbait / num_total * 100
pct_non_clickbait = 100 - pct_clickbait

# Print results
print(f"Clickbait: {num_clickbait} / {num_total} ({pct_clickbait:.2f}%)")
print(f"Non-clickbait: {num_non_clickbait} / {num_total} ({pct_non_clickbait:.2f}%)")
