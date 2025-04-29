import pandas as pd
from sklearn.metrics import classification_report

# Load the predictions
df = pd.read_csv("predictions.csv")  # or whichever file you're evaluating

# Check if both 'clickbait' and 'prediction' columns exist
if 'clickbait' not in df.columns or 'prediction' not in df.columns:
    raise ValueError("CSV must contain 'clickbait' (ground truth) and 'prediction' columns.")

# Print evaluation metrics
report = classification_report(df['clickbait'], df['prediction'], target_names=["Non-Clickbait", "Clickbait"])
print("Evaluation Metrics:\n")
print(report)

# Optional: Save the report to a text file
# with open("evaluation_report.txt", "w") as f:
#     f.write("Evaluation Metrics:\n\n")
#     f.write(report)
