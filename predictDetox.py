import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# === Load BERT model ===
MODEL_DIR = "bert_clickbait_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Predict BERT clickbait probability in batches ===
def batch_predict_probs(headlines, batch_size=32):
    clickbait_probs = []
    for i in tqdm(range(0, len(headlines), batch_size), desc="Predicting"):
        batch = headlines[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            clickbait_probs.extend(probs[:, 1].tolist())  # class 1 = clickbait
    return clickbait_probs

# === Evaluate detoxified headlines with varying thresholds ===
def evaluate_detoxified_dataset(input_file, output_prefix):
    output_folder = "detox_threshold_results"
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_file)
    if "detoxified_headline" not in df.columns:
        raise ValueError("Missing 'detoxified_headline' column.")

    headlines = df["detoxified_headline"].fillna("").astype(str).tolist()
    clickbait_probs = batch_predict_probs(headlines)

    thresholds = [i / 100 for i in range(10, 91, 5)]
    clickbait_pcts = []

    print(f"\nDetox Evaluation for {input_file}")
    print("Threshold | Clickbait % | Non-Clickbait %")
    print("-----------------------------------------")

    for threshold in thresholds:
        preds = [1 if prob > threshold else 0 for prob in clickbait_probs]
        clickbait_pct = sum(preds) / len(preds) * 100
        non_clickbait_pct = 100 - clickbait_pct
        clickbait_pcts.append(clickbait_pct)

        print(f"{threshold:.2f}      | {clickbait_pct:10.2f}% | {non_clickbait_pct:15.2f}%")

        output_df = pd.DataFrame({
            "detoxified_headline": headlines,
            "confidence": clickbait_probs,
            "predicted_label": preds
        })
        output_path = os.path.join(output_folder, f"{output_prefix}_threshold_{threshold:.2f}.csv")
        output_df.to_csv(output_path, index=False)

    # === Plot results ===
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, clickbait_pcts, marker='o', label="Clickbait %")
    plt.xlabel("Threshold")
    plt.ylabel("Clickbait Percentage")
    plt.title(f"Threshold vs Clickbait % ({output_prefix})")
    plt.grid(True)
    plt.legend()
    plt.show()

# === Run on both detox output datasets ===
if __name__ == "__main__":
    evaluate_detoxified_dataset("new_dataset_results.csv", "new_dataset")
    evaluate_detoxified_dataset("kaggle_dataset_results.csv", "kaggle_dataset")
