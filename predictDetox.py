import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Same procedures as in predictClickbait, instead we're evaluating the "detox success rate" by re-running our BERT model on the detoxified headlines
MODEL_DIR = "bert_clickbait_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def batch_predict_probs(headlines, batch_size=32):
    clickbait_probs = []
    for i in tqdm(range(0, len(headlines), batch_size), desc="Predicting"):
        batch = headlines[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            clickbait_probs.extend(probs[:, 1].tolist())  
    return clickbait_probs

def evaluate_detoxified_dataset(input_file, output_prefix):
    output_folder = "detox_threshold_results"
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_file)
    if "detoxified_headline" not in df.columns:
        raise ValueError("Missing 'detoxified_headline' column.")
    if "clickbait" not in df.columns:
        raise ValueError("Missing 'clickbait' ground-truth label column.")

    headlines = df["detoxified_headline"].fillna("").astype(str).tolist()
    true_labels = df["clickbait"].tolist()
    clickbait_probs = batch_predict_probs(headlines)

    thresholds = [i / 100 for i in range(10, 91, 5)]
    clickbait_pcts = []
    precisions = []
    recalls = []
    f1_scores = []

    print(f"\nDetox Evaluation for {input_file}")
    print("Threshold | Clickbait % | Precision | Recall | F1 Score")
    print("---------------------------------------------------------")

    for threshold in thresholds:
        preds = [1 if prob > threshold else 0 for prob in clickbait_probs]
        clickbait_pct = sum(preds) / len(preds) * 100
        clickbait_pcts.append(clickbait_pct)

        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        print(f"{threshold:.2f}      | {clickbait_pct:10.2f}% | {prec:9.4f} | {rec:6.4f} | {f1:7.4f}")

        output_df = pd.DataFrame({
            "detoxified_headline": headlines,
            "confidence": clickbait_probs,
            "predicted_label": preds,
            "true_label": true_labels
        })
        output_path = os.path.join(output_folder, f"{output_prefix}_threshold_{threshold:.2f}.csv")
        output_df.to_csv(output_path, index=False)


    plt.figure(figsize=(8, 6))
       # Ultra-condensed Threshold vs Clickbait % table
    fig, ax = plt.subplots(figsize=(3.8, 0.6 + len(thresholds) * 0.33))  # even narrower
    ax.axis('off')

    col_labels = ["Threshold", "% Clickbait"]
    table_data = [
        [f"{t:.2f}", f"{cb:.1f}%"]
        for t, cb in zip(thresholds, clickbait_pcts)
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        colColours=["#40466e"] * 2,
        cellColours=[["#f1f1f2"] * 2 if i % 2 == 0 else ["#ffffff"] * 2 for i in range(len(table_data))]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.65, 1.1)  # max horizontal compression

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_text_props(color='#333333')

    plt.title(f"T vs %CB ({output_prefix})", fontsize=10, pad=8)
    plt.tight_layout()
    plt.show()



    plt.plot(thresholds, clickbait_pcts, marker='o', label="Clickbait %")
    plt.xlabel("Threshold")
    plt.ylabel("Clickbait Percentage")
    plt.title(f"Threshold vs Clickbait % ({output_prefix})")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores, marker='o', label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"Threshold vs F1 Score ({output_prefix})")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, marker='o', label="Precision")
    plt.plot(thresholds, recalls, marker='o', label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold vs Precision and Recall ({output_prefix})")
    plt.grid(True)
    plt.legend()
    plt.show()

# === Run on both detox output datasets ===
if __name__ == "__main__":
    evaluate_detoxified_dataset("new_dataset_results.csv", "detoxified_new_dataset")
    evaluate_detoxified_dataset("kaggle_dataset_results.csv", "detoxified_kaggle_dataset")
