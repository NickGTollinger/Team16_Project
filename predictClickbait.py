import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

# Note: This uses the original dataset that we trained BERT on. run_BERT and detox.py uses a new one for reasons explained in run_BERT

# Load our BERT model and tokenizer, same initial set-up as our run_BERT
MODEL_DIR = "bert_clickbait_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def batch_predict_probs(headlines, batch_size=32):
    all_clickbait_probs = []

    for i in tqdm(range(0, len(headlines), batch_size), desc="Predicting"):
        batch = headlines[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            all_clickbait_probs.extend(probs[:, 1].tolist()) 

    return all_clickbait_probs

def evaluate_dataset(input_file, output_prefix):
    batch_size = 32

    # Create output folder for CSVs
    output_folder = "threshold_results"
    os.makedirs(output_folder, exist_ok=True)

    # Load headlines and ground truth labels
    df = pd.read_csv(input_file)
    if "headline" not in df.columns or "clickbait" not in df.columns:
        raise ValueError("Input CSV must have 'headline' and 'clickbait' columns.")

    headlines = df["headline"].tolist()
    true_labels = df["clickbait"].tolist()

    print(f"\nLoaded {len(headlines)} headlines for evaluation from {input_file}.")

    # Predict clickbait probabilities
    clickbait_probs = batch_predict_probs(headlines, batch_size=batch_size)

    # Here we sweep over different BERT "thresholds." This means that if the probability of something being clickbait
    # is over this threshold value, it will be considered clickbait. Otherwise it's considered non-clickbait.
    # I selected 0.3 as our threshold for run_BERT and clickbait_detoxification based on these statistics
    thresholds = [i/100 for i in range(10, 91, 5)]  

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Decent formatting for results, we could see if there's a way to output a true table
    print(f"\nResults for {input_file}")
    print("Threshold | Accuracy | Precision | Recall | F1 Score")
    print("---------------------------------------------------------")
    for threshold in thresholds:
        preds = [1 if prob > threshold else 0 for prob in clickbait_probs]

        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, zero_division=0)
        rec = recall_score(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, zero_division=0)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        print(f"{threshold:.2f}      | {acc:.4f}  | {prec:.4f}    | {rec:.4f} | {f1:.4f}")

        # Save CSV for this threshold
        results_df = pd.DataFrame({
            "headline": headlines,
            "predicted_label": preds,
            "true_label": true_labels,
            "confidence": clickbait_probs
        })
        output_path = os.path.join(output_folder, f"{output_prefix}_threshold_{threshold:.2f}.csv")
        results_df.to_csv(output_path, index=False)

    # Plot F1 Score
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, f1_scores, marker='o', label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'Threshold vs F1 Score ({output_prefix})')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Precision and Recall
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, precisions, marker='o', label='Precision')
    plt.plot(thresholds, recalls, marker='o', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold vs Precision and Recall ({output_prefix})')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, accuracies, marker='o', label='Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'Threshold vs Accuracy ({output_prefix})')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_dataset("testing.csv", "original_set")
    evaluate_dataset("detoxify_dataset.csv", "detox_set")

# Results show an upward trend of accuracy vs threshold using the original dataset of 3200 samples
# Results for the new smaller dataset show a downward trend of accuracy vs threshold
# This is not necessarily unexpected, models will not generalize well to new data, so it suggests
# that our BERT model is trained for a specific pattern of clickbait. This is not necessarily bad
# it just means it could be improved given another large dataset. Our current and new dataset
# probably have their own sources of bias. A consistent precision of 1 suggests that BERT is very
# confident in what it predicts as clickbait, but not able to account for many false positives.