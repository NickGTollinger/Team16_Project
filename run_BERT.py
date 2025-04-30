import json
import torch
import html
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load up our trained BERT model

MODEL_DIR = "bert_clickbait_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Cuda can make things run faster if your device is compatible, not guaranteed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Threshold to classify as clickbait based on confidence/probability
THRESHOLD = 0.3

# Predict BERT clickbait probability in batches
def batch_predict(headlines, batch_size=32):
    confidences = []

    for i in tqdm(range(0, len(headlines), batch_size), desc="Predicting"):
        batch = headlines[i:i + batch_size]

        # Tokenize inputs
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        # Run model and compute softmax probabilities
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            batch_confidences = probs[:, 1].tolist()  

        confidences.extend(batch_confidences)

    return confidences

# Run BERT, get its performance stats
def run_on_dataset(input_file, output_prefix):
    df = pd.read_csv(input_file)
    headlines = df['headline'].tolist()
    true_labels = df['clickbait'].tolist()

    print(f"\nProcessing {input_file} with {len(headlines)} headlines...")

    # Predict clickbait probabilities
    confidences = batch_predict(headlines)
    predicted_labels = [1 if prob > THRESHOLD else 0 for prob in confidences]

    # The statistics we're looking at
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Print results of prediction accuracy
    print(f"=== Results for {output_prefix} ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Save all predictions
    full_output = f"bert_predictions_{output_prefix}.csv"
    pd.DataFrame({
        'headline': headlines,
        'true_label': true_labels,
        'predicted_confidence': confidences,
        'predicted_label': predicted_labels
    }).to_csv(full_output, index=False)
    print(f"Saved full predictions to {full_output}")

    # Save only clickbait predictions
    clickbait_output = f"bert_predictions_{output_prefix}_clickbait_only.csv"
    pd.DataFrame({
        'headline': [h for h, p in zip(headlines, predicted_labels) if p == 1],
        'clickbait': [1] * sum(predicted_labels)
    }).to_csv(clickbait_output, index=False)
    print(f"Saved clickbait-only subset to {clickbait_output}")

if __name__ == "__main__":
    datasets = [
        ("detoxify_dataset.csv", "detoxify"),
        ("testing.csv", "testing")
    ]
    for input_file, prefix in datasets:
        run_on_dataset(input_file, prefix)
