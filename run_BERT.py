import json
import torch
import html
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load up our BERT model that we trained
MODEL_DIR = "bert_clickbait_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# If your device is compatible with cuda this could help speed things up, otherwise it's just relying on your CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# We decide on the BERT threshold given the results given our finding in predictClickbait.py. 10 was actually the threshold with the highest accuracy
THRESHOLD = 0.1

# We run BERT in batches of 32 headlines. Confidence intervals are documented too, though we don't really need them for anything specific
def batch_predict(headlines, batch_size=32):
    confidences = []

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
            batch_confidences = probs[:, 1].tolist()  # Clickbait probability

        confidences.extend(batch_confidences)

    return confidences

# Run BERT over our dataset. This is a different dataset so we can test how well the model generalizes to unseen data.
# Accuracy has never been below 70%. The outputted csv is all pieces of data BERT predicted to be clickbait. Used by detox.py.
# The 20% drop in accuracy compared to the original dataset is expected. Even a large dataset may have biases to it, so by using
# a new dataset we observe how well our BERT model generalizes (which seems to be moderately well)
if __name__ == "__main__":
    input_file = "detoxify_dataset.csv"
    batch_size = 32

    # Load CSV
    df = pd.read_csv(input_file)
    headlines = df['headline'].tolist()
    true_labels = df['clickbait'].tolist()

    print(f"Loaded {len(headlines)} headlines.")

    # Predict clickbait probabilities
    confidences = batch_predict(headlines, batch_size=batch_size)

    # Test probabilities in comparison to threshold to filter out false positives
    predicted_labels = [1 if prob > THRESHOLD else 0 for prob in confidences]

    # Evaluate the model's performance in predicting on this unseen data
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save all predictions
    output_file = "bert_predictions.csv"
    output_df = pd.DataFrame({
        'headline': headlines,
        'true_label': true_labels,
        'predicted_confidence': confidences,
        'predicted_label': predicted_labels
    })
    output_df.to_csv(output_file, index=False)
    print(f"\nSaved detailed predictions to {output_file}")

    # Save just the clickbait predictions (for detox.py)
    detox_input_df = pd.DataFrame({
        'headline': [h for h, p in zip(headlines, predicted_labels) if p == 1],
        'clickbait': [1] * sum(predicted_labels)
    })
    detox_input_df.to_csv("bert_predictions_clean.csv", index=False)
    print("Saved clickbait-only subset to clickbait_headlines_for_detox.csv")
