from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load model and tokenizer
MODEL_DIR = "bert_clickbait_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Use GPU if available, if not it'll take 10 years sorry guys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def batch_predict(headlines, batch_size=32):
    predictions = []
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
            batch_preds = torch.argmax(probs, dim=1).tolist()
            batch_confidences = probs.max(dim=1).values.tolist()

        predictions.extend(batch_preds)
        confidences.extend(batch_confidences)

    return predictions, confidences
# Self explanatory variables, input file of csv data, an output file of its predictions, and the batch size
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="CSV file containing headlines")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="CSV file to save predictions")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    # Load testing data, needs to have the same columns as other data files to work (headline and clickbait columns)
    df = pd.read_csv(args.input_csv)

    if "headline" not in df.columns:
        raise ValueError("Input CSV must contain a 'headline' column.")

    headlines = df["headline"].tolist()
    predictions, confidences = batch_predict(headlines, args.batch_size)

    # Add predictions to DataFrame, it includes the predicted label and a confidence level
    df["prediction"] = predictions
    df["confidence"] = confidences

    # Compare prediction with classification and calculate average accuracy
    if "clickbait" in df.columns:
        accuracy = accuracy_score(df["clickbait"], df["prediction"])
        print(f"\nAccuracy: {accuracy:.2%}")
    else:
        print("\nNo ground truth labels found — skipping accuracy calculation.")

    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")
