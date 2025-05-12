import cohere
import pandas as pd
from tqdm import tqdm
import time
import torch
import csv
from transformers import BertTokenizer, BertForSequenceClassification

# Set up API key, cohere, API key limitations, etc etc
API_KEY = "API_KEY_NOT_SHOWN_FOR_SECURITY_PURPOSES"
co = cohere.Client(API_KEY)

RATE_LIMIT = 40                     # Max API calls per minute for trial key
SLEEP_DURATION = 60                # Seconds to sleep after hitting rate limit
THRESHOLD = 0.3                    # BERT confidence threshold to classify as clickbait
MAX_CALLS = 1000                   # Max total Cohere API calls allowed (due to trial key)

# Global counter to track total API usage
total_api_calls = 0

BERT_MODEL_DIR = "bert_clickbait_model"
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

# Prompt function, needed for each API call
def build_prompt(headline):
    return f"""
Rephrase the following headlines to remove hyperbole and clickbait while keeping them informative and grammatically complete.

Clickbait: You won’t believe what this AI startup just did!
Neutral: AI startup announces major breakthrough.

Clickbait: This one DIY trick will change your life forever!
Neutral: A simple DIY technique improves home organization.

Clickbait: The shocking truth about climate change will blow your mind!
Neutral: New study reveals key findings about climate change.

Clickbait: What happened next will surprise you: space exploration breakthrough
Neutral: Scientists report new milestone in space exploration.

Clickbait: {headline}
Neutral:"""

# Cohere API call with our previously shown prompt and with a low temperature value for consistent-ish results
def detoxify_headline(headline):
    prompt = build_prompt(headline)
    try:
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=60,
            temperature=0.3,
            stop_sequences=["\n"]
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"[ERROR: {str(e)}]"

# Runs the BERT model at the assigned threshold to assess the detox success rate
def run_bert_on_headlines(headlines, batch_size=32):
    confidences = []
    for i in tqdm(range(0, len(headlines), batch_size), desc="Re-evaluating with BERT"):
        batch = headlines[i:i + batch_size]
        inputs = bert_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidences.extend(probs[:, 1].tolist())  
    predicted = [1 if c > THRESHOLD else 0 for c in confidences]
    return predicted, confidences

# This is just tracking the progress as you run the code, don't worry too much about this.
def process_file(input_file, output_file):
    global total_api_calls
    if total_api_calls >= MAX_CALLS:
        print("Global API call limit reached. Skipping:", input_file)
        return

    print(f"\nProcessing: {input_file}")
    df = pd.read_csv(input_file, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')

    detoxified = []
    rate_limit_counter = 0

    for headline in tqdm(df['headline'], desc="Detoxifying"):
        if total_api_calls >= MAX_CALLS:
            print("Reached 1000 total API calls. Stopping.")
            break

        if rate_limit_counter >= RATE_LIMIT:
            print(f"\nRate limit hit. Sleeping for {SLEEP_DURATION + 1} seconds...\n")
            time.sleep(SLEEP_DURATION + 1)
            rate_limit_counter = 0

        detoxified.append(detoxify_headline(headline))
        rate_limit_counter += 1
        total_api_calls += 1

    df = df.iloc[:len(detoxified)].copy()
    df['detoxified_headline'] = detoxified

    print("Running BERT on detoxified headlines...")
    pred_labels, confidences = run_bert_on_headlines(detoxified)
    df['detox_confidence'] = confidences
    df['detox_predicted_label'] = pred_labels

    pct_clickbait = sum(pred_labels) / len(pred_labels) * 100
    print(f"BERT thinks {sum(pred_labels)} / {len(pred_labels)} ({pct_clickbait:.2f}%) of detoxified are still clickbait")

    df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved output to: {output_file}")

# Detoxify the kaggle dataset and new dataset
if __name__ == "__main__":
    datasets = [
        ("bert_predictions_detoxify_clickbait_only.csv", "new_dataset_results.csv"),
        ("bert_predictions_testing_clickbait_only.csv", "kaggle_dataset_results.csv")
    ]

    for input_path, output_path in datasets:
        process_file(input_path, output_path)
        if total_api_calls >= MAX_CALLS:
            print("== 1000-call limit reached. Terminating script. ==")
            break
