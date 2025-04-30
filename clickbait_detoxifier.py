import cohere
import pandas as pd
from tqdm import tqdm
import time
import csv  # for quoting mode

# Initialize Cohere client with API key (please don't re-push this with the key, make sure you remove it if you need to re-push)
co = cohere.Client("PUT_COHERE_API_KEY_HERE")

# Optional: Checks for any strange formatting or extra commas, I mainly just threw this in for safety reasons
with open("bert_predictions_clean.csv", encoding='utf-8') as f:
    for i, line in enumerate(f, start=1):
        if line.count(",") != 1:
            print(f"Line {i}: {line.strip()}")

# Load dataset
df = pd.read_csv("bert_predictions_clean.csv", quoting=csv.QUOTE_MINIMAL, encoding='utf-8')

# Filter only clickbait headlines
clickbait_df = df[df['clickbait'] == 1].copy()

# For our prompt we're using few-shot prompting, specifically 4-shot prompting to show the model what we're aiming for as an output
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

# Detoxification function, uses Cohere's command-r-plus model
def detoxify_headline(headline):
    prompt = build_prompt(headline)
    try:
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=60,
            temperature=0.7,
            stop_sequences=["\n"]
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"[ERROR: {str(e)}]"

# When using Cohere's trial keys we're limited to 40 calls per minute. To circumvent this we track the amount of calls/headlines we send within a minute
# and immediately stunt the process for 60 seconds to avoid hitting this rate limit. Not an amazing speed when detoxifying, but the quality
# of the new headlines is well-worth the wait
detoxified = []
counter = 0
RATE_LIMIT = 40
SLEEP_DURATION = 60  

for i, headline in enumerate(tqdm(clickbait_df["headline"], desc="Detoxifying")):
    if counter == RATE_LIMIT:
        print(f"\nRate limit hit at {i} items. Sleeping for {SLEEP_DURATION} seconds...\n")
        time.sleep(SLEEP_DURATION)
        counter = 0

    detoxified.append(detoxify_headline(headline))
    counter += 1

# Assign detoxified headlines
clickbait_df["detoxified_headline"] = detoxified

# Save to CSV
output_path = "clickbait_headlines_detoxified.csv"
clickbait_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

print(f"\nSaved detoxified headlines to: {output_path}")
