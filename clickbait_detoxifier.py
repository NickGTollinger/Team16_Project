from datasets import load_dataset, DatasetDict
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import pandas as pd
import os

# Step 1: Take our full dataset and split it 80/10/10 for training, validation, and testing data respectively
if not all(os.path.exists(f) for f in ["training.csv", "validation.csv", "testing.csv"]):
    print("Splitting dataset and saving to CSV...")

    full_dataset = load_dataset("csv", data_files="clickbait_data.csv")["train"].shuffle(seed=42)

    split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
    val_test_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

    # Assign names to our datasets for future use
    dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
    })

    # For convenience, save each of the datasets as a CSV
    dataset["train"].to_pandas().to_csv("training.csv", index=False)
    dataset["validation"].to_pandas().to_csv("validation.csv", index=False)
    dataset["test"].to_pandas().to_csv("testing.csv", index=False)

    print("Saved: training.csv, validation.csv, testing.csv")

# Step 2: With data sets saved, redeclare our the dataset variable as a dictionary of loaded data sets in preparation for training
print("Loading training, validation, and testing sets from CSV...")
dataset = DatasetDict({
    "train": load_dataset("csv", data_files="training.csv")["train"],
    "validation": load_dataset("csv", data_files="validation.csv")["train"],
    "test": load_dataset("csv", data_files="testing.csv")["train"]
})

# Step 3: Load Tokenizer & BERT Model, as we decided before BERT is a good choice for the clickbait classification part of our assignment
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Step 4: Tokenize our data sets and declare the headline column and rename the labels column to "clickbait" to conform with our original csv column names
def tokenize_function(example):
    return tokenizer(example["headline"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("clickbait", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Step 5: This is just general arguments. As a quick summary of what each of these means...
#output_dir is where our model will be exported
#eval_str and #save strategy means we do a validation batch and save the model at the end of each epoch
#train batch and eval batch is the amount of samples checked per CPU cores, training is set lower because it takes much longer and can be a bit intensive on our computers
#num_train_epochs is self explanatory
#logging doesn't matter too much, just shows what went on during the training and when logging occurs
#save_total_limit: 2 saves exist at a time, so the model from the 2nd and 3rd epochs
#metric_for_best_model: Obviously we just check our model's efficiency for accuracy. On the test set the current accuracy is 98.8%.
#fp16 is not compatible with all GPUs, but will boost training speed. I suggest you run with batch sizes that are multiples of 8, it's just good practice.

training_args = TrainingArguments(
    output_dir="./bert_clickbait_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True  # Set to False if you're using CPU
)

# This returns our stats of applying the model to test data, most importantly accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Step 6: We already established our training arguments, initialize the model and get ready to train 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

# Step 8: Train
trainer.train()

# Step 9: Somewhat unnecessary since the model should be saved to bert_click_classifier, but I threw this here just in case since bert_clickbait_classifier gets rewritten if the model is retrained. Bert_clickbait_model contains the original model resulting from training.
model_dir = "bert_clickbait_model"
print(f"\nSaving trained model and tokenizer to: {model_dir}")
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)

# Step 10: Get results on the test set
print("\nTest set results:")
metrics = trainer.evaluate(tokenized_datasets["test"])
print(metrics)
