Usage:

To run clickbait_detoxifier.py, you need a Cohere account from which you make a Trial Key. Paste that trial key into the API key variable.

Ideally you should be on Python 3.11 for the best results. It is unlikely to work on later versions, but may work on earlier versions.
Simply run the below line to get all necessary libraries and packages needed to run the code:

py -3.11 -m pip install -r requirements.txt


run_BERT.py: Used to test our BERT model over a new dataset
py -3.11 run_BERT.py

predictClickbait.py: Provides statistics and visuals on how BERT runs on the original dataset
py -3.11 run_BERT.py

BERT_trainer.py: Was used to train BERT, not needed if you have the zip on the GitHub
py -3.11 BERT_trainer.py

detoxify.py: Detoxifies the dataset produced from run_BERT
py -3.11 BERT_trainer.py

checkDetoxifierAccuracy.py is exactly what you think, it checks the accuracy of the resulting file (currently kaggle_dataset_results.csv)

CSVs Used:
bert_predictions_testing and bert_predictions_testing_clickbait_only are the full predictions on the Kaggle dataset, and its predictions of what is clickbait respectively

bert_predictions_detoxify and bert_predictions_detoxify_clickbait_only are the same as above, but for the new dataset

clickbait_data.csv is the full Kaggle dataset

new_dataset_results is the detoxified headlines of the new dataset

kaggle_dataset_results is the detoxified headlines of the kaggle dataset

training.csv, validation.csv, and testing.csv are an 80/10/10 split of the kaggle dataset respectively. They were used to train our BERT model.