Usage:

Put the Cohere API key I gave you into detox.py

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

