# LLM AI-TEXT DEETECTION USING LSTM AND NEURAL NETWORKS

Greetings to our AI-Generated Text Detection project! Within this repository, we unveil a powerful solution designed to identify AI-generated text using two methods-
### 1. (LSTM and Neural Networks) 
### 2. RoBERTaForSequenceClassification model

# The Repository contains 3 notebooks.
1. Dataset_preparation.ipynb
2. TextDetection_using_LSTM_NN.ipynb with accuracy 0.89%
3. TextDetection_using_RoBERTa.ipynb with accuracy 0.78%


# Getting Started

->The Training Dataset provided in the Problem is highly imbalanced wrt to the counts of generated and human written text. So we need to use other potential dataset along with the provided dataset for training.
The Dataset_preparation.ipynb include the required for code for formation of the external dataset.
The external data is taken from various different Datasets from Kaggle.
The final data was then concatenated with the given data and any duplicate entries were dropped.

## Kaggle Datasets used include-
4k Mixtral87b Crafted Essays for Detect AI Comp
daigt-external-dataset
daigt-data-llama-70b-and-falcon180b
daigt-proper-train-dataset
daigt-gemini-pro-8-5k-essays
daigt-v2-train-dataset
daigt-misc
hello-claude-1000-essays-from-anthropic
LLM-generated essay using PaLM from Google Gen-AI
llm-7-prompt-training-dataset
llm-mistral-7b-instruct-texts

->the data was then preprocessed (like removing stopwords,punctuation marks etc.) and vectorized for feeding into the model.

From here I have worked upon two different approaches as mentioned above.
1.
->The model is then constructed which includes several Conv layers, LSTM etc. and the parameters are set. Model is then Trained on the prepared dataset.
->Then we load the testing dataset and process it to obtain predictions from model.
----------------------> Accuracy with this method is obtained to be 0.89% <-----------------------------------


2.
Second approach includes the use of RoBERTa model for sequence classification.
RoBERTa-> or Robustly optimized BERT approach, is an advanced natural language processing (NLP) model. It is based on BERT but incorporates key modifications for enhanced performance. RoBERTa achieves state-of-the-art results in various NLP tasks by employing dynamic masking, larger batch sizes, and omitting the "Next Sentence Prediction" task during training. Its architecture, built upon transformers, captures rich contextual embeddings and has proven to be robust and effective in understanding and generating human-like text.
-
->->prepared dataset is convereted into Dataset Object and later tokenized for feeding into the model.
->the pretrained RoBERTa model is further trained on our dataset. 
->The major setback with this way is it has a high computation cost. the model takes really large amount of time to be trained and requires powerful GPU
->Test dataset is then processed to obtain predictions by feeding in model.
----------------------> Accuracy with this method is obtained to be 0.78% <-----------------------------------




