# LLM AI-TEXT DETECTION USING LSTM AND NEURAL NETWORKS

Greetings to our AI-Generated Text Detection project! Within this repository, we unveil a powerful solution designed to identify AI-generated text using two methods-
### 1. (LSTM and Neural Networks) 
### 2. RoBERTaForSequenceClassification model

## The Repository contains 3 notebooks.
1. Dataset_preparation.ipynb
2. TextDetection_using_LSTM_NN.ipynb with accuracy 0.89%
3. TextDetection_using_RoBERTa.ipynb with accuracy 0.78%


## Getting Started

->The Training Dataset provided in the Problem is highly imbalanced wrt to the counts of generated and human written text. So we need to use other potential dataset along with the provided dataset for training.<br>
The Dataset_preparation.ipynb include the required for code for formation of the external dataset.<br>
The external data is taken from various different Datasets from Kaggle.<br>
The final data was then concatenated with the given data and any duplicate entries were dropped.<br>

## Kaggle Datasets used include-
4k Mixtral87b Crafted Essays for Detect AI Comp <br>
daigt-external-dataset <br>
daigt-data-llama-70b-and-falcon180b <br>
daigt-proper-train-dataset <br>
daigt-gemini-pro-8-5k-essays <br>
daigt-v2-train-dataset <br>
daigt-misc <br>
hello-claude-1000-essays-from-anthropic<br>
LLM-generated essay using PaLM from Google Gen-AI<br>
llm-7-prompt-training-dataset<br>
llm-mistral-7b-instruct-texts<br>

->the data was then preprocessed (like removing stopwords,punctuation marks etc.) and vectorized for feeding into the model. <br>

From here I have worked upon two different approaches as mentioned above.<br>
### 1. Using NN ans LSTM
->The model is then constructed which includes several Conv layers, LSTM etc. and the parameters are set. Model is then Trained on the prepared dataset.<br>
->Then we load the testing dataset and process it to obtain predictions from model.<br>

#### ----------------------> Accuracy with this method is obtained to be 0.89% <-----------------------------------<br>


### 2. Using RoBERTa Model
Second approach includes the use of RoBERTa model for sequence classification.<br>
RoBERTa-> or Robustly optimized BERT approach, is an advanced natural language processing (NLP) model. It is based on BERT but incorporates key modifications for enhanced performance. RoBERTa achieves state-of-the-art results in various NLP tasks by employing dynamic masking, larger batch sizes, and omitting the "Next Sentence Prediction" task during training. Its architecture, built upon transformers, captures rich contextual embeddings and has proven to be robust and effective in understanding and generating human-like text.
<br>
->->prepared dataset is convereted into Dataset Object and later tokenized for feeding into the model.<br>
->the pretrained RoBERTa model is further trained on our dataset. <br>
->The major setback with this way is it has a high computation cost. the model takes really large amount of time to be trained and requires powerful GPU<br>
->Test dataset is then processed to obtain predictions by feeding in model.<br>
#### ----------------------> Accuracy with this method is obtained to be 0.78% <-----------------------------------<br>

🚀 **Connect With Me:**
- LinkedIn: [LinkedIn Profile]( https://www.linkedin.com/in/krish-khadria-034401271/)
- Kaggle: [Kaggle Profile](https://www.kaggle.com/krishkhadria)
- GitHub: [GitHub Profile](https://github.com/krish-sky1ark/LLM-AI-Text-Detection/tree/main)


