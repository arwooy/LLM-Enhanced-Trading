#!/usr/bin/env python
# coding: utf-8

# # IBM Granite 3.0

# ### IBM Granite is a state-of-the-art language model developed by IBM, designed for a variety of natural language processing tasks, including sentiment analysis, text summarization, and question-answering. Leveraging advanced transformer-based architectures, Granite excels in generating coherent and contextually accurate outputs.
# 
# ### IBM Granite 3.0-8B-Instruct is an advanced language model developed by IBM, featuring 8 billion parameters. It has been fine-tuned from the Granite 3.0-8B-Base model using a combination of open-source instruction datasets with permissive licenses and internally collected synthetic datasets. This fine-tuning process employs diverse techniques within a structured chat format, including supervised fine-tuning, model alignment using reinforcement learning, and model merging.
# 
# ### Hugging Face link: https://huggingface.co/ibm-granite/granite-3.0-8b-instruct

# In[1]:


get_ipython().system('pip install torch torchvision torchaudio')
get_ipython().system('pip install accelerate')
get_ipython().system('pip install transformers')


# ## Loading the Model and Tokenizer

# ### Model: IBM Granite 3.0-8B-Instruct is an advanced transformer model with 8 billion parameters, leveraging a decoder-only architecture. It is fine-tuned using Reinforcement Learning with Human Feedback (RLHF), designed to excel in instruction-following tasks like summarization, classification, and multilingual dialogue generation.
# 
# ### Tokenizer: Its tokenizer is based on a custom implementation of Byte Pair Encoding (BPE), tailored for efficient tokenization of multilingual text and optimized for long-context scenarios. This allows the model to handle complex and diverse language tasks with improved accuracy and efficiency.
# 
# ### The model is loaded from Hugging Face with half-precision (float16) for optimized performance 

# In[2]:


import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# In[4]:


# Loading the model and tokenizer
MODEL_NAME = "ibm-granite/granite-3.0-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)


# ## Preparing sentiment analysis prompts

# ### This section constructs prompts for the sentiment analysis task using a predefined template and dynamically inserts each text input from the dataset. The prompts are designed to elicit a response strictly as 'positive', 'negative', or 'neutral'. A separate list is initialized to store the model's predicted sentiment labels for further analysis.

# In[5]:


df = pd.read_csv("data.csv")
true_labels = df["Sentiment"].str.lower()


# In[45]:


# Preparing the prompts for predictions
prompt_template = "Classify the sentiment of the text as 'positive', 'negative', or 'neutral': {text}. Respond with only one word: 'positive', 'negative', or 'neutral'."
prompts = df["Sentence"].apply(lambda sentence: prompt_template.format(text=sentence)).tolist()

# creating a list to store the predicted labels
predicted_labels = []


# ## Batch Processing for sentiment prediction

# ### This section processes the sentiment analysis prompts in batches to optimize computational efficiency. Each batch is tokenized and passed to the model for prediction. The output is decoded, and the predicted sentiment ('positive', 'negative', or 'neutral') is extracted and stored in the predicted_labels list for further analysis.
# 
# ### The batch_size here is chosen in accordance with the GPU Memory.

# In[46]:


from tqdm import tqdm
import re


# In[47]:


batch_size = 16


# In[48]:


# Processing prompts in batches
for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Batches"):
    batch_prompts = prompts[i:i + batch_size]

    # Tokenizing batch
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

    # Generating output for the batch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=500)

    # Decoding and extracting sentiments
    for output in outputs:
        output_text = tokenizer.decode(output, skip_special_tokens=True)
        sentiment = output_text.split()[-1].lower().rstrip('.')
        predicted_labels.append(sentiment)
        #print(sentiment)


# In[51]:


df['Predicted Sentiment'] = predicted_labels


# ## Evaluating Model Performance

# ### This section evaluates the sentiment analysis performance of IBM's Granite-3.0-8B-Instruct model. The true and predicted sentiment labels are compared to calculate key metrics: accuracy, precision, recall, and F1 score. These metrics provide insights into the model's effectiveness in classifying sentiments accurately and consistently.

# In[52]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# In[56]:


true_labels = df["Sentiment"]
predicted_labels = df["Predicted Sentiment"]

# Calculating metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

# Printing the results
print("Sentiment Analysis metrics for IBMs Granite-3.0-8B-Instruct model on Hugging Face")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# ## Benchmarking Conclusion

# ### The sentiment analysis performance of IBM's Granite-3.0-8B-Instruct model, as reflected by its accuracy (58.61%), precision (69.42%), recall (58.61%), and F1 score (62.07%), falls short compared to FinGPT. This highlights FinGPT's superior capability in handling sentiment classification tasks, making it a more reliable choice for financial and general sentiment analysis.
# 
# ### IBM's Granite-3.0-8B-Instruct model underperforms compared to FinGPT in sentiment analysis due to its general-purpose training objectives. Unlike FinGPT, which is fine-tuned on financial and sentiment-specific datasets, Granite is designed for broader, instruction-based tasks and lacks domain-specific optimization. This results in FinGPT being more accurate and effective in understanding sentiment nuances, especially in financial contexts.
