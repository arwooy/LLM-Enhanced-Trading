#!/usr/bin/env python
# coding: utf-8

# # Meta LLAMA 3.1

# ### Meta's LLAMA 3.1 (Large Language Model Meta AI) represents the latest advancement in Meta's series of powerful foundational models for natural language understanding and generation. LLAMA 3.1 is designed to deliver high performance across a variety of tasks, including text classification, summarization, and conversational AI, with a focus on efficiency and scalability.
# 
# ### The Meta Llama 3.1-8B-Instruct is a multilingual language model by Meta AI, designed for instruction-following tasks in languages like English, German, Hindi, and Spanish. With 8 billion parameters, it excels in conversational AI and diverse NLP scenarios. 
# 
# ### Hugging Face link: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# In[1]:


get_ipython().system('pip install torch torchvision torchaudio')
get_ipython().system('pip install accelerate')
get_ipython().system('pip install transformers')
get_ipython().system('pip instal tqdm')


# ## Model and Tokenizer Initialization

# ### Model: The Meta Llama 3.1-8B-Instruct is a state-of-the-art large language model featuring 8 billion parameters. It is built on a transformer-based architecture with highly optimized multi-head attention mechanisms, layer normalization, and feed-forward networks for efficient processing of textual data. Trained on diverse and high-quality datasets using instruction-tuning techniques, it excels in following user prompts across a wide range of tasks. The training process emphasizes supervised fine-tuning on human-annotated datasets to ensure accurate, context-aware, and instruction-adherent outputs.
# 
# ### Tokenizer: The tokenizer for Meta Llama 3.1-8B-Instruct employs a byte-pair encoding (BPE) algorithm, designed to tokenize text efficiently by breaking it into subword units. This approach ensures a balance between token granularity and vocabulary size, allowing for handling of rare words and out-of-vocabulary tokens effectively. The tokenizer supports a maximum context length suitable for processing large inputs, enabling the model to capture dependencies across long sequences while maintaining computational efficiency.
# 
# ### The model is loaded from Hugging Face with half-precision (float16) for optimized performance 

# In[2]:


from transformers import AutoTokenizer, AutoModelForCausalLM


# In[3]:


from huggingface_hub import login
login()


# In[4]:


import torch

# Loading the model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)


# #### As I will be using batch processing later, I am assigning the End of Sentence (eos) token as the pad token to ensure that the sequences in a batch are all of the same length.

# In[28]:


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ## Preparing Dataset and Prompts for Sentiment Analysis

# ### This section loads the dataset and prepares structured prompts for sentiment classification. Each sentence from the dataset is dynamically formatted into a predefined template, instructing the model to classify sentiment as 'positive', 'negative', or 'neutral' while ensuring consistent output formatting. True labels are standardized to lowercase, and an empty list is initialized to store the model’s predictions.

# In[29]:


# Loading dataset
import pandas as pd

df = pd.read_csv("data.csv")
true_labels = df["Sentiment"].str.lower()


# In[30]:


# Preparing the prompts for predictions
prompt_template = (
        "Classify the sentiment of the following text as either 'positive', 'negative', or 'neutral'. Reply strictly in the format:\n\n"
        "Sentiment: <positive/negative/neutral>\n\n"
        "Do not include explanations, additional text, or comments. Reply with only the sentiment.\n\n"
        "Text: {text}\n\nSentiment:"
    )
prompts = df["Sentence"].apply(lambda sentence: prompt_template.format(text=sentence)).tolist()

# creating a list to store the predicted labels
predicted_labels = []


# ## Batch Processing for Sentiment Analysis
# 
# ### This section processes the prepared prompts in batches for efficient sentiment classification using the model. Prompts are tokenized with padding and truncation to ensure compatibility with the model’s input requirements. The model generates predictions in a no-gradient environment for faster inference. Extracted outputs are decoded and analyzed using regular expressions to identify the sentiment ('positive', 'negative', or 'neutral') from the model’s response. If no valid sentiment is found, a default fallback of 'neutral' is used. The final sentiment for each prompt is appended to a list for further evaluation.

# In[31]:


from tqdm import tqdm
import re


# In[32]:


batch_size = 16


# In[33]:


# Processing prompts in batches
for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Batches"):
    batch_prompts = prompts[i:i + batch_size]

    # Tokenizing batch
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

    # Generating output for the batch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=400)

    # Decoding and extracting sentiments
    for output in outputs:
        output_text = tokenizer.decode(output, skip_special_tokens=True)
        # using Regex to match inline or newline cases
        matches = re.findall(r"(?i)sentiment:\s*(positive|negative|neutral)|(?i)sentiment:\s*$", output_text)

        # Extracting the last valid sentiment, as the sentiment would be last occurence after "sentiment:" in the output_text
        if matches:
            for match in reversed(matches):
                if isinstance(match, tuple):
                    sentiment_candidate = match[0] or match[1]
                else:
                    sentiment_candidate = match

                if sentiment_candidate:
                    sentiment = sentiment_candidate.lower()
                    break
            else:
                sentiment = "neutral"  # Default fallback
        else:
            # If no valid sentiment is found, default to neutral
            sentiment = "neutral"

        predicted_labels.append(sentiment)
        #  debugging
        #print(f"Extracted Matches: {matches}")
        #print(f"Final Extracted Sentiment: {sentiment}")


# In[34]:


df['Predicted Sentiment'] = predicted_labels


# ## Evaluating Sentiment Analysis Metrics
# 
# ### This section evaluates the performance of Meta's Llama-3.1-8B-Instruct model in predicting sentiment. True sentiment labels from the dataset and predicted labels generated by the model are compared to compute key evaluation metrics: accuracy, precision, recall, and F1 score. These metrics, calculated using a weighted average to handle class imbalances, provide insights into the model's classification performance. The results are printed for analysis and benchmarking.

# In[35]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# In[36]:


true_labels = df["Sentiment"]
predicted_labels = df["Predicted Sentiment"]

# Calculating metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

# Printing the results
print("Sentiment Analysis metrics for Meta's Llama-3.1-8B-Instruct model on Hugging Face")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# ## Benchmarking Conclusion
# 
# ### The performance metrics of Meta's Llama-3.1-8B-Instruct model—accuracy, precision, recall, and F1 score—are notably lower compared to FinGPT on the same sentiment analysis task. This underperformance highlights the superiority of FinGPT for financial and sentiment-specific tasks, likely due to its specialized training on financial and sentiment-rich datasets, which aligns closely with the nuances of this domain. In contrast, Llama-3.1, while powerful and general-purpose, lacks the fine-tuning necessary to excel in such specialized scenarios.
