#!/usr/bin/env python
# coding: utf-8

# # FinGPT: LoRA-Optimized Financial Language Model

# ### FinGPT is a financial domain-focused large language model that employs Low-Rank Adaptation (LoRA) optimization for efficient fine-tuning on specialized datasets. It is built on top of existing foundational models like GPT, adapting them to excel in financial tasks such as sentiment analysis, market prediction, and financial text interpretation. The LoRA optimization enables FinGPT to integrate financial-specific knowledge effectively without requiring extensive computational resources, making it a powerful tool for financial AI applications.
# 
# ### FinGPT builds on the LLaMA 3 architecture, leveraging its transformer-based design for processing sequential data efficiently. It incorporates LoRA (Low-Rank Adaptation) optimization, enabling fine-tuning with reduced computational overhead. This approach allows FinGPT to focus on financial-specific tasks while retaining the general capabilities of the base model.
# 
# ### The model is trained on diverse financial datasets, including market reports, news articles, earnings calls, and social media data, particularly from platforms like Reddit and Twitter. This specialized training equips FinGPT with a nuanced understanding of financial language, trends, and sentiment, making it adept at tasks like sentiment analysis, market forecasting, and financial text summarization.
# 
# ### Hugging Face link: https://huggingface.co/FinGPT/fingpt-mt_llama3-8b_lora

# In[1]:


get_ipython().system('pip install transformers==4.40.1 peft==0.5.0')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install accelerate')
get_ipython().system('pip install torch')
get_ipython().system('pip install datasets')
get_ipython().system('pip install bitsandbytes')


# ## Loading FinGPT and Preparing the Dataset

# ### This section initializes the FinGPT model, a fine-tuned version of Meta-Llama-3-8B enhanced using LoRA (Low-Rank Adaptation). FinGPT specializes in financial tasks and sentiment analysis by leveraging datasets like market reports, financial news, and social media data. The model operates efficiently with 16-bit precision for optimized performance on GPUs.
# 
# ### The tokenizer, derived from Meta-Llama-3-8B, ensures compatibility with financial text, offering robust handling of sequential data. 
# 
# ### The Kaggle dataset containing financial sentences and their respective sentiments is loaded, preparing the groundwork for evaluating the model's predictions against ground truth labels.

# In[2]:


from huggingface_hub import login
login()


# In[3]:


import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Base model and PEFT (LoRA) model
base_model = "meta-llama/Meta-Llama-3-8B"
peft_model = "FinGPT/fingpt-mt_llama3-8b_lora"

# Load tokenizer
tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with 16-bit precision
model = LlamaForCausalLM.from_pretrained(base_model,
                                         trust_remote_code=True,
                                         device_map="auto",
                                         torch_dtype=torch.float16)  # Enable 16-bit precision

# Apply LoRA-based PEFT model
model = PeftModel.from_pretrained(model, peft_model, torch_dtype=torch.float16)
model = model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the Kaggle dataset
dataset_path = "/content/data.csv"  # Replace with the actual path
df = pd.read_csv(dataset_path)

# Assuming the dataset has a column 'Sentence' for news text and 'Sentiment' for ground truth labels
sentences = df['Sentence'].tolist()
true_labels = df['Sentiment'].tolist()

# Prepare for storing results
predicted_labels = []


# ## Batch Processing for Sentiment Predictions
# 
# ### This section processes financial news prompts in batches to predict sentiment using FinGPT. The prompts are constructed to include an instruction for identifying sentiment, followed by the input sentence. The model generates responses in batches, leveraging GPU for efficient processing.
# 
# ### The decoded outputs are parsed to extract the predicted sentiment after the "Answer:" field. Any errors in sentiment detection are marked appropriately. Finally, the predicted sentiments are stored for further evaluation.

# In[4]:


from tqdm import tqdm

# Set batch size
batch_size = 16  # Adjust based on your GPU memory

# Prepare prompts
df['Prompt'] = df['Sentence'].apply(lambda sentence: f'''Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
Input: {sentence}
Answer: ''')

prompts = df['Prompt'].tolist()

predicted_labels = []

# Process prompts in batches
for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Batches"):
    batch_prompts = prompts[i:i + batch_size]

    # Tokenize the batch
    tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)

    # Generate responses
    with torch.no_grad():
        outputs = model.generate(**tokens, max_length=200)

    # Decode and extract sentiments
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        if "Answer:" in response:
            sentiment = response.split("Answer:")[-1].strip()
        else:
            sentiment = "Error: Sentiment not detected"
        predicted_labels.append(sentiment)

# Add predictions to the DataFrame
df['Predicted Sentiment'] = predicted_labels


# ## Evaluating Sentiment Analysis Metrics 
# 
# ### This section evaluates the performance of FinGPT on the sentiment analysis task by comparing the predicted sentiments with the ground truth. It calculates the accuracy, precision, recall, and F1 score as performance metrics.
# 
# ### To ensure meaningful results, invalid predictions (e.g., errors or undefined sentiment values) are filtered out. The filtered true and predicted labels are then used to compute the metrics, providing a detailed assessment of the model's sentiment prediction capabilities.

# In[7]:


# Extract true and predicted labels
true_labels = df["Sentiment"].str.lower()
predicted_labels = df["Predicted Sentiment"].str.lower()

# Filter out invalid predictions (e.g., "Error")
valid_indices = predicted_labels.isin(["negative", "neutral", "positive"])
filtered_true_labels = true_labels[valid_indices]
filtered_predicted_labels = predicted_labels[valid_indices]

# Calculate metrics
accuracy = accuracy_score(filtered_true_labels, filtered_predicted_labels)
precision = precision_score(filtered_true_labels, filtered_predicted_labels, average="weighted")
recall = recall_score(filtered_true_labels, filtered_predicted_labels, average="weighted")
f1 = f1_score(filtered_true_labels, filtered_predicted_labels, average="weighted")


# In[8]:


print("Metrics for Sentiment Analysis with FinGPT model on Hugging Face:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# In[9]:


df.to_csv("kaggle_sentiment_data.csv", index=False)


# ## Benchmarking Conclusion
# 
# ### The FinGPT model outperforms both Llama 3.1 and Granite 3.0 across all key metrics—accuracy (0.7462), precision (0.7675), recall (0.7462), and F1 score (0.7488). This superior performance highlights FinGPT's effectiveness and suitability for sentiment analysis tasks, particularly in the financial domain. The reason why FinGPT performed better than the other two models are as follows:
# 
# ### 1. Task-Specific Fine-Tuning: FinGPT leverages LoRA-based fine-tuning on financial data, enabling it to understand the nuances of financial news and sentiments effectively. This targeted training allows the model to make more contextually accurate predictions.
# 
# ### 2. Domain-Specific Optimization: Unlike Llama 3.1 and Granite 3.0, which are general-purpose models, FinGPT is optimized for financial language processing. This optimization gives it a significant edge in capturing sentiment in domain-specific texts.
# 
# ### 3. Efficient Training with LoRA: The use of LoRA (Low-Rank Adaptation) ensures efficient fine-tuning without requiring retraining the entire model. This adaptation aligns the model closely with the sentiment nuances in financial datasets.
# 
# ### 4. Robust Handling of Financial Terminology: Financial texts often include jargon and market-specific terminology. FinGPT’s training on financial datasets equips it to handle such complexities better than general-purpose models.
# 
# ### FinGPT's specialization in financial language processing makes it a natural fit for sentiment analysis in financial contexts. Its superior performance metrics validate its ability to provide precise and reliable sentiment predictions, making it an excellent choice for tasks requiring sentiment classification in financial news or related texts.
