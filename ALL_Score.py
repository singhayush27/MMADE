#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


# In[2]:


import pandas as pd
# Read CSV dataset from Pandas
df_blipv2 = pd.read_csv('blip_image.csv') #, nrows = nRowsRead
df_blipv2.dataframeName = 'blip_image.csv'
nRow, nCol = df_blipv2.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[3]:


df_blipv2


# ## ROUGE Score

# In[5]:


from rouge_score import rouge_scorer


# In[6]:


scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)


# In[7]:


total_scores_p = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
total_scores_r = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
total_scores_f = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
num_rows = len(df_blipv2)
for index,row in df_blipv2.iterrows(): 
    IMDB_summary = str(row['Text'])
    generated_summary = str(row['Output'])
#     print('<start>',IMDB_summary,'\n<mid>\n',generated_summary,'<end>')
    scores = scorer.score(IMDB_summary,generated_summary)
#     for key in scores:
#         print(f'{key}: {scores[key]}')
        # Accumulate scores for averaging
    for key in scores:
        total_scores_p[key] += scores[key].precision
        total_scores_r[key] += scores[key].recall
        total_scores_f[key] += scores[key].fmeasure  # Extract the fmeasure from Score object

# Calculate the average ROUGE scores
average_scores_p = {key: total_scores_p[key] / num_rows for key in total_scores_p}
average_scores_r = {key: total_scores_r[key] / num_rows for key in total_scores_r}
average_scores_f = {key: total_scores_f[key] / num_rows for key in total_scores_f}

# Print average ROUGE scores
for key in average_scores_p:
    print(f'Average_p {key}: {average_scores_p[key]}')
print('\n')
for key in average_scores_r:
    print(f'Average_r {key}: {average_scores_r[key]}')
print('\n')
for key in average_scores_f:
    print(f'Average_f {key}: {average_scores_f[key]}')


# ## BLEU Score

# In[8]:


import numpy as np
import evaluate
from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu_score(references, outputs):
    
    bleu_scores = []
    for reference, output in zip(references, outputs):
        bleu_score = sentence_bleu(reference, output, weights=(1, 0, 0, 0))
        bleu_scores.append(bleu_score)

    return np.mean(bleu_scores)

# Get the reference sentences and predicted sentences.
reference_sentences = df_blipv2['Text'].tolist()
output_sentences = df_blipv2['Output'].tolist()

bleu = evaluate.load("bleu")
results = bleu.compute(predictions=output_sentences, references=reference_sentences)
print(results)
# Calculate the BLEU score.
# bleu_score = calculate_bleu_score(reference_sentences, output_sentences)

# # Print the BLEU score.
# print('BLEU score:', bleu_score)


# In[9]:


# Calculate the BLEU score.
bleu_score = calculate_bleu_score(reference_sentences, output_sentences)

# Print the BLEU score.
print('BLEU score:', bleu_score)


# ## BERT Score

# In[10]:


import pandas as pd
from bert_score import score

# Load your data frame
data = pd.read_csv("blip_image.csv")

# Extract the predictions and references from the data frame
predictions = data["Output"].tolist()
references = data["Text"].tolist()

# Calculate BERT Score
# Calculate BERT Score
P, R, F1 = score(predictions, references, lang='en')

# Add BERT Score metrics to the data frame
data["BERT_P"] = P.tolist()
data["BERT_R"] = R.tolist()
data["BERT_F1"] = F1.tolist()

# Calculate the average of all P, R, and F1 values
avg_P = sum(data["BERT_P"]) / len(data)
avg_R = sum(data["BERT_R"]) / len(data)
avg_F1 = sum(data["BERT_F1"]) / len(data)

# Print or use the average values
print("Average BERT Precision (P):", avg_P)
print("Average BERT Recall (R):", avg_R)
print("Average BERT F1 Score (F1):", avg_F1)

# Save the data frame with BERT Score metrics to a new CSV file
# data.to_csv("data_with_bertscore.csv", index=False)


# ## MOVER Score

# In[11]:


import pandas as pd
from moverscore_v2 import get_idf_dict, word_mover_score

# Assuming you have a DataFrame named 'df' with 'translations' and 'references' columns
# You can load your data using pandas, e.g., df = pd.read_csv('your_data.csv')

translations = df_blipv2['Output'].tolist()
references = df_blipv2['Text'].tolist()


# Create IDF dictionaries for translations and references
idf_dict_hyp = get_idf_dict(translations)
idf_dict_ref = get_idf_dict(references)

# Calculate Word Mover Score for the entire DataFrame
scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)

# Now, 'scores' contains the MoverScore for each pair of translation and reference


# In[12]:


import numpy as np

# Assuming you have a list of MoverScore results stored in 'scores'
average_score = np.mean(scores)

print(f"Average MoverScore: {average_score}")


# In[ ]:




