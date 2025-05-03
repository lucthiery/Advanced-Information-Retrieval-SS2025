import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
import requests
from nltk import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
import emoji
import zipfile

# ----------------------------------------
# 1. Load the files
# ----------------------------------------

# Load document base (pkl format)
doc_url = "https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/raw/main/task4/subtask_4b/subtask4b_collection_data.pkl"
doc_file = "subtask4b_collection_data.pkl"

if not os.path.exists(doc_file):
    with open(doc_file, "wb") as f:
        f.write(requests.get(doc_url).content)

with open(doc_file, "rb") as f:
    documents = pickle.load(f)

# Load query tweets (TSV format)
query_url = "https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/raw/main/task4/subtask_4b/subtask4b_query_tweets_dev.tsv"
query_file = "subtask4b_query_tweets_dev.tsv"

if not os.path.exists(query_file):
    with open(query_file, "wb") as f:
        f.write(requests.get(query_url).content)

queries = pd.read_csv(query_file, sep="\t")

# ----------------------------------------
# 2. Load GloVe embeddings
# ----------------------------------------

model = "glove.6B"
glove_url = "http://nlp.stanford.edu/data/%s.zip" % model
glove_path = "%s.100d.txt" % model

# Download and unzip if needed
if not os.path.exists(glove_path):
    glove_zip = "%s.zip" % model
    response = requests.get(glove_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(glove_zip, "wb") as f, tqdm(
        desc="Downloading " + model,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))
    with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(glove_zip)

# Load GloVe vectors
def load_glove_embeddings(path):
    embeddings = {}
    with open(path, encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove = load_glove_embeddings(glove_path)

# ----------------------------------------
# 3. Text preprocessing and vectorization
# ----------------------------------------

def preprocess(text):
    text = emoji.replace_emoji(text)  # Remove emojis
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    # tokens = word_tokenize(text)
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def get_avg_embedding(text, embeddings, dim=100):
    words = preprocess(text)
    valid_vectors = [embeddings[word] for word in words if word in embeddings]
    if not valid_vectors:
        return np.zeros(dim)
    return np.mean(valid_vectors, axis=0)

def get_weighted_avg_embedding(text, embeddings, tfidf_vectorizer, dim=100):
    words = preprocess(text)
    feature_vector = np.zeros(dim, dtype=np.float32)
    weight_sum = 0.0
    for word in words:
        if word in embeddings and word in tfidf_vocabulary:
            weight = tfidf_vectorizer.idf_[tfidf_vocabulary[word]]
            feature_vector += embeddings[word] * weight
            weight_sum += weight
    if weight_sum > 0:
        return feature_vector / weight_sum
    else:
        return np.zeros(dim)


# Compute document embeddings
doc_ids = documents['cord_uid'].tolist()
doc_texts = [data['title'] + ' ' + data['abstract'] for _, data in documents.iterrows()]

processed_doc_texts = [" ".join(preprocess(text)) for text in doc_texts]
# Initialize and fit TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(processed_doc_texts)
tfidf_vocabulary = tfidf_vectorizer.vocabulary_

doc_embeddings = np.array([
    get_weighted_avg_embedding(text, glove, tfidf_vectorizer) for text in tqdm(doc_texts, desc="Embedding documents")
])

# ----------------------------------------
# 4. Process queries and compute similarity
# ----------------------------------------

results = []

for _, row in tqdm(queries.iterrows(), total=len(queries), desc="Processing queries"):
    tweet_id = row['post_id']
    tweet_text = row['tweet_text']
    tweet_embedding = get_weighted_avg_embedding(tweet_text, glove, tfidf_vectorizer)

    sims = cosine_similarity([tweet_embedding], doc_embeddings)[0]
    top_indices = np.argsort(sims)[-25:][::-1]  # Top 25
    top_doc_ids = [doc_ids[i] for i in top_indices]
    results.append({
        "id": tweet_id,
        "doc_ids": top_doc_ids
    })

# ----------------------------------------
# 5. compute MRR
# ----------------------------------------

df = pd.DataFrame(results)
print(df.head())

def get_performance_mrr(data, col_gold, col_pred, list_k = [1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        d_performance[k] = data["in_topx"].mean()
    return d_performance

print("Mean Reciprocal Rank (MRR) for top 1, 5, 10:")
print(get_performance_mrr(pd.merge(queries, df, left_on='post_id', right_on='id', how='left'), 'cord_uid', 'doc_ids'))
# {1: 0.155, 5: 0.19347619047619047, 10: 0.19996882086167797}
# {1: 0.15714285714285714, 5: 0.19311904761904763, 10: 0.20062528344671202}
# word_tokenize with emoji remove {1: 0.2, 5: 0.23891666666666667, 10: 0.24508390022675736}
# tweet tokenizer {1: 0.2092857142857143, 5: 0.2466190476190476, 10: 0.25320521541950114}
# twitter model {1: 0.15714285714285714, 5: 0.19215476190476188, 10: 0.19848412698412699}
# remove stop words {1: 0.20714285714285716, 5: 0.25065476190476194, 10: 0.25725595238095234}


# Optional: Save to file
import json
with open("top25_results.json", "w") as f:
    json.dump(results, f, indent=2)
