import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import pickle as pkl
import sys

from Model_Evaluation import embed_texts, calculate_mrr5, calculate_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device, end="\n\n")

model_name = sys.argv[1] if len(sys.argv) > 1 else "all-MiniLM-L6-v2"

print("Using model:", model_name, end="\n\n")

model = SentenceTransformer(model_name, device=device)

model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")

dev_df = pd.read_csv("../X_Data/subtask4b_query_tweets_dev.tsv", sep="\t")

with open("../X_Data/subtask4b_collection_data.pkl", "rb") as f:
    data = pkl.load(f)
    collection = pd.DataFrame(data)

collection["combined"] = (
    "Title: "
    + collection["title"].astype(str)
    + " Abstract: "
    + collection["abstract"].astype(str)
    + " Authors: "
    + collection["authors"].astype(str)
)

print("Combined columns", collection["combined"].shape)


# Calculate embeddings of combined text
paper_embeddings = embed_texts(collection["combined"].tolist(), model, batch_size=32)


# Calculate embeddings of tweet text
tweet_embeddings = embed_texts(dev_df["tweet_text"].tolist(), model, batch_size=32)

# Calculate similarity
similatrity_tweet_input = [
    (row["post_id"], row["cord_uid"], row["tweet_text"], tweet_embeddings[i])
    for i, (_, row) in tqdm(enumerate(dev_df.iterrows()), total=len(dev_df))
]


similarity_collection_input = [
    (row["cord_uid"], row["title"], paper_embeddings[i])
    for i, (_, row) in tqdm(enumerate(collection.iterrows()), total=len(collection))
]

similarity_scores = calculate_similarity(
    similatrity_tweet_input,
    similarity_collection_input,
)


mrr5 = calculate_mrr5(similarity_scores)

print(f"MRR@5: {mrr5:.4f}")
