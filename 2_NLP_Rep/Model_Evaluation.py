import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
from tqdm import tqdm
import torch
from torch.nn.functional import cosine_similarity
import pickle as pkl
from collections import defaultdict
import sys
import os

CHUNK_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device, end="\n\n")

model_name = sys.argv[1] if len(sys.argv) > 1 else "all-MiniLM-L6-v2"

print("Using model:", model_name, end="\n\n")

model = SentenceTransformer(model_name, device=device)

model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")


def embed_texts(texts, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        output = model.encode(batch, convert_to_tensor=True).cpu()
        embeddings.extend(output)
    return embeddings


def calculate_mrr5(results):
    mrr5 = 0.0
    for result in results:
        gold_paper = result["gold_paper"]
        retrieved_papers = result["retrieved"]
        # Find the rank of the gold paper
        if gold_paper in retrieved_papers:
            rank = retrieved_papers.index(gold_paper) + 1
            mrr5 += 1.0 / rank

    mrr5 /= len(results)
    return mrr5


def pre_process(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
        length_function=lambda x: len(model.encode(x, add_special_tokens=True)),
    )
    pairs = [
        (chunk, row[0])
        for row in tqdm(
            data[["cord_uid", "abstract"]].to_numpy(), desc="Creating text-uid pairs"
        )
        for chunk in splitter.split_text(row[1])
    ]

    docs = [
        Document(page_content=pair[0], metadata={"cord_uid": pair[1]}) for pair in pairs
    ]
    titles = [
        Document(page_content=row[1], metadata={"cord_uid": row[0]})
        for row in data[["cord_uid", "title"]].to_numpy()
    ]

    docs.extend(titles)
    return docs


def get_or_create_chunks():
    if os.path.exists("./docs.json"):
        print("Loading existing chunks", end="\n\n")
        with open("./docs.json", "r") as f:
            paper_chunks = json.load(f)
    else:
        print("Creating chunks from scratch\n")
        with open("../X_Data/subtask4b_collection_data.pkl", "rb") as f:
            data = pkl.load(f)
            collection = pd.DataFrame(data)

        docs = pre_process(collection)

        paper_chunks = [
            {
                "cord_uid": doc.metadata["cord_uid"],
                "text": doc.page_content,
            }
            for doc in docs
        ]

        with open("docs.json", "w") as f:
            json.dump(paper_chunks, f)
    return paper_chunks


paper_chunks = get_or_create_chunks()
paper_dict = {}
for entry in paper_chunks:
    uid = entry["cord_uid"]
    paper_dict.setdefault(uid, []).append(entry["text"])


df = pd.read_csv("../X_Data/subtask4b_query_tweets_train.tsv", sep="\t")
train_df = df[["tweet_text", "cord_uid"]].dropna()
dev_df = pd.read_csv("../X_Data/subtask4b_query_tweets_dev.tsv", sep="\t")


def get_or_create_paper_embeddings():
    if os.path.exists(f"./paper_embeddings_{model_name}.pt"):
        print("Loading existing paper embeddings\n")
        paper_embeddings = torch.load(f"./paper_embeddings_{model_name}.pt")
    else:
        print("Creating paper embeddings from scratch\n")
        paper_embeddings = []
        for uid, texts in tqdm(paper_dict.items()):
            # Embed the text
            embeddings = model.encode(texts, convert_to_tensor=True).cpu()
            for embedding, text in zip(embeddings, texts):
                paper_embeddings.append((uid, text, embedding))
            # Save the embeddings to a file
        torch.save(paper_embeddings, f"./paper_embeddings_{model_name}.pt")
    return paper_embeddings


def get_or_create_tweet_embeddings():
    if os.path.exists(f"./tweet_embeddings_{model_name}.pt"):
        print("Loading existing tweet embeddings\n")
        tweet_embeddings = torch.load(f"./tweet_embeddings_{model_name}.pt")
    else:
        print("Creating tweet embeddings from scratch\n")
        tweet_embeddings = []
        for uid, texts in tqdm(dev_df.iterrows(), desc="Creating tweet embeddings"):
            # Embed the text
            embedding = model.encode(texts["tweet_text"], convert_to_tensor=True).cpu()
            tweet_embeddings.append(
                (texts["post_id"], texts["cord_uid"], texts["tweet_text"], embedding)
            )
        # Save the embeddings to a file
        torch.save(tweet_embeddings, f"./tweet_embeddings_{model_name}.pt")
    return tweet_embeddings


paper_embeddings = get_or_create_paper_embeddings()
tweet_embeddings = get_or_create_tweet_embeddings()


# Calculate cosine similarity and rank the papers, each paper appears at most once
def calculate_similarity(tweet_embeddings, paper_embeddings):
    similarities = []
    for idx, (id, uid, _, tweet_embedding) in tqdm(
        enumerate(tweet_embeddings), desc="Tweets"
    ):
        scores = defaultdict(float)

        for chunk_idx, (paper_uid, _, chunk_emb) in tqdm(
            enumerate(paper_embeddings), desc="Chunks", leave=False
        ):
            sim = cosine_similarity(tweet_embedding, chunk_emb, dim=0).item()
            scores[paper_uid] = max(
                scores[paper_uid], sim
            )  # Keep best similarity per paper

        # Sort by similarity and get top_k papers
        top_papers = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        similarities.append(
            {
                "tweet": id,
                "gold_paper": uid,
                "retrieved": [uid for uid, score in top_papers],
            }
        )

    torch.save(similarities, f"./similarities_{model_name}.pt")

    return similarities


if os.path.exists(f"./similarities_{model_name}.pt"):
    print("\nLoading existing similarities\n")
    results = torch.load(f"./similarities_{model_name}.pt")
else:
    # Calculate cosine similarity and rank the papers
    print("\nCalculating similarities\n")
    results = calculate_similarity(tweet_embeddings, paper_embeddings)

print(results[:5])
mrr5 = calculate_mrr5(results)

print("MRR5:", mrr5)

with open(f"results_{model_name}.json", "w") as f:
    json.dump(results, f)

# Open the results csv as dataframe and write the mrr5
if not os.path.exists("results.csv"):
    df = pd.DataFrame(columns=["model", "mrr5"])
else:
    df = pd.read_csv("results.csv", usecols=["model", "mrr5"])

df = pd.concat(
    [
        df,
        pd.DataFrame.from_dict(
            {
                "model": [model_name],
                "mrr5": [mrr5],
            },
        ),
    ],
    ignore_index=True,
)

df.to_csv("results.csv", index=False)

with open(f"submission_{model_name}.json", "w") as f:
    formatted_results = [
        {
            "id": result["tweet"],
            "doc_ids": result["retrieved"],
        }
        for result in results
    ]
    json.dump(formatted_results, f)
