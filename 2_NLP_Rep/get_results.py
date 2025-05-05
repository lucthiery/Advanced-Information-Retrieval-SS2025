import pandas as pd
import json
import os
import torch
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 300
TOP_K = 75
DATA_DIR = "../X_Data"
OUTPUT_DIR = "reranker_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def embed_texts(texts, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i : i + batch_size]
        output = model.encode(batch, convert_to_tensor=True).cpu()
        embeddings.extend(output)
    return embeddings

def calculate_mrr5(results):
    mrr5 = 0.0
    for result in results:
        gold_paper = result["gold_paper"]
        retrieved_papers = result["retrieved"]
        if gold_paper in retrieved_papers:
            rank = retrieved_papers.index(gold_paper) + 1
            mrr5 += 1.0 / rank
    return mrr5 / len(results)

def pre_process(data, model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
        length_function=lambda x: len(model.encode(x, add_special_tokens=True)),
    )
    pairs = []
    for row in tqdm(data.to_numpy(), desc="Creating text-uid pairs"):
        uid, abstract = row
        if isinstance(abstract, str):
            for chunk in splitter.split_text(abstract):
                pairs.append((chunk, uid))
    return pairs

def run_for_model(model_name, split):
    print(f"\nRunning {model_name} on {split}")
    model = SentenceTransformer(model_name)

    with open(os.path.join(DATA_DIR, "subtask4b_collection_data.pkl"), "rb") as f:
        corpus = pkl.load(f)

    query_file = os.path.join(DATA_DIR, f"subtask4b_query_tweets_{split}.tsv")
    queries = {}
    with open(query_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]

    corpus_df = pd.DataFrame(list(corpus.items()), columns=["cord_uid", "abstract"])
    pairs = pre_process(corpus_df, model)
    texts = [x[0] for x in pairs]
    uids = [x[1] for x in pairs]

    embeddings = embed_texts(texts, model)

    grouped_embeddings = defaultdict(list)
    for uid, emb in zip(uids, embeddings):
        grouped_embeddings[uid].append(emb)

    corpus_embeddings = {
        uid: torch.stack(emb_list).mean(dim=0) for uid, emb_list in grouped_embeddings.items()
    }

    query_embeddings = {
        qid: model.encode(query, convert_to_tensor=True).cpu()
        for qid, query in queries.items()
    }

    results = []
    for qid, q_emb in tqdm(query_embeddings.items(), desc="Calculating similarities"):
        sims = {
            uid: cosine_similarity(q_emb, c_emb, dim=0).item()
            for uid, c_emb in corpus_embeddings.items()
        }
        top75 = sorted(sims, key=sims.get, reverse=True)[:TOP_K]
        results.append({
            "query": qid,
            "gold_paper": qid,
            "retrieved": top75
        })

    mrr5 = calculate_mrr5(results)
    print(f"MRR@5: {mrr5:.4f}")

    # Save results
    safe_model_name = model_name.replace("/", "_")
    json_path = os.path.join(OUTPUT_DIR, f"{safe_model_name}_{split}_top75.json")
    csv_path = os.path.join(OUTPUT_DIR, f"{safe_model_name}_{split}_top75.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(results).to_csv(csv_path, index=False)

if __name__ == "__main__":
    models = [
        "intfloat/multilingual-e5-small",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    splits = ["train", "dev"]

    for model in models:
        for split in splits:
            run_for_model(model, split)
