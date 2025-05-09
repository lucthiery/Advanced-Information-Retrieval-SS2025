### Current approach

#### Basic

- Abstracts split more or less semantically into < 256 token chunks
- Calculate Title and chunk embeddings
- Calculate tweet embeddings
- Calculate cosine distance between tweet and chunk
- Rank per paper
- Then use highest score per paper to rank papers

#### Hybrid

- Same approach as basic
- Uses postgres full text search in addition to vector search
- Cross ranks using Reciprocal Rank Fusion


| MODEL                               | MRR@5  | Tokenlimit |File Name     | comment                         |
|-------------------------------------|--------|------------|--------------|---------------------------------|
| SBERT                               | 0.5297 | 512        |              |no loss implementation for now   |
| text-embeddings-3-large             | 0.71   | 8192       |              |API-based, zero-shot             |
| granite-embedding-278m-multilingual | 0.5813 | 8192       |              |Multilingual, IBM Granite family |
| intfloat/multilingual-e5-small      | 0.5675 | 512        |              |Multilingual                     |

Furhter models:
- nomic-ai/nomic-embed-text-v1 - 8192 tokens
- thenlper/gte-large - 2048 tokens 

Next steps:

- Try models with high token limits
- Vary what is embedded: ie. full abstact, title + abstract, authors + title + abstract, etc.
- Analysis of failures to retrieve -> Why is the paper not identified or why is it ranked lower than others
- Try hybrid approaches, reranker + bm25 for example





