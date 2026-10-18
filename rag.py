import json
import os
import uuid

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models
from langsmith import traceable
import streamlit as st

# ---------------------------------------------------------------- models
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
)
sparse_model = SparseTextEmbedding("Qdrant/bm25")
reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------- config
CORPUS_FILE = "hotpot_corpus.jsonl"
EVAL_FILE = "hotpot_eval.json"

N_EVAL = 500
N_UNANS = 100

COLLECTION = "hotpot"

CANDIDATES_K = 20   
FINAL_K = 5         


# ----------------------------------------------------------- data loading
def download_and_save():
    print("Downloading HotpotQA (first run only)...")
    train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    dev = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    paras = {}
    for ds in (train, dev):
        for ctx in ds["context"]:
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                if title not in paras:
                    paras[title] = " ".join(s.strip() for s in sents).strip()

    # Build the test questions
    eval_qs = dev.select(range(N_EVAL))
    unans_gold = set()
    for ex in eval_qs.select(range(N_UNANS)):
        unans_gold.update(ex["supporting_facts"]["title"])

    eval_rows = []
    for i, ex in enumerate(eval_qs):
        gold = sorted(set(ex["supporting_facts"]["title"]))
        unanswerable = i < N_UNANS
        if not unanswerable and set(gold) & unans_gold:
            continue
        eval_rows.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_titles": gold,
            "unanswerable": unanswerable,
        })

    # Remove gold paragraphs of the unanswerable questions from the corpus
    for t in unans_gold:
        paras.pop(t, None)

    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for title, text in paras.items():
            f.write(json.dumps({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, title)),   
                "page_content": f"{title}: {text}",                 
                "metadata": {"title": title, "text": text},
            }, ensure_ascii=False) + "\n")
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2, ensure_ascii=False)


def load_documents():
    """Read from the saved file if it exists, otherwise download and build it first."""
    if not os.path.exists(CORPUS_FILE):
        download_and_save()
    docs = []
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs.append(Document(id=r["id"], page_content=r["page_content"], metadata=r["metadata"]))
    return docs


# ---------------------------------------------------------- vector database
_client = None

def get_client():
    global _client
    if _client is None:
        try:
            url = st.secrets["QDRANT_URL"]
            key = st.secrets["QDRANT_API_KEY"]
        except Exception:
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            key = os.getenv("QDRANT_API_KEY")
        _client = QdrantClient(url=url.rstrip("/"), api_key=key, timeout=120)
    return _client

# ---------------------------------------------------------- retrieval
@traceable(name="hybrid_search", run_type="retriever")
def hybrid_search(query, k=CANDIDATES_K):
    """Dense + sparse search inside Qdrant, merged with Reciprocal Rank Fusion (RRF)."""
    client = get_client()

    dense_q = embedding_model.embed_query(query)
    sparse_q = next(iter(sparse_model.query_embed(query)))

    result = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_q, using="dense", limit=k),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_q.indices.tolist(),
                    values=sparse_q.values.tolist(),
                ),
                using="sparse",
                limit=k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return result.points


@traceable(name="rerank", run_type="tool")
def rerank(query, points, top_k=FINAL_K):
    """Re-score candidates with a cross-encoder and keep the best top_k."""
    if not points:
        return []

    texts = [f"{p.payload['title']}: {p.payload['text']}" for p in points]
    scores = list(reranker.rerank(query, texts))

    ranked = sorted(zip(points, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        Document(
            id=str(p.id),
            page_content=f"{p.payload['title']}: {p.payload['text']}",
            metadata={
                "title": p.payload["title"],
                "text": p.payload["text"],
                "rerank_score": float(s),
            },
        )
        for p, s in ranked
    ]


@traceable(name="retrieve", run_type="retriever")
def retrieve(query, top_k=FINAL_K, candidates=CANDIDATES_K):
    """Full pipeline: hybrid search -> re-rank. Returns a list of LangChain Documents."""
    return rerank(query, hybrid_search(query, candidates), top_k)