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

# ---------------------------------------------------------------- models
# Must be the same model that was used on Kaggle to create the stored vectors.
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

QDRANT_URL = "http://localhost:6333"   # Qdrant server running in Docker
COLLECTION = "hotpot"

CANDIDATES_K = 20   # how many docs hybrid search returns before re-ranking
FINAL_K = 5         # how many docs are kept after re-ranking


# ---------------------------------------------------------------- data loading
# NOTE: the vector database is already built (507,294 points).
# These two functions are not used by the app. Do not call them.
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
        # Skip answerable questions whose gold paragraphs will be removed from the corpus
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

    # Save to disk
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for title, text in paras.items():
            f.write(json.dumps({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, title)),   # stable id
                "page_content": f"{title}: {text}",                 # title prefix is embedded with the text
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
    """One shared client that talks to the Qdrant server running in Docker."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, prefer_grpc=True, timeout=120)
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