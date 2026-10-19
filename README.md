# Veritas · Self-RAG
[![Lint](https://github.com/rishavmodi17/veritas/actions/workflows/lint.yml/badge.svg)](https://github.com/rishavmodi17/veritas/actions/workflows/lint.yml)
A self-correcting RAG system over HotpotQA that routes each question, grades its own evidence, verifies its own answers, and says so when it does not know.

🔗 **Live App:** https://veritas-9nppirqgaitroxnrqm6fbs.streamlit.app

![Veritas Screenshot](assets/screenshot.png)

---

## Overview

Veritas is a self-reflective RAG pipeline. Instead of running a fixed `Query → Retrieve → Generate` chain, it checks its own work at every stage and decides what to do next.

For each question, Veritas can:

- Filter unsafe or manipulative input before anything else runs
- Decide whether the question needs retrieval or can be answered directly
- Search a private knowledge base of about 507k Wikipedia paragraphs using hybrid dense + sparse retrieval
- Re-score the candidates with a cross-encoder reranker
- Grade every retrieved passage for relevance
- Fall back to live web search when the local documents are not enough
- Generate an answer using only the approved context
- Verify that the answer is fully supported by that context, and revise it if it is not
- Check that the answer actually addresses the question
- Rewrite the question and retry once, with an optional human-in-the-loop pause
- Refuse to answer when no reliable evidence is found

The result is a research workflow that separates routing, retrieval, grading, generation, verification and abstention, with every decision visible in the interface.

---

## Workflow

```mermaid
flowchart TD
    S([START]) --> input_guard
    input_guard -->|blocked| blocked_response --> E([END])
    input_guard -->|allowed| decide_retrieval
    decide_retrieval -->|direct| generate_direct --> E
    decide_retrieval -->|retrieve| retrieve
    retrieve --> is_relevant
    is_relevant -->|has relevant docs| generate_from_context
    is_relevant -->|no relevant docs| tavily_search
    tavily_search -->|first search| is_relevant
    tavily_search -->|already tried| no_answer
    generate_from_context --> is_supported
    is_supported -->|supported| is_answer_relevant
    is_supported -->|"unsupported (max 3 revisions)"| revise_answer --> is_supported
    is_answer_relevant -->|relevant| E
    is_answer_relevant -->|"not relevant (first time)"| human_review
    is_answer_relevant -->|already rewrote| no_answer
    human_review -->|approve| rewrite_question --> retrieve
    human_review -->|reject| E
    no_answer --> E
```

---

## Graph Architecture

Veritas is a compiled **LangGraph** `StateGraph` with 14 nodes. Each node has one focused responsibility.

### 1. Input Guard
Rejects empty input, input over 1000 characters, prompt-injection attempts and harmful requests. Factual questions about sensitive topics (history, crime, medicine asked in an encyclopedic way) are allowed. When unsure, it chooses *safe*.

### 2. Retrieval Router
Decides whether the question needs the knowledge base. Specific facts, named entities, comparisons and multi-step lookups go to retrieval. Greetings, arithmetic, writing tasks and widely known stable facts are answered directly.

### 3. Retrieve
Runs hybrid search and reranking (see [Hybrid Retrieval](#hybrid-retrieval)).

### 4. Relevance Grading
Every retrieved passage is graded in parallel by an LLM. A passage is kept if the grader marks it relevant, or if its cross-encoder score is 4.0 or higher. The grader is deliberately lenient for multi-hop questions: a passage that covers one of the entities, or one step of the reasoning, counts as relevant.

### 5. Web Fallback
If no passage is relevant, Tavily searches the web (up to 5 results). This happens at most once per question, and the results go through the same relevance grading.

### 6. Grounded Generation
The answer is generated using only the approved context. The model returns the answer and a flag saying whether the context was sufficient.

### 7. Answer Verification and Revision
A strict verifier checks whether the entire answer is supported by the context. If not, the answer is revised and verified again, up to 3 times.

### 8. Answer Relevance Check
Confirms the answer addresses the question that was asked, not a different one. An answer based on insufficient context fails this check.

### 9. Human Review and Question Rewriting
When the answer fails the relevance check for the first time, the graph can pause and ask the user to approve an automatic rewrite, retry with their own question, or stop. With human review switched off, the rewrite is approved automatically. The rewritten question goes through retrieval again, and this happens at most once.

### 10. No Answer
If every route fails, Veritas says that it could not find enough reliable information. It does not guess.

---

## Hybrid Retrieval

Veritas does not rely on vector similarity alone. The knowledge base uses three stages:

1. **Dense retrieval** with `BAAI/bge-base-en-v1.5` (768 dimensions) for semantic similarity
2. **Sparse retrieval** with `Qdrant/bm25` for exact terms, names and acronyms
3. **Reciprocal Rank Fusion (RRF)**, performed inside Qdrant, to merge both rankings
4. **Cross-encoder reranking** with `Xenova/ms-marco-MiniLM-L-6-v2` to pick the best passages

```
                 User Query
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Dense Retrieval        Sparse Retrieval
   (bge-base-en-v1.5)        (BM25)
          │                     │
          ▼                     ▼
     20 candidates         20 candidates
          │                     │
          └──────────┬──────────┘
                     ▼
          Reciprocal Rank Fusion
              (inside Qdrant)
                     │
                     ▼
             20 fused candidates
                     │
                     ▼
           Cross-Encoder Reranker
                     │
                     ▼
                Top 5 passages
```

### Why hybrid retrieval?

Dense retrieval understands meaning but can miss exact names, titles and rare terms. BM25 matches those exactly but misses paraphrases. HotpotQA questions mix both: they name specific people, films and places, and they connect them in loosely worded ways. Fusing both rankings handles each case.

RRF works on ranks, not scores, so the two incompatible scoring systems never need to be normalized.

### Why a reranker?

Fusion returns candidates that are individually similar to the query. A cross-encoder reads the query and each passage together, which gives a much sharper relevance score. Veritas retrieves 20 candidates, then keeps the top 5.

---

## Knowledge Base

The knowledge base is built from [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) (distractor setting):

- Paragraphs from the train and validation splits, de-duplicated by title
- Each paragraph is embedded as `Title: text`, so the title is part of the searchable content
- **507,294 paragraphs** in the Qdrant collection `hotpot`
- Every point stores a named dense vector (`dense`) and a named sparse vector (`sparse`, IDF modifier)
- The payload holds the title and the full paragraph text

### Built-in unanswerable questions

The supporting paragraphs of the first 100 validation questions were removed from the corpus on purpose. This means a set of questions exists that Veritas *cannot* answer from its documents, which makes it possible to demonstrate its refusal behavior.

The Playground also ships with example buttons for a multi-hop question, a direct question and a prompt-injection guard test.

---

## Persistent Vector Store

Veritas uses **Qdrant** as its vector database: Docker locally, Qdrant Cloud in production.

The index is built once and reused. The app never rebuilds the corpus on startup, so deploying only needs a connection to an existing collection.

On the Qdrant Cloud free tier (1 GiB RAM, 4 GiB disk), the collection is stored with:

- Vectors on disk
- Binary quantization kept in RAM
- Payloads on disk

This keeps memory use far below the size of the full vector data.

---

## Research Pipeline

A complete question follows this workflow:

```
1. User submits a question
              │
              ▼
2. Input guard checks the text
              │
              ▼
3. Router chooses: direct answer or retrieval
              │
              ▼
4. Hybrid search + reranker return the top 5 passages
              │
              ▼
5. Each passage is graded for relevance
              │
        ┌─────┴─────┐
        │           │
     relevant    none relevant
        │           │
        │           ▼
        │     Tavily web search (once)
        │           │
        └─────┬─────┘
              ▼
6. Answer is generated from approved context only
              │
              ▼
7. Verifier checks the answer is fully supported
              │
              ▼
8. Relevance check confirms it answers the question
              │
              ▼
9. Answer streams to the interface with its sources
```

If any stage fails, Veritas revises, rewrites the question, or declines to answer.

---

## User Interface

The Streamlit app has six pages:

| Page | What it shows |
|---|---|
| **Playground** | Chat with the live LangGraph, streaming answers, source cards, human-review panel |
| **Sources** | Documents used in the answer, retrieved, graded relevant, and returned by Tavily |
| **Execution Trace** | Per-node timings, every decision, and the state updates from the last run |
| **Architecture** | The workflow graph, with the path of the last run highlighted |
| **Tech Stack** | The technologies behind the project |
| **About** | Summary and feature table |

The sidebar shows a live Qdrant status, a Human-in-the-loop toggle, and a Clear chat button.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (loops, interrupts, checkpointing) |
| LLM framework | LangChain, Pydantic structured output |
| Fast LLM | Groq `openai/gpt-oss-20b` |
| Large LLM | Google Gemini `gemini-2.5-flash` |
| Dense embeddings | HuggingFace `BAAI/bge-base-en-v1.5` |
| Sparse retrieval | `Qdrant/bm25` via FastEmbed |
| Retrieval fusion | Reciprocal Rank Fusion (server-side in Qdrant) |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder |
| Vector database | Qdrant (Docker locally, Qdrant Cloud in production) |
| Web search | Tavily |
| Tracing | LangSmith (optional) |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
veritas/
│
├── app.py
│   └── Streamlit interface (Playground, Sources, Trace, Architecture, Tech Stack, About)
│
├── main.py
│   └── LangGraph workflow
│       ├── State and Pydantic schemas
│       ├── Graph nodes
│       ├── Routing functions
│       └── Graph compilation and checkpointer
│
├── prompts.py
│   └── Prompts for guard, routing, grading, generation,
│       verification, revision, relevance and rewriting
│
├── rag.py
│   └── Retrieval
│       ├── Embedding models
│       ├── Hybrid search (dense + sparse, RRF)
│       └── Cross-encoder reranking
│
├── assets/
│   └── screenshot.png
│
├── .streamlit/
│   └── config.toml
│
└── requirements.txt
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/rishavmodi17/veritas.git
cd veritas
```

### 2. Create a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Qdrant

Run Qdrant in Docker, or use a Qdrant Cloud cluster, and make sure the `hotpot` collection is loaded.

```bash
docker run -p 6333:6333 -p 6334:6334 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

# Optional tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=veritas
```

Leave `QDRANT_API_KEY` empty for a local Docker instance. For Qdrant Cloud, use the cluster URL with port `6333` and your API key.

`GOOGLE_API_KEY`, `GROQ_API_KEY` and `TAVILY_API_KEY` are required.

### 6. Run Veritas

```bash
streamlit run app.py
```

On the first run, the embedding, sparse and reranker models are downloaded and loaded, so the first question takes longer than the rest.

### Deploying on Streamlit Community Cloud

Add the same keys under **Settings → Secrets** in TOML format, using the Qdrant Cloud URL and API key. Never commit `.env` or any key to GitHub.

---

## Design Principles

### 1. Check Before You Trust
No step is assumed to be correct. Retrieved passages are graded, answers are verified, and the final answer is checked against the question.

### 2. Route Before Retrieve
Not every question needs a knowledge-base search. Small talk, arithmetic and stable general facts are answered directly.

### 3. Combine Semantic and Lexical Retrieval
Dense vectors catch meaning, BM25 catches exact terms, and RRF merges both without score normalization.

### 4. Rerank for Precision
A cross-encoder gives a sharper relevance judgment on a small candidate set than any embedding similarity can.

### 5. Prefer Silence to Fabrication
When the evidence is not there, Veritas declines to answer instead of inventing one.

### 6. Keep a Human in the Loop
The graph can pause at the point where it is most likely to go wrong and let the person decide.

### 7. Make Every Decision Visible
Every run shows its path, timings, decisions and sources, so the reasoning can be inspected.

### 8. Separate Fast Checks From Slow Writing
Cheap, frequent checks use a small fast model. Answer writing uses a larger one.

---

## Current Limitations

Veritas is a working demonstration with some known limits:

- **Narrow knowledge base:** the corpus is Wikipedia text from HotpotQA, so topics such as anime, current events and how-to questions are not covered and often end in a refusal.
- **Conditional web fallback:** Tavily runs only when no local passage is graded relevant. If loosely related passages pass grading but the answer fails verification, Veritas refuses without trying the web.
- **Vague multi-hop questions:** questions that name no entity, such as "this first round draft pick", give search nothing to match.
- **Factual questions only:** opinion and hypothetical questions, such as "who would win", have no factual answer and are refused.
- **Single retry:** the question is rewritten once, and the web is searched once.
- **Fixed fusion:** dense and sparse results are merged with standard RRF and no tuned weighting.
- **Free-tier speed:** a full failing run (retrieve, grade, generate, verify, rewrite, retry) can take 15 to 20 seconds.
- **Short-term state:** graph state is held in memory per run and is not saved between sessions.

---

## Future Improvements

- Second-hop retrieval to find the missing paragraph in multi-hop questions
- Web search as a last resort when answer verification fails
- Persistent conversation memory and saved research history

---

## Project Goal

Veritas explores how RAG can be made self-aware. Instead of a fixed pipeline:

```
Query → Retrieve → Generate
```

it uses a loop that checks itself:

```
Query
  ↓
Guard
  ↓
Route
  ↓
Retrieve + Rerank
  ↓
Grade
  ↓
Generate
  ↓
Verify
  ↓
Check relevance
  ↓
Answer, Rewrite, or Abstain
```

The goal is a system that knows when it has good evidence, can tell when it does not, and says so instead of guessing.

Based on the Self-RAG pattern from [Asai et al. (2023)](https://arxiv.org/abs/2310.11511).

---

## Author

Built by Rishav as an engineering project exploring self-correcting RAG, hybrid retrieval, and agentic workflows with LangGraph.

LinkedIn: https://www.linkedin.com/in/rishav-modi-738524412/

GitHub: https://github.com/rishavmodi17

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
