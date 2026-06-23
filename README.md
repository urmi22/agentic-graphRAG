# Agentic GraphRAG

A from-scratch comparison of three retrieval strategies — **naive vector RAG**, **GraphRAG**, and an **agentic LangGraph-based GraphRAG** — evaluated on [HotpotQA](https://hotpotqa.github.io/), a multi-hop QA dataset built specifically to require combining facts across multiple documents.

The goal isn't to prove graphs always win. It's to build all three honestly, on the same corpus, and show *exactly where and why* one approach succeeds where another fails.

## Why HotpotQA

HotpotQA questions come in two flavors:
- **comparison** — both entities are named in the question ("Were Scott Derrickson and Ed Wood the same nationality?"). Vector search handles these fine.
- **bridge** — the question never names the entity that connects the answer to the question ("What government position was held by the woman who portrayed Corliss Archer in *Kiss and Tell*?" — never says "Shirley Temple"). This is where naive retrieval structurally breaks, and where a knowledge graph's multi-hop traversal should earn its keep.

Each question ships with `supporting_facts` — the exact sentences a human used to answer it — so retrieval quality can be checked against ground truth, not just final-answer string matching.

## Case study: one bridge question, two outcomes

To make the comparison concrete rather than aggregate-metric abstract, here are two real runs against the same pipeline, on the same partially-built graph.

### Where the graph succeeds

> *"The director of the romantic comedy 'Big Stone Gap' is based in what New York city?"* → **Greenwich Village, New York City**

Vector search retrieves the *Big Stone Gap* film page itself, plus four irrelevant films that happen to mention "New York" — it never finds the *Adriana Trigiani* passage that actually answers the question, and gives up:

> System A (vector only): *"I don't know."*

The graph retriever links the query to the `big stone gap` node, walks one hop (`directed_by → adriana trigiani`), then a second hop (`based in → greenwich village, new york city`) — because the LLM extractor named the bridging person identically in both source passages, the chain stays connected end to end:

> GraphRAG (vector + graph): **"Greenwich Village, New York City."**

![Bridge resolved successfully via the graph](assets/bridge_success_diagram.png)

### Where the graph also fails — and why

> *"What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"* → **Chief of Protocol**

Here the bridging person was extracted under **two different surface forms** from her two source passages: `Shirley Temple` (from the film's page) and `Shirley Temple Black` (from her biography page). Plain `lower() + strip()` normalization doesn't catch that these are the same person, so they end up as two disconnected graph nodes. The walk reaches `shirley temple` and dead-ends — the fact that actually answers the question lives on the other, unreachable node. Both the vector-only and graph-augmented answers come back **"I don't know."**

![Why the graph path fails](assets/bridge_gap_diagram.png)

This is the central, recurring failure mode of any LLM-extracted knowledge graph: **entity resolution, not graph traversal, is the hard part.** Fixing it means clustering aliases (edit-distance / token-overlap, or a second LLM canonicalization pass) before the graph is built — not just at query time.

Reproduce either case yourself: [`scripts/demo_bridge_success.py`](scripts/demo_bridge_success.py) / [`scripts/demo_bridge_question.py`](scripts/demo_bridge_question.py).

## Architecture

```
HotpotQA (500 questions, distractor config)
        │
        ▼
  src/ingest.py  ──────────────►  data/corpus.jsonl   (4,937 deduped passages)
                                   data/questions.jsonl

        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
 System A: naive vector RAG                              System B: GraphRAG
 src/vector_store.py                                     src/extract_triples.py
   - bge-small-en-v1.5 embeddings                          - LLM-extracted (head, relation, tail)
   - FAISS IndexFlatIP (cosine)                             triples, batched + resumable
                                                           src/graph_store.py
                                                             - networkx.MultiDiGraph
                                                             - query → entity-linking → k-hop walk
                                                             - edge-level source-chunk collection
        └───────────────────────────────┬───────────────────────────────┘
                                         ▼
                              src/generate.py
                         (shared context-grounded prompt,
                          shared across all systems via LiteLLM)

 System C (planned): LangGraph agent that routes between vector/graph
 retrieval, grades retrieved context, and rewrites the query on a miss.
```

All three systems share the same corpus, the same embedding model, and the same generation prompt — only the retrieval step differs. That's deliberate: it isolates retrieval strategy as the only variable being compared.

## Status

| Phase | Component | Status |
|---|---|---|
| 0 | Config (`config.yaml`, `src/config.py`) | done |
| 1 | Ingest HotpotQA → corpus (`src/ingest.py`) | done — 4,937 passages from 500 questions |
| 2 | System A: vector index + retrieval (`src/vector_store.py`) | done |
| 3 | Triple extraction (`src/extract_triples.py`) | in progress — 350 / 4,937 chunks extracted |
| 3 | Graph construction + k-hop retrieval (`src/graph_store.py`) | done, validated on the partial graph above |
| 4 | System C: LangGraph agentic router/grader/rewriter | not started |
| 5 | Evaluation harness (ragas) across all three systems | not started |

Triple extraction is bottlenecked by free-tier LLM rate limits (Groq: 100k tokens/day; Gemini: 20 requests/day on the current API key) rather than anything algorithmic — it resumes safely across runs via an on-disk JSONL ledger keyed by `chunk_id`, so it's just a matter of letting it run across multiple days, or switching to a paid tier.

## Setup

```bash
conda create -n agentic-graphrag python=3.11
conda activate agentic-graphrag
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY / GROQ_API_KEY
```

## Running the pipeline

```bash
python -m src.ingest              # build data/corpus.jsonl + data/questions.jsonl
python -m src.vector_store        # build + cache the FAISS index (System A)
python -m src.extract_triples     # LLM-extract triples → data/triples.jsonl (resumable)
python -m src.graph_store         # build the graph from cached triples (System B)
python -m scripts.demo_bridge_success    # walk + visualize a working bridge case
python -m scripts.demo_bridge_question   # walk + visualize the entity-resolution failure case
```

Each script is independently resumable and reads from cached `data/*.jsonl` / `data/*.pkl` artifacts where available, so re-running the pipeline doesn't redo expensive embedding or LLM-extraction work.

## Repository layout

```
src/
  config.py          # loads config.yaml
  ingest.py           # HotpotQA → corpus.jsonl / questions.jsonl
  vector_store.py      # FAISS-backed System A retriever
  generate.py          # shared context-grounded generation prompt (LiteLLM)
  extract_triples.py   # batched, resumable LLM triple extraction
  graph_store.py        # graph construction + k-hop GraphRetriever (System B)
  llm.py               # thin LiteLLM wrapper (model-agnostic completion)
scripts/
  demo_bridge_success.py   # case study: graph traversal succeeds
  demo_bridge_question.py  # case study: graph traversal fails (entity-alias gap)
config.yaml          # models, embedding model, top_k, k_hops, dataset params
```

`data/` (HotpotQA cache, FAISS index, extracted triples, graph pickle) is gitignored — regenerate it locally via the commands above.
