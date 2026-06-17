# Agentic GraphRAG with a Real Evaluation Harness — Build Plan

A complete, follow-along plan for a GitHub portfolio project that fuses RAG + knowledge graphs + an agentic router + a rigorous evaluation harness. Designed to run end-to-end on Kaggle / Google Colab free tier.

**The point of this repo:** anyone can build naive RAG. This project proves you can (a) build a graph-aware retrieval system, (b) wrap it in an agent that *decides* how to retrieve, and (c) measure all of it like an engineer, not just demo it. The headline deliverable is a **three-way benchmark table** (naive RAG vs GraphRAG vs agentic GraphRAG) with quality *and* latency numbers, plus a live demo.

---

## 0. What you're building (mental model)

You build three retrieval systems on the same corpus and the same questions, then benchmark them head-to-head:

| System | Retrieval strategy | What it proves |
|---|---|---|
| **A. Naive RAG** | Vector search only (top-k chunks) | Baseline |
| **B. GraphRAG** | Vector search + knowledge-graph traversal, always both | Graph adds multi-hop reasoning |
| **C. Agentic GraphRAG** | A LangGraph agent *routes* between vector / graph / hybrid, grades retrieved context, and re-retrieves when confidence is low | You can build adaptive, self-correcting systems |

The story your benchmark should tell: **B beats A on multi-hop questions; C matches B on quality while avoiding wasted retrieval, and recovers from bad first-pass retrieval via its self-correction loop.**

---

## 1. Tech stack (all free-tier friendly)

- **Orchestration:** LangGraph (the agent), LangChain (loaders/utilities)
- **LLM (generation + extraction):** Gemini 2.5 Flash (1,500 req/day free) as primary; Groq Llama 3.3 70B as fallback. Access both through **LiteLLM** so the model is a one-line config swap.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (fast, 33M params) for the baseline; optionally upgrade to `Alibaba-NLP/gte-base-en-v1.5` or `google/embeddinggemma-300m`. All run on free Colab/Kaggle GPU.
- **Vector store:** FAISS (in-memory, zero infra). Weaviate Embedded as an optional upgrade.
- **Graph store:** `networkx` in-memory to start (free, simple). Neo4j Aura Free tier as the "look, I can use a real graph DB" upgrade.
- **Evaluation:** RAGAS for the four canonical metrics; HotpotQA gold answers for Exact-Match / F1; `matplotlib` for charts; Weights & Biases (free) to log runs.
- **Serving:** FastAPI (API) + Gradio (UI) → deploy on a Hugging Face Space (free).

Install:
```bash
pip install langgraph langchain langchain-community litellm \
  sentence-transformers faiss-cpu networkx datasets \
  ragas matplotlib gradio fastapi uvicorn wandb python-dotenv
```

---

## 2. Dataset

**Use HotpotQA.** It is purpose-built for multi-hop QA, which is exactly where graph traversal beats flat vector search — so it lets your benchmark *show* the graph advantage instead of just asserting it. Critically, it ships with:
- gold short answers (for Exact-Match / F1),
- gold "supporting facts" (sentence-level relevance labels — free retrieval ground truth),
- 2–10 context paragraphs per question (a ready-made small corpus).

```python
from datasets import load_dataset
ds = load_dataset("hotpot_qa", "distractor", split="validation")
# Take a working subset for free-tier compute
subset = ds.select(range(500))   # 500 questions is plenty for a portfolio benchmark
```

Build your corpus from the union of all context paragraphs across the subset (dedup by title). You'll get a few thousand passages — small enough for free tier, large enough to be credible.

**Alternatives if you want a different flavor:** 2WikiMultiHopQA or MuSiQue (also multi-hop with reasoning paths), or arXiv/PubMed abstracts if you'd rather a domain corpus (but then you lose free gold labels and must build a golden set yourself — see §6).

---

## 3. Phase-by-phase build

### Phase 0 — Repo + config (Day 1)
Set up the structure (see §8), a `.env` for API keys, and a `config.yaml` holding model names, `top_k`, chunk size, etc. Get a Gemini API key from Google AI Studio and a Groq key. Wire LiteLLM:

```python
# llm.py
import litellm, os
def llm(prompt, model="gemini/gemini-2.5-flash", temperature=0.0):
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content
```
Keep `temperature=0` everywhere in the pipeline so benchmarks are reproducible.

### Phase 1 — Ingest + chunk (Day 1–2)
Flatten HotpotQA context into passages. These are already paragraph-sized, so chunking is light: keep each paragraph as a chunk, attach metadata `{title, para_id}`. Store as a list of `{id, text, title}` dicts — this same list feeds both the vector index and the graph builder.

### Phase 2 — Vector index = System A (Day 2)
This is your baseline naive RAG.
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
emb = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(np.array(emb))

def vector_search(query, k=5):
    q = embedder.encode([query], normalize_embeddings=True)
    scores, idx = index.search(np.array(q), k)
    return [(chunks[i], float(s)) for i, s in zip(idx[0], scores[0])]
```
Generation: stuff top-k chunks into a prompt → `llm(...)`. That's System A complete.

### Phase 3 — Build the knowledge graph (Day 3–5)
Extract `(head, relation, tail)` triples from each passage with the free LLM, then load them into networkx. Prompt the model for strict JSON:

```python
EXTRACT = """Extract factual relationships from the text as JSON triples.
Return ONLY a JSON list of objects: [{{"head": ..., "relation": ..., "tail": ...}}].
Use canonical entity names. Text:
{text}"""

import json, networkx as nx
G = nx.MultiDiGraph()
for c in chunks:
    try:
        triples = json.loads(llm(EXTRACT.format(text=c["text"])))
        for t in triples:
            G.add_edge(t["head"], t["tail"], relation=t["relation"], source=c["id"])
    except json.JSONDecodeError:
        continue   # log and skip malformed extractions
```
Practical notes:
- This is the most LLM-call-heavy step. With 1,500 Gemini req/day, a few thousand passages may span 2–3 days, or batch several passages per call. Cache every extraction to disk (`jsonl`) so you never re-run.
- Normalize entities (lowercase, strip) and optionally merge near-duplicates with embedding similarity to avoid graph fragmentation.
- Keep a map from each graph node → source chunk ids, so graph hits can pull the underlying text for generation.

**Graph retrieval:** given a query, (1) link query entities to graph nodes (embed node names, nearest-neighbor match), (2) pull the k-hop subgraph around them, (3) collect the source passages of those edges as context. This is what lets multi-hop questions connect facts that live in *different* passages.

### Phase 4 — GraphRAG = System B (Day 5–6)
System B = vector results **+** graph-subgraph results, concatenated (dedup), then generate. No agent yet. This isolates the value of the graph itself before you add adaptivity.

### Phase 5 — The agent = System C (Day 6–9)
Build a LangGraph state machine. This is the centerpiece — describe it clearly in your README with a diagram.

Nodes:
1. **Router** — LLM classifies the query: `vector` (simple factual lookup), `graph` (relational / multi-hop), or `hybrid`. Output constrained to one token.
2. **Retrieve** — runs the chosen retriever(s).
3. **Grade** — LLM scores whether retrieved context is sufficient/relevant (`yes`/`no` + brief reason).
4. **Rewrite** — if grade is `no`, the LLM rewrites the query (decomposes multi-hop questions, adds entities) and loops back to Retrieve. Cap at 2 retries to bound cost.
5. **Generate** — produce the final grounded answer with citations to source chunk ids.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class S(TypedDict):
    question: str
    route: str
    context: List[dict]
    grade: str
    tries: int
    answer: str

def route(s):  s["route"] = classify(s["question"]); return s
def retrieve(s):
    s["context"] = run_retriever(s["route"], s["question"]); return s
def grade(s):  s["grade"] = grade_context(s["question"], s["context"]); return s
def rewrite(s):
    s["question"] = rewrite_query(s["question"]); s["tries"] += 1; return s
def generate(s):
    s["answer"] = answer_with_context(s["question"], s["context"]); return s

g = StateGraph(S)
for n, f in [("route",route),("retrieve",retrieve),("grade",grade),
             ("rewrite",rewrite),("generate",generate)]:
    g.add_node(n, f)
g.set_entry_point("route")
g.add_edge("route", "retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade",
    lambda s: "generate" if s["grade"]=="yes" or s["tries"]>=2 else "rewrite",
    {"generate":"generate", "rewrite":"rewrite"})
g.add_edge("rewrite", "retrieve")
g.add_edge("generate", END)
agent = g.compile()
```
That's System C. The self-correction loop (grade → rewrite → re-retrieve) is the behavior that makes this "agentic" rather than a fixed pipeline — make sure your eval includes questions where the first retrieval fails so the loop earns its keep.

---

## 4. Phase 6 — Evaluation harness (Day 9–11)

Run all three systems over the same question set and score two layers:

**Retrieval + generation quality (RAGAS, LLM-as-judge):**
- `context_precision`, `context_recall` — did retrieval fetch the right stuff?
- `faithfulness` — is the answer grounded in the retrieved context (no hallucination)?
- `answer_relevancy` — does the answer address the question?

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_ds = Dataset.from_dict({
    "question":      questions,
    "answer":        system_answers,
    "contexts":      retrieved_contexts,   # list[list[str]]
    "ground_truth":  gold_answers,         # HotpotQA gold
})
report = evaluate(eval_ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```
Configure RAGAS to use your free Gemini/Groq model as the judge (pass a LangChain-wrapped LLM + your embedder). Track the LLM-judge cost mentally — a few-hundred-question run is well within free limits if you throttle to the RPM cap.

**Answer correctness (deterministic, no LLM):** HotpotQA has gold short answers, so also compute classic **Exact Match** and **token-level F1** — these are cheap, reproducible, and reviewers trust them. Implement the standard SQuAD-style normalization (lowercase, strip punctuation/articles).

**Why both:** RAGAS catches hallucination and retrieval quality; EM/F1 gives a hard, judge-free correctness number. Reporting both signals evaluation maturity.

---

## 5. Phase 7 — Benchmarking (Day 11–12)

Run the full matrix and produce the table that anchors the whole repo:

| Metric | A: Naive RAG | B: GraphRAG | C: Agentic GraphRAG |
|---|---|---|---|
| Exact Match | | | |
| F1 | | | |
| Faithfulness | | | |
| Answer relevancy | | | |
| Context precision | | | |
| Context recall | | | |
| Avg latency (s/query) | | | |
| Avg LLM calls/query | | | |

Then **slice by question type** (single-hop vs multi-hop — HotpotQA labels this). The multi-hop slice is where B and C should pull ahead of A; show that explicitly with a grouped bar chart. Also log latency and calls-per-query: C will use more LLM calls when it self-corrects, and that honest tradeoff (quality vs cost) is exactly the kind of analysis hiring managers want to see.

Save raw per-question results to `results/*.jsonl` and commit them — reproducibility is part of the signal.

---

## 6. If you swap HotpotQA for a custom corpus

You lose free gold labels, so generate a **golden evaluation set**: use RAGAS's synthetic test-set generator (or hand-write 50–100 Q/A pairs with reference answers). Keep it small but clean; quality of the golden set matters more than size.

---

## 7. Phase 8 — Serve + demo (Day 12–14)

- **FastAPI** endpoint `POST /ask` → runs the agent, returns `{answer, route_taken, retries, sources}`. Exposing the route and retry count makes the agent's decisions visible (great for the demo).
- **Gradio** UI: a question box, the answer, and a panel showing which retriever fired, how many self-correction loops ran, and the source passages.
- Deploy the Gradio app to a **free Hugging Face Space**. Put the live link at the top of your README — recruiters click it.

---

## 8. Repo structure

```
agentic-graphrag/
├── README.md                  # results table + architecture diagram up top
├── requirements.txt
├── config.yaml
├── .env.example
├── src/
│   ├── llm.py                 # LiteLLM wrapper
│   ├── ingest.py              # load + chunk HotpotQA
│   ├── vector_store.py        # FAISS index (System A)
│   ├── graph_store.py         # triple extraction + networkx (System B)
│   ├── agent.py               # LangGraph agent (System C)
│   ├── retrievers.py          # vector / graph / hybrid
│   └── generate.py            # grounded answer prompts
├── eval/
│   ├── run_benchmark.py       # runs A/B/C over the set
│   ├── ragas_eval.py
│   ├── em_f1.py
│   └── plots.py
├── results/                   # committed jsonl + charts
├── app/
│   ├── api.py                 # FastAPI
│   └── gradio_app.py          # demo UI
└── notebooks/
    └── 01_explore.ipynb
```

---

## 9. README skeleton (this is what gets you the call)

1. **One-line pitch** + live demo badge/link.
2. **Architecture diagram** (LangGraph flow: route → retrieve → grade → rewrite/generate). Draw it; don't describe it in prose.
3. **Results table** (the §5 matrix) — above the fold.
4. **Key finding** in one sentence (e.g. "GraphRAG improves multi-hop F1 by X points; the agent matches it while cutting wasted retrieval on simple queries").
5. Quickstart (install + run in <5 commands).
6. How each system works (short).
7. Evaluation methodology (metrics, dataset, judge model).
8. Limitations + next steps.

Goal: a reviewer understands the problem, your approach, and the outcome in **30 seconds**.

---

## 10. Timeline (≈2–3 weeks, part-time)

| Days | Milestone |
|---|---|
| 1–2 | Repo, config, ingest, FAISS baseline (System A working end-to-end) |
| 3–5 | Triple extraction + graph build + graph retrieval (System B) |
| 6–9 | LangGraph agent with router + self-correction (System C) |
| 9–11 | Evaluation harness (RAGAS + EM/F1) |
| 11–12 | Full benchmark + charts + slice analysis |
| 12–14 | FastAPI + Gradio + deploy to HF Space + polish README |

---

## 11. Pitfalls to avoid

- **Don't skip the baseline.** System A is what makes your numbers *mean* something. Build it first.
- **Cache LLM extractions to disk.** The graph-build step is your biggest free-tier cost; never re-run it from scratch.
- **Pin `temperature=0`** across the pipeline or your benchmark won't be reproducible.
- **Throttle to the RPM cap** (15/min Gemini, 30/min Groq) with a small sleep/retry wrapper, or you'll hit rate-limit errors mid-benchmark.
- **Make the agent's self-correction actually fire** on some questions — if it never loops, System C is just System B with extra steps. Include hard multi-hop questions in your eval set.
- **Commit raw results.** "Trust me, it scored 0.8" is weak; committed `results/*.jsonl` is strong.
- **Show the tradeoff honestly.** If C is slower or uses more calls, say so. That candor reads as senior.
