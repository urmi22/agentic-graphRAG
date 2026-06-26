"""Evaluation harness: runs Systems A, B, and C over the same question set and
scores two independent layers, per the project's eval design --

  1. RAGAS (LLM-as-judge): faithfulness, answer_relevancy, context_precision,
     context_recall. Catches hallucination and retrieval quality.
  2. SQuAD-style Exact Match / F1 against HotpotQA's gold short answer.
     Deterministic, judge-free, cheap to reproduce.

Reporting both is the point: RAGAS can be fooled by a fluent wrong answer,
EM/F1 can't see whether a *correct* answer was actually grounded. Together
they triangulate.

RAGAS judging only runs on the 3 anchor questions, not the full sample: each
judged example costs up to ~7 LLM calls (faithfulness, answer_relevancy,
context_precision x N contexts, context_recall), and Groq's free-tier daily
token cap can't sustain that across the whole stratified sample plus the
generation calls for systems A/B/C. EM/F1 still run on the full sample, since
they're free and judge-independent.
"""
import json
import os
import random
import time

from src.agent import ask, build_agent
from src.config import load_config
from src.generate import answer_with_context
from src.graph_store import GraphRetriever, build_graph
from src.ingest import CORPUS_PATH, QUESTIONS_PATH, load_jsonl
from src.ragas_judge import build_judge, score_example
from src.squad_metrics import exact_match_score, f1_score
from src.vector_store import load_vector_store

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_results.jsonl")
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_summary.json")
GEN_MODEL = "groq/llama-3.3-70b-versatile"
NA_RAGAS_SCORES = {
    "faithfulness": None, "answer_relevancy": None,
    "context_precision": None, "context_recall": None,
}

ANCHOR_QUESTION_IDS = {
    "5a8b57f25542995d1e6f1371",  # comparison, both systems agree trivially
    "5a8e3ea95542995a26add48d",  # bridge: graph succeeds, vector alone fails
    "5a8c7595554299585d9e36b6",  # bridge: graph also fails (entity-alias split); agent recovers via rewrite
}


def select_questions(cfg):
    questions = load_jsonl(QUESTIONS_PATH)
    by_id = {q["id"]: q for q in questions}
    anchors = [by_id[qid] for qid in ANCHOR_QUESTION_IDS]

    rng = random.Random(cfg["eval"]["seed"])
    remaining = [q for q in questions if q["id"] not in ANCHOR_QUESTION_IDS]
    n = cfg["eval"]["sample_per_type"]
    sampled = []
    for qtype in ("comparison", "bridge"):
        pool = [q for q in remaining if q["type"] == qtype]
        sampled.extend(rng.sample(pool, min(n, len(pool))))

    return anchors + sampled


def run_system_a(vs, question):
    chunks = [c for c, _ in vs.search(question, k=5)]
    answer = answer_with_context(question, chunks, model=GEN_MODEL)
    return answer, [c["text"] for c in chunks]


def run_system_b(vs, graph_retriever, chunks_by_id, question):
    vec_chunks = [c for c, _ in vs.search(question, k=5)]
    graph_ids = graph_retriever.subgraph_chunk_ids(question, top_n=3, max_chunks=10)
    seen = {c["id"] for c in vec_chunks}
    graph_chunks = [chunks_by_id[cid] for cid in graph_ids if cid in chunks_by_id and cid not in seen]
    combined = vec_chunks + graph_chunks
    answer = answer_with_context(question, combined, model=GEN_MODEL)
    return answer, [c["text"] for c in combined]


def run_system_c(agent, question):
    result = ask(agent, question)
    return result["answer"], [c["text"] for c in result["context"]]


def main():
    cfg = load_config()
    questions = select_questions(cfg)
    print(f"Evaluating {len(questions)} questions across systems A, B, C "
          f"({len(ANCHOR_QUESTION_IDS)} anchor cases + a stratified random sample).")

    chunks_by_id = {c["id"]: c for c in load_jsonl(CORPUS_PATH)}
    vs = load_vector_store()
    G = build_graph()
    graph_retriever = GraphRetriever(G, cfg["embedding"]["model"], k_hops=cfg["graph"]["k_hops"])
    agent = build_agent()
    judge = build_judge()
    sleep_s = cfg["eval"]["judge_sleep_s"]

    rows = []
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            print(f"\n[{i + 1}/{len(questions)}] {q['question']}")
            for system, run in [
                ("A", lambda question: run_system_a(vs, question)),
                ("B", lambda question: run_system_b(vs, graph_retriever, chunks_by_id, question)),
                ("C", lambda question: run_system_c(agent, question)),
            ]:
                try:
                    answer, contexts = run(q["question"])
                except Exception as e:
                    print(f"  [{system}] generation failed, recording as a miss: {e}")
                    answer, contexts = "[ERROR: generation failed]", []

                em = exact_match_score(answer, q["answer"])
                f1 = f1_score(answer, q["answer"])
                if q["id"] in ANCHOR_QUESTION_IDS and contexts:
                    ragas_scores = score_example(
                        judge, q["question"], answer, contexts, q["answer"], sleep_s=sleep_s
                    )
                else:
                    ragas_scores = dict(NA_RAGAS_SCORES)
                row = {
                    "question_id": q["id"], "question": q["question"], "type": q["type"],
                    "system": system, "gold": q["answer"], "answer": answer,
                    "n_contexts": len(contexts), "em": em, "f1": f1, **ragas_scores,
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()

                def fmt(v):
                    return f"{v:.2f}" if v is not None else "n/a"

                print(f"  [{system}] em={em} f1={f1:.2f} "
                      f"faith={fmt(ragas_scores['faithfulness'])} "
                      f"rel={fmt(ragas_scores['answer_relevancy'])} "
                      f"ctx_prec={fmt(ragas_scores['context_precision'])} "
                      f"ctx_rec={fmt(ragas_scores['context_recall'])}")
                time.sleep(sleep_s if q["id"] in ANCHOR_QUESTION_IDS else 1.0)

    summarize(rows)


def summarize(rows):
    metrics = ["em", "f1", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = {}
    for system in ("A", "B", "C"):
        sys_rows = [r for r in rows if r["system"] == system]
        summary[system] = {}
        for m in metrics:
            values = [r[m] for r in sys_rows if r[m] is not None]
            summary[system][m] = round(sum(values) / len(values), 3) if values else None

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'system':<8}" + "".join(f"{m:<18}" for m in metrics))
    for system, scores in summary.items():
        print(f"{system:<8}" + "".join(f"{scores[m] if scores[m] is not None else 'n/a':<18}" for m in metrics))
    print(f"\nSaved per-question rows to {RESULTS_PATH}")
    print(f"Saved aggregate summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
