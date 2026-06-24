from typing import List, TypedDict

from langgraph.graph import END, StateGraph

from src.config import load_config
from src.generate import format_context
from src.graph_store import graph_search, load_graph_retriever
from src.ingest import CORPUS_PATH, load_jsonl
from src.llm import llm
from src.vector_store import load_vector_store

ROUTE_PROMPT = """Classify the question into exactly one retrieval strategy. Reply with ONLY one word: \
vector, graph, or hybrid.

vector: a simple factual lookup answerable from a single passage.
graph: requires connecting facts across two or more entities (multi-hop).
hybrid: unclear, or likely needs both a direct match and a connecting fact.

Question: {question}
Answer:"""

GRADE_PROMPT = """Does the context below contain enough information to fully answer the question? \
Reply on the first line with exactly "yes" or "no", then a brief reason on the second line.

Context:
{context}

Question: {question}"""

REWRITE_PROMPT = """Retrieval failed to find enough context to answer this question. Rewrite it to make \
the missing fact easier to retrieve: either decompose it into its bridging sub-question (e.g. "who/what is \
the connecting entity?"), or, if you already know the connecting entity, name it explicitly in the rewritten \
question. Return ONLY the rewritten question, no preamble.

Original question: {question}"""

GENERATE_PROMPT = """Answer the question using only the provided context. After each claim, cite the chunk \
id(s) it came from in square brackets, e.g. [chunk_42]. If the context does not contain the answer, say \
"I don't know".

Context:
{context}

Question: {question}
Answer:"""


class AgentState(TypedDict):
    question: str           # current (possibly rewritten) retrieval query
    original_question: str  # the user's question, kept stable for the final answer
    route: str
    context: List[dict]
    grade: str
    tries: int
    answer: str


def build_agent(model=None):
    """Compiles the System C LangGraph agent. Heavy resources (embedder, FAISS
    index, graph retriever) are loaded once here and closed over by the node
    functions, not reloaded per question."""
    cfg = load_config()
    # Gemini's free-tier cap (20 requests/day) is far too low for an agent that
    # can issue 3-7 LLM calls per question; Groq's token-based daily cap fits
    # this workload better, so it's the default unless the caller overrides it.
    model = model or cfg["llm"]["fallback_model"]
    top_k = cfg["retrieval"]["top_k"]
    max_retries = cfg["agent"]["max_retries"]

    chunks_by_id = {c["id"]: c for c in load_jsonl(CORPUS_PATH)}
    vs = load_vector_store()
    retriever = load_graph_retriever()

    def classify(question):
        resp = llm(ROUTE_PROMPT.format(question=question), model=model).strip().lower()
        for r in ("vector", "graph", "hybrid"):
            if r in resp:
                return r
        return "hybrid"

    def run_retriever(route, question):
        chunks = []
        if route in ("vector", "hybrid"):
            chunks.extend(c for c, _ in vs.search(question, k=top_k))
        if route in ("graph", "hybrid"):
            seen = {c["id"] for c in chunks}
            chunks.extend(
                c for c in graph_search(retriever, chunks_by_id, question, top_n=3, max_chunks=10)
                if c["id"] not in seen
            )
        return chunks

    def grade_context(question, context):
        resp = llm(GRADE_PROMPT.format(context=format_context(context), question=question), model=model)
        first_line = resp.strip().splitlines()[0].strip().lower() if resp.strip() else ""
        return "yes" if "yes" in first_line else "no"

    def rewrite_query(question):
        return llm(REWRITE_PROMPT.format(question=question), model=model).strip()

    def node_route(s):
        s["route"] = classify(s["question"])
        return s

    def node_retrieve(s):
        # Accumulate rather than replace: a rewritten query that finds the
        # missing bridge fact shouldn't discard a connecting passage a
        # prior attempt already found.
        new_chunks = run_retriever(s["route"], s["question"])
        seen = {c["id"] for c in s["context"]}
        s["context"] = s["context"] + [c for c in new_chunks if c["id"] not in seen]
        return s

    def node_grade(s):
        s["grade"] = grade_context(s["original_question"], s["context"])
        return s

    def node_rewrite(s):
        s["question"] = rewrite_query(s["question"])
        s["tries"] += 1
        return s

    def node_generate(s):
        ctx = "\n\n".join(f"[{c['id']}] ({c['title']}) {c['text']}" for c in s["context"])
        prompt = GENERATE_PROMPT.format(context=ctx, question=s["original_question"])
        s["answer"] = llm(prompt, model=model)
        return s

    def after_grade(s):
        return "generate" if s["grade"] == "yes" or s["tries"] >= max_retries else "rewrite"

    g = StateGraph(AgentState)
    for name, fn in [
        ("route", node_route), ("retrieve", node_retrieve), ("grade", node_grade),
        ("rewrite", node_rewrite), ("generate", node_generate),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


def ask(agent, question):
    return agent.invoke({
        "question": question,
        "original_question": question,
        "route": "",
        "context": [],
        "grade": "",
        "tries": 0,
        "answer": "",
    })


if __name__ == "__main__":
    agent = build_agent()
    result = ask(agent, "Were Scott Derrickson and Ed Wood of the same nationality?")
    print("route:", result["route"], "| tries:", result["tries"])
    print("answer:", result["answer"])
