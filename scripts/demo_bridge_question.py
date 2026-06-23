import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.config import load_config
from src.generate import answer_with_context
from src.graph_store import GraphRetriever, build_graph
from src.ingest import CORPUS_PATH, load_jsonl
from src.vector_store import load_vector_store

TRIPLES_SNAPSHOT = os.path.join(os.path.dirname(__file__), "..", "data", "triples_snapshot.jsonl")
OUT_PNG = os.path.join(os.path.dirname(__file__), "..", "data", "bridge_subgraph.png")
GAP_PNG = os.path.join(os.path.dirname(__file__), "..", "data", "bridge_gap_diagram.png")

QUESTION = ("What government position was held by the woman who portrayed "
            "Corliss Archer in the film Kiss and Tell?")
GOLD = "Chief of Protocol"
GEN_MODEL = "groq/llama-3.3-70b-versatile"


def main():
    cfg = load_config()
    chunks_by_id = {c["id"]: c for c in load_jsonl(CORPUS_PATH)}

    print("=" * 70)
    print("QUESTION:", QUESTION)
    print("GOLD ANSWER:", GOLD)
    print("=" * 70)

    # Step 1: vector search (System A)
    vs = load_vector_store()
    vec_results = vs.search(QUESTION, k=5)
    print("\n[1] Vector search top-5:")
    for c, s in vec_results:
        print(f"    {s:.3f}  {c['title']}")
    vec_chunks = [c for c, _ in vec_results]
    vec_answer = answer_with_context(QUESTION, vec_chunks, model=GEN_MODEL)
    print("\n[System A answer]:", vec_answer)

    # Step 2: graph retrieval
    G = build_graph(triples_path=TRIPLES_SNAPSHOT)
    retriever = GraphRetriever(G, cfg["embedding"]["model"], k_hops=cfg["graph"]["k_hops"])
    seed_nodes = retriever.link_entities(QUESTION, top_n=3)
    print("\n[2] Linked entities (seed nodes):", seed_nodes)

    graph_chunk_ids = retriever.subgraph_chunk_ids(QUESTION, top_n=3, max_chunks=10)
    graph_chunks = [chunks_by_id[cid] for cid in graph_chunk_ids if cid in chunks_by_id]
    print("\n[Graph-retrieved passages]:")
    for c in graph_chunks:
        print(f"    {c['title']}")

    # Step 3: combined vector+graph generation (System B-ish)
    seen_ids = {c["id"] for c in vec_chunks}
    combined = vec_chunks + [c for c in graph_chunks if c["id"] not in seen_ids]
    graphrag_answer = answer_with_context(QUESTION, combined, model=GEN_MODEL)
    print("\n[3] GraphRAG (vector+graph) answer:", graphrag_answer)

    # Step 4: visualize the k-hop subgraph actually traversed
    visited = set(seed_nodes)
    frontier = set(seed_nodes)
    hop_of = {n: 0 for n in seed_nodes}
    edges = []
    for hop in range(1, retriever.k_hops + 1):
        next_frontier = set()
        for node in frontier:
            if node not in G:
                continue
            for _, nbr, data in G.out_edges(node, data=True):
                edges.append((node, nbr, data.get("relation", "")))
                if nbr not in visited:
                    next_frontier.add(nbr)
                    visited.add(nbr)
                    hop_of[nbr] = hop
            for nbr, _, data in G.in_edges(node, data=True):
                edges.append((nbr, node, data.get("relation", "")))
                if nbr not in visited:
                    next_frontier.add(nbr)
                    visited.add(nbr)
                    hop_of[nbr] = hop
        frontier = next_frontier

    H = nx.DiGraph()
    for u, v, rel in edges:
        H.add_edge(u, v, relation=rel)

    color_by_hop = {0: "tab:red", 1: "tab:orange"}
    colors = [color_by_hop.get(hop_of.get(n), "tab:blue") for n in H.nodes()]

    plt.figure(figsize=(16, 11))
    pos = nx.spring_layout(H, k=1.0, seed=42)
    nx.draw_networkx_nodes(H, pos, node_color=colors, node_size=900)
    nx.draw_networkx_labels(H, pos, font_size=7)
    nx.draw_networkx_edges(H, pos, arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.05")
    edge_labels = {(u, v): d["relation"] for u, v, d in H.edges(data=True)}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=6)
    plt.title(f"k-hop subgraph (k={retriever.k_hops}) for:\n{QUESTION}", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\n[4] saved subgraph visualization to {OUT_PNG}")
    print(f"    nodes: {H.number_of_nodes()}, edges: {H.number_of_edges()}")

    save_gap_diagram()


def save_gap_diagram():
    """Hand-curated diagram isolating why the bridge fails: the answer-bearing
    entity is split across two unmerged surface forms ('shirley temple' vs
    'shirley temple black'), so the 2-hop walk dead-ends before reaching the
    gold answer, while a lexically-similar distractor gets pulled in instead."""
    pos = {
        "kiss and tell": (0.0, 3.0),
        "shirley temple": (0.0, 1.8),
        "shirley temple black": (2.4, 1.8),
        "chief of protocol of the united states": (2.4, 0.6),
        "meet corliss archer": (-2.6, 3.0),
        "assistant secretary of state for legislative affairs": (-1.6, 0.6),
        'douglas joseph "doug" bennet, jr.': (-1.6, -0.6),
    }
    seeds = {"kiss and tell", "meet corliss archer", "assistant secretary of state for legislative affairs"}
    reached = {"shirley temple", 'douglas joseph "doug" bennet, jr.'}
    answer_node = "chief of protocol of the united states"
    orphan_node = "shirley temple black"

    edges = [
        ("kiss and tell", "shirley temple", "stars"),
        ('douglas joseph "doug" bennet, jr.', "assistant secretary of state for legislative affairs", "served_as"),
        ("shirley temple black", answer_node, "served_as"),
    ]

    fig, ax = plt.subplots(figsize=(11, 8))
    for u, v, rel in edges:
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5,
                             shrinkA=22, shrinkB=22),
        )
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        ax.text(mx, my, rel, fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8))

    # the missing alias link
    x1, y1 = pos["shirley temple"]
    x2, y2 = pos[orphan_node]
    ax.plot([x1, x2], [y1, y2], linestyle="--", color="crimson", lw=1.8)
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18,
            "same real person —\nnever merged into one node",
            fontsize=8, color="crimson", ha="center", fontstyle="italic")

    for node, (x, y) in pos.items():
        if node == answer_node:
            color, ec = "white", "tab:green"
        elif node == orphan_node:
            color, ec = "white", "crimson"
        elif node in seeds:
            color, ec = "tab:red", "tab:red"
        elif node in reached:
            color, ec = "tab:orange", "tab:orange"
        else:
            color, ec = "tab:blue", "tab:blue"
        ax.scatter([x], [y], s=2600, color=color, edgecolors=ec, linewidths=2.2, zorder=3)
        ax.text(x, y, node, fontsize=8, ha="center", va="center", wrap=True, zorder=4)

    ax.text(2.4, 0.05, "GOLD ANSWER — never reached", fontsize=8.5,
            color="tab:green", ha="center", fontweight="bold")

    legend_items = [
        ("tab:red", "seed (entity-linked from query text)"),
        ("tab:orange", "reached via 1-hop traversal"),
        ("crimson", "holds the answer, but disconnected"),
        ("tab:green", "gold answer node"),
    ]
    for i, (c, label) in enumerate(legend_items):
        ax.scatter([-3.4], [-1.2 - i * 0.35], s=120, color=c if c != "crimson" else "white",
                   edgecolors=c, linewidths=2)
        ax.text(-3.2, -1.2 - i * 0.35, label, fontsize=8, va="center")

    ax.set_title(f"Why the graph path fails on:\n\"{QUESTION}\"", fontsize=10)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-2.4, 3.8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(GAP_PNG, dpi=150)
    print(f"\n[5] saved gap diagram to {GAP_PNG}")


if __name__ == "__main__":
    main()
