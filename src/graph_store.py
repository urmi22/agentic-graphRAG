import json
import os
import pickle

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import load_config
from src.extract_triples import TRIPLES_PATH

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "graph.pkl")


def normalize_entity(name):
    return " ".join(name.strip().lower().split())


def build_graph(triples_path=TRIPLES_PATH):
    G = nx.MultiDiGraph()
    with open(triples_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunk_id = rec["chunk_id"]
            for t in rec["triples"]:
                head = normalize_entity(t["head"])
                tail = normalize_entity(t["tail"])
                if not head or not tail:
                    continue
                G.add_edge(head, tail, relation=t["relation"], source=chunk_id)
                for node in (head, tail):
                    G.nodes[node].setdefault("source_chunks", set()).add(chunk_id)
    return G


def save_graph(G, path=GRAPH_PATH):
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_graph(path=GRAPH_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


class GraphRetriever:
    """Links a query to graph entities, walks the k-hop neighborhood, and
    returns the source chunk ids of every edge touched along the way."""

    def __init__(self, G, embedder_model, k_hops=2):
        self.G = G
        self.embedder = SentenceTransformer(embedder_model)
        self.k_hops = k_hops
        self.node_list = list(G.nodes())
        self.node_emb = (
            self.embedder.encode(self.node_list, normalize_embeddings=True)
            if self.node_list else np.zeros((0, 1))
        )

    def link_entities(self, query, top_n=3):
        if not self.node_list:
            return []
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.node_emb @ q
        top_idx = np.argsort(-scores)[:top_n]
        return [self.node_list[i] for i in top_idx]

    def subgraph_chunk_ids(self, query, top_n=3, max_chunks=None):
        seed_nodes = self.link_entities(query, top_n=top_n)
        visited = set(seed_nodes)
        frontier = set(seed_nodes)
        seen, ordered = set(), []

        def add_source(cid):
            if cid and cid not in seen:
                seen.add(cid)
                ordered.append(cid)

        for _ in range(self.k_hops):
            if not frontier:
                break
            next_frontier = set()
            for node in frontier:
                if node not in self.G:
                    continue
                for _, nbr, data in self.G.out_edges(node, data=True):
                    add_source(data.get("source"))
                    if nbr not in visited:
                        next_frontier.add(nbr)
                        visited.add(nbr)
                for nbr, _, data in self.G.in_edges(node, data=True):
                    add_source(data.get("source"))
                    if nbr not in visited:
                        next_frontier.add(nbr)
                        visited.add(nbr)
            frontier = next_frontier

        return ordered[:max_chunks] if max_chunks else ordered


def load_graph_retriever(rebuild=False):
    cfg = load_config()
    if not rebuild and os.path.exists(GRAPH_PATH):
        G = load_graph()
    else:
        G = build_graph()
        save_graph(G)
    return GraphRetriever(G, cfg["embedding"]["model"], k_hops=cfg["graph"]["k_hops"])


def graph_search(retriever, chunks_by_id, query, top_n=3, max_chunks=None):
    chunk_ids = retriever.subgraph_chunk_ids(query, top_n=top_n, max_chunks=max_chunks)
    return [chunks_by_id[cid] for cid in chunk_ids if cid in chunks_by_id]


if __name__ == "__main__":
    retriever = load_graph_retriever()
    print(f"graph: {retriever.G.number_of_nodes()} nodes, {retriever.G.number_of_edges()} edges")
    print(retriever.link_entities("Were Scott Derrickson and Ed Wood of the same nationality?"))
