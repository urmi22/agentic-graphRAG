import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import load_config
from src.ingest import CORPUS_PATH, load_jsonl

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")


class VectorStore:
    def __init__(self, chunks, model_name):
        self.chunks = chunks
        self.embedder = SentenceTransformer(model_name)
        self.index = None

    def build(self):
        emb = self.embedder.encode(
            [c["text"] for c in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.array(emb, dtype="float32"))
        return self

    def save(self, path=INDEX_PATH):
        faiss.write_index(self.index, path)

    def load(self, path=INDEX_PATH):
        self.index = faiss.read_index(path)
        return self

    def search(self, query, k=5):
        q = self.embedder.encode([query], normalize_embeddings=True)
        scores, idx = self.index.search(np.array(q, dtype="float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(idx[0], scores[0])]


def load_vector_store(rebuild=False):
    cfg = load_config()
    chunks = load_jsonl(CORPUS_PATH)
    vs = VectorStore(chunks, cfg["embedding"]["model"])
    if not rebuild and os.path.exists(INDEX_PATH):
        vs.load()
    else:
        vs.build()
        vs.save()
    return vs


if __name__ == "__main__":
    vs = load_vector_store()
    for chunk, score in vs.search("Were Scott Derrickson and Ed Wood of the same nationality?"):
        print(round(score, 3), chunk["title"])
