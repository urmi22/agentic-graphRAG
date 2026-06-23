import json
import os
import time

from src.config import load_config
from src.ingest import CORPUS_PATH, load_jsonl
from src.llm import llm

TRIPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "triples.jsonl")

EXTRACT_PROMPT = """Extract factual relationships from each passage below as JSON triples.
Return ONLY a JSON list of objects: [{{"chunk_id": "...", "head": "...", "relation": "...", "tail": "..."}}].
Use canonical entity names (full names, not pronouns). Tag every triple with the chunk_id \
of the passage it came from.

Passages:
{passages}"""


def format_batch(batch):
    return "\n\n".join(f'[{c["id"]}] ({c["title"]}) {c["text"]}' for c in batch)


def parse_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def load_done_chunk_ids(path=TRIPLES_PATH):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["chunk_id"])
    return done


def extract_all(batch_size=5, sleep_s=2.5, model=None):
    cfg = load_config()
    model = model or cfg["llm"]["fallback_model"]  # Groq: higher free-tier RPM for this bulk step
    chunks = load_jsonl(CORPUS_PATH)

    done = load_done_chunk_ids()
    pending = [c for c in chunks if c["id"] not in done]
    print(f"{len(done)} chunks already extracted, {len(pending)} remaining")

    n_batches = -(-len(pending) // batch_size)
    os.makedirs(os.path.dirname(TRIPLES_PATH), exist_ok=True)
    with open(TRIPLES_PATH, "a", encoding="utf-8") as f:
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            batch_ids = {c["id"] for c in batch}

            try:
                resp = llm(EXTRACT_PROMPT.format(passages=format_batch(batch)), model=model)
                triples = parse_response(resp)
            except Exception as e:
                print(f"batch {i // batch_size + 1}/{n_batches}: skipped ({e}), will retry next run")
                time.sleep(sleep_s * 4)
                continue

            by_chunk = {cid: [] for cid in batch_ids}
            for t in triples:
                cid = t.get("chunk_id")
                if cid in by_chunk and {"head", "relation", "tail"} <= t.keys():
                    by_chunk[cid].append({"head": t["head"], "relation": t["relation"], "tail": t["tail"]})

            for cid, triple_list in by_chunk.items():
                f.write(json.dumps({"chunk_id": cid, "triples": triple_list}) + "\n")
            f.flush()

            print(f"batch {i // batch_size + 1}/{n_batches}: {sum(len(v) for v in by_chunk.values())} triples")
            time.sleep(sleep_s)


if __name__ == "__main__":
    extract_all()
