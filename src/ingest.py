import json
import os

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions.jsonl")


def load_hotpotqa_subset(n=500, split="validation"):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    return ds.select(range(n))


def build_corpus(subset):
    """Flatten HotpotQA context paragraphs into a deduped corpus, plus a
    parallel question list carrying gold answers and supporting-fact titles."""
    chunks_by_title = {}
    questions = []
    for ex in subset:
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title not in chunks_by_title:
                chunks_by_title[title] = {
                    "id": f"chunk_{len(chunks_by_title)}",
                    "title": title,
                    "text": " ".join(sentences),
                }
        questions.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex["type"],
            "level": ex["level"],
            "supporting_titles": list(ex["supporting_facts"]["title"]),
        })
    return list(chunks_by_title.values()), questions


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    subset = load_hotpotqa_subset(n=500)
    chunks, questions = build_corpus(subset)
    save_jsonl(chunks, CORPUS_PATH)
    save_jsonl(questions, QUESTIONS_PATH)
    print(f"corpus: {len(chunks)} passages, questions: {len(questions)}")
