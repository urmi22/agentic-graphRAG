from src.llm import llm

PROMPT = """Answer the question using only the provided context. \
If the context does not contain the answer, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""


def format_context(chunks):
    return "\n\n".join(f"[{c['title']}] {c['text']}" for c in chunks)


def answer_with_context(question, chunks, model="gemini/gemini-2.5-flash"):
    prompt = PROMPT.format(context=format_context(chunks), question=question)
    return llm(prompt, model=model)
