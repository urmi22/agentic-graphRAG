"""Wires ragas's metric classes to this project's free-tier LLM (Groq, via the
same LiteLLM call path as everywhere else) and local embedder, instead of the
default OpenAI client ragas examples assume.

Also works around a packaging bug in ragas 0.4.3: it eagerly imports
`ChatVertexAI` from a langchain-community submodule that recent
langchain-community releases no longer ship (langchain-community is being
sunset in favor of standalone integration packages). We never use Vertex AI,
so a stub module satisfies the import without pulling in real Vertex code.
"""
import asyncio
import sys
import time
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

import instructor
import litellm
from dotenv import load_dotenv
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)
from sentence_transformers import SentenceTransformer

from src.config import load_config

load_dotenv()

MAX_JUDGED_CONTEXTS = 3  # context_precision makes one LLM call per context
RATE_LIMIT_BACKOFF_S = 20
TRANSIENT_BACKOFF_S = 5
MAX_ATTEMPTS = 4

# Groq's llama-3.3-70b-versatile occasionally emits a tool call as literal
# "<function=...>" text instead of a structured call ("tool_use_failed");
# it's nondeterministic, so a plain retry usually succeeds on the next try.
TRANSIENT_MARKERS = ("rate_limit", "tool_use_failed", "failed to call a function")


async def _ascore_with_retry(metric, **kwargs):
    """Same free-tier-RPM/TPM problem as src/llm.py, on the judge's own call
    path (instructor -> litellm.acompletion, bypassing src/llm.py entirely).

    Returns None (rather than raising) once retries are exhausted: a free-tier
    daily quota can run out mid-run, and losing every already-scored question
    to one unlucky call is worse than reporting that one metric as unavailable.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            return (await metric.ascore(**kwargs)).value
        except Exception as e:
            msg = str(e).lower()
            is_transient = any(marker in msg for marker in TRANSIENT_MARKERS)
            if attempt == MAX_ATTEMPTS - 1 or not is_transient:
                print(f"  [judge] giving up on {metric.__class__.__name__}: {e}")
                return None
            backoff = RATE_LIMIT_BACKOFF_S if "rate_limit" in msg else TRANSIENT_BACKOFF_S
            time.sleep(backoff)


class STEmbedding(BaseRagasEmbedding):
    """Wraps the project's existing sentence-transformers embedder so
    answer_relevancy's cosine-similarity step is local and free, not a
    second billed API call."""

    def __init__(self, model_name):
        super().__init__()
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text, **kwargs):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    async def aembed_text(self, text, **kwargs):
        return self.embed_text(text)


def build_judge(model=None):
    """Builds the four RAGAS metrics, all backed by one LLM judge."""
    cfg = load_config()
    model = model or cfg["llm"]["fallback_model"]  # groq/llama-3.3-70b-versatile
    provider = model.split("/")[0]
    client = instructor.from_litellm(litellm.acompletion)
    llm = llm_factory(model, provider=provider, client=client, adapter="litellm")
    embeddings = STEmbedding(cfg["embedding"]["model"])
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


def score_example(metrics, question, answer, contexts, reference, sleep_s=1.5):
    """Runs all four RAGAS metrics for one (question, answer, contexts, gold)
    tuple. Sleeps between calls to stay under the judge model's free-tier RPM
    cap -- each call here is a real LLM request, not a cache hit."""
    judged_contexts = contexts[:MAX_JUDGED_CONTEXTS] or ["(no context retrieved)"]
    scores = {}

    async def run():
        scores["faithfulness"] = await _ascore_with_retry(
            metrics["faithfulness"],
            user_input=question, response=answer, retrieved_contexts=judged_contexts,
        )
        time.sleep(sleep_s)

        scores["answer_relevancy"] = await _ascore_with_retry(
            metrics["answer_relevancy"], user_input=question, response=answer,
        )
        time.sleep(sleep_s)

        scores["context_precision"] = await _ascore_with_retry(
            metrics["context_precision"],
            user_input=question, reference=reference, retrieved_contexts=judged_contexts,
        )
        time.sleep(sleep_s)

        scores["context_recall"] = await _ascore_with_retry(
            metrics["context_recall"],
            user_input=question, retrieved_contexts=judged_contexts, reference=reference,
        )

    asyncio.run(run())
    return scores
