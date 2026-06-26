import time

import litellm
from dotenv import load_dotenv

load_dotenv()              # reads .env locally; harmless on Colab
litellm.drop_params = True # ignore params a provider doesn't support


def llm(prompt, model="gemini/gemini-2.5-flash", temperature=0.0, max_attempts=4, backoff_s=20):
    """Single-prompt completion via LiteLLM. Keys come from the environment.

    Retries with a fixed backoff on rate-limit errors: free-tier RPM/TPM caps
    (Groq, Gemini) are the project's main operational constraint, and a single
    eval run issues enough calls in a short window to hit them.
    """
    for attempt in range(max_attempts):
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except litellm.RateLimitError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(backoff_s)

if __name__ == "__main__":   # quick smoke test: python llm.py
    print(llm("Say OK if you can hear me."))