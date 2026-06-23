import litellm
from dotenv import load_dotenv

load_dotenv()              # reads .env locally; harmless on Colab
litellm.drop_params = True # ignore params a provider doesn't support

def llm(prompt, model="gemini/gemini-2.5-flash", temperature=0.0):
    """Single-prompt completion via LiteLLM. Keys come from the environment."""
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":   # quick smoke test: python llm.py
    print(llm("Say OK if you can hear me."))