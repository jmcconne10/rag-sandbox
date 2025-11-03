import os
from typing import List
import httpx
from fastembed import TextEmbedding

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# Create a singleton embedder (downloads ONNX model the first time)
_embedder = TextEmbedding(model_name=EMBED_MODEL)

# ---------------------------------------------------------------------------
# Embedding and Chat Helpers
# ---------------------------------------------------------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embedding vectors for a list of texts using FastEmbed."""
    return [
        vec.tolist() if hasattr(vec, "tolist") else list(vec)
        for vec in _embedder.embed(texts)
    ]


async def ollama_chat(prompt: str, *, model: str | None = None) -> str:
    """Send a chat-style prompt to an Ollama model and return the text response."""
    model = model or CHAT_MODEL
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You answer using only the provided context. "
                            "If the context is insufficient, say you don't know."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = res
