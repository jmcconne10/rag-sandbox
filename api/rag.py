# api/rag.py
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable, List, Dict, Any

import chromadb
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# ----------------------------
# Config
# ----------------------------
# Where to store Chroma’s on-disk data (docker-compose should mount ./chroma -> /data/chroma)
PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/chroma")
# Name of the collection
COLLECTION = os.getenv("CHROMA_COLLECTION", "local_knowledge")

# Embedding model (fastembed runs on CPU; small + quick to download)
# Alternatives you can try later: "BAAI/bge-base-en-v1.5", "BAAI/bge-m3"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # characters overlap


# ----------------------------
# Utilities
# ----------------------------

def _iter_files(folder: Path) -> Iterable[Path]:
    """Yield text-like files we know how to parse."""
    exts = {".md", ".markdown", ".txt"}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # Best-effort fallback
        return p.read_text(errors="ignore")


def _chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple character-based chunking with overlap (robust and language-agnostic)."""
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# ----------------------------
# Chroma (new API) + Embeddings
# ----------------------------

# Create a persistent client (no legacy Settings())
_client: chromadb.ClientAPI = PersistentClient(path=PERSIST_DIR)

# FastEmbed embedding function (CPU only; no Ollama dependency)
# We use Chroma’s PortableEmbeddingFunction interface to supply vectors directly.
try:
    from fastembed import TextEmbedding

    _fe_model = TextEmbedding(model_name=EMBED_MODEL)
    # Wrap it into a Chroma-compatible callable
    class _FastEmbedEF(embedding_functions.EmbeddingFunction):
        def __call__(self, texts: List[str]) -> List[List[float]]:
            # fastembed expects an iterable of strings and returns an iterator of vectors
            # Convert to list for Chroma
            return [vec for vec in _fe_model.embed(texts)]

except Exception as e:
    # If fastembed import fails, raise a helpful error early
    raise RuntimeError(
        f"Failed to initialize FastEmbed model '{EMBED_MODEL}'. "
        f"Make sure fastembed and onnxruntime are installed. Under Docker, "
        f'check your "api/requirements.txt". Original error: {e}'
    )

_embedding_fn = _FastEmbedEF()

# Create or get the collection
_collection = _client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},   # cosine works well for bge-family models
    embedding_function=_embedding_fn,    # let Chroma call our FastEmbed wrapper
)


# ----------------------------
# Public API
# ----------------------------

def ingest_folder(folder_path: str) -> Dict[str, Any]:
    """
    Ingest all .md/.markdown/.txt files from the folder into the Chroma collection.
    Returns a small dict with counts.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Ingest folder does not exist or is not a directory: {folder}")

    total_files = 0
    total_chunks = 0

    # We upsert in small batches to keep memory steady
    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict[str, Any]] = []
    BATCH_SIZE = 64

    for f in _iter_files(folder):
        total_files += 1
        text = _read_text_file(f)
        chunks = _chunk(text)
        for idx, ch in enumerate(chunks):
            total_chunks += 1
            doc_id = f"{f.as_posix()}::{idx:06d}"
            batch_ids.append(doc_id)
            batch_docs.append(ch)
            batch_metas.append({"source": f.as_posix(), "chunk": idx})

            if len(batch_ids) >= BATCH_SIZE:
                _collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                batch_ids, batch_docs, batch_metas = [], [], []

    # Flush remainder
    if batch_ids:
        _collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    # Persist to disk (PersistentClient does this, but force a flush)
    try:
        _client.persist()
    except Exception:
        # Some backends persist automatically; ignore if not supported
        pass

    return {
        "status": "ok",
        "files_ingested": total_files,
        "chunks_ingested": total_chunks,
        "collection": COLLECTION,
        "persist_dir": PERSIST_DIR,
        "embed_model": EMBED_MODEL,
    }


def retrieve(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Return top-k chunks for a user query.
    """
    if not query or not query.strip():
        return {"results": [], "k": k}

    res = _collection.query(query_texts=[query], n_results=max(1, k))
    # Chroma returns dicts with lists; normalize to a simple list of hits
    hits: List[Dict[str, Any]] = []
    if res and "documents" in res and res["documents"]:
        docs = res["documents"][0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]  # cosine distance
        ids = res.get("ids", [[]])[0]
        for i, doc in enumerate(docs):
            hits.append({
                "id": ids[i] if i < len(ids) else str(uuid.uuid4()),
                "document": doc,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            })

    return {"results": hits, "k": k}
