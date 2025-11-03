from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: str
    files_ingested: int
    chunks_ingested: int
    collection: str
    persist_dir: str
    embed_model: str


class QueryRequest(BaseModel):
    query: str
    k: int = 5


class QueryHit(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any] = {}
    distance: Optional[float] = None


class QueryResponse(BaseModel):
    results: List[QueryHit]
    k: int
