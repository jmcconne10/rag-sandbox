from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from rag import ingest_folder, retrieve
from models import IngestResponse, QueryRequest, QueryResponse, QueryHit

app = FastAPI(title="RAG Sandbox API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.getenv("DATA_DIR", "/data/knowledge")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest() -> IngestResponse:
    # run sync function off the event loop
    res: Dict[str, Any] = await run_in_threadpool(ingest_folder, DATA_DIR)
    return IngestResponse(**res)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    res: Dict[str, Any] = await run_in_threadpool(retrieve, req.query, req.k)
    hits = [QueryHit(**h) for h in res.get("results", [])]
    return QueryResponse(results=hits, k=res.get("k", req.k))
