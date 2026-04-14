#!/usr/bin/env python3
"""
RAG Knowledge Graph MCP Server
==============================
By MEOK AI Labs | https://meok.ai

Hybrid vector + knowledge graph RAG for regulatory and compliance data.
Tools: semantic_search, knowledge_graph_query, index_document, extract_entities, cross_reference
"""

import json
import os
import re
import hashlib
from typing import Optional, List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rag-knowledge-mcp")

# In-memory vector index (replace with pgvector/Neo4j in production)
_DOCUMENTS: dict[str, dict] = {}
_ENTITIES: dict[str, list] = {}
_GRAPH: dict[str, list] = {}


def _embed(text: str) -> List[float]:
    """Simple deterministic embedding for demo."""
    h = hashlib.md5(text.lower().encode()).hexdigest()
    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-9)


def _extract_entities(text: str) -> List[dict]:
    patterns = {
        "regulation": r"(GDPR|EU AI Act|NIST|ISO 42001|HIPAA|SOC 2|CSOAI)",
        "article": r"Article\s+\d+",
        "penalty": r"(€\d+[\d,.]*\s*million|fine|penalty)",
        "date": r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    }
    found = []
    for ent_type, pat in patterns.items():
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.append({"type": ent_type, "value": m.group(0), "start": m.start()})
    return found


@mcp.tool(name="semantic_search")
async def semantic_search(query: str, top_k: int = 5) -> str:
    """Semantic search over indexed documents."""
    if not _DOCUMENTS:
        return json.dumps({"results": [], "message": "No documents indexed yet. Use index_document first."})
    q_vec = _embed(query)
    scores = []
    for doc_id, doc in _DOCUMENTS.items():
        score = _cosine(q_vec, doc["embedding"])
        scores.append({"id": doc_id, "score": round(score, 4), "title": doc["title"], "snippet": doc["text"][:200]})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps({"query": query, "results": scores[:top_k]})


@mcp.tool(name="knowledge_graph_query")
async def knowledge_graph_query(entity: str, relation: Optional[str] = None) -> str:
    """Query the knowledge graph by entity and optional relation."""
    entity = entity.lower()
    matches = []
    for src, rels in _GRAPH.items():
        if entity in src.lower():
            for r in rels:
                if relation is None or relation.lower() in r["relation"].lower():
                    matches.append({"from": src, "to": r["target"], "relation": r["relation"], "weight": r.get("weight", 1.0)})
    return json.dumps({"entity": entity, "relation": relation, "matches": matches})


@mcp.tool(name="index_document")
async def index_document(title: str, text: str, doc_id: Optional[str] = None) -> str:
    """Index a document into vector store and knowledge graph."""
    doc_id = doc_id or hashlib.md5(text.encode()).hexdigest()[:12]
    embedding = _embed(text)
    _DOCUMENTS[doc_id] = {"title": title, "text": text, "embedding": embedding}
    entities = _extract_entities(text)
    _ENTITIES[doc_id] = entities
    # Build simple graph edges between regulations and articles
    regs = [e["value"] for e in entities if e["type"] == "regulation"]
    arts = [e["value"] for e in entities if e["type"] == "article"]
    for r in regs:
        if r not in _GRAPH:
            _GRAPH[r] = []
        for a in arts:
            _GRAPH[r].append({"target": a, "relation": "has_article", "source_doc": doc_id, "weight": 1.0})
    return json.dumps({"doc_id": doc_id, "indexed": True, "entities_found": len(entities)})


@mcp.tool(name="extract_entities")
async def extract_entities_tool(text: str) -> str:
    """Extract regulatory entities from text."""
    return json.dumps({"entities": _extract_entities(text)})


@mcp.tool(name="cross_reference")
async def cross_reference(term: str, framework_a: str, framework_b: str) -> str:
    """Find cross-references between two frameworks for a term."""
    q_vec = _embed(term)
    docs_a = []
    docs_b = []
    for doc_id, doc in _DOCUMENTS.items():
        score = _cosine(q_vec, doc["embedding"])
        fa = framework_a.lower()
        fb = framework_b.lower()
        if fa in doc["text"].lower() or fa in doc["title"].lower():
            docs_a.append({"id": doc_id, "score": round(score, 4), "title": doc["title"]})
        if fb in doc["text"].lower() or fb in doc["title"].lower():
            docs_b.append({"id": doc_id, "score": round(score, 4), "title": doc["title"]})
    docs_a.sort(key=lambda x: x["score"], reverse=True)
    docs_b.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps({
        "term": term,
        framework_a: docs_a[:3],
        framework_b: docs_b[:3],
        "note": "Cross-reference confidence based on semantic similarity."
    })


if __name__ == "__main__":
    mcp.run()
