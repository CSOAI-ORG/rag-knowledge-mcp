# Rag Knowledge

> By [MEOK AI Labs](https://meok.ai) — MEOK AI Labs MCP Server

RAG Knowledge Graph MCP Server

## Installation

```bash
pip install rag-knowledge-mcp
```

## Usage

```bash
# Run standalone
python server.py

# Or via MCP
mcp install rag-knowledge-mcp
```

## Tools

### `semantic_search`
Semantic search over indexed documents.

**Parameters:**
- `query` (str)
- `top_k` (int)

### `knowledge_graph_query`
Query the knowledge graph by entity and optional relation.

**Parameters:**
- `entity` (str)
- `relation` (str)

### `index_document`
Index a document into vector store and knowledge graph.

**Parameters:**
- `title` (str)
- `text` (str)
- `doc_id` (str)

### `extract_entities_tool`
Extract regulatory entities from text.

**Parameters:**
- `text` (str)

### `cross_reference`
Find cross-references between two frameworks for a term.

**Parameters:**
- `term` (str)
- `framework_a` (str)
- `framework_b` (str)


## Authentication

Free tier: 15 calls/day. Upgrade at [meok.ai/pricing](https://meok.ai/pricing) for unlimited access.

## Links

- **Website**: [meok.ai](https://meok.ai)
- **GitHub**: [CSOAI-ORG/rag-knowledge-mcp](https://github.com/CSOAI-ORG/rag-knowledge-mcp)
- **PyPI**: [pypi.org/project/rag-knowledge-mcp](https://pypi.org/project/rag-knowledge-mcp/)

## License

MIT — MEOK AI Labs
