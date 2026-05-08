<div align="center">

# Rag Knowledge MCP

**MCP server for rag knowledge mcp operations**

[![PyPI](https://img.shields.io/pypi/v/meok-rag-knowledge-mcp)](https://pypi.org/project/meok-rag-knowledge-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MEOK AI Labs](https://img.shields.io/badge/MEOK_AI_Labs-MCP_Server-purple)](https://meok.ai)

</div>

## Overview

Rag Knowledge MCP provides AI-powered tools via the Model Context Protocol (MCP).

## Tools

| Tool | Description |
|------|-------------|
| `semantic_search` | Semantic search over indexed documents. |
| `knowledge_graph_query` | Query the knowledge graph by entity and optional relation. |
| `index_document` | Index a document into vector store and knowledge graph. |
| `extract_entities_tool` | Extract regulatory entities from text. |
| `cross_reference` | Find cross-references between two frameworks for a term. |

## Installation

```bash
pip install meok-rag-knowledge-mcp
```

## Usage with Claude Desktop

Add to your Claude Desktop MCP config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rag-knowledge": {
      "command": "python",
      "args": ["-m", "meok_rag_knowledge_mcp.server"]
    }
  }
}
```

## Usage with FastMCP

```python
from mcp.server.fastmcp import FastMCP

# This server exposes 5 tool(s) via MCP
# See server.py for full implementation
```

## License

MIT © [MEOK AI Labs](https://meok.ai)
