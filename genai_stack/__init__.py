"""Code knowledge graph over a TS/JS repository, and GraphRAG on top of it.

Layout:
    settings.py     environment configuration, read once
    extract.py      tree-sitter parsing of a repo into symbols and edges
    code_index.py   reshaping that index, resolving imports to files
    schema.py       Neo4j labels, constraints and indexes
    ingest.py       writing nodes and relationships into Neo4j
    retrieval.py    the graph traversal query and the vector store
    llm.py          loading chat and embedding models
    chains.py       the LLM-only and GraphRAG chains

The apps in ../apps are thin entry points over this package.
"""
