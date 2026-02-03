# RAG Q&A Course Info

A lightweight Retrieval-Augmented Generation (RAG) demo that serves course information via a Gradio chat interface. It uses ChromaDB for embeddings and cosine similarity, with an ID lookup path for special cases.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

## Installation

```bash
# Install dependencies with uv (creates .venv by default) need a uv.lock file
uv sync
```

## Running

```bash
uv run asm1Rag.py
```

Starts the Gradio UI in your browser.

## Features

- **Gradio chat interface** for course Q&A.
- **ChromaDB** as the embedding database with **cosine similarity** retrieval.
- **ID lookup** to handle special cases.
- Toggle retrieval options via callable functions in the UI.

## Notes

- Ensure your embeddings and course data are initialized per your ChromaDB setup before running.
- Adjust environment variables or config in `asm1Rag.py` as needed.
