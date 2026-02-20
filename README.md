# RAG Q&A Course Info

A lightweight Retrieval-Augmented Generation (RAG) demo that serves course information via a Gradio chat interface. It uses ChromaDB for embeddings and cosine similarity, with an ID lookup path for special cases. Model is Granite (IBM) 4.0

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Ollama (Granite 4: 3B)

## Installation

```bash
ollama run granite4:3b #download granite4:3b model
ollama serve # start the ollama server to receive input
cd Design-Pattern
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
- Chromasqlite3 is located in **.chroma_course_desc** this will always run and recreate new chroma regardless of exist.
- **ID lookup** to handle special cases.
- Toggle retrieval options via callable functions in the UI.

## Notes for TA, Grader

- Ensure your embeddings and course data are initialized ChromaDB setup, it will take time for the script to load every information to embedding space.
- Adjust environment variables or config in `asm1Rag.py` as needed.
