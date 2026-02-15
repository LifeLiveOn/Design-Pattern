from __future__ import annotations

import os
import re
import json
import shutil
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

import gradio as gr
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Provider implementations (Ollama now; OpenAI/Gemini later)
from langchain_ollama import OllamaEmbeddings, ChatOllama


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class RagConfig:
    file_path: str = "course-desc.jsonl"
    retrieve_k: int = 5
    persist_dir: str = ".chroma_course_desc"
    collection: str = "course_desc"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "granite4:3b"
    temperature: float = 0.4


SYSTEM = """You are an expert academic advisor.
Use ONLY the provided course context to answer.
Always cite course ID and title in your answer.
Only recommend courses that are in the provided context.
Answer only questions related to academic courses.
(IF) the question is not related to academic courses, respond with:
"I can only answer questions related to academic courses."
(IF) no relevant courses are found, explicitly say:
"No relevant courses were found in the provided context."
"""


# =========================
# ABSTRACT FACTORY (3rd pattern)
# =========================
class LlmProviderFactory(Protocol):
    def create_chat(self) -> Any: ...
    def create_embeddings(self) -> Any: ...


@dataclass(frozen=True)
class OllamaFactory:
    base_url: str
    model: str
    temperature: float = 0.4

    def create_chat(self) -> Any:
        return ChatOllama(
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )

    def create_embeddings(self) -> Any:
        return OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
        )


# =========================
# STRATEGY PATTERN
# =========================
class RetrievalStrategy(Protocol):
    def retrieve(self, query: str, k: int) -> Sequence[Document]: ...


@dataclass(frozen=True)
class ChromaTopKStrategy:
    vectorstore: Any

    def retrieve(self, query: str, k: int) -> Sequence[Document]:
        return self.vectorstore.similarity_search(query, k=k)


# =========================
# CHAIN OF RESPONSIBILITY
# =========================
@dataclass(frozen=True)
class HandlerResult:
    handled: bool
    # final answer to show user (for ID/title/fallback)
    answer: Optional[str] = None
    # context for LLM generation (for semantic handler)
    context: Optional[str] = None
    # retrieved docs (for deterministic formatting)
    docs: Optional[Sequence[Document]] = None


class BaseHandler:
    def __init__(self) -> None:
        self._next: Optional["BaseHandler"] = None

    def set_next(self, nxt: "BaseHandler") -> "BaseHandler":
        self._next = nxt
        return nxt

    def _pass(self, query: str) -> HandlerResult:
        if not self._next:
            return HandlerResult(handled=True, answer="No relevant courses were found in the provided context.")
        return self._next.handle(query)

    def handle(self, query: str) -> HandlerResult:
        return self._pass(query)


@dataclass
class CourseIdHandler(BaseHandler):
    course_index: dict[str, dict]

    def __post_init__(self) -> None:
        super().__init__()

    def handle(self, query: str) -> HandlerResult:
        # Match standalone 4-digit IDs (e.g., "4361") and typical phrases.
        m = re.search(r"\b(\d{4})\b", query)
        if not m:
            return self._pass(query)

        cid = m.group(1)
        course = self.course_index.get(cid)
        if not course:
            return HandlerResult(handled=True, answer=f"No course found with ID {cid}")

        ans = (
            f"Course ID: {cid}\n"
            f"Title: {course.get('title')}\n"
            f"Description: {course.get('description')}"
        )
        return HandlerResult(handled=True, answer=ans)


@dataclass
class CourseTitleHandler(BaseHandler):
    course_index: dict[str, dict]

    def __post_init__(self) -> None:
        super().__init__()

    def handle(self, query: str) -> HandlerResult:
        q = query.lower()

        # Only trigger if user is probably asking by title.
        # (You can tune this if your professor tests weird phrasing.)
        title_trigger = any(kw in q for kw in [
                            "title", "titled", "called", "named"])
        has_quotes = ("'" in query) or ('"' in query)
        if not (title_trigger or has_quotes):
            return self._pass(query)

        # Prefer quoted title if present
        qm = re.search(r"['\"]([^'\"]+)['\"]", query)
        needle = (qm.group(1) if qm else query).strip().lower()

        for course in self.course_index.values():
            title = str(course.get("title", "")).strip().lower()
            if needle and needle in title:
                ans = (
                    f"Course ID: {course.get('id')}\n"
                    f"Title: {course.get('title')}\n"
                    f"Description: {course.get('description')}"
                )
                return HandlerResult(handled=True, answer=ans)

        return HandlerResult(handled=True, answer=f"No course found with title '{needle}'")


@dataclass
class SemanticRagHandler(BaseHandler):
    retrieval: RetrievalStrategy
    k: int

    def __post_init__(self) -> None:
        super().__init__()

    def handle(self, query: str) -> HandlerResult:
        docs = list(self.retrieval.retrieve(query, self.k))
        if not docs:
            return self._pass(query)

        context = "\n\n".join(
            f"Course ID: {d.metadata.get('id')}\n"
            f"Title: {d.metadata.get('title')}\n"
            f"Description: {d.page_content}"
            for d in docs
        )
        return HandlerResult(handled=True, context=context, docs=docs)


class FallbackHandler(BaseHandler):
    def handle(self, query: str) -> HandlerResult:
        return HandlerResult(handled=True, answer="No relevant courses were found in the provided context.")


# =========================
# APP
# =========================
class RagApp:
    def __init__(self, cfg: RagConfig, factory: LlmProviderFactory):
        self.cfg = cfg

        # Abstract Factory creates provider-specific implementations
        self.embeddings = factory.create_embeddings()
        self.llm = factory.create_chat()

        self.course_index = self._load_course_index(cfg.file_path)
        docs = self._load_documents(cfg.file_path)
        self.vectorstore = self._build_vectorstore(docs)

        # Strategy
        self.retrieval_strategy: RetrievalStrategy = ChromaTopKStrategy(
            self.vectorstore)

        # Chain of Responsibility
        self.chain = self._build_chain()

    def _build_chain(self) -> BaseHandler:
        id_h = CourseIdHandler(self.course_index)
        title_h = CourseTitleHandler(self.course_index)
        sem_h = SemanticRagHandler(
            self.retrieval_strategy, k=self.cfg.retrieve_k)
        fb_h = FallbackHandler()

        id_h.set_next(title_h).set_next(sem_h).set_next(fb_h)
        return id_h

    def _load_course_index(self, path: str) -> dict[str, dict]:
        index: dict[str, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = str(obj.get("id", "")).strip()
                if cid:
                    index[cid] = obj
        return index

    def _load_documents(self, path: str) -> list[Document]:
        docs: list[Document] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = obj.get("id")
                desc = obj.get("description", "")
                title = obj.get("title", "Untitled")

                if cid and desc:
                    docs.append(
                        Document(
                            page_content=str(desc).strip(),
                            metadata={
                                "id": str(cid).strip(),
                                "title": str(title).strip(),
                            },
                        )
                    )
        return docs

    def _vector_count(self, vectorstore: Chroma) -> int:
        """Best-effort count of vectors; returns 0 on failure."""
        try:
            # type: ignore[attr-defined]
            return vectorstore._collection.count()
        except Exception:
            return 0

    def _build_vectorstore(self, docs: list[Document]) -> Chroma:
        cfg = self.cfg
        # Try loading an existing vector store first; if it fails, rebuild from documents.
        if os.path.exists(cfg.persist_dir):
            print("Loading existing vectorstore from disk...")
            try:
                vs = Chroma(
                    embedding_function=self.embeddings,
                    collection_name=cfg.collection,
                    persist_directory=cfg.persist_dir,
                )
                if self._vector_count(vs) > 0:
                    return vs
                print("Existing vectorstore is empty; rebuilding...")
                shutil.rmtree(cfg.persist_dir, ignore_errors=True)
            except Exception as exc:
                print(f"Failed to load vectorstore, rebuilding... ({exc})")

        print("Creating vectorstore from documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        return Chroma.from_documents(
            documents=chunks,
            embedding_function=self.embeddings,
            collection_name=cfg.collection,
            persist_directory=cfg.persist_dir,
        )

    def _format_docs_answer(self, docs: Sequence[Document]) -> str:
        lines: list[str] = [
            "Here are relevant courses from the provided context:"]
        for d in docs:
            cid = d.metadata.get("id", "?")
            title = d.metadata.get("title", "Untitled")
            desc = d.page_content
            lines.append(
                f"- Course ID: {cid} — {title}\n  Description: {desc}")
        return "\n".join(lines)

    def _generate_from_context(self, question: str, context: str) -> str:
        messages: list = [SystemMessage(content=SYSTEM)]

        prompt = (
            f"Course context:\n{context}\n\n"
            f"Student question:\n{question}\n\n"
            f"Instructions:\n"
            f"- Use ONLY the course context.\n"
            f"- Always cite course ID and title.\n"
            f"- At least one course is provided in the context; do NOT answer with the fallback unless the context is empty.\n"
            f"- If and only if the context is empty, say: "
            f"\"No relevant courses were found in the provided context.\""
        )
        messages.append(HumanMessage(content=prompt))

        ai = self.llm.invoke(messages)
        raw = (ai.content or "").strip()
        if not raw:
            return "Model returned no answer."
        if raw.lower().strip() == "no relevant courses were found in the provided context.":
            return "Model returned fallback unexpectedly. Please try rephrasing."
        return raw

    def answer(self, student_query: str, history=None) -> str:
        q = str(student_query or "").strip()
        if not q:
            return "Ask a course-related question."

        result = self.chain.handle(q)

        # ID/title/fallback produce direct answers (no LLM rewrite needed)
        if result.answer is not None:
            return result.answer

        # semantic handler returns context -> generate natural answer
        if result.context is not None:
            llm_answer = self._generate_from_context(q, result.context)
            if (not llm_answer or llm_answer.lower().strip().startswith("model returned fallback")
                    or llm_answer.lower().strip().startswith("no relevant courses were found")) and result.docs:
                return self._format_docs_answer(result.docs)
            return llm_answer

        # should never happen
        return "No relevant courses were found in the provided context."


def main():
    cfg = RagConfig()

    # Choose provider factory (Ollama now)
    factory = OllamaFactory(base_url=cfg.base_url,
                            model=cfg.model, temperature=cfg.temperature)

    app = RagApp(cfg, factory)

    with gr.Blocks() as demo:
        gr.Markdown(
            "## Academic Course Advisor (Strategy + CoR + Abstract Factory)")

        gr.ChatInterface(
            fn=app.answer,
            title="Chat Advisor",
            examples=[
                "What are introductory machine learning courses?",
                "Look up course ID 4361",
                "Give me the course description for 4361",
                "Can you find a course titled 'Data Mining'?",
                "Which course teaches design patterns?",
            ],
        )

    demo.launch()


if __name__ == "__main__":
    main()
