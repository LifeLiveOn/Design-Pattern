import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import gradio as gr
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class RagConfig:
    file_path: str = "course-desc.jsonl"
    retrieve_k: int = 5
    persist_dir: str = ".chroma_course_desc"
    collection: str = "course_desc"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "granite4:3b"
    temperature: float = 0.1


SYSTEM = """You are an expert academic advisor.
Use provided context and tools to answer.
Always cite course ID and title in your answer.
Only recommend courses that are in the provided context or returned by tools.
Answer only questions related to academic courses.
If no relevant courses are found, say:
\"No relevant courses were found in the provided context.\"
"""

FALLBACK = "No relevant courses were found in the provided context."


def format_course(course: dict[str, Any]) -> str:
    return (
        f"Course ID: {course.get('id')}\n"
        f"Title: {course.get('title')}\n"
        f"Description: {course.get('description')}"
    )


def format_docs(docs: Sequence[Document]) -> str:
    lines = [f"Here are relevant courses found: {len(docs)} courses.\n"]
    for d in docs:
        lines.append(
            f"- Course ID: {d.metadata.get('id', '?')} — {d.metadata.get('title', 'Untitled')}\n"
            f"  Description: {d.page_content}"
        )
    return "\n".join(lines)


class RetrievalStrategy(Protocol):
    def retrieve(self, query: str, k: int) -> Sequence[Document]: ...


@dataclass(frozen=True)
class TopNStrategy:
    vs: Any

    def retrieve(self, query: str, k: int) -> Sequence[Document]:
        return self.vs.similarity_search(query, k=k)


@dataclass(frozen=True)
class WindowStrategy:
    vs: Any
    chunks_by_course: dict[str, list[Document]]
    radius: int = 1

    def retrieve(self, query: str, k: int) -> Sequence[Document]:
        anchors = list(self.vs.similarity_search(query, k=max(1, min(3, k))))
        out: list[Document] = []
        seen: set[tuple[str, int]] = set()
        for anchor in anchors:
            cid = str(anchor.metadata.get("id", "")).strip()
            idx = int(anchor.metadata.get("chunk_index", 0))
            chunks = self.chunks_by_course.get(cid, [])
            for c in chunks[max(0, idx - self.radius): idx + self.radius + 1]:
                key = (str(c.metadata.get("id", "")).strip(),
                       int(c.metadata.get("chunk_index", 0)))
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
        return out[:k] or list(self.vs.similarity_search(query, k=k))


@dataclass(frozen=True)
class DocumentStrategy:
    vs: Any

    def retrieve(self, query: str, k: int) -> Sequence[Document]:
        docs = list(self.vs.similarity_search(query, k=max(10, k * 4)))
        out: list[Document] = []
        seen_ids: set[str] = set()
        for d in docs:
            cid = str(d.metadata.get("id", "")).strip()
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            out.append(d)
            if len(out) >= k:
                break
        return out


@dataclass(frozen=True)
class HierarchicalStrategy:
    vs: Any

    def retrieve(self, query: str, k: int) -> Sequence[Document]:
        coarse_ids = {
            str(d.metadata.get("id", "")).strip()
            for d in DocumentStrategy(self.vs).retrieve(query, max(2, k))
        }
        fine = list(self.vs.similarity_search(query, k=max(20, k * 6)))
        ranked = [d for d in fine if str(d.metadata.get("id", "")).strip() in coarse_ids] + [
            d for d in fine if str(d.metadata.get("id", "")).strip() not in coarse_ids
        ]
        out: list[Document] = []
        seen: set[tuple[str, int]] = set()
        for d in ranked:
            key = (str(d.metadata.get("id", "")).strip(),
                   int(d.metadata.get("chunk_index", 0)))
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
            if len(out) >= k:
                break
        return out


class RagApp:
    def __init__(self, cfg: RagConfig):
        self.cfg = cfg
        self.llm = ChatOllama(
            model=cfg.model, temperature=cfg.temperature, base_url=cfg.base_url)
        self.emb = OllamaEmbeddings(model=cfg.model, base_url=cfg.base_url)

        self.course_index, docs = self._load_courses(cfg.file_path)
        chunks = self._chunk_docs(docs)
        self.chunks_by_course = self._group_by_course(chunks)
        # check dir exists and has data, otherwise add to vector store

        self.vs = Chroma.from_documents(
            documents=chunks,
            embedding=self.emb,
            collection_name=cfg.collection,
            persist_directory=cfg.persist_dir,
        )

        self.strategies: dict[str, RetrievalStrategy] = {
            "Top N": TopNStrategy(self.vs),
            "Window": WindowStrategy(self.vs, self.chunks_by_course),
            "Document": DocumentStrategy(self.vs),
            "Hierarchical": HierarchicalStrategy(self.vs),
        }
        self.current_strategy = "Top N"
        self.tools = self._build_tools()
        self.tool_map = {t.name: t for t in self.tools}

    def _build_tools(self):
        @tool
        def get_course_by_id(course_id: str) -> str:
            """Get exact course details by numeric course ID (e.g., 4361)."""
            cid = str(course_id).strip()
            course = self.course_index.get(cid)
            return format_course(course) if course else f"No course found with ID {cid}"

        @tool
        def find_course_by_title(title: str) -> str:
            """Find the first course whose title contains the provided text. or similar to the provided text.
            ex: "design patterns" should match "software design patterns"
            """
            t = str(title).strip().lower()
            if not t:
                return "No title provided."
            for course in self.course_index.values():
                if t in str(course.get("title", "")).lower():
                    return format_course(course)
            return f"No course found with title '{title}'"

        return [get_course_by_id, find_course_by_title]

    def _retrieve_docs(self, query: str) -> Sequence[Document]:
        """
        Retrieve relevant course documents based on the current strategy.
        """
        docs = list(self.strategies[self.current_strategy].retrieve(
            query, self.cfg.retrieve_k))
        out: list[Document] = []
        seen_ids: set[str] = set()
        for d in docs:
            cid = str(d.metadata.get("id", "")).strip()
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            out.append(d)
        return out

    def _load_courses(self, path: str) -> tuple[dict[str, dict[str, Any]], list[Document]]:
        index: dict[str, dict[str, Any]] = {}
        docs: list[Document] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = str(obj.get("id", "")).strip()
                title = str(obj.get("title", "Untitled")).strip()
                desc = str(obj.get("description", "")).strip()
                if not cid:
                    continue
                index[cid] = {"id": cid, "title": title, "description": desc}
                if desc:
                    docs.append(Document(page_content=desc, metadata={
                                "id": cid, "title": title}))
        return index, docs

    def _chunk_docs(self, docs: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100)
        out: list[Document] = []
        for d in docs:
            for i, text in enumerate(splitter.split_text(d.page_content)):
                md = dict(d.metadata)
                md["chunk_index"] = i
                out.append(Document(page_content=text, metadata=md))
        return out

    def _group_by_course(self, chunks: list[Document]) -> dict[str, list[Document]]:
        grouped: dict[str, list[Document]] = {}
        for c in chunks:
            grouped.setdefault(
                str(c.metadata.get("id", "")).strip(), []).append(c)
        for cid in grouped:
            grouped[cid].sort(key=lambda d: int(
                d.metadata.get("chunk_index", 0)))
        return grouped

    def _ask_llm(self, question: str, context: str) -> str:
        llm_with_tools = self.llm.bind_tools(self.tools)
        messages: list = [
            SystemMessage(content=SYSTEM),
            HumanMessage(
                content=(
                    f"Course context:\n{context or '(none)'}\n\n"
                    f"Student question:\n{question}\n\n"
                    f"Use tools when user asks by ID or title."
                )
            ),
        ]

        ai = llm_with_tools.invoke(messages)
        for _ in range(5):
            tool_calls = getattr(ai, "tool_calls", [])
            if not tool_calls:
                break
            messages.append(ai)
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args", {})
                tool_fn = self.tool_map.get(name)
                result = tool_fn.invoke(
                    args) if tool_fn else f"Tool '{name}' not found"
                messages.append(ToolMessage(content=str(
                    result), tool_call_id=call.get("id")))
            ai = llm_with_tools.invoke(messages)

        return (ai.content or "").strip()

    def answer(self, student_query: str, history=None, retrieval_mode: str = "Top N") -> str:
        question = str(student_query or "").strip()
        if not question:
            return "Ask a course-related question."
        if retrieval_mode in self.strategies:
            self.current_strategy = retrieval_mode

        docs = list(self._retrieve_docs(question))
        context = "\n\n".join(
            f"Course ID: {d.metadata.get('id')}\nTitle: {d.metadata.get('title')}\nDescription: {d.page_content}"
            for d in docs
        )
        answer = self._ask_llm(question, context)
        if not answer or answer.lower().startswith("no relevant courses were found"):
            return format_docs(docs) if docs else FALLBACK
        return answer


def main() -> None:
    app = RagApp(RagConfig())
    with gr.Blocks() as demo:
        gr.Markdown("## Academic Course Advisor (Strategy + Tool Calling)")
        strategy = gr.Dropdown(
            ["Top N", "Window", "Document", "Hierarchical"],
            value="Top N",
            label="Retrieval Strategy",
        )
        gr.ChatInterface(
            fn=app.answer,
            additional_inputs=[strategy],
            title="Chat Advisor",
            examples=[
                ["What courses teach machine learning?", "Top N"],
                ["Look up course ID 4361", "Document"],
                ["Find a course titled Data Mining", "Window"],
                ["Which course teaches materials for design patterns?", "Document"],
                ["Can you recommend courses related to databases?", "Hierarchical"]
            ],
        )
    demo.launch()


if __name__ == "__main__":
    main()
