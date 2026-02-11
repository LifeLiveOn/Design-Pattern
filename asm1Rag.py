import os
import json
import gradio as gr

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# ---------------- CONFIG ----------------
FILE_PATH = "course-desc.jsonl"
RETRIEVE_K = 5
PERSIST_DIR = ".chroma_course_desc"
COLLECTION = "course_desc"
BASE_URL = "http://127.0.0.1:11434"

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

# ---------------- LLM + EMBEDDINGS ----------------
embeddings = OllamaEmbeddings(
    model="granite4:3b",
    base_url=BASE_URL
)

llm = ChatOllama(
    model="granite4:3b",
    temperature=0.4,
    base_url=BASE_URL
)


# ---------------- DATA LOADING ----------------
def load_course_index(path: str) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = str(obj.get("id", "")).strip()
            if cid:
                index[cid] = obj
    return index


def load_documents(path: str) -> list[Document]:
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
                        page_content=desc.strip(),
                        metadata={"id": str(cid).strip(),
                                  "title": str(title).strip()},
                    )
                )
    return docs


COURSE_INDEX = load_course_index(FILE_PATH)
DOCS = load_documents(FILE_PATH)


# ---------------- VECTORSTORE ----------------
def build_vectorstore(docs: list[Document]) -> Chroma:
    if os.path.exists(PERSIST_DIR):
        return Chroma(
            embedding_function=embeddings,
            collection_name=COLLECTION,
            persist_directory=PERSIST_DIR,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )


VECTORSTORE = build_vectorstore(DOCS)


# ---------------- TOOLS ----------------
@tool
def search_by_id(course_id: str) -> str:
    """Look up a course by exact course ID."""
    course_id = str(course_id).strip()
    course = COURSE_INDEX.get(course_id)
    if not course:
        return f"No course found with ID {course_id}"

    return (
        f"Course ID: {course_id}\n"
        f"Title: {course.get('title')}\n"
        f"Description: {course.get('description')}"
    )


@tool
def search_by_title(title: str) -> str:
    """Look up a course by contain matching title in query."""
    title_norm = str(title).strip().lower()
    for course in COURSE_INDEX.values():
        if title_norm in str(course.get("title", "")).strip().lower():
            return (
                f"Course ID: {course.get('id')}\n"
                f"Title: {course.get('title')}\n"
                f"Description: {course.get('description')}"
            )
    return f"No course found with title '{title_norm}'"


@tool
def rag_search(query: str) -> str:
    """Semantic search over course descriptions; returns top matches."""
    query = str(query).strip()
    docs = VECTORSTORE.similarity_search(query, k=RETRIEVE_K)

    if not docs:
        return "No relevant courses found in the provided context."

    out = []
    for d in docs:
        out.append(
            f"Course ID: {d.metadata.get('id')}\n"
            f"Title: {d.metadata.get('title')}\n"
            f"Description: {d.page_content}"
        )
    print("RAG Search Results:", len(out))  # debug
    return "\n\n".join(out)


TOOLS = [rag_search, search_by_id, search_by_title]
llm = llm.bind_tools(TOOLS)

TOOL_MAP = {
    "rag_search": rag_search,
    "search_by_id": search_by_id,
    "search_by_title": search_by_title,
}


# ---------------- TOOL LOOP ----------------
def run_with_tools(messages: list, max_iters: int = 5) -> str:
    """
    messages: List[BaseMessage] (SystemMessage/HumanMessage/AIMessage/ToolMessage)
    """
    for _ in range(max_iters):
        ai = llm.invoke(messages)
        # print("AI:", ai)  # debug
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None)
        if not tool_calls:
            return (ai.content or "").strip() or "Model returned no answer."

        # Execute each tool call and add ToolMessage responses
        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})

            # Tool args may come as a JSON string in some backends
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {"query": tool_args}

            tool_fn = TOOL_MAP.get(tool_name)
            if not tool_fn:
                result = f"Unknown tool: {tool_name}"
            else:
                # LangChain tool.invoke expects a dict for structured args
                # e.g. rag_search({"query":"..."}) or search_by_id({"course_id":"..."})
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception as e:
                    result = f"Tool execution error in {tool_name}: {e}"

            messages.append(
                ToolMessage(
                    tool_call_id=call.get("id", ""),
                    content=str(result),
                )
            )
    final = llm.invoke(messages)
    return (final.content or "").strip() or "Model returned no answer."


# ---------------- GRADIO ----------------
def answer(student_query: str, history=None):
    messages: list = [SystemMessage(content=SYSTEM)]

    # Gradio ChatInterface history is typically list[tuple[str,str]] or list[dict]
    # Handle both safely.
    if history:
        for item in history[-6:]:
            # tuple format: (user, assistant)
            if isinstance(item, (list, tuple)) and len(item) == 2:
                u, a = item
                if u:
                    messages.append(HumanMessage(content=str(u)))
                if a:
                    messages.append(AIMessage(content=str(a)))
            # dict format: {"role": "...", "content": "..."}
            elif isinstance(item, dict) and "role" in item and "content" in item:
                role = item["role"]
                content = item["content"]
                if role == "user":
                    messages.append(HumanMessage(content=str(content)))
                elif role == "assistant":
                    messages.append(AIMessage(content=str(content)))

    messages.append(HumanMessage(content=str(student_query)))

    return run_with_tools(messages)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## Academic Course Advisor")

        gr.ChatInterface(
            fn=answer,
            title="Chat Advisor",
            examples=[
                "What are introductory machine learning courses?",
                "Look up course ID 4309",
                "Can you find a course titled 'Data Mining'?",
                "What courses should I take if I'm interested in natural language processing?",
            ],
        )

    demo.launch()
