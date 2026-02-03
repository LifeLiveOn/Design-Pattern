import os
import json
import gradio as gr

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
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
Only recommend courses that are in the provided context. Answer only questions related to academic courses. (IF) the question is not related to academic courses, respond with "I can only answer questions related to academic courses."
"""

embeddings = OllamaEmbeddings(
    model="granite4:3b",
    base_url=BASE_URL
)

llm = ChatOllama(
    model="granite4:3b",
    temperature=0.1,
    base_url=BASE_URL
)


def load_course_index(path: str) -> dict[str, dict]:
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = str(obj.get("id", "")).strip()
            if cid:
                index[cid] = obj
    return index


def load_documents(path: str) -> list[Document]:
    docs = []
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
                        metadata={"id": cid.strip(), "title": title.strip()},
                    )
                )
    return docs


COURSE_INDEX = load_course_index(FILE_PATH)
DOCS = load_documents(FILE_PATH)


def build_vectorstore(docs: list[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )


VECTORSTORE = build_vectorstore(DOCS)


@tool
def search_by_id(course_id: str) -> str:
    """Look up a course by exact course ID."""
    course = COURSE_INDEX.get(course_id.strip())
    if not course:
        return f"No course found with ID {course_id}"

    return (
        f"Course ID: {course_id}\n"
        f"Title: {course.get('title')}\n"
        f"Description: {course.get('description')}"
    )


@tool
def rag_search(query: str) -> str:
    """Search for relevant courses using semantic retrieval."""
    docs = VECTORSTORE.similarity_search(query, k=RETRIEVE_K)

    if not docs:
        return "No relevant courses found."

    out = []
    for d in docs:
        out.append(
            f"Course ID: {d.metadata.get('id')}\n"
            f"Title: {d.metadata.get('title')}\n"
            f"Description: {d.page_content}"
        )

    return "\n\n".join(out)


TOOLS = [search_by_id, rag_search]
llm = llm.bind_tools(TOOLS)


def run_with_tools(messages: list[dict]) -> str:
    response = llm.invoke(messages)

    # No tool → normal response
    if not response.tool_calls:
        return response.content or ""

    # Execute tools
    for call in response.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        if tool_name == "search_by_id":
            result = search_by_id.invoke(tool_args)

        elif tool_name == "rag_search":
            result = rag_search.invoke(tool_args)

        else:
            result = f"Unknown tool: {tool_name}"

        messages.append(
            ToolMessage(
                tool_call_id=call["id"],
                content=result
            )
        )

    # FORCE final reasoning turn (prevents blank output)
    messages.append({
        "role": "user",
        "content": "Using the tool results above, answer the question clearly."
    })

    final = llm.invoke(messages)
    return final.content or ""


def answer(student_query: str, history=None) -> str:
    messages = [{"role": "system", "content": SYSTEM}]

    for msg in (history or [])[-6:]:
        if "role" in msg and "content" in msg:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    messages.append({"role": "user", "content": student_query})

    # print("Messages:", messages)  # debug

    return run_with_tools(messages)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## Academic Course Advisor")

        gr.ChatInterface(
            fn=answer,
            title="Chat Advisor",
            examples=[
                "What are introductory machine learning courses?",
                "Look up course ID 4309"
            ],
        )

    demo.launch()
