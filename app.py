"""College Chatbot — local RAG powered by Ollama and Streamlit."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import ollama
import streamlit as st

MODEL = "qwen3:8b"
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40


@dataclass
class Chunk:
    text: str
    source: str
    number: int


def extract_text(uploaded_file) -> str:
    """Read text, PDF, or DOCX uploads without saving report contents to disk."""
    data = uploaded_file.getvalue()
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if suffix == "txt":
        return data.decode("utf-8", errors="replace")
    if suffix == "pdf":
        from pypdf import PdfReader
        from io import BytesIO

        return "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
    if suffix == "docx":
        from docx import Document
        from io import BytesIO

        document = Document(BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError("Please upload a .txt, .pdf, or .docx report.")


def chunk_text(text: str, source: str) -> list[Chunk]:
    words = re.sub(r"\s+", " ", text).strip().split()
    chunks = []
    for start in range(0, len(words), CHUNK_WORDS - CHUNK_OVERLAP):
        section = " ".join(words[start : start + CHUNK_WORDS])
        if section:
            chunks.append(Chunk(section, source, len(chunks) + 1))
    return chunks


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def retrieve(question: str, chunks: list[Chunk], limit: int = 4) -> list[Chunk]:
    """Small, dependency-free BM25 retriever: ideal for a single college report."""
    query = tokens(question)
    if not query or not chunks:
        return []
    corpus = [tokens(chunk.text) for chunk in chunks]
    document_frequency = Counter(term for doc in corpus for term in set(doc))
    average_length = sum(map(len, corpus)) / len(corpus)
    scores = []
    for chunk, doc in zip(chunks, corpus):
        frequencies = Counter(doc)
        score = 0.0
        for term in query:
            if term not in frequencies:
                continue
            idf = math.log(1 + (len(corpus) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * (frequencies[term] * 2.0) / (frequencies[term] + 1.0 * (1 - 0.75 + 0.75 * len(doc) / average_length))
        scores.append(score)
    ranked = [pair for pair in sorted(zip(scores, chunks), key=lambda item: item[0], reverse=True) if pair[0] > 0]
    return [chunk for _, chunk in ranked[:limit]]


def is_conversational(question: str) -> bool:
    """Recognise short social messages that should not need report retrieval."""
    normalized = " ".join(tokens(question))
    phrases = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "thanks", "thank you", "help", "what can you do",
    }
    return normalized in phrases


def conversational_reply(question: str) -> str:
    if "thank" in question.lower():
        return "You’re welcome! Ask me anything about the uploaded college report."
    if "how are you" in question.lower():
        return "I’m ready to help you explore the uploaded college report. What would you like to know?"
    return (
        "Hi! I’m your college report assistant. Ask me about departments, programmes, "
        "staff roles, facilities, or anything else covered in the uploaded report."
    )


def answer(question: str, context: list[Chunk], history: list[dict]) -> str:
    sources = "\n\n".join(f"[Source {item.number}]\n{item.text}" for item in context)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful college-report assistant. Answer only using the supplied report excerpts. "
                "If the excerpts do not contain the answer, say so plainly. Do not invent facts. "
                "Cite factual claims with [Source N]. Be concise and clear."
            ),
        },
        *history[-6:],
        {"role": "user", "content": f"Report excerpts:\n{sources}\n\nQuestion: {question}"},
    ]
    response = ollama.chat(model=MODEL, messages=messages, options={"temperature": 0.2})
    return response["message"]["content"]


st.set_page_config(page_title="College Chatbot", page_icon="🎓", layout="wide")
st.title("🎓 College Chatbot")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Report")
    uploaded = st.file_uploader("Upload college report", type=["txt", "pdf", "docx"])
    if uploaded and (st.session_state.get("report_name") != uploaded.name):
        try:
            text = extract_text(uploaded)
            st.session_state.chunks = chunk_text(text, uploaded.name)
            st.session_state.report_name = uploaded.name
            st.session_state.messages = []
            st.success(f"Indexed {len(st.session_state.chunks)} sections from {uploaded.name}")
        except Exception as error:
            st.error(f"Could not read that report: {error}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"Answer model: `{MODEL}`")

if not st.session_state.chunks:
    st.info("Upload a college report in the sidebar to begin.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about departments, roles, programs, or people…")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        context = retrieve(question, st.session_state.chunks)
        with st.chat_message("assistant"):
            if is_conversational(question):
                response = conversational_reply(question)
                st.markdown(response)
            elif not context:
                response = "I couldn't find relevant information in the uploaded report. Try rephrasing your question."
                st.markdown(response)
            else:
                try:
                    with st.spinner("Searching the report and drafting an answer…"):
                        response = answer(question, context, st.session_state.messages)
                    st.markdown(response)
                    with st.expander("Retrieved report excerpts"):
                        for item in context:
                            st.markdown(f"**[Source {item.number}] {item.source}**\n\n{item.text}")
                except Exception as error:
                    response = (
                        "I couldn't reach Ollama. Start it with `ollama serve`, confirm the model is installed "
                        f"with `ollama list`, then retry. Details: `{error}`"
                    )
                    st.error(response)
        st.session_state.messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ])
