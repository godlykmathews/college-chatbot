# College Chatbot

### Mulearn AI Chatbot task - #cl-ai-chatbot

A local chatbot for asking focused questions about a college report: departments, programmes, staff roles, and other information in the supplied document. It runs entirely on your computer and generates answers with the Ollama model `gemma4:12b`.

## How it works

This project uses **retrieval-augmented generation (RAG)**. When you upload a `.txt`, `.pdf`, or `.docx` college report, the app splits it into small overlapping sections. For each question, a local BM25 search selects the most relevant sections and sends only those excerpts, along with the question, to Gemma through Ollama. The response cites the retrieved sections as `[Source N]` and is instructed to say when the report does not contain an answer.

This makes the chatbot useful for a small, changing college report without training or fine-tuning a model. The report is processed in memory and is not uploaded to a third-party service.

[![Watch the walkthrough](https://img.youtube.com/vi/qt1IcDaCbO0/hqdefault.jpg)](https://youtu.be/qt1IcDaCbO0)

Watch the [How it works video walkthrough](https://youtu.be/qt1IcDaCbO0).

## Setup

Prerequisites:

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed
- The `gemma4:12b` model available locally

From this project folder, create a virtual environment and install the app dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start Ollama in another terminal if it is not already running:

```bash
ollama serve
```

Verify that `gemma4:12b` appears in `ollama list`. If it does not, pull the exact model tag supplied by your Ollama installation, for example:

```bash
ollama pull gemma4:12b
```

Then start the web interface:

```bash
streamlit run app.py
```

Open the local address Streamlit prints (usually `http://localhost:8501`), upload your college report in the sidebar, and ask questions in the chat field.

## Using the chatbot

Try questions such as:

- “Which departments are described in the report?”
- “Who is the head of the Computer Science department?”
- “What programmes does the college offer?”

Use the **Retrieved report excerpts** panel beneath an answer to check its supporting passages. Uploading a different report starts a fresh conversation.

## Challenges and solutions

- **Grounding answers:** language models can make plausible but unsupported statements. The app retrieves relevant passages first, requires source markers, and tells the model to admit when evidence is missing.
- **Keeping the app simple and local:** rather than requiring a cloud vector database or embedding service, the app uses a lightweight BM25 lexical retriever. This is reliable and easy to run for one or a few reports.
- **Mixed document formats:** the uploader supports plain text, PDF, and DOCX. Some scanned PDFs have no extractable text; convert those to searchable/OCR text before uploading.

## Project files

- `app.py` — Streamlit interface, document extraction, retrieval, and Ollama chat call
- `requirements.txt` — Python dependencies
