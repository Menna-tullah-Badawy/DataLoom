# 🔬 DataLoom — AI Research Intelligence Platform

> An agentic **SQL + RAG** assistant over the arXiv corpus. A **LangGraph** router classifies every question, dispatches it to a **DuckDB analytics engine**, a **ChromaDB semantic index with CrossEncoder re-ranking**, or both, and synthesises a grounded answer with an LLM — served through **FastAPI** and a **Gradio** chat UI.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agentic%20routing-1C3C3C)
![DuckDB](https://img.shields.io/badge/DuckDB-analytics-FFF000?logo=duckdb&logoColor=black)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6F00)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-demo%20UI-F97316)

---

## ✨ What it does

DataLoom answers three kinds of research questions from a single chat interface:

| Route | Example question | What happens under the hood |
|---|---|---|
| **SQL** | *"What are the top 5 research categories?"* | DuckDB runs analytical queries over the processed Parquet layer (overview, category counts, temporal trends). |
| **RAG** | *"Find papers discussing neural networks."* | ChromaDB retrieves candidate papers → a CrossEncoder re-ranks them → top‑k are passed to the LLM. |
| **HYBRID** | *"What are the top categories, and find papers about language models."* | SQL analytics **and** semantic retrieval are executed, then merged into one grounded answer. |

The route is chosen automatically by a structured intent classifier — the user never has to specify a mode.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │    arXiv Dataset    │   Cornell-University/arxiv (Kaggle)
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │   DuckDB + Polars   │   read_json_auto → filter → clean
                    │   Ingestion Layer   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  silver_arxiv.parquet│  Silver data layer
                    └───────┬─────┬───────┘
                            │     │
                 ┌──────────▼─┐ ┌─▼──────────────┐
                 │  DuckDB    │ │   ChromaDB      │
                 │  Analytics │ │   Vector Store  │
                 └──────┬─────┘ └─┬──────────────┘
                        │         │  candidates (2·k, min 10)
                        │   ┌─────▼──────────────┐
                        │   │ CrossEncoder       │  ms-marco-MiniLM-L-6-v2
                        │   │ Re-ranking → top-k │
                        │   └─────┬──────────────┘
                        └────┬────┘
                             ▼
                   ┌───────────────────┐
                   │  LangGraph Agent  │  router → sql / rag / hybrid → synthesizer
                   └─────────┬─────────┘
                             ▼
              ┌──────────────┴──────────────┐
              │  FastAPI  POST /chat        │  Gradio ChatInterface
              └─────────────────────────────┘
```

### Agent graph (LangGraph)

```text
router ──sql────▶ sql_executor ──────────────▶ synthesizer ──▶ END
   │                    │ (hybrid)                 ▲
   ├──hybrid──▶ sql_executor ──▶ rag_executor ─────┤
   └──rag───────────────────────▶ rag_executor ────┘
```

* **`router`** – classifies the query into `sql | rag | hybrid` using **Instructor** + a Pydantic `IntentClassification` model, so the routing decision is *schema‑validated*, not free text.
* **`sql_executor`** – dataset overview, top categories and temporal trends via DuckDB.
* **`rag_executor`** – ChromaDB candidate retrieval followed by CrossEncoder re‑ranking (top‑5).
* **`synthesizer`** – builds a grounded prompt from the analytics + retrieved papers and calls the LLM.
* Conversation state is checkpointed per `session_id` with LangGraph's `MemorySaver`.

---

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| Data processing | **Polars**, **DuckDB**, Parquet |
| Vector search | **ChromaDB** (persistent client), **sentence-transformers** CrossEncoder |
| Agent orchestration | **LangGraph**, LangChain core |
| LLM access | OpenAI‑compatible client via **OpenRouter** (`meta-llama/llama-3.3-70b-instruct`) |
| Structured outputs | **Instructor** + **Pydantic** |
| Observability | **LangSmith** tracing (`DataLoom-Production` project) |
| Serving | **FastAPI** + Uvicorn (REST), **Gradio** (chat demo) |
| Data source | **kagglehub** → `Cornell-University/arxiv` |

---

## 🛡️ Grounding & responsible‑AI controls

DataLoom is designed to be *honest about what it knows*:

* **Grounded generation only** – the synthesis prompt instructs the model to answer **only** from the supplied analytics and retrieved papers, to never invent statistics or papers, and to state explicitly when the data is insufficient.
* **Scope transparency** – every answer is scoped to the indexed subset (5,000 records in the demo) and the model is told not to generalise to the full arXiv corpus.
* **Clear provenance** – dataset statistics and semantically retrieved papers are kept distinct in the answer, and paper titles are cited when used.
* **Schema‑enforced routing** – intent classification is validated against a Pydantic model, eliminating malformed or ambiguous routing decisions.
* **Traceability** – every run is traced in LangSmith, giving a full audit trail of prompts, tool calls and latencies.

---

## 📁 Project structure

The notebook bootstraps a proper package layout and writes each module to disk:

```text
DataLoom/
├── dataloom.ipynb          # End-to-end notebook (setup → ingestion → index → agent → API → demo)
├── src/
│   ├── config.py           # Paths, LLM provider settings, LangSmith tracing, config validation
│   ├── ingestion.py        # ArxivIngestion: DuckDB read_json_auto + Polars cleaning → silver Parquet
│   ├── analytics.py        # AnalyticsEngine: DuckDB views & analytical queries
│   ├── vector_store.py     # VectorStoreManager: ChromaDB indexing + lazy CrossEncoder re-ranking
│   ├── agent.py            # LangGraph workflow: router → sql / rag / hybrid → synthesizer
│   └── main.py             # FastAPI application (GET /, POST /chat)
├── data/
│   ├── raw/                # Downloaded source data
│   └── processed/          # silver_arxiv.parquet
└── chroma_db/              # Persistent ChromaDB collection (`arxiv_papers`)
```

---

## 🚀 Getting started

### 1. Install

```bash
pip install polars duckdb openai instructor pydantic chromadb langgraph langchain-core \
            langchain-openai pandas kagglehub pandera sentence-transformers fastapi uvicorn gradio
```

### 2. Configure credentials

Provide the keys through environment variables (or Kaggle / Colab secrets) — never commit them:

```bash
export OPENROUTER_API_KEY="..."      # LLM access (OpenAI-compatible endpoint)
export LANGCHAIN_API_KEY="..."       # optional – LangSmith tracing
```

### 3. Run the notebook

Open `dataloom.ipynb` (Kaggle or local Jupyter) and run the cells in order:

1. **Setup** – installs dependencies and creates the project directories.
2. **Configuration** – writes `src/config.py` and validates it.
3. **Ingestion** – downloads the arXiv metadata via `kagglehub`, processes the first 5,000 valid records with DuckDB/Polars and writes `data/processed/silver_arxiv.parquet`.
4. **Indexing** – upserts title + abstract documents into ChromaDB in batches of 500.
5. **Agent** – builds and compiles the LangGraph workflow.
6. **System validation** – runs one SQL, one RAG and one HYBRID query end‑to‑end.
7. **Demo** – launches the Gradio chat interface.

### 4. Serve the REST API

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Find papers about transformer architectures.", "session_id": "demo"}'
```

```json
{
  "query": "Find papers about transformer architectures.",
  "response": "**🔀 DataLoom Route:** RAG\n\n..."
}
```

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service status |
| `/chat` | POST | `{ "query": str, "session_id": str }` → routed, grounded answer |

### 5. Launch the chat demo

The last notebook cell starts a Gradio `ChatInterface` with ready‑made example prompts:

* *What are the top research categories?*
* *Find papers discussing neural networks.*
* *Find papers about transformer architectures.*
* *What are the top categories, and find papers about language models.*
* *Show me research trends and papers related to deep learning.*

---

## ⚙️ Design decisions

* **DuckDB + Polars instead of pandas** – the raw arXiv snapshot is a multi‑GB JSON file; DuckDB's `read_json_auto` streams it and Polars handles cleaning without loading everything into memory.
* **Two‑stage retrieval** – cheap vector similarity for recall, CrossEncoder for precision. The re‑ranker is **lazy‑loaded** so indexing stays fast and memory‑light.
* **Instructor only where structure matters** – used for the routing decision; the synthesis step uses a plain chat completion to keep answers natural.
* **Medallion‑style data layer** – raw → silver Parquet, giving both the SQL and the vector paths a single, cleaned source of truth.

---

## 🗺️ Limitations & roadmap

* Demo indexes a bounded subset (5,000 papers); scaling to the full corpus needs incremental ingestion and a batched embedding job.
* ChromaDB uses its default embedding function — swapping in a domain‑specific embedder (e.g. SPECTER / E5) should improve recall.
* Conversation memory is in‑process (`MemorySaver`); a persistent checkpointer is required for multi‑replica deployments.
* Planned: input guardrails (length / prompt‑injection checks), evaluation set with LangSmith datasets, Dockerfile + CI pipeline.

---

## 👤 Author

**Menna‑tullah Badawy** — [GitHub](https://github.com/Menna-tullah-Badawy)
