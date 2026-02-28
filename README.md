# 🛍️ ShopSense

> AI-powered product recommendation system with semantic search, session personalization, RAG review summaries, and a LangGraph agent — deployed on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-ShopSense-green)](https://huggingface.co/spaces/RithikaBaskaran/shopsense)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-orange)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 🔗 Live Demo

**[https://huggingface.co/spaces/RithikaBaskaran/shopsense](https://huggingface.co/spaces/RithikaBaskaran/shopsense)**

Type a natural language query like *"compact coffee maker for small kitchen"* or *"non stick frying pan under $30"* and ShopSense will retrieve, rerank, summarize, and explain the best matching products — getting smarter with every search.

---

## 📸 Overview

ShopSense is a full end-to-end ML portfolio project built across 9 phases, covering everything from raw data preparation to a publicly deployed web application.

| Feature | Description |
|---|---|
| 🔍 Semantic Search | FAISS vector search with `all-MiniLM-L6-v2` embeddings |
| 📊 Reranking | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) for precision ordering |
| 📝 Review Summaries | RAG pipeline — retrieve relevant review chunks → Llama 3 summarization |
| 🤖 LangGraph Agent | Intent extraction, filter parsing, clarification, explanation generation |
| 🧠 Session Memory | Tracks likes, dismissals, price preferences across the conversation |
| 📈 Evaluation | NDCG@5 and MRR metrics — reranking improves NDCG by **+9.7%** |
| 🔧 Fine-tuning | LoRA fine-tuning of `flan-t5-base` on synthetic training data |
| 🚀 Deployment | FastAPI backend + Gradio frontend on Hugging Face Spaces |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         LangGraph Agent             │
│  • Analyze intent                   │
│  • Extract filters (price, rating)  │
│  • Detect vague queries → clarify   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│         FAISS Search                │
│  all-MiniLM-L6-v2 embeddings        │
│  10,000 products · top 20 results   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│      Session Personalization        │
│  Re-score based on:                 │
│  +0.15 keyword overlap with likes   │
│  +0.10 price matches liked range    │
│  -0.10 already seen this session    │
│  -0.20 explicitly dismissed         │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│       RAG Review Summaries          │
│  Review FAISS index → top chunks    │
│  → Llama 3 → pros/cons/verdict      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│      Explanation Generation         │
│  Groq Llama 3 → friendly summary    │
│  referencing session history        │
└────────────────┬────────────────────┘
                 │
                 ▼
            Results UI
```

---

## 📊 Evaluation Results

Evaluated on 8 queries using automated relevance judgements (0–3 scale).

| Pipeline | MRR | NDCG@5 |
|---|---|---|
| FAISS Only | 1.000 | 0.910 |
| FAISS + Cross-Encoder Reranking | 1.000 | 0.998 |
| **Improvement** | **+0.0%** | **+9.7%** |

**MRR = 1.0 for both** — FAISS already surfaces a relevant result at position 1 every time.
**NDCG improved +9.7%** — reranking fixes the ordering within the top 5, pushing higher relevance results to the top.

---

## 📁 Project Structure

```
ShopSense/
├── data/
│   ├── products_clean.csv          # 10,000 Amazon Home & Kitchen products
│   ├── reviews_clean.csv           # 5,566 review chunks (2,003 products)
│   ├── faiss_index.bin             # Product embeddings (384-dim)
│   ├── faiss_metadata.json         # Product metadata
│   ├── reviews_faiss_index.bin     # Review chunk embeddings
│   ├── reviews_metadata.json       # Review chunk metadata
│   ├── evaluation_results.csv      # Phase 7 evaluation metrics
│   └── evaluation_plot.png         # MRR / NDCG comparison chart
│
├── app.py                          # FastAPI backend (5 endpoints)
├── gradio_app.py                   # Gradio frontend (3-tab UI)
├── agent.py                        # LangGraph agent module
├── session.py                      # Session memory module
├── requirements.txt                # Dependencies
│
├── data_preparation.ipynb          # Phase 1 — data cleaning
├── semantic_retrieval.ipynb        # Phase 2 — FAISS search
├── reranking.ipynb                 # Phase 3 — cross-encoder (Colab)
├── rag_review_summarization.ipynb  # Phase 4 — RAG pipeline
├── langgraph_agent.ipynb           # Phase 5 — LangGraph agent
├── session_personalization.ipynb   # Phase 6 — session memory
├── evaluation.ipynb                # Phase 7 — NDCG + MRR
├── finetuning.ipynb                # Phase 8 — LoRA fine-tuning (Colab)
└── deployment.ipynb                # Phase 9 — HF Spaces deployment
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search | FAISS (cosine similarity) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq API — `llama-3.1-8b-instant` |
| Agent Framework | LangGraph |
| Fine-tuning | LoRA via PEFT — `google/flan-t5-base` |
| Backend | FastAPI + Uvicorn |
| Frontend | Gradio 5 |
| Hosting | Hugging Face Spaces |
| Dataset | Amazon Reviews 2023 — Home & Kitchen |

---

## 🗂️ Dataset

- **Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — Home & Kitchen category
- **Products:** 10,000
- **Reviews:** 5,566 chunks across 2,003 products (20% coverage)
- **Fields used:** title, description, price, rating, rating count, ASIN

> **Note:** 80% of products have no review text in the dataset. This is a data collection limitation — the RAG pipeline works correctly for all products that have reviews, and gracefully returns "No reviews available" for others.

---

## 🚀 Running Locally

### Prerequisites
- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone the repo
git clone https://github.com/RithikaBaskaran/ShopSense.git
cd ShopSense

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env
```

### Run the Gradio UI

```bash
python gradio_app.py
# Open http://localhost:7860
```

### Run the FastAPI backend

```bash
uvicorn app:app --reload
# API docs at http://localhost:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/search` | Main search endpoint |
| `POST` | `/feedback` | Record like / dismiss |
| `GET` | `/session` | Get session summary |
| `POST` | `/session/reset` | Clear session memory |

---

## 📓 Phase Breakdown

| Phase | Notebook | Environment | Description |
|---|---|---|---|
| 1 | `data_preparation.ipynb` | VS Code | Load, clean and sample Amazon dataset |
| 2 | `semantic_retrieval.ipynb` | VS Code | FAISS index + semantic search |
| 3 | `reranking.ipynb` | Colab (GPU) | Cross-encoder reranking |
| 4 | `rag_review_summarization.ipynb` | VS Code | Review RAG pipeline |
| 5 | `langgraph_agent.ipynb` | VS Code | LangGraph agent with intent extraction |
| 6 | `session_personalization.ipynb` | VS Code | Session memory + personalization scoring |
| 7 | `evaluation.ipynb` | VS Code | NDCG + MRR evaluation |
| 8 | `finetuning.ipynb` | Colab (GPU) | LoRA fine-tuning of flan-t5-base |
| 9 | `deployment.ipynb` | VS Code | FastAPI + Gradio + HF Spaces |

---

## 🔍 Example Queries

Try these on the live demo:

```
non stick frying pan under $30
compact coffee maker for small kitchen
highly rated storage bins for closet
gift for someone who loves cooking under $25
best knife set under $50 with 4 stars and above
durable water bottle for gym
cute kitchen decor for modern home
```

---

## 🧠 Session Personalization

ShopSense gets smarter the longer you use it within a session.

**Explicit signals** (you tell it directly):
- 👍 Like a product → future results biased toward similar items
- 👎 Dismiss a product → future results penalize that item

**Implicit signals** (extracted automatically):
- Keywords from your searches → influence query refinement
- Price range of liked items → biases future price scoring
- Products already shown → slight novelty penalty to surface new items

The **Session Memory tab** shows exactly what ShopSense has learned about you at any point.

---

## ⚙️ Fine-tuning Notes

Phase 8 fine-tuned `google/flan-t5-base` using LoRA on 118 synthetically generated product explanation examples.

- **Method:** LoRA (`r=8`, `lora_alpha=16`, 0.356% trainable params)
- **Data:** 118 examples generated via Groq Llama 3
- **Training:** 20 epochs, loss reduced from 17.4 → 7.9
- **Key fix:** Tied weights initialization bug in flan-t5 checkpoint loader causing `nan` loss — resolved by explicitly setting `tie_word_embeddings=True` and manually assigning `lm_head.weight = shared.weight`
- **Limitation:** 106 training examples is below the ~1,000 minimum for meaningful text quality improvement — production deployment uses Groq Llama 3 for explanation generation

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset
- [Groq](https://groq.com) for fast LLM inference
- [Hugging Face](https://huggingface.co) for model hosting and Spaces
- [LangGraph](https://langchain-ai.github.io/langgraph/) for agent framework
