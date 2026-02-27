
"""
ShopSense Agent Module
Reusable agent for Phase 9 FastAPI backend.
Usage:
    from agent import run_shopsense_agent
    result = run_shopsense_agent("compact coffee maker")
"""

import os, json, re, faiss, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

load_dotenv()
DATA_DIR   = Path("data")
GROQ_MODEL = "llama-3.1-8b-instant"

# Load all components
embed_model      = SentenceTransformer("all-MiniLM-L6-v2")
product_index    = faiss.read_index(str(DATA_DIR / "faiss_index.bin"))
review_index     = faiss.read_index(str(DATA_DIR / "reviews_faiss_index.bin"))
groq_client      = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open(DATA_DIR / "faiss_metadata.json") as f:
    product_metadata = json.load(f)
with open(DATA_DIR / "reviews_metadata.json") as f:
    review_chunks = json.load(f)

df_products  = pd.read_csv(DATA_DIR / "products_clean.csv")
median_price = df_products["price"].median()


class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    filters: Optional[dict]
    search_results: Optional[list]
    final_results: Optional[list]
    clarify_question: Optional[str]
    explanation: Optional[str]
    needs_clarification: bool


def search_tool(query, top_k=20):
    q = embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = product_index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1: continue
        p = product_metadata[idx].copy()
        p["similarity_score"] = round(float(score), 4)
        if not p.get("price") or str(p.get("price")) == "nan": p["price"] = median_price
        results.append(p)
    return results


def filter_tool(products, max_price=None, min_price=None, min_rating=None, keyword=None):
    f = products
    if max_price: f = [p for p in f if p.get("price", 0) <= max_price]
    if min_price: f = [p for p in f if p.get("price", 0) >= min_price]
    if min_rating: f = [p for p in f if p.get("rating", 0) >= min_rating]
    if keyword:
        kw = keyword.lower()
        f = [p for p in f if kw in p.get("title","").lower() or kw in p.get("description","").lower()]
    return f if len(f) >= 3 else products[:10]


def analyze_node(state):
    query = state["query"]
    prompt = f"""Analyze this shopping query: "{query}"
Respond with JSON: {{"intent": "search" or "clarify", "needs_clarification": bool,
"search_query": "improved query", "filters": {{"max_price": null or number,
"min_price": null or number, "min_rating": null or number, "keyword": null or string}}}}
Only clarify if extremely vague. JSON only."""
    r = groq_client.chat.completions.create(model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}], max_tokens=200, temperature=0.1)
    try:
        p = json.loads(re.sub(r"```json|```", "", r.choices[0].message.content).strip())
        fi = p.get("filters", {})
        return {**state, "intent": p.get("intent", "search"),
                "needs_clarification": p.get("needs_clarification", False),
                "query": p.get("search_query", query),
                "filters": {"max_price": fi.get("max_price"), "min_price": fi.get("min_price"),
                             "min_rating": fi.get("min_rating"), "keyword": fi.get("keyword")}}
    except: return {**state, "intent": "search", "needs_clarification": False, "filters": {}}


def clarify_node(state):
    r = groq_client.chat.completions.create(model=GROQ_MODEL,
        messages=[{"role": "user", "content": f'Query: "{state["query"]}" is vague. Ask ONE clarifying question under 20 words.'}],
        max_tokens=50, temperature=0.5)
    return {**state, "clarify_question": r.choices[0].message.content.strip()}


def search_node(state):
    return {**state, "search_results": search_tool(state["query"])}


def filter_node(state):
    fi = state.get("filters") or {}
    filtered = filter_tool(state.get("search_results") or [],
        max_price=fi.get("max_price"), min_price=fi.get("min_price"),
        min_rating=fi.get("min_rating"), keyword=fi.get("keyword"))
    return {**state, "final_results": filtered[:5]}


def explain_node(state):
    results = state.get("final_results") or []
    if not results: return {**state, "explanation": "No products found."}
    summary = "
".join([f"{i+1}. {r['title'][:60]} (${r.get('price',0):.2f} | {r.get('rating')}⭐)"
                          for i, r in enumerate(results)])
    r = groq_client.chat.completions.create(model=GROQ_MODEL,
        messages=[{"role": "user", "content": f"User searched: "{state['query']}"
Recommended:
{summary}
Write 3-4 friendly sentences explaining why these match."}],
        max_tokens=150, temperature=0.7)
    return {**state, "explanation": r.choices[0].message.content.strip()}


# Build graph
_graph = StateGraph(AgentState)
_graph.add_node("analyze", analyze_node)
_graph.add_node("clarify", clarify_node)
_graph.add_node("search",  search_node)
_graph.add_node("filter",  filter_node)
_graph.add_node("explain", explain_node)
_graph.set_entry_point("analyze")
_graph.add_conditional_edges("analyze",
    lambda s: "clarify" if s.get("needs_clarification") else "search",
    {"clarify": "clarify", "search": "search"})
_graph.add_edge("clarify", END)
_graph.add_edge("search",  "filter")
_graph.add_edge("filter",  "explain")
_graph.add_edge("explain", END)
_agent = _graph.compile()


def run_shopsense_agent(query: str) -> dict:
    """Main entry point for the ShopSense agent. Used by FastAPI in Phase 9."""
    state = _agent.invoke({
        "query": query, "intent": None, "filters": {},
        "search_results": None, "final_results": None,
        "clarify_question": None, "explanation": None,
        "needs_clarification": False
    })
    return {
        "query"              : state["query"],
        "needs_clarification": state.get("needs_clarification", False),
        "clarify_question"   : state.get("clarify_question"),
        "results"            : state.get("final_results") or [],
        "explanation"        : state.get("explanation"),
        "filters_applied"    : state.get("filters") or {}
    }
