"""
ShopSense — Gradio Frontend
Phase 9: Full UI with search, filters, session memory, review summaries, feedback
"""

import gradio as gr
import os
import json
import re
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from collections import Counter

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR   = Path('data')
GROQ_MODEL = 'llama-3.1-8b-instant'

# ── Load components ────────────────────────────────────────────
print('Loading ShopSense...')
embed_model   = SentenceTransformer('all-MiniLM-L6-v2')
product_index = faiss.read_index(str(DATA_DIR / 'faiss_index.bin'))
groq_client   = Groq(api_key=os.getenv('GROQ_API_KEY'))

with open(DATA_DIR / 'faiss_metadata.json') as f:
    product_metadata = json.load(f)

with open(DATA_DIR / 'reviews_metadata.json') as f:
    review_chunks = json.load(f)

review_index = faiss.read_index(str(DATA_DIR / 'reviews_faiss_index.bin'))
df_products  = pd.read_csv(DATA_DIR / 'products_clean.csv')
median_price = float(df_products['price'].median())
print('✅ Ready')


# ── Session Memory ─────────────────────────────────────────────
STOPWORDS = {'the','a','an','for','and','or','in','on','at','to',
             'with','of','is','it','this','that','my','me','i',
             'set','inch','pack','piece'}

def extract_keywords(text):
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]

@dataclass
class SessionMemory:
    liked_asins     : list    = field(default_factory=list)
    dismissed_asins : list    = field(default_factory=list)
    search_history  : list    = field(default_factory=list)
    seen_asins      : set     = field(default_factory=set)
    keyword_counts  : Counter = field(default_factory=Counter)
    liked_keywords  : list    = field(default_factory=list)
    avg_price_liked : float   = field(default=None)
    turn_count      : int     = field(default=0)

    def context_string(self):
        if self.turn_count == 0:
            return 'No history yet.'
        parts = []
        if self.search_history:
            parts.append(f"Searches: {', '.join(repr(q) for q in self.search_history[-3:])}")
        if self.liked_asins:
            parts.append(f"Liked: {len(self.liked_asins)}")
        if self.liked_keywords:
            parts.append(f"Liked keywords: {', '.join(self.liked_keywords[:4])}")
        if self.avg_price_liked:
            parts.append(f"Avg liked price: ${self.avg_price_liked:.2f}")
        top_kw = [w for w, _ in self.keyword_counts.most_common(4) if len(w) > 3]
        if top_kw:
            parts.append(f"Top keywords: {', '.join(top_kw)}")
        return ' | '.join(parts)

    def reset(self):
        self.__init__()

session = SessionMemory()


# ── Core functions ─────────────────────────────────────────────
def clean_price(p):
    try:
        v = float(p)
        return v if not np.isnan(v) else median_price
    except:
        return median_price

def personalize(results):
    if session.turn_count == 0:
        return results
    liked_kw = set(session.liked_keywords)
    for r in results:
        score = r.get('similarity_score', 0.0)
        if liked_kw & set(extract_keywords(r.get('title', ''))):
            score += 0.15
        if session.avg_price_liked:
            if abs(clean_price(r.get('price')) - session.avg_price_liked) < session.avg_price_liked * 0.3:
                score += 0.10
        if r['asin'] in session.dismissed_asins:
            score -= 0.20
        if r['asin'] in session.seen_asins:
            score -= 0.10
        r['p_score'] = round(score, 4)
    return sorted(results, key=lambda x: x['p_score'], reverse=True)

def search_products(query, max_price, min_rating):
    q_emb = embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype('float32')
    scores, indices = product_index.search(q_emb, 20)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        p = product_metadata[idx].copy()
        p['similarity_score'] = round(float(score), 4)
        p['price'] = clean_price(p.get('price'))
        results.append(p)
    if max_price:
        filtered = [r for r in results if r['price'] <= max_price]
        results   = filtered if len(filtered) >= 3 else results
    if min_rating:
        filtered = [r for r in results if r.get('rating', 0) >= min_rating]
        results   = filtered if len(filtered) >= 3 else results
    return personalize(results)[:5]

def get_review_summary(asin, query):
    chunks = [c for c in review_chunks if c['asin'] == asin]
    if not chunks:
        return None
    q_emb = embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype('float32')
    scores, indices = review_index.search(q_emb, min(100, len(review_chunks)))
    prod_idx = {i for i, c in enumerate(review_chunks) if c['asin'] == asin}
    relevant = []
    for score, idx in zip(scores[0], indices[0]):
        if idx in prod_idx:
            relevant.append(review_chunks[idx])
        if len(relevant) >= 4:
            break
    if not relevant:
        return None
    context = '\n'.join([f"- [{c['rating']}★] {c['chunk_text']}" for c in relevant])
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content':
                f'Query: "{query}"\nReviews:\n{context}\n\n'
                f'JSON only: {{"pros": ["...", "..."], "cons": ["..."], "verdict": "..."}}'
            }],
            max_tokens=200, temperature=0.3
        )
        raw = re.sub(r'```json|```', '', resp.choices[0].message.content).strip()
        return json.loads(raw)
    except:
        return None

def get_explanation(query, results):
    summary = '\n'.join([
        f"{i+1}. {r['title'][:55]} (${r['price']:.2f} | {r.get('rating')}⭐)"
        for i, r in enumerate(results)
    ])
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content':
                f'User searched: "{query}"\n'
                f'Session: {session.context_string()}\n'
                f'Results:\n{summary}\n\n'
                f'Write 2-3 friendly sentences explaining why these match.'
            }],
            max_tokens=100, temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except:
        return ''


# ── Format results as HTML ─────────────────────────────────────
def format_results_html(results, query):
    if not results:
        return '<p style="color:#888;">No results found.</p>'

    html = ''
    for i, r in enumerate(results, 1):
        title   = r.get('title', 'Unknown')[:80]
        price   = f"${r['price']:.2f}"
        rating  = r.get('rating', 'N/A')
        r_count = int(r.get('rating_count', 0))
        asin    = r.get('asin', '')
        score   = r.get('p_score', r.get('similarity_score', 0))

        # Review summary
        review_html = ''
        rev = get_review_summary(asin, query)
        if rev and (rev.get('pros') or rev.get('verdict')):
            pros_html = ''.join([f'<li>✅ {p}</li>' for p in rev.get('pros', [])[:3]])
            cons_html = ''.join([f'<li>❌ {c}</li>' for c in rev.get('cons', [])[:2]])
            verdict   = rev.get('verdict', '')
            review_html = f"""
            <div style="margin-top:8px; padding:8px;
                        background:#f8f9fa; border-radius:6px; font-size:0.85em;">
                <strong>📝 Review Summary:</strong>
                <ul style="margin:4px 0; padding-left:16px;">
                    {pros_html}{cons_html}
                </ul>
                <em style="color:#555;">💬 {verdict}</em>
            </div>"""

        # Feedback buttons (stored as data attributes for JS)
        html += f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px;
                    margin-bottom:12px; background:white;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="flex:1;">
                    <span style="font-size:0.8em; color:#888; font-weight:600;">#{i}</span>
                    <strong style="font-size:0.95em; color:#1a1a1a;"> {title}</strong>
                </div>
                <div style="text-align:right; margin-left:12px; white-space:nowrap;">
                    <span style="font-size:1.1em; font-weight:700;
                                 color:#2d7d46;">💰 {price}</span>
                </div>
            </div>
            <div style="margin-top:6px; font-size:0.85em; color:#555;">
                ⭐ {rating} &nbsp;·&nbsp; {r_count:,} reviews
                &nbsp;·&nbsp; Score: {score:.3f}
                &nbsp;·&nbsp; <code style="font-size:0.8em;">{asin}</code>
            </div>
            {review_html}
        </div>"""

    return html


# ── Main search function ───────────────────────────────────────
def run_search(query, max_price, min_rating, show_reviews):
    if not query.strip():
        return '', '<p style="color:#888;">Enter a search query above.</p>', session_summary()

    # Update session
    session.search_history.append(query)
    session.turn_count += 1
    session.keyword_counts.update(extract_keywords(query))

    # Search
    results = search_products(
        query,
        float(max_price) if max_price else None,
        float(min_rating) if min_rating else None
    )

    for r in results:
        session.seen_asins.add(r['asin'])

    # Explanation
    explanation = get_explanation(query, results)

    # Format results
    if show_reviews:
        results_html = format_results_html(results, query)
    else:
        results_html = format_results_html_no_reviews(results)

    return explanation, results_html, session_summary()


def format_results_html_no_reviews(results):
    if not results:
        return '<p style="color:#888;">No results found.</p>'
    html = ''
    for i, r in enumerate(results, 1):
        title   = r.get('title', '')[:80]
        price   = f"${r['price']:.2f}"
        rating  = r.get('rating', 'N/A')
        r_count = int(r.get('rating_count', 0))
        score   = r.get('p_score', r.get('similarity_score', 0))
        html += f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px;
                    margin-bottom:12px; background:white;">
            <strong>#{i} {title}</strong><br>
            <span style="color:#2d7d46; font-weight:700;">{price}</span>
            &nbsp;·&nbsp; ⭐ {rating} ({r_count:,} reviews)
            &nbsp;·&nbsp; Score: {score:.3f}
        </div>"""
    return html


def session_summary():
    ctx = session.context_string()
    top_kw = [w for w, _ in session.keyword_counts.most_common(5) if len(w) > 3]
    return f"""**Turn:** {session.turn_count}  |  **Searches:** {len(session.search_history)}  |  **Liked:** {len(session.liked_asins)}  |  **Dismissed:** {len(session.dismissed_asins)}

**Top keywords:** {', '.join(top_kw) if top_kw else 'none yet'}

**Context:** {ctx}"""


def record_feedback(asin_and_title, feedback_type):
    if not asin_and_title:
        return '⚠️ Enter an ASIN to give feedback.', session_summary()
    parts = asin_and_title.strip().split('|')
    asin  = parts[0].strip()
    title = parts[1].strip() if len(parts) > 1 else asin

    product = next((p for p in product_metadata if p['asin'] == asin), None)
    if not product:
        return f'⚠️ ASIN {asin} not found.', session_summary()

    if feedback_type == 'like':
        if asin not in session.liked_asins:
            session.liked_asins.append(asin)
        kws = extract_keywords(title)
        session.liked_keywords.extend(kws[:3])
        liked_prices = [
            clean_price(p['price']) for p in product_metadata
            if p['asin'] in session.liked_asins and p.get('price')
        ]
        if liked_prices:
            session.avg_price_liked = sum(liked_prices) / len(liked_prices)
        return f'👍 Liked: {title[:50]}', session_summary()
    else:
        if asin not in session.dismissed_asins:
            session.dismissed_asins.append(asin)
        return f'👎 Dismissed: {title[:50]}', session_summary()


def reset_session_fn():
    session.reset()
    return '', '<p style="color:#888;">Session reset. Start a new search!</p>', session_summary()


# ── Build Gradio UI ────────────────────────────────────────────
with gr.Blocks(
    title='ShopSense',
    theme=gr.themes.Soft(primary_hue='green'),
    css="""
    .gradio-container { max-width: 900px !important; margin: auto; }
    #title { text-align: center; margin-bottom: 4px; }
    #subtitle { text-align: center; color: #666; margin-bottom: 20px; }
    """
) as demo:

    gr.Markdown('# 🛍️ ShopSense', elem_id='title')
    gr.Markdown(
        'AI-powered product recommendations with semantic search, '
        'session personalization, and review summaries.',
        elem_id='subtitle'
    )

    with gr.Tabs():

        # ── Tab 1: Search ──────────────────────────────────────
        with gr.Tab('🔍 Search'):
            with gr.Row():
                query_box = gr.Textbox(
                    placeholder='e.g. compact coffee maker for small kitchen...',
                    label='What are you looking for?',
                    scale=4
                )
                search_btn = gr.Button('Search', variant='primary', scale=1)

            with gr.Accordion('⚙️ Filters', open=False):
                with gr.Row():
                    max_price  = gr.Number(label='Max Price ($)', value=None, precision=0)
                    min_rating = gr.Number(label='Min Rating (1-5)', value=None, precision=1)
                    show_reviews = gr.Checkbox(label='Show Review Summaries', value=True)

            explanation_box = gr.Markdown(label='Agent Explanation')

            results_box = gr.HTML(
                value='<p style="color:#888; text-align:center;">'
                      'Enter a query above to get started.</p>'
            )

            reset_btn = gr.Button('🔄 Reset Session', variant='secondary', size='sm')

        # ── Tab 2: Feedback ────────────────────────────────────
        with gr.Tab('👍 Feedback'):
            gr.Markdown("""
            ### Give feedback on results
            Copy the **ASIN** from a result card and paste it below.
            Format: `ASIN | Product Title` (title is optional but helpful)

            Example: `B0B12JWHF2 | Anyfish 8 Inch Frying Pan`
            """)

            asin_input    = gr.Textbox(label='ASIN | Title', placeholder='B0B12JWHF2 | Product name')
            with gr.Row():
                like_btn    = gr.Button('👍 Like',    variant='primary')
                dismiss_btn = gr.Button('👎 Dismiss', variant='secondary')
            feedback_status = gr.Markdown()

        # ── Tab 3: Session Memory ──────────────────────────────
        with gr.Tab('🧠 Session Memory'):
            gr.Markdown("""
            ### What ShopSense has learned about you this session
            This updates automatically as you search and give feedback.
            """)
            session_display = gr.Markdown(value=session_summary())
            refresh_btn = gr.Button('🔄 Refresh', size='sm')

    # ── Wire up events ─────────────────────────────────────────
    search_btn.click(
        fn=run_search,
        inputs=[query_box, max_price, min_rating, show_reviews],
        outputs=[explanation_box, results_box, session_display]
    )

    query_box.submit(
        fn=run_search,
        inputs=[query_box, max_price, min_rating, show_reviews],
        outputs=[explanation_box, results_box, session_display]
    )

    reset_btn.click(
        fn=reset_session_fn,
        inputs=[],
        outputs=[explanation_box, results_box, session_display]
    )

    like_btn.click(
        fn=lambda x: record_feedback(x, 'like'),
        inputs=[asin_input],
        outputs=[feedback_status, session_display]
    )

    dismiss_btn.click(
        fn=lambda x: record_feedback(x, 'dismiss'),
        inputs=[asin_input],
        outputs=[feedback_status, session_display]
    )

    refresh_btn.click(
        fn=session_summary,
        inputs=[],
        outputs=[session_display]
    )


if __name__ == '__main__':
    demo.launch()
