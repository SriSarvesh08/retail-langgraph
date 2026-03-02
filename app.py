import streamlit as st
import os, json
import PyPDF2
import faiss
import numpy as np
from typing import TypedDict, Literal
from groq import Groq

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

@st.cache_resource(show_spinner="⚡ Loading AI model (first time only)...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

@st.cache_resource
def get_langchain_groq():
    return ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=800
    )

EMBED_MODEL = load_embed_model()
groq_client = get_groq_client()
llm         = get_langchain_groq()

st.set_page_config(page_title="Retail AI — LangGraph", page_icon="🕸️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0a0a0f; color: #f0ece4; }
.main { background-color: #0a0a0f; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

.store-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a1f2e 100%);
    border: 1px solid #58a6ff; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem; text-align: center;
}
.store-header h1 { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0; }
.store-header p  { color: #8b949e; margin: 0.4rem 0 0 0; font-size: 0.95rem; }
.accent { color: #58a6ff; }

.graph-flow {
    background: #0d1117; border: 1px solid #30363d;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem;
    font-size: 0.82rem; color: #8b949e;
}
.node-active {
    display: inline-block; background: #388bfd26;
    border: 1px solid #58a6ff; color: #58a6ff;
    border-radius: 20px; padding: 2px 12px;
    font-size: 0.75rem; font-weight: 700; margin-bottom: 0.4rem;
}
.node-done {
    display: inline-block; background: #23863626;
    border: 1px solid #3fb950; color: #3fb950;
    border-radius: 20px; padding: 2px 12px;
    font-size: 0.75rem; font-weight: 600; margin-bottom: 0.4rem;
}
.chat-user      { background: #161b22; border-left: 3px solid #58a6ff; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.chat-assistant { background: #0d1117; border-left: 3px solid #3fb950; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; }
.status-ok   { color: #3fb950; font-weight: 600; }
.status-warn { color: #d29922; font-weight: 600; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

for key, default in {
    "rag_index":    None,
    "rag_chunks":   [],
    "pdf_loaded":   False,
    "pdf_name":     "",
    "chat_history": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def extract_and_chunk_pdf(pdf_file, chunk_size=600, overlap=60):
    reader, all_chunks, buffer = PyPDF2.PdfReader(pdf_file), [], []
    for page in reader.pages:
        buffer.extend((page.extract_text() or "").split())
        while len(buffer) >= chunk_size:
            all_chunks.append(" ".join(buffer[:chunk_size]))
            buffer = buffer[chunk_size - overlap:]
    if buffer:
        all_chunks.append(" ".join(buffer))
    return all_chunks

def build_faiss_index(chunks, progress_bar):
    batch_size, all_embeddings = 32, []
    for i in range(0, len(chunks), batch_size):
        emb = EMBED_MODEL.encode(chunks[i:i+batch_size], convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(emb)
        progress_bar.progress(min((i+batch_size)/len(chunks), 1.0),
                              text=f"⚡ Indexing {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    embeddings = np.vstack(all_embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query: str, top_k: int = 5) -> str:
    if not st.session_state.pdf_loaded:
        return "NO_PDF"
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = st.session_state.rag_index.search(q_emb, top_k)
    chunks = [st.session_state.rag_chunks[i] for i in indices[0] if i < len(st.session_state.rag_chunks)]
    return "\n\n".join(chunks) if chunks else "NOT_FOUND"

class RetailState(TypedDict):
    user_query:    str
    intent:        str
    context:       str
    final_answer:  str
    agent_used:    str
    error:         str

def supervisor_node(state: RetailState) -> RetailState:
    query = state["user_query"]
    prompt = f"""You are a retail assistant router. Classify the user query into exactly ONE category.

Categories:
- product_info     : specs, features, warranty, materials, description
- stock_level      : availability, inventory, quantity, in stock, out of stock
- sales_summary    : revenue, sales, best sellers, performance, report
- product_search   : find, search, show me, list, filter, price range, category
- discount_info    : discount, offer, coupon, promo, deal, price reduction

User query: "{query}"

Reply with ONLY the category name, nothing else."""

    response = llm.invoke([HumanMessage(content=prompt)])
    intent   = response.content.strip().lower()
    valid = ["product_info", "stock_level", "sales_summary", "product_search", "discount_info"]
    if intent not in valid:
        intent = "product_info"
    return {**state, "intent": intent}

def product_info_agent(state: RetailState) -> RetailState:
    query   = state["user_query"]
    context = retrieve(f"{query} specifications features warranty materials details description")
    if context == "NO_PDF":
        return {**state, "final_answer": "Please upload a product PDF first.", "agent_used": "📋 Product Info Agent", "error": "no_pdf"}
    if context == "NOT_FOUND":
        return {**state, "final_answer": "This information is not available in the uploaded dataset.", "agent_used": "📋 Product Info Agent", "error": "not_found"}
    prompt = f"""You are a Product Information Agent for a retail store.
Using ONLY the context below from the product dataset, answer the user question.
Do not fabricate any information. Be clear and structured.

CONTEXT FROM PDF:
{context}

USER QUESTION: {query}

Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "context": context, "final_answer": response.content.strip(), "agent_used": "📋 Product Info Agent", "error": ""}

def stock_level_agent(state: RetailState) -> RetailState:
    query   = state["user_query"]
    context = retrieve(f"{query} stock inventory quantity available units in stock out of stock")
    if context == "NO_PDF":
        return {**state, "final_answer": "Please upload a product PDF first.", "agent_used": "📦 Stock Level Agent", "error": "no_pdf"}
    if context == "NOT_FOUND":
        return {**state, "final_answer": "Stock information is not available in the uploaded dataset.", "agent_used": "📦 Stock Level Agent", "error": "not_found"}
    prompt = f"""You are a Stock Level Agent for a retail store.
Using ONLY the context below from the inventory dataset, answer the user question about stock availability.
Be direct about availability status.

CONTEXT FROM PDF:
{context}

USER QUESTION: {query}

Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "context": context, "final_answer": response.content.strip(), "agent_used": "📦 Stock Level Agent", "error": ""}

def sales_summary_agent(state: RetailState) -> RetailState:
    query   = state["user_query"]
    context = retrieve(f"{query} sales revenue units sold best selling performance daily weekly monthly report")
    if context == "NO_PDF":
        return {**state, "final_answer": "Please upload a product PDF first.", "agent_used": "📊 Sales Summary Agent", "error": "no_pdf"}
    if context == "NOT_FOUND":
        return {**state, "final_answer": "Sales data is not available in the uploaded dataset.", "agent_used": "📊 Sales Summary Agent", "error": "not_found"}
    prompt = f"""You are a Sales Summary Agent for a retail store.
Using ONLY the context below from the sales dataset, provide a clear sales summary.
Include revenue figures, units, and trends if available.

CONTEXT FROM PDF:
{context}

USER QUESTION: {query}

Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "context": context, "final_answer": response.content.strip(), "agent_used": "📊 Sales Summary Agent", "error": ""}

def product_search_agent(state: RetailState) -> RetailState:
    query   = state["user_query"]
    context = retrieve(f"{query} product list category price range filter search", top_k=6)
    if context == "NO_PDF":
        return {**state, "final_answer": "Please upload a product PDF first.", "agent_used": "🔎 Product Search Agent", "error": "no_pdf"}
    if context == "NOT_FOUND":
        return {**state, "final_answer": "No matching products found in the uploaded dataset.", "agent_used": "🔎 Product Search Agent", "error": "not_found"}
    prompt = f"""You are a Product Search Agent for a retail store.
Using ONLY the context below from the product dataset, find and list products matching the user query.
Present results clearly with product names, prices, and key details if available.

CONTEXT FROM PDF:
{context}

USER QUESTION: {query}

Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "context": context, "final_answer": response.content.strip(), "agent_used": "🔎 Product Search Agent", "error": ""}

def discount_info_agent(state: RetailState) -> RetailState:
    query   = state["user_query"]
    context = retrieve(f"{query} discount offer promo coupon deal eligibility price reduction sale")
    if context == "NO_PDF":
        return {**state, "final_answer": "Please upload a product PDF first.", "agent_used": "🏷️ Discount Info Agent", "error": "no_pdf"}
    if context == "NOT_FOUND":
        return {**state, "final_answer": "No discount information found in the uploaded dataset.", "agent_used": "🏷️ Discount Info Agent", "error": "not_found"}
    prompt = f"""You are a Discount & Offers Agent for a retail store.
Using ONLY the context below from the dataset, answer the user question about discounts, offers, or promotions.
Be specific about eligibility criteria and discount values if available.

CONTEXT FROM PDF:
{context}

USER QUESTION: {query}

Answer:"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "context": context, "final_answer": response.content.strip(), "agent_used": "🏷️ Discount Info Agent", "error": ""}

def route_to_agent(state: RetailState) -> Literal[
    "product_info_agent",
    "stock_level_agent",
    "sales_summary_agent",
    "product_search_agent",
    "discount_info_agent"
]:
    return {
        "product_info":  "product_info_agent",
        "stock_level":   "stock_level_agent",
        "sales_summary": "sales_summary_agent",
        "product_search":"product_search_agent",
        "discount_info": "discount_info_agent",
    }.get(state["intent"], "product_info_agent")

@st.cache_resource
def build_graph():
    graph = StateGraph(RetailState)
    graph.add_node("supervisor",          supervisor_node)
    graph.add_node("product_info_agent",  product_info_agent)
    graph.add_node("stock_level_agent",   stock_level_agent)
    graph.add_node("sales_summary_agent", sales_summary_agent)
    graph.add_node("product_search_agent",product_search_agent)
    graph.add_node("discount_info_agent", discount_info_agent)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_to_agent, {
        "product_info_agent":  "product_info_agent",
        "stock_level_agent":   "stock_level_agent",
        "sales_summary_agent": "sales_summary_agent",
        "product_search_agent":"product_search_agent",
        "discount_info_agent": "discount_info_agent",
    })
    for agent in ["product_info_agent","stock_level_agent","sales_summary_agent","product_search_agent","discount_info_agent"]:
        graph.add_edge(agent, END)
    return graph.compile()

retail_graph = build_graph()

def run_graph(user_query: str):
    initial_state: RetailState = {
        "user_query":   user_query,
        "intent":       "",
        "context":      "",
        "final_answer": "",
        "agent_used":   "",
        "error":        ""
    }
    result = retail_graph.invoke(initial_state)
    return result["final_answer"], result["agent_used"], result["intent"]

# ─── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Upload Product Dataset PDF")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_pdf and uploaded_pdf.name != st.session_state.pdf_name:
        st.info(f"📄 `{uploaded_pdf.name}`")
        with st.spinner("📄 Reading PDF..."):
            chunks = extract_and_chunk_pdf(uploaded_pdf)
        st.success(f"✅ {len(chunks)} chunks extracted")
        progress_bar = st.progress(0, text="⚡ Indexing...")
        index = build_faiss_index(chunks, progress_bar)
        progress_bar.progress(1.0, text="✅ Done!")
        st.session_state.rag_index  = index
        st.session_state.rag_chunks = chunks
        st.session_state.pdf_loaded = True
        st.session_state.pdf_name   = uploaded_pdf.name
        st.success("🚀 PDF ready! Start asking questions.")

    st.markdown("---")
    st.markdown("### 🕸️ LangGraph Architecture")
    st.markdown("""
<div class="graph-flow">
User Query<br>
&nbsp;&nbsp;&nbsp;↓<br>
🧠 <b>Supervisor Node</b> (intent routing)<br>
&nbsp;&nbsp;&nbsp;↓<br>
┌──────────────────┐<br>
│ 📋 Product Info  │<br>
│ 📦 Stock Level   │<br>
│ 📊 Sales Summary │<br>
│ 🔎 Product Search│<br>
│ 🏷️ Discount Info │<br>
└──────────────────┘<br>
&nbsp;&nbsp;&nbsp;↓<br>
✅ Final Answer
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💬 Example Questions")
    for q in [
        "What are the specs of [product]?",
        "Is [product] available in stock?",
        "Show monthly sales summary",
        "Find products under $500",
        "What discounts are available?",
    ]:
        st.markdown(f"• *{q}*")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ─── MAIN ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="store-header">
    <h1>🕸️ Retail <span class="accent">AI</span> — LangGraph</h1>
    <p>Multi-Agent Architecture · PDF-Powered · Groq LLaMA3-8B</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<p class='status-ok'>✅ {st.session_state.pdf_name}</p>" if st.session_state.pdf_loaded else "<p class='status-warn'>⚠️ Upload PDF to begin</p>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<p class='status-ok'>💬 {len(st.session_state.chat_history)} messages</p>", unsafe_allow_html=True)
with col3:
    st.markdown("<p class='status-ok'>🕸️ LangGraph · 6 Nodes</p>", unsafe_allow_html=True)

st.markdown("---")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'>👤 <b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        badge = f"<span class='node-done'>{msg.get('agent_used','')}</span><br>" if msg.get("agent_used") else ""
        st.markdown(f"<div class='chat-assistant'>{badge}🤖 <b>Assistant:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

if not st.session_state.pdf_loaded:
    st.warning("⬆️ Please upload your product PDF from the sidebar to start chatting.")
    st.stop()

with st.form(key="chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_input = st.text_input("", placeholder="Ask anything about your products...", label_visibility="collapsed")
    with col_btn:
        submitted = st.form_submit_button("Send 🚀")

if submitted and user_input.strip():
    with st.spinner("🕸️ LangGraph agents processing..."):
        response, agent_used, intent = run_graph(user_input)
    st.session_state.chat_history.append({"role": "user",      "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response, "agent_used": agent_used, "intent": intent})
    st.rerun()
