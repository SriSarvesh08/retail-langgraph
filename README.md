# 🕸️ Retail AI — LangGraph Multi-Agent App

A Streamlit-based AI assistant for retail, powered by LangGraph multi-agent architecture, Groq LLaMA3, and PDF-based RAG (Retrieval-Augmented Generation).

## 🧠 Architecture

```
User Query
    ↓
🧠 Supervisor Node (intent routing)
    ↓
┌─────────────────────┐
│ 📋 Product Info     │
│ 📦 Stock Level      │
│ 📊 Sales Summary    │
│ 🔎 Product Search   │
│ 🏷️  Discount Info   │
└─────────────────────┘
    ↓
✅ Final Answer
```

## 🚀 Features

- Upload any retail product PDF (catalog, inventory, sales report)
- Automatic intent detection routes to the right agent
- FAISS vector search for fast PDF retrieval
- 5 specialized agents powered by Groq LLaMA-3.1-8B

## 🛠️ Setup & Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/retail-ai-langgraph.git
cd retail-ai-langgraph
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GROQ_API_KEY` in **Secrets** settings
5. Deploy!

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com) |

## 📦 Tech Stack

- **Streamlit** — UI framework
- **LangGraph** — Multi-agent orchestration
- **Groq** — LLM inference (LLaMA 3.1 8B)
- **FAISS** — Vector similarity search
- **Sentence Transformers** — Text embeddings
- **PyPDF2** — PDF parsing
