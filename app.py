from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.document_loader import load_and_split_pdfs
from src.vector_store import create_or_get_vector_store
from src.retriever import retrieve_and_rerank
from src.answer_generator import generate_answer_stream, LLMUnavailableError
from src.settings import get_setting

# Page setup
st.set_page_config(
    page_title="DocuQuery AI | Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "indexed_signature" not in st.session_state:
    st.session_state["indexed_signature"] = None
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "document_names" not in st.session_state:
    st.session_state["document_names"] = []
if "chunk_count" not in st.session_state:
    st.session_state["chunk_count"] = 0
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

groq_ready = bool(get_setting("GROQ_API_KEY"))
openrouter_ready = bool(get_setting("OPENROUTER_API_KEY"))

# Theme Toggle in Sidebar
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    dark_mode_enabled = st.toggle(
        "🌙 Dark Mode",
        value=(st.session_state["theme"] == "dark"),
        help="Switch between Dark and Light visual themes.",
    )
    st.session_state["theme"] = "dark" if dark_mode_enabled else "light"
    is_dark = st.session_state["theme"] == "dark"

    st.markdown("---")
    st.markdown("### ⚙️ Engine Settings")

    provider_options = []
    if groq_ready:
        provider_options.append("Groq (Ultra-Fast)")
    if openrouter_ready:
        provider_options.append("OpenRouter")

    if not provider_options:
        st.error(
            "No API key found. Add `GROQ_API_KEY` or `OPENROUTER_API_KEY` "
            "to `.env` locally or Streamlit Settings → Secrets when hosted."
        )
        selected_provider = "Groq"
    else:
        chosen_provider = st.selectbox(
            "Primary LLM Provider",
            options=provider_options,
            index=0,
            help="Groq provides sub-second streaming answers. OpenRouter serves as an automatic backup.",
        )
        selected_provider = "Groq" if "Groq" in chosen_provider else "OpenRouter"

    # Model selector based on provider
    if selected_provider == "Groq":
        groq_model = st.selectbox(
            "Groq Model",
            options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
            index=0,
            help="gpt-oss-120b delivers deep comprehension; gpt-oss-20b is ultra-fast for low latency.",
        )
        active_model = groq_model
    else:
        active_model = get_setting("OPENROUTER_MODEL", "openrouter/free")
        st.text(f"Model: {active_model}")

    st.markdown("---")
    st.markdown("### ⚡ Retrieval Speed")
    retrieval_mode = st.radio(
        "Search Pipeline",
        options=["Fast (Direct Vector Similarity)", "Precision (Vector + Cross-Encoder Rerank)"],
        index=0,
        help="Fast mode runs in <20ms using ChromaDB cosine distance. Precision mode reranks top candidates with a cross-encoder model.",
    )
    use_reranker = "Precision" in retrieval_mode

    st.markdown("---")
    st.markdown("### 📡 Provider Status")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Groq**: {'🟢 Active' if groq_ready else '🔴 Missing'}")
    with c2:
        st.markdown(f"**OpenRouter**: {'🟢 Active' if openrouter_ready else '⚪ Not set'}")

    st.markdown("---")
    if st.button("🔄 Reset Chat & Index", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["indexed_signature"] = None
        st.session_state["vector_store"] = None
        st.session_state["document_names"] = []
        st.session_state["chunk_count"] = 0
        st.rerun()

# Theme CSS Definition
if is_dark:
    theme_css = """
    :root {
        --bg-main: #070908;
        --bg-card: #0f1411;
        --bg-card-subtle: #141b17;
        --bg-sidebar: #0a0d0b;
        --border-color: rgba(255, 255, 255, 0.08);
        --border-hover: #10b981;
        --border-accent: rgba(16, 185, 129, 0.3);
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --accent: #10b981;
        --accent-dark: #059669;
        --accent-gradient: linear-gradient(135deg, #34d399 0%, #10b981 50%, #059669 100%);
        --chat-user-bg: #0e1e16;
        --chat-user-border: rgba(16, 185, 129, 0.3);
        --chat-bot-bg: #0f1411;
        --chat-bot-border: rgba(255, 255, 255, 0.08);
        --chip-bg: rgba(16, 185, 129, 0.12);
        --chip-text: #34d399;
        --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.6);
    }
    """
else:
    theme_css = """
    :root {
        --bg-main: #f7f9f8;
        --bg-card: #ffffff;
        --bg-card-subtle: #f0fdf4;
        --bg-sidebar: #ffffff;
        --border-color: #e5e7eb;
        --border-hover: #059669;
        --border-accent: rgba(5, 150, 105, 0.3);
        --text-primary: #111827;
        --text-secondary: #4b5563;
        --text-muted: #9ca3af;
        --accent: #047857;
        --accent-dark: #064e3b;
        --accent-gradient: linear-gradient(135deg, #065f46 0%, #047857 50%, #059669 100%);
        --chat-user-bg: #f0fdf4;
        --chat-user-border: #bbf7d0;
        --chat-bot-bg: #ffffff;
        --chat-bot-border: #e5e7eb;
        --chip-bg: #ecfdf5;
        --chip-text: #065f46;
        --shadow-card: 0 4px 15px -1px rgba(0, 0, 0, 0.05);
    }
    """

# Inject comprehensive responsive styling
st.markdown(
    f"""
    <style>
    {theme_css}

    /* Base Streamlit App Overrides */
    .stApp {{
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }}
    [data-testid="stAppViewContainer"] {{
        background-color: var(--bg-main) !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }}
    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    [data-testid="stBottom"] > div {{
        background-color: var(--bg-main) !important;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, label, div {{
        color: var(--text-primary);
    }}
    p, span, label {{
        color: var(--text-secondary);
    }}

    /* Modern Hero Header */
    .hero-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.5rem;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        box-shadow: var(--shadow-card);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }}
    .hero-title-group {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }}
    .hero-icon {{
        font-size: 2rem;
        line-height: 1;
        padding: 0.5rem;
        background: var(--chip-bg);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .hero-title {{
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }}
    .hero-subtitle {{
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0.2rem 0 0 0;
        font-weight: 500;
    }}
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: var(--chip-bg);
        color: var(--chip-text);
        border: 1px solid var(--border-color);
    }}

    /* Dashboard Metric Card */
    .dashboard-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        box-shadow: var(--shadow-card);
        margin-bottom: 1.25rem;
    }}
    .dashboard-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
    }}
    .metric-item {{
        display: flex;
        flex-direction: column;
    }}
    .metric-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 0.2rem;
    }}
    .metric-value {{
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* File Uploader Customization */
    [data-testid="stFileUploader"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
        box-shadow: var(--shadow-card) !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: var(--bg-card-subtle) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 12px !important;
        transition: border-color 0.2s ease;
    }}
    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: var(--accent) !important;
    }}

    /* Chat Messages Overhaul */
    [data-testid="stChatMessage"] {{
        background: var(--chat-bot-bg) !important;
        border: 1px solid var(--chat-bot-border) !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: var(--shadow-card) !important;
        margin-bottom: 0.85rem !important;
        transition: transform 0.2s ease;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: var(--chat-user-bg) !important;
        border: 1px solid var(--chat-user-border) !important;
    }}
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] div {{
        color: var(--text-primary) !important;
        line-height: 1.6 !important;
    }}

    /* Chat Input Area */
    [data-testid="stChatInput"] textarea {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow-card) !important;
        font-size: 0.95rem !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--border-hover) !important;
    }}

    /* Expanders & Citations */
    [data-testid="stExpander"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        margin-top: 0.75rem !important;
        overflow: hidden;
    }}
    .source-box {{
        padding: 0.75rem;
        border-left: 3px solid var(--accent);
        background: var(--bg-card-subtle);
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.6rem;
    }}
    .source-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        color: var(--accent);
    }}
    .source-content {{
        font-size: 0.82rem;
        line-height: 1.5;
        color: var(--text-secondary);
    }}

    /* Buttons */
    .stButton > button {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        transform: translateY(-1px);
    }}

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {{
        .hero-container {{
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 0.85rem !important;
            padding: 1rem !important;
        }}
        .hero-title {{
            font-size: 1.35rem !important;
        }}
        .hero-subtitle {{
            font-size: 0.8rem !important;
        }}
        .dashboard-grid {{
            grid-template-columns: 1fr 1fr !important;
            gap: 0.75rem !important;
        }}
        [data-testid="stChatMessage"] {{
            padding: 0.75rem 0.9rem !important;
        }}
        .block-container {{
            padding: 1rem 0.75rem !important;
        }}
    }}
    @media (max-width: 480px) {{
        .hero-title {{
            font-size: 1.2rem !important;
        }}
        .dashboard-grid {{
            grid-template-columns: 1fr !important;
        }}
        .hero-icon {{
            font-size: 1.6rem !important;
            padding: 0.4rem !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Professional Header without the old promo subtitle
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title-group">
            <div class="hero-icon">📄</div>
            <div>
                <h1 class="hero-title">DocuQuery AI</h1>
                <p class="hero-subtitle">Interactive Document Intelligence & Analysis</p>
            </div>
        </div>
        <div>
            <span class="badge-pill">⚡ Powered by Groq & ChromaDB</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Document uploader
uploaded_files = st.file_uploader(
    "Upload PDF document(s)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload selectable-text PDFs to index into the local vector database.",
)

# Process uploads only when files change
if uploaded_files:
    file_signature = tuple(sorted((f.name, f.size) for f in uploaded_files))
    if st.session_state["indexed_signature"] != file_signature:
        with st.status("Processing and indexing documents...", expanded=True) as status:
            t0 = time.time()
            st.write("📖 Reading PDF pages and extracting text chunks...")
            chunks, content_hash = load_and_split_pdfs(uploaded_files)

            st.write(f"🧠 Generating embeddings and indexing {len(chunks)} chunks in ChromaDB...")
            collection_name = "docs_" + re.sub(r"[^a-zA-Z0-9_-]", "_", content_hash)
            store = create_or_get_vector_store(chunks, collection_name)

            elapsed = round(time.time() - t0, 2)
            st.session_state["vector_store"] = store
            st.session_state["indexed_signature"] = file_signature
            st.session_state["document_names"] = [f.name for f in uploaded_files]
            st.session_state["chunk_count"] = len(chunks)
            st.session_state["content_hash"] = content_hash
            status.update(
                label=f"✅ Indexed {len(uploaded_files)} file(s) ({len(chunks)} chunks) in {elapsed}s",
                state="complete",
                expanded=False,
            )

# Active document stats banner
if st.session_state["vector_store"] is not None:
    doc_list = ", ".join(st.session_state["document_names"])
    mode_label = "Precision (Reranked)" if use_reranker else "Fast (Cosine)"

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="dashboard-grid">
                <div class="metric-item">
                    <span class="metric-label">Active Document(s)</span>
                    <span class="metric-value" title="{doc_list}">{doc_list}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Text Chunks</span>
                    <span class="metric-value">{st.session_state['chunk_count']} chunks</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Search Pipeline</span>
                    <span class="metric-value">{mode_label}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Engine Provider</span>
                    <span class="metric-value">{selected_provider}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render previous messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"🔍 Citations & Sources ({len(msg['sources'])})"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <div class="source-header">
                                    <span>Source {i}: {src['source']}</span>
                                    <span>Page {src['page']}</span>
                                </div>
                                <div class="source-content">{src['text']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat input for new questions
    user_question = st.chat_input("Ask a question about your document(s)...")

    if user_question:
        # Display user question
        st.session_state["messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Retrieve and generate answer
        with st.chat_message("assistant"):
            try:
                t_retrieval_start = time.time()
                retrieved_docs = retrieve_and_rerank(
                    st.session_state["vector_store"],
                    user_question,
                    candidate_count=6,
                    final_count=4,
                    use_reranker=use_reranker,
                )
                retrieval_time = round(time.time() - t_retrieval_start, 3)

                if not retrieved_docs:
                    reply_text = "No relevant passages were found in the uploaded documents. Please try rephrasing your question."
                    st.warning(reply_text)
                    st.session_state["messages"].append({"role": "assistant", "content": reply_text, "sources": []})
                else:
                    stream_gen, provider_used = generate_answer_stream(
                        user_question,
                        retrieved_docs,
                        preferred_provider=selected_provider,
                        model_override=active_model,
                    )

                    # Stream tokens in real time
                    full_answer = st.write_stream(stream_gen)
                    st.caption(f"⚡ Retrieval: {retrieval_time}s | Engine: {provider_used}")

                    sources_meta = [
                        {
                            "source": doc.metadata.get("source", "Document"),
                            "page": doc.metadata.get("page", "?"),
                            "text": doc.page_content,
                        }
                        for doc in retrieved_docs
                    ]

                    with st.expander(f"🔍 Citations & Sources ({len(sources_meta)})"):
                        for i, src in enumerate(sources_meta, start=1):
                            st.markdown(
                                f"""
                                <div class="source-box">
                                    <div class="source-header">
                                        <span>Source {i}: {src['source']}</span>
                                        <span>Page {src['page']}</span>
                                    </div>
                                    <div class="source-content">{src['text']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": full_answer,
                            "sources": sources_meta,
                        }
                    )
            except LLMUnavailableError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Error generating answer: {exc}")
else:
    st.info("👆 Upload one or more text-based PDF files above to start asking questions.")
