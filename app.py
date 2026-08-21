# app.py
# Streamlit UI: upload a PDF, build embeddings, chat with the document.

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from create_database import build_vectorstore
from main import get_retriever, get_llm, answer_question

load_dotenv()

st.set_page_config(
    page_title="Marginalia — chat with your PDF",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design system
# --------------------------------------------------------------------------
# Palette: a reading room at night — near-black ink background, warm brass
# lamplight accent for actions, cool teal for the assistant's voice, and a
# paper-toned card reserved for quoted source material (so a quote always
# reads visually as "lifted from the page").
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink-bg: #0F1116;
    --ink-bg-soft: #161A24;
    --surface: #1C2130;
    --surface-border: rgba(255,255,255,0.08);
    --paper: #F4EDDC;
    --paper-ink: #2B2417;
    --text: #E9ECF4;
    --text-muted: #99A1B8;
    --amber: #E3A23C;
    --amber-soft: rgba(227,162,60,0.14);
    --teal: #5FC9BC;
    --danger: #E2685A;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(1200px 600px at 15% -10%, rgba(227,162,60,0.06), transparent 60%),
        var(--ink-bg);
    color: var(--text);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--ink-bg-soft);
    border-right: 1px solid var(--surface-border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.brand-mark {
    display: flex;
    align-items: baseline;
    gap: 0.55rem;
    margin-bottom: 0.15rem;
}
.brand-mark .glyph {
    font-family: 'Source Serif 4', serif;
    font-size: 1.9rem;
    color: var(--amber);
    line-height: 1;
}
.brand-mark .name {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 1.55rem;
    color: var(--text);
    letter-spacing: 0.2px;
}
.brand-sub {
    color: var(--text-muted);
    font-size: 0.86rem;
    margin-bottom: 1.1rem;
    line-height: 1.4;
}

.key-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    padding: 0.32rem 0.65rem;
    border-radius: 999px;
    background: rgba(95,201,188,0.12);
    color: var(--teal);
    border: 1px solid rgba(95,201,188,0.3);
    margin-bottom: 0.75rem;
}
.key-pill.dot::before {
    content: "";
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--teal);
    display: inline-block;
}

.doc-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: var(--text);
    background: var(--surface);
    border: 1px solid var(--surface-border);
    border-left: 3px solid var(--amber);
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 0.4rem;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid var(--surface-border);
    transition: all 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: var(--amber);
    color: #1A140A;
    border: none;
}
.stButton > button[kind="primary"]:hover {
    background: #F0B25A;
    box-shadow: 0 0 0 3px var(--amber-soft);
}
.stButton > button[kind="secondary"] {
    background: transparent;
    color: var(--text-muted);
}
.stButton > button[kind="secondary"]:hover {
    color: var(--text);
    border-color: var(--text-muted);
}

/* Inputs */
.stTextInput input, .stFileUploader, textarea {
    border-radius: 8px !important;
}

/* Expander (fine-tune panel) */
.streamlit-expanderHeader, div[data-testid="stExpander"] summary {
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* ---------- Main area ---------- */
.hero {
    border: 1px solid var(--surface-border);
    background: linear-gradient(160deg, var(--surface) 0%, var(--ink-bg-soft) 100%);
    border-radius: 16px;
    padding: 2.6rem 2.8rem;
    margin-bottom: 1.5rem;
}
.hero .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.72rem;
    color: var(--amber);
    margin-bottom: 0.9rem;
}
.hero h1 {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 2.3rem;
    line-height: 1.15;
    margin: 0 0 0.7rem 0;
    color: var(--text);
}
.hero p.lead {
    color: var(--text-muted);
    font-size: 1.02rem;
    max-width: 46ch;
    margin-bottom: 1.6rem;
}

.steps { display: flex; gap: 1.1rem; flex-wrap: wrap; }
.step {
    flex: 1 1 220px;
    background: var(--ink-bg);
    border: 1px solid var(--surface-border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
}
.step .tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--teal);
    margin-bottom: 0.4rem;
    display: block;
}
.step .title { font-weight: 600; margin-bottom: 0.25rem; }
.step .desc { color: var(--text-muted); font-size: 0.86rem; line-height: 1.45; }

.doc-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--surface-border);
    padding-bottom: 0.9rem;
    margin-bottom: 1.2rem;
}
.doc-header h2 {
    font-family: 'Source Serif 4', serif;
    font-size: 1.5rem;
    margin: 0;
}
.doc-header .subtitle { color: var(--text-muted); font-size: 0.88rem; margin-top: 0.15rem; }

/* Citation "index card" — the signature element */
.cite-card {
    background: var(--paper);
    color: var(--paper-ink);
    border-radius: 8px;
    padding: 0.75rem 0.9rem 0.7rem 0.9rem;
    margin-bottom: 0.55rem;
    border-left: 4px solid var(--amber);
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}
.cite-card .tab {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.65;
    margin-bottom: 0.3rem;
}
.cite-card .snippet {
    font-family: 'Source Serif 4', serif;
    font-size: 0.92rem;
    line-height: 1.5;
}

div[data-testid="stChatMessage"] {
    border-radius: 12px;
    border: 1px solid var(--surface-border);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "vectorstore": None,
    "retriever": None,
    "llm": None,
    "messages": [],
    "processed_files": [],
    "persist_dir": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_session():
    if st.session_state.persist_dir and os.path.exists(st.session_state.persist_dir):
        shutil.rmtree(st.session_state.persist_dir, ignore_errors=True)
    for key, value in defaults.items():
        st.session_state[key] = value


# Plain-language answer styles, mapped to the technical chunking/retrieval
# parameters behind the scenes. Nobody uploading a PDF should have to know
# what a "chunk" is.
ANSWER_STYLES = {
    "Quick": dict(chunk_size=600, chunk_overlap=100, k=3,
                  desc="Fast, to-the-point answers. Best for short documents."),
    "Balanced": dict(chunk_size=1000, chunk_overlap=200, k=4,
                      desc="A good mix of speed and depth. Works well for most documents."),
    "Thorough": dict(chunk_size=1400, chunk_overlap=250, k=6,
                      desc="Reads more of the document per answer. Best for long or dense PDFs."),
}

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-mark"><span class="glyph">§</span>'
        '<span class="name">Marginalia</span></div>'
        '<div class="brand-sub">Upload a PDF and ask it questions directly — '
        'answers come only from what\'s on the page.</div>',
        unsafe_allow_html=True,
    )

    env_key_present = bool(os.getenv("MISTRAL_API_KEY"))

    if env_key_present:
        st.markdown('<span class="key-pill dot">Connected</span>', unsafe_allow_html=True)
        api_key = os.environ["MISTRAL_API_KEY"]
    else:
        api_key = st.text_input(
            "Mistral API key",
            type="password",
            value="",
            help="Required to generate answers. Kept only in this browser session, never pre-filled.",
        )
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key

    st.divider()

    uploaded_files = st.file_uploader(
        "Your document",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload more than one PDF — they'll be searched together.",
    )

    with st.expander("Fine-tune answers (optional)"):
        style = st.select_slider(
            "Answer style",
            options=list(ANSWER_STYLES.keys()),
            value="Balanced",
        )
        st.caption(ANSWER_STYLES[style]["desc"])

    process_clicked = st.button(
        "Read this PDF →",
        use_container_width=True,
        type="primary",
        disabled=not uploaded_files,
    )

    if st.session_state.processed_files:
        st.markdown("**Currently loaded**")
        for name in st.session_state.processed_files:
            st.markdown(f'<div class="doc-chip">📄 {name}</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Start over", use_container_width=True, type="secondary"):
        reset_session()
        st.rerun()

# --------------------------------------------------------------------------
# Process uploaded PDFs
# --------------------------------------------------------------------------
if process_clicked and uploaded_files:
    if not os.getenv("MISTRAL_API_KEY"):
        st.sidebar.error("Add your Mistral API key first.")
    else:
        params = ANSWER_STYLES[style]
        with st.status("Reading your document...", expanded=True) as status:
            reset_session()

            st.write("Saving upload...")
            tmp_dir = tempfile.mkdtemp(prefix="pdf_upload_")
            saved_paths = []
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(path)

            st.write("Splitting into readable sections and creating embeddings...")
            persist_dir = tempfile.mkdtemp(prefix="chroma_")
            vectorstore, num_pages, num_chunks = build_vectorstore(
                saved_paths,
                persist_directory=persist_dir,
                chunk_size=params["chunk_size"],
                chunk_overlap=params["chunk_overlap"],
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = get_retriever(vectorstore, k=params["k"])
            st.session_state.llm = get_llm()
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.session_state.persist_dir = persist_dir
            st.session_state.messages = []

            shutil.rmtree(tmp_dir, ignore_errors=True)
            status.update(
                label=f"Ready — read {num_pages} pages.", state="complete", expanded=False
            )
        st.rerun()

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
if not st.session_state.retriever:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Marginalia</div>
            <h1>Turn any PDF into a conversation.</h1>
            <p class="lead">Drop a document in the sidebar and ask it anything —
            a contract, a research paper, a manual. Answers are grounded only
            in what's actually written on the page, with the source passage
            shown alongside every reply.</p>
            <div class="steps">
                <div class="step">
                    <span class="tag">01</span>
                    <div class="title">Upload</div>
                    <div class="desc">Add one or more PDFs in the sidebar.</div>
                </div>
                <div class="step">
                    <span class="tag">02</span>
                    <div class="title">Read this PDF</div>
                    <div class="desc">We index the document — usually takes a few seconds.</div>
                </div>
                <div class="step">
                    <span class="tag">03</span>
                    <div class="title">Ask away</div>
                    <div class="desc">Chat naturally. Every answer cites where it came from.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    doc_list = ", ".join(st.session_state.processed_files)
    st.markdown(
        f"""
        <div class="doc-header">
            <div>
                <h2>Ask about your document</h2>
                <div class="subtitle">Currently reading: {doc_list}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def render_sources(docs):
        with st.expander(f"📖 {len(docs)} passage(s) referenced"):
            for doc in docs:
                page = doc.metadata.get("page", "?")
                snippet = doc.page_content[:400].strip()
                st.markdown(
                    f"""
                    <div class="cite-card">
                        <div class="tab">Page {page}</div>
                        <div class="snippet">{snippet}…</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    for msg in st.session_state.messages:
        avatar = "🖋️" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    query = st.chat_input("Ask a question about the document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🖋️"):
            with st.spinner("Reading the relevant pages..."):
                if not os.getenv("MISTRAL_API_KEY"):
                    answer = "Add your Mistral API key in the sidebar to get answers."
                    docs = []
                else:
                    try:
                        answer, docs = answer_question(
                            st.session_state.retriever, st.session_state.llm, query
                        )
                    except Exception as e:
                        answer = f"Something went wrong: {e}"
                        docs = []
            st.markdown(answer)
            if docs:
                render_sources(docs)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": docs}
        )