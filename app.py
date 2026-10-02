#!/usr/bin/env python3
"""
app.py

NyayaSahayak — Privacy-Preserving Legal AI Assistant for India
Streamlit UI

This app:
1. Accepts a legal query from the user.
2. Retrieves relevant Supreme Court judgments using hybrid search.
3. Streams an LLM-generated research summary token-by-token.
4. Displays verified sources with full judgment PDF download access.
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure backend modules are importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval_service import HybridRetriever, RetrievalConfig
from backend.context_builder import build_messages
from backend.generation_service import GenerationService, GenerationConfig
from backend.citation_verifier import verify_citations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PDF_DIR = PROJECT_ROOT / "docs" / "judgments"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_OPTIONS = {
    "Mistral 7B Instruct (Q5 — Higher Quality)": "mistral-7b-instruct-q5.gguf",
    "Mistral 7B Instruct (Q4 — Faster)": "mistral-7b-instruct.gguf",
}

DISCLAIMER = (
    "**NyayaSahayak** is a legal research assistant that describes Supreme Court "
    "of India precedents. It is **not a lawyer** and does **not** provide legal "
    "advice. The information presented is for research and educational purposes "
    "only. Always consult a qualified advocate for legal matters."
)


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading retrieval models...")
def load_retriever() -> HybridRetriever:
    config = RetrievalConfig(
        vault_dir=str(PROJECT_ROOT / "vault"),
        collection_name="nyaya_sahayak_v1",
        keyword_db_path=str(PROJECT_ROOT / "vault" / "nyaya_keyword_porter.db"),
        device="cuda",
    )
    return HybridRetriever(config).load()


@st.cache_resource(show_spinner="Loading LLM...")
def load_generator(model_filename: str) -> GenerationService:
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    config = GenerationConfig(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=8192,
        max_tokens=1024,
        temperature=0.1,
        top_p=0.9,
    )
    return GenerationService(config).load()


# ---------------------------------------------------------------------------
# Helper: Find and read a judgment PDF
# ---------------------------------------------------------------------------
def find_judgment_pdf(source_file: str) -> Path | None:
    """
    Searches for the judgment PDF in multiple locations.
    Returns the Path if found, None otherwise.
    """
    if not source_file or source_file == "Unknown":
        return None

    # Search order: docs/judgments/ → demo_pdfs/ → data/raw/
    search_dirs = [
        PROJECT_ROOT / "docs" / "judgments",
        PROJECT_ROOT / "demo_pdfs",
        PROJECT_ROOT / "data" / "raw",
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        pdf_path = search_dir / source_file
        if pdf_path.exists():
            return pdf_path
        # Case-insensitive fallback
        for f in search_dir.iterdir():
            if f.name.lower() == source_file.lower():
                return f

    return None


def read_pdf_bytes(pdf_path: Path) -> bytes | None:
    try:
        return pdf_path.read_bytes()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="NyayaSahayak",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("⚖️ NyayaSahayak")
        st.caption("Privacy-Preserving Legal AI Assistant")
        st.divider()

        # Model selector
        selected_model_label = st.selectbox(
            "Select LLM Model",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            help="Q5 provides higher quality but is slower. Q4 is faster.",
        )
        selected_model_file = MODEL_OPTIONS[selected_model_label]

        st.divider()

        # Retrieval settings
        st.subheader("Retrieval Settings")
        top_k = st.slider("Number of sources", min_value=3, max_value=10, value=6)

        st.divider()

        # Disclaimer
        st.info(DISCLAIMER)

        st.divider()
        st.caption(
            "🔒 **Fully Offline** — All data, models, and processing remain "
            "on your machine. Nothing is sent to any server."
        )

    # -----------------------------------------------------------------------
    # Main content
    # -----------------------------------------------------------------------
    st.header("⚖️ NyayaSahayak")
    st.markdown(DISCLAIMER)
    st.divider()

    # Initialize session state
    if "query_submitted" not in st.session_state:
        st.session_state.query_submitted = False
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""

    # Query input
    query = st.text_area(
        "Describe your legal question:",
        placeholder=(
            "e.g., What has the Supreme Court held regarding preventive detention "
            "under Article 21 and Article 22 of the Constitution?"
        ),
        height=100,
        key="query_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("🔍 Research", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)

    if clear_button:
        st.session_state.query_submitted = False
        st.session_state.current_query = ""
        st.rerun()

    if submit_button and query.strip():
        st.session_state.current_query = query.strip()
        st.session_state.query_submitted = True
        st.rerun()

    # -----------------------------------------------------------------------
    # Process query
    # -----------------------------------------------------------------------
    if st.session_state.query_submitted and st.session_state.current_query:
        user_query = st.session_state.current_query

        # Step 1: Retrieval
        st.subheader("📚 Retrieving Authorities")

        with st.spinner("Searching Supreme Court corpus..."):
            try:
                retriever = load_retriever()
                authorities = retriever.retrieve(
                    query=user_query,
                    final_top_k=top_k,
                )
            except Exception as e:
                st.error(f"Retrieval failed: {e}")
                st.stop()

        if not authorities:
            st.warning(
                "No relevant authorities found for this query. "
                "Try rephrasing or using different legal terms."
            )
            st.session_state.query_submitted = False
            st.stop()

        st.success(f"Found **{len(authorities)}** relevant authorities.")

        # Step 2: Build context and generate
        st.subheader("🤖 Research Summary")

        try:
            messages = build_messages(authorities, user_query)
            generator = load_generator(selected_model_file)

            # Stream the response
            response_placeholder = st.empty()
            full_response = ""

            with st.spinner("Generating research summary..."):
                for token in generator.generate_stream(messages):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

            # Final display without cursor
            response_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

        # Step 3: Citation verification
        st.subheader("✅ Citation Verification")
        verified = verify_citations(full_response, authorities)

        if verified:
            st.success(f"**{len(verified)}** citations verified against retrieved authorities.")
        else:
            st.warning("No citations could be verified. Exercise caution.")

        # Step 4: Display sources with PDF access
        st.subheader("📜 Sources & Full Judgments")
        st.markdown(
            "The authorities below were used to generate the summary above. "
            "Click **Download Judgment** to access the full text of each judgment."
        )

        for idx, citation in enumerate(verified, start=1):
            with st.expander(
                f"**{idx}. {citation.case_name}** — {citation.judgment_date}",
                expanded=(idx == 1),
            ):
                col_info, col_download = st.columns([3, 1])

                with col_info:
                    st.markdown(f"**Case:** {citation.case_name}")
                    st.markdown(f"**Date:** {citation.judgment_date}")
                    st.markdown(f"**Source File:** `{citation.source_file}`")

                with col_download:
                    pdf_path = find_judgment_pdf(citation.source_file)
                    if pdf_path:
                        pdf_bytes = read_pdf_bytes(pdf_path)
                        if pdf_bytes:
                            st.download_button(
                                label="📥 Download Judgment",
                                data=pdf_bytes,
                                file_name=citation.source_file,
                                mime="application/pdf",
                                use_container_width=True,
                            )
                        else:
                            st.warning("⚠️ Could not read PDF file.")
                    else:
                        st.caption("📄 Full PDF not available locally.")

        st.divider()
        st.info(DISCLAIMER)


if __name__ == "__main__":
    main()