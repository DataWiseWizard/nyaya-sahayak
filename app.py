"""
app.py
------
Streamlit chat UI for NyayaSahayak.

Key design point: this is a long-lived process, so I did NOT use the
one-shot `unlocked_vault()` context manager. Instead:

  1. On passphrase submit -> encryptor.open_vault() ONCE, keep the
     decrypted directory open in st.session_state for the whole session.
  2. Every question -> model_loader.answer_question_open_db() reuses that
     already-open directory + already-loaded model.
  3. "Lock Vault" button (or closing the app / Ctrl+C in the terminal) ->
     re-encrypts and securely wipes the decrypted copy.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import time  # noqa: E402

from encryptor import open_vault  # noqa: E402
from ocr import extract_text_from_image, OCRNotAvailableError  # noqa: E402
from model_loader import (  # noqa: E402
    load_model, prepare_generation, build_messages, finalize_generation,
    MODEL_PATH, get_available_models,
)

VAULT_ENCRYPTED = Path("vault/chroma_store.enc")
SALT_PATH = Path("vault/salt.bin")
TMP_DECRYPTED_DIR = Path(".vault_session")

MAX_TOKENS = 1536
BATCH_SECONDS = 0.15  # tokens to stream per rerun

st.set_page_config(page_title="NyayaSahayak", page_icon="⚖️", layout="wide")

# --- Custom CSS for a cleaner UI ----------------------------------------------
st.markdown("""
<style>
    div[data-testid="stFileUploader"] {
        background-color: #2b2b36 !important;
        border-radius: 28px !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease !important;
        height: 56px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin-bottom: 0 !important;
        overflow: hidden !important;
    }

    div[data-testid="stFileUploader"]:hover {
        background-color: #3a3a46 !important;
    }

    div[data-testid="stFileUploader"] section {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 0 !important;
        min-width: 0 !important;
        width: 100% !important;
        height: 100% !important;
    }

    div[data-testid="stFileUploader"] section button {
        background: transparent !important;
        border: none !important;
        color: inherit !important;
    }

    div[data-testid="stFileUploaderDropzoneInstructions"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Cached resources ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading local LLM (first load can take ~30s)...")
def get_llm(model_path_str: str):
    return load_model(Path(model_path_str))


# --- Session state init --------------------------------------------------------

if "vault_open" not in st.session_state:
    st.session_state.vault_open = False
    st.session_state.db_dir = None
    st.session_state.close_vault_fn = None
    st.session_state.messages = []  # [{role, content, citations?}]
    st.session_state.stream_active = False
    st.session_state.stream_gen = None
    st.session_state.stream_text = ""
    st.session_state.stream_retrieved = None
    st.session_state.stream_history = None
    st.session_state.stream_question = None
    st.session_state.stop_requested = False
    st.session_state.pending_attachment_text = None
    st.session_state.pending_attachment_name = None
    st.session_state.selected_model_path = str(MODEL_PATH)


def lock_vault():
    if st.session_state.close_vault_fn:
        st.session_state.close_vault_fn()
    st.session_state.vault_open = False
    st.session_state.db_dir = None
    st.session_state.close_vault_fn = None


# --- Sidebar: unlock / lock controls -------------------------------------------

with st.sidebar:
    st.title("⚖️ NyayaSahayak")
    st.caption("Privacy-preserving legal AI — fully offline, encrypted at rest.")

    st.subheader("🧠 Model")
    available_models = get_available_models()
    downloaded = [m for m in available_models if m["available"]]

    if not downloaded:
        st.error(
            "No model files found in `models/`. Download at least one GGUF "
            "model (see options below) before unlocking the vault."
        )
    else:
        labels = [m["label"] for m in downloaded]
        current_idx = next(
            (i for i, m in enumerate(downloaded) if str(m["path"]) == st.session_state.selected_model_path),
            0,
        )
        chosen_label = st.selectbox("Choose a model", labels, index=current_idx)
        chosen = next(m for m in downloaded if m["label"] == chosen_label)
        new_path = str(chosen["path"])
        if new_path != st.session_state.selected_model_path:
            get_llm.clear()
            st.session_state.selected_model_path = new_path
            st.rerun()
        st.caption(chosen["ram_note"])

    with st.expander("Need a different model? / hardware guide"):
        st.markdown(
            "Pick based on your hardware — none of these are pip-installable, "
            "you need to manually download a `.gguf` file into `models/`:"
        )
        for m in available_models:
            status = "✅ downloaded" if m["available"] else "⬇️ not downloaded"
            st.markdown(
                f"**{m['label']}** — {status}\n"
                f"- {m['ram_note']}\n"
                f"- Save as: `models/{m['filename']}`\n"
                f"- Search Hugging Face for: *\"{m['search_hint']}\"*"
            )

    st.divider()

    if not VAULT_ENCRYPTED.exists():
        st.error(
            f"No vault found at `{VAULT_ENCRYPTED}`. Run "
            f"`python backend/rag_pipeline.py ingest --passphrase ...` first."
        )

    if not st.session_state.vault_open:
        st.subheader("🔒 Vault Locked")
        passphrase = st.text_input("Enter your passphrase", type="password")
        if st.button("Unlock", type="primary", use_container_width=True, disabled=not downloaded):
            if not passphrase:
                st.warning("Enter a passphrase first.")
            else:
                with st.spinner("Decrypting vault (this reads the whole "
                                "store once, may take a moment)..."):
                    try:
                        db_dir, close_fn = open_vault(
                            VAULT_ENCRYPTED, SALT_PATH, passphrase, TMP_DECRYPTED_DIR
                        )
                        st.session_state.db_dir = db_dir
                        st.session_state.close_vault_fn = close_fn
                        st.session_state.vault_open = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to unlock: {e}")
    else:
        st.subheader("🔓 Vault Unlocked")
        st.caption(
            "Stays open for this session for fast responses. Lock it when "
            "you're done, or it auto-locks if you stop the app (Ctrl+C)."
        )
        if st.button("🔒 Lock Vault", use_container_width=True):
            lock_vault()
            st.rerun()

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main chat area -------------------------------------------------------------

st.title("NyayaSahayak — Legal Research Assistant")
st.caption(
    "Talk through your legal matter — I'll research relevant Indian Supreme "
    "Court precedent and describe patterns found in similar cases, with sources."
)


def render_message(content: str, citations: list, truncated: bool = False,
                    unverified_citations: list | None = None):
    """Renders a message with citations and links to full-text pages."""
    st.markdown(content)
    if truncated:
        st.warning("⚠️ This response hit the length limit and may be cut off...")
    if unverified_citations:
        st.error(
            "⚠️ **Unverified citation warning:** ..."
        )
    if citations:
        with st.expander(f"📚 {len(citations)} source(s)"):
            for i, c in enumerate(citations):
                st.markdown(f"**{c['case_name']} ({c['year']})** — relevance {c['score']:.2f}")
                st.caption(c["excerpt"])

                source_url = c.get("source_url")
                source_file = c.get("source_file")
                if source_url:
                    # Preferred: an Indian Kanoon URL that was embedded in
                    # the judgment text itself at ingestion time — opens
                    # the real full judgment on an actual legal database,
                    # no custom viewer needed.
                    st.markdown(f"📄 [Read full judgment]({source_url})")
                elif source_file and Path(source_file).exists():
                    # Fallback: no URL found in this judgment's text, but
                    # I have the local source PDF — offer it as a direct
                    # download (a plain file:// link doesn't work reliably
                    # from a browser-served Streamlit app).
                    try:
                        with open(source_file, "rb") as f:
                            st.download_button(
                                "📄 Open full judgment (PDF)",
                                data=f.read(),
                                file_name=Path(source_file).name,
                                mime="application/pdf",
                                key=f"dl_{i}_{Path(source_file).name}",
                            )
                    except OSError:
                        st.caption("_(source file couldn't be read)_")
                else:
                    st.caption(
                        "_(no link available for this citation — re-run "
                        "ingestion to enable links/downloads for older data)_"
                    )


def start_generation(llm, display_text: str, history: list, question: str,
                    top_k: int = 5):
    """
    Shared kickoff for both the chat_input path. Always retrieves (unless
    casual, which is handled inside prepare_generation). Appends the user
    message, prepares the model input, and starts streaming.
    """
    st.session_state.messages.append({"role": "user", "content": display_text})

    messages, retrieved = prepare_generation(
        st.session_state.db_dir, history, question, top_k=top_k
    )

    st.session_state.stream_gen = llm.create_chat_completion(
        messages=messages, max_tokens=MAX_TOKENS, temperature=0.3, stream=True,
    )
    st.session_state.stream_text = ""
    st.session_state.stream_retrieved = retrieved
    st.session_state.stream_history = history
    st.session_state.stream_question = question
    st.session_state.stream_active = True
    st.session_state.stop_requested = False


if not st.session_state.vault_open:
    st.info("👈 Unlock the vault in the sidebar with your passphrase to start asking questions.")
else:
    # Load the model once the vault is open
    llm = get_llm(st.session_state.selected_model_path)

    # Render all already‑finalized messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_message(
                    msg["content"], msg.get("citations", []),
                    msg.get("truncated", False), msg.get("unverified_citations", []),
                )
            elif len(msg["content"]) > 400:
                with st.expander("📷 Attached document + message", expanded=False):
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

    # --- Streaming message area (if active) ------------------------------------
    if st.session_state.stream_active:
        with st.chat_message("assistant"):
            # Show a Stop button above the streaming text
            stop_clicked = st.button("⏹ Stop generating")
            if stop_clicked:
                st.session_state.stop_requested = True

            placeholder = st.empty()
            placeholder.markdown(st.session_state.stream_text + " ▌")

            finished = False
            truncated = False
            if not st.session_state.stop_requested:
                deadline = time.time() + BATCH_SECONDS
                try:
                    while time.time() < deadline:
                        chunk = next(st.session_state.stream_gen)
                        delta = chunk["choices"][0].get("delta", {})
                        st.session_state.stream_text += delta.get("content", "") or ""
                        if chunk["choices"][0].get("finish_reason") == "length":
                            truncated = True
                except StopIteration:
                    finished = True

            if finished or st.session_state.stop_requested:
                result = finalize_generation(
                    st.session_state.stream_text.strip(),
                    st.session_state.stream_retrieved,
                    st.session_state.stream_history,
                    st.session_state.stream_question,
                    truncated=truncated and not st.session_state.stop_requested,
                )
                placeholder.empty()
                render_message(
                    result["answer"], result["citations"],
                    result["truncated"], result["unverified_citations"],
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result["citations"],
                    "truncated": result["truncated"],
                    "unverified_citations": result["unverified_citations"],
                })
                st.session_state.stream_active = False
                st.session_state.stream_gen = None
                st.session_state.stream_text = ""
                st.session_state.stop_requested = False
                st.rerun()
            else:
                time.sleep(0.02)
                st.rerun()

    # --- Input area: uploader + chat input + stop button ------------------------
    # I used columns to place upload icon and a stop button (when streaming)
    # above the chat input. The chat input itself stays as a separate widget.
    # If streaming, I'll show a stop button; otherwise I'll show the uploader.

    col1, col2 = st.columns([1, 13], vertical_alignment="bottom")
    with col1:
        uploaded_image = st.file_uploader(
            "📷",
            type=["png", "jpg", "jpeg"],
            key="ocr_uploader",
            label_visibility="collapsed",
            accept_multiple_files=False,
        )
        if uploaded_image is not None:
            max_size = 10 * 1024 * 1024  # 10 MB
            if uploaded_image.size > max_size:
                st.error(f"File too large ({uploaded_image.size / 1024 / 1024:.1f} MB). Please upload an image smaller than 10 MB.")
                st.session_state.pending_attachment_text = None
                st.session_state.pending_attachment_name = None
            else:
                if st.session_state.pending_attachment_name != uploaded_image.name:
                    with st.spinner("Reading text from the image..."):
                        try:
                            extracted = extract_text_from_image(uploaded_image.getvalue())
                            st.session_state.pending_attachment_text = extracted
                            st.session_state.pending_attachment_name = uploaded_image.name
                        except OCRNotAvailableError as e:
                            st.error(str(e))
                        except ValueError as e:
                            st.error(f"Couldn't process this file: {e}")

        # Show attachment preview if present
        if st.session_state.pending_attachment_text:
            st.caption(f"📎 {st.session_state.pending_attachment_name}")
            if st.button("✕ Remove"):
                st.session_state.pending_attachment_text = None
                st.session_state.pending_attachment_name = None
                st.rerun()

    with col2:
        # If streaming, show a Stop button; else show the chat input.
        if st.session_state.stream_active:
            st.button("⏹ Stop generating", type="primary", use_container_width=True,
                    on_click=lambda: setattr(st.session_state, 'stop_requested', True))
        else:
            question = st.chat_input(
                "Ask about your case, or chat with NyayaSahayak (English or Hinglish)...",
                disabled=st.session_state.stream_active,
            )
            if question and not st.session_state.stream_active:
                # If a document photo is attached, fold its text into the question.
                if st.session_state.pending_attachment_text:
                    full_question = (
                        f"[Attached document text]\n{st.session_state.pending_attachment_text}"
                        f"\n\n{question}"
                    )
                    st.session_state.pending_attachment_text = None
                    st.session_state.pending_attachment_name = None
                else:
                    full_question = question

                history = list(st.session_state.messages)

                with st.chat_message("user"):
                    if len(full_question) > 400:
                        with st.expander("📷 Attached document + message", expanded=False):
                            st.markdown(full_question)
                    else:
                        st.markdown(full_question)

                with st.spinner("Retrieving relevant precedent..." if len(full_question.split()) > 2 else "Thinking..."):
                    start_generation(llm, full_question, history, question)
                st.rerun()

    st.caption(
        "⚠️ NyayaSahayak researches historical case law and describes patterns "
        "in similar precedent — it does not provide legal advice and cannot "
        "predict your specific outcome. Consult a qualified lawyer for your situation."
    )