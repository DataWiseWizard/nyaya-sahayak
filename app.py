"""
app.py
------
Streamlit chat UI for NyayaSahayak.

Key design point: this is a long-lived process, so we do NOT use the
one-shot `unlocked_vault()` context manager (which decrypts/re-encrypts
the whole multi-GB vault around a single operation — fine for a CLI
script, far too slow to repeat before every chat message). Instead:

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
    load_model, prepare_generation, prepare_generation_forced, build_messages,
    finalize_generation, MODEL_PATH, get_available_models, OFFER_MARKER,
    ACCEPTED_OFFER_INSTRUCTION,
)

VAULT_ENCRYPTED = Path("vault/chroma_store.enc")
SALT_PATH = Path("vault/salt.bin")
TMP_DECRYPTED_DIR = Path(".vault_session")

MAX_TOKENS = 1536
BATCH_SECONDS = 0.15  # how many tokens to pull before yielding back to Streamlit for a rerun

st.set_page_config(page_title="NyayaSahayak", page_icon="⚖️", layout="wide")


# --- Cached, expensive, load-once resources -----------------------------------

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
            # Switching models: evict the old one from cache so we don't
            # hold two models resident in VRAM/RAM at the same time.
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


# --- Main chat area --------------------------------------------------------------

st.title("NyayaSahayak — Legal Research Assistant")
st.caption(
    "Talk through your legal matter — I'll research relevant Indian Supreme "
    "Court precedent and describe patterns found in similar cases, with sources."
)

def render_message(content: str, citations: list, truncated: bool = False,
                    unverified_citations: list | None = None):
    st.markdown(content)
    if truncated:
        st.warning(
            "⚠️ This response hit the length limit and may be cut off mid-sentence. "
            "Ask a follow-up like \"continue\" if you'd like the rest."
        )
    if unverified_citations:
        st.error(
            "⚠️ **Unverified citation warning:** this response mentions "
            + ", ".join(f'"{c}"' for c in unverified_citations)
            + " — this doesn't match anything in the retrieved sources or "
            "what you provided. Treat this specific reference with caution "
            "and verify it independently before relying on it."
        )
    if citations:
        with st.expander(f"📚 {len(citations)} source(s)"):
            for c in citations:
                st.markdown(f"**{c['case_name']} ({c['year']})** — relevance {c['score']:.2f}")
                st.caption(c["excerpt"])


def start_generation(llm, display_text: str, history: list, retrieval_mode: str = "auto",
                      effective_question: str | None = None, top_k: int = 5):
    """
    Shared kickoff for both the chat_input path and the offer-response
    buttons. retrieval_mode:
      - "auto"   -> text-based should_retrieve gate (normal typed messages)
      - "forced" -> ALWAYS retrieves, no text detection — used by the
                    "🔍 Search now" button, a deterministic signal
      - "skip"   -> NEVER retrieves, no text detection — used by the
                    "💬 Keep talking" button, equally deterministic
    Appends the user-visible message, prepares the model input, and starts
    streaming (sets session_state so the mid-stream branch picks it up).
    """
    effective_question = effective_question or display_text
    st.session_state.messages.append({"role": "user", "content": display_text})

    if retrieval_mode == "forced":
        messages, retrieved = prepare_generation_forced(
            st.session_state.db_dir, history, effective_question, top_k=top_k
        )
    elif retrieval_mode == "skip":
        messages, retrieved = build_messages(history, effective_question, None), None
    else:
        messages, retrieved = prepare_generation(
            st.session_state.db_dir, history, effective_question, top_k=top_k
        )

    st.session_state.stream_gen = llm.create_chat_completion(
        messages=messages, max_tokens=MAX_TOKENS, temperature=0.3, stream=True,
    )
    st.session_state.stream_text = ""
    st.session_state.stream_retrieved = retrieved
    st.session_state.stream_history = history
    st.session_state.stream_question = effective_question
    st.session_state.stream_active = True
    st.session_state.stop_requested = False


if not st.session_state.vault_open:
    st.info("👈 Unlock the vault in the sidebar with your passphrase to start asking questions.")
else:
    # Load the model once the vault is open (avoids loading a 4GB+ model
    # before the user has even unlocked anything)
    llm = get_llm(st.session_state.selected_model_path)

    # Render everything already finalized in history (not the in-progress stream)
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

    # --- Offer response buttons: if the assistant's last message made a
    # research offer, show explicit Yes/Not-yet buttons instead of relying
    # on free-text "yes"/"no" detection, which was unreliable (missed
    # phrasings like "yes please do X" or full sentences that weren't an
    # exact match to a fixed phrase list).
    last_msg = st.session_state.messages[-1] if st.session_state.messages else None
    show_offer_buttons = (
        not st.session_state.stream_active
        and last_msg is not None
        and last_msg["role"] == "assistant"
        and last_msg.get("made_offer")
    )
    if show_offer_buttons:
        col1, col2 = st.columns(2)
        with col1:
            yes_clicked = st.button("🔍 Yes, search now", use_container_width=True, key="offer_yes_btn")
        with col2:
            no_clicked = st.button("💬 Not yet, let's talk more", use_container_width=True, key="offer_no_btn")

        if yes_clicked or no_clicked:
            history = list(st.session_state.messages)
            if yes_clicked:
                display_text = "✅ Yes, please search now."
                with st.chat_message("user"):
                    st.markdown(display_text)
                with st.spinner("Retrieving relevant precedent..."):
                    start_generation(llm, display_text, history, retrieval_mode="forced",
                                      effective_question=ACCEPTED_OFFER_INSTRUCTION)
            else:
                display_text = "💬 Not yet, let's keep talking."
                with st.chat_message("user"):
                    st.markdown(display_text)
                with st.spinner("Thinking..."):
                    start_generation(llm, display_text, history, retrieval_mode="skip")
            st.rerun()

    if st.session_state.stream_active:
        # --- Mid-stream: pull a small batch of tokens, show a Stop button,
        # then rerun to keep going (or finish up). This is the actual
        # pause/resume mechanic: the generator object itself is persisted
        # in session_state across reruns, and we just call next() on it a
        # few more times each rerun rather than all at once.
        with st.chat_message("assistant"):
            stop_clicked = st.button("⏹ Stop generating")
            if stop_clicked:
                st.session_state.stop_requested = True

            placeholder = st.empty()
            placeholder.markdown(st.session_state.stream_text.replace(OFFER_MARKER, "") + " ▌")

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
                    "made_offer": result["made_offer"],
                })
                st.session_state.stream_active = False
                st.session_state.stream_gen = None
                st.session_state.stream_text = ""
                st.session_state.stop_requested = False
                # Without this rerun, the script just falls through to the
                # rest of the page once — the offer-buttons check earlier
                # in the script already ran for THIS execution and won't
                # see this message until some other interaction (like the
                # next chat_input submit) happens to trigger a rerun. That
                # was the reported bug: buttons only flashing on the next
                # unrelated interaction instead of appearing right away.
                st.rerun()
            else:
                time.sleep(0.02)
                st.rerun()

    with st.expander("📷 Attach a photo of a document (optional — OCR reads the text, not the image)"):
        uploaded_image = st.file_uploader(
            "Upload a photo of a notice, order, or judgment page",
            type=["png", "jpg", "jpeg"], key="ocr_uploader",
        )
        if uploaded_image is not None:
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

        if st.session_state.pending_attachment_text:
            st.caption(
                "OCR isn't perfect — please check this matches the document "
                "before sending. Edit anything that looks wrong."
            )
            st.session_state.pending_attachment_text = st.text_area(
                "Extracted text (edit if needed)",
                value=st.session_state.pending_attachment_text,
                height=150,
            )
            if st.button("✕ Remove attachment"):
                st.session_state.pending_attachment_text = None
                st.session_state.pending_attachment_name = None
                st.rerun()

    question = st.chat_input(
        "Ask about your case, or chat with NyayaSahayak (English or Hinglish)...",
        disabled=st.session_state.stream_active,
    )
    if question and not st.session_state.stream_active:
        # If a document photo is attached, fold its OCR'd text into the
        # actual question sent to the model — this becomes the real record
        # of what was asked, so it's also what gets stored/displayed.
        if st.session_state.pending_attachment_text:
            full_question = (
                f"[Attached document text]\n{st.session_state.pending_attachment_text}"
                f"\n\n{question}"
            )
            st.session_state.pending_attachment_text = None
            st.session_state.pending_attachment_name = None
        else:
            full_question = question

        # Conversation history so far (before this new question) — used for
        # multi-turn retrieval context and prompt continuity.
        history = list(st.session_state.messages)

        with st.chat_message("user"):
            if len(full_question) > 400:
                with st.expander("📷 Attached document + message", expanded=False):
                    st.markdown(full_question)
            else:
                st.markdown(full_question)

        with st.spinner("Retrieving relevant precedent..." if len(full_question.split()) > 2 else "Thinking..."):
            start_generation(llm, full_question, history, retrieval_mode="auto")
        st.rerun()

    st.caption(
        "⚠️ NyayaSahayak researches historical case law and describes patterns "
        "in similar precedent — it does not provide legal advice and cannot "
        "predict your specific outcome. Consult a qualified lawyer for your situation."
    )