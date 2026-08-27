"""
model_loader.py
----------------
Loads a local GGUF LLM and uses it to answer questions grounded in retrieved
Indian Supreme Court precedent. The model always retrieves relevant case
excerpts for every substantive question (except trivial small‑talk), so it
always has context. It never gives prescriptive advice – it only describes
patterns found in the retrieved cases.

Usage (standalone CLI test):
    python backend/model_loader.py "What did Vineeta Sharma vs Rakesh Sharma
      decide about daughters' coparcenary rights?" --passphrase "..."
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from llama_cpp import Llama

from rag_pipeline import query as retrieve, query_open_db  # reuses retrieval

MODEL_PATH = Path("models/mistral-7b-instruct.gguf")

# Pre-vetted tiers – users choose based on hardware.
# The "Light" tier was removed due to incoherent output on this pipeline.
MODEL_OPTIONS = [
    {
        "key": "balanced",
        "label": "Balanced — Mistral-7B-Instruct, Q4_K_M (default)",
        "filename": "mistral-7b-instruct.gguf",
        "ram_note": "~5GB RAM or ~5GB VRAM with GPU offload — good general default",
        "search_hint": "Mistral-7B-Instruct-v0.2 GGUF Q4_K_M",
    },
    {
        "key": "quality",
        "label": "Best Quality — Mistral-7B-Instruct, Q5_K_M or higher",
        "filename": "mistral-7b-instruct-q5.gguf",
        "ram_note": "~6-7GB VRAM — noticeably better reasoning; needs a real GPU (8GB+ VRAM) to stay fast",
        "search_hint": "Mistral-7B-Instruct-v0.2 GGUF Q5_K_M",
    },
]


def get_available_models(models_dir: Path = Path("models")) -> list[dict]:
    """Returns MODEL_OPTIONS annotated with availability."""
    result = []
    for opt in MODEL_OPTIONS:
        path = models_dir / opt["filename"]
        result.append({**opt, "path": path, "available": path.exists()})
    return result


SYSTEM_PROMPT = (
    "You are NyayaSahayak, a professional legal research assistant focused "
    "exclusively on Indian law and Indian Supreme Court precedent. Talk "
    "naturally, like a normal chat assistant – greet the user, ask "
    "clarifying questions about their situation (dates, parties involved, "
    "whether there was a written agreement, prior rulings), and have a "
    "real back-and-forth conversation. Do NOT treat every message as a "
    "database search. You do not provide emotional support or discuss "
    "unrelated topics.\n\n"
    "IMPORTANT – do not repeat yourself: Before asking a question, check "
    "what's already been said in this conversation. Never ask for a detail "
    "(names, dates, amounts, whether there was an agreement, etc.) that the "
    "user already gave you earlier in this same conversation.\n\n"
    "When you have enough understanding of the user's situation, and if the "
    "user asks you to look up case law or precedent, you will be provided "
    "with relevant CASE EXCERPTS from Indian Supreme Court judgments. When "
    "you receive those excerpts, describe the PATTERN found in the retrieved "
    "excerpts as specifically as possible, e.g. 'in the retrieved cases, "
    "courts ruled in favor of X in N out of M instances, generally reasoning "
    "that...' – using real counts and case names from the excerpts you were "
    "given. NEVER tell the user what they personally should do, what decision "
    "to make, or predict their specific outcome. This includes SOFT advice, "
    "not just direct commands – do not suggest 'best practices', risk-"
    "management steps, due diligence measures, policies, safeguards, or say "
    "something 'can help' or 'is essential' for their situation. Even generic-"
    "sounding recommendations count as advice-giving and are not allowed. "
    "State only what the retrieved precedent shows; let the user draw their "
    "own conclusion. This is legal research, not legal advice or consulting.\n\n"
    "If the user asks about anything outside Indian legal matters or their "
    "case, politely decline in your own words and redirect them back – for "
    "example, something like: 'I'm built specifically to help with legal "
    "research on Indian case law – I can't help with that, but I'm happy "
    "to continue discussing your case.'\n\n"
    "Only use citation details (case names, years, SCC numbers, paragraph "
    "numbers) that literally appear in the excerpts you're given – never "
    "invent or infer a citation, paragraph number, or reporter reference "
    "not explicitly shown. If the excerpts don't contain enough information "
    "to answer, say so plainly instead of guessing."
)

# --- Fast-path heuristic for obvious small talk --------------------------------
_CASUAL_PHRASES = {
    "hi", "hii", "hiii", "hello", "hey", "heya", "yo",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "whats up", "what's up", "sup",
    "thanks", "thank you", "thankyou", "ty", "thanks a lot",
    "ok", "okay", "cool", "nice", "great", "got it", "understood",
    "bye", "goodbye", "see you", "see ya", "cya",
}


def is_casual_message(text: str) -> bool:
    normalized = re.sub(r"[!?.,]+$", "", text.strip().lower())
    return normalized in _CASUAL_PHRASES


def should_retrieve(question: str) -> bool:
    """
    Returns True unless the question is casual small‑talk (skip retrieval
    for performance only). All substantive questions go through retrieval.
    """
    return not is_casual_message(question)


def build_retrieval_query(history: list[dict], question: str,
                        max_prior_turns: int = 2) -> str:
    """
    Combines the current question with the last few user turns for
    retrieval, so context from earlier messages is included.
    """
    prior_user_turns = [m["content"] for m in history if m["role"] == "user"]
    prior_user_turns = prior_user_turns[-max_prior_turns:]
    if not prior_user_turns:
        return question
    return " ".join(prior_user_turns + [question])


def _normalize_case_name(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


_SENTENCE_FILLER_WORDS = {
    "based", "as", "according", "also", "see", "furthermore", "however",
    "additionally", "moreover", "note", "importantly", "overall",
    "therefore", "thus", "hence", "this", "that", "these", "those",
    "consistent", "similarly", "likewise", "in", "on", "with",
}
_CONNECTOR_WORDS = {"of", "and", "the", "&", "for", "in", "de"}


def _extract_case_names(text: str) -> list[str]:
    tokens = [(m.group(), m.start()) for m in re.finditer(r"\S+", text)]
    sep_re = re.compile(r"^v(?:s\.?|ersus)?\.?[,:]?$", re.IGNORECASE)
    names = []

    for i, (tok, _) in enumerate(tokens):
        if not sep_re.fullmatch(tok):
            continue

        left_words = []
        j = i - 1
        while j >= 0 and len(left_words) < 8:
            w = tokens[j][0]
            core = w.strip(".,;:()'\"")
            if not core:
                break
            if core.lower() in _SENTENCE_FILLER_WORDS:
                break
            if core[0].isupper() or core.lower() in _CONNECTOR_WORDS or core == "&":
                left_words.insert(0, w)
                if w.endswith(",") or w.endswith("."):
                    break
                j -= 1
            else:
                break

        right_words = []
        k = i + 1
        while k < len(tokens) and len(right_words) < 8:
            w = tokens[k][0]
            core = w.strip(".,;:()'\"")
            if not core:
                break
            if core.lower() in _SENTENCE_FILLER_WORDS:
                break
            if core[0].isupper() or core.lower() in _CONNECTOR_WORDS or core == "&":
                right_words.append(w)
                k += 1
                if w.endswith(",") or (w.endswith(".") and len(core) > 4):
                    break
            else:
                break

        if left_words and right_words:
            names.append(f"{' '.join(left_words)} {tok} {' '.join(right_words)}".strip(" ,."))

    return names


_STOPWORDS_FOR_MATCHING = _CONNECTOR_WORDS | {"v", "vs", "versus", "anr", "ors", "ltd", "etc"}


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS_FOR_MATCHING and len(w) > 1}


def _verify_citations(answer: str, retrieved: dict | None,
                    history: list[dict] | None = None,
                    question: str = "") -> list[str]:
    legitimate_word_sets = []
    if retrieved and retrieved["documents"][0]:
        legitimate_word_sets += [
            _significant_words(m["case_name"]) for m in retrieved["metadatas"][0]
        ]

    user_supplied_text = question + " " + " ".join(
        m["content"] for m in (history or []) if m["role"] == "user"
    )
    user_words = _significant_words(user_supplied_text)

    unverified = []
    for match in _extract_case_names(answer):
        candidate_words = _significant_words(match)
        if not candidate_words:
            continue

        overlap_with_user = len(candidate_words & user_words) / len(candidate_words)

        overlap_with_retrieved = 0.0
        for legit_words in legitimate_word_sets:
            if not legit_words:
                continue
            overlap = len(candidate_words & legit_words) / min(len(candidate_words), len(legit_words))
            overlap_with_retrieved = max(overlap_with_retrieved, overlap)

        if overlap_with_user < 0.6 and overlap_with_retrieved < 0.6:
            unverified.append(match.strip())
    return unverified


def load_model(model_path: Path = MODEL_PATH, n_ctx: int = 8192,
                n_gpu_layers: int = -1) -> Llama:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Download a GGUF model "
            f"(e.g. Mistral-7B-Instruct-v0.2, Q4_K_M quantization) from "
            f"Hugging Face and place it there, or pass a different path."
        )
    print(f"Loading model from {model_path} (this can take ~10-30s)...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=None,
        verbose=False,
    )
    return llm


def _safe_history_window(history: list[dict], max_len: int = 6) -> list[dict]:
    start = max(0, len(history) - max_len)
    if start % 2 != 0:
        start -= 1
    window = history[start:]
    if window and window[-1].get("role") != "assistant":
        window = window[:-1]
    return window


def build_messages(history: list[dict], question: str, retrieved: dict | None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in _safe_history_window(history):
        messages.append({"role": turn["role"], "content": turn["content"]})

    if retrieved and retrieved["documents"][0]:
        context_blocks = []
        for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0]):
            label = f"{meta['case_name']} ({meta.get('year')})"
            context_blocks.append(f"[{label}]\n{doc}")
        context = "\n\n---\n\n".join(context_blocks)
        messages.append({
            "role": "user",
            "content": (
                f"CASE EXCERPTS:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                "INSTRUCTIONS FOR THIS RESPONSE: You already have real case "
                "excerpts above – this is the answer, not another information-"
                "gathering step. Do NOT ask clarifying questions unless the "
                "excerpts are completely insufficient. Analyze the pattern "
                "found in the excerpts above and answer the question directly, "
                "citing the case names given. If the excerpts don't actually "
                "address the question, say so plainly instead of asking for "
                "more details."
            ),
        })
    else:
        messages.append({"role": "user", "content": question})

    return messages


def prepare_generation(db_dir: Path | None, history: list[dict], question: str,
                        top_k: int = 5, passphrase: str | None = None) -> tuple[list[dict], dict | None]:
    """
    Always retrieves (except for casual small talk) and builds the final messages.
    Returns (messages, retrieved) – retrieved is None for casual messages.
    """
    if not should_retrieve(question):
        return build_messages(history, question, None), None

    retrieval_query = build_retrieval_query(history, question)
    if db_dir is not None:
        retrieved = query_open_db(db_dir, retrieval_query, top_k=top_k)
    else:
        retrieved = retrieve(retrieval_query, passphrase, top_k=top_k)

    if not retrieved["documents"][0]:
        retrieved = None
    return build_messages(history, question, retrieved), retrieved


def build_citations_list(retrieved: dict | None) -> list[dict]:
    if not retrieved or not retrieved["documents"][0]:
        return []
    return [
        {
            "case_name": meta["case_name"],
            "year": meta.get("year"),
            "score": score,
            "excerpt": doc[:400] + ("..." if len(doc) > 400 else ""),
            "source_file": meta.get("source_file"),
            "source_url": meta.get("source_url"),
        }
        for doc, meta, score in zip(
            retrieved["documents"][0], retrieved["metadatas"][0], retrieved["scores"][0]
        )
    ]


def finalize_generation(answer: str, retrieved: dict | None, history: list[dict],
                        question: str, truncated: bool) -> dict:
    """
    Returns the final answer, citations, truncation flag, and unverified
    citation warnings. No made_offer flag anymore.
    """
    clean_answer = answer.strip()
    return {
        "answer": clean_answer,
        "citations": build_citations_list(retrieved),
        "truncated": truncated,
        "unverified_citations": _verify_citations(clean_answer, retrieved, history, question),
    }


def answer_question(llm: Llama, question: str, passphrase: str,
                    history: list[dict] | None = None,
                    top_k: int = 5, max_tokens: int = 1536) -> str:
    """One‑off CLI call."""
    history = history or []
    messages, retrieved = prepare_generation(None, history, question, top_k, passphrase)
    output = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.3)
    answer = output["choices"][0]["message"]["content"].strip()
    truncated = output["choices"][0].get("finish_reason") == "length"
    return finalize_generation(answer, retrieved, history, question, truncated)["answer"]


def answer_question_open_db(llm: Llama, db_dir: Path, history: list[dict],
                            question: str, top_k: int = 5,
                            max_tokens: int = 1536) -> dict:
    """Fast path for already‑decrypted vault."""
    messages, retrieved = prepare_generation(db_dir, history, question, top_k)
    output = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.3)
    answer = output["choices"][0]["message"]["content"].strip()
    truncated = output["choices"][0].get("finish_reason") == "length"
    return finalize_generation(answer, retrieved, history, question, truncated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--passphrase", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    args = parser.parse_args()

    llm = load_model(Path(args.model_path))
    answer = answer_question(llm, args.question, args.passphrase, top_k=args.top_k)
    print("\n--- Answer ---\n")
    print(answer)