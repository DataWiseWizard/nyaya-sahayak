"""
model_loader.py
----------------
Loads a local GGUF LLM (Mistral-7B-Instruct or similar) and uses it to hold
a focused, persona-scoped conversation about the user's legal matter,
grounded in retrieved Indian Supreme Court precedent.

Design (agreed with the user):
  - The bot chats CONVERSATIONALLY FIRST — like a normal chat model, not a
    research pipeline. It does NOT search the vault on every message. Only
    once it judges it understands the user's situation well enough does it
    ASK PERMISSION to research similar precedent, rather than diving in
    unprompted. See OFFER_MARKER below for how that offer is detected.
  - Retrieval fires on the current turn only when: (a) the user explicitly
    asks for case research/precedent, or (b) the user affirmatively
    accepts a research offer the assistant just made. Otherwise the
    conversation stays purely conversational — no vault query at all.
  - If the user declines an offer (or gives an ambiguous/unclear reply),
    the assistant keeps chatting normally and is upfront that it won't
    search case law until asked — it does not repeatedly re-offer or
    silently research anyway.
  - It NEVER tells the user what they personally should do or predicts
    their specific outcome — even once research does happen. It states
    the PATTERN found in retrieved precedent (e.g. "in 4 of 5 similar
    cases, courts ruled X") and lets the user draw their own conclusion.
    This is a deliberate line: descriptive precedent, not prescriptive
    advice — see SYSTEM_PROMPT.
  - Obvious small-talk (hi/bye/thanks) skips the expensive retrieval step
    entirely for speed — this is a performance fast-path only.
  - Retrieval uses recent conversation turns, not just the latest message,
    so a case described across several messages retrieves well.

All of the on/off-topic judgment, the offer-timing judgment, and the
never-prescriptive rule are enforced through prompting, not a separate
classifier model — consistent with keeping this simple rather than
building a full agentic router. The offer/consent mechanic and the
never-prescriptive rule both rely on the model reliably following
instructions, which is inherently probabilistic, not mechanically
guaranteed (the citation-verification step is the one place we added a
mechanical safety net on top of prompting, since that specific failure
mode — invented citations — was directly observed).

Usage (standalone CLI test, single question, no conversation history):
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

# Pre-vetted tiers so people on different hardware can self-select honestly
# rather than the app silently being slow/unusable on lower-end machines.
# Filenames are just the expected local name in models/ — the user must
# download the actual GGUF file themselves (can't pip-install model
# weights); search Hugging Face for the model name + "GGUF" to find a
# quantized build from a reputable uploader (e.g. bartowski, or the
# model's own org).
#
# NOTE: a "Light" tier (Phi-4-mini-instruct, ~3.8B) was tried and dropped —
# real testing produced a repetition loop and, separately, incoherent
# word-salad output. Root cause wasn't fully confirmed (candidates: prompt
# instruction density exceeding what a ~3.8B model can hold alongside
# dense retrieved context, or a poor-quality GGUF conversion), but a
# broken "recommended for low-end machines" option is worse than not
# offering one, so it's removed rather than left in as a known-bad choice.
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
    """Returns MODEL_OPTIONS annotated with whether each one's file is
    actually present locally, so the UI can show what's ready to use vs.
    what still needs downloading."""
    result = []
    for opt in MODEL_OPTIONS:
        path = models_dir / opt["filename"]
        result.append({**opt, "path": path, "available": path.exists()})
    return result

OFFER_MARKER = "[[CAN_RESEARCH]]"

SYSTEM_PROMPT = (
    "You are NyayaSahayak, a professional legal research assistant focused "
    "exclusively on Indian law and Indian Supreme Court precedent. Talk "
    "naturally, like a normal chat assistant — greet the user, ask "
    "clarifying questions about their situation (dates, parties involved, "
    "whether there was a written agreement, prior rulings), and have a "
    "real back-and-forth conversation. Do NOT treat every message as a "
    "database search. You do not provide emotional support or discuss "
    "unrelated topics.\n\n"
    "IMPORTANT — do not repeat yourself: Before asking a question or "
    "making an offer, check what's already been said in this conversation. "
    "Never ask for a detail (names, dates, amounts, whether there was an "
    "agreement, etc.) that the user already gave you earlier in this same "
    "conversation. Never repeat a research offer you already made if it's "
    "still pending or was just answered — wait for the user's next message "
    "before considering whether to raise it again.\n\n"
    "IMPORTANT — only research case law when invited to: Do not search or "
    "reference case precedent until either (a) the user explicitly asks "
    "you to look up/research/find similar cases, or (b) you have chatted "
    "enough to understand their situation AND you have asked their "
    "permission to research similar precedent AND they agreed. When you "
    "reach that second point — you understand their situation well enough "
<<<<<<< HEAD
    "to research it — ask permission naturally, IN YOUR OWN WORDS, varying "
    "your phrasing rather than reusing a fixed sentence. For example (vary "
    "these, don't reuse one verbatim): 'Want me to check how courts have "
    "handled similar situations?' / 'I could look into similar Supreme "
    "Court rulings if that'd help — should I?' / 'Should I dig into past "
    "cases like this one?' Immediately after asking that, on a new line at "
    "the very end of your reply, output exactly this marker and nothing "
    "else on that line: " + OFFER_MARKER + " — this is an internal signal, "
    "never mention it or explain it to the user.\n\n"
=======
    "to research it — ask permission naturally. For example, you might ask "
    "if they would like you to research similar Supreme Court rulings. "
    "Immediately after asking that, on a new line at the very end "
    "of your reply, output exactly this marker and nothing else on that "
    "line: " + OFFER_MARKER + " — this is an internal signal, never "
    "mention it or explain it to the user.\n\n"
    "CRITICAL — Avoid repetition: Do not re-ask questions the user has "
    "already answered. Do not repeat a research offer you have already "
    "made in a previous turn unless the user explicitly asks you to. "
>>>>>>> 260ae74e9e0f9af06438bee1ff4d560988fdd2b8
    "If the user declines your offer, or their reply is unclear/not a "
    "clear yes, do NOT research anyway — keep chatting normally and let "
    "them know you'll hold off on researching case law until they ask.\n\n"
    "Once research does happen (excerpts will be provided to you "
    "separately as 'CASE EXCERPTS'): describe the PATTERN found in the "
    "retrieved excerpts as specifically as possible, e.g. 'in the "
    "retrieved cases, courts ruled in favor of X in N out of M instances, "
    "generally reasoning that...' — using real counts and case names from "
    "the excerpts you were given. NEVER tell the user what they personally "
    "should do, what decision to make, or predict their specific outcome. "
    "This includes SOFT advice, not just direct commands — do not suggest "
    "'best practices', risk-management steps, due diligence measures, "
    "policies, safeguards, or say something 'can help' or 'is essential' "
    "for their situation. Even generic-sounding recommendations count as "
    "advice-giving and are not allowed. State only what the retrieved "
    "precedent shows; let the user draw their own conclusion. This is "
    "legal research, not legal advice or consulting.\n\n"
    "If the user asks about anything outside Indian legal matters or their "
    "case, politely decline in your own words and redirect them back — for "
    "example, something like: 'I'm built specifically to help with legal "
    "research on Indian case law — I can't help with that, but I'm happy "
    "to continue discussing your case.'\n\n"
    "Only use citation details (case names, years, SCC numbers, paragraph "
    "numbers) that literally appear in the excerpts you're given — never "
    "invent or infer a citation, paragraph number, or reporter reference "
    "not explicitly shown. If the excerpts don't contain enough information "
    "to answer, say so plainly instead of guessing."
)

# --- Fast-path heuristic for obvious small talk --------------------------------
# Performance optimization ONLY: skips retrieval+re-ranking for the common
# "hi"/"thanks"/"bye" cases so they respond quickly. It is NOT the on/off
# -topic judgment — anything that doesn't match this still goes through
# full retrieval, and whether to actually engage with it is entirely up to
# the model's persona instructions above.
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


# Keyword-based heuristic for "the user is explicitly asking for case
# research/precedent right now" — independent of whether the assistant
# ever made an offer. Best-effort, like the other heuristics here; not a
# classifier, just a fast keyword scan.
_RESEARCH_REQUEST_PHRASES = (
    "search", "look up", "lookup", "look into", "find case", "find similar",
    "find precedent", "similar case", "similar judgment", "case law",
    "check precedent", "check case", "research this", "research my",
    "any judgments", "any cases", "past cases", "previous cases",
    "supreme court ruling", "supreme court case", "precedent on",
    "case history", "search the database", "check the database",
)


def is_explicit_research_request(text: str) -> bool:
    normalized = text.strip().lower()
    return any(phrase in normalized for phrase in _RESEARCH_REQUEST_PHRASES)


# Heuristic for "this reply reads as a clear yes" to a pending research
# offer. Deliberately conservative: anything NOT matching here is treated
# as a decline/ambiguous reply (agreed default — don't research on an
# unclear signal), not as an accept.
_AFFIRMATIVE_PHRASES = {
    "yes", "yes please", "yeah", "yep", "yup", "sure", "sure thing",
    "ok", "okay", "please do", "go ahead", "do it", "please proceed",
    "proceed", "sounds good", "that would help", "please", "yes go ahead",
    "please research", "yes research", "y",
}


def is_affirmative_reply(text: str) -> bool:
    normalized = re.sub(r"[!?.,]+$", "", text.strip().lower())
    return normalized in _AFFIRMATIVE_PHRASES


def should_retrieve(history: list[dict], question: str) -> bool:
    """
    The core retrieval-trigger decision — deliberately flipped from
    "retrieve by default" to "only retrieve when invited to":
<<<<<<< HEAD
      1. An explicit research request ALWAYS triggers retrieval, checked
         FIRST — regardless of whether there's a pending offer. (Bug fix:
         this used to be checked only when there was NO pending offer,
         which meant a fully explicit request like "please look at similar
         cases" got silently swallowed by the narrower yes/no phrase check
         whenever it happened to follow an offer.)
      2. If the assistant's last turn made a research offer (flagged via
         OFFER_MARKER at generation time — see finalize_generation) and
         the message isn't already caught by #1, retrieval fires only if
         this reply reads as a clear yes via the narrow phrase list.
         Anything else (decline, unclear, off-topic) => no retrieval.
      3. Casual small talk never triggers retrieval either way.

    Note: in the Streamlit UI, the button-based accept/decline (see app.py)
    bypasses this function's text-based offer detection entirely for
    reliability — this function's pending-offer branch mainly matters for
    the CLI path, which has no buttons to click.
    """
=======
      1. Retrieval fires if the user explicitly asked for case research
         in this message (e.g. "search the database", "find similar cases").
      2. If the assistant's last turn made a research offer (flagged via
         OFFER_MARKER at generation time — see finalize_generation),
         retrieval ALSO fires if this reply reads as a clear yes.
      3. Casual small talk never triggers retrieval.
    """
    if is_casual_message(question):
        return False

>>>>>>> 260ae74e9e0f9af06438bee1ff4d560988fdd2b8
    if is_explicit_research_request(question):
        return True

    pending_offer = bool(history) and history[-1].get("role") == "assistant" \
        and history[-1].get("made_offer")
    if pending_offer:
        return is_affirmative_reply(question)

    return False


def build_retrieval_query(history: list[dict], question: str,
                           max_prior_turns: int = 2) -> str:
    """
    Combines the current question with the last few user turns for
    retrieval, so a case described across multiple messages (e.g. "it's a
    property dispute between siblings" in one message, "the property was
    already partitioned in 1995" in the next) retrieves relevant precedent
    even when the latest message alone lacks context.
    """
    prior_user_turns = [m["content"] for m in history if m["role"] == "user"]
    prior_user_turns = prior_user_turns[-max_prior_turns:]
    if not prior_user_turns:
        return question
    return " ".join(prior_user_turns + [question])


def _normalize_case_name(name: str) -> str:
    """Lowercase + strip punctuation/whitespace variance for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


# Matches "X vs Y", "X v. Y", "X versus Y" style case name mentions in the
# model's generated answer, so we can cross-check them against what was
# actually retrieved.
# Words that are capitalized only because they start a sentence, not
# because they're part of a case name — used to stop backward/forward
# scanning so we don't swallow surrounding sentence text into the "name".
_SENTENCE_FILLER_WORDS = {
    "based", "as", "according", "also", "see", "furthermore", "however",
    "additionally", "moreover", "note", "importantly", "overall",
    "therefore", "thus", "hence", "this", "that", "these", "those",
    "consistent", "similarly", "likewise", "in", "on", "with",
}
_CONNECTOR_WORDS = {"of", "and", "the", "&", "for", "in", "de"}


def _extract_case_names(text: str) -> list[str]:
    """
    Token-based extraction of "X v. Y" / "X vs Y" / "X versus Y" style case
    name mentions. Scans outward from each "v."/"vs"/"versus" token,
    collecting capitalized words (and common connectors like "of"/"and"/
    "the") on each side, and stops at sentence-filler words or punctuation
    boundaries so it doesn't swallow surrounding sentence text into the
    extracted "name" (a plain regex over character ranges was too greedy
    for this — this is more robust, though still a heuristic, not a parser).
    """
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
                    break  # likely end of a clause/sentence boundary
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
                    break  # likely end of a clause/sentence boundary
            else:
                break

        if left_words and right_words:
            names.append(f"{' '.join(left_words)} {tok} {' '.join(right_words)}".strip(" ,."))

    return names


_STOPWORDS_FOR_MATCHING = _CONNECTOR_WORDS | {"v", "vs", "versus", "anr", "ors", "ltd", "etc"}


def _significant_words(text: str) -> set[str]:
    """Lowercased word set, stripped of connectors/procedural suffixes that
    aren't distinctive enough to matter for matching (e.g. "of", "and",
    "Anr", "Ltd")."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS_FOR_MATCHING and len(w) > 1}


def _verify_citations(answer: str, retrieved: dict | None,
                       history: list[dict] | None = None,
                       question: str = "") -> list[str]:
    """
    Cross-checks every case-name-shaped mention in the generated answer
    against case names that were legitimately available to the model:
    either (a) retrieved from the vault, or (b) supplied directly by the
    user themselves (e.g. pasting in a case they want discussed/advised
    on — perfectly legitimate, not a hallucination). Only flags a mention
    that appears in NEITHER source — i.e. one the model likely invented
    from nowhere.

    Uses word-overlap matching rather than exact substring matching, since
    the model often paraphrases slightly when restating a name (e.g. "the
    case of Pooja Ramesh Singh..." vs. the user's original "Case: Pooja
    Ramesh Singh...") — exact substring matching broke on that harmless
    variation and produced false-positive warnings.

    This is a mechanical safety net on top of the system prompt's "only
    cite what's given" instruction, since that instruction alone isn't
    reliable enough — local 7B-class models can and do still invent
    plausible-looking case names/citation numbers.
    """
    legitimate_word_sets = []
    if retrieved and retrieved["documents"][0]:
        legitimate_word_sets += [
            _significant_words(m["case_name"]) for m in retrieved["metadatas"][0]
        ]

    # Anything the user themselves typed (this message + prior turns) is a
    # legitimate source too — they may be pasting in a real case for advice,
    # which is a normal, expected use of this tool. Treated as one big bag
    # of words rather than requiring contiguous/exact phrasing.
    user_supplied_text = question + " " + " ".join(
        m["content"] for m in (history or []) if m["role"] == "user"
    )
    user_words = _significant_words(user_supplied_text)

    unverified = []
    for match in _extract_case_names(answer):
        candidate_words = _significant_words(match)
        if not candidate_words:
            continue

        # Verified if MOST of the candidate's distinctive words appear in
        # the user's own text (not necessarily contiguous/exact phrasing).
        overlap_with_user = len(candidate_words & user_words) / len(candidate_words)

        # Verified if it substantially overlaps with any single retrieved
        # case name (checked both directions so partial-name mentions of a
        # longer retrieved title still count).
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
    """
    n_ctx is the model's total context WINDOW — prompt + generated response
    combined must fit within it. It's not a fixed hardware limit, just a
    number we choose; Mistral-7B-Instruct supports up to 32k. We raised
    this from 4096 to 8192 since long pasted-in case text (like a full
    judgment someone wants advice on) plus conversation history plus a long
    generated answer can otherwise exceed a smaller window. Larger n_ctx
    uses more VRAM for the KV cache — 8192 is a safe increase on an 8GB
    card with a Q4 7B model; go much higher only if you confirm you still
    have VRAM headroom (check `nvidia-smi` while running).

    n_gpu_layers=-1 offloads ALL model layers to GPU — the fastest option,
    and an RTX 4060 (8GB VRAM) has comfortable headroom for a Q4-quantized
    7B model (~4-5GB) plus context. Requires a CUDA-enabled build of
    llama-cpp-python; with the plain CPU build this argument is silently
    ignored and everything still runs on CPU (slower, but won't error out).
    Pass n_gpu_layers=0 to force CPU-only regardless of what's installed.
    """
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
        n_threads=None,  # None = let llama.cpp auto-detect CPU cores
        verbose=False,
    )
    return llm


def build_messages(history: list[dict], question: str, retrieved: dict | None) -> list[dict]:
    """
    Builds a model-AGNOSTIC list of {"role", "content"} messages (like the
    OpenAI chat format), instead of a hand-built Mistral-specific prompt
    string. This is important: different GGUF models expect different
    chat templates (Mistral's "[INST]...[/INST]", Phi's ChatML-style
    template, etc.) — hardcoding one model's format and feeding it to a
    different model causes it to lose track of turn boundaries, which is
    exactly what produced repetition loops when Phi-4-mini was fed a
    Mistral-shaped prompt. Passing plain messages to
    llm.create_chat_completion() lets llama.cpp apply whichever model's
    own correct template is embedded in that GGUF file.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Cap history length to keep the prompt within the model's context window
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    if retrieved and retrieved["documents"][0]:
        context_blocks = []
        for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0]):
            label = f"{meta['case_name']} ({meta.get('year')})"
            context_blocks.append(f"[{label}]\n{doc}")
        context = "\n\n---\n\n".join(context_blocks)
        # This reinforcement is deliberately placed LAST, right next to the
        # excerpts, rather than relying only on the system prompt (which
        # sits far away and competes with many other rules — persona,
        # anti-repetition, offer-phrasing, citation rules — for attention).
        # Real testing showed the model ignoring provided excerpts and
        # just re-asking earlier questions / making another offer instead
        # of analyzing what it was just given; a distant general
        # instruction wasn't enough to prevent that.
        messages.append({
            "role": "user",
            "content": (
                f"CASE EXCERPTS:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                "INSTRUCTIONS FOR THIS RESPONSE: You already researched and "
                "have real case excerpts above — this is the answer, not "
                "another information-gathering step. Do NOT ask clarifying "
                "questions. Do NOT make another research offer or repeat "
                "the offer marker in this response — you already have what "
                "you need. Analyze the pattern found in the excerpts above "
                "and answer the question directly, citing the case names "
                "given. If the excerpts don't actually address the "
                "question, say so plainly instead of asking for more details."
            ),
        })
    else:
        messages.append({"role": "user", "content": question})

    return messages


ACCEPTED_OFFER_INSTRUCTION = (
    "Yes — please go ahead and research similar Supreme Court precedent "
    "for the situation I described earlier in our conversation, and share "
    "your analysis of the pattern found in those cases."
)


def prepare_generation_forced(db_dir: Path | None, history: list[dict],
                               effective_question: str, top_k: int = 5,
                               passphrase: str | None = None) -> tuple[list[dict], dict | None]:
    """
    ALWAYS retrieves, bypassing should_retrieve entirely. Use this when
    something outside this module already made a deterministic decision to
    research — e.g. the Streamlit UI's "🔍 Search now" button — so there's
    no need to (and no benefit to) re-deriving that decision from fuzzy
    text matching. `effective_question` should be a real, actionable
    instruction (not necessarily the user's literal message) — see
    ACCEPTED_OFFER_INSTRUCTION for why a bare "yes" doesn't work here.
    """
    retrieval_query = build_retrieval_query(history, effective_question)
    if db_dir is not None:
        retrieved = query_open_db(db_dir, retrieval_query, top_k=top_k)
    else:
        retrieved = retrieve(retrieval_query, passphrase, top_k=top_k)

    if not retrieved["documents"][0]:
        retrieved = None
    return build_messages(history, effective_question, retrieved), retrieved


def prepare_generation(db_dir: Path | None, history: list[dict], question: str,
                        top_k: int = 5, passphrase: str | None = None) -> tuple[list[dict], dict | None]:
    """
    Shared prep step for both the blocking and streaming (app.py) code
    paths: decides whether retrieval is needed via should_retrieve's
    text-based judgment (skips it for casual small talk or an unclear
    reply), runs it if so, and builds the final messages list. Returns
    (messages, retrieved) — retrieved is None for casual messages or when
    nothing relevant was found.

    Pass db_dir for the already-open-vault fast path (Streamlit), or leave
    db_dir=None and pass passphrase for a one-off CLI-style call.

    For a DETERMINISTIC trigger (e.g. a UI button click) rather than
    inferring intent from text, use prepare_generation_forced() instead.
    """
    if not should_retrieve(history, question):
        return build_messages(history, question, None), None

    # If retrieval is firing because the user just accepted a research
    # offer, their literal reply (e.g. "yes") carries no case information
    # by itself. Handing the model "CASE EXCERPTS: [...] QUESTION: yes"
    # gives it nothing real to act on — which is exactly what caused it to
    # just repeat its previous offer instead of answering. Substitute an
    # explicit instruction here; the actual case description still comes
    # from the conversation history included in build_messages/retrieval.
    pending_offer = bool(history) and history[-1].get("role") == "assistant" \
        and history[-1].get("made_offer")
    effective_question = (
        ACCEPTED_OFFER_INSTRUCTION
        if pending_offer and is_affirmative_reply(question)
        else question
    )

    return prepare_generation_forced(db_dir, history, effective_question, top_k, passphrase)


def build_citations_list(retrieved: dict | None) -> list[dict]:
    if not retrieved or not retrieved["documents"][0]:
        return []
    return [
        {
            "case_name": meta["case_name"],
            "year": meta.get("year"),
            "score": score,
            "excerpt": doc[:400] + ("..." if len(doc) > 400 else ""),
        }
        for doc, meta, score in zip(
            retrieved["documents"][0], retrieved["metadatas"][0], retrieved["scores"][0]
        )
    ]


def finalize_generation(answer: str, retrieved: dict | None, history: list[dict],
                         question: str, truncated: bool) -> dict:
    """Shared finalization for both code paths: strips the internal
    research-offer marker (if present) from the visible answer, builds the
    citations list, and runs hallucination-detection citation verification.
    Returns made_offer=True when this response ended with a research
    offer, so the caller can store that flag on the message — checked by
    should_retrieve() on the NEXT turn to know a reply is answering an
    offer rather than starting a new topic."""
    made_offer = OFFER_MARKER in answer
    clean_answer = answer.replace(OFFER_MARKER, "").strip()
    return {
        "answer": clean_answer,
        "citations": build_citations_list(retrieved),
        "truncated": truncated,
        "unverified_citations": _verify_citations(clean_answer, retrieved, history, question),
        "made_offer": made_offer,
    }


def answer_question(llm: Llama, question: str, passphrase: str,
                     history: list[dict] | None = None,
                     top_k: int = 5, max_tokens: int = 1536) -> str:
    """One-off CLI-style call: opens the vault, retrieves, re-locks, generates.
    For a long-lived session (many questions), use answer_question_open_db
    instead — see app.py."""
    history = history or []
    messages, retrieved = prepare_generation(None, history, question, top_k, passphrase)
    output = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.3)
    answer = output["choices"][0]["message"]["content"].strip()
    truncated = output["choices"][0].get("finish_reason") == "length"
    return finalize_generation(answer, retrieved, history, question, truncated)["answer"]


def answer_question_open_db(llm: Llama, db_dir: Path, history: list[dict],
                             question: str, top_k: int = 5,
                             max_tokens: int = 1536) -> dict:
    """
    Fast path for a long-lived session where the vault is already
    decrypted (via encryptor.open_vault) and the model is already loaded.
    `history` is the prior conversation (list of {"role", "content"} dicts,
    NOT including the current `question`) — used both for multi-turn
    retrieval context and for prompt continuity.

    Returns {"answer": str, "citations": [...], "truncated": bool,
    "unverified_citations": [...]} so the UI can show sources, flag a
    cut-off response, and warn about any case-name mention that couldn't
    be matched to something actually retrieved (a likely hallucination).

    For a STREAMING version of this (token-by-token, with a stop button),
    see prepare_generation() + finalize_generation() used directly in
    app.py instead of this all-in-one blocking call.
    """
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