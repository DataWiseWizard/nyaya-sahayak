# NyayaSahayak — Privacy-Preserving Legal AI Assistant for India

**Talk through your legal matter with a fully offline, encrypted AI assistant that researches relevant Indian Supreme Court precedent — and describes what it finds, without pretending to be your lawyer.**

This document reflects the project **as it actually stands today**, not the original plan. Quite a lot changed along the way — some of that is called out explicitly below, because the reasoning behind those changes is part of the engineering story.

---

## 1. Status at a glance

| Area | Status |
|---|---|
| Dataset pipeline (PDF corpus → cleaned text) | ✅ Done |
| Encrypted vector store (AES-256 vault) | ✅ Done, hardened through real bugs |
| Retrieval + re-ranking | ✅ Done |
| Citation hallucination verification | ✅ Done |
| Conversational persona (legal-only, never-prescriptive) | ✅ Done, prompt occasionally drifts — see Known Issues |
| Consent-gated research ("ask before searching") | 🟡 **Currently broken — being redesigned, see below** |
| Streaming responses + stop button | ✅ Done |
| OCR document upload | ✅ Done |
| Multi-model tiers (Light/Balanced/Quality) | 🟡 Light tier unreliable — likely being dropped |
| GPU acceleration (CUDA) | ✅ Done |
| Fine-tuning (QLoRA) | ❌ Not started |
| Formal evaluation / benchmarking | ❌ Not started |
| Automated tests | ❌ Not started (all testing has been manual/interactive so far) |
| Setup documentation for others | ❌ Not started (this doc is the first step) |

---

## 2. What this project actually is

NyayaSahayak is a fully offline, privacy-first assistant for exploring Indian Supreme Court precedent. A person describes a legal situation conversationally; once the assistant understands it well enough, it asks permission to research similar precedent, retrieves relevant judgments from a local encrypted corpus, and describes the **pattern** found in those cases — without ever telling the user what they personally should do.

Everything runs locally: the language model, the embeddings, the re-ranking, the vector store. Nothing is sent to a third-party API. This is the actual differentiator, not an incidental detail — see §4.

**What it is not:** a lawyer, a source of legal advice, or a system that predicts case outcomes. It describes precedent; it does not prescribe action. This distinction is enforced (imperfectly — see Known Issues) through the system prompt.

---

## 3. How the plan changed from the original draft

The original plan (see git history / earlier draft) assumed:
- A single flat CSV of judgments from Kaggle
- A 6-week solo timeline including QLoRA fine-tuning by week 2
- A simple "encrypt the ChromaDB file" wrapper
- A single fixed LLM
- An always-retrieve chatbot (every message triggers a database search)

What actually happened, and why:

- **Dataset turned out to be a folder of ~35,000+ PDFs organized by year (1950–2025), not a CSV.** The Kaggle dataset's real structure only became clear after downloading it — the ingestion pipeline (`scripts/clean_dataset.py`) was rewritten around PDF text extraction and filename parsing (`Party1_vs_Party2_on_DD_Month_YYYY.PDF`) instead of CSV columns.
- **Fine-tuning was deprioritized in favor of RAG quality and safety engineering.** Early on, the recommendation was to treat QLoRA as a stretch goal rather than a week-2 dependency, since system design and retrieval correctness matter more for a working portfolio piece than whether the model was fine-tuned. This held — fine-tuning still hasn't happened, and the project is stronger for having spent that time elsewhere instead (citation verification, encryption lifecycle correctness, persona design).
- **The encryption design is significantly more involved than "encrypt a file."** ChromaDB's persistent store is a *directory* (SQLite + index files), not a single file, which wasn't obvious until it broke. The vault now zips the whole directory in memory, encrypts that, and follows a decrypt-to-temp → use → re-encrypt-and-securely-wipe lifecycle with crash safety (`atexit`/signal handlers, adapted again once it turned out Streamlit runs scripts in a non-main thread where `signal.signal()` isn't allowed).
- **The chatbot's entire interaction model was redesigned, twice.** First from "answer any question via RAG" to a scoped persona that redirects off-topic chat and never gives prescriptive advice. Then from "retrieve on every non-greeting message" to "chat conversationally first, only research when explicitly asked or when offered and accepted" — closer to how an actual consultation works. **This second change is the part currently being debugged** (see §6).
- **A citation-verification safety net was added that wasn't in any original plan.** During testing, the model fabricated a citation for a case that couldn't exist in the corpus's date range. Prompt instructions alone ("only cite what's given") weren't reliable enough, so a mechanical post-generation check now cross-references every case name the model mentions against what was actually retrieved (or supplied directly by the user), flagging anything that doesn't match. This is arguably the single most distinctive piece of engineering in the project.
- **Multi-model support and GPU offload were added** once real hardware constraints (CPU-only response times of 60–90s) made it clear a single fixed model wasn't going to be a good experience across different machines.
- **Streaming + stop button and OCR upload** were added as direct responses to real usability friction encountered during testing, not part of the original scope.

---

## 4. Why privacy-first, and why local-only (not "private cloud")

Legal documents are sensitive; sending them to third-party APIs is the exact objection most Indian law firms have to existing AI tools. This project's core claim — *nothing leaves your machine* — is a **structural guarantee**, not a policy promise: there's no server that could see the data even if it wanted to, because there is no server.

This was deliberately weighed against alternatives (see project discussion history):
- **Third-party hosted APIs** (even for open-weight models) were rejected — they would directly reintroduce the exact problem the project exists to solve, regardless of any provider's data-handling promises.
- **Self-hosted cloud / on-prem servers** (a firm's own private infrastructure) is a legitimate middle ground for future enterprise deployment, but is **not implemented** — it's noted here as a possible future direction, not a current feature.
- **Confidential computing / homomorphic encryption** exists in theory but is out of scope — too immature for practical local LLM inference at this project's scale.

The tradeoff this creates — heavier setup, hardware-dependent performance — is treated as inherent to the actual value proposition, not a flaw to be engineered away.

---

## 5. Architecture

```
┌─────────────┐     ┌────────────────┐     ┌──────────────────┐
│  Streamlit  │────▶│  model_loader  │────▶│   ChromaDB +      │
│   Chat UI   │◀────│  (llama-cpp,   │◀────│   AES-256 vault   │
│  (app.py)   │     │  chat API)     │     │  (rag_pipeline,    │
└─────────────┘     └───────┬────────┘     │   encryptor)       │
                             │              └──────────────────┘
                     ┌───────▼────────┐
                     │  Local GGUF    │
                     │  LLM (GPU/CPU) │
                     └────────────────┘
```

- **Frontend:** Streamlit (`app.py`) — chat interface, passphrase-gated vault unlock, model tier picker, OCR upload, streaming responses with a stop button.
- **Retrieval:** `backend/rag_pipeline.py` — chunks and embeds judgments (`sentence-transformers`, multilingual model for English/Hinglish), stores in ChromaDB, retrieves + re-ranks via a cross-encoder (`backend/reranker.py`).
- **Encryption:** `backend/encryptor.py` — AES-256-CBC, PBKDF2 key derivation (480,000 iterations), whole-directory encryption (zip-then-encrypt) so the entire ChromaDB store is encrypted at rest. Supports both one-shot (`unlocked_vault()`, CLI-style) and long-lived (`open_vault()`/`close_fn()`, for the Streamlit session) usage patterns.
- **Generation:** `backend/model_loader.py` — loads a GGUF model via `llama-cpp-python`, builds model-agnostic chat messages (not a hardcoded prompt template, so different model families' own chat templates are respected), runs the persona/safety system prompt, and performs post-generation citation verification.
- **OCR:** `backend/ocr.py` — Tesseract-based text extraction from uploaded document photos, with auto-detection of common install paths across OS.
- **Data prep:** `scripts/clean_dataset.py` — walks the year-folder PDF corpus, extracts and cleans text, parses case name/date from filenames.

---

## 6. Known issues (actively being worked on)

**The consent-gated research mechanic is unreliable.** The design: the assistant chats conversationally, and only searches the case-law database when it either (a) is explicitly asked to, or (b) has offered to research and the user has agreed. In practice, real transcripts show two compounding problems:

1. **A logic bug in the trigger gate** (`should_retrieve` in `model_loader.py`): once the assistant has made an offer, the code checks *only* whether the reply matches a short fixed list of "yes"-type phrases — it doesn't also check whether the reply is itself an explicit, unambiguous research request. A reply like *"please look at how the Supreme Court has ruled in similar cases"* doesn't match the narrow yes/no phrase list and gets silently treated as a decline. This is a straightforward, understood bug (fix identified, not yet applied — see §7).
2. **The model (even the largest/best tier) tends to append a near-identical offer sentence to almost every response**, rather than reserving it for the one moment it's actually decided it understands the situation — and separately, it sometimes **re-asks for information the user already provided** in the same conversation. The leading hypothesis is prompt-related: the system prompt gives one literal example offer sentence, which the model appears to be reusing as a template rather than treating as illustrative, and the prompt has grown dense enough (persona scope + never-prescriptive rule + citation rules + offer mechanic, all at once) that adherence to any single instruction — including "don't repeat yourself" — may be degrading.

**Current direction (agreed, not yet implemented):** replace free-text "yes/no" detection with an explicit two-button choice ("search now" vs. "keep talking first") rendered under the offer, removing the fuzzy-matching problem for the accept/decline step entirely. The gate-logic bug and the prompt-density/repetition issue still need separate fixes regardless of the button UI, since they affect *when* the offer is made in the first place, not just how the reply is interpreted.

**The "Light" model tier (Phi-4-mini) has produced incoherent output** in testing, including one severe repetition loop (since partially addressed by switching to model-agnostic chat templating) and, in a later test, syntactically broken word-salad output. Root cause not fully confirmed — candidates include prompt/instruction density exceeding what a ~3.8B model can reliably hold alongside dense retrieved context, or a poor-quality GGUF conversion. **Leaning toward dropping in-app model tiers entirely** in favor of documenting recommended models (with hardware requirements) in this README and letting users download accordingly — see §7.

**A softer prompt-adherence issue** was also observed independent of the above: even the best-performing tier has, at least once, drifted into generic prescriptive-sounding advice ("you should implement...", "you should establish...") despite explicit prompt instructions against exactly this. Not confirmed as a regression from any specific change — may be normal variance — but flagged as worth re-testing once the other fixes land.

---

## 7. Roadmap / not yet done

**Near-term (next up):**
- Fix the `should_retrieve` gate bug (explicit requests should always work, regardless of pending-offer state)
- Replace free-text accept/decline with explicit buttons
- Add an explicit anti-repetition instruction ("don't re-ask for info already given, don't repeat an offer you just made")
- Remove or loosen the literal example offer sentence in the system prompt to reduce template-copying behavior
- Decide finally on dropping the in-app "Light" tier vs. fixing it; move model guidance into documentation regardless
- Re-test the soft-advice-drift issue once the above land

**Documentation (this pass):**
- Full step-by-step `SETUP.md` (Kaggle dataset download and folder structure, GGUF model download + placement, Tesseract install per OS, optional CUDA Toolkit setup for GPU acceleration)
- A setup-verification script (`check_setup.py` or similar) that checks: is the vault ingested, is a model file present, is Tesseract found, is CUDA available — surfacing clear guidance instead of cryptic errors one at a time
- A recorded demo video/GIF — given the genuinely heavy setup requirements (dataset, model weights, OCR binary, optional CUDA toolkit), a demo is the realistic way most reviewers will experience this project, not a from-scratch local run

**Not started, still valuable:**
- **QLoRA fine-tuning** — originally planned, deprioritized early in favor of RAG/safety work. Still worth doing, and arguably more useful now than originally conceived: it could bake the "descriptive not prescriptive" behavior more durably into model weights, which matters more for smaller/weaker tiers that drift under plain prompting.
- **A real evaluation set** — currently all correctness testing has been manual, interactive, and anecdotal (this conversation's entire debugging history). A hand-labeled set of ~20–30 questions with known-correct source cases, plus a script measuring retrieval precision and citation accuracy, would be the single highest-value addition for turning "seems to work" into a defensible, demonstrable claim.
- **Automated tests** — several real bugs in this project were caught only through manual testing (a ChromaDB directory-vs-file bug, a Windows file-locking issue, two separate regex bugs in citation matching, a `langchain` import break on newer versions, the chat-template/repetition bug). A small `pytest` suite covering the encryption round-trip and citation-matching edge cases would harden the project against regressions going forward.
- **Grammar-constrained output (GBNF)** — a structural alternative/complement to prompting for enforcing the never-prescriptive rule and reducing incoherent output on weaker models, by mechanically constraining what the model's output can contain rather than only asking nicely.

**Original "future roadmap" items, still aspirational, not begun:**
- OCR was implemented (text-extraction only, by design — see original scope discussion; not a vision-capable model)
- Voice input/output
- Peer-to-peer encrypted case-summary sharing
- Integration plugin for existing legal research platforms
- Fine-tuning on High Court / lower court judgments
- Self-hosted/on-prem enterprise deployment option (see §4)

---

## 8. Repository structure (current)

```
nyaya-sahayak/
├── app.py                      # Streamlit entry point — chat UI, streaming, model picker, OCR
├── backend/
│   ├── rag_pipeline.py         # Ingestion + retrieval + re-ranking
│   ├── reranker.py             # Cross-encoder re-ranking
│   ├── encryptor.py            # AES-256 vault (directory-based, crash-safe lifecycle)
│   ├── model_loader.py         # LLM loading, persona/prompt logic, citation verification
│   └── ocr.py                  # Tesseract-based document photo → text
├── scripts/
│   └── clean_dataset.py        # PDF corpus → cleaned parquet
├── models/                     # (gitignored) GGUF model files go here
├── vault/                      # (gitignored) encrypted vector store lives here
├── data/                       # (gitignored) raw + processed judgment data
├── requirements.txt
└── README.md                   # this file
```

---

## 9. Target roles / portfolio framing

ML/AI Engineer (RAG, LLM safety, retrieval systems) · AI/ML roles at legal tech startups · Backend engineer with privacy-first system design focus · Roles valuing demonstrated debugging depth and honest engineering tradeoffs over polish alone.

The most distinctive, non-tutorial pieces of this project, worth calling out explicitly in an interview: the citation-hallucination verification safety net (built in direct response to an observed failure, not preemptively), the directory-based encrypted vault with a correct crash-safe lifecycle (found and fixed through several real bugs, not designed perfectly upfront), and the ongoing, honestly-documented struggle to get a consent-gated conversational flow to behave reliably — which is a genuinely hard, underspecified problem, not a solved one.

---

## 10. License

MIT License — free to use, modify, and distribute.
