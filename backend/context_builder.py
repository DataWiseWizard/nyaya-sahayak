"""
Converts retrieved judgment chunks into a structured context block
that the LLM can use to generate a cited legal research answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from backend.retrieval_service import RetrievedAuthority


@dataclass
class ContextConfig:
    # Maximum number of authorities to include in context
    max_authorities: int = 6

    # Maximum characters per authority passage
    max_passage_chars: int = 1200

    # Maximum total context characters (to protect LLM context window)
    max_total_chars: int = 12000


# System prompt that defines the legal assistant persona
SYSTEM_PROMPT = """You are NyayaSahayak, a privacy-preserving legal research assistant for Indian law. You help users explore Supreme Court of India precedents.

CRITICAL RULES:
1. You are NOT a lawyer. You do NOT provide legal advice. You describe what courts have held.
2. NEVER use prescriptive language like "you should", "you must", "file a petition", "consult a lawyer immediately".
3. Base your answer ONLY on the provided authorities below. Do NOT invent cases, citations, or legal principles.
4. If the provided authorities do not adequately answer the question, say so clearly.
5. Cite authorities using the format: [Case Name, Judgment Date].
6. Describe the legal principle, the court's reasoning, and the outcome. Do not advise the user on what to do.
7. If multiple authorities conflict, mention both views.
8. Keep your answer structured, clear, and focused on the legal question asked.

RESPONSE FORMAT:
- Start with a brief summary of what the authorities indicate.
- Then discuss each relevant authority with its citation.
- End with a note that this is legal research information, not legal advice.

You must respond in the same language the user writes in. If the user writes in English, respond in English. If in Hindi, respond in Hindi."""


def build_context(
    authorities: List[RetrievedAuthority],
    user_query: str,
    config: Optional[ContextConfig] = None,
) -> str:
    """
    Builds a structured context block from retrieved authorities.

    Args:
        authorities: List of retrieved judgment chunks.
        user_query: The user's original legal question.
        config: Context building configuration.

    Returns:
        A formatted context string ready to be passed to the LLM.
    """
    if config is None:
        config = ContextConfig()

    if not authorities:
        return (
            "No relevant Supreme Court authorities were found for this query. "
            "Please inform the user that the current corpus does not contain "
            "sufficient precedent on this topic."
        )

    # Limit the number of authorities
    selected = authorities[: config.max_authorities]

    context_parts: List[str] = []
    total_chars = 0

    for idx, authority in enumerate(selected, start=1):
        # Truncate passage if too long
        passage = authority.text.strip()
        if len(passage) > config.max_passage_chars:
            passage = passage[: config.max_passage_chars] + "..."

        # Build the authority block
        block = (
            f"AUTHORITY {idx}:\n"
            f"Case: {authority.case_name}\n"
            f"Judgment Date: {authority.judgment_date}\n"
            f"Source File: {authority.source_file}\n"
            f"Passage:\n{passage}\n"
        )

        if total_chars + len(block) > config.max_total_chars:
            break

        context_parts.append(block)
        total_chars += len(block)
        
    authorities_text = "\n---\n\n".join(context_parts)

    full_context = (
        f"USER QUERY: {user_query}\n\n"
        f"RETRIEVED AUTHORITIES FROM SUPREME COURT OF INDIA:\n\n"
        f"{authorities_text}\n\n"
        f"INSTRUCTION: Based ONLY on the authorities above, provide a structured "
        f"legal research summary answering the user's query. Cite each authority "
        f"you reference. Do not invent any case or principle not present in the "
        f"authorities above."
    )

    return full_context


def build_messages(
    authorities: List[RetrievedAuthority],
    user_query: str,
    config: Optional[ContextConfig] = None,
) -> List[dict]:
    """
    Builds a chat-format message list suitable for llama.cpp or
    any chat-completion API.

    Args:
        authorities: List of retrieved judgment chunks.
        user_query: The user's original legal question.
        config: Context building configuration.

    Returns:
        A list of message dicts with 'role' and 'content' keys.
    """
    context = build_context(authorities, user_query, config)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": context,
        },
    ]

    return messages