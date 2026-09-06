"""
Scans the LLM's generated text to identify which of the retrieved authorities
were actually cited. Maps the cited cases to their source PDF files so the 
UI can provide download/open links.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import List

from backend.retrieval_service import RetrievedAuthority

logger = logging.getLogger("nyaya.verifier")


@dataclass
class VerifiedCitation:
    # Represents a verified citation that the LLM used in its response.
    case_name: str
    judgment_date: str
    source_file: str
    doc_id: str
    chunk_id: str


def verify_citations(
    generated_text: str, 
    authorities: List[RetrievedAuthority]
) -> List[VerifiedCitation]:
    verified: List[VerifiedCitation] = []
    text_lower = generated_text.lower()
    
    for auth in authorities:
        case_name = auth.case_name
        
        # Clean the case name for matching
        # Remove common legal noise words and punctuation
        noise_words = r'\b(vs|v\.?|and|ors|anr|etc|date|of|dead|through|lrs|union|state|intervener)\b'
        clean_name = re.sub(noise_words, '', case_name, flags=re.IGNORECASE).strip()
        clean_name = re.sub(r'[^\w\s]', ' ', clean_name) # remove punctuation
        
        # Get significant words (length > 2)
        words = [w for w in clean_name.split() if len(w) > 2]
        
        if not words:
            continue
            
        # Count how many significant words from the case name appear in the text
        matched_words = sum(1 for w in words if w.lower() in text_lower)
        
        # we consider this case to be cited in the text.
        is_cited = (matched_words >= 2) or (matched_words / len(words) > 0.4)
        
        if is_cited:
            verified.append(
                VerifiedCitation(
                    case_name=auth.case_name,
                    judgment_date=auth.judgment_date,
                    source_file=auth.source_file,
                    doc_id=auth.doc_id,
                    chunk_id=auth.chunk_id
                )
            )
    unique_citations = {}
    for cite in verified:
        if cite.doc_id not in unique_citations:
            unique_citations[cite.doc_id] = cite
            
    final_citations = list(unique_citations.values())
    
    logger.info(f"Verified {len(final_citations)} citations out of {len(authorities)} retrieved authorities.")
    return final_citations