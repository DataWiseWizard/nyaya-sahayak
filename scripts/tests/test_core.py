"""
Core unit tests for NyayaSahayak backend.
Run with: pytest scripts/tests/test_core.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestContextBuilder:
    # Tests for the context builder module.

    def test_context_respects_max_authorities(self):
        # Context should not include more than max_authorities chunks.
        from backend.context_builder import build_context, ContextConfig
        from backend.retrieval_service import RetrievedAuthority

        # Create 10 fake authorities
        authorities = []
        for i in range(10):
            authorities.append(RetrievedAuthority(
                chunk_id=f"test_chunk_{i}",
                doc_id=f"test_doc_{i}",
                case_name=f"Test Case {i} vs State",
                judgment_date="2020-01-01",
                source_file=f"test_{i}.PDF",
                chunk_index=i,
                text=f"This is test passage {i} for context builder testing.",
                retrieval_sources=["dense"],
            ))

        config = ContextConfig(max_authorities=5)
        context = build_context(authorities, "test query", config)

        # Should only contain 5 authorities
        assert context.count("AUTHORITY") == 5

    def test_context_includes_disclaimer_instruction(self):
        # System prompt must include legal disclaimer.
        from backend.context_builder import SYSTEM_PROMPT

        assert "NOT a lawyer" in SYSTEM_PROMPT
        assert "NOT provide legal advice" in SYSTEM_PROMPT

    def test_context_empty_authorities(self):
        #Empty authority list should produce a graceful message.
        from backend.context_builder import build_context

        context = build_context([], "test query")
        assert "No relevant" in context or "no" in context.lower()


class TestCitationVerifier:
    # Tests for the citation verifier module.

    def test_verifier_detects_cited_case(self):
        # Verifier should identify cases mentioned in generated text.
        from backend.citation_verifier import verify_citations
        from backend.retrieval_service import RetrievedAuthority

        authorities = [
            RetrievedAuthority(
                chunk_id="chunk_1",
                doc_id="doc_1",
                case_name="A.K. Gopalan vs The State of Madras",
                judgment_date="1950-05-19",
                source_file="A_K_Gopalan.PDF",
                chunk_index=0,
                text="test",
                retrieval_sources=["dense"],
            )
        ]

        generated_text = "In the case of A.K. Gopalan vs The State of Madras (1950), the Supreme Court held..."

        verified = verify_citations(generated_text, authorities)
        assert len(verified) >= 1

    def test_verifier_flags_hallucinated_case(self):
        # Verifier should not mark a case as cited if it is not in the text.
        from backend.citation_verifier import verify_citations
        from backend.retrieval_service import RetrievedAuthority

        authorities = [
            RetrievedAuthority(
                chunk_id="chunk_1",
                doc_id="doc_1",
                case_name="A.K. Gopalan vs The State of Madras",
                judgment_date="1950-05-19",
                source_file="A_K_Gopalan.PDF",
                chunk_index=0,
                text="test",
                retrieval_sources=["dense"],
            )
        ]

        generated_text = "In the case of Kesavananda Bharati vs State of Kerala, the Supreme Court held..."

        verified = verify_citations(generated_text, authorities)
        assert len(verified) == 0


class TestRetrievalService:
    # Tests for the retrieval service.

    def test_retrieval_returns_results(self):
        # Retrieval should return at least one result for a valid legal query.
        from backend.retrieval_service import HybridRetriever, RetrievalConfig

        config = RetrievalConfig()
        retriever = HybridRetriever(config).load()

        results = retriever.retrieve(
            query="Article 21 right to life personal liberty",
            final_top_k=5,
        )

        assert len(results) >= 1

    def test_retrieval_metadata_contains_required_fields(self):
        # Every result must have case_name, judgment_date, source_file
        from backend.retrieval_service import HybridRetriever, RetrievalConfig

        config = RetrievalConfig()
        retriever = HybridRetriever(config).load()

        results = retriever.retrieve(
            query="preventive detention",
            final_top_k=3,
        )

        for result in results:
            assert result.case_name != "Unknown", f"Missing case_name: {result.chunk_id}"
            assert result.judgment_date != "Unknown", f"Missing judgment_date: {result.chunk_id}"
            assert result.source_file != "Unknown", f"Missing source_file: {result.chunk_id}"