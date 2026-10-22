import logging
import json
import time
import sys
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.retrieval_service import HybridRetriever, RetrievalConfig
from backend.context_builder import build_messages, ContextConfig
from backend.generation_service import GenerationService, GenerationConfig
from backend.citation_verifier import verify_citations

# =====================================================================
# 1. STRUCTURED JSON LOGGING (MLOps Observability)
# =====================================================================
class JSONFormatter(logging.Formatter):
    """Formats logs as JSON for ingestion by Datadog/CloudWatch/ELK."""
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach MLOps metrics and guardrail events if they exist
        for key in ["guardrail_event", "metrics", "query_id", "latency_ms"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        return json.dumps(log_data)

logger = logging.getLogger("nyaya.api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# =====================================================================
# 2. FASTAPI APP & PYDANTIC MODELS
# =====================================================================
app = FastAPI(
    title="NyayaSahayak API", 
    version="1.0.0", 
    description="Privacy-preserving Legal RAG API with Guardrail Observability"
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 6

class CitationResponse(BaseModel):
    case_name: str
    judgment_date: str
    source_file: str

class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: List[CitationResponse]
    guardrails_triggered: List[str]
    latency_ms: float
    metrics: dict

# =====================================================================
# 3. GLOBAL STATE (Load models once on startup)
# =====================================================================
retriever = None
generator = None

@app.on_event("startup")
async def load_models():
    global retriever, generator
    logger.info("Loading ML models for API...")
    # Device auto-detection is handled inside the configs now
    retriever = HybridRetriever(RetrievalConfig()).load()
    generator = GenerationService(GenerationConfig()).load()
    logger.info("Models loaded successfully. API ready.")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": retriever is not None and generator is not None
    }

# =====================================================================
# 4. QUERY ENDPOINT WITH GUARDRAIL OBSERVABILITY
# =====================================================================
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()
    query_id = str(uuid.uuid4())[:8]
    guardrails_triggered = []
    metrics = {}
    
    # --- A. RETRIEVAL & CONFIDENCE GATE ---
    authorities = retriever.retrieve(query=request.query, final_top_k=request.top_k)
    
    if not authorities:
        guardrails_triggered.append("empty_retrieval")
        logger.info("Guardrail triggered: Empty Retrieval", extra={
            "guardrail_event": "empty_retrieval", 
            "query_id": query_id
        })
        return QueryResponse(
            query_id=query_id,
            answer="I could not find any relevant Supreme Court authorities for this query in the current corpus.",
            citations=[],
            guardrails_triggered=guardrails_triggered,
            latency_ms=(time.time() - start_time) * 1000,
            metrics={"retrieved_chunks": 0}
        )
        
    # Confidence Gate: ms-marco-MiniLM returns logits. < -5.0 usually means irrelevant.
    top_score = authorities[0].rerank_score if authorities[0].rerank_score is not None else -999.0
    metrics["top_rerank_score"] = round(top_score, 4)
    metrics["retrieved_chunks"] = len(authorities)
    
    if top_score < -5.0: 
        guardrails_triggered.append("low_confidence_retrieval")
        logger.info("Guardrail triggered: Low Confidence", extra={
            "guardrail_event": "low_confidence_retrieval", 
            "query_id": query_id,
            "metrics": metrics
        })

    # --- B. GENERATION ---
    messages = build_messages(authorities, request.query)
    full_response = "".join([token for token in generator.generate_stream(messages)])
        
    # --- C. CITATION VERIFICATION ---
    verified = verify_citations(full_response, authorities)
    metrics["verified_citations"] = len(verified)
    
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    # Log final success metrics
    logger.info("Query processed successfully", extra={
        "query_id": query_id,
        "latency_ms": latency_ms,
        "metrics": metrics
    })
    
    return QueryResponse(
        query_id=query_id,
        answer=full_response,
        citations=[
            CitationResponse(
                case_name=c.case_name,
                judgment_date=c.judgment_date,
                source_file=c.source_file
            ) for c in verified
        ],
        guardrails_triggered=guardrails_triggered,
        latency_ms=latency_ms,
        metrics=metrics
    )