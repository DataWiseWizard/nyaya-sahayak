"""
Loads the local Mistral LLM and streams the generated legal research answer
token-by-token based on the context built by context_builder.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Dict, Optional

from llama_cpp import Llama

logger = logging.getLogger("nyaya.generation")


class GenerationConfig:
    def __init__(
        self,
        model_path: str = "models/mistral-7b-instruct-q5.gguf",
        n_gpu_layers: int = -1,  # -1 offloads all layers to GPU (RTX 4060)
        n_ctx: int = 8192,       # Context window
        max_tokens: int = 1024,  # Max generation tokens
        temperature: float = 0.1,# Low temperature for factual legal answers
        top_p: float = 0.9,
    ):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p


class GenerationService:
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.llm: Optional[Llama] = None

    def load(self) -> "GenerationService":
        if self.llm is not None:
            return self

        model_path = Path(self.config.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"LLM model not found at: {model_path}")

        logger.info("Loading LLM from %s...", model_path)
        
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            chat_format="mistral-instruct",  # Ensures correct [INST] formatting for Mistral
            verbose=False,
        )
        
        logger.info("LLM loaded successfully.")
        return self

    def generate_stream(
        self, 
        messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        # Streams the LLM response token-by-token.
        if self.llm is None:
            self.load()

        logger.info("Starting generation stream...")
        
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                token = delta["content"]
                if token:
                    yield token
                    
        logger.info("Generation stream finished.")