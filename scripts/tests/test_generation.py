 #!/usr/bin/env python3
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.retrieval_service import HybridRetriever, RetrievalConfig
from backend.context_builder import build_messages
from backend.generation_service import GenerationService, GenerationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main():
    # 1. Retrieve (Using top 3 to keep the test fast)
    retriever = HybridRetriever(RetrievalConfig()).load()
    query = "validity of preventive detention under Article 21 and Article 22"
    print(f"Retrieving for: {query}")
    authorities = retriever.retrieve(query, final_top_k=3) 
    
    # 2. Build Context
    messages = build_messages(authorities, query)
    
    # 3. Generate
    # IMPORTANT: Update the model_path below to your actual Mistral Q4 or Q5 GGUF file path!
    gen_config = GenerationConfig(model_path="models/mistral-7b-instruct-q5.gguf")
    generator = GenerationService(gen_config).load()
    
    print("\n" + "="*80)
    print("STREAMING LLM RESPONSE:")
    print("="*80 + "\n")
    
    # Stream token by token to the console
    for token in generator.generate_stream(messages):
        print(token, end="", flush=True)
        
    print("\n\n" + "="*80)

if __name__ == "__main__":
    main()