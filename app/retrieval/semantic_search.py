import json
import faiss
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = Path("generated/faiss.index")
METADATA_PATH = Path("generated/vector_metadata.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class SemanticSearchEngine:
    def __init__(self):
        # Determine device (Optimized for your MacBook M1)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        try:
            print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))

            print("Loading metadata...")
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            print(f"Loading embedding model ({MODEL_NAME}) on {self.device}...")
            self.model = SentenceTransformer(MODEL_NAME, device=self.device)
            
            print("Semantic search engine ready.")
        except Exception as e:
            print(f"Initialization failed: {e}")
            raise

    def search(self, query, top_k=10, threshold=0.3):
        """
        Search for assessments with an optional similarity threshold.
        """
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_embedding, top_k)
        
        # Flatten results
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            # idx -1 means no neighbor found; threshold check for relevance
            if idx == -1 or score < threshold:
                continue

            item = self.metadata[idx]
            results.append({
                "score": round(float(score), 4),
                "name": item["name"],
                "entity_id": item["entity_id"],
                "link": item["metadata"].get("link"),
                "job_levels": item["metadata"].get("job_levels", []),
                "searchable_text": item["searchable_text"]
            })

        return results

if __name__ == "__main__":
    engine = SemanticSearchEngine()
    
    # Example search
    query = "Assessment for backend python developer"
    hits = engine.search(query, top_k=5)

    print(f"\nTop matches for: '{query}'")
    for i, hit in enumerate(hits, 1):
        print(f"{i}. [{hit['score']}] {hit['name']} (ID: {hit['entity_id']})")