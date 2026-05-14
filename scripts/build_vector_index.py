import json
import faiss
import numpy as np
import torch  # Added for device check
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("generated/searchable_documents.json")
FAISS_OUTPUT = Path("generated/faiss.index")
EMBEDDINGS_OUTPUT = Path("generated/embeddings.npy")
METADATA_OUTPUT = Path("generated/vector_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    # 1. Device Selection (Performance recommendation)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Loading searchable documents...")
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found. Run the extraction script first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # 2. Validation: Filter out items with no searchable text
    valid_docs = [doc for doc in documents if doc.get("searchable_text")]
    texts = [doc["searchable_text"] for doc in valid_docs]

    print(f"Loaded {len(texts)} valid documents (out of {len(documents)})")

    # 3. Load embedding model
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 4. Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for IndexFlatIP (Cosine Similarity)
    )

    # 5. Save raw embeddings
    np.save(EMBEDDINGS_OUTPUT, embeddings)

    # 6. Build FAISS index
    dimension = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dimension}")
    
    # We use IndexFlatIP because normalized vectors + Inner Product = Cosine Similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_OUTPUT))

    # 7. Save metadata mapping
    metadata = []
    for idx, doc in enumerate(valid_docs):
        metadata.append({
            "faiss_id": idx,
            "doc_id": doc["doc_id"],
            "entity_id": doc["entity_id"],
            "name": doc["name"],
            "metadata": doc["metadata"],
            "searchable_text": doc["searchable_text"]
        })

    with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDONE. Saved index and metadata to {FAISS_OUTPUT.parent}")

    # 8. Recommendation: Test Search
    test_query = "Assessments for software developers using .NET"
    print(f"\n--- Running Test Search: '{test_query}' ---")
    query_vector = model.encode([test_query], normalize_embeddings=True)
    distances, indices = index.search(query_vector, k=3)
    
    for i, idx in enumerate(indices[0]):
        score = distances[0][i]
        match = metadata[idx]
        print(f"Rank {i+1} [Score: {score:.4f}]: {match['name']} (ID: {match['entity_id']})")

if __name__ == "__main__":
    main()
