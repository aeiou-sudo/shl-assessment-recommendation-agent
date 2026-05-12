import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


SEARCH_DOCS_PATH = Path(
    "data/processed/search_documents.json"
)

INDEX_OUTPUT_PATH = Path(
    "data/processed/faiss_index.bin"
)

METADATA_OUTPUT_PATH = Path(
    "data/processed/vector_metadata.pkl"
)


MODEL_NAME = "all-MiniLM-L6-v2"


def load_search_documents():

    with open(SEARCH_DOCS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("\nLoading search documents...")

    documents = load_search_documents()

    print(f"Loaded {len(documents)} documents")

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        doc["search_document"]
        for doc in documents
    ]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    print(f"\nEmbedding dimension: {dimension}")

    print("\nBuilding FAISS index...")

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print(f"Indexed {index.ntotal} vectors")

    INDEX_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_OUTPUT_PATH)
    )

    with open(METADATA_OUTPUT_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("\nSaved vector index and metadata")


if __name__ == "__main__":
    main()
