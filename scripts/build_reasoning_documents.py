import json
import re

from pathlib import Path
from tqdm import tqdm


INPUT_FILE = Path("data/shl_product_catalog.json")

OUTPUT_FILE = Path(
    "generated/reasoning_documents.json"
)


# -----------------------------------------
# Stopwords
# -----------------------------------------

STOPWORDS = {
    "assessment",
    "test",
    "new",
    "general",
    "using",
    "knowledge",
    "skills",
    "measures",
    "measure",
    "ability",
    "abilities",
    "candidate",
    "candidates",
    "job",
    "role",
    "roles",
    "professional",
    "individual",
    "contributor"
}

# Add these to your STOPWORDS to clean it up further
EXTRA_STOPWORDS = {"designed", "provides", "including", "available", "assessment", "participants"}
STOPWORDS.update(EXTRA_STOPWORDS)

def tokenize(text):
    """
    Extract semantic terms from text.
    """

    if not text:
        return []

    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9\.\#\+\-]+\b",
        text
    )

    tokens = [
        token
        for token in tokens
        if (
            len(token) > 2
            and token not in STOPWORDS
        )
    ]

    return list(set(tokens))


def build_reasoning_document(entry):

    name = entry.get("name", "")

    description = entry.get("description", "")

    keys = entry.get("keys", [])

    job_levels = entry.get("job_levels", [])

    # -----------------------------------------
    # Semantic term extraction
    # -----------------------------------------

    semantic_source = " ".join([
        name,
        description,
        " ".join(keys)
    ])

    semantic_terms = tokenize(semantic_source)

    reasoning_doc = {

        "entity_id": entry.get("entity_id"),

        "name": name,

        "description": description,

        "job_levels": job_levels,

        "categories": keys,

        "semantic_terms": semantic_terms
    }

    return reasoning_doc


def main():
    print("Loading SHL catalog...")

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # Read the file content as a string first
        content = f.read()
        # Use loads with strict=False to ignore invalid control characters
        catalog = json.loads(content, strict=False)

    reasoning_docs = []

    print(f"Processing {len(catalog)} entries...")

    for entry in tqdm(catalog):

        reasoning_doc = build_reasoning_document(entry)

        reasoning_docs.append(reasoning_doc)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(reasoning_docs, f, indent=2)

    print("\nDONE.")
    print(f"Saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
