import json
from pathlib import Path


RAW_CATALOG_PATH = Path(
    "data/raw/shl_product_catalog.json"
)

SEMANTIC_PATH = Path(
    "data/processed/semantic_dimensions.json"
)

OUTPUT_PATH = Path(
    "data/processed/search_documents.json"
)


def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, strict=False)


def build_search_document(item, semantics):

    fields = []

    # Core fields
    fields.append(item.get("name", ""))
    fields.append(item.get("description", ""))

    # Structured metadata
    fields.extend(item.get("job_levels", []))
    fields.extend(item.get("languages", []))
    fields.extend(item.get("keys", []))

    # Semantic dimensions
    for category, values in semantics.items():

        if isinstance(values, list):
            fields.extend(values)

    # Remove empty values
    fields = [
        str(f).strip()
        for f in fields
        if str(f).strip()
    ]

    return " ".join(fields)


def main():

    catalog = load_json(RAW_CATALOG_PATH)

    semantic_data = load_json(SEMANTIC_PATH)

    semantic_lookup = {
        item["name"]: item["semantics"]
        for item in semantic_data["assessment_semantics"]
    }

    processed_documents = []

    for item in catalog:

        name = item.get("name")

        semantics = semantic_lookup.get(name, {})

        search_document = build_search_document(
            item,
            semantics
        )

        processed_documents.append({
            "name": name,
            "entity_id": item.get("entity_id"),
            "url": item.get("link"),
            "search_document": search_document,
            "metadata": {
                "job_levels": item.get("job_levels", []),
                "languages": item.get("languages", []),
                "duration": item.get("duration"),
                "remote": item.get("remote"),
                "adaptive": item.get("adaptive"),
                "keys": item.get("keys", [])
            },
            "semantic_dimensions": semantics
        })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_documents, f, indent=2)

    print(f"\nSaved search documents to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
