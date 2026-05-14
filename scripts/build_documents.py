import json
from pathlib import Path
from tqdm import tqdm

# Ensure these paths match your local setup
INPUT_FILE = Path("data/shl_product_catalog.json")
OUTPUT_FILE = Path("generated/searchable_documents.json")

def build_searchable_text(entry):
    """
    Convert raw SHL catalog entry into semantic searchable text.
    """
    sections = []

    # Assessment name
    if entry.get("name"):
        sections.append(f"Assessment Name: {entry['name']}")

    # Description
    if entry.get("description"):
        sections.append(f"Description:\n{entry['description']}")

    # Job levels
    if entry.get("job_levels"):
        levels = ", ".join(entry["job_levels"])
        sections.append(f"Applicable Job Levels:\n{levels}")

    # Assessment keys
    if entry.get("keys"):
        keys = ", ".join(entry["keys"])
        sections.append(f"Assessment Type/Categories:\n{keys}")

    # Languages
    if entry.get("languages"):
        langs = ", ".join(entry["languages"])
        sections.append(f"Languages Available:\n{langs}")

    # Duration
    if entry.get("duration"):
        sections.append(f"Duration: {entry['duration']}")

    # Added: Remote Administration
    if entry.get("remote"):
        remote_status = "Available for remote administration" if entry["remote"].lower() == "yes" else "Not remote"
        sections.append(f"Remote Access: {remote_status}")

    # Added: Adaptive Format
    if entry.get("adaptive"):
        adaptive_status = "Adaptive assessment technology" if entry["adaptive"].lower() == "yes" else "Standard format"
        sections.append(f"Format: {adaptive_status}")

    return "\n\n".join(sections)

def main():
    print("Loading SHL catalog...")

    # Using strict=False to handle the control character issues in the source JSON
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        # We read and use loads with strict=False because of hidden control chars
        content = f.read()
        catalog = json.loads(content, strict=False)

    searchable_docs = []

    print(f"Processing {len(catalog)} catalog entries...")

    for idx, entry in enumerate(tqdm(catalog)):
        if not isinstance(entry, dict):
            continue

        searchable_text = build_searchable_text(entry)

        searchable_docs.append({
            "doc_id": idx,
            "entity_id": entry.get("entity_id"),
            "name": entry.get("name"),
            "searchable_text": searchable_text,
            # Enhanced metadata for filtering
            "metadata": {
                "job_levels": entry.get("job_levels", []),
                "keys": entry.get("keys", []),
                "languages": entry.get("languages", []),
                "duration": entry.get("duration"),
                "link": entry.get("link"),
                "remote": entry.get("remote"),
                "adaptive": entry.get("adaptive")
            }
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(searchable_docs, f, indent=2)

    print(f"\nSaved {len(searchable_docs)} searchable documents to:")
    print(OUTPUT_FILE)

if __name__ == "__main__":
    main()
