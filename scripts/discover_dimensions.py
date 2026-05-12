import json
from collections import Counter, defaultdict
from pathlib import Path


RAW_CATALOG_PATH = Path("data/raw/shl_product_catalog.json")
OUTPUT_PATH = Path("data/processed/discovered_dimensions.json")


def load_catalog():
    with open(RAW_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f, strict=False)


def discover_explicit_dimensions(catalog):
    """
    Discover structured fields directly present in catalog.
    """

    field_presence = Counter()

    for item in catalog:
        for key, value in item.items():

            # Ignore empty/null values
            if value is None:
                continue

            if isinstance(value, str) and value.strip() == "":
                continue

            if isinstance(value, list) and len(value) == 0:
                continue

            field_presence[key] += 1

    return field_presence


def collect_field_values(catalog, fields):
    """
    Aggregate values for important structured fields.
    """

    aggregated = defaultdict(set)

    for item in catalog:

        for field in fields:

            if field not in item:
                continue

            value = item[field]

            if isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        aggregated[field].add(v.strip())

            elif isinstance(value, str):
                aggregated[field].add(value.strip())

    # Convert sets -> sorted lists
    return {
        field: sorted(list(values))
        for field, values in aggregated.items()
    }


def main():

    catalog = load_catalog()

    print(f"Loaded {len(catalog)} catalog entries")

    # Step 1 — discover fields
    field_presence = discover_explicit_dimensions(catalog)

    print("\n=== DISCOVERED FIELDS ===\n")

    for field, count in field_presence.most_common():
        print(f"{field}: present in {count} entries")

    # Step 2 — choose useful fields
    useful_fields = [
        "job_levels",
        "languages",
        "keys",
        "remote",
        "adaptive",
        "duration"
    ]

    # Step 3 — aggregate values
    aggregated_values = collect_field_values(
        catalog,
        useful_fields
    )

    output = {
        "discovered_fields": dict(field_presence),
        "aggregated_dimensions": aggregated_values
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved discovered dimensions to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
