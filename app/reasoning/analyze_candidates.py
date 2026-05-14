import re
import json
from collections import Counter
from pathlib import Path
import sys

# Import your previously created engine
# Ensure the path corresponds to your local project structure
root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root))

from app.retrieval.semantic_search import SemanticSearchEngine

# Expanded stopwords list to refine technical term extraction
STOPWORDS = {
    "assessment", "test", "new", "general", "using", "developer", 
    "developers", "programming", "knowledge", "skills", "level", 
    "management", "professional", "individual", "contributor",
    "designed", "covers", "measures", "information", "completed"
}

def tokenize(text):
    """
    Tokenizer designed to preserve technical identifiers like C# and .NET
    while filtering out non-descriptive common words.
    """
    if not text:
        return []
    
    text = text.lower()
    # Captures alphanumeric strings and common tech symbols (. and #)
    words = re.findall(r"\b[a-zA-Z0-9\.\#]+\b", text)
    
    return [
        w for w in words 
        if len(w) > 1 and w not in STOPWORDS
    ]

def analyze_candidates(candidates):
    """
    Performs a deep aggregation of search results to identify trends 
    in job levels, categories, and technical terminology.
    """
    analysis = {
        "job_levels": Counter(),
        "categories": Counter(),
        "assessment_names": [],
        "term_frequencies": Counter(),
        "unique_terms_count": 0
    }

    for candidate in candidates:
        # 1. Track Names
        analysis["assessment_names"].append(candidate["name"])

        # 2. Aggregate Job Levels
        # Using .get() with a default empty list to prevent crashes
        for level in candidate.get("job_levels", []):
            analysis["job_levels"][level] += 1

        # 3. Aggregate Categories/Keys from metadata
        metadata = candidate.get("metadata", {})
        for key in metadata.get("keys", []):
            analysis["categories"][key] += 1

        # 4. Analyze Technical Terms
        searchable_text = candidate.get("searchable_text", "")
        tokens = tokenize(searchable_text)
        analysis["term_frequencies"].update(tokens)

    analysis["unique_terms_count"] = len(analysis["term_frequencies"])
    return analysis

def print_analysis(analysis, query):
    """
    Prints a structured report of the candidate analysis.
    """
    divider = "=" * 80
    print(f"\n{divider}")
    print(f"CANDIDATE DIFFERENCE ANALYSIS | QUERY: '{query}'")
    print(divider)

    print("\n[1] RETRIEVED ASSESSMENTS")
    for i, name in enumerate(analysis["assessment_names"], 1):
        print(f" {i}. {name}")

    print("\n[2] JOB LEVEL DISTRIBUTION")
    if not analysis["job_levels"]:
        print(" No job level data found.")
    for level, count in analysis["job_levels"].most_common():
        print(f" • {level:35} | {count} hits")

    print("\n[3] ASSESSMENT CATEGORIES")
    if not analysis["categories"]:
        print(" No category data found.")
    for category, count in analysis["categories"].most_common():
        print(f" • {category:35} | {count} hits")

    print("\n[4] MOST FREQUENT TECHNICAL TERMS")
    # Display the top 20 terms to get a clear technical profile
    for term, count in analysis["term_frequencies"].most_common(20):
        print(f" • {term:35} | {count} occurrences")

    print(f"\nSummary: {analysis['unique_terms_count']} unique meaningful terms across results.")
    print(f"{divider}\n")

if __name__ == "__main__":
    # 1. Initialize the Search Engine
    # Note: Ensure your faiss.index and vector_metadata.json are in 'generated/'
    try:
        engine = SemanticSearchEngine()

        # 2. Define your search parameters
        search_query = "Assessment for backend python developer"
        top_k_count = 10

        # 3. Perform the search
        # If your search function uses a threshold, you can pass it here
        candidates = engine.search(search_query, top_k=top_k_count)

        if not candidates:
            print("No matching assessments found for this query.")
        else:
            # 4. Analyze the results
            results_analysis = analyze_candidates(candidates)

            # 5. Output the final report
            print_analysis(results_analysis, search_query)
            
    except FileNotFoundError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
