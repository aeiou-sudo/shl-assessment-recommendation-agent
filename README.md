# SHL Assessment Recommendation Agent

A state-driven conversational retrieval convergence agent for helping recruiters prepare SHL assessment plans from a semi-structured product catalogue.

This is not a generic chatbot. The runtime preserves a staged pipeline:

1. Query extraction and out-of-context rejection
2. Structured conversation state update
3. Deterministic positive-intent retrieval with FAISS
4. Conservative LLM filtering using negative constraints
5. Query-strength analysis with score spread and entropy
6. Query strengthening before ambiguity clarification, rendered as recruiter-facing consultation
7. Focused ambiguity clarification
8. Catalogue-grounded recommendation and assessment plan generation

The public interface does not expose orchestration states, confidence spread,
entropy, retrieval ranks, or filtering diagnostics. Those remain backend-only.

## Architecture

- `app/rag/catalogue_processor.py` transforms catalogue entries into semantic documents while preserving references to source entries.
- `app/rag/faiss_store.py` builds deterministic embedding search over those documents and aggregates chunk hits back to catalogue entries.
- `app/state/conversation_state.py` defines primary intents, linked intents, domains, skills, specializations, rejected intents, ambiguity history, clarification answers, confidence history, and negative constraints.
- `app/state/llm_state_interpreter.py` performs constrained query classification: `OUT_OF_CONTEXT`, `NEW_QUERY`, `REFINEMENT_QUERY`, `INTENT_SHIFT`, or `CLARIFICATION_RESPONSE`.
- `app/retrieval/query_synthesizer.py` creates the FAISS query from positive state only. Negative constraints are intentionally excluded from vector search.
- `app/reasoning/intent_filters.py` conservatively removes noisy candidates, domain drift, and negative-constraint violations after retrieval.
- `app/reasoning/query_strength_engine.py` and `app/reasoning/query_gap_analyzer.py` detect weak queries and ask strengthening questions from shared top-k concepts.
- `app/reasoning/analyze_candidates_enriched.py` compares high-ranking candidates and asks focused ambiguity questions.
- `app/reasoning/convergence_engine.py` orchestrates the staged loop without collapsing it into one prompt.
- `app/services/conversation_presenter.py` converts backend convergence results into natural recruiter-facing guidance and hides retrieval internals.
- `app/services/assessment_agent.py` generates final assessment plans only from matched catalogue entries and preserved recruiter clarifications.

## Conversation Contract

Every `/chat` call advances the same loop: state update, deterministic retrieval,
filtering, convergence evaluation, and either clarification or stable
recommendation. The recruiter-facing response shape is intentionally small:

```json
{
  "message": "I'm keeping the .NET backend focus and excluding MVC in view. Would success in this role depend more on hands-on implementation, systems design, or operational ownership?",
  "recommendations": null,
  "assessment_plan": null,
  "end_of_conversation": false
}
```

Recommendations only appear after convergence is stable enough for the assistant
to be useful rather than noisy.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build a persisted FAISS index:

```bash
python scripts/build_faiss_index.py
```

Run a smoke turn:

```bash
python scripts/smoke_turn.py "I need the .NET Framework 4.5 assessment"
```

Start the API:

```bash
uvicorn app.api.main:app --reload
```

Then call:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","query":"I need an assessment for a .NET backend developer, not MVC"}'
```

## LLM Use

LLM calls are constrained to classification, filtering, query-strength clarification, and ambiguity clarification. Retrieval remains deterministic and embedding-based. If `GROQ_API_KEY` is unset, deterministic fallbacks keep the local pipeline runnable.
