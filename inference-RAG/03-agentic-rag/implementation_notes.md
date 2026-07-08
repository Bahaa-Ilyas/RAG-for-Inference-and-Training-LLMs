# Implementation Notes: Agentic RAG / multi-step retrieval

## Evidence classification
Paper-supported by ReAct and tool-use literature; official-doc-supported by LangGraph and Azure agentic retrieval docs.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Iterative retrieval and tool use selected by a planner, often over hybrid, graph, web, and API retrievers..
5. Add tracing for every retrieval, ranking, generation, and verification step.
6. Compare against a simple hybrid RAG baseline.

## Engineering defaults
- Use stable source IDs and content hashes.
- Store original text plus cleaned text.
- Keep page, URL, section, table, and version metadata on every chunk.
- Batch embeddings and reranker calls.
- Use idempotent ingestion so recrawls and PDF updates are safe.
- Preserve raw evidence for audits.

## Method-specific notes
Plan subgoals; select a retrieval or tool action; inspect evidence; revise the plan; stop when confidence and coverage thresholds are met; synthesize with citations.

## PDF notes
Agentic retrieval can jump from a PDF summary to sections, appendices, tables, and cited references, but every hop must preserve page-grounded citations. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For webpages, the agent must enforce domain allowlists, freshness checks, canonical URLs, and prompt-injection isolation before using crawled text. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
