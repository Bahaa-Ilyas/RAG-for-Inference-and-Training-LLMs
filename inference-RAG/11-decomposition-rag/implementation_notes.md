# Implementation Notes: Decomposition RAG / sub-question RAG

## Evidence classification
Engineering best practice and framework-doc-supported; related to agentic retrieval and multi-hop QA literature.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Multiple focused retrieval calls planned from decomposed sub-questions..
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
Generate sub-questions; retrieve evidence per sub-question; answer each with citations; synthesize; verify each final claim against sub-evidence.

## PDF notes
For PDFs, decomposition can search separate sections, appendices, and tables, then align sub-answers with page citations. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For webpages, decomposition should keep domain, version, and freshness constraints attached to each sub-query. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
