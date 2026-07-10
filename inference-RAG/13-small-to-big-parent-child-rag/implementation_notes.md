# Implementation Notes: Small-to-big retrieval / parent-child retrieval

## Evidence classification
Framework-doc-supported and widely used engineering pattern.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Dense/sparse retrieval over child chunks with parent expansion for final context..
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
Create parent sections; create child chunks with parent IDs; retrieve children; group by parent; select windows; rerank or compress context.

## PDF notes
This is one of the best PDF defaults: retrieve paragraph/table/caption children and expand to page, section, or heading parents with page numbers. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For webpages, retrieve paragraph or heading blocks and expand to the canonical page section, not the entire site page if it is too long. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
