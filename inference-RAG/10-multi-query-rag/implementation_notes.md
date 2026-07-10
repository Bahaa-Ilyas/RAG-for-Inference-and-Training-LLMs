# Implementation Notes: Multi-query RAG

## Evidence classification
Framework-doc-supported and engineering best practice; related to query expansion research.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Parallel retrieval over multiple generated query variants..
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
Generate N variants; run retrievers in parallel; union results; deduplicate by source span; fuse ranks; rerank and answer.

## PDF notes
For PDFs, variants can target titles, section labels, definitions, and table headers while retaining page and document constraints. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For webpages, variants should include canonical product or version names and should deduplicate pages by content hash and canonical URL. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
