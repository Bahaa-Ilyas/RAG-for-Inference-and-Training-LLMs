# Implementation Notes: RAPTOR / hierarchical retrieval

## Evidence classification
Paper-supported and official-repo-supported by RAPTOR.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Hierarchical vector retrieval over leaf chunks and recursively generated summary nodes..
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
Chunk documents; embed chunks; cluster; summarize clusters; repeat to root; retrieve from multiple levels; expand selected summaries to source chunks.

## PDF notes
RAPTOR is useful for long PDFs if the hierarchy respects document sections, page spans, figure captions, and appendices instead of clustering arbitrary fragments only. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For websites, build per-page and per-site hierarchies with crawl timestamps; rebuild summaries when page hashes change. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
