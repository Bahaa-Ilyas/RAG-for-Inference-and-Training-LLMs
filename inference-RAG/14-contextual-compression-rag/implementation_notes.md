# Implementation Notes: Contextual compression RAG

## Evidence classification
Framework-doc-supported and engineering best practice.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Post-retrieval compression after sparse, dense, hybrid, or hierarchical retrieval..
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
Retrieve; split into sentences/spans; score spans against query; keep relevant spans under budget; preserve source offsets; generate with citations.

## PDF notes
Compression should preserve page, table headers, captions, equation labels, and adjacent definitions so snippets remain auditable. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For webpages, compression removes nav/footer boilerplate and repeated templates but must retain canonical URL and crawl timestamp. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
