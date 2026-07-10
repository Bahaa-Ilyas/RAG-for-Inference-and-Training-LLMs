# Implementation Notes: GraphRAG

## Evidence classification
Paper-supported and official-repo-supported by Microsoft GraphRAG.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Graph traversal plus chunk/vector retrieval; local search for entities and global search over community summaries..
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
Extract entities and relations; build graph; cluster communities; summarize communities; retrieve relevant entities and summaries; ground answer in source spans.

## PDF notes
GraphRAG can connect entities across long PDFs, references, captions, and tables, but extraction must keep source spans and page numbers for auditability. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For websites, graph nodes should include canonical URL, crawl timestamp, publisher, and page version to avoid merging stale or duplicated entities blindly. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
