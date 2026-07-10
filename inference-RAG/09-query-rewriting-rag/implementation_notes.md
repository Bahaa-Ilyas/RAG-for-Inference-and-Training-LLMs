# Implementation Notes: Query rewriting / query expansion RAG

## Evidence classification
Paper-supported by HyDE and retrieval literature; framework-doc-supported by LlamaIndex and LangChain.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement First-stage retrieval over rewritten or expanded queries..
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
Classify query; generate retrieval rewrites; validate preserved constraints; retrieve for each rewrite; fuse and rerank.

## PDF notes
For PDFs, rewrites should preserve exact document IDs, page references, section names, and quoted phrases while expanding concepts around them. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
For websites, rewrites can add site-specific terms, product names, release versions, or API namespace terms but must avoid broad web drift. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
