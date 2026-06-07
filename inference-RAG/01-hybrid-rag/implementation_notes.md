# Implementation Notes: Hybrid RAG: BM25 + dense vector retrieval

## Ownership scope
The implementation owner is responsible for the complete RAG lifecycle: ingestion, parsing, chunking, metadata, sparse and dense indexing, retrieval evaluation, context assembly, response quality, citation behavior, and operational monitoring. Do not treat the retriever score as the final product; the final product is a grounded response that can be traced back to source evidence.

## Evidence classification
Paper-supported for dense retrieval and RAG; engineering best practice and official-doc-supported for hybrid search in production search engines.

## Practical build order
1. Define the metadata schema before ingestion.
2. Create a 50 to 200 question evaluation set with expected sources.
3. Ingest a small corpus slice and inspect extracted text, tables, images, and webpage metadata.
4. Implement Sparse BM25 plus dense ANN with reciprocal rank fusion or weighted score fusion..
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
Index each chunk into sparse and dense stores; retrieve top candidates from both; normalize or rank-fuse; deduplicate by document and span; optionally rerank; build context.

## PDF notes
Hybrid retrieval is strong for PDFs because BM25 preserves exact page labels, exhibit names, equation identifiers, and table headers while dense retrieval handles paraphrased concepts. Use page-aware citations and validate scanned/OCR documents before trusting answers.

## Web notes
On webpages, hybrid retrieval should store canonical URL, title, headings, crawl timestamp, hash, and anchor text so exact terms and semantic content both survive HTML cleaning. Store crawl timestamps and treat webpage text as untrusted input.



## Minimal implementation pseudocode
```python
records = ingest_sources(paths_or_urls)
records = attach_metadata(records)
index = build_indexes(records)
results = retrieve_with_method(index, query)
evidence = prepare_context(results)
answer = generate_and_verify(query, evidence)
```
