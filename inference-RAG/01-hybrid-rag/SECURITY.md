# Security Notes

This cookbook uses local fixtures, but the same risks apply in production Hybrid RAG systems.

## Retrieved Content Is Untrusted

PDFs, OCR text, webpages, tables, and tool outputs can contain malicious or misleading instructions. Treat retrieved content as data, not as system or developer instructions.

## Minimum Production Controls

- Apply authorization filters before retrieval.
- Store source IDs, page numbers, canonical URLs, content hashes, and crawl timestamps.
- Keep raw source text separate from trusted prompts.
- Validate tool arguments before calling APIs.
- Log retrieval traces for audit and incident review.
- Add no-answer behavior when evidence is missing or stale.

## Reporting

If this folder is published as a public repository, add a repository-level security contact or GitHub private vulnerability reporting policy.
