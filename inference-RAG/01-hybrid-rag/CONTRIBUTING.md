# Contributing

Keep this folder small, runnable, and pedagogical.

## Standards

- Preserve the end-to-end RAG responsibility framing: document ingestion, chunking strategy, retrieval evaluation, and response quality all belong to the system design.
- Use only the Python standard library unless a dependency is clearly justified.
- Keep examples local and inspectable.
- Do not commit generated files such as `data/local_index.json`.
- Preserve stable document IDs in `data/corpus.jsonl`; evaluation depends on them.
- Run validation before submitting changes:

```bash
python validate_project.py
```

## Good changes

- Add a new source fixture with matching metadata.
- Improve the explanation in a numbered script.
- Add a query to `data/queries.jsonl` and a matching qrel to `data/qrels.jsonl`.
- Improve architecture diagrams without breaking `flowchart LR`.

## Avoid

- Adding hidden API requirements to the default tutorial path.
- Replacing the local demo with a black-box framework call.
- Removing citation, page, URL, version, or source-type metadata.
