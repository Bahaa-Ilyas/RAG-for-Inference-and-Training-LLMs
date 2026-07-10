"""Add left-to-right Mermaid architecture files to every inference method."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent

METHOD_COMPONENTS = {
    "01-hybrid-rag": ("Hybrid Retrieval", "BM25 + dense vector search + rank fusion"),
    "02-reranked-rag": ("Reranker", "Cross-encoder or LLM reranking over broad candidates"),
    "03-agentic-rag": ("Agent Loop", "Planner chooses retrieval, tools, and follow-up searches"),
    "04-managed-enterprise-rag": ("Managed Search Service", "Cloud index, ACL trim, semantic/vector search"),
    "05-graphrag": ("Graph Search", "Entity extraction, graph traversal, and community context"),
    "06-raptor-hierarchical-rag": ("Hierarchy Search", "Leaf chunks, clustered summaries, and tree expansion"),
    "07-self-rag": ("Adaptive Controller", "Model decides whether to retrieve and critique support"),
    "08-corrective-rag-crag": ("Corrective Branch", "Grade retrieval quality and repair weak evidence"),
    "09-query-rewriting-rag": ("Query Rewriter", "Generate retrieval-optimized query variants"),
    "10-multi-query-rag": ("Multi-Query Fanout", "Run diverse query variants and fuse results"),
    "11-decomposition-rag": ("Subquestion Planner", "Break complex tasks into answerable retrieval steps"),
    "12-metadata-filtered-rag": ("Metadata Filter", "Apply version, source, ACL, and freshness constraints"),
    "13-small-to-big-parent-child-rag": ("Parent Expansion", "Retrieve child chunks and expand to parent context"),
    "14-contextual-compression-rag": ("Context Compressor", "Extract only query-relevant spans under token budget"),
    "15-table-aware-rag": ("Table Engine", "Preserve rows, columns, units, and executable operations"),
    "16-layout-aware-multimodal-pdf-rag": ("Layout + Multimodal Parser", "Use OCR, page layout, figures, captions, and tables"),
    "17-webpage-rag-freshness": ("Freshness Controller", "Crawl, canonicalize, hash, deduplicate, and recrawl"),
    "18-version-aware-rag": ("Version Resolver", "Select current or requested document version"),
    "19-tool-augmented-rag": ("Tool/API Call", "Combine retrieved policy with live structured tool results"),
    "20-claim-level-verification-rag": ("Claim Verifier", "Check each generated claim against evidence"),
}


def title_from_readme(folder: Path) -> str:
    readme = folder / "README.md"
    if not readme.exists():
        return folder.name
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            if title.startswith("Building ") and title.endswith(" From Scratch"):
                return title.removeprefix("Building ").removesuffix(" From Scratch")
            return title
    return folder.name


def mermaid_for(folder: Path) -> str:
    method_title = title_from_readme(folder)
    component, component_note = METHOD_COMPONENTS.get(folder.name, ("Method Logic", "Apply method-specific retrieval behavior"))
    return f"""%% Architecture for {method_title}
%% Direction is left-to-right for later conversion into rendered or 3D visuals.
flowchart LR
    A["User query"] --> B["Query understanding"]
    S1["PDF fixture<br/>examples/sample_policy.pdf"] --> C
    S2["Scanned OCR fixture<br/>examples/scanned_page_ocr.txt"] --> C
    S3["Webpage fixture<br/>examples/sample_webpage.html"] --> C
    S4["Table fixture<br/>examples/sample_table.csv"] --> C
    S5["Tool/API fixture<br/>examples/tool_response.json"] --> C
    B --> C["Mixed-source corpus<br/>PDFs, scanned OCR, webpages, tables, tools"]
    C --> D["Ingestion and normalization<br/>parse, clean, chunk, metadata"]
    D --> E["Local / production indexes<br/>BM25, vectors, graph, table, version store"]
    E --> F["{component}<br/>{component_note}"]
    F --> G["Evidence assembly<br/>dedupe, rerank, compress, cite"]
    G --> H["LLM or deterministic answer step"]
    H --> I["Verification and evaluation<br/>faithfulness, citations, recall, MRR"]
    I --> J["Final answer with trace"]

    classDef source fill:#E8F3FF,stroke:#2B6CB0,color:#102A43;
    classDef process fill:#F7FAFC,stroke:#4A5568,color:#1A202C;
    classDef method fill:#FFF4D6,stroke:#B7791F,color:#3D2C00;
    classDef output fill:#E6FFFA,stroke:#2C7A7B,color:#123B3C;
    class S1,S2,S3,S4,S5,C source;
    class B,D,E,G,H,I process;
    class F method;
    class J output;
"""


def update_readme(folder: Path) -> None:
    readme = folder / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if "architecture.mmd" not in text:
        text = text.replace(
            "├── 5-evaluate.py\n├── README.md",
            "├── 5-evaluate.py\n├── architecture.mmd\n├── README.md",
        )
        marker = "## Included example data\n"
        section = (
            "## Mermaid architecture file\n\n"
            "This method includes [`architecture.mmd`](architecture.mmd), a left-to-right `flowchart LR` Mermaid architecture designed for later rendering or conversion into a 3D visual. "
            "A duplicate copy is also stored at [`assets/architecture.mmd`](assets/architecture.mmd).\n\n"
        )
        text = text.replace(marker, section + marker)
        readme.write_text(text, encoding="utf-8")


def main() -> None:
    folders = sorted(p for p in ROOT.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name))
    for folder in folders:
        content = mermaid_for(folder)
        (folder / "architecture.mmd").write_text(content, encoding="utf-8")
        assets = folder / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "architecture.mmd").write_text(content, encoding="utf-8")
        update_readme(folder)
    print(f"Wrote left-to-right Mermaid architecture files for {len(folders)} method folders.")


if __name__ == "__main__":
    main()
