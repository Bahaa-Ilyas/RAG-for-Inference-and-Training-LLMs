"""Add paper-informed diagram images to every method README.

The generated SVGs are local redraws/summaries based on canonical papers or
canonical documentation. They are not copied paper figures. This keeps the
repository useful for study while avoiding ambiguous reuse rights for original
paper artwork.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


DIAGRAMS = {
    "01-hybrid-rag": {
        "source_title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks + DPR",
        "source_url": "https://arxiv.org/abs/2005.11401",
        "source_kind": "Canonical RAG paper; hybrid BM25+dense is an engineering extension.",
        "nodes": [
            "User Query",
            "BM25 Sparse Search",
            "Dense Retriever",
            "Rank Fusion",
            "Grounded Generator",
            "Cited Answer",
        ],
    },
    "02-reranked-rag": {
        "source_title": "ColBERT and cross-encoder reranking literature",
        "source_url": "https://arxiv.org/abs/2004.12832",
        "source_kind": "Canonical neural reranking / late-interaction source.",
        "nodes": ["Query", "First-stage Retriever", "Candidate Pool", "Reranker", "Top Evidence", "Answer"],
    },
    "03-agentic-rag": {
        "source_title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "source_url": "https://arxiv.org/abs/2210.03629",
        "source_kind": "Canonical reasoning/action loop paper.",
        "nodes": ["Task", "Think / Plan", "Retrieve or Tool", "Observe Evidence", "Revise Plan", "Final Answer"],
    },
    "04-managed-enterprise-rag": {
        "source_title": "Azure AI Search RAG documentation",
        "source_url": "https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview",
        "source_kind": "Canonical managed-platform documentation; no single original paper.",
        "nodes": ["Enterprise Sources", "Managed Ingestion", "ACL + Index", "Vector/Semantic Search", "Grounding Data", "Chat App"],
    },
    "05-graphrag": {
        "source_title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "source_url": "https://arxiv.org/abs/2404.16130",
        "source_kind": "Canonical GraphRAG paper.",
        "nodes": ["Documents", "Entity Extraction", "Knowledge Graph", "Community Summaries", "Local/Global Search", "Answer"],
    },
    "06-raptor-hierarchical-rag": {
        "source_title": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
        "source_url": "https://arxiv.org/abs/2401.18059",
        "source_kind": "Canonical RAPTOR paper.",
        "nodes": ["Leaf Chunks", "Cluster", "Summarize", "Recursive Tree", "Tree Retrieval", "Answer"],
    },
    "07-self-rag": {
        "source_title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "source_url": "https://arxiv.org/abs/2310.11511",
        "source_kind": "Canonical Self-RAG paper.",
        "nodes": ["Input", "Need Retrieval?", "Retrieve Passages", "Generate", "Critique Tokens", "Supported Answer"],
    },
    "08-corrective-rag-crag": {
        "source_title": "Corrective Retrieval Augmented Generation",
        "source_url": "https://arxiv.org/abs/2401.15884",
        "source_kind": "Canonical CRAG paper.",
        "nodes": ["Query", "Initial Retrieval", "Retrieval Evaluator", "Correct / Refine / Web Search", "Knowledge Strip", "Answer"],
    },
    "09-query-rewriting-rag": {
        "source_title": "HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels",
        "source_url": "https://arxiv.org/abs/2212.10496",
        "source_kind": "Canonical query rewriting / hypothetical-document retrieval source.",
        "nodes": ["User Query", "Rewrite / HyDE", "Expanded Query", "Retriever", "Fused Evidence", "Answer"],
    },
    "10-multi-query-rag": {
        "source_title": "LangChain MultiQueryRetriever documentation",
        "source_url": "https://python.langchain.com/docs/how_to/MultiQueryRetriever/",
        "source_kind": "Canonical framework implementation; no single original paper.",
        "nodes": ["User Query", "Query Variant 1", "Query Variant 2", "Query Variant N", "Union + Deduplicate", "Answer"],
    },
    "11-decomposition-rag": {
        "source_title": "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
        "source_url": "https://arxiv.org/abs/2205.10625",
        "source_kind": "Canonical decomposition reasoning paper used as the method analogue.",
        "nodes": ["Complex Query", "Decompose", "Subquestion Retrieval", "Intermediate Answers", "Synthesis", "Verified Answer"],
    },
    "12-metadata-filtered-rag": {
        "source_title": "LlamaIndex auto-retrieval documentation",
        "source_url": "https://docs.llamaindex.ai/en/stable/examples/vector_stores/chroma_auto_retriever/",
        "source_kind": "Canonical framework pattern; no single original paper.",
        "nodes": ["Query", "Filter Extraction", "Metadata Constraints", "Filtered Retrieval", "Scoped Evidence", "Answer"],
    },
    "13-small-to-big-parent-child-rag": {
        "source_title": "LangChain ParentDocumentRetriever documentation",
        "source_url": "https://python.langchain.com/docs/how_to/parent_document_retriever/",
        "source_kind": "Canonical framework implementation; no single original paper.",
        "nodes": ["Small Chunks", "Child Retriever", "Hit Child IDs", "Parent Lookup", "Expanded Context", "Answer"],
    },
    "14-contextual-compression-rag": {
        "source_title": "LangChain contextual compression documentation",
        "source_url": "https://python.langchain.com/docs/how_to/contextual_compression/",
        "source_kind": "Canonical framework implementation; no single original paper.",
        "nodes": ["Query", "Retrieve Chunks", "Compress / Extract", "Relevant Spans", "Compact Context", "Answer"],
    },
    "15-table-aware-rag": {
        "source_title": "TAPAS: Weakly Supervised Table Parsing via Pre-training",
        "source_url": "https://arxiv.org/abs/2004.02349",
        "source_kind": "Canonical table QA architecture source used as the table-reasoning analogue.",
        "nodes": ["Question", "Table Parser", "Rows + Columns", "Cell/Row Selection", "Aggregation", "Cited Table Answer"],
    },
    "16-layout-aware-multimodal-pdf-rag": {
        "source_title": "LayoutLMv3: Pre-training for Document AI",
        "source_url": "https://arxiv.org/abs/2204.08387",
        "source_kind": "Canonical layout-aware document AI source.",
        "nodes": ["PDF Page", "OCR + Layout", "Text + Boxes + Images", "Multimodal Index", "Page Evidence", "Answer"],
    },
    "17-webpage-rag-freshness": {
        "source_title": "Trafilatura documentation",
        "source_url": "https://trafilatura.readthedocs.io/",
        "source_kind": "Canonical web extraction documentation; freshness RAG is an engineering pattern.",
        "nodes": ["Seed URLs", "Crawler", "Clean HTML", "Canonical URL + Hash", "Freshness Filter", "Answer"],
    },
    "18-version-aware-rag": {
        "source_title": "OpenSearch vector search and filtering documentation",
        "source_url": "https://opensearch.org/docs/latest/search-plugins/vector-search/",
        "source_kind": "Canonical search/filtering source; version-aware RAG is an engineering pattern.",
        "nodes": ["Query + Time", "Version Resolver", "Validity Window", "Filtered Retrieval", "Current Evidence", "Answer"],
    },
    "19-tool-augmented-rag": {
        "source_title": "Toolformer and ReAct tool-use papers",
        "source_url": "https://arxiv.org/abs/2302.04761",
        "source_kind": "Canonical tool-use paper; ReAct is also relevant.",
        "nodes": ["Task", "Retrieve Policy", "Choose Tool", "Call API", "Merge Evidence", "Answer"],
    },
    "20-claim-level-verification-rag": {
        "source_title": "FActScore: Fine-grained Atomic Evaluation of Factual Precision",
        "source_url": "https://arxiv.org/abs/2305.14251",
        "source_kind": "Canonical claim-level factuality evaluation source.",
        "nodes": ["Draft Answer", "Atomic Claims", "Evidence Retrieval", "Support Check", "Revise / Abstain", "Verified Answer"],
    },
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


def svg_for(folder: Path, metadata: dict[str, object]) -> str:
    method_title = title_from_readme(folder)
    source_title = str(metadata["source_title"])
    source_kind = str(metadata["source_kind"])
    nodes = [str(node) for node in metadata["nodes"]]
    width = 1280
    height = 460
    box_w = 165
    box_h = 72
    gap = 30
    start_x = 40
    y = 190
    title = html.escape(method_title)
    source = html.escape(source_title)
    kind = html.escape(source_kind.rstrip("."))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title} paper-informed diagram">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/></marker>',
        '<linearGradient id="method" x1="0" x2="1"><stop offset="0%" stop-color="#fff7ed"/><stop offset="100%" stop-color="#fef3c7"/></linearGradient>',
        "</defs>",
        '<rect width="1280" height="460" fill="#ffffff"/>',
        f'<text x="40" y="52" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#0f172a">{title}</text>',
        f'<text x="40" y="84" font-family="Arial, sans-serif" font-size="15" fill="#475569">Paper-informed explanatory diagram based on: {source}</text>',
        f'<text x="40" y="108" font-family="Arial, sans-serif" font-size="13" fill="#64748b">{kind}. Local redraw; not a copied paper figure.</text>',
    ]

    for i, node in enumerate(nodes):
        x = start_x + i * (box_w + gap)
        fill = "#e0f2fe" if i == 0 else "url(#method)" if i in {2, 3} else "#f8fafc"
        stroke = "#0284c7" if i == 0 else "#b45309" if i in {2, 3} else "#64748b"
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        words = html.escape(node).split()
        lines: list[str] = []
        current = ""
        for word in words:
            if len(current + " " + word) > 18:
                lines.append(current.strip())
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)
        line_y = y + 30 - (len(lines) - 1) * 9
        for line in lines[:3]:
            parts.append(f'<text x="{x + box_w / 2}" y="{line_y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="600" fill="#0f172a">{line}</text>')
            line_y += 18
        if i < len(nodes) - 1:
            x1 = x + box_w
            x2 = x + box_w + gap - 8
            parts.append(f'<line x1="{x1}" y1="{y + box_h / 2}" x2="{x2}" y2="{y + box_h / 2}" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)"/>')

    parts.extend(
        [
            '<text x="40" y="350" font-family="Arial, sans-serif" font-size="14" fill="#475569">How to read it: source data flows left to right through retrieval/control stages, then into answer construction and verification.</text>',
            '<text x="40" y="376" font-family="Arial, sans-serif" font-size="14" fill="#475569">Use this SVG as a clean visual base for later 3D rendering or slide generation.</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def image_block(folder: Path, metadata: dict[str, object]) -> str:
    source_title = str(metadata["source_title"])
    source_url = str(metadata["source_url"])
    return f"""## Paper-Informed Method Diagram

![Paper-informed method diagram](assets/paper_diagram.svg)

Source: [{source_title}]({source_url}). This local SVG is a redraw/synthesis of the important method architecture from the canonical paper or source, not a copied paper figure.

"""


def update_readme(folder: Path, metadata: dict[str, object]) -> None:
    readme = folder / "README.md"
    text = readme.read_text(encoding="utf-8")
    block = image_block(folder, metadata)
    pattern = r"## Paper-Informed Method Diagram\n\n!\[Paper-informed method diagram\]\(assets/paper_diagram\.svg\)\n\nSource: .+?\n\n"
    if re.search(pattern, text, flags=re.S):
        text = re.sub(pattern, block, text, count=1, flags=re.S)
    elif "## Mermaid architecture file" in text:
        text = text.replace("## Mermaid architecture file\n\n", block + "## Mermaid architecture file\n\n", 1)
    elif "## Included example data" in text:
        text = text.replace("## Included example data\n\n", block + "## Included example data\n\n", 1)
    else:
        text = text.rstrip() + "\n\n" + block
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    count = 0
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name)):
        metadata = DIAGRAMS.get(folder.name)
        if not metadata:
            continue
        assets = folder / "assets"
        assets.mkdir(exist_ok=True)
        (assets / "paper_diagram.svg").write_text(svg_for(folder, metadata), encoding="utf-8")
        update_readme(folder, metadata)
        count += 1
    print(f"Wrote paper-informed SVG diagrams and README image blocks for {count} methods.")


if __name__ == "__main__":
    main()
