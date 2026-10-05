from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    from sentence_transformers import SentenceTransformer
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n+', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    global _semantic_model
    try:
        _semantic_model
    except NameError:
        _semantic_model = None

    if _semantic_model is None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = _semantic_model.encode(sentences)

    def cos_sim(a, b):
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = cos_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(
            text=" ".join(g),
            metadata={**metadata, "strategy": "semantic", "chunk_index": idx}
        )
        for idx, g in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        if text.strip():
            paragraphs = [text.strip()]
        else:
            return ([], [])

    parents: list[Chunk] = []
    children: list[Chunk] = []

    parent_groups: list[list[str]] = []
    curr_group: list[str] = []
    curr_len = 0

    for para in paragraphs:
        para_len = len(para)
        if curr_group and (curr_len + para_len + 2 > parent_size):
            parent_groups.append(curr_group)
            curr_group = [para]
            curr_len = para_len
        else:
            curr_group.append(para)
            curr_len += para_len + 2

    if curr_group:
        parent_groups.append(curr_group)

    source_prefix = f"{metadata.get('source', '')}_" if metadata.get('source') else ""
    for p_idx, p_paras in enumerate(parent_groups):
        pid = f"{source_prefix}parent_{p_idx}"
        p_text = "\n\n".join(p_paras).strip()
        parents.append(Chunk(
            text=p_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": p_idx},
            parent_id=pid
        ))

        # Split parent into children chunks smaller than child_size
        child_units: list[str] = []
        for para in p_paras:
            if len(para) <= child_size:
                child_units.append(para)
            else:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
                for s in sentences:
                    if len(s) <= child_size:
                        child_units.append(s)
                    else:
                        words = s.split()
                        curr_w = ""
                        for w in words:
                            if curr_w and (len(curr_w) + len(w) + 1 > child_size):
                                child_units.append(curr_w)
                                curr_w = w
                            else:
                                curr_w = f"{curr_w} {w}".strip() if curr_w else w
                        if curr_w:
                            child_units.append(curr_w)

        curr_c_text = ""
        for u in child_units:
            if curr_c_text and (len(curr_c_text) + len(u) + 1 > child_size):
                children.append(Chunk(
                    text=curr_c_text.strip(),
                    metadata={**metadata, "chunk_type": "child", "parent_id": pid, "child_index": len(children)},
                    parent_id=pid
                ))
                curr_c_text = u
            else:
                curr_c_text = f"{curr_c_text} {u}".strip() if curr_c_text else u

        if curr_c_text.strip():
            children.append(Chunk(
                text=curr_c_text.strip(),
                metadata={**metadata, "chunk_type": "child", "parent_id": pid, "child_index": len(children)},
                parent_id=pid
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    chunks = []
    current_header = ""
    current_content = ""

    for sec in sections:
        if not sec:
            continue
        header_match = re.match(r'^#{1,3}\s+(.+)$', sec.strip())
        if header_match:
            combined = f"{current_header}\n\n{current_content}".strip() if current_header else current_content.strip()
            if combined:
                chunks.append(Chunk(
                    text=combined,
                    metadata={**metadata, "section": current_header or "Intro", "strategy": "structure", "chunk_index": len(chunks)}
                ))
            current_header = sec.strip()
            current_content = ""
        else:
            current_content += sec

    combined = f"{current_header}\n\n{current_content}".strip() if current_header else current_content.strip()
    if combined:
        chunks.append(Chunk(
            text=combined,
            metadata={**metadata, "section": current_header or "Intro", "strategy": "structure", "chunk_index": len(chunks)}
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
