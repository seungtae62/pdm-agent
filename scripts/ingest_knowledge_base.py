"""Knowledge Base → Qdrant 문서 적재 스크립트.

data/knowledge_base/ 하위 문서를 파싱·청킹·임베딩하여
Qdrant의 equipment_manual / maintenance_history 컬렉션에 적재한다.

Usage:
    python scripts/ingest_knowledge_base.py              # 전체 적재
    python scripts/ingest_knowledge_base.py --dry-run     # 적재 없이 확인만
    python scripts/ingest_knowledge_base.py --collection equipment_manual
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
MANUAL_DIR = KB_DIR / "equipment_manual"
WO_DIR = KB_DIR / "maintenance_history" / "work_orders"
CR_DIR = KB_DIR / "maintenance_history" / "completion_reports"
PARSED_RECORDS = PROJECT_ROOT / "data" / "parsed_maintenance_records.json"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = 20

# Deterministic UUID namespace
UUID_NS = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL

# doc_id prefix → doc_type mapping
DOC_TYPE_MAP = {
    "SPEC": "spec",
    "FAULT": "fault_guide",
    "MAINT": "procedure",
    "TOOL_LIST": "tool_list",
}

# Filename line letter → equipment_id mapping
LINE_MAP = {
    "A": "LINE-A",
    "B": "LINE-B",
    "C": "LINE-C",
    "D": "LINE-D",
}

# Regex for maintenance history filenames
# e.g. WO-20250113-001_A_AX-01_instruction.pdf
MAINT_FILENAME_RE = re.compile(
    r"^(WO-\d{8}-\d{3})_([A-D])_([A-Z]{2}-\d{2})_(instruction|completion)\.pdf$"
)


# ---------------------------------------------------------------------------
# 텍스트 추출
# ---------------------------------------------------------------------------


def extract_pdf_text(pdf_path: Path) -> str:
    """PDF에서 전체 텍스트 추출 (PyMuPDF)."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages).strip()


def extract_md_text(md_path: Path) -> str:
    """Markdown 파일 텍스트 읽기."""
    return md_path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# 설비매뉴얼 처리 (Recursive Chunking)
# ---------------------------------------------------------------------------


def _detect_doc_type(filename: str) -> str:
    """파일명에서 doc_type 추출."""
    for prefix, dtype in DOC_TYPE_MAP.items():
        if prefix in filename:
            return dtype
    return "unknown"


def process_equipment_manuals() -> list[dict]:
    """설비매뉴얼을 recursive chunking하여 문서 리스트 반환."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents: list[dict] = []

    # EM-001/002/003: .md 우선 사용
    for md_file in sorted(MANUAL_DIR.glob("EM-*.md")):
        doc_id = md_file.name.split("_")[0]  # e.g. EM-001
        doc_type = _detect_doc_type(md_file.name)
        text = extract_md_text(md_file)
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "doc_type": doc_type,
                "equipment_model": "ZA2115",
                "doc_id": doc_id,
                "chunk_index": i,
                "source_file": md_file.name,
            })

    # TOOL_LIST_V1.pdf: PDF만 존재
    tool_list_pdf = MANUAL_DIR / "TOOL_LIST_V1.pdf"
    if tool_list_pdf.exists():
        text = extract_pdf_text(tool_list_pdf)
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "doc_type": "tool_list",
                "equipment_model": "ZA2115",
                "doc_id": "TOOL_LIST_V1",
                "chunk_index": i,
                "source_file": tool_list_pdf.name,
            })

    logger.info(
        f"equipment_manual: {len(documents)} chunks from "
        f"{len(list(MANUAL_DIR.glob('EM-*.md')))} MD + 1 PDF"
    )
    return documents


# ---------------------------------------------------------------------------
# 정비이력 처리 (Single Page)
# ---------------------------------------------------------------------------


def _load_parsed_records() -> dict[str, dict]:
    """parsed_maintenance_records.json에서 wo_number 기반 lookup dict 생성."""
    if not PARSED_RECORDS.exists():
        logger.warning(f"parsed_maintenance_records.json not found: {PARSED_RECORDS}")
        return {}

    with open(PARSED_RECORDS, encoding="utf-8") as f:
        data = json.load(f)

    lookup: dict[str, dict] = {}
    for record in data.get("records", []):
        instruction = record.get("instruction", {})
        completion = record.get("completion", {})
        wo = instruction.get("wo_number") or completion.get("wo_number")
        if wo:
            lookup[wo] = {
                "fault_type": (
                    instruction.get("summary", "")
                ),
                "work_type": instruction.get("work_type", ""),
                "instruction_summary": instruction.get("summary", ""),
                "completion_summary": completion.get("result_summary", ""),
            }
    return lookup


def process_maintenance_history() -> list[dict]:
    """작업지시서 + 완료보고서 PDF를 단일 페이지 문서로 처리."""
    lookup = _load_parsed_records()
    documents: list[dict] = []

    for directory, doc_subtype in [(WO_DIR, "instruction"), (CR_DIR, "completion")]:
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            continue

        for pdf_file in sorted(directory.glob("*.pdf")):
            match = MAINT_FILENAME_RE.match(pdf_file.name)
            if not match:
                logger.warning(f"Filename pattern mismatch: {pdf_file.name}")
                continue

            wo_number = match.group(1)
            line_letter = match.group(2)
            bearing_id = match.group(3)
            equipment_id = LINE_MAP.get(line_letter, f"LINE-{line_letter}")

            text = extract_pdf_text(pdf_file)
            if not text:
                logger.warning(f"Empty text extracted: {pdf_file.name}")
                continue

            # 메타데이터 보강
            meta = lookup.get(wo_number, {})

            doc = {
                "text": text,
                "equipment_id": equipment_id,
                "bearing_id": bearing_id,
                "fault_type": meta.get("fault_type", ""),
                "wo_number": wo_number,
                "doc_subtype": doc_subtype,
                "work_type": meta.get("work_type", ""),
                "summary": (
                    meta.get("instruction_summary", "")
                    if doc_subtype == "instruction"
                    else meta.get("completion_summary", "")
                ),
            }
            documents.append(doc)

    logger.info(f"maintenance_history: {len(documents)} documents")
    return documents


# ---------------------------------------------------------------------------
# 임베딩 & Qdrant Upsert
# ---------------------------------------------------------------------------


def batch_embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    """OpenAI 배치 임베딩."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([d.embedding for d in response.data])
        logger.info(
            f"  Embedded batch {i // EMBEDDING_BATCH_SIZE + 1}"
            f"/{(len(texts) - 1) // EMBEDDING_BATCH_SIZE + 1}"
        )
    return all_embeddings


def make_point_id(source_file: str, chunk_index: int | None = None) -> str:
    """Deterministic UUID5 생성."""
    key = source_file
    if chunk_index is not None:
        key = f"{source_file}::{chunk_index}"
    return str(uuid.uuid5(UUID_NS, key))


def upsert_documents(
    collection_name: str,
    documents: list[dict],
    qdrant: QdrantClient,
    openai_client: OpenAI,
) -> None:
    """문서 임베딩 후 Qdrant upsert."""
    if not documents:
        logger.info(f"  No documents to upsert for '{collection_name}'.")
        return

    texts = [doc["text"] for doc in documents]
    logger.info(f"  Embedding {len(texts)} texts for '{collection_name}'...")
    embeddings = batch_embed(texts, openai_client)

    points: list[PointStruct] = []
    for doc, embedding in zip(documents, embeddings):
        source_file = doc.get("source_file", doc.get("wo_number", "unknown"))
        chunk_index = doc.get("chunk_index")
        point_id = make_point_id(source_file, chunk_index)

        # payload = doc without embedding, keep text and all metadata
        payload = {k: v for k, v in doc.items()}
        points.append(
            PointStruct(id=point_id, vector=embedding, payload=payload)
        )

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant.upsert(collection_name=collection_name, points=batch)

    logger.info(f"  Upserted {len(points)} points to '{collection_name}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_dry_run(collection_name: str, documents: list[dict]) -> None:
    """Dry-run: 파싱/메타데이터 결과 출력."""
    print(f"\n{'='*60}")
    print(f"Collection: {collection_name} ({len(documents)} documents)")
    print(f"{'='*60}")

    for i, doc in enumerate(documents[:5]):
        print(f"\n--- Document {i+1} ---")
        for k, v in doc.items():
            if k == "text":
                print(f"  text: {v[:120]}...")
            else:
                print(f"  {k}: {v}")

    if len(documents) > 5:
        print(f"\n  ... and {len(documents) - 5} more documents")


def main() -> None:
    """메인 실행."""
    parser = argparse.ArgumentParser(
        description="Knowledge Base → Qdrant 문서 적재"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="적재 없이 파싱/메타데이터/청킹 결과만 확인",
    )
    parser.add_argument(
        "--collection",
        choices=["equipment_manual", "maintenance_history"],
        help="특정 컬렉션만 적재",
    )
    args = parser.parse_args()

    targets = {}

    if args.collection is None or args.collection == "equipment_manual":
        targets["equipment_manual"] = process_equipment_manuals()

    if args.collection is None or args.collection == "maintenance_history":
        targets["maintenance_history"] = process_maintenance_history()

    if args.dry_run:
        for name, docs in targets.items():
            print_dry_run(name, docs)
        print("\n[dry-run] No data was uploaded to Qdrant.")
        return

    # Real upsert
    openai_client = OpenAI()
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    for name, docs in targets.items():
        upsert_documents(name, docs, qdrant, openai_client)

    print("\nDone. All documents ingested.")


if __name__ == "__main__":
    main()
