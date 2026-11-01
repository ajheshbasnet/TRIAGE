"""
Build ChromaDB vector store from PrimeKG triples.

Converts each knowledge graph edge (head, relation, tail) into a natural language
sentence, embeds it with S-PubMedBERT-MS-MARCO, and stores in ChromaDB with
rich metadata for filtered retrieval.

Usage:
    python -m src.data.build_vector_store
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any

import chromadb
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    VECTORDB_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
import os

logger = logging.getLogger(__name__)

# GPU encode batch for the embedding model. Small by default so the build
# fits alongside resident LLMs on a single consumer GPU (e.g. 16 GB T4);
# raise it on an idle A100 for faster ingestion.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))

# Shared with the retriever so every channel renders a triple identically.
from src.data.triple_text import triple_to_sentence as _triple_to_sentence

def build_vector_store(
    kg_csv_path: Optional[Path] = None,
    batch_size: int = 2048,
    max_rows: Optional[int] = None,
    relation_filter: Optional[set[str]] = None,
) -> chromadb.Collection:
    """
    Build ChromaDB vector store from PrimeKG triples.

    Args:
        kg_csv_path: Path to PrimeKG kg.csv. Defaults to RAW_DATA_DIR/kg.csv.
        batch_size: Embedding batch size.
        max_rows: Limit edges (None = all ~4M edges; use small number for testing).

    Returns:
        ChromaDB Collection object.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")

    import pandas as pd
    from sentence_transformers import SentenceTransformer

    kg_csv_path = kg_csv_path or (
        RAW_DATA_DIR / "primekg" / "kg.csv"
        if (RAW_DATA_DIR / "primekg" / "kg.csv").exists()
        else RAW_DATA_DIR / "kg.csv"
    )
    if not kg_csv_path.exists():
        raise FileNotFoundError(
            f"PrimeKG CSV not found at {kg_csv_path}. "
            "Run `python src/main.py prepare` first."
        )

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info(f"Loading PrimeKG from {kg_csv_path}")
    df = pd.read_csv(
        kg_csv_path,
        usecols=["relation", "display_relation", "x_name", "y_name", "x_type", "y_type"],
        nrows=max_rows,
    )
    logger.info(f"Loaded {len(df):,} edges")

    # Filter to clinically relevant relation types if specified
    if relation_filter:
        df = df[df['relation'].isin(relation_filter)]
        logger.info(f"Filtered to {len(df)} rows with clinical relations")

    # build into a STAGING *directory* and
    # swap atomically. The previous code deleted the live collection first,
    # so any concurrent query failed and a failed build left no store at all.
    # chromadb has no rename_collection, so the atomic unit is the directory.
    import shutil
    import sqlite3
    staging_dir = VECTORDB_DIR.parent / "vectordb_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    sqlite_db_path = staging_dir / "chroma.sqlite3"
    try:
        conn = sqlite3.connect(str(sqlite_db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")  # avoid 'database is locked' under concurrency
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not apply SQLite WAL pragmas: {e}")

    client = chromadb.PersistentClient(path=str(staging_dir))
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Convert DataFrame columns to lists for fast iteration
    x_names = df["x_name"].astype(str).tolist()
    y_names = df["y_name"].astype(str).tolist()
    disp_rels = df["display_relation"].astype(str).tolist()
    x_types = df["x_type"].astype(str).tolist()
    y_types = df["y_type"].astype(str).tolist()
    relations = df["relation"].astype(str).tolist()

    total_rows = len(x_names)
    logger.info(f"Processing and indexing {total_rows:,} edges in batches of {batch_size}...")

    # Process in batches
    for start_idx in tqdm(range(0, total_rows, batch_size), desc="Building Vector Store"):
        end_idx = min(start_idx + batch_size, total_rows)
        
        batch_texts = [
            _triple_to_sentence(x_names[i], relations[i], y_names[i])  # rel_map is keyed by `relation`
            for i in range(start_idx, end_idx)
        ]
        batch_ids = [f"edge_{i}" for i in range(start_idx, end_idx)]
        batch_metadatas = [
            {
                "head_name": x_names[i],
                "head_type": x_types[i],
                "tail_name": y_names[i],
                "tail_type": y_types[i],
                "relation": relations[i],
                "display_relation": disp_rels[i],
            }
            for i in range(start_idx, end_idx)
        ]

        # GPU encoding; EMBED_BATCH (default 128) keeps activations small
        # enough to coexist with ollama on one GPU.
        embeddings = model.encode(
            batch_texts,
            batch_size=min(len(batch_texts), EMBED_BATCH),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        collection.add(
            documents=batch_texts,
            embeddings=embeddings.tolist(),
            metadatas=batch_metadatas,
            ids=batch_ids,
        )

    built = collection.count()
    logger.info(f"Vector store built: {built:,} documents in staging directory")
    if built == 0:
        raise RuntimeError("Refusing to publish an empty vector store ")

    # atomic publish: directory-level rename (POSIX). The live store remains
    # queryable until this instant; a failure before this point changes nothing.
    import os
    backup_dir = VECTORDB_DIR.parent / "vectordb_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    had_live = VECTORDB_DIR.exists() and any(VECTORDB_DIR.iterdir())
    if had_live:
        os.replace(VECTORDB_DIR, backup_dir)
    os.replace(staging_dir, VECTORDB_DIR)
    if had_live:
        shutil.rmtree(backup_dir, ignore_errors=True)
    final = chromadb.PersistentClient(path=str(VECTORDB_DIR)).get_collection(CHROMA_COLLECTION_NAME)
    logger.info(
        f"Published vector store '{CHROMA_COLLECTION_NAME}' with {final.count():,} documents"
    )
    return final

def load_vector_store() -> chromadb.Collection:
    """Load existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    logger.info(f"Loaded vector store: {collection.count():,} documents")
    return collection

_SHARED_MODEL = None

def _get_shared_model():
    global _SHARED_MODEL
    if _SHARED_MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _SHARED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device, local_files_only=True)
        except Exception:
            _SHARED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=_device)
    return _SHARED_MODEL

def search_similar(
    query: str,
    n_results: int = 5,
    where: Optional[dict] = None,
    collection: Optional[chromadb.Collection] = None,
    embedding_model: Optional[Any] = None,
) -> list[dict]:
    """
    Search for similar contexts using vector similarity.

    Args:
        query: Natural language query.
        n_results: Number of results to return.
        where: Optional ChromaDB metadata filter dict.
        collection: Pre-loaded collection (loaded if None).
        embedding_model: Pre-loaded model.

    Returns:
        List of dicts with keys: document, metadata, distance, score.
    """
    if collection is None:
        collection = load_vector_store()

    model = embedding_model or _get_shared_model()
    try:
        query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    except Exception:
        # GPU OOM fallback to CPU encoding
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from sentence_transformers import SentenceTransformer
        cpu_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        query_embedding = cpu_model.encode([query], normalize_embeddings=True).tolist()

    kwargs = {"query_embeddings": query_embedding, "n_results": n_results}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    for i in range(len(results["documents"][0])):
        output.append(
            {
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1.0 - results["distances"][0][i],  # cosine: closer = higher score
            }
        )
    return output

def ingest_clinical_chunks(
    collection: Optional[chromadb.Collection] = None,
    chunks_path: Optional[Path] = None,
    batch_size: int = 256,
    max_chunks: Optional[int] = None,
) -> int:
    """
    Ingest pre-processed PMC clinical text chunks into ChromaDB.

    Reads chunks.jsonl and meta.jsonl, embeds each chunk with S-PubMedBERT,
    and stores in the same ChromaDB collection with doc_type='clinical_text'.

    Args:
        collection: ChromaDB collection (loaded if None).
        chunks_path: Path to chunks.jsonl. Defaults to PROCESSED_DATA_DIR/chunks.jsonl.
        batch_size: Embedding batch size.
        max_chunks: Limit number of chunks (None = all).

    Returns:
        Number of chunks ingested.
    """
    import json as _json
    from sentence_transformers import SentenceTransformer

    chunks_path = chunks_path or (PROCESSED_DATA_DIR / "chunks.jsonl")
    if not chunks_path.exists():
        logger.warning(f"chunks.jsonl not found at {chunks_path}. Skipping clinical text ingestion.")
        return 0

    if collection is None:
        collection = load_vector_store()

    model = _get_shared_model()

    logger.info(f"Ingesting clinical text chunks from {chunks_path}...")
    texts, ids, metadatas = [], [], []
    ingested = 0

    with open(chunks_path) as f:
        for line_num, line in enumerate(f):
            if max_chunks and line_num >= max_chunks:
                break
            try:
                chunk = _json.loads(line.strip())
            except _json.JSONDecodeError:
                continue

            text = chunk.get("text", "").strip()
            if not text or len(text) < 20:
                continue

            doc_id = str(chunk.get("doc_id", f"unknown_{line_num}"))
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, (int, float)):
                chunk_id = line_num  # non-numeric chunk_id crashed ingestion

            texts.append(text)
            ids.append(f"pmc_{doc_id.replace('/', '_')}_{chunk_id}")
            metadatas.append({
                "doc_type": "clinical_text",
                "source": "pmc",
                "doc_id": str(doc_id),
                "chunk_id": int(chunk_id),
                "head_name": "",
                "tail_name": "",
                "relation": "",
                "display_relation": "",
                "head_type": "",
                "tail_type": "",
            })

            # Flush batch
            if len(texts) >= batch_size:
                embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                collection.upsert(  # idempotent re-ingestion
                    documents=texts,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                    ids=ids,
                )
                ingested += len(texts)
                if ingested % 1000 == 0:
                    logger.info(f"  Ingested {ingested:,} clinical chunks...")
                texts, ids, metadatas = [], [], []

    # Final batch
    if texts:
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
        )
        ingested += len(texts)

    logger.info(f"Clinical text ingestion complete: {ingested:,} chunks added to ChromaDB")
    logger.info(f"Total collection size: {collection.count():,} documents")
    return ingested

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CLINICAL_RELATIONS = {
        'indication', 'contraindication', 'off-label use',
        'disease_phenotype_positive', 'disease_phenotype_negative',
        'drug_effect', 'disease_disease', 'drug_protein', 'disease_protein',
    }
    build_vector_store(max_rows=None, batch_size=2048, relation_filter=CLINICAL_RELATIONS)
    ingest_clinical_chunks(batch_size=2048)
    print("Vector store ready.")
