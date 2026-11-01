import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("triage")

from src.wandb_tracking import tracked_run

def cmd_prepare(args):
    """Download datasets, prepare text chunks, build knowledge graph + SapBERT index, create vector store."""
    import time
    from src.data.download_datasets import download_all
    from src.data.build_knowledge_graph import load_knowledge_graph
    from src.data.build_vector_store import build_vector_store, ingest_clinical_chunks

    start = time.time()

    # ── Step 1: Download all raw datasets ──────────────────────────
    print("\n" + "=" * 60)
    print("Step 1/5: Downloading datasets (PrimeKG, MedHallu, PMC)...")
    print("=" * 60)
    paths = download_all()
    if "primekg" not in paths:
        # PrimeKG is the substrate - nothing downstream works without it.
        raise RuntimeError("Dataset preparation incomplete - PrimeKG download failed")
    if "medhallu" not in paths:
        # MedHallu is only needed by the hallucination evaluator, not by the
        # base-paper benchmark - a failure here must not block prepare.
        logger.warning(
            "MedHallu unavailable - hallucination evaluation will not run "
            "until it downloads. Continuing."
        )
    if "pmc" not in paths:
        logger.warning(
            "PMC articles unavailable - the vector store will contain PrimeKG "
            "triples only (no clinical prose chunks). Continuing."
        )

    # ── Step 2: Prepare PMC clinical text chunks ───────────────────
    print("\n" + "=" * 60)
    print("Step 2/5: Preparing PMC clinical text chunks...")
    print("=" * 60)
    from src.config import PROJECT_ROOT as CFG_ROOT
    chunks_path = CFG_ROOT / "data" / "processed" / "chunks.jsonl"
    if chunks_path.exists():
        print(f"  Chunks already exist at {chunks_path} - skipping.")
    else:
        pmc_dir = CFG_ROOT / "data" / "raw" / "pmc"
        if pmc_dir.exists() and list(pmc_dir.glob("*.txt")):
            import subprocess
            # absolute paths + cwd - the previous
            # CWD-relative invocation failed outside the repo root.
            subprocess.run(
                [sys.executable, str(CFG_ROOT / "data" / "prepare_chunks.py")],
                check=True, cwd=str(CFG_ROOT),
            )
            print(f"  Chunks prepared → {chunks_path}")
        else:
            print("  No PMC articles found in data/raw/pmc/ - skipping chunk preparation.")

    # ── Step 3: Build Knowledge Graph + SapBERT Index ──────────────
    print("\n" + "=" * 60)
    print("Step 3/5: Building PrimeKG knowledge graph...")
    print("=" * 60)
    kg = load_knowledge_graph(force_rebuild=args.rebuild)
    num_nodes = kg.graph.number_of_nodes() if hasattr(kg, "graph") else 0
    num_edges = kg.graph.number_of_edges() if hasattr(kg, "graph") else 0
    print(f"  Graph: {num_nodes:,} nodes, {num_edges:,} edges")

    # ── Step 4: Build SapBERT FAISS entity linking index ───────────
    print("\n" + "=" * 60)
    print("Step 4/5: Building SapBERT FAISS entity linking index...")
    print("=" * 60)
    try:
        from src.core.entity_linker import SapBERTLinker
        linker = SapBERTLinker()
        linker.build_index(kg)
        print(f"  SapBERT index: {len(linker.node_names):,} nodes indexed")
    except ImportError as e:
        print(f"  ⚠ SapBERT index skipped (missing dependency: {e})")
    except Exception as e:
        print(f"  ⚠ SapBERT index failed: {e}")

    # ── Step 5: Build ChromaDB vector store with clinical relations ─
    print("\n" + "=" * 60)
    print("Step 5/5: Building ChromaDB vector store...")
    print("=" * 60)
    CLINICAL_RELATIONS = {
        'indication', 'contraindication', 'off-label use',
        'disease_phenotype_positive', 'disease_phenotype_negative',
        'drug_effect', 'disease_disease', 'drug_protein', 'disease_protein',
    }
    col = build_vector_store(
        max_rows=args.max_rows,
        # Chroma add batch; the GPU encode batch inside is capped separately
        # by EMBED_BATCH (default 128) so VRAM stays small alongside ollama.
        batch_size=int(os.getenv("VECTOR_BATCH", "512")),
        relation_filter=CLINICAL_RELATIONS,
    )
    print(f"  KG triples indexed: {col.count():,} documents")

    # Ingest clinical text chunks
    n_chunks = ingest_clinical_chunks(collection=col, batch_size=int(os.getenv("VECTOR_BATCH", "512")))
    print(f"  Clinical text chunks added: {n_chunks:,}")
    print(f"  Total vector store size: {col.count():,} documents")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"✓ Data preparation complete in {elapsed/60:.1f} minutes.")
    print("  Run `python src/main.py demo` to test.")
    print("=" * 60)

# Command: demo

def cmd_demo(args):
    """Run a single query through all three systems for comparison."""
    query = args.query

    print("\n" + "=" * 60)
    print("HCAD-RAG: Interactive Demo")
    print("=" * 60)
    print(f"\nQuery: {query}\n")

    # Load systems
    try:
        from src.baselines.baseline_llm import BaselineLLM
        from src.baselines.vanilla_vector_rag import VanillaVectorRAG
        from src.core.hcad_retriever import HCADRetriever

        print("[1/3] Baseline LLM (no retrieval)...")
        baseline = BaselineLLM()
        b_answer = baseline.generate(query)
        print(f"  Answer:\n{b_answer.strip()}\n")

        print("[2/3] Vanilla Vector-RAG (Top-K retrieval)...")
        vanilla = VanillaVectorRAG()
        v_answer = vanilla.retrieve_and_generate(query)
        print(f"  Answer:\n{v_answer.strip()}\n")

        print("[3/3] HCAD-RAG (Proposed)...")
        hcad = HCADRetriever()
        h_answer = hcad.retrieve_and_generate(query)
        print(f"  Answer:\n{h_answer.strip()}\n")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n⚠ Demo error: {e}")
        print("Make sure you have run `python src/main.py prepare` first,")
        print("and that Ollama is running (`ollama pull llama3.1:8b`).")
        raise

def _load_baseline():
    from src.baselines.baseline_llm import BaselineLLM
    return BaselineLLM()

def _load_vanilla():
    from src.baselines.vanilla_vector_rag import VanillaVectorRAG
    return VanillaVectorRAG()

def _load_hcad():
    from src.core.hcad_retriever import HCADRetriever
    return HCADRetriever()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage",
        description="TRIAGE: Gated Evidence Selection and Context Decoupling for Clinical RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py prepare
  python src/main.py demo --query "Can Metformin be given to renal failure patients?"
  python src/main.py evaluate --system hcad-rag --samples 200
  python src/main.py benchmark --samples 200
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    p_prep = subparsers.add_parser("prepare", help="Download data and build KG/vector store")
    p_prep.add_argument("--rebuild", action="store_true", help="Force rebuild even if files exist")
    p_prep.add_argument("--max-rows", type=int, default=None,
                        dest="max_rows",
                        help="Limit PrimeKG edges for quick testing (e.g. 50000)")
    p_prep.set_defaults(func=cmd_prepare)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run a single query through all systems")
    p_demo.add_argument(
        "--query", "-q",
        default="Can Metformin be given to patients with severe renal failure?",
        help="Medical question to answer",
    )
    p_demo.set_defaults(func=cmd_demo)

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate", help="Alias of base-paper: run the paper-protocol benchmark")
    p_eval.add_argument(
        "--question-pool", choices=["test", "dev"], default="test",
        help="Question pool: test (held-out evaluation) or dev (training/tuning)",
    )
    p_eval.add_argument(
        "--samples", type=int, default=50,
        help="Number of paper-protocol questions ",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    # benchmark
    p_bench = subparsers.add_parser(
        "benchmark", help="Alias of base-paper: full 3-way comparison (Baseline vs Vanilla RAG vs HCAD-RAG)")
    p_bench.add_argument(
        "--question-pool", choices=["test", "dev"], default="test",
        help="Question pool: test (held-out evaluation) or dev (training/tuning)",
    )
    p_bench.add_argument(
        "--samples", type=int, default=50,
        help="Number of paper-protocol questions ",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # base-paper (the project's primary benchmark: arXiv:2504.08758 protocol)
    p_bp = subparsers.add_parser(
        "base-paper", help="Run the base-paper protocol benchmark (scoring + selection + latency)",
    )
    p_bp.add_argument(
        "--pool", "--question-pool", dest="question_pool", choices=["test", "dev"], default="test",
        help="Question pool: test (held-out evaluation) or dev (training/tuning)",
    )
    p_bp.add_argument(
        "--samples", type=int, default=50,
        help="Number of paper-protocol questions ",
    )
    p_bp.add_argument("--plot", action="store_true", help="Render result figures to results/figures/")
    p_bp.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking for this run")
    p_bp.set_defaults(func=cmd_base_paper)

    # save-pipeline
    p_tt = subparsers.add_parser(
        "train-tensor", help="Train the query-adaptive anisotropic tensor on the DEV question pool")
    p_tt.add_argument("--samples", type=int, default=60,
                      help="Number of dev-pool questions to train on")
    p_tt.add_argument("--epochs", type=int, default=3)
    p_tt.add_argument("--lr", type=float, default=1e-3)
    p_save = subparsers.add_parser("save-pipeline", help="Serialize entire HCAD-RAG pipeline to bundle")
    p_save.add_argument(
        "--output-dir", type=str, default="models/hcad_pipeline_v1",
        help="Target bundle output directory",
    )
    p_save.add_argument(
        "--file", type=str, default="models/hcad_rag_model.joblib",
        help="Target joblib model artifact file path",
    )
    p_tt.set_defaults(func=cmd_train_tensor)

    # retrieval-eval (Deliverable-B dev-pool harness)
    p_re = subparsers.add_parser(
        "retrieval-eval",
        help="Deliverable-B: paired before/after retrieval eval (Recall@5/NDCG@5) on the dev pool",
    )
    p_re.add_argument(
        "--pool", "--question-pool", dest="question_pool", choices=["dev", "test"], default="dev",
        help="Question pool (default dev - test stays untouched until GO)",
    )
    p_re.add_argument("--samples", type=int, default=60,
                      help="Number of pool questions (default 60)")
    p_re.set_defaults(func=cmd_retrieval_eval)

    # hub-eval (Phase 4a ablation)
    p_hub = subparsers.add_parser(
        "hub-eval",
        help="Phase 4a: neighbor-expansion-rule ablation (desc/soft/asc_cap) on the dev pool",
    )
    p_hub.add_argument(
        "--pool", "--question-pool", dest="question_pool", choices=["dev", "test"], default="dev",
        help="Question pool (default dev)",
    )
    p_hub.add_argument("--samples", type=int, default=60,
                       help="Number of pool questions (default 60)")
    p_hub.set_defaults(func=cmd_hub_eval)
    p_save.set_defaults(func=cmd_save_pipeline)

    # serve (workbench API)
    p_serve = subparsers.add_parser(
        "serve", help="Serve the TRIAGE workbench HTTP API (FastAPI)",
    )
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port")
    p_serve.set_defaults(func=cmd_serve)

    return parser

def cmd_serve(args):
    """Run the FastAPI server for the Next.js workbench."""
    import uvicorn
    from src.api.server import app
    logger.info("Starting TRIAGE API on %s:%d ...", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)

def cmd_evaluate(args):
    """Alias of base-paper: the benchmark evaluates all three systems."""
    from src.evaluation.run_base_paper_benchmark import run_benchmark
    logger.info("Evaluating all three systems (%d samples)...", args.samples)
    run_benchmark(num_samples=args.samples, question_pool=args.question_pool)

def cmd_benchmark(args):
    """Run full 3-way benchmark comparison (Baseline vs Vanilla RAG vs HCAD-RAG)."""
    from src.evaluation.run_base_paper_benchmark import run_benchmark
    logger.info("Running full 3-way benchmark (%d samples per system)...", args.samples)
    run_benchmark(num_samples=args.samples, question_pool=args.question_pool)

def cmd_base_paper(args):
    """Run the base-paper (Hyper-RAG, arXiv:2504.08758) evaluation protocol."""
    if getattr(args, "no_wandb", False):
        import os
        os.environ["WANDB_MODE"] = "disabled"
        import src.config as _cfg
        _cfg.WANDB_MODE = "disabled"
    from src.evaluation.run_base_paper_benchmark import run_benchmark
    results = run_benchmark(num_samples=args.samples, question_pool=args.question_pool)
    if getattr(args, "plot", False) and results:
        from src.evaluation.plots import plot_all
        from pathlib import Path as _P
        paths = plot_all(results, hallu_path=_P("results/tables/hallucination_hcad-rag.json"))
        for p_ in paths:
            logger.info("Figure saved: %s", p_)

def cmd_train_tensor(args):
    """Train the dynamic anisotropic tensor on dev-pool questions (leakage-safe)."""
    from src.core.hcad_retriever import HCADRetriever
    from src.core.tensor_trainer import train_tensor
    from src.evaluation.run_base_paper_benchmark import generate_paper_protocol_questions

    logger.info("Generating/loading DEV question pool (%d questions)...", args.samples)
    questions = generate_paper_protocol_questions(args.samples, pool="dev")
    retriever = HCADRetriever()
    result = train_tensor(retriever, questions, epochs=args.epochs, lr=args.lr)
    logger.info("Tensor training complete: %s", result)

def cmd_retrieval_eval(args):
    """Deliverable-B: paired before/after retrieval eval on the dev pool."""
    from src.evaluation.dev_pool_retrieval import run_retrieval_eval

    results = run_retrieval_eval(
        num_samples=args.samples, pool=args.question_pool
    )
    decision = (results.get("go_stop") or {}).get("decision", "n/a")
    logger.info("GO/STOP decision: %s", decision)
    for name in ("baseline", "masking", "masking_vocab", "hybrid", "full"):
        cfg = results.get(name)
        if cfg:
            logger.info(
                "  %-13s Recall@5=%.4f  NDCG@5=%.4f",
                name, cfg["mean_recall_at_k"], cfg["mean_ndcg_at_k"],
            )
    if results.get("delta_vs_baseline"):
        d = results["delta_vs_baseline"]
        logger.info(
            "  delta vs baseline: dRecall@5=%+.4f  dNDCG@5=%+.4f",
            d["recall_at_k"], d["ndcg_at_k"],
        )

def cmd_hub_eval(args):
    """Phase 4a: neighbor-expansion-rule ablation on the dev pool."""
    from src.evaluation.dev_pool_retrieval import run_hub_ablation

    results = run_hub_ablation(
        num_samples=args.samples, pool=args.question_pool
    )

def cmd_save_pipeline(args):
    """Serialize the entire HCAD-RAG pipeline into a joblib model artifact."""
    from src.core.model_serializer import save_hcad_model_joblib
    save_hcad_model_joblib(output_path=args.file)

# Main

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
