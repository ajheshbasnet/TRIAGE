# TRIAGE
**Gated Evidence Selection and Context Decoupling for Clinical RAG**

A novel Retrieval-Augmented Generation framework that decomposes clinical semantics into orthogonal Entity–Relation–Modifier fields, applies graph-topology anisotropic transformations, and uses adaptive statistical Heaviside filtering combined with a calibrated JEV System One decision gate to mitigate context distraction and hallucinations in medical LLMs.

---

## Key Innovations

1. **Semantic Decoupling**: Represents each medical statement as a 3×d matrix $X_{\text{HCAD}} = [E; R; M]$ instead of a flat 1×d vector, decoupling entities, predicates, and modifiers.
2. **Hybrid Decoupled Scoring**: Ranks candidates by $\Phi = w_d \cdot \cos(q, c) + (1 - w_d) \cdot \Phi_{\text{struct}}$, fusing the full-text dense match with the presence-masked Entity/Relation/Modifier channel score. The legacy topology-block weighting $W_A$ (node degree, edge confidence, IDF) remains available as an ablation (`HCAD_ANISO_DIM_WEIGHTING=1`); a query-adaptive channel-weight adapter can be trained on the dev pool (`train-tensor`, `HCAD_TENSOR_MODE=learned`).
3. **Adaptive Heaviside Filtering**: Replaces rigid Top-K retrieval with a dual-gated dynamic threshold:
   $$\mathcal{M}_{\text{HCAD}}(t_i) = \Theta(\Phi_{\text{Final}}(t_i) - (\mu + k\sigma))$$
4. **JEV System One Decision Gate**: Employs an intelligent decision gate (`typesafe-sdk`) to evaluate factual groundedness and prune irrelevant or ungrounded context before prompt assembly, preventing token bloat and hallucination injection.
5. **Bidirectional Hypergraph Expansion**: Discovers high-order biomedical relations and therapy associations from PrimeKG using multi-hop neighborhood hyperedges.

---

## Architecture

```
Clinical Query
      │
      ▼
Biomedical NER (SciSpacy / SapBERT) ──► Entity Linking (PrimeKG)
      │                                       │
      ▼                                       ▼
Relation & Modifier Extraction        Hyperedge Construction & Expansion
      │                                       │
      ▼                                       ▼
Semantic Decoupling (3×768 Matrix)    Candidate Context Pool
      │                                       │
      └──────────────────┬────────────────────┘
                         │
                         ▼
        Anisotropic Transformation & Tensor Scoring
                         │
                         ▼
       Adaptive Statistical Filter (Heaviside Gate)
                         │
                         ▼
       JEV System One Decision Gate (Groundedness Verification)
                         │
                         ▼
       Structured Clinical Prompt Construction
                         │
                         ▼
       Clinical LLM (Llama-3.1-8B-Instruct) ──► Verified Clinical Synthesis
```

---

## Quick Start

### 1. Prerequisites & Environment Setup

```bash
git clone https://github.com/Nirajd071/HCAD-RAG.git
cd HCAD-RAG

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SciSpacy biomedical NER models
# (--no-deps: their metadata pins spacy<3.8 but they run fine under 3.8)
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz
```

Use the repository-root `.env` as the single local environment file — copy
`.env.example` and keep one profile:

- **Profile A (all LLMs local via ollama, zero LLM API keys):**
  `LLM_PROVIDER=ollama` + `JUDGE_OLLAMA_MODEL=qwen2.5:14b`. Generator
  (`llama3.1:8b`) and judge both run via ollama. Everything else keeps its
  existing behavior — W&B stays online if `WANDB_API_KEY` is set, the JEV
  gate stays live if `TYPESAFE_API_KEY` is set.
- **Profile B (cloud-judge upgrade, opt-in):** `JUDGE_PROVIDER=huggingface`
  puts the judge on paper-parity `Qwen/Qwen2.5-72B-Instruct` via HF Inference
  Providers while the generator stays local.

```bash
LLM_PROVIDER="huggingface"                              # "huggingface", "nvidia" (NIM), or "ollama"
# JUDGE_PROVIDER="huggingface"                          # Optional hybrid: generator local (ollama),
                                                        # judge remote (huggingface = Qwen2.5-72B, or nvidia)
WANDB_API_KEY="your_rotated_wandb_api_key"              # Optional, enables experiment tracking
WANDB_PROJECT="hcad-rag"
# WANDB_ENTITY="your_wandb_username_or_team"            # Optional
```

The root `.env` is git-ignored and is loaded centrally by `src.config`; add
settings there rather than creating another environment file. The W&B API key
is loaded automatically for each evaluation, benchmark, hallucination
evaluation, and weight-learning run. Do not put API keys in source code or
commit them. Set `WANDB_MODE="disabled"` (or pass `--no-wandb`) to turn
tracking off.

### 2. Prepare Knowledge Corpus & Indices

```bash
# Build knowledge graph indices and vector database with clinical relations
python src/main.py prepare --rebuild
```

### 3. Run Benchmark (Base Paper Protocol)

Replicate the evaluation protocol from Hyper-RAG ([arXiv:2504.08758](https://arxiv.org/abs/2504.08758)):

```bash
# Run base paper benchmark comparing Baseline LLM, Vanilla RAG, and HCAD-RAG
python src/evaluation/run_base_paper_benchmark.py 4
```

Outputs:
- **5-Dimension Scoring Assessment** (0–100 scale: Comprehensiveness, Diversity, Empowerment, Logical, Readability)
- **8-Criteria Selection Assessment** (Pairwise Win-Rate with order counterbalancing)
- **Retrieval Latency Profiling**
- **Retrieval Quality** (`src/evaluation/metrics.py`): NDCG@5 / Recall@5 over the paper-protocol gold relevance labels — the source chunk that generated each question
- **Significance Testing**: exact McNemar test on paired per-question correctness (overall score ≥ 70 of 100 counts as correct)
- **Citation Precision**: fraction of `[Evidence i]` tags in generated answers that resolve to actually retrieved items
- **MedHallu hallucination evaluation** via `src/evaluation/hallucination_evaluator.py`

Per-question results are checkpointed to `results/base_paper_checkpoint.jsonl`
and all aggregates are recomputed from the checkpoint, so interrupted runs
resume without losing or double-counting samples. W&B logs the aggregate and
per-question numeric metrics when `WANDB_API_KEY` is configured.

Weight learning tunes only `filter_k`, the learned parameter currently consumed by
the retriever. Anisotropic and component-scoring weights remain at their configured
values; retrieval and model architecture are unchanged.

### Hybrid judging (small-GPU machines)

`JUDGE_PROVIDER` decouples the judge from the generator. With
`LLM_PROVIDER=ollama` (local `llama3.1:8b` via ollama) plus
`JUDGE_PROVIDER=huggingface`, the judge runs on the paper-parity
`Qwen/Qwen2.5-72B-Instruct` through Hugging Face Inference Providers
(pay-per-token, needs HF billing; roughly $2 for a 50-sample run) while all
generation stays free and local. `JUDGE_PROVIDER=nvidia` does the same via
NVIDIA NIM. Without the variable, the judge follows `LLM_PROVIDER`.

### Retrieval scoring knobs (ablations)

| Variable | Default | Effect |
|----------|---------|--------|
| `HCAD_DENSE_WEIGHT` | `0.5` | Weight of the dense full-text cosine in Φ; `0` = legacy structure-only score |
| `HCAD_ANISO_DIM_WEIGHTING` | `0` | `1` re-enables the legacy topology block weighting of embedding dimensions |
| `HCAD_PRESENCE_MASKING` | `1` | Query-conditioned channel-presence masking |
| `HCAD_RELATION_VOCAB` | `1` | PrimeKG relation labels / query relations mapped to canonical verb phrases |
| `HCAD_EXPANSION_RULE` | `soft` | Member order inside relation hyperedges: `desc`, `soft`, `asc_cap` |
| `HCAD_TAU_SAT` | unset | Gate-2 saturation ceiling, calibrated by `retrieval-eval` on the dev pool |

`python src/main.py retrieval-eval --pool dev` reports paired Recall@5/NDCG@5
for the legacy baseline, each added change, and the full configuration.

All three benchmark arms (Baseline LLM, Vanilla RAG, HCAD-RAG) receive the
same answer-synthesis instructions (`synthesis_instructions` in
`src/core/prompt_builder.py`); they differ only in the evidence supplied.

---

## Project Structure

```
hcad-rag/
├── requirements.txt                   # Project dependencies
├── README.md                          # Documentation
├── .gitignore                         # Git exclusion rules
├── src/
│   ├── config.py                      # Global configurations, model weights & paths
│   ├── main.py                        # CLI entry point
│   ├── wandb_tracking.py              # Shared W&B helpers (auto-enabled when configured)
│   ├── api/
│   │   └── server.py                   # FastAPI workbench API (`python src/main.py serve`)
│   ├── data/
│   │   ├── download_datasets.py       # Dataset fetchers (PrimeKG, MedHallu)
│   │   ├── build_knowledge_graph.py   # PrimeKG MultiDiGraph & index construction
│   │   └── build_vector_store.py      # ChromaDB clinical context ingestion
│   ├── core/
│   │   ├── ner_extractor.py           # SciSpacy + Regex Biomedical NER
│   │   ├── entity_linker.py           # SapBERT FAISS cosine entity resolution
│   │   ├── relation_extractor.py      # Rule-based clinical predicate extraction
│   │   ├── semantic_decoupler.py      # Innovation 1: Entity-Relation-Modifier decoupling
│   │   ├── anisotropic_transformer.py # Innovation 2: Graph topology W_A transformation
│   │   ├── hypergraph_builder.py      # Innovation 3: Bidirectional hyperedge expansion
│   │   ├── fact_hypergraph.py         # N-ary clinical fact hypergraph from PrimeKG shared-tail contexts
│   │   ├── statistical_filter.py      # Innovation 4: Adaptive Heaviside cutoff
│   │   ├── decision_gate.py           # Innovation 5: JEV System One groundedness gate
│   │   ├── dynamic_tensor.py          # (P3, opt-in) trainable anisotropic tensor
│   │   ├── entropy_gate.py            # (P3, opt-in) entropy-based abstention gate
│   │   ├── weight_learner.py          # filter_k weight learning
│   │   ├── prompt_builder.py          # Standardized 4-section clinical prompt builder
│   │   ├── llm_client.py              # Unified LLM interface (HuggingFace / NVIDIA NIM / Ollama)
│   │   └── hcad_retriever.py          # End-to-end integrated HCAD-RAG pipeline
│   ├── baselines/
│   │   ├── baseline_llm.py            # Parametric LLM baseline (zero-retrieval)
│   │   └── vanilla_vector_rag.py      # Standard dense Top-K vector baseline
│   └── evaluation/
│       ├── base_paper_evaluator.py    # Scoring & selection rubric evaluator (LLM-as-a-judge)
│       ├── run_base_paper_benchmark.py# Base paper benchmark runner (checkpointed, resume-safe)
│       ├── hallucination_evaluator.py # MedHallu hallucination evaluation
│       ├── metrics.py                 # NDCG@K, Recall@K, McNemar, citation precision
│       └── plots.py                   # Benchmark figures rendered from results JSON
└── results/
    ├── figures/                       # Benchmark charts
    └── tables/                        # Summary metrics & logs
```

---

## Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| [PrimeKG](https://doi.org/10.7910/DVN/IXA7BM) | Harvard Dataverse | Precision medicine knowledge graph (129K nodes, 4M+ edges) |
| [MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | HuggingFace | Medical hallucination detection dataset |
| PMC Open Access | NCBI / PMC | Curated open-access clinical text corpus |

---

## References

- **Hyper-RAG**: *Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation* ([arXiv:2504.08758](https://arxiv.org/abs/2504.08758))
- **SIRG**: *Structured Information Retrieval Gating* ([arXiv:2601.03052](https://arxiv.org/abs/2601.03052))
- **PrimeKG**: Chandak et al., *Building a knowledge graph to enable precision medicine*, Scientific Data (2023)
- **SapBERT**: Liu et al., *Self-Alignment Pretraining for Biomedical Entity Representations*, NAACL (2021)

---

## License

Research and academic use only. Individual components are governed by their respective dataset and model licenses.
