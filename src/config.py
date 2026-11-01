import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORDB_DIR = DATA_DIR / "vectordb"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
TENSOR_MODEL_PATH = MODELS_DIR / "dynamic_tensor.pt"
TABLES_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORDB_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Datasets
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
PRIMEKG_FILENAME = "kg.csv"

MEDHALLU_DATASET_ID = "UTAustin-AIHealth/MedHallu"
MEDHALLU_CONFIG_LABELED = "pqa_labeled"
MEDHALLU_CONFIG_ARTIFICIAL = "pqa_artificial"

# Embedding & Entity Linking Models
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIMENSION = 768
ENTITY_LINKING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# HCAD Core Parameters
NUM_SEMANTIC_COMPONENTS = 3  # Entity, Relation, Modifier
HCAD_MATRIX_SHAPE = (NUM_SEMANTIC_COMPONENTS, EMBEDDING_DIMENSION)

# Anisotropic Transformation Weights
ANISO_ALPHA = 0.4   # Node degree
ANISO_BETA = 0.4    # Edge confidence
ANISO_GAMMA = 0.2   # Inverse document frequency

# Retrieval Score Weights
SCORE_WEIGHT_ENTITY = 0.3
SCORE_WEIGHT_RELATION = 0.5
SCORE_WEIGHT_MODIFIER = 0.2

# Statistical Filtering
FILTERING_K = 1.0
MIN_PHI_SCORE = 0.10

# Hybrid scoring: Phi = w_d * dense_cosine(query, document)
#                      + (1 - w_d) * Phi_struct(E/R/M channels).
# The structured channels compare short extracted fragments (entity names,
# relation phrase, modifiers); on their own they discard the full-text
# semantic match that dense retrieval ranks by. Fusing the two keeps the
# dense signal as a floor and lets the decoupled channels re-rank on top
# of it. HCAD_DENSE_WEIGHT=0 restores the legacy structure-only score.
HYBRID_DENSE_WEIGHT = 0.5


def dense_weight() -> float:
    """Active dense-score weight w_d in [0, 1] (env HCAD_DENSE_WEIGHT)."""
    raw = os.environ.get("HCAD_DENSE_WEIGHT")
    if raw is None or raw == "":
        return HYBRID_DENSE_WEIGHT
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return HYBRID_DENSE_WEIGHT


# Anisotropic per-dimension weighting. The topology statistics (degree,
# edge confidence, IDF) were mapped onto three contiguous 256-dim blocks of
# the sentence embedding, but embedding coordinates carry no such meaning,
# PrimeKG edges have no confidence attribute (constant 1.0) and IDF used a
# constant document frequency - the weights were effectively a fixed vector
# that suppressed the first third of every embedding. Default OFF
# (isotropic weighted cosine == plain cosine); HCAD_ANISO_DIM_WEIGHTING=1
# re-enables the legacy block weighting for ablations.
ANISO_DIM_WEIGHTING = False


def aniso_dim_weighting_enabled() -> bool:
    default = "1" if ANISO_DIM_WEIGHTING else "0"
    return os.environ.get("HCAD_ANISO_DIM_WEIGHTING", default) == "1"


# Entity linking: exact (case-insensitive) PrimeKG name match first, then
# SapBERT nearest neighbour above this cosine threshold.
ENTITY_LINK_THRESHOLD = 0.65

# Candidate pool composition.
CANDIDATE_POOL_MAX = 80          # total candidates scored per query
GRAPH_TRIPLES_PER_ENTITY = 12    # hyperedge triples emitted per linked entity
GRAPH_TRIPLES_PER_HYPEREDGE = 3  # spread over relation-typed hyperedges
NARY_FACTS_PER_QUERY = 10

# DecisionGate: at most this many evidence items may share one relation
# type before lower-scored items of other relations/prose are preferred.
MAX_EVIDENCE_PER_RELATION = 3

# Semantic Decoupling 
ORTHO_STRENGTH = 0.70        # soft Gram-Schmidt blend weight (eta)
GS_EPS = 1e-8
GS_RESIDUAL_TOL = 1e-3      # below this, a component is marked absent (presence mask)
NEUTRAL_RELATION_TEXT = "biomedical relation"   # anchor, never equal to entity text
NEUTRAL_MODIFIER_TEXT = "no qualifier stated"

# Knowledge-graph relation exclusion policy
KG_EXCLUDED_RELATIONS = {"drug_drug"}

# ---------------------------------------------------------------------------
# Deliverable-B (Phase 3): query-conditioned channel masking, relation
# vocabulary mapping, and Gate-2 saturation cap.
# ---------------------------------------------------------------------------

# Calibrated prior similarity assigned to a candidate that is missing a
# channel the query actively uses (p_k^Q = 1, p_k^C = 0). Replaces both the
# old silent 1.0 (neutral-anchor cosine) and the proposed drop-from-
# denominator bug; range [0.15, 0.25] per the epistemic audit.
CHANNEL_MISSING_PRIOR = 0.20

# Query-conditioned presence masking: omega_k = w_k * p_k^Q / sum_j w_j * p_j^Q.
# Toggles default ON (the recommended experiment); HCAD_PRESENCE_MASKING=0
# disables for paired before/after measurement.
CHANNEL_PRESENCE_MASKING = True

# PrimeKG schema labels -> canonical clinical verb phrases. Raw schema
# strings ("drug protein") barely cosine-match natural query verbs
# ("targets") in S-PubMedBert; the mapped phrases close that gap. drug_drug
# stays excluded (KG_EXCLUDED_RELATIONS) to preserve the benchmark protocol
# and the memory footprint.
RELATION_VOCAB_MAPPING = True
RELATION_VOCAB_MAP = {
    "contraindication": "is contraindicated in and must not be used for",
    "indication": "is indicated for the treatment of",
    "off-label use": "is used off-label for",
    "drug_effect": "causes effect or adverse effect",
    "disease_disease": "is associated with",
    "drug_protein": "targets or binds protein",
    "disease_protein": "is associated with altered protein",
    "disease_phenotype_positive": "presents with phenotype",
    "disease_phenotype_negative": "excludes phenotype and is not characterized by",
    "biomarker": "is a biomarker for",
    "phenotype_protein": "phenotype associated with protein",
    "protein_protein": "interacts with protein",
}

# Gate-2 saturation ceiling: tau_eff = min(tau_sat, max(Phi_min, mu + k*sigma)).
# None disables the cap until it is calibrated on the development pool as the
# 25th percentile of gold development candidate scores; set via
# HCAD_TAU_SAT (e.g. "0.42") once calibrated.
TAU_SAT = None
TAU_SAT_PERCENTILE = 25.0


def presence_masking_enabled() -> bool:
    """Runtime toggle for query-conditioned channel masking."""
    default = "1" if CHANNEL_PRESENCE_MASKING else "0"
    return os.environ.get("HCAD_PRESENCE_MASKING", default) == "1"


def relation_vocab_enabled() -> bool:
    """Runtime toggle for PrimeKG relation-vocabulary mapping."""
    default = "1" if RELATION_VOCAB_MAPPING else "0"
    return os.environ.get("HCAD_RELATION_VOCAB", default) == "1"


# Query-side relation types (RelationExtractor rule labels) rendered with
# the same canonical verb phrases the candidates use, so the R channel
# compares like with like ("TREATS" vs "is indicated for the treatment of").
QUERY_RELATION_PHRASES = {
    "TREATS": RELATION_VOCAB_MAP["indication"],
    "INDICATED_FOR": RELATION_VOCAB_MAP["indication"],
    "CONTRAINDICATED": RELATION_VOCAB_MAP["contraindication"],
    "NOT_RECOMMENDED": RELATION_VOCAB_MAP["contraindication"],
    "CAUSES": RELATION_VOCAB_MAP["drug_effect"],
    "TARGETS": RELATION_VOCAB_MAP["drug_protein"],
    "INHIBITS": RELATION_VOCAB_MAP["drug_protein"],
    "ACTIVATES": RELATION_VOCAB_MAP["drug_protein"],
    "BIOMARKER_FOR": RELATION_VOCAB_MAP["biomarker"],
    "ASSOCIATED_WITH": RELATION_VOCAB_MAP["disease_disease"],
}

# Query relation label -> PrimeKG relations it asks about. Hyperedges of
# these relations are expanded first for the query's entities.
QUERY_RELATION_TO_PRIMEKG = {
    "TREATS": {"indication", "off-label use"},
    "INDICATED_FOR": {"indication", "off-label use"},
    "CONTRAINDICATED": {"contraindication"},
    "NOT_RECOMMENDED": {"contraindication"},
    "CAUSES": {"drug_effect", "disease_phenotype_positive"},
    "TARGETS": {"drug_protein"},
    "INHIBITS": {"drug_protein"},
    "ACTIVATES": {"drug_protein"},
    "BIOMARKER_FOR": {"biomarker"},
    "ASSOCIATED_WITH": {"disease_disease", "disease_protein"},
}

# PrimeKG stores every edge in both directions. For directed relations the
# head must be of this node type; reversed copies are swapped back so the
# evidence reads correctly ("Metformin is indicated for X", never
# "X is indicated for Metformin") and duplicates collapse.
RELATION_HEAD_TYPE = {
    "indication": "drug",
    "contraindication": "drug",
    "off-label use": "drug",
    "drug_effect": "drug",
    "drug_protein": "drug",
    "disease_protein": "disease",
    "disease_phenotype_positive": "disease",
    "disease_phenotype_negative": "disease",
    "phenotype_protein": "effect/phenotype",
}


def map_query_relation(raw_relation: str) -> str:
    """Canonical verb phrase for a query-side rule relation label."""
    if not raw_relation:
        return ""
    if relation_vocab_enabled():
        phrase = QUERY_RELATION_PHRASES.get(raw_relation.strip().upper())
        if phrase:
            return phrase
    return raw_relation.replace("_", " ").lower()


def map_relation_text(raw_relation: str) -> str:
    """Map a raw PrimeKG relation label to a canonical verb phrase.

    Unknown relations keep the legacy underscores-to-spaces rendering.
    """
    if not raw_relation:
        return ""
    key = raw_relation.strip().lower()
    if relation_vocab_enabled() and key in RELATION_VOCAB_MAP:
        return RELATION_VOCAB_MAP[key]
    return key.replace("_", " ")


def tau_sat() -> float | None:
    """Calibrated Gate-2 saturation ceiling, if set."""
    raw = os.environ.get("HCAD_TAU_SAT")
    if raw:
        try:
            return float(raw)
        except ValueError:
            return None
    return TAU_SAT


# ---------------------------------------------------------------------------
# Phase 4a: hub-degree bias correction (soft, no hard cap).
#
# PrimeKG is scale-free: promiscuous hubs (NR3C1, CYP3A4, ALB, TP53, major
# diseases) have degrees orders of magnitude above specific targets
# (SERPINA6, RBP4). Descending-degree neighbor truncation deterministically
# floods candidate slices with hubs; a HARD degree cap would blind the
# system to canonical receptors and comorbidities (deg > 500 is normal for
# them), so the deployed rule is a soft inverse-log-degree weighting:
# hubs stay reachable but specific neighbors win the truncation slice.
# ---------------------------------------------------------------------------

# "desc"     - legacy descending-degree truncation (ablation baseline)
# "soft"     - soft inverse-log-degree weighting, NO hard drop (default)
# "asc_cap"  - ascending degree with hard deg<=cap drop (ablation-only;
#              expected to lose canonical high-degree targets)
HUB_EXPANSION_RULE = "soft"
HUB_DEGREE_CAP = 500
HUB_SOFT_WEIGHT_EPS = 1.0


def expansion_rule() -> str:
    """Active neighborhood-expansion rule (env HCAD_EXPANSION_RULE)."""
    rule = os.environ.get("HCAD_EXPANSION_RULE", HUB_EXPANSION_RULE)
    if rule not in ("desc", "soft", "asc_cap"):
        raise ValueError(f"Unknown HCAD_EXPANSION_RULE: {rule!r}")
    return rule


def hub_degree_cap() -> int:
    return int(os.environ.get("HCAD_HUB_DEGREE_CAP", HUB_DEGREE_CAP))


def hub_weight(degree: int) -> float:
    """Soft inverse-log-degree specificity weight, in (0, 1].

    1 / (1 + ln(1 + deg)) - monotonically decreasing in degree, never zero:
    a 10^5-degree hub keeps ~12% weight (reachable), while a degree-3
    specific target keeps ~50%.
    """
    import math

    return 1.0 / (1.0 + math.log1p(max(0, int(degree))) * HUB_SOFT_WEIGHT_EPS)


# ---------------------------------------------------------------------------
# Phase 4b: latency precompute with a config-versioned decoupler cache.
# ---------------------------------------------------------------------------

DECOUPLER_CACHE_ENABLED = True
DECOUPLER_EMBED_CACHE_MAXSIZE = 20_000  # ~61 MB at d=768 float32


def decoupler_cache_enabled() -> bool:
    default = "1" if DECOUPLER_CACHE_ENABLED else "0"
    return os.environ.get("HCAD_DECOUPLER_CACHE", default) == "1"

# LLM
LLM_OLLAMA_TIMEOUT_S = 120.0

# Graph Expansion
GRAPH_HOP_DEPTH = 2
MAX_NEIGHBORS_PER_HOP = 50
MAX_EVIDENCE_ITEMS = 8
EVIDENCE_WORD_BUDGET = 350

# Vanilla Baseline
VANILLA_TOP_K = 5

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")  # "nvidia", "huggingface", or "ollama"
LLM_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Separate judge to avoid self-preference bias
# Base-paper parity : 72B-class judge as in the paper's
# protocol. Fits A100 80GB locally (Q4). Override to qwen3:14b for small GPUs.
JUDGE_OLLAMA_MODEL = os.getenv("JUDGE_OLLAMA_MODEL", "qwen2.5:72b")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Text-optimized NIM models (vision-instruct variants underperform on clinical
# prose and waste tokens on the image tower). Override via env if needed.
LLM_NVIDIA_MODEL = os.getenv("LLM_NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
JUDGE_NVIDIA_MODEL = os.getenv("JUDGE_NVIDIA_MODEL", "qwen/qwen2.5-72b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Hub downloads (SapBERT/S-PubMedBert) still need this
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "hcad-rag")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None
WANDB_MODE = os.getenv("WANDB_MODE", "online")
LLM_MODEL = "llama3.1:8b"
# Base-paper Fig. 3 parity: evaluate augmentation generality across generators.
BENCHMARK_GENERATORS = [LLM_MODEL]  # e.g. ["llama3.1:8b", "qwen3:14b", "deepseek-r1:8b"]
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
ENABLE_FEW_SHOT = False  # zero-shot default; opt into few-shot explicitly
FEW_SHOT_COUNT = 4  # consumed by PromptBuilder.build_mcq_prompt

# Embeddings are always L2-normalized at encode time (cosine consistency)

LLM_SYSTEM_PROMPT = """You are a biomedical research assistant specializing in clinical medicine.
Use the provided context as primary evidence, and synthesize it with your expert clinical
domain knowledge to deliver comprehensive, accurate, and actionable answers.
Clearly cite specific evidence from the context where applicable, and supplement
with established medical knowledge when the context is incomplete.
Be precise, thorough, and clinically rigorous."""

# System prompt for any arm answering WITHOUT retrieved evidence: the
# no-retrieval baseline and HCAD-RAG when its gate abstains. Shared so both
# no-evidence cases are prompted identically.
NO_CONTEXT_SYSTEM_PROMPT = (
    "You are a biomedical research assistant specializing in clinical medicine. "
    "Use your expert clinical domain knowledge to deliver comprehensive, accurate, "
    "and actionable answers. Be precise, thorough, and clinically rigorous."
)

# Vector Database
CHROMA_COLLECTION_NAME = "biomedical_contexts"
CHROMA_DISTANCE_METRIC = "cosine"

# NER Models
NER_MODELS = {
    "drug_disease": "en_ner_bc5cdr_md",
    "molecular": "en_ner_jnlpba_md",
}

# Relation Types & Negation
RELATION_TYPES = [
    "TREATS",
    "CAUSES",
    "CONTRAINDICATED",
    "INTERACTS_WITH",
    "TARGETS",
    "ASSOCIATED_WITH",
    "INHIBITS",
    "ACTIVATES",
    "UPREGULATES",
    "DOWNREGULATES",
    "BIOMARKER_FOR",
    "NOT_RECOMMENDED",
    "INDICATED_FOR",
]

NEGATION_CUES = [
    "not", "no", "never", "without", "avoid", "contraindicated",
    "should not", "must not", "do not", "doesn't", "does not",
    "isn't", "is not", "wasn't", "was not", "cannot", "can't",
    "not recommended", "discontinued", "withdrawn", "prohibited",
    "lack of", "absence of", "failure to",
    "rules out", "ruled out", "unlikely", "fails to improve", "cannot tolerate",
    "refractory to", "excluded",
]

# Evaluation Parameters
DEFAULT_EVAL_SAMPLES = 200
RANDOM_SEED = 42

