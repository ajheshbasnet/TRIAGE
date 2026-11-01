"""Config-versioned decoupler cache (Phase 4b: latency precompute).

Online retrieval decouples every candidate on every query: SciSpacy NER on
prose chunks plus S-PubMedBert encoding of three strings per candidate -
the dominant share of HCAD-RAG's retrieval latency. The candidate corpus
is static after indexing, so this module caches:

  1. extracted component strings (t_E, t_R, t_M) per document text, and
  2. embeddings per unique component string (bounded in-memory LRU).

Every cache key embeds a CONFIG VERSION hash of everything that influences
extraction or orthogonalization - the Gram-Schmidt formula version,
ORTHO_STRENGTH, residual tolerance, neutral anchors, the relation-vocab
map, negation cues, keyword lists, and the embedding model. Any change to
those silently invalidates old entries by construction: a new version
means a new cache file namespace, and a mismatched on-disk file is
ignored.

Memory budget: the embedding LRU is capped (default 20k entries x 768
float32 ~ 61 MB, under the review's 100 MB ceiling); component strings
persist to a small JSONL sidecar so restored sessions keep their NER work.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    CACHE_DIR,
    DECOUPLER_EMBED_CACHE_MAXSIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    GS_EPS,
    GS_RESIDUAL_TOL,
    NEUTRAL_MODIFIER_TEXT,
    NEUTRAL_RELATION_TEXT,
    ORTHO_STRENGTH,
    RELATION_VOCAB_MAP,
    decoupler_cache_enabled,
)

logger = logging.getLogger(__name__)

# Bump when the extraction or Gram-Schmidt math itself changes in a way not
# captured by the config constants below (e.g. the MGS pass structure).
DECOUPLER_FORMULA_VERSION = "mgs2+presence+vocab+wordbound"


def decoupler_config_version(extra_components: Optional[list] = None) -> str:
    """Stable hash of every input that changes decoupled outputs.

    A cache entry written under one version must never be read under
    another: the review flags stale cached embeddings against changed
    eta / MGS formula / vocab map / negation rules as a silent-corruption
    hazard, so the version key is content-addressed over all of them.
    Config constants are read at call time (module attribute access) so
    runtime edits invalidate immediately.
    """
    import src.config as _cfg

    parts = [
        DECOUPLER_FORMULA_VERSION,
        f"eta={_cfg.ORTHO_STRENGTH}",
        f"gs_eps={_cfg.GS_EPS}",
        f"gs_tol={_cfg.GS_RESIDUAL_TOL}",
        f"neutral_rel={_cfg.NEUTRAL_RELATION_TEXT}",
        f"neutral_mod={_cfg.NEUTRAL_MODIFIER_TEXT}",
        f"embed_model={_cfg.EMBEDDING_MODEL_NAME}",
        f"embed_dim={_cfg.EMBEDDING_DIMENSION}",
        json.dumps(_cfg.RELATION_VOCAB_MAP, sort_keys=True),
    ]
    parts.extend(str(p) for p in (extra_components or []))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _text_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


class DecouplerCache:
    """Component-string cache (memory + JSONL sidecar) + embedding LRU."""

    def __init__(
        self,
        version: str,
        dim: int = EMBEDDING_DIMENSION,
        embed_maxsize: int = DECOUPLER_EMBED_CACHE_MAXSIZE,
        disk_dir: Optional[Path] = None,
    ) -> None:
        self.version = version
        self.dim = dim
        self.embed_maxsize = embed_maxsize
        self.disk_dir = Path(disk_dir) if disk_dir else CACHE_DIR
        self._components: dict[str, tuple[str, str, str]] = {}
        self._embed: OrderedDict[str, np.ndarray] = OrderedDict()
        self._disk_path = self.disk_dir / f"decoupler_components_{version}.jsonl"
        self._disk_dirty = 0
        self.n_component_hits = 0
        self.n_component_misses = 0
        self.n_embed_hits = 0
        self.n_embed_misses = 0
        self._load_disk()
        atexit.register(self.flush)

    # -- component strings (t_E, t_R, t_M) ---------------------------------

    def get_components(self, text: str) -> Optional[tuple[str, str, str]]:
        key = _text_key(text)
        hit = self._components.get(key)
        if hit is not None:
            self.n_component_hits += 1
        else:
            self.n_component_misses += 1
        return hit

    def put_components(
        self, text: str, entity_text: str, relation_text: str, modifier_text: str
    ) -> None:
        key = _text_key(text)
        if key in self._components:
            return
        self._components[key] = (entity_text, relation_text, modifier_text)
        self._disk_dirty += 1
        if self._disk_dirty >= 64:
            self.flush()

    def _load_disk(self) -> None:
        if not self._disk_path.exists():
            return
        try:
            with open(self._disk_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # truncated tail from a killed session
                    if record.get("version") != self.version:
                        continue  # never trust a foreign version
                    self._components[record["key"]] = tuple(record["components"])
            logger.info(
                "Decoupler cache: %d component entries loaded from %s",
                len(self._components), self._disk_path,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Decoupler cache load failed (%s); starting cold.", exc)

    def flush(self) -> None:
        """Persist new component entries to the versioned sidecar."""
        if not self._components or self._disk_dirty == 0:
            return
        try:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            new_entries = [
                {"version": self.version, "key": k, "components": list(v)}
                for k, v in self._components.items()
            ]
            # rewrite atomically (small file: strings only)
            tmp = self._disk_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for record in new_entries:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp.replace(self._disk_path)
            self._disk_dirty = 0
        except Exception as exc:  # pragma: no cover - cache is best-effort
            logger.warning("Decoupler cache flush failed: %s", exc)

    # -- embeddings ---------------------------------------------------------

    def get_embedding(self, text: str, dim: Optional[int] = None) -> Optional[np.ndarray]:
        """Cached embedding for *text*, or None.

        The entry's dimension must match the requested (runtime) *dim*:
        a vector cached under a different embedding dimension is a miss,
        never a wrong-shape hit.
        """
        key = _text_key(text)
        entry = self._embed.get(key)
        if entry is not None:
            entry_dim, vec = entry
            if dim is None or entry_dim == dim:
                self._embed.move_to_end(key)
                self.n_embed_hits += 1
                return vec
            self.n_embed_misses += 1
            return None
        self.n_embed_misses += 1
        return None

    def put_embedding(self, text: str, vec: np.ndarray) -> None:
        key = _text_key(text)
        vec = np.asarray(vec, dtype=np.float32)
        if key not in self._embed and len(self._embed) >= self.embed_maxsize:
            self._embed.popitem(last=False)  # drop the least-recently-used
        self._embed[key] = (int(vec.shape[0]), vec)
        self._embed.move_to_end(key)

    def stats(self) -> dict:
        return {
            "version": self.version,
            "component_entries": len(self._components),
            "component_hits": self.n_component_hits,
            "component_misses": self.n_component_misses,
            "embed_entries": len(self._embed),
            "embed_hits": self.n_embed_hits,
            "embed_misses": self.n_embed_misses,
            "disk_path": str(self._disk_path),
        }


_SHARED: Optional[DecouplerCache] = None


def get_decoupler_cache(
    extra_version_components: Optional[list] = None,
) -> Optional[DecouplerCache]:
    """Process-wide cache, or None when disabled via HCAD_DECOUPLER_CACHE=0."""
    global _SHARED
    if not decoupler_cache_enabled():
        return None
    version = decoupler_config_version(extra_version_components)
    if _SHARED is None or _SHARED.version != version:
        if _SHARED is not None:
            _SHARED.flush()
        _SHARED = DecouplerCache(version)
    return _SHARED
