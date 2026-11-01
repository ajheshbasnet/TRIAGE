"""
Hypergraph construction from PrimeKG.

Converts PrimeKG's pairwise graph into hyperedges that simultaneously connect
multiple biomedical entities (Drug, Disease, Gene, Protein, etc.), capturing
higher-order semantic relationships.

Mathematical reference:
    H = (V, EH)  where EH = set of hyperedges
    Importance(EH_i) = Σ λ_j × NodeScore_j
    Multi-hop: N²(v) = ⋃ N(u) for u ∈ N(v)
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Any

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import GRAPH_HOP_DEPTH, MAX_NEIGHBORS_PER_HOP

logger = logging.getLogger(__name__)

class HypergraphBuilder:
    """
    Builds hyperedges from PrimeKG neighborhood structures.

    A hyperedge connects multiple biomedical entities simultaneously,
    capturing higher-order relationships beyond simple pairwise edges.
    """

    def __init__(self, graph: nx.MultiDiGraph):
        """
        Args:
            graph: PrimeKG as a NetworkX MultiDiGraph (from build_knowledge_graph).
        """
        if graph is None:
            raise ValueError("graph cannot be None for HypergraphBuilder")
        self.graph = graph
        self._degree_cache: dict[str, int] = {}
        self._density: Optional[float] = None

    def _resolve_entity(self, entity_name: Any) -> Any:
        """Resolve string entity name to node key in self.graph if needed."""
        if entity_name in self.graph:
            return entity_name
        name_to_node = getattr(self.graph, "name_to_node", None)
        if name_to_node and isinstance(entity_name, str):
            if entity_name in name_to_node:
                return name_to_node[entity_name]
            node_lower = entity_name.lower()
            if node_lower in name_to_node:
                return name_to_node[node_lower]
        return entity_name

    # Core Hyperedge Construction

    def build_hyperedges(self, entity_name: str) -> list[dict]:
        """
        Build hyperedges centered on a given entity.

        Groups the entity's neighborhood into hyperedges, where each hyperedge
        connects a set of related biomedical entities.

        Args:
            entity_name: Name or ID of the central entity (e.g., "Metformin").

        Returns:
            List of hyperedge dicts, each with:
                - nodes: list of graph node keys in the hyperedge
                - types: list of corresponding node types
                - relations: list of relation types involved
                - importance: float importance score
        """
        node_id = self._resolve_entity(entity_name)
        if node_id not in self.graph:
            logger.warning(f"Entity '{entity_name}' not found in graph.")
            return []

        # Get multi-hop neighborhood
        neighborhood = self.expand_neighborhood(node_id, hops=GRAPH_HOP_DEPTH)

        # Group neighbors by relation type to form hyperedges
        relation_groups: dict[str, list[Any]] = {}
        for neighbor in neighborhood:
            if neighbor == node_id:
                continue
            # Get all edges between entity and neighbor
            # union BOTH directions - the previous
            # forward-first check dropped reciprocal edges on MultiDiGraph.
            edges = dict(self.graph.get_edge_data(node_id, neighbor) or {})
            edges.update(self.graph.get_edge_data(neighbor, node_id) or {})
            for edge_key, edge_data in edges.items():
                relation = edge_data.get("relation", "unknown")
                relation_groups.setdefault(relation, []).append(neighbor)

        # Build hyperedges: one per relation type + the central entity
        hyperedges = []
        for relation, members in relation_groups.items():
            # Cap hyperedge size
            members = members[:MAX_NEIGHBORS_PER_HOP]
            nodes = [node_id] + members
            types = [
                self.graph.nodes[n].get("node_type", "unknown")
                for n in nodes
                if n in self.graph
            ]
            he = {
                "nodes": nodes,
                "types": types,
                "relations": [relation] * len(members),
                "importance": self.compute_hyperedge_importance(
                    {"nodes": nodes, "types": types}
                ),
            }
            hyperedges.append(he)

        # Also build a "global" hyperedge with all neighbors
        all_neighbors = list(neighborhood - {node_id})[:MAX_NEIGHBORS_PER_HOP]
        if all_neighbors:
            all_nodes = [node_id] + all_neighbors
            all_types = [
                self.graph.nodes[n].get("node_type", "unknown")
                for n in all_nodes
                if n in self.graph
            ]
            all_relations = list(
                set(
                    data.get("relation", "unknown")
                    for _, _, data in self.graph.edges(node_id, data=True)
                )
            )
            hyperedges.append(
                {
                    "nodes": all_nodes,
                    "types": all_types,
                    "relations": all_relations,
                    "importance": self.compute_hyperedge_importance(
                        {"nodes": all_nodes, "types": all_types}
                    ),
                }
            )

        # Sort by importance descending
        hyperedges.sort(key=lambda h: h["importance"], reverse=True)
        logger.debug(
            f"Built {len(hyperedges)} hyperedges for '{entity_name}'"
        )
        return hyperedges

    def relation_hyperedges(self, entity_name: Any) -> list[dict]:
        """Relation-typed hyperedges over the entity's full 1-hop neighbourhood.

        One hyperedge per relation type: the centre plus EVERY neighbour it
        shares that relation with, members ordered by the active expansion
        rule (soft = specific neighbours first, hubs kept). Each hyperedge
        carries the concrete edges (u, v) aligned with its members, so the
        retriever can emit the facts the hyperedge actually contains.

        build_hyperedges() groups only the 50 neighbours kept by the
        2-hop expansion cap, so a drug's indications could be cut in favour
        of rarer protein targets; the retriever previously also read
        triples from the centre's first CSV edges for every hyperedge,
        ignoring both the grouping and the expansion rule.

        Returns hyperedges with keys: relation, nodes ([centre] + members),
        edges (list of (u, v) aligned with members), importance.
        """
        node_id = self._resolve_entity(entity_name)
        if node_id not in self.graph:
            return []

        groups: dict[str, dict[Any, tuple]] = {}
        for u, v, data in self.graph.out_edges(node_id, data=True):
            if v != node_id:
                groups.setdefault(data.get("relation", "unknown"), {}).setdefault(v, (u, v))
        if hasattr(self.graph, "in_edges"):
            for u, v, data in self.graph.in_edges(node_id, data=True):
                if u != node_id:
                    groups.setdefault(data.get("relation", "unknown"), {}).setdefault(u, (u, v))

        hyperedges = []
        for relation, neighbour_edges in groups.items():
            members = self._ordered_neighbors(set(neighbour_edges))
            if not members:
                continue
            nodes = [node_id] + members
            hyperedges.append({
                "relation": relation,
                "nodes": nodes,
                "edges": [neighbour_edges[m] for m in members],
                "importance": self.compute_hyperedge_importance(
                    {"nodes": nodes[: 1 + MAX_NEIGHBORS_PER_HOP]}
                ),
            })
        hyperedges.sort(key=lambda h: h["importance"], reverse=True)
        return hyperedges

    # Incidence Matrix

    def get_incidence_matrix(self, hyperedges: list[dict]) -> np.ndarray:
        """
        Build binary incidence matrix H where H[v][e] = 1 if vertex v ∈ hyperedge e.

        Args:
            hyperedges: List of hyperedge dicts from build_hyperedges().

        Returns:
            np.ndarray of shape (num_nodes, num_hyperedges).
        """
        all_nodes = list(
            {n for he in hyperedges for n in he["nodes"]}
        )
        node_idx = {n: i for i, n in enumerate(all_nodes)}

        H = np.zeros((len(all_nodes), len(hyperedges)), dtype=np.float32)
        for j, he in enumerate(hyperedges):
            for node in he["nodes"]:
                if node in node_idx:
                    H[node_idx[node], j] = 1.0
        return H

    # Importance Scoring

    def compute_hyperedge_importance(
        self, hyperedge: dict, weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute importance score for a hyperedge.

        Legacy ("desc"): Importance(EH) = sum_v ln(1 + deg(v)) - strictly
        monotonically increasing in cardinality and promiscuity, so
        scale-free hubs (NR3C1, CYP3A4) dominate every slice.

        Phase 4 ("soft"): soft inverse-entity-frequency importance

            Importance_IEF(EH) = (1/|EH|) * sum_v 1 / (1 + ln(1 + deg(v)))

        Specific low-degree hyperedges outrank hub-stuffed ones, but hubs
        keep positive weight - canonical receptors and major diseases stay
        reachable (no hard degree cap).

        Args:
            hyperedge: Dict with 'nodes' and 'types' keys.
            weights: Optional anisotropic weights (defaults to degree-based).

        Returns:
            Float importance score.
        """
        nodes = hyperedge.get("nodes", [])
        if not nodes:
            return 0.0

        from src.config import expansion_rule

        rule = expansion_rule()
        scores = []
        for node in nodes:
            if node in self.graph:
                degree = self.graph.degree(node)
                if rule == "desc":
                    node_score = np.log1p(degree)  # legacy log-degree
                else:
                    # soft (and the ablation-only asc_cap variant): inverse
                    # log-degree specificity weight
                    node_score = 1.0 / (1.0 + np.log1p(degree))
            else:
                node_score = 0.0
            scores.append(node_score)

        if rule == "desc":
            total = float(np.sum(scores))
        else:
            # mean, not sum: decouples importance from hyperedge cardinality
            total = float(np.mean(scores)) if scores else 0.0

        if weights is not None and len(weights) > 0:
            # Use scalar weight mean as a scaling factor
            scale = float(np.mean(weights))
            return total * scale

        return total

    # Neighborhood Expansion

    def _ordered_neighbors(self, neighbors: set) -> list:
        """Deterministic neighbor order under the active expansion rule.

        desc:     descending degree (legacy - hub-favoring truncation)
        soft:     descending soft inverse-log-degree weight (hub-damped,
                  no hard drop - the Phase-4 default)
        asc_cap:  ascending degree with a hard deg <= cap drop
                  (ablation-only; expected to lose canonical hubs)
        """
        from src.config import expansion_rule, hub_degree_cap

        rule = expansion_rule()
        if rule == "desc":
            return sorted(neighbors, key=lambda n: (-self.graph.degree(n), str(n)))
        if rule == "asc_cap":
            cap = hub_degree_cap()
            return sorted(
                (n for n in neighbors if self.graph.degree(n) <= cap),
                key=lambda n: (self.graph.degree(n), str(n)),
            )
        # soft: high specificity weight first; hubs stay in the list
        return sorted(
            neighbors,
            key=lambda n: (
                -(1.0 / (1.0 + np.log1p(self.graph.degree(n)))),
                str(n),
            ),
        )

    def expand_neighborhood(self, node_name: str, hops: int = 2) -> set:
        """
        Multi-hop neighborhood expansion: N(v), N²(v), N³(v).

        N^k(v) = ⋃ N(u) for u ∈ N^(k-1)(v)

        Args:
            node_name: Starting node name or ID.
            hops: Expansion depth (default from config).

        Returns:
            Set of all node names reachable within `hops` steps.
        """
        node_id = self._resolve_entity(node_name)
        if node_id not in self.graph:
            return set()

        visited = {node_id}
        frontier = {node_id}

        for hop in range(hops):
            new_frontier = set()
            for node in frontier:
                # Both successors and predecessors for richer context
                neighbors = set(self.graph.successors(node)) | set(
                    self.graph.predecessors(node)
                )
                # deterministic cap under the active expansion rule before
                # slicing (set order is hash-seed dependent).
                neighbors = set(
                    self._ordered_neighbors(neighbors)[:MAX_NEIGHBORS_PER_HOP]
                )
                new_frontier |= neighbors - visited

            visited |= new_frontier
            frontier = new_frontier
            logger.debug(f"Hop {hop+1}: {len(frontier)} new nodes, {len(visited)} total")

        return visited

    def get_subgraph_for_entities(
        self, entity_names: list[str], hops: int = 1
    ) -> nx.MultiDiGraph:
        """
        Get the induced subgraph for a list of entities with neighborhood expansion.

        Args:
            entity_names: List of entity names to center on.
            hops: Hop depth for each entity.

        Returns:
            NetworkX MultiDiGraph subgraph.
        """
        all_nodes = set()
        for name in entity_names:
            all_nodes |= self.expand_neighborhood(name, hops=hops)

        # Only keep nodes that exist in the graph
        valid_nodes = [n for n in all_nodes if n in self.graph]
        return self.graph.subgraph(valid_nodes).copy()

    # Graph Statistics

    def compute_graph_density(self) -> float:
        """
        Compute graph density.

        Density = 2E / V(V - 1)
        """
        if self._density is not None:
            return self._density

        V = self.graph.number_of_nodes()
        E = self.graph.number_of_edges()
        if V <= 1:
            return 0.0
        # directed-graph density is E / V(V-1);
        # the undirected 2E / V(V-1) formula overcounts by 2x here.
        self._density = E / (V * (V - 1))
        return self._density

    def get_node_degree(self, node_name: str) -> int:
        """Get degree of a node (total in + out degree)."""
        node_id = self._resolve_entity(node_name)
        if node_id not in self._degree_cache:
            if node_id in self.graph:
                self._degree_cache[node_id] = self.graph.degree(node_id)
            else:
                self._degree_cache[node_id] = 0
        return self._degree_cache[node_id]

    def get_triples_for_entities(
        self, entity_names: list[str]
    ) -> list[tuple[str, str, str]]:
        """
        Get all (head, relation, tail) triples involving the given entities.

        Args:
            entity_names: List of entity names.

        Returns:
            List of (head, relation, tail) tuples.
        """
        triples = []
        seen = set()
        for name in entity_names:
            node_id = self._resolve_entity(name)
            if node_id not in self.graph:
                continue
            # graph.edges(node) yields OUT-edges only on
            # a MultiDiGraph; include in-edges so facts where the entity is the
            # tail (e.g., a disease treated by a drug) are not lost. Capped at
            # 50 per direction to bound hub explosion.
            edge_iter = list(self.graph.out_edges(node_id, data=True))
            if hasattr(self.graph, "in_edges"):
                edge_iter += list(self.graph.in_edges(node_id, data=True))
            for count, (u, v, data) in enumerate(edge_iter):
                if count >= 100:
                    break
                key = (u, data.get("relation", ""), v)
                if key not in seen:
                    triples.append(key)
                    seen.add(key)
        return triples

    def summarize(self) -> dict:
        """Return summary statistics for the hypergraph structure."""
        V = self.graph.number_of_nodes()
        E = self.graph.number_of_edges()
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("node_type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1

        return {
            "num_nodes": V,
            "num_edges": E,
            "density": self.compute_graph_density(),
            "node_type_distribution": node_types,
        }
