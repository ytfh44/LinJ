"""
LinJ Dependency Edges Definition

Implements dependency and mapping as defined in Section 8 of the specification
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EdgeKind(str, Enum):
    """Dependency type (Section 8.1)"""

    DATA = "data"
    CONTROL = "control"
    RESOURCE = "resource"


class MapRule(BaseModel):
    """
    Mapping rule (Section 8.2)

    - from_: source path
    - to: target path
    - default: optional default value
    """

    from_: str = Field(alias="from")
    to: str
    default: Optional[Any] = None


class Edge(BaseModel):
    """
    Dependency edge (Section 8.1)

    Must contain:
    - from_: source node id
    - to: target node id
    - kind: data/control/resource
    """

    from_: str = Field(alias="from")
    to: str
    kind: EdgeKind
    weight: float = 1.0
    map_rules: Optional[List[MapRule]] = Field(default=None, alias="map")
    resource_name: Optional[str] = None

    # Compatibility property
    @property
    def map(self) -> Optional[List[MapRule]]:
        """Mapping rule list (compatibility property)"""
        return self.map_rules

    def is_data(self) -> bool:
        return self.kind == EdgeKind.DATA

    def is_control(self) -> bool:
        return self.kind == EdgeKind.CONTROL

    def is_resource(self) -> bool:
        return self.kind == EdgeKind.RESOURCE

    def apply_map(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply mapping rules to state

        Section 8.2: Apply mapping rules before target node execution
        """
        from .path import PathResolver

        if not self.map:
            return {}

        result: Dict[str, Any] = {}
        for rule in self.map:
            source_value = PathResolver.get(state, rule.from_)

            if source_value is not None:
                PathResolver.set(result, rule.to, source_value)
            elif rule.default is not None:
                PathResolver.set(result, rule.to, rule.default)
            # Otherwise this rule is a no-op

        return result


class DependencyGraph:
    """
    Dependency Graph

    Manages dependency relationships between nodes, provides dependency resolution functionality
    """

    def __init__(self, edges: List[Edge]):
        self.edges = edges
        self._build_index()

    def _build_index(self):
        """Build index to accelerate queries"""
        self._incoming: Dict[str, List[Edge]] = {}
        self._outgoing: Dict[str, List[Edge]] = {}

        for edge in self.edges:
            # Incoming edges
            if edge.to not in self._incoming:
                self._incoming[edge.to] = []
            self._incoming[edge.to].append(edge)

            # Outgoing edges
            if edge.from_ not in self._outgoing:
                self._outgoing[edge.from_] = []
            self._outgoing[edge.from_].append(edge)

    def get_incoming(self, node_id: str) -> List[Edge]:
        """Get all incoming edges of a node"""
        return self._incoming.get(node_id, [])

    def get_outgoing(self, node_id: str) -> List[Edge]:
        """Get all outgoing edges of a node"""
        return self._outgoing.get(node_id, [])

    def get_data_dependencies(self, node_id: str) -> List[str]:
        """Get data dependency nodes of a node"""
        return [edge.from_ for edge in self.get_incoming(node_id) if edge.is_data()]

    def get_control_dependencies(self, node_id: str) -> List[str]:
        """Get control dependency nodes of a node"""
        return [edge.from_ for edge in self.get_incoming(node_id) if edge.is_control()]

    def get_data_mapping(self, node_id: str) -> List[Edge]:
        """Get data mapping edges (edges with map) of a node"""
        return [
            edge for edge in self.get_incoming(node_id) if edge.is_data() and edge.map
        ]

    def has_incoming(self, node_id: str) -> bool:
        """Check if a node has incoming edges"""
        return node_id in self._incoming and len(self._incoming[node_id]) > 0

    def has_outgoing(self, node_id: str) -> bool:
        """Check if a node has outgoing edges"""
        return node_id in self._outgoing and len(self._outgoing[node_id]) > 0

    def resolve_map_conflicts(
        self, target_node_id: str, current_maps: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Resolve multiple incoming edge mapping conflicts for the same target node (Section 8.3, 22.2)

        When multiple incoming edges exist for the same target node and write to the same or intersecting paths,
        they must be applied according to the following deterministic priority:

        1. Higher weight wins
        2. Earlier in edges array wins
        3. If still tied, lexicographic order by (from, to) wins

        Args:
            target_node_id: target node ID
            current_maps: current mapping list, each item contains 'from', 'to', 'default', 'weight', 'from_node', 'edge_index'
                If None, collect mapping rules from incoming edges

        Returns:
            Conflict-resolved mapping list, each item contains mapping info and coverage reason

        Raises:
            None (uses deterministic merge, no errors raised)
        """
        from .path import PathResolver

        # Collect all mapping rules
        if current_maps is None:
            current_maps = []
            incoming_edges = self.get_incoming(target_node_id)
            for edge_idx, edge in enumerate(incoming_edges):
                if edge.map:
                    for rule in edge.map:
                        current_maps.append(
                            {
                                "from": rule.from_,
                                "to": rule.to,
                                "default": rule.default,
                                "weight": edge.weight,
                                "from_node": edge.from_,
                                "edge_index": edge_idx,
                                "edge_weight": edge.weight,
                            }
                        )

        if len(current_maps) <= 1:
            return current_maps

        # Sort by priority: weight desc -> edge_index asc -> (from, to) lexicographic
        def sort_key(item: Dict[str, Any]) -> Tuple[float, int, str, str]:
            return (
                -item["weight"],  # weight desc
                item["edge_index"],  # edge_index asc
                item["from"],  # from lexicographic
                item["to"],  # to lexicographic
            )

        sorted_maps = sorted(current_maps, key=sort_key)

        # Detect intersecting paths and resolve conflicts
        # Strategy: Higher priority items come first in the sorted list
        # For intersecting paths, only keep the highest priority (first one)
        resolved_maps: List[Dict[str, Any]] = []
        covered_maps: List[Dict[str, Any]] = []

        for i, current in enumerate(sorted_maps):
            is_covered = False

            # Check if it intersects with already resolved paths
            for j, existing in enumerate(resolved_maps):
                if PathResolver.intersect(current["to"], existing["to"]):
                    # Current item has lower priority than resolved item, is covered
                    is_covered = True

                    # Determine coverage reason
                    if existing["weight"] > current["weight"]:
                        cover_reason = (
                            f"weight_higher:{existing['weight']}>{current['weight']}"
                        )
                    elif existing["weight"] == current["weight"]:
                        if existing["edge_index"] < current["edge_index"]:
                            cover_reason = f"edge_order:{existing['edge_index']}<{current['edge_index']}"
                        else:
                            cover_reason = f"lexicographic:({existing['from']},{existing['to']})<({current['from']},{current['to']})"
                    else:
                        cover_reason = f"lexicographic:({existing['from']},{existing['to']})<({current['from']},{current['to']})"

                    # Record covered mapping
                    covered_maps.append(
                        {
                            **current,
                            "covered_by": existing["from_node"],
                            "cover_reason": cover_reason,
                            "original_index": i,
                        }
                    )
                    break

            if not is_covered:
                resolved_maps.append(current)

        # Record diagnostic info
        if covered_maps:
            for covered in covered_maps:
                logger.info(
                    "Map rule covered during conflict resolution",
                    extra={
                        "target_node": target_node_id,
                        "covered_from": covered["from"],
                        "covered_to": covered["to"],
                        "covered_by": covered["covered_by"],
                        "cover_reason": covered["cover_reason"],
                    },
                )

        return resolved_maps


@dataclass
class MapConflictInfo:
    """
    Mapping conflict information

    Used to record detailed information during conflict resolution
    """

    target_node: str
    covered_map: Dict[str, Any]
    winning_map: Dict[str, Any]
    reason: str  # weight, edge_order, lexicographic


@dataclass
class MapResolutionResult:
    """
    Mapping conflict resolution result
    """

    resolved_maps: List[Dict[str, Any]]
    conflicts: List[MapConflictInfo]


def resolve_map_conflicts(
    edges: List[Edge],
    target_node_id: Optional[str] = None,
    record_diagnostics: bool = True,
) -> MapResolutionResult:
    """
    Resolve multiple incoming edge mapping conflicts (Section 8.3)

    Default behavior: raise ConflictError
    If deterministic override is supported, apply priority:
    1. Higher weight wins
    2. Earlier in edges array wins
    3. If still tied, lexicographic order by (from, to) wins

    Args:
        edges: list of edges
        target_node_id: target node ID (optional)
        record_diagnostics: whether to record diagnostic info

    Returns:
        MapResolutionResult: contains resolved mappings and conflict info
    """
    # Build dependency graph
    graph = DependencyGraph(edges)

    # Collect all mapping rules
    all_maps: List[Dict[str, Any]] = []
    for edge_idx, edge in enumerate(edges):
        if edge.map:
            for rule in edge.map:
                all_maps.append(
                    {
                        "from": rule.from_,
                        "to": rule.to,
                        "default": rule.default,
                        "weight": edge.weight,
                        "from_node": edge.from_,
                        "edge_index": edge_idx,
                        "edge_weight": edge.weight,
                    }
                )

    if target_node_id:
        # Only resolve conflicts for target node
        resolved = graph.resolve_map_conflicts(target_node_id, all_maps)
    else:
        # Resolve all node conflicts (simplified handling)
        resolved = all_maps

    # Record diagnostic info
    conflicts: List[MapConflictInfo] = []
    if record_diagnostics:
        # Analyze covered rules
        all_map_set = set((m["from"], m["to"]) for m in all_maps)
        resolved_set = set((m["from"], m["to"]) for m in resolved)

        for om in all_maps:
            if (om["from"], om["to"]) not in resolved_set:
                # Find the winning rule
                for rm in resolved:
                    if rm["from"] == om["from"] and rm["to"] == om["to"]:
                        # Determine coverage reason
                        if om["weight"] < rm["weight"]:
                            reason = "weight"
                        elif (
                            om["weight"] == rm["weight"]
                            and om["edge_index"] > rm["edge_index"]
                        ):
                            reason = "edge_order"
                        else:
                            reason = "lexicographic"

                        conflicts.append(
                            MapConflictInfo(
                                target_node=target_node_id or "",
                                covered_map=om,
                                winning_map=rm,
                                reason=reason,
                            )
                        )
                        break

    return MapResolutionResult(resolved_maps=resolved, conflicts=conflicts)
