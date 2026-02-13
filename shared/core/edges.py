"""
LinJ dependency edge definitions

Implements dependencies and mappings defined in Section 8 of the specification
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Tuple

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class EdgeKind(str, Enum):
    """Dependency type (Section 8.1)"""

    DATA = "data"
    CONTROL = "control"
    RESOURCE = "resource"


class MapRule(BaseModel):
    """
    Mapping rules (Section 8.2)

    - from: source path
    - to: target path
    - default: optional default value
    """

    model_config = {"populate_by_name": True}

    from_: str = Field(default="", validation_alias="from")
    to: str = ""
    default: Optional[Any] = None


class Edge(BaseModel):
    """
    Dependency edge (Section 8.1)

    Must contain:
    - from: source node id
    - to: target node id
    - kind: data/control/resource
    """

    model_config = {"populate_by_name": True}

    from_: str = Field(default="", validation_alias="from")
    to: str = ""
    kind: EdgeKind = EdgeKind.DATA
    weight: float = 1.0
    map_rules: Optional[List[MapRule]] = Field(default=None, validation_alias="map")
    resource_name: Optional[str] = None

    # compatibility property
    @property
    def map(self) -> Optional[List[MapRule]]:
        """Mapping rules list (compatibility property)"""
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
    Dependency graph

    Manages dependencies between nodes, provides dependency resolution
    """

    def __init__(self, edges: List[Edge]):
        self.edges = edges
        self._build_index()

    def _build_index(self):
        """Build index for faster queries"""
        self._incoming: Dict[str, List[Edge]] = {}
        self._outgoing: Dict[str, List[Edge]] = {}

        for edge in self.edges:
            # incoming edge
            if edge.to not in self._incoming:
                self._incoming[edge.to] = []
            self._incoming[edge.to].append(edge)

            # outgoing edge
            if edge.from_ not in self._outgoing:
                self._outgoing[edge.from_] = []
            self._outgoing[edge.from_].append(edge)

    def get_incoming(self, node_id: str) -> List[Edge]:
        """Get all incoming edges for a node"""
        return self._incoming.get(node_id, [])

    def get_outgoing(self, node_id: str) -> List[Edge]:
        """Get all outgoing edges for a node"""
        return self._outgoing.get(node_id, [])

    def get_data_dependencies(self, node_id: str) -> List[str]:
        """Get data dependency nodes for a node"""
        return [edge.from_ for edge in self.get_incoming(node_id) if edge.is_data()]

    def get_control_dependencies(self, node_id: str) -> List[str]:
        """Get control dependency nodes for a node"""
        return [edge.from_ for edge in self.get_incoming(node_id) if edge.is_control()]

    def get_data_mapping(self, node_id: str) -> List[Edge]:
        """Get data mapping edges for a node (edges with map)"""
        return [
            edge for edge in self.get_incoming(node_id) if edge.is_data() and edge.map
        ]

    def has_incoming(self, node_id: str) -> bool:
        """Check if node has incoming edges"""
        return node_id in self._incoming and len(self._incoming[node_id]) > 0

    def has_outgoing(self, node_id: str) -> bool:
        """Check if node has outgoing edges"""
        return node_id in self._outgoing and len(self._outgoing[node_id]) > 0

    def resolve_map_conflicts(
        self, target_node_id: str, current_maps: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Resolve mapping conflicts for multiple incoming edges to the same target node (Section 8.3, 22.2)

        When multiple incoming edges map to the same target node and write to the same or intersecting paths,
        they must be applied in the following deterministic priority order:

        1. Higher weight first
        2. Earlier position in edges array first
        3. If still tied, lexicographic order by (from, to)

        Args:
            target_node_id: target node ID
            current_maps: current mapping list, each item contains 'from', 'to', 'default', 'weight', 'from_node', 'edge_index'
                If None, collects mapping rules from incoming edges

        Returns:
            Resolved mapping list after conflict resolution, each item contains mapping info and coverage reason

        Raises:
            None (uses deterministic merge, no errors)
        """
        from .path import PathResolver

        # collect all mapping rules
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

        # sort by priority: weight descending -> edge_index ascending -> (from, to) lexicographic
        def sort_key(item: Dict[str, Any]) -> Tuple[float, int, str, str]:
            return (
                -item["weight"],  # weight descending
                item["edge_index"],  # edge_index ascending
                item["from"],  # from lexicographic
                item["to"],  # to lexicographic
            )

        sorted_maps = sorted(current_maps, key=sort_key)

        # detect intersecting paths and resolve conflicts
        # strategy: in sorted list, higher priority comes first
        # for intersecting paths, only keep the highest priority (first one)
        resolved_maps: List[Dict[str, Any]] = []
        covered_maps: List[Dict[str, Any]] = []

        for i, current in enumerate(sorted_maps):
            is_covered = False

            # check if intersects with already resolved paths
            for j, existing in enumerate(resolved_maps):
                if PathResolver.intersect(current["to"], existing["to"]):
                    # current item has lower priority than resolved item, is covered
                    is_covered = True

                    # determine coverage reason
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

                    # record covered mapping
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

        # record diagnostics
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

    Default behavior: produces ConflictError
    """
    # Simple implementation for now
    return MapResolutionResult(resolved_maps=[], conflicts=[])
