"""
LinJ Document Model

Implements LinJ document structure defined in Section 4
"""

import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from .nodes import Node, parse_node, NodeType
from .edges import Edge, DependencyGraph, EdgeKind
from .errors import (
    ValidationError,
    InvalidRequirements,
    InvalidPlacement,
    ResourceConstraintUnsatisfied,
)


class Policies(BaseModel):
    """
    Global Policies (Section 10.1)
    """

    max_steps: Optional[int] = None
    max_rounds: Optional[int] = None
    timeout_ms: Optional[int] = None
    retry: Optional[Dict[str, Any]] = None
    max_array_length: Optional[int] = None
    max_local_state_bytes: Optional[int] = None


class Loop(BaseModel):
    """
    Explicit Loop (Section 12)
    """

    id: str
    entry: str
    members: List[str]
    mode: str = "finite"  # finite or infinite
    stop_condition: Optional[str] = None
    max_rounds: Optional[int] = None


class Placement(BaseModel):
    """
    Placement Declaration (Section 15.2)
    """

    target: str  # node id or resource_name
    domain: str  # execution domain label


class Requirements(BaseModel):
    """
    Runtime Requirements (Section 15.1)
    """

    allow_parallel: bool = False
    allow_child_units: bool = False
    require_resume: bool = False


class LinJDocument(BaseModel):
    """
    LinJ Document (Section 4.1)

    Must contain:
    - linj_version: version number
    - nodes: node array
    - edges: dependency edge array
    """

    linj_version: str
    nodes: List[NodeType] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    loops: Optional[List[Loop]] = None
    policies: Optional[Policies] = None
    requirements: Optional[Requirements] = None
    placement: Optional[List[Placement]] = None

    @field_validator("nodes", mode="before")
    @classmethod
    def parse_nodes(cls, v):
        """Parse node list"""
        if v is None:
            return []
        return [parse_node(node) if isinstance(node, dict) else node for node in v]

    @field_validator("linj_version")
    @classmethod
    def validate_version(cls, v):
        """Validate version format"""
        parts = v.split(".")
        if len(parts) != 2:
            raise ValidationError(f"Invalid version format: {v}. Expected: major.minor")
        try:
            int(parts[0])
            int(parts[1])
        except ValueError:
            raise ValidationError(f"Invalid version numbers: {v}")
        return v

    def get_major_version(self) -> int:
        """Get major version"""
        return int(self.linj_version.split(".")[0])

    def get_minor_version(self) -> int:
        """Get minor version"""
        return int(self.linj_version.split(".")[1])

    def check_version_compatibility(
        self, supported_major: int, supported_minor: int
    ) -> bool:
        """
        Check version compatibility (Section 4.2)

        - Major version mismatch: must refuse to run
        - Major version matches but minor version is higher: can run, but ignore unrecognized fields
        """
        doc_major = self.get_major_version()
        doc_minor = self.get_minor_version()

        if doc_major != supported_major:
            raise ValidationError(
                f"Version mismatch: document requires {self.linj_version}, "
                f"but runtime supports {supported_major}.{supported_minor}"
            )

        # Warn but don't block when minor version is higher
        return doc_minor <= supported_minor

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by id"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_ids(self) -> List[str]:
        """Get all node ids"""
        return [node.id for node in self.nodes]

    def build_dependency_graph(self) -> DependencyGraph:
        """Build dependency graph"""
        return DependencyGraph(self.edges)

    def validate_references(self) -> List[str]:
        """
        Validate reference validity

        Check that all nodes referenced by edges exist
        """
        errors = []
        node_ids = set(self.get_node_ids())

        for edge in self.edges:
            if edge.from_ not in node_ids:
                errors.append(f"Edge references unknown source node: {edge.from_}")
            if edge.to not in node_ids:
                errors.append(f"Edge references unknown target node: {edge.to}")

        return errors

    def validate_loop_constraints(self) -> List[str]:
        """
        Validate loop constraints (Section 11.2)

        Check that implicit loops (loops not declared in loops) have max_rounds
        """
        errors = []

        # Get explicit loops and their members
        explicit_loops = self.loops or []
        loop_members = set()
        for loop in explicit_loops:
            loop_members.update(loop.members)

        # Build dependency graph
        graph = self.build_dependency_graph()
        node_ids = self.get_node_ids()

        # Use DFS to detect cycles
        visited = set()
        stack = []  # Nodes on current path

        def find_cycles(u: str):
            visited.add(u)
            stack.append(u)

            # Get all types of prerequisites (incoming edges are dependencies)
            # Note: In our dependency graph, graph.get_incoming(u) returns edges pointing to u
            # But to detect cycles, we need to see node a -> node b -> node a
            # Here get_outgoing(u) returns edges outgoing from u
            for edge in graph.get_outgoing(u):
                v = edge.to
                if v in stack:
                    # Cycle found: v -> u -> v
                    cycle_nodes = stack[stack.index(v) :]

                    # Check if this cycle is already covered by an explicit loop
                    is_covered = False
                    for loop in explicit_loops:
                        # If all nodes in the cycle are in the explicit loop's members, it's covered
                        if all(node in loop.members for node in cycle_nodes):
                            is_covered = True
                            break

                    if not is_covered:
                        # Implicit cycle, must have max_rounds
                        if not self.policies or not self.policies.max_rounds:
                            cycle_str = " -> ".join(cycle_nodes + [v])
                            errors.append(
                                f"Implicit cycle detected: {cycle_str}. "
                                f"Implicit cycles must have max_rounds policy."
                            )
                elif v not in visited:
                    find_cycles(v)

            stack.pop()

        for node_id in node_ids:
            if node_id not in visited:
                find_cycles(node_id)

        return errors

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinJDocument":
        """Create document from dictionary"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.model_dump(by_alias=True, exclude_none=True)


logger = logging.getLogger(__name__)


def validate_resource_constraints(
    doc: "LinJDocument",
    edges: Optional[List[Edge]] = None,
    available_domains: Optional[Set[str]] = None,
) -> List[ValidationError]:
    """
    Validate resource domain constraints are satisfied (Sections 15, 25)

    Validates:
    1. Whether requirements field is boolean
    2. Whether same-domain constraints declared in placement can be satisfied
    3. Whether nodes depended on by kind=resource can be scheduled to the same domain

    Args:
        doc: LinJ document object
        edges: Dependency edge list (optional, defaults to doc.edges)
        available_domains: Available execution domain set (optional, used for simulating test environment)

    Returns:
        Validation error list (empty list means validation passed)
    """
    errors: List[ValidationError] = []

    if edges is None:
        edges = doc.edges

    # 1. Validate requirements field (Section 15.1)
    if doc.requirements:
        req_errors = _validate_requirements(doc.requirements)
        errors.extend(req_errors)

    # 2. Validate placement declarations (Section 15.2)
    if doc.placement:
        placement_errors = _validate_placement(
            doc, doc.placement, edges, available_domains
        )
        errors.extend(placement_errors)

    # 3. Validate kind=resource dependencies (Section 25)
    resource_errors = _validate_resource_dependencies(doc, edges, available_domains)
    errors.extend(resource_errors)

    return errors


def _validate_requirements(req: Requirements) -> List[InvalidRequirements]:
    """
    Validate that requirements field is boolean

    Args:
        req: Requirements object

    Returns:
        Error list
    """
    errors: List[InvalidRequirements] = []

    # Check standard fields
    standard_fields = ["allow_parallel", "allow_child_units", "require_resume"]
    for field in standard_fields:
        value = getattr(req, field, None)
        if value is not None and not isinstance(value, bool):
            errors.append(
                InvalidRequirements(
                    f"requirements.{field} must be boolean, got {type(value).__name__}",
                    details={"field": field, "value": value, "expected_type": "bool"},
                )
            )

    return errors


def _validate_placement(
    doc: "LinJDocument",
    placement: List[Placement],
    edges: List[Edge],
    available_domains: Optional[Set[str]] = None,
) -> List[InvalidPlacement]:
    """
    Validate same-domain constraints for placement declarations

    Args:
        doc: LinJ document object
        placement: Placement declaration list
        edges: Dependency edge list
        available_domains: Available execution domain set

    Returns:
        Error list
    """
    errors: List[InvalidPlacement] = []
    node_ids = set(doc.get_node_ids())

    # Group placement entries by domain
    domain_targets: Dict[str, Set[str]] = {}

    for p in placement:
        # Validate target is valid
        if p.target not in node_ids and not _is_valid_resource_name(p.target):
            errors.append(
                InvalidPlacement(
                    f"Invalid placement target: {p.target}. Must be a node id or resource_name",
                    details={"target": p.target, "domain": p.domain},
                )
            )
            continue

        # Collect all targets for the same domain
        if p.domain not in domain_targets:
            domain_targets[p.domain] = set()
        domain_targets[p.domain].add(p.target)

    # Validate that targets in the same domain can all run in the same execution domain
    for domain, targets in domain_targets.items():
        # Check if any nodes cannot coexist with other nodes in the same domain
        conflict_info = _check_domain_conflicts(targets, edges)
        if conflict_info:
            errors.append(
                InvalidPlacement(
                    f"Placement conflict in domain '{domain}': {conflict_info}",
                    details={
                        "domain": domain,
                        "targets": list(targets),
                        "conflict": conflict_info,
                    },
                )
            )

    return errors


def _validate_resource_dependencies(
    doc: "LinJDocument", edges: List[Edge], available_domains: Optional[Set[str]] = None
) -> List[ResourceConstraintUnsatisfied]:
    """
    Validate that nodes depended on by kind=resource can be scheduled to the same domain (Section 25)

    Args:
        doc: LinJ document object
        edges: Dependency edge list
        available_domains: Available execution domain set

    Returns:
        Error list
    """
    errors: List[ResourceConstraintUnsatisfied] = []
    node_ids = set(doc.get_node_ids())

    # Group edges by resource_name
    resource_edges: Dict[str, List[Edge]] = {}
    for edge in edges:
        if edge.is_resource() and edge.resource_name:
            if edge.resource_name not in resource_edges:
                resource_edges[edge.resource_name] = []
            resource_edges[edge.resource_name].append(edge)

    # Check if dependency nodes for each resource can coexist
    for resource_name, resource_deps in resource_edges.items():
        # Collect all nodes using the same resource
        nodes_in_resource: Set[str] = set()
        for edge in resource_deps:
            if edge.from_ in node_ids:
                nodes_in_resource.add(edge.from_)
            if edge.to in node_ids:
                nodes_in_resource.add(edge.to)

        # Check if these nodes have conflicts (cannot run in the same execution domain)
        if len(nodes_in_resource) > 1:
            # Directly check if these nodes have mutual dependencies
            conflict_info = _check_resource_conflicts(nodes_in_resource, edges)
            if conflict_info:
                errors.append(
                    ResourceConstraintUnsatisfied(
                        f"Resource '{resource_name}' depends on nodes that cannot share execution domain: {conflict_info}",
                        details={
                            "resource_name": resource_name,
                            "nodes": list(nodes_in_resource),
                            "conflict": conflict_info,
                        },
                    )
                )

    return errors


def _check_resource_conflicts(targets: Set[str], edges: List[Edge]) -> Optional[str]:
    """
    Check if there are conflicts (mutual dependencies) between nodes depended on by resource

    Args:
        targets: Target set
        edges: Dependency edge list

    Returns:
        Conflict description, None if no conflict
    """
    if len(targets) <= 1:
        return None

    # Build dependency graph
    graph = DependencyGraph(edges)

    targets_list = list(targets)
    for i, target_a in enumerate(targets_list):
        for target_b in targets_list[i + 1 :]:
            # Check if two targets have mutual dependencies (cycle)
            if _has_mutual_dependency(graph, target_a, target_b):
                return f"mutual dependency between {target_a} and {target_b}"

    return None


def _is_valid_resource_name(name: str) -> bool:
    """
    Check if it's a valid resource_name

    Args:
        name: Name to check

    Returns:
        Whether it's a valid resource_name
    """
    # resource_name should start with a letter, can contain letters, numbers, underscores, hyphens
    if not name or not isinstance(name, str):
        return False
    return len(name) > 0 and name[0].isalpha()


def _check_domain_conflicts(targets: Set[str], edges: List[Edge]) -> Optional[str]:
    """
    Check if a group of targets has conflicts that prevent coexistence

    Args:
        targets: Target set (node id or resource_name)
        edges: Dependency edge list

    Returns:
        Conflict description, None if no conflict
    """
    if len(targets) <= 1:
        return None

    # Build dependency graph
    graph = DependencyGraph(edges)

    # Check for circular dependencies (cycles in the same execution domain will cause problems)
    targets_list = list(targets)
    for i, target_a in enumerate(targets_list):
        for target_b in targets_list[i + 1 :]:
            # Check if two targets have mutual dependencies (cycle)
            if _has_mutual_dependency(graph, target_a, target_b):
                return f"mutual dependency between {target_a} and {target_b}"

    return None


def _has_mutual_dependency(graph: DependencyGraph, node_a: str, node_b: str) -> bool:
    """
    Check if there is mutual dependency (cycle) between two nodes

    Includes data, control, and resource dependencies

    Args:
        graph: Dependency graph
        node_a: First node
        node_b: Second node

    Returns:
        Whether mutual dependency exists
    """
    # Get all incoming edge source nodes for node_a (including all types)
    incoming_a = [edge.from_ for edge in graph.get_incoming(node_a)]
    # Get all incoming edge source nodes for node_b (including all types)
    incoming_b = [edge.from_ for edge in graph.get_incoming(node_b)]

    # Check for mutual dependency: a depends on b and b depends on a
    # a depends on b means b is in a's incoming edge sources
    # b depends on a means a is in b's incoming edge sources
    return (node_b in incoming_a) and (node_a in incoming_b)
