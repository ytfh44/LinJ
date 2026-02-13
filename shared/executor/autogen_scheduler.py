"""
Scheduler implementation

Scheduler implementation migrated and refactored from autogen/executor/scheduler.py, compatible with existing AutoGen scheduling logic.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Mapping, Tuple

# Try importing existing modules for compatibility
try:
    from ..core.nodes import Node
    from ..core.path import PathResolver
    from ..core.document import LinJDocument, Placement
    from ..core.edges import Edge, DependencyGraph, EdgeKind
except ImportError:
    # Fallback to basic types
    Node = Any
    PathResolver = Any
    LinJDocument = Any
    Placement = Any
    Edge = Any
    DependencyGraph = Any
    EdgeKind = Any

from .scheduler import (
    BaseScheduler,
    DeterministicScheduler,
    SchedulingDecision,
    SchedulingStrategy,
)


@dataclass
class ExecutionDomain:
    """
    Execution Domain

    Represents a set of nodes and resources that can be executed together (Section 15.2)
    """

    node_ids: Set[str]
    resource_names: Set[str]
    domain_label: Optional[str] = None


class DomainAllocator:
    """
    Execution Domain Allocator

    Allocates execution domains based on placement declarations and resource dependencies (Section 15.2, Section 25)
    """

    def __init__(self, available_domains: Optional[Set[str]] = None):
        """
        Initialize domain allocator

        Args:
            available_domains: Set of available execution domains (optional, None means no limit)
        """
        self.available_domains = available_domains

    def allocate_domains(
        self, doc: LinJDocument, edges: Optional[List[Edge]] = None
    ) -> Mapping[str, ExecutionDomain]:
        """
        Allocate execution domains for nodes

        Allocation strategy:
        1. First apply same-domain constraints based on placement declarations
        2. Then apply same-domain constraints based on kind=resource dependencies
        3. Remaining nodes are assigned to default domain

        Args:
            doc: LinJ document object
            edges: List of dependency edges (optional, uses doc.edges by default)

        Returns:
            Mapping from node IDs to execution domains
        """
        if edges is None:
            edges = doc.edges

        node_ids = doc.get_node_ids()
        graph = DependencyGraph(edges)

        # Initialize: one domain per node
        domain_map: Dict[str, ExecutionDomain] = {}
        for node_id in node_ids:
            domain_map[node_id] = ExecutionDomain(
                node_ids={node_id}, resource_names=set(), domain_label=None
            )

        # Apply placement same-domain constraints
        domain_map = self._apply_placement_constraints(doc.placement, domain_map)

        # Apply resource dependency same-domain constraints
        domain_map = self._apply_resource_constraints(edges, domain_map, graph)

        return domain_map

    def can_share_domain(self, node_a: str, node_b: str, edges: List[Edge]) -> bool:
        """
        Determine if two nodes can share the same execution domain

        Two nodes can share an execution domain if and only if:
        - There is no mutual dependency between them (cycle)
        - There are no conflicting writes paths between them

        Args:
            node_a: First node
            node_b: Second node
            edges: List of dependency edges

        Returns:
            Whether the nodes can share an execution domain
        """
        graph = DependencyGraph(edges)

        # Check for mutual dependencies
        deps_a = graph.get_data_dependencies(node_a) + graph.get_control_dependencies(
            node_a
        )
        deps_b = graph.get_data_dependencies(node_b) + graph.get_control_dependencies(
            node_b
        )

        if node_b in deps_a and node_a in deps_b:
            return False  # Mutual dependency, cannot be in same domain

        return True

    def _apply_placement_constraints(
        self,
        placement: Optional[List[Placement]],
        domain_map: Dict[str, ExecutionDomain],
    ) -> Dict[str, ExecutionDomain]:
        """
        Apply placement same-domain constraints

        Args:
            placement: List of placement declarations
            domain_map: Current domain mapping

        Returns:
            Domain mapping after applying constraints
        """
        if not placement:
            return domain_map

        # Group by domain
        domain_groups: Dict[str, Set[str]] = {}
        for p in placement:
            if p.target not in domain_map:
                continue

            if p.domain not in domain_groups:
                domain_groups[p.domain] = set()
            domain_groups[p.domain].add(p.target)

        # Merge nodes in same domain
        for domain, targets in domain_groups.items():
            domain_map = self._merge_domains(targets, domain_map, domain_label=domain)

        return domain_map

    def _apply_resource_constraints(
        self,
        edges: List[Edge],
        domain_map: Dict[str, ExecutionDomain],
        graph: DependencyGraph,
    ) -> Dict[str, ExecutionDomain]:
        """
        Apply kind=resource dependency same-domain constraints

        Args:
            edges: List of dependency edges
            domain_map: Current domain mapping
            graph: Dependency graph

        Returns:
            Domain mapping after applying constraints
        """
        # Group edges by resource_name
        resource_edges: Dict[str, List[Edge]] = {}
        for edge in edges:
            if edge.is_resource() and edge.resource_name:
                if edge.resource_name not in resource_edges:
                    resource_edges[edge.resource_name] = []
                resource_edges[edge.resource_name].append(edge)

        # Merge nodes using same resource into same domain
        for resource_name, resource_deps in resource_edges.items():
            nodes_in_resource: Set[str] = set()
            for edge in resource_deps:
                if edge.from_ in domain_map:
                    nodes_in_resource.add(edge.from_)
                if edge.to in domain_map:
                    nodes_in_resource.add(edge.to)

            if len(nodes_in_resource) > 1:
                domain_map = self._merge_domains(
                    nodes_in_resource, domain_map, resource_name
                )

        return domain_map

    def _merge_domains(
        self,
        targets: Set[str],
        domain_map: Dict[str, ExecutionDomain],
        domain_label: Optional[str] = None,
    ) -> Dict[str, ExecutionDomain]:
        """
        Merge a set of targets into the same execution domain

        Args:
            targets: Set of targets to merge
            domain_map: Current domain mapping
            domain_label: Domain label

        Returns:
            Merged domain mapping
        """
        if len(targets) <= 1:
            return domain_map

        targets_list = list(targets)
        first_target = targets_list[0]
        merged_domain = domain_map[first_target]

        # Merge all targets into the first target's domain
        for target in targets_list[1:]:
            target_domain = domain_map[target]

            # Merge node sets
            merged_domain.node_ids.update(target_domain.node_ids)

            # Merge resource sets
            merged_domain.resource_names.update(target_domain.resource_names)

            # Update domain reference for target nodes
            for node_id in target_domain.node_ids:
                domain_map[node_id] = merged_domain

        # Set domain label
        if domain_label:
            merged_domain.domain_label = domain_label

        return domain_map


class ExecutionState:
    """
    Execution State Tracker

    Tracks node execution states, used for dependency resolution
    """

    def __init__(self):
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.cancelled: Set[str] = set()

    def is_terminal(self, node_id: str) -> bool:
        """Check if node has reached terminal state"""
        return (
            node_id in self.completed
            or node_id in self.failed
            or node_id in self.cancelled
        )

    def is_successful(self, node_id: str) -> bool:
        """Check if node completed successfully"""
        return node_id in self.completed


class AutoGenDeterministicScheduler(DeterministicScheduler):
    """
    AutoGen-compatible deterministic scheduler

    Refactored from original AutoGen scheduling logic, maintaining compatibility
    """

    def __init__(self, nodes: List[Node]):
        super().__init__(nodes)
        # Add AutoGen-specific state management
        self._execution_state = ExecutionState()

    def select_nodes(
        self, ready_nodes: List[Any], context, max_concurrency: Optional[int] = None
    ) -> SchedulingDecision:
        """Select nodes for execution (AutoGen-compatible)"""
        # Filter executable nodes
        executable_nodes = []
        executed_this_round = getattr(context, "executed_this_round", set())

        for node in ready_nodes:
            node_id = getattr(node, "id", "unknown")

            # Check if already executing
            if self.is_executing(node_id):
                continue

            # Check if already executed this round
            allow_reenter = getattr(node, "policy", None)
            allow_reenter = (
                getattr(allow_reenter, "allow_reenter", False)
                if allow_reenter
                else False
            )

            if node_id in executed_this_round and not allow_reenter:
                continue

            # Check if prerequisites are satisfied
            if not self._are_dependencies_satisfied(node_id, context):
                continue

            executable_nodes.append(node)

        if not executable_nodes:
            return SchedulingDecision(
                selected_nodes=[],
                execution_order=[],
                concurrency_level=0,
                strategy=SchedulingStrategy.DETERMINISTIC,
                metadata={"reason": "no_executable_nodes"},
            )

        # Sort by deterministic rules
        sorted_nodes = self._sort_deterministically(executable_nodes)
        selected_node = sorted_nodes[0]

        # Record batch size
        self._stats["batch_sizes"].append(1)

        return SchedulingDecision(
            selected_nodes=[selected_node],
            execution_order=[getattr(selected_node, "id", "unknown")],
            concurrency_level=1,
            strategy=SchedulingStrategy.DETERMINISTIC,
            metadata={
                "total_ready": len(ready_nodes),
                "executable": len(executable_nodes),
                "selected_rank": getattr(selected_node, "rank", 0),
            },
        )

    def _are_dependencies_satisfied(self, node_id: str, context) -> bool:
        """Check if node dependencies are satisfied"""
        # Get dependency graph and execution state
        graph = getattr(context, "dependency_graph", None)
        if not graph:
            return True

        # Get all prerequisite nodes
        deps = graph.get_data_dependencies(node_id)
        deps.extend(graph.get_control_dependencies(node_id))

        # Deduplicate
        deps = list(set(deps))

        if not deps:
            return True

        # Check if all dependencies are completed
        return all(self._execution_state.is_terminal(dep) for dep in deps)

    def get_dependencies(self, node: Any) -> List[str]:
        """Get node's dependency list"""
        return getattr(node, "dependencies", [])


def select_next_node(
    ready_nodes: List[Node], node_order: Dict[str, int]
) -> Optional[Node]:
    """
    Select next node in deterministic order (Section 11.3)

    Priority:
    1. Higher rank first (0 if not provided)
    2. Earlier in nodes array first (via node_order dict)
    3. If still tied, by node_id lexicographically

    Args:
        ready_nodes: List of schedulable nodes
        node_order: Node order in original array

    Returns:
        Selected node, or None if no ready nodes
    """
    if not ready_nodes:
        return None

    def sort_key(node: Node) -> Tuple[float, int, str]:
        rank = getattr(node, "rank", 0.0)
        rank = rank if rank is not None else 0.0
        order = node_order.get(getattr(node, "id", "unknown"), float("inf"))
        node_id = getattr(node, "id", "unknown")
        return (-rank, order, node_id)  # Negative rank for descending order

    sorted_nodes = sorted(ready_nodes, key=sort_key)
    return sorted_nodes[0]


def get_node_path_set(node: Node, use_writes: bool = True) -> Set[str]:
    """
    Get node's path set

    If node doesn't declare reads/writes, consider entire state (return {"$"})
    """
    if use_writes:
        paths = getattr(node, "writes", [])
    else:
        paths = getattr(node, "reads", [])

    if not paths:
        # Section 6.1: Missing declaration means read/write entire main state
        return {"$"}

    return set(paths)


def check_path_intersection(paths_a: Set[str], paths_b: Set[str]) -> bool:
    """
    Check if two path sets intersect (Section 11.4)

    Two paths intersect if and only if:
    - One path is a prefix of the other
    - Two paths are exactly the same
    """
    for path_a in paths_a:
        for path_b in paths_b:
            if PathResolver.intersect(path_a, path_b):
                return True
    return False


def check_concurrent_safety(node_a: Node, node_b: Node) -> bool:
    """
    Check if two nodes can be safely executed concurrently (Section 11.5)

    Concurrent safety conditions:
    - Both nodes' writes sets are disjoint
    - Neither node's reads intersect with the other's writes

    Args:
        node_a: First node
        node_b: Second node

    Returns:
        True if safe to execute concurrently, False otherwise
    """
    # Get path sets
    reads_a = get_node_path_set(node_a, use_writes=False)
    writes_a = get_node_path_set(node_a, use_writes=True)
    reads_b = get_node_path_set(node_b, use_writes=False)
    writes_b = get_node_path_set(node_b, use_writes=True)

    # Check writes are disjoint
    if check_path_intersection(writes_a, writes_b):
        return False

    # Check reads don't intersect with other's writes
    if check_path_intersection(reads_a, writes_b):
        return False
    if check_path_intersection(reads_b, writes_a):
        return False

    return True


def find_concurrent_groups(
    nodes: List[Node], domain_map: Optional[Mapping[str, ExecutionDomain]] = None
) -> List[List[Node]]:
    """
    Group nodes such that nodes within each group can be safely executed concurrently

    Uses greedy algorithm: iterate through nodes, try to add node to existing group,
    if cannot join any group, create new group

    Section 11.5 & 25:
    - Nodes within a group must have disjoint writes/reads
    - Nodes within a group must belong to different execution domains
    """
    if not nodes:
        return []

    groups: List[List[Node]] = []

    for node in nodes:
        placed = False
        node_domain = (
            domain_map.get(getattr(node, "id", "unknown")) if domain_map else None
        )

        for group in groups:
            # Check if can join the group
            # 1. Check concurrent safety (path intersection)
            can_join = all(check_concurrent_safety(node, member) for member in group)

            # 2. Check execution domain constraint (same-domain nodes must be serial)
            if can_join and node_domain:
                can_join = all(
                    domain_map.get(getattr(member, "id", "unknown")) is not node_domain
                    for member in group
                )

            if can_join:
                group.append(node)
                placed = True
                break

        if not placed:
            groups.append([node])

    return groups


def are_dependencies_satisfied(
    node_id: str,
    graph,  # DependencyGraph
    exec_state: ExecutionState,
    check_all: bool = True,
) -> bool:
    """
    Check if node dependencies are satisfied

    All data/control prerequisite dependencies of a node must have reached terminal state

    Args:
        node_id: Node ID
        graph: Dependency graph
        exec_state: Execution state
        check_all: Whether to check all dependencies.
                  For cycle entry nodes, if check_all=False, only need to satisfy one dependency (OR semantics)
                  But under LinJ specification, default is still AND semantics.
    """
    # Get all prerequisite nodes
    deps = graph.get_data_dependencies(node_id)
    deps.extend(graph.get_control_dependencies(node_id))

    # Deduplicate
    deps = list(set(deps))

    if not deps:
        return True

    # Simple AND semantics
    if check_all:
        return all(exec_state.is_terminal(dep) for dep in deps)
    else:
        # OR semantics: only need one dependency satisfied
        return any(exec_state.is_terminal(dep) for dep in deps)


# Logger
logger = logging.getLogger(__name__)
