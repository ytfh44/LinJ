"""
LangGraph Deterministic Scheduler

Based on AutoGen implementation migration, ensuring full compliance with LinJ specification Section 11
Provides scheduling behavior identical to the AutoGen version
"""

import logging
from typing import Any, Dict, List, Optional, Set, Mapping, Tuple
from dataclasses import dataclass

from ..executor.scheduler import (
    BaseScheduler,
    DeterministicScheduler,
    SchedulingDecision,
    SchedulingStrategy,
)
from ..executor.autogen_scheduler import (
    ExecutionDomain,
    DomainAllocator,
    ExecutionState,
    select_next_node,
    get_node_path_set,
    check_path_intersection,
    check_concurrent_safety,
    find_concurrent_groups,
    are_dependencies_satisfied,
)

logger = logging.getLogger(__name__)


class LangGraphDeterministicScheduler(DeterministicScheduler):
    """
    LangGraph Deterministic Scheduler

    Maintains identical scheduling behavior with the AutoGen version:
    - Same deterministic sorting rules (Section 11.3)
    - Same dependency checking logic
    - Same concurrency safety checks
    - Same execution domain handling
    """

    def __init__(self, nodes: List[Any], enable_parallel: bool = False):
        super().__init__(nodes)
        self.enable_parallel = enable_parallel

        # Reuse AutoGen's state management and domain allocation logic
        self._execution_state = ExecutionState()
        self._domain_allocator = DomainAllocator()

        # Domain mapping cache
        self._domain_map: Optional[Mapping[str, ExecutionDomain]] = None

    def _initialize_domain_map(self, document: Any) -> None:
        """Initialize execution domain mapping"""
        if self._domain_map is None:
            try:
                self._domain_map = self._domain_allocator.allocate_domains(
                    document, document.edges if hasattr(document, "edges") else []
                )
            except Exception as e:
                logger.warning(
                    f"Domain allocation failed: {e}, using default allocation"
                )
                self._domain_map = {}

    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: Any,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """
        Select nodes for execution (logic identical to AutoGen)

        Args:
            ready_nodes: List of ready nodes
            context: Execution context (contains document, etc.)
            max_concurrency: Maximum concurrency limit

        Returns:
            Scheduling decision result
        """
        # Initialize domain mapping (if needed)
        if hasattr(context, "document"):
            self._initialize_domain_map(context.document)

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

        # Select execution strategy based on whether parallel is enabled
        if self.enable_parallel and max_concurrency and max_concurrency > 1:
            return self._select_parallel_group(
                executable_nodes, context, max_concurrency
            )
        else:
            return self._select_single_node(executable_nodes, context)

    def _select_single_node(
        self, executable_nodes: List[Any], context: Any
    ) -> SchedulingDecision:
        """Select single node for execution (serial mode)"""
        # Sort by deterministic rules (identical to AutoGen)
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
                "total_ready": len(executable_nodes),
                "executable": len(executable_nodes),
                "selected_rank": getattr(selected_node, "rank", 0),
                "execution_mode": "serial",
            },
        )

    def _select_parallel_group(
        self, executable_nodes: List[Any], context: Any, max_concurrency: int
    ) -> SchedulingDecision:
        """Select group of nodes that can execute in parallel"""
        # Use the same concurrent grouping logic as AutoGen
        concurrent_groups = find_concurrent_groups(executable_nodes, self._domain_map)

        if not concurrent_groups:
            return self._select_single_node(executable_nodes, context)

        # Select the largest parallel group (but not exceeding concurrency limit)
        selected_group = concurrent_groups[0]
        selected_nodes = selected_group[:max_concurrency]

        # Record batch size
        self._stats["batch_sizes"].append(len(selected_nodes))

        return SchedulingDecision(
            selected_nodes=selected_nodes,
            execution_order=[getattr(node, "id", "unknown") for node in selected_nodes],
            concurrency_level=len(selected_nodes),
            strategy=SchedulingStrategy.PARALLEL,
            metadata={
                "total_ready": len(executable_nodes),
                "executable": len(executable_nodes),
                "parallel_groups": len(concurrent_groups),
                "selected_group_size": len(selected_group),
                "max_concurrency": max_concurrency,
                "execution_mode": "parallel",
            },
        )

    def _are_dependencies_satisfied(self, node_id: str, context: Any) -> bool:
        """Check if node dependencies are satisfied (identical to AutoGen)"""
        # Get dependency graph and execution state
        graph = getattr(context, "dependency_graph", None)
        if not graph:
            return True

        # Use the same dependency checking logic as AutoGen
        return are_dependencies_satisfied(node_id, graph, self._execution_state)

    def get_dependencies(self, node: Any) -> List[str]:
        """Get node's dependency list"""
        return getattr(node, "dependencies", [])

    def can_execute(self, node: Any, context: Any) -> bool:
        """
        Check if node can execute (enhanced version)

        Includes domain constraint checking
        """
        node_id = getattr(node, "id", "unknown")

        # Basic check
        if not super().can_execute(node, context):
            return False

        # Domain constraint check
        if self._domain_map and node_id in self._domain_map:
            node_domain = self._domain_map[node_id]

            # Check if there are nodes executing in the same domain
            for executing_id in self._executing:
                if executing_id in self._domain_map:
                    executing_domain = self._domain_map[executing_id]
                    if executing_domain is node_domain:
                        return False

        return True

    def mark_completed(self, node_id: str, success: bool = True) -> None:
        """Mark node execution complete (update execution state)"""
        super().mark_completed(node_id, success)

        # Update execution state
        if success:
            self._execution_state.completed.add(node_id)
        else:
            self._execution_state.failed.add(node_id)

    def reset(self) -> None:
        """Reset scheduler state"""
        super().reset()
        self._execution_state = ExecutionState()
        self._domain_map = None

    def get_domain_info(self) -> Dict[str, Any]:
        """Get execution domain information (for debugging)"""
        if not self._domain_map:
            return {"domains": "not_initialized"}

        domain_info = {}
        for node_id, domain in self._domain_map.items():
            domain_info[node_id] = {
                "domain_label": domain.domain_label,
                "node_count": len(domain.node_ids),
                "resource_names": list(domain.resource_names),
            }

        return {
            "total_domains": len(
                set(d.domain_label for d in self._domain_map.values() if d.domain_label)
            ),
            "domain_mapping": domain_info,
        }
