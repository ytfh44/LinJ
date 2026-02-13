"""
Scheduler Abstract Class

Defines the abstract interface and base implementation for node scheduling, supporting multiple scheduling strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from .types import ExecutionContext, ExecutionStatus


class SchedulingStrategy(Enum):
    """Scheduling strategy enumeration"""

    DETERMINISTIC = "deterministic"  # Deterministic scheduling
    PRIORITY = "priority"  # Priority scheduling
    ROUND_ROBIN = "round_robin"  # Round-robin scheduling
    PARALLEL = "parallel"  # Parallel scheduling
    ADAPTIVE = "adaptive"  # Adaptive scheduling


@dataclass
class SchedulingDecision:
    """Scheduling decision result"""

    selected_nodes: List[Any]  # Selected node list
    execution_order: List[str]  # Execution order
    concurrency_level: int  # Concurrency level
    strategy: SchedulingStrategy
    metadata: Dict[str, Any]  # Additional scheduling information


class Scheduler(ABC):
    """
    Scheduler Abstract Base Class

    Defines the core interface for node scheduling:
    - Node selection and ordering
    - Dependency analysis
    - Concurrency safety checks
    - Execution state management
    """

    @abstractmethod
    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: ExecutionContext,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """
        Select nodes to execute from ready nodes

        Args:
            ready_nodes: List of ready nodes
            context: Execution context
            max_concurrency: Maximum concurrency limit

        Returns:
            Scheduling decision result
        """
        pass

    @abstractmethod
    def can_execute(self, node: Any, context: ExecutionContext) -> bool:
        """
        Check if a node can execute

        Args:
            node: Node object
            context: Execution context

        Returns:
            True if can execute, False otherwise
        """
        pass

    @abstractmethod
    def get_dependencies(self, node: Any) -> List[str]:
        """
        Get the dependency list for a node

        Args:
            node: Node object

        Returns:
            List of dependency node IDs
        """
        pass

    @abstractmethod
    def mark_executing(self, node_id: str) -> None:
        """Mark a node as starting execution"""
        pass

    @abstractmethod
    def mark_completed(self, node_id: str, success: bool = True) -> None:
        """Mark a node as completed execution"""
        pass

    def allocate_step_id(self) -> int:
        """Allocate a step ID"""
        # Default implementation: simple incrementing counter
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        self._step_counter += 1
        return self._step_counter

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        return {
            "total_scheduled": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_batch_size": 0.0,
        }


class BaseScheduler(Scheduler):
    """
    Base Scheduler Implementation

    Provides common scheduling logic and state management functionality
    """

    def __init__(self):
        self._executing: Set[str] = set()
        self._completed: Set[str] = set()
        self._failed: Set[str] = set()
        self._stats = {
            "total_scheduled": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "batch_sizes": [],
        }

    def can_execute(self, node: Any, context: ExecutionContext) -> bool:
        """Check if a node can execute"""
        node_id = getattr(node, "id", "unknown")

        # Check if already executing
        if node_id in self._executing:
            return False

        # Check if already completed
        if node_id in self._completed or node_id in self._failed:
            return False

        # Check if dependencies are satisfied
        dependencies = self.get_dependencies(node)
        for dep_id in dependencies:
            if dep_id not in self._completed:
                return False

        return True

    def mark_executing(self, node_id: str) -> None:
        """Mark a node as starting execution"""
        self._executing.add(node_id)
        self._stats["total_scheduled"] += 1

    def mark_completed(self, node_id: str, success: bool = True) -> None:
        """Mark a node as completed execution"""
        self._executing.discard(node_id)

        if success:
            self._completed.add(node_id)
            self._stats["successful_executions"] += 1
        else:
            self._failed.add(node_id)
            self._stats["failed_executions"] += 1

    def is_executing(self, node_id: str) -> bool:
        """Check if a node is currently executing"""
        return node_id in self._executing

    def is_completed(self, node_id: str) -> bool:
        """Check if a node has completed"""
        return node_id in self._completed

    def is_failed(self, node_id: str) -> bool:
        """Check if a node has failed execution"""
        return node_id in self._failed

    def get_pending_nodes(self) -> Set[str]:
        """Get pending node IDs"""
        return (self._completed | self._failed) - self._executing

    def reset(self) -> None:
        """Reset scheduler state"""
        self._executing.clear()
        self._completed.clear()
        self._failed.clear()
        self._stats = {
            "total_scheduled": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "batch_sizes": [],
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        batch_sizes = self._stats["batch_sizes"]
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0

        return {
            "total_scheduled": self._stats["total_scheduled"],
            "successful_executions": self._stats["successful_executions"],
            "failed_executions": self._stats["failed_executions"],
            "average_batch_size": avg_batch_size,
            "currently_executing": len(self._executing),
            "completed_count": len(self._completed),
            "failed_count": len(self._failed),
        }


class DeterministicScheduler(BaseScheduler):
    """
    Deterministic Scheduler

    Deterministic scheduling based on node priority and document order
    """

    def __init__(self, nodes: List[Any]):
        super().__init__()
        self._node_order = {
            getattr(node, "id", str(i)): i for i, node in enumerate(nodes)
        }
        self._nodes = {
            getattr(node, "id", str(i)): node for i, node in enumerate(nodes)
        }

    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: ExecutionContext,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """Select nodes for execution"""
        # Filter executable nodes
        executable_nodes = [
            node for node in ready_nodes if self.can_execute(node, context)
        ]

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

        # Select the first node (single-threaded execution)
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

    def _sort_deterministically(self, nodes: List[Any]) -> List[Any]:
        """Sort nodes by deterministic rules"""

        def sort_key(node):
            rank = getattr(node, "rank", 0)
            rank = rank if rank is not None else 0
            order = self._node_order.get(getattr(node, "id", "unknown"), float("inf"))
            node_id = getattr(node, "id", "unknown")
            return (-rank, order, node_id)  # Negative rank for descending order

        return sorted(nodes, key=sort_key)

    def get_dependencies(self, node: Any) -> List[str]:
        """Get the dependency list for a node"""
        return getattr(node, "dependencies", [])


class ParallelScheduler(BaseScheduler):
    """
    Parallel Scheduler

    Supports multi-node parallel execution while ensuring safety
    """

    def __init__(self, max_concurrency: int = 4):
        super().__init__()
        self.max_concurrency = max_concurrency

    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: ExecutionContext,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """Select nodes for parallel execution"""
        # Determine maximum concurrency
        actual_max = min(max_concurrency or self.max_concurrency, self.max_concurrency)

        # Filter executable nodes
        executable_nodes = [
            node for node in ready_nodes if self.can_execute(node, context)
        ]

        if not executable_nodes:
            return SchedulingDecision(
                selected_nodes=[],
                execution_order=[],
                concurrency_level=0,
                strategy=SchedulingStrategy.PARALLEL,
                metadata={"reason": "no_executable_nodes"},
            )

        # Group nodes that can execute in parallel
        parallel_groups = self._find_parallel_groups(executable_nodes)

        # Select the first group (execute as many in parallel as possible)
        selected_group = parallel_groups[0] if parallel_groups else executable_nodes[:1]

        # Limit concurrency
        selected_nodes = selected_group[:actual_max]

        # Record batch size
        self._stats["batch_sizes"].append(len(selected_nodes))

        return SchedulingDecision(
            selected_nodes=selected_nodes,
            execution_order=[getattr(node, "id", "unknown") for node in selected_nodes],
            concurrency_level=len(selected_nodes),
            strategy=SchedulingStrategy.PARALLEL,
            metadata={
                "total_ready": len(ready_nodes),
                "executable": len(executable_nodes),
                "parallel_groups": len(parallel_groups),
                "group_size": len(selected_group),
                "max_concurrency": actual_max,
            },
        )

    def _find_parallel_groups(self, nodes: List[Any]) -> List[List[Any]]:
        """Find groups of nodes that can execute in parallel"""
        groups = []

        for node in nodes:
            placed = False
            reads = self._get_node_reads(node)
            writes = self._get_node_writes(node)

            for group in groups:
                # Check if can join this group
                can_join = True

                for member in group:
                    member_reads = self._get_node_reads(member)
                    member_writes = self._get_node_writes(member)

                    # Check for write conflicts
                    if self._has_path_conflict(writes, member_writes):
                        can_join = False
                        break

                    # Check for read-write conflicts
                    if self._has_path_conflict(
                        writes, member_reads
                    ) or self._has_path_conflict(reads, member_writes):
                        can_join = False
                        break

                if can_join:
                    group.append(node)
                    placed = True
                    break

            if not placed:
                groups.append([node])

        return groups

    def _get_node_reads(self, node: Any) -> List[str]:
        """Get node read paths"""
        return getattr(node, "reads", [])

    def _get_node_writes(self, node: Any) -> List[str]:
        """Get node write paths"""
        return getattr(node, "writes", [])

    def _has_path_conflict(self, paths_a: List[str], paths_b: List[str]) -> bool:
        """Check if paths have conflicts"""
        # Simplified implementation: check for identical paths
        # Should implement more complex path intersection check in production
        set_a = set(paths_a)
        set_b = set(paths_b)
        return bool(set_a & set_b)

    def get_dependencies(self, node: Any) -> List[str]:
        """Get the dependency list for a node"""
        return getattr(node, "dependencies", [])


class PriorityScheduler(BaseScheduler):
    """
    Priority Scheduler

    Schedules nodes based on priority
    """

    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: ExecutionContext,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """Select nodes by priority"""
        executable_nodes = [
            node for node in ready_nodes if self.can_execute(node, context)
        ]

        if not executable_nodes:
            return SchedulingDecision(
                selected_nodes=[],
                execution_order=[],
                concurrency_level=0,
                strategy=SchedulingStrategy.PRIORITY,
                metadata={"reason": "no_executable_nodes"},
            )

        # Sort by priority
        sorted_nodes = sorted(
            executable_nodes,
            key=lambda n: (-getattr(n, "priority", 0), getattr(n, "id", "unknown")),
        )

        # Select the highest priority node
        selected_node = sorted_nodes[0]

        self._stats["batch_sizes"].append(1)

        return SchedulingDecision(
            selected_nodes=[selected_node],
            execution_order=[getattr(selected_node, "id", "unknown")],
            concurrency_level=1,
            strategy=SchedulingStrategy.PRIORITY,
            metadata={
                "total_ready": len(ready_nodes),
                "executable": len(executable_nodes),
                "selected_priority": getattr(selected_node, "priority", 0),
            },
        )

    def get_dependencies(self, node: Any) -> List[str]:
        """Get the dependency list for a node"""
        return getattr(node, "dependencies", [])
