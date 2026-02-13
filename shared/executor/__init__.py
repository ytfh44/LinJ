"""
Shared Execution Engine Abstractions

Provides abstract interfaces for multi-backend architecture, supporting different execution engine implementations.
"""

# Core Abstract Interfaces
from .backend import ExecutionBackend
from .adapter import ToolAdapter
from .scheduler import Scheduler, SchedulingStrategy, SchedulingDecision
from .evaluator import Evaluator, EvaluationResult, EvaluationStrategy
from .context import ExecutionContext, StateManager, ContextManager, PathResolver
from .types import ExecutionResult, ToolResult, NodeExecution, ExecutionStatus

# AutoGen Compatible Implementations
from .autogen_scheduler import (
    ExecutionState,
    AutoGenDeterministicScheduler,
    DomainAllocator,
    select_next_node,
    get_node_path_set,
    check_path_intersection,
    check_concurrent_safety,
    find_concurrent_groups,
    are_dependencies_satisfied,
)

from .autogen_evaluator import AutoGenConditionEvaluator, evaluate_condition

__all__ = [
    # Core interfaces
    "ExecutionBackend",
    "ToolAdapter",
    "Scheduler",
    "Evaluator",
    "ExecutionContext",
    "StateManager",
    "ContextManager",
    "PathResolver",
    # Type definitions
    "ExecutionResult",
    "ToolResult",
    "NodeExecution",
    "ExecutionStatus",
    "SchedulingStrategy",
    "SchedulingDecision",
    "EvaluationResult",
    "EvaluationStrategy",
    # AutoGen Compatible Implementations
    "ExecutionState",
    "AutoGenDeterministicScheduler",
    "AutoGenConditionEvaluator",
    "DomainAllocator",
    # Convenience functions
    "select_next_node",
    "get_node_path_set",
    "check_path_intersection",
    "check_concurrent_safety",
    "find_concurrent_groups",
    "are_dependencies_satisfied",
    "evaluate_condition",
]
