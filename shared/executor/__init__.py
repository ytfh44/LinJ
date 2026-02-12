"""
共享执行引擎抽象层

提供多后端架构的抽象接口，支持不同执行引擎的实现。
"""

# 核心抽象接口
from .backend import ExecutionBackend
from .adapter import ToolAdapter
from .scheduler import Scheduler, SchedulingStrategy, SchedulingDecision
from .evaluator import Evaluator, EvaluationResult, EvaluationStrategy
from .context import ExecutionContext, StateManager, ContextManager, PathResolver
from .types import ExecutionResult, ToolResult, NodeExecution, ExecutionStatus

# AutoGen兼容实现
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
    # 核心接口
    "ExecutionBackend",
    "ToolAdapter",
    "Scheduler",
    "Evaluator",
    "ExecutionContext",
    "StateManager",
    "ContextManager",
    "PathResolver",
    # 类型定义
    "ExecutionResult",
    "ToolResult",
    "NodeExecution",
    "ExecutionStatus",
    "SchedulingStrategy",
    "SchedulingDecision",
    "EvaluationResult",
    "EvaluationStrategy",
    # AutoGen兼容实现
    "ExecutionState",
    "AutoGenDeterministicScheduler",
    "AutoGenConditionEvaluator",
    "DomainAllocator",
    # 便捷函数
    "select_next_node",
    "get_node_path_set",
    "check_path_intersection",
    "check_concurrent_safety",
    "find_concurrent_groups",
    "are_dependencies_satisfied",
    "evaluate_condition",
]
