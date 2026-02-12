"""
LinJ Autogen - LinJ/ContiText 的 Autogen 实现

支持：
- 作为 Autogen 插件导入使用
- 独立运行 LinJ 工作流
"""

__version__ = "0.1.0"

# For now, keep original core imports to maintain compatibility
# Future update: migrate to shared components
from .core import (
    # Errors
    LinJError,
    ValidationError,
    ExecutionError,
    MappingError,
    ConditionError,
    ConflictError,
    # Core classes
    LinJDocument,
    Node,
    HintNode,
    ToolNode,
    JoinNode,
    GateNode,
    Edge,
    ChangeSet,
    ChangeSetBuilder,
    PathResolver,
    StateManager,
)
from .executor.runner import LinJExecutor, load_document
from .executor.evaluator import evaluate_condition

__all__ = [
    "__version__",
    # Errors
    "LinJError",
    "ValidationError",
    "ExecutionError",
    "MappingError",
    "ConditionError",
    "ConflictError",
    # Core
    "LinJDocument",
    "Node",
    "HintNode",
    "ToolNode",
    "JoinNode",
    "GateNode",
    "Edge",
    "ChangeSet",
    "ChangeSetBuilder",
    "PathResolver",
    "StateManager",
    # Executor
    "LinJExecutor",
    "load_document",
    "evaluate_condition",
]
