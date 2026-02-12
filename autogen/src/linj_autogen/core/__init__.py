"""LinJ 核心数据模型"""

from .errors import (
    LinJError,
    ValidationError,
    ExecutionError,
    MappingError,
    ConditionError,
    ConflictError,
    HandleExpired,
    ResourceConstraintUnsatisfied,
    ContractViolation,
)
from .path import PathResolver, PathSegment
from .changeset import ChangeSet, ChangeSetBuilder, WriteOp, DeleteOp
from .state import StateManager, StateView
from .nodes import (
    Node,
    HintNode,
    ToolNode,
    JoinNode,
    GateNode,
    NodeType,
    ValueRef,
    Contract,
    NodePolicy,
    ToolCall,
    Effect,
    GlossaryItem,
    parse_node,
)
from .edges import Edge, EdgeKind, MapRule, DependencyGraph
from .document import (
    LinJDocument,
    Policies,
    Loop,
    Placement,
    Requirements,
)
from .contract_validator import (
    ContractValidator,
    ValidationResult,
    ContractValidationError,
)

__all__ = [
    # Errors
    "LinJError",
    "ValidationError",
    "ExecutionError",
    "MappingError",
    "ConditionError",
    "ConflictError",
    "HandleExpired",
    "ResourceConstraintUnsatisfied",
    "ContractViolation",
    # Path
    "PathResolver",
    "PathSegment",
    # ChangeSet
    "ChangeSet",
    "ChangeSetBuilder",
    "WriteOp",
    "DeleteOp",
    # State
    "StateManager",
    "StateView",
    # Nodes
    "Node",
    "HintNode",
    "ToolNode",
    "JoinNode",
    "GateNode",
    "NodeType",
    "ValueRef",
    "Contract",
    "NodePolicy",
    "ToolCall",
    "Effect",
    "GlossaryItem",
    "parse_node",
    # Edges
    "Edge",
    "EdgeKind",
    "MapRule",
    "DependencyGraph",
    # Document
    "LinJDocument",
    "Policies",
    "Loop",
    "Placement",
    "Requirements",
    # Contract Validator
    "ContractValidator",
    "ValidationResult",
    "ContractValidationError",
]
