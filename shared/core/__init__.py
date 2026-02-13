"""LinJ Core Components Module

Provides core components across multiple backend implementations, including:
- Path resolution and state management
- Node and edge definitions
- Change sets and contract validation
- Document models
"""

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
from .document import LinJDocument, Policies, Loop, Placement, Requirements
from .contract_validator import (
    ContractValidator,
    ValidationResult,
    ContractValidationError,
)
from .tracing import (
    DiagnosticTracer,
    TraceEntry,
    PerformanceMetrics,
    ConflictRecord,
    TracingMixin,
    LogLevel,
)

__all__ = [
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
    # Contract Validation
    "ContractValidator",
    "ValidationResult",
    "ContractValidationError",
    # Tracing
    "DiagnosticTracer",
    "TraceEntry",
    "PerformanceMetrics",
    "ConflictRecord",
    "TracingMixin",
    "LogLevel",
]
