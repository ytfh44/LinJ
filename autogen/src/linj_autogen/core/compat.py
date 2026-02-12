"""
Backward compatibility layer for autogen core components

This module re-exports shared components to maintain backward compatibility.
Existing code can continue to import from here while we migrate to shared components.
"""

# Re-export shared components with local names
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    # Import from shared components
    from shared.core.document import (
        LinJDocument,
        Policies,
        Loop,
        Placement,
        Requirements,
    )
    from shared.core.nodes import Node, NodeType, HintNode, ToolNode, JoinNode, GateNode
    from shared.core.edges import Edge, DependencyGraph, EdgeKind
    from shared.core.state import StateManager
    from shared.core.changeset import ChangeSet, ChangeSetBuilder
    from shared.core.errors import (
        ExecutionError,
        ValidationError,
        ContractViolation,
        InvalidRequirements,
        InvalidPlacement,
        ResourceConstraintUnsatisfied,
    )
    from shared.core.contract_validator import ContractValidator
    from shared.core.tracing import DiagnosticTracer, TraceEntry, TracingMixin
    from shared.core.path import PathResolver

    # Re-export for backward compatibility
    __all__ = [
        "LinJDocument",
        "Policies",
        "Loop",
        "Placement",
        "Requirements",
        "Node",
        "NodeType",
        "HintNode",
        "ToolNode",
        "JoinNode",
        "GateNode",
        "Edge",
        "DependencyGraph",
        "EdgeKind",
        "StateManager",
        "ChangeSet",
        "ChangeSetBuilder",
        "ExecutionError",
        "ValidationError",
        "ContractViolation",
        "InvalidRequirements",
        "InvalidPlacement",
        "ResourceConstraintUnsatisfied",
        "ContractValidator",
        "DiagnosticTracer",
        "TraceEntry",
        "TracingMixin",
        "PathResolver",
    ]

except ImportError as e:
    print(f"Warning: Could not import shared components: {e}")
    # Fallback: keep local imports available
    from .document import LinJDocument, Policies, Loop, Placement, Requirements
    from .nodes import Node, NodeType, HintNode, ToolNode, JoinNode, GateNode
    from .edges import Edge, DependencyGraph, EdgeKind
    from .state import StateManager
    from .changeset import ChangeSet, ChangeSetBuilder
    from .errors import (
        ExecutionError,
        ValidationError,
        ContractViolation,
        InvalidRequirements,
        InvalidPlacement,
        ResourceConstraintUnsatisfied,
    )
    from .contract_validator import ContractValidator
    from .tracing import DiagnosticTracer, TraceEntry, TracingMixin
    from .path import PathResolver

    __all__ = [
        "LinJDocument",
        "Policies",
        "Loop",
        "Placement",
        "Requirements",
        "Node",
        "NodeType",
        "HintNode",
        "ToolNode",
        "JoinNode",
        "GateNode",
        "Edge",
        "DependencyGraph",
        "EdgeKind",
        "StateManager",
        "ChangeSet",
        "ChangeSetBuilder",
        "ExecutionError",
        "ValidationError",
        "ContractViolation",
        "InvalidRequirements",
        "InvalidPlacement",
        "ResourceConstraintUnsatisfied",
        "ContractValidator",
        "DiagnosticTracer",
        "TraceEntry",
        "TracingMixin",
        "PathResolver",
    ]
