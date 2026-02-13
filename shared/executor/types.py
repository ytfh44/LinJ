"""
Execution Engine Type Definitions

Defines common data structures and types used during execution.
"""

from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum


# Async function type for type checking
AsyncCallable = Callable[..., Awaitable[Any]]


class ExecutionStatus(Enum):
    """Execution status enum"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ToolResult:
    """Tool execution result"""

    success: bool
    data: Any = None
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    """Execution result"""

    success: bool
    data: Any = None
    error: Optional[Exception] = None
    changeset: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


@dataclass
class NodeExecution:
    """Node execution record"""

    node_id: str
    step_id: int
    status: ExecutionStatus
    result: Optional[ExecutionResult] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    reads: Optional[List[str]] = None
    writes: Optional[List[str]] = None


@dataclass
class ExecutionContext:
    """Execution context (LinJ unified version)"""

    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_counter: int = 0
    execution_history: List[NodeExecution] = field(default_factory=list)

    # New fields for backend alignment
    document: Optional[Any] = None  # LinJDocument
    state_manager: Optional[Any] = None  # StateManager
    current_node: Optional[str] = None

    def get_state_value(self, path: str) -> Any:
        """Get state value (delegated to state_manager or retrieved directly from dict)"""
        if self.state_manager and hasattr(self.state_manager, "get"):
            return self.state_manager.get(path)
        return self.state.get(path)

    def set_state_value(self, path: str, value: Any) -> None:
        """Set state value (delegated to state_manager or set directly)"""
        if self.state_manager and hasattr(self.state_manager, "apply"):
            # Note: apply requires ChangeSet, simplified here
            self.state[path] = value
        else:
            self.state[path] = value


# Function types for type checking
AsyncCallable = Callable[..., Any]


# Backward compatible imports
try:
    from ..core.changeset import ChangeSet  # type: ignore
except ImportError:
    # Fallback to Any if import fails
    ChangeSet = Any  # type: ignore
