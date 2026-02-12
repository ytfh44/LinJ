"""
LangGraph Type Definitions

This module contains type definitions used throughout the LangGraph implementation.
Provides consistent typing for nodes, workflows, and execution contexts.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Literal,
    Type,
    Generic,
    TypeVar,
    Protocol,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .state import LangGraphStateView

    try:
        from shared.core.changeset import ChangeSet
    except ImportError:
        ChangeSet = Any  # Fallback for now
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

T = TypeVar("T")
NodeType = TypeVar("NodeType")  # Simplified for now


class RetryPolicy(str, Enum):
    """Retry policies for node execution"""

    NONE = "none"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"


class NodeType(str, Enum):
    """Types of nodes in the workflow"""

    DECISION = "decision"
    ACTION = "action"
    CONDITION = "condition"
    TRANSFORM = "transform"
    AGGREGATOR = "aggregator"
    SPLITTER = "splitter"
    JOINER = "joiner"
    CUSTOM = "custom"


@dataclass
class NodeConfig:
    """Configuration for individual nodes"""

    node_id: str
    node_type: NodeType
    description: Optional[str] = None

    # Execution configuration
    timeout: Optional[float] = None
    max_retries: int = 0
    retry_policy: RetryPolicy = RetryPolicy.NONE
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0

    # Conditional execution
    condition: Optional[str] = None
    skip_on_condition_false: bool = True

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    inputs: Dict[str, str] = field(default_factory=dict)  # input_name -> state_path

    # Output configuration
    outputs: Dict[str, str] = field(default_factory=dict)  # output_name -> state_path
    persist_output: bool = True

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowConfig:
    """Configuration for the entire workflow"""

    workflow_id: str
    name: str
    description: Optional[str] = None

    # Execution configuration
    max_steps: Optional[int] = None
    timeout: Optional[float] = None
    parallel_execution: bool = False

    # State management
    state_persistence: bool = True
    checkpoint_interval: int = 10

    # Error handling
    continue_on_node_failure: bool = False
    failure_handling: Literal["stop", "skip", "retry"] = "stop"

    # Logging and monitoring
    log_level: str = "INFO"
    enable_tracing: bool = True
    metrics_collection: bool = True

    # Node configurations
    nodes: Dict[str, NodeConfig] = field(default_factory=dict)

    # Workflow graph definition
    edges: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Execution context for a node"""

    workflow_id: str
    step_id: int
    node_id: str

    # State access
    state_view: Any  # Will be properly typed when BaseNode is created
    input_data: Dict[str, Any]

    # Configuration
    node_config: NodeConfig
    workflow_config: WorkflowConfig

    # Execution metadata
    attempt_number: int = 1
    parent_step_id: Optional[int] = None
    trigger_reason: Optional[str] = None

    # Services and utilities
    logger: Any = None  # Logger instance
    services: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of node execution"""

    success: bool
    node_id: str
    step_id: int

    # Output data
    output_data: Optional[Dict[str, Any]] = None
    changeset: Optional[Any] = None  # Simplified for now

    # Error information
    error: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None

    # Execution metadata
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    tokens_used: Optional[int] = None

    # Next steps
    next_nodes: List[str] = field(default_factory=list)
    should_continue: bool = True
    should_retry: bool = False
    retry_delay: Optional[float] = None

    # Logs and artifacts
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class WorkflowStatus:
    """Status of the entire workflow"""

    workflow_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]

    # Progress tracking
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    skipped_nodes: int
    current_step: int
    total_steps: Optional[int] = None

    # Timing information
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None

    # Results
    final_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Node statuses
    node_statuses: Dict[
        str, Literal["pending", "running", "completed", "failed", "skipped"]
    ] = field(default_factory=dict)

    # Metrics
    total_duration_ms: float = 0.0
    total_tokens_used: int = 0
    total_cost_usd: Optional[float] = None


class NodeCallable(Protocol):
    """Protocol for node execution functions"""

    def __call__(self, context: ExecutionContext) -> ExecutionResult:
        """Execute the node with given context"""
        ...


class StateValidator(Protocol):
    """Protocol for state validation functions"""

    def __call__(self, state: Dict[str, Any], path: str) -> bool:
        """Validate state at given path"""
        ...


class ErrorHandler(Protocol):
    """Protocol for error handling functions"""

    def __call__(self, context: ExecutionContext, error: Exception) -> ExecutionResult:
        """Handle an error during node execution"""
        ...


class Middleware(Protocol):
    """Protocol for middleware functions"""

    def __call__(
        self, context: ExecutionContext, next_func: Callable[[], ExecutionResult]
    ) -> ExecutionResult:
        """Execute middleware with next function"""
        ...


# Union types for common scenarios
NodeInput = Union[Dict[str, Any], str, int, float, bool, None]
NodeOutput = Union[Dict[str, Any], str, int, float, bool, None]
StatePath = str
ConditionExpression = str
NodeDependency = str

# Type aliases for better readability
NodeID = str
WorkflowID = str
StepID = int
RetryCount = int
DurationMs = float
MemoryUsageMb = float
TokenCount = int
