"""ContiText Framework

Framework-agnostic continuation execution engine for parallel processing.
Implements LinJ specification sections 17-26 for continuation management,
signal handling, and deterministic execution.
"""

from .continuation import Continuation, Status, ContinuationView, StateManager
from .engine import ContiTextEngine, JoinResult, ChangeSet
from .signal import Signal, WaitCondition, SignalQueue
from .commit_manager import (
    CommitManager,
    CommitResult,
    PendingChangeSet,
    PendingStatus,
    LinJError,
    ConflictError,
)
from .mapper import (
    LinJToContiTextMapper,
    ParallelLinJExecutor,
    LinJDocument,
    Node,
    DeterministicScheduler,
    LinJExecutor,
)

__version__ = "1.0.0"

__all__ = [
    # Core continuation components
    "Continuation",
    "Status",
    "ContinuationView",
    "StateManager",
    # Engine components
    "ContiTextEngine",
    "JoinResult",
    "ChangeSet",
    # Signal and waiting mechanisms
    "Signal",
    "WaitCondition",
    "SignalQueue",
    # Commit management
    "CommitManager",
    "CommitResult",
    "PendingChangeSet",
    "PendingStatus",
    "LinJError",
    "ConflictError",
    # LinJ mapping
    "LinJToContiTextMapper",
    "ParallelLinJExecutor",
    # Protocol interfaces for framework integration
    "LinJDocument",
    "Node",
    "DeterministicScheduler",
    "LinJExecutor",
]
