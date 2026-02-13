"""
Execution Context and State Management Interface

Defines the context management and state operation interfaces during execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
import time

from .types import NodeExecution, ExecutionStatus, ExecutionContext
from ..core.path import PathResolver as CorePathResolver

# PathResolver reference kept for compatibility
PathResolver = CorePathResolver


class StateScope(Enum):
    """State scope enum"""

    GLOBAL = "global"  # Global scope
    SESSION = "session"  # Session scope
    STEP = "step"  # Step scope
    NODE = "node"  # Node scope
    TEMPORARY = "temporary"  # Temporary scope


@dataclass
class StateEntry:
    """State entry"""

    value: Any
    scope: StateScope
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    read_count: int = 0
    write_count: int = 0

    def mark_read(self) -> None:
        """Mark as read"""
        self.read_count += 1

    def mark_write(self) -> None:
        """Mark as written"""
        self.write_count += 1
        self.timestamp = time.time()


class StateManager(ABC):
    """
    Abstract State Manager Interface

    Defines the unified interface for state operations:
    - State read/write and query
    - Scope management
    - Change tracking
    - Persistence and recovery
    """

    @abstractmethod
    def get(self, path: str, scope: Optional[StateScope] = None) -> Any:
        """
        Get state value

        Args:
            path: State path
            scope: State scope

        Returns:
            State value
        """
        pass

    @abstractmethod
    def set(self, path: str, value: Any, scope: Optional[StateScope] = None) -> None:
        """
        Set state value

        Args:
            path: State path
            value: Value to set
            scope: State scope
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Check if path exists

        Args:
            path: State path

        Returns:
            True if exists, False if not exists
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """
        Delete state item

        Args:
            path: State path

        Returns:
            True if deleted successfully, False if path does not exist
        """
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List all keys

        Args:
            prefix: Key prefix filter

        Returns:
            Key list
        """
        pass

    @abstractmethod
    def clear(self, scope: Optional[StateScope] = None) -> None:
        """
        Clear state

        Args:
            scope: Scope to clear, None means clear all
        """
        pass

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        Create state snapshot

        Returns:
            State snapshot dictionary
        """
        pass

    @abstractmethod
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        Restore state snapshot

        Args:
            snapshot: State snapshot
        """
        pass


class BaseStateManager(StateManager):
    """
    Base State Manager Implementation

    Provides in-memory state management functionality
    """

    def __init__(self):
        self._state: Dict[str, StateEntry] = {}
        self._watchers: Dict[str, List[Callable]] = {}
        self._change_log: List[Dict[str, Any]] = []

    def get(self, path: str, scope: Optional[StateScope] = None) -> Any:
        """Get state value"""
        if path not in self._state:
            return None

        entry = self._state[path]
        entry.mark_read()

        # Check scope filter
        if scope and entry.scope != scope:
            return None

        # Trigger watchers
        self._notify_watchers(path, "read", entry.value)

        return entry.value

    def set(self, path: str, value: Any, scope: Optional[StateScope] = None) -> None:
        """Set state value"""
        scope = scope or StateScope.GLOBAL

        old_value = None
        if path in self._state:
            old_entry = self._state[path]
            old_value = old_entry.value
            # Update existing entry
            old_entry.value = value
            old_entry.mark_write()
        else:
            # Create new entry
            self._state[path] = StateEntry(value=value, scope=scope)

        # Record change
        change = {
            "path": path,
            "old_value": old_value,
            "new_value": value,
            "scope": scope.value,
            "timestamp": time.time(),
        }
        self._change_log.append(change)

        # Trigger watchers
        self._notify_watchers(path, "write", value)

    def exists(self, path: str) -> bool:
        """Check if path exists"""
        return path in self._state

    def delete(self, path: str) -> bool:
        """Delete state item"""
        if path not in self._state:
            return False

        old_value = self._state[path].value
        del self._state[path]

        # Record change
        change = {
            "path": path,
            "old_value": old_value,
            "new_value": None,
            "deleted": True,
            "timestamp": time.time(),
        }
        self._change_log.append(change)

        # Trigger watchers
        self._notify_watchers(path, "delete", None)

        return True

    def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys"""
        if prefix:
            return [key for key in self._state.keys() if key.startswith(prefix)]
        return list(self._state.keys())

    def clear(self, scope: Optional[StateScope] = None) -> None:
        """Clear state"""
        if scope is None:
            self._state.clear()
        else:
            keys_to_delete = [
                key for key, entry in self._state.items() if entry.scope == scope
            ]
            for key in keys_to_delete:
                del self._state[key]

        self._change_log.append(
            {
                "action": "clear",
                "scope": scope.value if scope else "all",
                "timestamp": time.time(),
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        """Create state snapshot"""
        return {key: entry.value for key, entry in self._state.items()}

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state snapshot"""
        # Clear current state
        self._state.clear()

        # Restore snapshot
        for path, value in snapshot.items():
            self._state[path] = StateEntry(value=value, scope=StateScope.GLOBAL)

        # Record restore operation
        self._change_log.append(
            {
                "action": "restore",
                "timestamp": time.time(),
            }
        )

    def watch(self, path: str, callback: Callable) -> None:
        """Add state watcher"""
        if path not in self._watchers:
            self._watchers[path] = []
        self._watchers[path].append(callback)

    def unwatch(self, path: str, callback: Callable) -> None:
        """Remove state watcher"""
        if path in self._watchers:
            try:
                self._watchers[path].remove(callback)
                if not self._watchers[path]:
                    del self._watchers[path]
            except ValueError:
                pass

    def _notify_watchers(self, path: str, action: str, value: Any) -> None:
        """Notify watchers"""
        if path in self._watchers:
            for callback in self._watchers[path]:
                try:
                    callback(path, action, value)
                except Exception:
                    # Watcher errors should not affect main flow
                    pass

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Get change log"""
        return self._change_log.copy()

    def clear_change_log(self) -> None:
        """Clear change log"""
        self._change_log.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get state statistics"""
        scope_counts = {}
        for entry in self._state.values():
            scope_name = entry.scope.value
            scope_counts[scope_name] = scope_counts.get(scope_name, 0) + 1

        total_reads = sum(entry.read_count for entry in self._state.values())
        total_writes = sum(entry.write_count for entry in self._state.values())

        return {
            "total_entries": len(self._state),
            "scope_distribution": scope_counts,
            "total_reads": total_reads,
            "total_writes": total_writes,
            "change_log_size": len(self._change_log),
            "active_watchers": len(self._watchers),
        }


class ContextManager:
    """
    Context Manager

    Manages the lifecycle and state transitions of execution contexts
    """

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self._contexts: Dict[str, ExecutionContext] = {}
        self._active_context: Optional[ExecutionContext] = None

    def create_context(
        self,
        context_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionContext:
        """
        Create execution context

        Args:
            context_id: Context ID
            initial_state: Initial state
            metadata: Metadata

        Returns:
            Execution context
        """
        context = ExecutionContext(
            state=initial_state or {},
            metadata=metadata or {},
            step_counter=0,
            execution_history=[],
        )

        self._contexts[context_id] = context
        return context

    def get_context(self, context_id: str) -> Optional[ExecutionContext]:
        """Get execution context"""
        return self._contexts.get(context_id)

    def set_active_context(self, context_id: str) -> bool:
        """Set active context"""
        if context_id in self._contexts:
            self._active_context = self._contexts[context_id]
            return True
        return False

    def get_active_context(self) -> Optional[ExecutionContext]:
        """Get active context"""
        return self._active_context

    def update_context(
        self,
        context_id: str,
        state_updates: Dict[str, Any],
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update context"""
        if context_id not in self._contexts:
            return False

        context = self._contexts[context_id]
        context.state.update(state_updates)

        if metadata_updates:
            context.metadata.update(metadata_updates)

        return True

    def delete_context(self, context_id: str) -> bool:
        """Delete context"""
        if context_id not in self._contexts:
            return False

        if self._active_context == self._contexts[context_id]:
            self._active_context = None

        del self._contexts[context_id]
        return True

    def list_contexts(self) -> List[str]:
        """List all context IDs"""
        return list(self._contexts.keys())

    def clear_all_contexts(self) -> None:
        """Clear all contexts"""
        self._contexts.clear()
        self._active_context = None

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context statistics"""
        return {
            "total_contexts": len(self._contexts),
            "active_context_id": (
                next(
                    (k for k, v in self._contexts.items() if v == self._active_context),
                    None,
                )
            ),
            "total_steps": sum(ctx.step_counter for ctx in self._contexts.values()),
            "total_executions": sum(
                len(ctx.execution_history) for ctx in self._contexts.values()
            ),
        }


# PathResolver is now imported from ..core.path
