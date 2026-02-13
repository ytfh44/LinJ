"""
Continuation

Implements continuation state and behavior for LinJ specification sections 17-18
Framework-agnostic continuation implementation, supports cross-framework usage
"""

import uuid
import time
from enum import Enum
from typing import Any, Dict, Optional, List, Protocol, runtime_checkable
from copy import deepcopy

from pydantic import BaseModel, Field, ConfigDict

# Import path resolver
from ..core.path import PathResolver


class HandleExpired(Exception):
    """
    Continuation handle expired error (section 17.2)

    Raised when attempting to access an expired continuation handle
    """

    def __init__(self, handle: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Continuation handle expired: {handle}")
        self.handle = handle
        self.details = details or {}


@runtime_checkable
class StateManager(Protocol):
    """State manager protocol for framework-agnostic state management"""

    def get_full_state(self) -> Dict[str, Any]:
        """Get the full state"""
        ...

    def get_revision(self) -> int:
        """Get the current revision"""
        ...

    def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
        """Apply a changeset"""
        ...


class Status(str, Enum):
    """Continuation status (section 18.1)"""

    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContinuationView:
    """
    Continuation view (section 18.2)

    Provides a controlled view of the main state object for the continuation
    Framework-agnostic view implementation
    """

    def __init__(
        self,
        state_manager: StateManager,
        step_id: int,
        pending_changes: Optional[List[Any]] = None,
    ):
        """
        Create a continuation view

        Args:
            state_manager: State manager
            step_id: Current step ID
            pending_changes: List of changesets to be committed
        """
        self._state_manager = state_manager
        self._step_id = step_id
        self._pending_changes = pending_changes or []

    def read(self, path: str) -> Any:
        """
        Read path value (section 5.2)

        Per LinJ spec: returns null when reading a non-existent path
        Supports $.a.b path syntax and [n] array index

        Args:
            path: State path, e.g. "$.a.b" or "$.arr[0]"

        Returns:
            The value at the path, or None if not found
        """
        state = self._state_manager.get_full_state()
        return PathResolver.get(state, path)

    def value(self, path: str) -> Any:
        """
        Get path value (section 14.1)

        Same as read(), used for conditional expression evaluation

        Args:
            path: State path

        Returns:
            The value at the path, or None if not found
        """
        return self.read(path)

    def exists(self, path: str) -> bool:
        """Check if the path exists and is not null"""
        return self.value(path) is not None

    def len(self, path: str) -> int:
        """
        Get array length (section 14.1)

        Returns the physical length of the array (including trailing null)
        Returns 0 for non-arrays or non-existent paths

        Args:
            path: State path

        Returns:
            Array length, 0 for non-arrays or non-existent
        """
        value = self.read(path)
        if isinstance(value, list):
            return len(value)
        return 0

    def get_full_state(self) -> Dict[str, Any]:
        """
        Get the full state (deep copy)

        Applies all pending changesets to construct the current view state
        Ensures the view reflects the logical snapshot corresponding to the current step_id

        Returns:
            Deep copy of the full state with all pending changes applied
        """
        state = self._state_manager.get_full_state()

        # Apply pending changesets to construct the current view state
        # Section 18.2: view must not include changesets with step_id >= current attempt
        for changeset in self._pending_changes:
            if hasattr(changeset, "apply_to"):
                # Use ChangeSet's apply_to method
                changeset.apply_to(state)
            elif hasattr(changeset, "apply_to_state"):
                # Use protocol's apply_to_state method
                state = changeset.apply_to_state(state)
            elif isinstance(changeset, dict):
                # Simple dict update (backward compatible)
                if "writes" in changeset:
                    for write in changeset["writes"]:
                        PathResolver.set(state, write["path"], write["value"])
                if "deletes" in changeset:
                    for delete in changeset["deletes"]:
                        PathResolver.delete(state, delete["path"])

        return deepcopy(state)

    def propose_changeset(self, changeset: Any) -> None:
        """Propose a changeset (used when suspended)"""
        self._pending_changes.append(changeset)


class Continuation(BaseModel):
    """
    Continuation

    Section 17.1: Each continuation has a unique handle, serializable to string
    Section 18.1: Contains status, local_state, view
    Framework-agnostic continuation implementation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Status = Status.RUNNING
    local_state: Dict[str, Any] = Field(default_factory=dict)
    step_id: int = 0
    base_revision: int = 0
    parent_handle: Optional[str] = None

    # Execution result
    result: Optional[Any] = None
    error: Optional[str] = None

    # Changeset generated when suspended
    pending_changeset: Optional[Any] = None

    # Expiration timestamp in milliseconds, None means never expires
    expires_at_ms: Optional[int] = None

    # View (runtime set, not serialized)
    view: Optional[ContinuationView] = Field(default=None, exclude=True)

    def is_active(self) -> bool:
        """Check if the continuation is still in an active state"""
        return self.status in (Status.RUNNING, Status.SUSPENDED)

    def is_terminal(self) -> bool:
        """Check if the continuation has reached a terminal state"""
        return self.status in (Status.COMPLETED, Status.FAILED, Status.CANCELLED)

    def can_submit_changes(self) -> bool:
        """Check if changes can be submitted (section 19.5: cannot submit after cancellation)"""
        return self.status != Status.CANCELLED

    def create_view(
        self,
        state_manager: StateManager,
        pending_changes: Optional[List[Any]] = None,
    ) -> ContinuationView:
        """
        Create or update the continuation view

        Args:
            state_manager: State manager
            pending_changes: List of changesets to be committed

        Returns:
            Continuation view
        """
        self.view = ContinuationView(state_manager, self.step_id, pending_changes)
        return self.view

    def get_logical_snapshot(self) -> Dict[str, Any]:
        """
        Get the logical snapshot (section 18.2)

        Returns the state snapshot corresponding to the step_id
        """
        if self.view:
            return self.view.get_full_state()
        return {}

    def update_step_id(self, new_step_id: int) -> None:
        """
        Update step_id and recreate the view

        Args:
            new_step_id: New step ID
        """
        self.step_id = new_step_id
        # View will be recreated on next access


class ContinuationRegistry:
    """
    Continuation registry

    Manages continuation registration and expiration handling
    Framework-agnostic registry implementation
    """

    def __init__(self, default_ttl_ms: Optional[int] = None):
        """
        Initialize the registry

        Args:
            default_ttl_ms: Default time-to-live in milliseconds, None means never expires
        """
        self._continuations: Dict[str, Continuation] = {}
        self._expiry_times: Dict[str, float] = {}  # handle -> expiry timestamp
        self._default_ttl_ms = default_ttl_ms

    def register(
        self,
        continuation: Continuation,
        ttl_ms: Optional[int] = None,
    ) -> None:
        """
        Register a continuation

        Args:
            continuation: Continuation to register
            ttl_ms: Time-to-live in milliseconds, uses default if None
        """
        self._continuations[continuation.handle] = continuation

        # Set expiration time
        if ttl_ms is not None or self._default_ttl_ms is not None:
            effective_ttl = ttl_ms if ttl_ms is not None else self._default_ttl_ms
            expiry_time = time.time() * 1000 + effective_ttl
            self._expiry_times[continuation.handle] = expiry_time

    def get(self, handle: str) -> Continuation:
        """
        Get a continuation

        Args:
            handle: Continuation handle

        Returns:
            Continuation object

        Raises:
            HandleExpired: Handle not found or expired
        """
        # Check if exists
        if handle not in self._continuations:
            raise HandleExpired(handle, {"reason": "not_found"})

        # Check if expired
        if handle in self._expiry_times:
            if time.time() * 1000 > self._expiry_times[handle]:
                # Clean up expired continuation
                del self._continuations[handle]
                del self._expiry_times[handle]
                raise HandleExpired(
                    handle,
                    {"reason": "expired", "expired_at": self._expiry_times[handle]},
                )

        return self._continuations[handle]

    def remove(self, handle: str) -> bool:
        """
        Remove a continuation

        Args:
            handle: Continuation handle

        Returns:
            True if successfully removed
        """
        if handle in self._continuations:
            del self._continuations[handle]
            if handle in self._expiry_times:
                del self._expiry_times[handle]
            return True
        return False

    def is_expired(self, handle: str) -> bool:
        """Check if the handle has expired"""
        if handle not in self._continuations:
            return True
        if handle in self._expiry_times:
            return time.time() * 1000 > self._expiry_times[handle]
        return False

    def cleanup_expired(self) -> int:
        """Clean up all expired continuations, returns the count cleaned"""
        now = time.time() * 1000
        expired_handles = [h for h, t in self._expiry_times.items() if t <= now]

        for handle in expired_handles:
            self.remove(handle)

        return len(expired_handles)

    def count(self) -> int:
        """Get the number of registered continuations"""
        return len(self._continuations)

    def clear(self) -> None:
        """Clear all registrations"""
        self._continuations.clear()
        self._expiry_times.clear()
