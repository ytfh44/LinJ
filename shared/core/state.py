"""
Main State Management

Implements main state management as defined in section 3.1 of the specification
"""

from typing import Any, Dict, List, Optional
import copy

from .changeset import ChangeSet
from .path import PathResolver


class StateManager:
    """
    Main State Manager

    Responsible for managing main state objects, applying change sets, creating views
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._state = {}
        if initial_state:
            for path, value in initial_state.items():
                if path.startswith("$."):
                    PathResolver.set(self._state, path, value)
                else:
                    # Compatibility for simple keys without $.
                    PathResolver.set(self._state, f"$.{path}", value)

        self._revision = 0
        self._history: List[Dict[str, Any]] = []  # Change history (for debugging)

    def get(self, path: str) -> Any:
        """Read path value"""
        return PathResolver.get(self._state, path)

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state (deep copy)"""
        return copy.deepcopy(self._state)

    def apply(self, changeset: ChangeSet, step_id: Optional[int] = None) -> None:
        """
        Apply change set to state

        Section 9.2: Change set application must be atomic
        """
        if changeset.is_empty():
            return

        # Apply to temporary state first for validation
        temp_state = copy.deepcopy(self._state)
        changeset.apply_to(temp_state)

        # Apply to actual state after validation passes
        changeset.apply_to(self._state)

        # Update revision
        self._revision += 1

        # Record history
        self._history.append(
            {
                "step_id": step_id,
                "revision": self._revision,
                "changeset": changeset.model_dump(),
            }
        )

    def get_revision(self) -> int:
        """Get current revision number"""
        return self._revision

    def snapshot(self) -> Dict[str, Any]:
        """Create state snapshot"""
        return copy.deepcopy(self._state)

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot"""
        self._state = copy.deepcopy(snapshot)
        self._revision = 0
        self._history.clear()


class StateView:
    """
    State View (Logical Snapshot)

    Section 18.2: Provide logical snapshot corresponding to step_id for each node attempt
    """

    def __init__(
        self,
        state_manager: StateManager,
        step_id: int,
        pending_changes: Optional[List[ChangeSet]] = None,
    ):
        """
        Create state view

        Args:
            state_manager: State manager
            step_id: Current step ID
            pending_changes: List of change sets pending submission
        """
        self._state_manager = state_manager
        self._step_id = step_id
        self._pending = pending_changes or []

        # Create logical snapshot
        self._snapshot = state_manager.snapshot()

        # Apply all pending changes with step_id less than current
        for cs in self._pending:
            cs.apply_to(self._snapshot)

    def read(self, path: str) -> Any:
        """Read path value"""
        return PathResolver.get(self._snapshot, path)

    def exists(self, path: str) -> bool:
        """Check if path exists and is not null"""
        return self.read(path) is not None

    def len(self, path: str) -> int:
        """Get array length"""
        value = self.read(path)
        if isinstance(value, list):
            return len(value)
        return 0

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state (deep copy)"""
        return copy.deepcopy(self._snapshot)

    def propose(self, changeset: ChangeSet) -> None:
        """
        Propose change set

        Change set is not applied directly, but submitted to pending queue
        """
        self._pending.append(changeset)
