"""
CommitManager - Change Set Submission Manager

Implements changeset submission management as defined in LinJ Specification Sections 20 and 24.3:
- Baseline Rule: Accept changesets serially in ascending step_id order
- Read-only Optimization: Empty changesets can be recorded as completed immediately
- Non-intersecting Optimization: Larger step_id can be accepted early, provided that
  write/delete paths are pairwise non-intersecting with all unaccepted smaller step_id changesets
- Baseline Revision Check: Check if base_revision matches current state version during submission
- Conflict Generation: ConflictError is generated when conflicts occur
Framework-agnostic submission manager implementation
"""

import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


@runtime_checkable
class StateManager(Protocol):
    """State manager protocol for framework-agnostic state management"""

    def get_full_state(self) -> Dict[str, Any]:
        """Get the complete state"""
        ...

    def get_revision(self) -> int:
        """Get the current revision number"""
        ...

    def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
        """Apply the changeset"""
        ...


class LinJError(Exception):
    """Base class for LinJ specification-related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ConflictError(LinJError):
    """Conflict error"""

    pass


class PendingStatus(str, Enum):
    """Status of pending changeset submissions"""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class PendingChangeSet(BaseModel):
    """
    Encapsulation of pending changeset submissions

    Contains changeset metadata and submission status tracking
    Framework-agnostic implementation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: int
    base_revision: int
    changeset: Any
    continuation_handle: str
    status: PendingStatus = PendingStatus.PENDING


class CommitResult(BaseModel):
    """
    Submission result

    Records the complete result of changeset submission
    Framework-agnostic implementation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    step_id: int
    new_revision: Optional[int] = None
    error: Optional[LinJError] = None


@runtime_checkable
class ChangeSet(Protocol):
    """Changeset protocol for framework-agnostic changeset operations"""

    def is_empty(self) -> bool:
        """Check if changeset is empty"""
        ...

    def intersects_with(self, other: Any) -> bool:
        """Check if it intersects with another changeset"""
        ...

    def apply_to_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply changeset to state"""
        ...


class CommitManager:
    """
    Change Submission Manager

    Implements the definitive submission rules defined in Section 24.3:
    - Baseline Rule: Accept serially in ascending step_id order
    - Read-only Optimization: Empty changesets can be accepted immediately
    - Non-intersecting Optimization: Non-intersecting changesets can be accepted early

    Thread-safe: All queue operations are protected by locks
    Framework-agnostic submission manager implementation
    """

    def __init__(self, state_manager: StateManager):
        """
        Initialize the submission manager

        Args:
            state_manager: State manager used to apply changesets
        """
        self._state_manager = state_manager
        self._pending: Dict[int, PendingChangeSet] = {}  # step_id -> pending
        self._accepted_step_ids: Set[int] = set()  # accepted step_id
        self._next_expected_step: int = 1  # next expected step_id
        self._lock = threading.RLock()  # reentrant lock for queue operations
        self._results: Dict[int, CommitResult] = {}  # step_id -> result

    def submit(
        self, step_id: int, base_revision: int, changeset: Any, handle: str
    ) -> CommitResult:
        """
        Submit a changeset

        Process:
        1. Check if baseline revision matches
        2. If changeset is empty -> Accept immediately (read-only optimization)
        3. Check if it intersects with changesets in the pending queue
        4. If step_id == next_expected -> Apply directly
        5. If step_id > next_expected and non-intersecting -> Non-intersecting optimization allows early acceptance
        6. Otherwise -> Add to pending queue

        Args:
            step_id: Step ID
            base_revision: Baseline revision number
            changeset: Changeset
            handle: Continuation handle

        Returns:
            CommitResult: Submission result
        """
        with self._lock:
            # Check if already submitted
            if step_id in self._results:
                return self._results[step_id]

            # Baseline revision check (Section 24.3)
            # During submission, base_revision must equal current revision
            current_revision = self._state_manager.get_revision()
            if base_revision != current_revision:
                error = ConflictError(
                    f"Base revision mismatch: expected {current_revision}, got {base_revision}",
                    {
                        "step_id": step_id,
                        "expected": current_revision,
                        "got": base_revision,
                    },
                )
                result = CommitResult(success=False, step_id=step_id, error=error)
                self._results[step_id] = result
                return result

            # Create pending submission record
            pending = PendingChangeSet(
                step_id=step_id,
                base_revision=base_revision,
                changeset=changeset,
                continuation_handle=handle,
            )

            # Read-only optimization: Empty changeset accepted immediately
            if self._is_changeset_empty(changeset):
                pending.status = PendingStatus.ACCEPTED
                self._pending[step_id] = pending
                self._accepted_step_ids.add(step_id)

                # Update next_expected
                if step_id == self._next_expected_step:
                    self._update_next_expected()

                result = CommitResult(
                    success=True,
                    step_id=step_id,
                    new_revision=current_revision,  # Empty changeset does not change revision
                )
                self._results[step_id] = result

                # Try to process queue
                self._process_queue_internal()
                return result

            # Check if can be accepted
            if self.can_accept(step_id, changeset):
                # Apply changeset
                try:
                    self._state_manager.apply(changeset, step_id)
                    new_revision = self._state_manager.get_revision()

                    pending.status = PendingStatus.ACCEPTED
                    self._pending[step_id] = pending
                    self._accepted_step_ids.add(step_id)

                    # Update next_expected
                    if step_id == self._next_expected_step:
                        self._update_next_expected()

                    result = CommitResult(
                        success=True, step_id=step_id, new_revision=new_revision
                    )
                except Exception as e:
                    pending.status = PendingStatus.REJECTED
                    self._pending[step_id] = pending

                    error = ConflictError(
                        f"Failed to apply changeset: {str(e)}",
                        {"step_id": step_id, "error": str(e)},
                    )
                    result = CommitResult(success=False, step_id=step_id, error=error)

                self._results[step_id] = result

                # Try to process queue
                if result.success:
                    self._process_queue_internal()

                return result
            else:
                # Add to pending queue
                self._pending[step_id] = pending

                result = CommitResult(
                    success=False,
                    step_id=step_id,
                    error=ConflictError(
                        f"Changeset {step_id} cannot be accepted yet, queued for processing",
                        {"step_id": step_id, "next_expected": self._next_expected_step},
                    ),
                )
                # Note: Result is not recorded here as it has not been finally processed
                return result

            # Create pending submission record
            pending = PendingChangeSet(
                step_id=step_id,
                base_revision=base_revision,
                changeset=changeset,
                continuation_handle=handle,
            )

            # Read-only optimization: Empty changeset accepted immediately
            if self._is_changeset_empty(changeset):
                pending.status = PendingStatus.ACCEPTED
                self._pending[step_id] = pending
                self._accepted_step_ids.add(step_id)

                # Update next_expected
                if step_id == self._next_expected_step:
                    self._update_next_expected()

                result = CommitResult(
                    success=True,
                    step_id=step_id,
                    new_revision=current_revision,  # Empty changeset does not change revision
                )
                self._results[step_id] = result

                # Try to process queue
                self._process_queue_internal()
                return result

            # Check if can be accepted
            if self.can_accept(step_id, changeset):
                # Apply changeset
                try:
                    self._state_manager.apply(changeset, step_id)
                    new_revision = self._state_manager.get_revision()

                    pending.status = PendingStatus.ACCEPTED
                    self._pending[step_id] = pending
                    self._accepted_step_ids.add(step_id)

                    # Update next_expected

                    result = CommitResult(
                        success=True, step_id=step_id, new_revision=new_revision
                    )
                except Exception as e:
                    pending.status = PendingStatus.REJECTED
                    self._pending[step_id] = pending

                    error = ConflictError(
                        f"Failed to apply changeset: {str(e)}",
                        {"step_id": step_id, "error": str(e)},
                    )
                    result = CommitResult(success=False, step_id=step_id, error=error)

                self._results[step_id] = result

                # Try to process queue
                if result.success:
                    self._process_queue_internal()

                return result
            else:
                # Add to pending queue
                self._pending[step_id] = pending

                result = CommitResult(
                    success=False,
                    step_id=step_id,
                    error=ConflictError(
                        f"Changeset {step_id} cannot be accepted yet, queued for processing",
                        {"step_id": step_id, "next_expected": self._next_expected_step},
                    ),
                )
                # Note: Result is not recorded here as it has not been finally processed
                return result

    def can_accept(self, step_id: int, changeset: Any) -> bool:
        """
        Check if changeset can be accepted

        Rules:
        1. If step_id == next_expected -> Can accept
        2. If step_id > next_expected and all smaller step_ids are submitted (in pending)
           and non-intersecting with them -> Can accept (non-intersecting optimization)

        Args:
            step_id: Step ID
            changeset: Changeset

        Returns:
            bool: Whether it can be accepted
        """
        with self._lock:
            # Baseline rule: Accept serially in ascending step_id order
            if step_id == self._next_expected_step:
                return True

            # Non-intersecting optimization: Check if can be accepted early
            if step_id > self._next_expected_step:
                # Check if all smaller step_ids are submitted (in pending)
                for sid in range(self._next_expected_step, step_id):
                    if sid not in self._pending:
                        # Smaller step_id not yet submitted, cannot determine intersection
                        return False

                # All smaller step_ids are submitted, check if intersecting with them
                if not self._check_intersection_with_pending(step_id, changeset):
                    return True

            return False

    def _is_changeset_empty(self, changeset: Any) -> bool:
        """
        Check if changeset is empty

        Framework-agnostic empty check implementation
        """
        # Check if is_empty method exists
        if hasattr(changeset, "is_empty"):
            return changeset.is_empty()

        # Check if it's an empty dict
        if isinstance(changeset, dict):
            return len(changeset) == 0

        # Check if it's an empty list
        if isinstance(changeset, list):
            return len(changeset) == 0

        # Check if it's None
        if changeset is None:
            return True

        # Default to non-empty
        return False

    def _changesets_intersect(self, changeset1: Any, changeset2: Any) -> bool:
        """
        Check if two changesets intersect

        Framework-agnostic intersection check implementation
        """
        # Check if intersects_with method exists
        if hasattr(changeset1, "intersects_with"):
            return changeset1.intersects_with(changeset2)

        # Simple intersection check: If both are dicts, check for overlapping keys
        if isinstance(changeset1, dict) and isinstance(changeset2, dict):
            return bool(set(changeset1.keys()) & set(changeset2.keys()))

        # Default to intersecting (conservative strategy)
        return True

    def _check_intersection_with_pending(self, step_id: int, changeset: Any) -> bool:
        """
        Check if changeset intersects with all pending smaller step_id changesets

        Determine write/delete path intersection according to Section 11.4 rules

        Args:
            step_id: Step ID
            changeset: Changeset

        Returns:
            bool: Whether it intersects with any pending changeset (True = intersects)
        """
        # Get all pending changesets with smaller step_id that are not accepted
        for other_step_id, pending in self._pending.items():
            if other_step_id < step_id and pending.status == PendingStatus.PENDING:
                if self._changesets_intersect(changeset, pending.changeset):
                    return True

        return False

    def process_queue(self) -> List[CommitResult]:
        """
        Process pending submission queue

        Process changesets in queue according to rules, return all newly processed results

        Returns:
            List[CommitResult]: All results generated from this processing
        """
        with self._lock:
            results = []

            while True:
                # Check all items in PENDING status sorted by step_id
                pending_steps = sorted(
                    [
                        sid
                        for sid, p in self._pending.items()
                        if p.status == PendingStatus.PENDING
                    ]
                )

                processed_any = False

                for step_id in pending_steps:
                    pending = self._pending[step_id]

                    # Check if baseline revision is still valid
                    current_revision = self._state_manager.get_revision()
                    if pending.base_revision < current_revision:
                        if self._check_intersection_since_revision(
                            pending.base_revision, pending.changeset
                        ):
                            pending.status = PendingStatus.REJECTED
                            result = CommitResult(
                                success=False,
                                step_id=step_id,
                                error=ConflictError(
                                    f"Base revision outdated by intersection: current {current_revision}, "
                                    f"base {pending.base_revision}",
                                    {"step_id": step_id},
                                ),
                            )
                            self._results[step_id] = result
                            results.append(result)
                            processed_any = True
                            continue
                    elif pending.base_revision > current_revision:
                        # This situation normally shouldn't happen unless state was rolled back
                        continue

                    # Check if can be accepted
                    if self.can_accept(step_id, pending.changeset):
                        try:
                            if self._is_changeset_empty(pending.changeset):
                                # Empty changeset, only update status
                                pending.status = PendingStatus.ACCEPTED
                                self._accepted_step_ids.add(step_id)

                                if step_id == self._next_expected_step:
                                    self._update_next_expected()

                                result = CommitResult(
                                    success=True,
                                    step_id=step_id,
                                    new_revision=current_revision,
                                )
                            else:
                                # Apply changeset
                                self._state_manager.apply(pending.changeset, step_id)
                                new_revision = self._state_manager.get_revision()

                                pending.status = PendingStatus.ACCEPTED
                                self._accepted_step_ids.add(step_id)

                                if step_id == self._next_expected_step:
                                    self._update_next_expected()

                                result = CommitResult(
                                    success=True,
                                    step_id=step_id,
                                    new_revision=new_revision,
                                )
                        except Exception as e:
                            pending.status = PendingStatus.REJECTED
                            result = CommitResult(
                                success=False,
                                step_id=step_id,
                                error=ConflictError(
                                    f"Failed to apply changeset: {str(e)}",
                                    {"step_id": step_id, "error": str(e)},
                                ),
                            )

                        self._results[step_id] = result
                        results.append(result)
                        processed_any = True

                # If no changesets were processed, exit loop
                if not processed_any:
                    break

            return results

    def _process_queue_internal(self) -> None:
        """Internal method: Process queue (must be called within lock)"""
        while True:
            pending_steps = sorted(
                [
                    sid
                    for sid, p in self._pending.items()
                    if p.status == PendingStatus.PENDING
                ]
            )

            processed_any = False

            for step_id in pending_steps:
                pending = self._pending[step_id]

                # Check baseline revision
                current_revision = self._state_manager.get_revision()
                if pending.base_revision < current_revision:
                    if self._check_intersection_since_revision(
                        pending.base_revision, pending.changeset
                    ):
                        pending.status = PendingStatus.REJECTED
                        self._results[step_id] = CommitResult(
                            success=False,
                            step_id=step_id,
                            error=ConflictError(
                                f"Base revision outdated with intersection: current {current_revision}, "
                                f"base {pending.base_revision}",
                                {"step_id": step_id},
                            ),
                        )
                        processed_any = True
                        continue
                elif pending.base_revision > current_revision:
                    continue

                # Check if can be accepted
                if self.can_accept(step_id, pending.changeset):
                    try:
                        if self._is_changeset_empty(pending.changeset):
                            pending.status = PendingStatus.ACCEPTED
                            self._accepted_step_ids.add(step_id)

                            if step_id == self._next_expected_step:
                                self._update_next_expected()

                            self._results[step_id] = CommitResult(
                                success=True,
                                step_id=step_id,
                                new_revision=current_revision,
                            )
                        else:
                            self._state_manager.apply(pending.changeset, step_id)
                            new_revision = self._state_manager.get_revision()

                            pending.status = PendingStatus.ACCEPTED
                            self._accepted_step_ids.add(step_id)

                            if step_id == self._next_expected_step:
                                self._update_next_expected()

                            self._results[step_id] = CommitResult(
                                success=True, step_id=step_id, new_revision=new_revision
                            )
                    except Exception as e:
                        pending.status = PendingStatus.REJECTED
                        self._results[step_id] = CommitResult(
                            success=False,
                            step_id=step_id,
                            error=ConflictError(
                                f"Failed to apply changeset: {str(e)}",
                                {"step_id": step_id, "error": str(e)},
                            ),
                        )

                    processed_any = True

            if not processed_any:
                break

    def _check_intersection_since_revision(
        self, base_revision: int, changeset: Any
    ) -> bool:
        """
        Check if changeset intersects with all applied changesets since base_revision

        Used to determine if an older version ChangeSet can be applied to a newer version state
        """
        # We need to find all accepted changesets with new_revision > base_revision
        # These changesets already exist in self._results
        for step_id, result in self._results.items():
            if (
                result.success
                and result.new_revision
                and result.new_revision > base_revision
            ):
                # Get the changeset corresponding to that step_id
                if step_id in self._pending:
                    other_cs = self._pending[step_id].changeset
                    if self._changesets_intersect(changeset, other_cs):
                        return True
        return False

    def _update_next_expected(self) -> None:
        """Update next_expected_step (must be called within lock)"""
        while self._next_expected_step in self._accepted_step_ids:
            self._next_expected_step += 1

    def get_pending(self) -> List[PendingChangeSet]:
        """
        Get all pending changesets

        Returns:
            List[PendingChangeSet]: Pending changeset list (sorted by step_id)
        """
        with self._lock:
            return sorted(
                [
                    p
                    for p in self._pending.values()
                    if p.step_id not in self._accepted_step_ids
                ],
                key=lambda p: p.step_id,
            )

    def get_result(self, step_id: int) -> Optional[CommitResult]:
        """
        Get the submission result for a specific step_id

        Args:
            step_id: Step ID

        Returns:
            Optional[CommitResult]: Submission result, or None if not processed
        """
        with self._lock:
            return self._results.get(step_id)

    def get_accepted_count(self) -> int:
        """
        Get the count of accepted changesets

        Returns:
            int: Accepted count
        """
        with self._lock:
            return len(self._accepted_step_ids)

    def get_pending_count(self) -> int:
        """
        Get the count of pending changesets

        Returns:
            int: Pending count
        """
        with self._lock:
            return len(
                [
                    p
                    for p in self._pending.values()
                    if p.step_id not in self._accepted_step_ids
                ]
            )

    def is_all_accepted(self) -> bool:
        """
        Check if all submitted changesets have been accepted

        Returns:
            bool: Whether all are accepted
        """
        with self._lock:
            total = len(self._pending)
            accepted = len(self._accepted_step_ids)
            return total > 0 and total == accepted

    def reset(self) -> None:
        """
        Reset submission manager state

        Clear all pending and accepted states
        """
        with self._lock:
            self._pending.clear()
            self._accepted_step_ids.clear()
            self._results.clear()
            self._next_expected_step = 1
