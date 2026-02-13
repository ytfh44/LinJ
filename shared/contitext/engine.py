"""
ContiText Engine

Implements the basic operations defined in Section 19 of the LinJ specification: derive, suspend, resume, join, cancel
Integrates CommitManager to implement deterministic change submission
Framework-agnostic continuation execution engine implementation
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import logging

from .continuation import Continuation, Status, ContinuationView, StateManager
from .signal import Signal, WaitCondition, SignalQueue
from .commit_manager import CommitManager, CommitResult

logger = logging.getLogger(__name__)


class JoinResult:
    """Join result (Section 19.4)"""

    def __init__(
        self,
        handle: str,
        status: Status,
        result: Any = None,
        error: Optional[str] = None,
    ):
        self.handle = handle
        self.status = status
        self.result = result
        self.error = error


@runtime_checkable
class ChangeSet(Protocol):
    """ChangeSet protocol for framework-agnostic changeset operations"""

    def is_empty(self) -> bool:
        """Check if changeset is empty"""
        ...

    def intersects_with(self, other: Any) -> bool:
        """Check if changeset intersects with another changeset"""
        ...

    def apply_to_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply changeset to state"""
        ...


class ContiTextEngine:
    """
    ContiText Execution Engine

    Manages continuation lifecycle: derive, suspend, resume, join, cancel
    Integrates deterministic change submission management
    Framework-agnostic engine implementation
    """

    def __init__(self, state_manager: Optional[StateManager] = None):
        """
        Initialize ContiText engine

        Args:
            state_manager: State manager for changeset submission
        """
        self._continuations: Dict[str, Continuation] = {}
        self._signal_queue = SignalQueue()
        self._children: Dict[str, List[str]] = {}  # parent -> children

        # State management and change submission
        if state_manager is None:
            # Create a simple default state manager
            self._state_manager = self._create_default_state_manager()
        else:
            self._state_manager = state_manager

        self._commit_manager = CommitManager(self._state_manager)

        # Pending changesets (ordered by step_id)
        self._pending_changes: Dict[int, Any] = {}

        logger.info("ContiText engine initialized")

    def _create_default_state_manager(self) -> StateManager:
        """Create default state manager implementation"""

        class DefaultStateManager:
            def __init__(self):
                self._state: Dict[str, Any] = {}
                self._revision: int = 0

            def get_full_state(self) -> Dict[str, Any]:
                return self._state.copy()

            def get_revision(self) -> int:
                return self._revision

            def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
                # Simple changeset application logic
                if hasattr(changeset, "apply_to_state"):
                    self._state = changeset.apply_to_state(self._state)
                else:
                    # Simple state update logic
                    if isinstance(changeset, dict):
                        self._state.update(changeset)
                self._revision += 1

        return DefaultStateManager()

    def derive(self, parent: Optional[Continuation] = None) -> Continuation:
        """
        Derive child continuation (Section 19.1)

        Args:
            parent: Parent continuation, None for root continuation

        Returns:
            Newly created continuation
        """
        cont = Continuation(
            parent_handle=parent.handle if parent else None,
            step_id=parent.step_id if parent else 0,
            base_revision=parent.base_revision if parent else 0,
        )

        self._continuations[cont.handle] = cont

        # Record parent-child relationship
        if parent:
            if parent.handle not in self._children:
                self._children[parent.handle] = []
            self._children[parent.handle].append(cont.handle)

        logger.debug(
            f"Derived continuation: {cont.handle} from parent: {parent.handle if parent else None}"
        )
        return cont

    def suspend(
        self,
        cont: Continuation,
        changeset: Optional[Any] = None,
        wait_condition: Optional[WaitCondition] = None,
    ) -> None:
        """
        Suspend continuation (Section 19.2)

        Args:
            cont: Continuation to suspend
            changeset: Changeset produced when suspending
            wait_condition: Wait condition (if any)
        """
        if cont.handle not in self._continuations:
            raise ValueError(f"Continuation {cont.handle} not found")

        cont.status = Status.SUSPENDED
        cont.pending_changeset = changeset

        # Register waiter
        if wait_condition:
            self._signal_queue.register_waiter(cont.handle, wait_condition)

        logger.debug(f"Suspended continuation: {cont.handle}")

    def resume(self, handle: str, input_data: Optional[Dict] = None) -> Continuation:
        """
        Resume continuation (Section 19.3)

        Args:
            handle: Continuation handle
            input_data: Injected input data

        Returns:
            Resumed continuation
        """
        if handle not in self._continuations:
            raise ValueError(f"Continuation {handle} expired or invalid")

        cont = self._continuations[handle]

        if cont.status != Status.SUSPENDED:
            raise ValueError(f"Cannot resume continuation in state: {cont.status}")

        cont.status = Status.RUNNING

        # Inject input data
        if input_data:
            cont.local_state.update(input_data)

        # Unregister waiter
        self._signal_queue.unregister_waiter(handle)

        logger.debug(f"Resumed continuation: {handle}")
        return cont

    async def join(self, handles: List[str]) -> List[JoinResult]:
        """
        Join wait (Section 19.4)

        Wait for all specified continuations to reach terminal state

        Args:
            handles: List of continuation handles

        Returns:
            Join result for each continuation
        """
        import asyncio

        results = []

        for handle in handles:
            if handle not in self._continuations:
                raise ValueError(f"Continuation {handle} not found")

            cont = self._continuations[handle]

            # Wait for continuation to reach terminal state
            while not cont.is_terminal():
                await asyncio.sleep(0.01)  # Simple polling

            results.append(
                JoinResult(
                    handle=handle,
                    status=cont.status,
                    result=cont.result,
                    error=cont.error,
                )
            )

        logger.debug(f"Joined {len(handles)} continuations")
        return results

    def cancel(self, handle: str) -> None:
        """
        Cancel continuation (Section 19.5)

        - Idempotent: repeated cancellation produces no new side effects
        - Propagation: child continuations should also be cancelled when parent is cancelled
        - No changesets should be submitted after cancellation

        Args:
            handle: Continuation handle to cancel
        """
        if handle not in self._continuations:
            return  # Idempotent: silent return for non-existent handle

        cont = self._continuations[handle]

        # If already terminal, skip processing
        if cont.is_terminal():
            return

        # Mark as cancelled
        cont.status = Status.CANCELLED

        logger.debug(f"Cancelled continuation: {handle}")

        # Recursively cancel child continuations
        if handle in self._children:
            for child_handle in self._children[handle]:
                self.cancel(child_handle)

    def complete(self, handle: str, result: Any) -> None:
        """Mark continuation as successfully completed"""
        if handle not in self._continuations:
            raise ValueError(f"Continuation {handle} not found")

        cont = self._continuations[handle]

        if cont.status == Status.CANCELLED:
            raise ValueError("Cannot complete cancelled continuation")

        cont.status = Status.COMPLETED
        cont.result = result

        logger.debug(f"Completed continuation: {handle}")

    def fail(self, handle: str, error: str) -> None:
        """Mark continuation as failed"""
        if handle not in self._continuations:
            raise ValueError(f"Continuation {handle} not found")

        cont = self._continuations[handle]

        if cont.status == Status.CANCELLED:
            raise ValueError("Cannot fail cancelled continuation")

        cont.status = Status.FAILED
        cont.error = error

        logger.debug(f"Failed continuation: {handle}, error: {error}")

    def send_signal(self, signal: Signal) -> None:
        """Send signal"""
        self._signal_queue.send(signal)
        logger.debug(f"Sent signal: {signal.name}")

    def check_signal(self, handle: str, state: Dict) -> Optional[Signal]:
        """Check if waiter received signal"""
        if handle not in self._continuations:
            return None

        cont = self._continuations[handle]
        if cont.status != Status.SUSPENDED:
            return None

        # Find matching signal
        for signal in self._signal_queue._signals:
            # Check if there's a wait condition
            wait_condition = self._signal_queue._waiters.get(handle)
            if wait_condition:
                if wait_condition.matches(signal, state):
                    logger.debug(
                        f"Signal matched for continuation {handle}: {signal.name}"
                    )
                    return signal
            else:
                # No wait condition, return first signal
                logger.debug(
                    f"Signal received for continuation {handle}: {signal.name}"
                )
                return signal

        return None

    def get_continuation(self, handle: str) -> Optional[Continuation]:
        """Get continuation"""
        return self._continuations.get(handle)

    def submit_changeset(
        self,
        step_id: int,
        changeset: Any,
        handle: str,
        base_revision: Optional[int] = None,
    ) -> CommitResult:
        """
        Submit changeset (Section 20.2)

        Integrate CommitManager to implement deterministic submission

        Args:
            step_id: Step ID
            changeset: Changeset
            handle: Continuation handle
            base_revision: Base revision, uses current revision if None

        Returns:
            Commit result
        """
        if base_revision is None:
            base_revision = self._state_manager.get_revision()

        result = self._commit_manager.submit(
            step_id=step_id,
            base_revision=base_revision,
            changeset=changeset,
            handle=handle,
        )

        if result.success:
            logger.debug(f"Changeset submitted successfully: step_id={step_id}")
        else:
            logger.warning(
                f"Changeset submission failed: step_id={step_id}, error={result.error}"
            )

        return result

    def process_pending_changes(self) -> List[CommitResult]:
        """
        Process pending changesets

        Returns:
            All results produced by this processing
        """
        results = self._commit_manager.process_queue()
        logger.debug(f"Processed {len(results)} pending changesets")
        return results

    def create_view(
        self, cont: Continuation, pending_changes: Optional[List[Any]] = None
    ) -> ContinuationView:
        """
        Create view for continuation (Section 18.2)

        Args:
            cont: Continuation
            pending_changes: List of pending changesets

        Returns:
            Continuation view
        """
        return cont.create_view(self._state_manager, pending_changes)

    def get_state_manager(self) -> StateManager:
        """Get state manager"""
        return self._state_manager

    def get_commit_manager(self) -> CommitManager:
        """Get commit manager"""
        return self._commit_manager
