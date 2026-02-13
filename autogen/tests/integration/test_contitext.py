"""
ContiText Continuation Mechanism Tests

Verifies continuation behavior defined in Sections 17-22:
- Continuation lifecycle management
- Signal waiting mechanism
- Changeset commit
- Conflict handling
"""

import pytest
import asyncio
from linj_autogen.contitext.engine import ContiTextEngine
from linj_autogen.contitext.continuation import Continuation, Status
from linj_autogen.contitext.signal import Signal, WaitCondition
from linj_autogen.core.changeset import ChangeSet
from linj_autogen.core.state import StateManager
from linj_autogen.core.errors import HandleExpired, ConflictError


class TestContinuationLifecycle:
    """Test continuation lifecycle (Sections 17-18)"""

    def test_continuation_creation(self):
        """Test continuation creation"""
        engine = ContiTextEngine()

        # Create root continuation
        root = engine.derive()
        assert root.status == Status.RUNNING
        assert root.handle is not None
        assert root.parent_handle is None
        assert root.step_id == 0
        assert root.base_revision == 0

        # Create child continuation
        child = engine.derive(root)
        assert child.parent_handle == root.handle
        assert child.step_id == root.step_id
        assert child.base_revision == root.base_revision

    def test_continuation_states(self):
        """Test continuation state transitions"""
        engine = ContiTextEngine()
        cont = engine.derive()

        # Initial state
        assert cont.status == Status.RUNNING
        assert cont.is_active()
        assert not cont.is_terminal()
        assert cont.can_submit_changes()

        # Suspend
        engine.suspend(cont)
        assert cont.status == Status.SUSPENDED
        assert cont.is_active()
        assert not cont.is_terminal()
        assert cont.can_submit_changes()

        # Resume
        engine.resume(cont.handle)
        assert cont.status == Status.RUNNING
        assert cont.is_active()
        assert not cont.is_terminal()
        assert cont.can_submit_changes()

        # Complete
        engine.complete(cont.handle, "result")
        assert cont.status == Status.COMPLETED
        assert not cont.is_active()
        assert cont.is_terminal()
        assert cont.can_submit_changes()
        assert cont.result == "result"

        # Cancel
        cont2 = engine.derive()
        engine.cancel(cont2.handle)
        assert cont2.status == Status.CANCELLED
        assert not cont2.is_active()
        assert cont2.is_terminal()
        assert not cont2.can_submit_changes()

    def test_continuation_view(self):
        """Test continuation view (Section 18.2)"""
        state_manager = StateManager({"$.test": "value"})
        engine = ContiTextEngine(state_manager)
        cont = engine.derive()

        # Create view
        view = engine.create_view(cont)
        assert view is not None

        # Test read
        assert view.read("$.test") == "value"
        assert view.exists("$.test")
        assert view.len("$.nonexistent") == 0

        # Test logical snapshot
        snapshot = cont.get_logical_snapshot()
        assert snapshot["test"] == "value"


class TestSignalMechanism:
    """Test signal mechanism (Section 21)"""

    def test_signal_creation(self):
        """Test signal creation"""
        signal = Signal(name="test", payload={"data": "value"}, correlation="id123")
        assert signal.name == "test"
        assert signal.payload == {"data": "value"}
        assert signal.correlation == "id123"

    def test_wait_condition_matching(self):
        """Test wait condition matching"""
        condition = WaitCondition(name="test_signal")
        signal = Signal(name="test_signal", payload="data")

        # Exact match
        assert condition.matches(signal)

        # Name mismatch
        wrong_signal = Signal(name="other_signal")
        assert not condition.matches(wrong_signal)

    def test_correlation_matching(self):
        """Test correlation identifier matching"""
        condition = WaitCondition(correlation="id123")
        signal1 = Signal(name="any", correlation="id123")
        signal2 = Signal(name="any", correlation="id456")

        assert condition.matches(signal1)
        assert not condition.matches(signal2)

    def test_predicate_matching(self):
        """Test predicate matching"""
        condition = WaitCondition(predicate='value("$.signal.payload") == "expected"')
        signal = Signal(name="test", payload="expected")
        state = {"signal": {"payload": "expected"}}

        assert condition.matches(signal, state)

        # Predicate mismatch
        signal2 = Signal(name="test", payload="different")
        assert not condition.matches(signal2, state)


class TestChangeSetCommit:
    """Test changeset commit (Section 20)"""

    def test_atomic_commit(self):
        """Test atomic commit"""
        state_manager = StateManager({"$.counter": 0})
        engine = ContiTextEngine(state_manager)

        # Create changeset
        changeset = ChangeSet.create_write("$.counter", 1)

        # Commit changeset
        result = engine.submit_changeset(step_id=1, changeset=changeset, handle="test")

        assert result.success
        assert result.step_id == 1
        assert result.new_revision == 1
        assert state_manager.get("$.counter") == 1

    def test_conflict_detection(self):
        """Test conflict detection"""
        state_manager = StateManager({"$.value": "original"})
        engine = ContiTextEngine(state_manager)

        # First changeset
        changeset1 = ChangeSet.create_write("$.value", "first")
        result1 = engine.submit_changeset(
            step_id=1, changeset=changeset1, handle="test1"
        )
        assert result1.success

        # Second changeset (same path, different step_id)
        changeset2 = ChangeSet.create_write("$.value", "second")
        result2 = engine.submit_changeset(
            step_id=2, changeset=changeset2, handle="test2"
        )
        assert result2.success

        # Verify final state (applied in step_id order)
        assert state_manager.get("$.value") == "second"

    def test_base_revision_mismatch(self):
        """Test base revision mismatch"""
        state_manager = StateManager({"$.value": "original"})
        engine = ContiTextEngine(state_manager)

        # Manually modify revision to simulate conflict
        state_manager._revision = 5

        changeset = ChangeSet.create_write("$.value", "new")
        result = engine.submit_changeset(
            step_id=1,
            changeset=changeset,
            handle="test",
            base_revision=3,  # Mismatched base revision
        )

        assert not result.success
        assert "Base revision mismatch" in str(result.error)


class TestContinuationEngine:
    """Test ContiText engine integration"""

    @pytest.mark.asyncio
    async def test_derive_and_cancel_propagation(self):
        """Test derive and cancel propagation (Section 19.5)"""
        engine = ContiTextEngine()

        # Create parent and child continuations
        parent = engine.derive()
        child1 = engine.derive(parent)
        child2 = engine.derive(parent)

        # Cancel parent continuation
        engine.cancel(parent.handle)

        # Verify propagation
        assert parent.status == Status.CANCELLED
        assert child1.status == Status.CANCELLED
        assert child2.status == Status.CANCELLED

    @pytest.mark.asyncio
    async def test_join_completion(self):
        """Test join completion (Section 19.4)"""
        engine = ContiTextEngine()

        # Create multiple continuations
        cont1 = engine.derive()
        cont2 = engine.derive()
        cont3 = engine.derive()

        # Complete continuations
        engine.complete(cont1.handle, "result1")
        engine.complete(cont2.handle, "result2")
        engine.fail(cont3.handle, "error")

        # Join wait
        results = await engine.join([cont1.handle, cont2.handle, cont3.handle])

        assert len(results) == 3
        assert results[0].status == Status.COMPLETED
        assert results[0].result == "result1"
        assert results[1].status == Status.COMPLETED
        assert results[1].result == "result2"
        assert results[2].status == Status.FAILED
        assert results[2].error == "error"

    def test_signal_waiting(self):
        """Test signal waiting"""
        engine = ContiTextEngine()
        cont = engine.derive()

        # Register wait condition
        condition = WaitCondition(name="expected_signal")
        engine.suspend(cont, wait_condition=condition)

        # Send non-matching signal
        wrong_signal = Signal(name="wrong_signal")
        engine.send_signal(wrong_signal)

        # Check wait (should not match)
        state = {}
        matched = engine.check_signal(cont.handle, state)
        assert matched is None

        # Send matching signal
        right_signal = Signal(name="expected_signal")
        engine.send_signal(right_signal)

        # Check wait (should match)
        matched = engine.check_signal(cont.handle, state)
        assert matched is not None
        assert matched.name == "expected_signal"


class TestErrorHandling:
    """Test error handling"""

    def test_handle_expired(self):
        """Test handle expiration"""
        engine = ContiTextEngine()

        # Try to resume non-existent continuation
        with pytest.raises(HandleExpired):
            engine.resume("nonexistent_handle")

        # Try to cancel non-existent continuation (should return silently)
        engine.cancel("nonexistent_handle")  # Should not raise exception

    def test_invalid_state_transitions(self):
        """Test invalid state transitions"""
        engine = ContiTextEngine()
        cont = engine.derive()

        # Try to resume non-suspended continuation
        with pytest.raises(ValueError):
            engine.resume(cont.handle)

        # Try to complete already cancelled continuation
        engine.cancel(cont.handle)
        with pytest.raises(ConflictError):
            engine.complete(cont.handle, "result")

        # Try to fail already cancelled continuation
        cont2 = engine.derive()
        engine.cancel(cont2.handle)
        with pytest.raises(ConflictError):
            engine.fail(cont2.handle, "error")


class TestPerformanceAndLimits:
    """Test performance and limits"""

    def test_large_number_of_continuations(self):
        """Test large number of continuations"""
        engine = ContiTextEngine()

        # Create large number of continuations
        continuations = []
        for i in range(1000):
            cont = engine.derive()
            continuations.append(cont)

        # Verify all continuations were created correctly
        assert len(continuations) == 1000
        for cont in continuations:
            assert cont.status == Status.RUNNING
            assert cont.handle is not None

    def test_concurrent_operations(self):
        """Test concurrent operations"""
        import threading
        import time

        engine = ContiTextEngine()
        results = []

        def create_continuations(start_id, count):
            for i in range(count):
                cont = engine.derive()
                results.append(cont.handle)

        # Concurrent continuation creation
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_continuations, args=(i * 100, 100))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 1000
        assert len(set(results)) == 1000  # All handles are unique
