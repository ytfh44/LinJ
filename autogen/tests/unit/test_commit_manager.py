"""
CommitManager Unit Tests

Test scope:
- Basic commit functionality
- Read-only optimization (empty changeset)
- Non-intersection optimization
- Base revision check
- Conflict handling
- Queue processing
- Thread safety
"""

import threading
import time
from typing import List

import pytest

from linj_autogen.contitext import CommitManager, PendingStatus
from linj_autogen.core.changeset import ChangeSet, WriteOp, DeleteOp
from linj_autogen.core.errors import ConflictError
from linj_autogen.core.state import StateManager


class TestCommitManagerBasic:
    """Basic functionality tests"""

    def test_init(self):
        """Test initialization"""
        state = StateManager()
        manager = CommitManager(state)

        assert manager._state_manager == state
        assert manager._next_expected_step == 1
        assert manager.get_pending_count() == 0
        assert manager.get_accepted_count() == 0

    def test_submit_single_changeset(self):
        """Test submitting a single changeset"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        cs = ChangeSet(writes=[WriteOp(path="$.data.value", value=42)])
        result = manager.submit(
            step_id=1, base_revision=0, changeset=cs, handle="handle-1"
        )

        assert result.success is True
        assert result.step_id == 1
        assert result.new_revision == 1
        assert state.get("$.data.value") == 42

    def test_submit_empty_changeset_readonly_optimization(self):
        """Test read-only optimization for empty changeset"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        cs = ChangeSet()  # Empty changeset
        result = manager.submit(
            step_id=1, base_revision=0, changeset=cs, handle="handle-1"
        )

        assert result.success is True
        assert result.step_id == 1
        assert result.new_revision == 0  # Empty changeset does not change revision
        assert manager.get_accepted_count() == 1

    def test_submit_out_of_order(self):
        """Test out-of-order submission"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Submit step 2 first (disjoint from step 1)
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # Since step 1 is not submitted, cannot determine intersection, step 2 should be queued
        assert result2.success is False
        assert manager.get_pending_count() == 1

        # Submit step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 should succeed, step 2 should be auto-processed (because paths are disjoint)
        assert result1.success is True
        assert manager.get_accepted_count() == 2
        assert state.get("$.data.a") == 1
        assert state.get("$.data.b") == 2


class TestBaseRevisionCheck:
    """Base revision check tests"""

    def test_base_revision_mismatch(self):
        """Test base version mismatch"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Submit step 1 first
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # Try to submit step 2 with old version
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # Should fail because current revision is already 1
        assert result2.success is False
        assert isinstance(result2.error, ConflictError)
        assert "Base revision mismatch" in str(result2.error.message)

    def test_base_revision_match(self):
        """Test base version match"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Submit step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # Submit step 2 with correct version
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        result2 = manager.submit(step_id=2, base_revision=1, changeset=cs2, handle="h2")

        assert result1.success is True
        assert result2.success is True


class TestNonIntersectionOptimization:
    """Non-intersection optimization tests"""

    def test_non_intersecting_changesets_can_accept_early(self):
        """Test non-intersecting changesets can be accepted after gap is filled"""
        state = StateManager({"a": {}, "b": {}})
        manager = CommitManager(state)

        # Submit step 2 (disjoint paths)
        cs2 = ChangeSet(writes=[WriteOp(path="$.b.value", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # Since step 1 is missing, must queue for safety
        assert result2.success is False
        assert manager.get_pending_count() == 1

        # Submit step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.a.value", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        assert result1.success is True
        # After step 1 completes, step 2 should be auto-processed and accepted (because disjoint)
        assert manager.get_accepted_count() == 2
        assert state.get("$.b.value") == 2

    def test_intersecting_changesets_cannot_accept_early(self):
        """Test intersecting changesets cannot be accepted early"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Submit step 2 first (paths that intersect with step 1)
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.value", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # Since step 1 is not yet submitted, cannot determine intersection, step 2 should queue
        assert result2.success is False
        assert manager.get_pending_count() == 1

        # Submit step 1 (intersecting paths)
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.value", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 should succeed
        assert result1.success is True

        # At this point, check_intersection_since_revision should find conflict
        # step 1 became revision 1. step 2 base=0.
        # step 2 intersects step 1. -> Rejected.
        assert manager.get_accepted_count() == 1
        # Step 2 status should be REJECTED.
        # Check results
        res2 = manager.get_result(2)
        assert res2 is not None
        assert res2.success is False
        assert isinstance(res2.error, ConflictError)

    def test_parent_child_path_intersection(self):
        """Test parent-child path intersection"""
        state = StateManager({"data": {"nested": {}}})
        manager = CommitManager(state)

        # step 2 modifies parent path (intersects with step 1's child path)
        cs2 = ChangeSet(writes=[WriteOp(path="$.data", value={"new": 1})])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # step 1 not submitted yet, cannot determine, should queue
        assert result2.success is False

        # step 1 modifies child path
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.nested.value", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        assert result1.success is True

        # intersection -> rejected
        res2 = manager.get_result(2)
        assert res2 is not None
        assert res2.success is False

    def test_array_index_non_intersection(self):
        """Test array index non-intersection"""
        state = StateManager({"items": [0, 0, 0]})
        manager = CommitManager(state)

        # step 2 modifies items[1]
        cs2 = ChangeSet(writes=[WriteOp(path="$.items[1]", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # Missing step 1 -> queue
        assert result2.success is False

        # Submit step 1 (disjoint)
        cs1 = ChangeSet(writes=[WriteOp(path="$.items[0]", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        assert result1.success is True
        # step 2 accepted
        assert manager.get_accepted_count() == 2
        assert state.get("$.items[1]") == 2

    def test_delete_operation_intersection(self):
        """Test delete operation non-intersection"""
        state = StateManager({"data": {"a": 1, "b": 2}})
        manager = CommitManager(state)

        # step 2 deletes data.a (disjoint from step 1's data.b)
        cs2 = ChangeSet(deletes=[DeleteOp(path="$.data.a")])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # step 1 not submitted yet, cannot determine, should queue
        assert result2.success is False

        # step 1 writes data.b (disjoint from step 2's delete path)
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.b", value=10)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 succeeds, step 2 should be auto-processed (because paths are disjoint)
        assert manager.get_accepted_count() == 2
        assert state.get("$.data.a") is None  # Deleted
        assert state.get("$.data.b") == 10


class TestQueueProcessing:
    """Queue processing tests"""

    def test_process_queue_empty(self):
        """Test processing empty queue"""
        state = StateManager()
        manager = CommitManager(state)

        results = manager.process_queue()
        assert len(results) == 0

    def test_process_queue_with_pending(self):
        """Test processing queue with pending items"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Let step 2 queue first (disjoint from step 1)
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # step 2 is in queue
        assert manager.get_pending_count() == 1

        # Submit step 1 (disjoint paths)
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 succeeds, step 2 should be auto-processed (because paths are disjoint)
        assert manager.get_accepted_count() == 2

    def test_process_queue_outdated_revision(self):
        """Test encountering outdated revision during processing (non-intersection optimization should allow through)"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # step 2 queues with old revision
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        assert manager.get_pending_count() == 1

        # Submit step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 succeeds. step 2 is disjoint, can merge.
        # Even though base_revision=0 < current=1.

        # step 2 should be ACCEPTED because of non-intersection check logic
        # in _process_queue_internal

        assert manager.get_accepted_count() == 2
        res2 = manager.get_result(2)
        assert res2 is not None
        assert res2.success is True


class TestGetters:
    """Getter methods tests"""

    def test_get_pending(self):
        """Test getting pending list"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Submit some changesets
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        manager.submit(step_id=2, base_revision=1, changeset=cs2, handle="h2")

        pending = manager.get_pending()
        assert len(pending) == 0  # All were accepted

        # Let step 3 queue
        cs3 = ChangeSet(writes=[WriteOp(path="$.data.c", value=3)])
        manager.submit(step_id=3, base_revision=2, changeset=cs3, handle="h3")

        # If revision doesn't match, it will be rejected rather than queued
        pending = manager.get_pending()
        assert len(pending) == 0

    def test_get_result(self):
        """Test getting results"""
        state = StateManager()
        manager = CommitManager(state)

        cs = ChangeSet(writes=[WriteOp(path="$.value", value=42)])
        manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")

        result = manager.get_result(1)
        assert result is not None
        assert result.success is True
        assert result.step_id == 1

        # Non-existent step_id
        assert manager.get_result(999) is None

    def test_is_all_accepted(self):
        """Test checking if all are accepted"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Empty state
        assert manager.is_all_accepted() is False

        # Submit one
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        assert manager.is_all_accepted() is True

        # Add a pending one (will be rejected due to revision mismatch)
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # step 2 is rejected, so only one is accepted
        # Note: rejected also counts as pending, but not accepted

    def test_reset(self):
        """Test reset"""
        state = StateManager()
        manager = CommitManager(state)

        cs = ChangeSet(writes=[WriteOp(path="$.value", value=42)])
        manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")

        assert manager.get_accepted_count() == 1

        manager.reset()

        assert manager.get_accepted_count() == 0
        assert manager.get_pending_count() == 0
        assert manager._next_expected_step == 1


class TestThreadSafety:
    """Thread safety tests - simplified version"""

    def test_lock_protection(self):
        """Test lock protection basic functionality"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Verify lock exists and is accessible
        assert manager._lock is not None

        # Test basic commit (single-threaded)
        cs = ChangeSet(writes=[WriteOp(path="$.data.value", value=42)])
        result = manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")
        assert result.success is True


class TestComplexScenarios:
    """Complex scenarios tests"""

    def test_mixed_operations(self):
        """Test mixed operations"""
        state = StateManager({"data": {"a": 1, "b": 2, "c": 3}})
        manager = CommitManager(state)

        # Write, delete, mixed
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=10)])
        cs2 = ChangeSet(deletes=[DeleteOp(path="$.data.b")])
        cs3 = ChangeSet(
            writes=[WriteOp(path="$.data.c", value=30)],
            deletes=[DeleteOp(path="$.data.d")],
        )

        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")
        manager.submit(step_id=2, base_revision=1, changeset=cs2, handle="h2")
        manager.submit(step_id=3, base_revision=2, changeset=cs3, handle="h3")

        assert state.get("$.data.a") == 10
        assert state.get("$.data.b") is None
        assert state.get("$.data.c") == 30

    def test_large_step_id_gap(self):
        """Test large step_id gap"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # Submit step 100 (disjoint from step 1)
        cs100 = ChangeSet(writes=[WriteOp(path="$.data.x", value=100)])
        result100 = manager.submit(
            step_id=100, base_revision=0, changeset=cs100, handle="h100"
        )

        # step 1 not submitted, cannot determine, should queue
        assert result100.success is False

        # Submit step 1 (disjoint paths)
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.y", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 succeeds
        assert result1.success is True

        # step 100 still has gap (2..99), should remain pending
        assert manager.get_accepted_count() == 1
        res100 = manager.get_result(100)
        # Result may be None or success=False (queued)
        if res100:
            assert res100.success is False

    def test_duplicate_submission(self):
        """Test duplicate submission"""
        state = StateManager()
        manager = CommitManager(state)

        cs = ChangeSet(writes=[WriteOp(path="$.value", value=42)])

        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")
        result2 = manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")

        # Second call should return cached result
        assert result1.success == result2.success
        assert result1.new_revision == result2.new_revision
