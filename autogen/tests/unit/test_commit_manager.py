"""
CommitManager 单元测试

测试范围：
- 基本提交功能
- 只读优化（空变更集）
- 非相交优化
- 基准修订检查
- 冲突处理
- 队列处理
- 线程安全
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
    """基本功能测试"""

    def test_init(self):
        """测试初始化"""
        state = StateManager()
        manager = CommitManager(state)

        assert manager._state_manager == state
        assert manager._next_expected_step == 1
        assert manager.get_pending_count() == 0
        assert manager.get_accepted_count() == 0

    def test_submit_single_changeset(self):
        """测试提交单个变更集"""
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
        """测试空变更集的只读优化"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        cs = ChangeSet()  # 空变更集
        result = manager.submit(
            step_id=1, base_revision=0, changeset=cs, handle="handle-1"
        )

        assert result.success is True
        assert result.step_id == 1
        assert result.new_revision == 0  # 空变更集不改变版本
        assert manager.get_accepted_count() == 1

    def test_submit_out_of_order(self):
        """测试乱序提交"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 先提交 step 2（与 step 1 不相交）
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # 由于 step 1 未提交，无法判断是否相交，step 2 应该排队
        assert result2.success is False
        assert manager.get_pending_count() == 1

        # 提交 step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 应该成功，step 2 应该被自动处理（因为路径不相交）
        assert result1.success is True
        assert manager.get_accepted_count() == 2
        assert state.get("$.data.a") == 1
        assert state.get("$.data.b") == 2


class TestBaseRevisionCheck:
    """基准修订检查测试"""

    def test_base_revision_mismatch(self):
        """测试基准版本不匹配"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 先提交 step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # 尝试用旧版本提交 step 2
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # 应该失败，因为当前版本已经是 1
        assert result2.success is False
        assert isinstance(result2.error, ConflictError)
        assert "Base revision mismatch" in str(result2.error.message)

    def test_base_revision_match(self):
        """测试基准版本匹配"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 提交 step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # 用正确版本提交 step 2
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        result2 = manager.submit(step_id=2, base_revision=1, changeset=cs2, handle="h2")

        assert result1.success is True
        assert result2.success is True


class TestNonIntersectionOptimization:
    """非相交优化测试"""

    def test_non_intersecting_changesets_can_accept_early(self):
        """测试不相交变更集在缺口填补后可以被接受"""
        state = StateManager({"a": {}, "b": {}})
        manager = CommitManager(state)

        # 提交 step 2（不相交路径）
        cs2 = ChangeSet(writes=[WriteOp(path="$.b.value", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # 由于 step 1 缺失，必须排队等待以确保安全
        assert result2.success is False
        assert manager.get_pending_count() == 1

        # 提交 step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.a.value", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        assert result1.success is True
        # step 1 完成后，step 2 应该被自动处理并接受（因为不相交）
        assert manager.get_accepted_count() == 2
        assert state.get("$.b.value") == 2

    def test_intersecting_changesets_cannot_accept_early(self):
        """测试相交变更集不能提前接受"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 先提交 step 2（与 step 1 相交的路径）
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.value", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # 由于 step 1 还未提交，无法判断是否相交，step 2 应该排队
        assert result2.success is False
        assert manager.get_pending_count() == 1

        # 提交 step 1（相交路径）
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.value", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 应该成功
        assert result1.success is True

        # 此时 check_intersection_since_revision 应该发现冲突
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
        """测试父子路径相交"""
        state = StateManager({"data": {"nested": {}}})
        manager = CommitManager(state)

        # step 2 修改父路径（与 step 1 的子路径相交）
        cs2 = ChangeSet(writes=[WriteOp(path="$.data", value={"new": 1})])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # step 1 还未提交，无法判断，应该排队
        assert result2.success is False

        # step 1 修改子路径
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.nested.value", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        assert result1.success is True

        # intersection -> rejected
        res2 = manager.get_result(2)
        assert res2 is not None
        assert res2.success is False

    def test_array_index_non_intersection(self):
        """测试数组下标不相交"""
        state = StateManager({"items": [0, 0, 0]})
        manager = CommitManager(state)

        # step 2 修改 items[1]
        cs2 = ChangeSet(writes=[WriteOp(path="$.items[1]", value=2)])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # 缺失 step 1 -> 排队
        assert result2.success is False

        # Submit step 1 (disjoint)
        cs1 = ChangeSet(writes=[WriteOp(path="$.items[0]", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        assert result1.success is True
        # step 2 accepted
        assert manager.get_accepted_count() == 2
        assert state.get("$.items[1]") == 2

    def test_delete_operation_intersection(self):
        """测试删除操作不相交"""
        state = StateManager({"data": {"a": 1, "b": 2}})
        manager = CommitManager(state)

        # step 2 删除 data.a（与 step 1 的 data.b 不相交）
        cs2 = ChangeSet(deletes=[DeleteOp(path="$.data.a")])
        result2 = manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # step 1 还未提交，无法判断，应该排队
        assert result2.success is False

        # step 1 写入 data.b（与 step 2 的删除路径不相交）
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.b", value=10)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 成功，step 2 应该被自动处理（因为路径不相交）
        assert manager.get_accepted_count() == 2
        assert state.get("$.data.a") is None  # 已删除
        assert state.get("$.data.b") == 10


class TestQueueProcessing:
    """队列处理测试"""

    def test_process_queue_empty(self):
        """测试处理空队列"""
        state = StateManager()
        manager = CommitManager(state)

        results = manager.process_queue()
        assert len(results) == 0

    def test_process_queue_with_pending(self):
        """测试处理有等待项的队列"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 先让 step 2 排队（与 step 1 不相交）
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # step 2 在队列中
        assert manager.get_pending_count() == 1

        # 提交 step 1（不相交路径）
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 成功，step 2 应该被自动处理（因为路径不相交）
        assert manager.get_accepted_count() == 2

    def test_process_queue_outdated_revision(self):
        """测试处理时遇到过时版本（不相交优化应允许通过）"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # step 2 用旧版本排队
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        assert manager.get_pending_count() == 1

        # 提交 step 1
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 成功。step 2 不相交，可以合并。
        # 即使 base_revision=0 < current=1.

        # step 2 should be ACCEPTED because of non-intersection check logic
        # in _process_queue_internal

        assert manager.get_accepted_count() == 2
        res2 = manager.get_result(2)
        assert res2.success is True


class TestGetters:
    """Getter 方法测试"""

    def test_get_pending(self):
        """测试获取待处理列表"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 提交一些变更集
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        manager.submit(step_id=2, base_revision=1, changeset=cs2, handle="h2")

        pending = manager.get_pending()
        assert len(pending) == 0  # 都被接受了

        # 让 step 3 排队
        cs3 = ChangeSet(writes=[WriteOp(path="$.data.c", value=3)])
        manager.submit(step_id=3, base_revision=2, changeset=cs3, handle="h3")

        # 如果版本不匹配，它会被拒绝而不是排队
        pending = manager.get_pending()
        assert len(pending) == 0

    def test_get_result(self):
        """测试获取结果"""
        state = StateManager()
        manager = CommitManager(state)

        cs = ChangeSet(writes=[WriteOp(path="$.value", value=42)])
        manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")

        result = manager.get_result(1)
        assert result is not None
        assert result.success is True
        assert result.step_id == 1

        # 不存在的 step_id
        assert manager.get_result(999) is None

    def test_is_all_accepted(self):
        """测试检查是否全部接受"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 空状态
        assert manager.is_all_accepted() is False

        # 提交一个
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.a", value=1)])
        manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        assert manager.is_all_accepted() is True

        # 添加一个待处理的（由于版本不匹配会被拒绝）
        cs2 = ChangeSet(writes=[WriteOp(path="$.data.b", value=2)])
        manager.submit(step_id=2, base_revision=0, changeset=cs2, handle="h2")

        # step 2 被拒绝，所以只有一个被接受
        # 注意：被拒绝的也算 pending，但不是 accepted

    def test_reset(self):
        """测试重置"""
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
    """线程安全测试 - 简化版本"""

    def test_lock_protection(self):
        """测试锁保护基本功能"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 验证锁存在且可访问
        assert manager._lock is not None

        # 测试基本提交（单线程）
        cs = ChangeSet(writes=[WriteOp(path="$.data.value", value=42)])
        result = manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")
        assert result.success is True


class TestComplexScenarios:
    """复杂场景测试"""

    def test_mixed_operations(self):
        """测试混合操作"""
        state = StateManager({"data": {"a": 1, "b": 2, "c": 3}})
        manager = CommitManager(state)

        # 写入、删除、混合
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
        """测试大间隔 step_id"""
        state = StateManager({"data": {}})
        manager = CommitManager(state)

        # 提交 step 100（与 step 1 不相交）
        cs100 = ChangeSet(writes=[WriteOp(path="$.data.x", value=100)])
        result100 = manager.submit(
            step_id=100, base_revision=0, changeset=cs100, handle="h100"
        )

        # step 1 未提交，无法判断，应该排队
        assert result100.success is False

        # 提交 step 1（不相交路径）
        cs1 = ChangeSet(writes=[WriteOp(path="$.data.y", value=1)])
        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs1, handle="h1")

        # step 1 成功
        assert result1.success is True

        # step 100 仍然有 gap (2..99)，应该继续 pending
        assert manager.get_accepted_count() == 1
        res100 = manager.get_result(100)
        # 结果可能为 None 或 success=False (queued)
        if res100:
            assert res100.success is False

    def test_duplicate_submission(self):
        """测试重复提交"""
        state = StateManager()
        manager = CommitManager(state)

        cs = ChangeSet(writes=[WriteOp(path="$.value", value=42)])

        result1 = manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")
        result2 = manager.submit(step_id=1, base_revision=0, changeset=cs, handle="h1")

        # 第二次应该返回缓存的结果
        assert result1.success == result2.success
        assert result1.new_revision == result2.new_revision
