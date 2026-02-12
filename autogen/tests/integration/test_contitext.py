"""
ContiText 续体机制测试

验证 17-22 节定义的续体行为：
- 续体生命周期管理
- 信号等待机制
- 变更集提交
- 冲突处理
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
    """测试续体生命周期 (17-18 节)"""

    def test_continuation_creation(self):
        """测试续体创建"""
        engine = ContiTextEngine()

        # 创建根续体
        root = engine.derive()
        assert root.status == Status.RUNNING
        assert root.handle is not None
        assert root.parent_handle is None
        assert root.step_id == 0
        assert root.base_revision == 0

        # 创建子续体
        child = engine.derive(root)
        assert child.parent_handle == root.handle
        assert child.step_id == root.step_id
        assert child.base_revision == root.base_revision

    def test_continuation_states(self):
        """测试续体状态转换"""
        engine = ContiTextEngine()
        cont = engine.derive()

        # 初始状态
        assert cont.status == Status.RUNNING
        assert cont.is_active()
        assert not cont.is_terminal()
        assert cont.can_submit_changes()

        # 挂起
        engine.suspend(cont)
        assert cont.status == Status.SUSPENDED
        assert cont.is_active()
        assert not cont.is_terminal()
        assert cont.can_submit_changes()

        # 恢复
        engine.resume(cont.handle)
        assert cont.status == Status.RUNNING
        assert cont.is_active()
        assert not cont.is_terminal()
        assert cont.can_submit_changes()

        # 完成
        engine.complete(cont.handle, "result")
        assert cont.status == Status.COMPLETED
        assert not cont.is_active()
        assert cont.is_terminal()
        assert cont.can_submit_changes()
        assert cont.result == "result"

        # 取消
        cont2 = engine.derive()
        engine.cancel(cont2.handle)
        assert cont2.status == Status.CANCELLED
        assert not cont2.is_active()
        assert cont2.is_terminal()
        assert not cont2.can_submit_changes()

    def test_continuation_view(self):
        """测试续体视图 (18.2 节)"""
        state_manager = StateManager({"$.test": "value"})
        engine = ContiTextEngine(state_manager)
        cont = engine.derive()

        # 创建视图
        view = engine.create_view(cont)
        assert view is not None

        # 测试读取
        assert view.read("$.test") == "value"
        assert view.exists("$.test")
        assert view.len("$.nonexistent") == 0

        # 测试逻辑快照
        snapshot = cont.get_logical_snapshot()
        assert snapshot["test"] == "value"


class TestSignalMechanism:
    """测试信号机制 (21 节)"""

    def test_signal_creation(self):
        """测试信号创建"""
        signal = Signal(name="test", payload={"data": "value"}, correlation="id123")
        assert signal.name == "test"
        assert signal.payload == {"data": "value"}
        assert signal.correlation == "id123"

    def test_wait_condition_matching(self):
        """测试等待条件匹配"""
        condition = WaitCondition(name="test_signal")
        signal = Signal(name="test_signal", payload="data")

        # 精确匹配
        assert condition.matches(signal)

        # 名称不匹配
        wrong_signal = Signal(name="other_signal")
        assert not condition.matches(wrong_signal)

    def test_correlation_matching(self):
        """测试关联标识匹配"""
        condition = WaitCondition(correlation="id123")
        signal1 = Signal(name="any", correlation="id123")
        signal2 = Signal(name="any", correlation="id456")

        assert condition.matches(signal1)
        assert not condition.matches(signal2)

    def test_predicate_matching(self):
        """测试谓词匹配"""
        condition = WaitCondition(predicate='value("$.signal.payload") == "expected"')
        signal = Signal(name="test", payload="expected")
        state = {"signal": {"payload": "expected"}}

        assert condition.matches(signal, state)

        # 谓词不匹配
        signal2 = Signal(name="test", payload="different")
        assert not condition.matches(signal2, state)


class TestChangeSetCommit:
    """测试变更集提交 (20 节)"""

    def test_atomic_commit(self):
        """测试原子性提交"""
        state_manager = StateManager({"$.counter": 0})
        engine = ContiTextEngine(state_manager)

        # 创建变更集
        changeset = ChangeSet.create_write("$.counter", 1)

        # 提交变更集
        result = engine.submit_changeset(step_id=1, changeset=changeset, handle="test")

        assert result.success
        assert result.step_id == 1
        assert result.new_revision == 1
        assert state_manager.get("$.counter") == 1

    def test_conflict_detection(self):
        """测试冲突检测"""
        state_manager = StateManager({"$.value": "original"})
        engine = ContiTextEngine(state_manager)

        # 第一个变更集
        changeset1 = ChangeSet.create_write("$.value", "first")
        result1 = engine.submit_changeset(
            step_id=1, changeset=changeset1, handle="test1"
        )
        assert result1.success

        # 第二个变更集（相同路径，不同 step_id）
        changeset2 = ChangeSet.create_write("$.value", "second")
        result2 = engine.submit_changeset(
            step_id=2, changeset=changeset2, handle="test2"
        )
        assert result2.success

        # 验证最终状态（按 step_id 顺序应用）
        assert state_manager.get("$.value") == "second"

    def test_base_revision_mismatch(self):
        """测试基准版本不匹配"""
        state_manager = StateManager({"$.value": "original"})
        engine = ContiTextEngine(state_manager)

        # 手动修改版本号模拟冲突
        state_manager._revision = 5

        changeset = ChangeSet.create_write("$.value", "new")
        result = engine.submit_changeset(
            step_id=1,
            changeset=changeset,
            handle="test",
            base_revision=3,  # 不匹配的基准版本
        )

        assert not result.success
        assert "Base revision mismatch" in str(result.error)


class TestContinuationEngine:
    """测试 ContiText 引擎集成"""

    @pytest.mark.asyncio
    async def test_derive_and_cancel_propagation(self):
        """测试派生和取消传播 (19.5 节)"""
        engine = ContiTextEngine()

        # 创建父子续体
        parent = engine.derive()
        child1 = engine.derive(parent)
        child2 = engine.derive(parent)

        # 取消父续体
        engine.cancel(parent.handle)

        # 验证传播
        assert parent.status == Status.CANCELLED
        assert child1.status == Status.CANCELLED
        assert child2.status == Status.CANCELLED

    @pytest.mark.asyncio
    async def test_join_completion(self):
        """测试合流完成 (19.4 节)"""
        engine = ContiTextEngine()

        # 创建多个续体
        cont1 = engine.derive()
        cont2 = engine.derive()
        cont3 = engine.derive()

        # 完成续体
        engine.complete(cont1.handle, "result1")
        engine.complete(cont2.handle, "result2")
        engine.fail(cont3.handle, "error")

        # 合流等待
        results = await engine.join([cont1.handle, cont2.handle, cont3.handle])

        assert len(results) == 3
        assert results[0].status == Status.COMPLETED
        assert results[0].result == "result1"
        assert results[1].status == Status.COMPLETED
        assert results[1].result == "result2"
        assert results[2].status == Status.FAILED
        assert results[2].error == "error"

    def test_signal_waiting(self):
        """测试信号等待"""
        engine = ContiTextEngine()
        cont = engine.derive()

        # 注册等待条件
        condition = WaitCondition(name="expected_signal")
        engine.suspend(cont, wait_condition=condition)

        # 发送不匹配的信号
        wrong_signal = Signal(name="wrong_signal")
        engine.send_signal(wrong_signal)

        # 检查等待（应该没有匹配）
        state = {}
        matched = engine.check_signal(cont.handle, state)
        assert matched is None

        # 发送匹配的信号
        right_signal = Signal(name="expected_signal")
        engine.send_signal(right_signal)

        # 检查等待（应该匹配）
        matched = engine.check_signal(cont.handle, state)
        assert matched is not None
        assert matched.name == "expected_signal"


class TestErrorHandling:
    """测试错误处理"""

    def test_handle_expired(self):
        """测试句柄过期"""
        engine = ContiTextEngine()

        # 尝试恢复不存在的续体
        with pytest.raises(HandleExpired):
            engine.resume("nonexistent_handle")

        # 尝试取消不存在的续体（应该静默返回）
        engine.cancel("nonexistent_handle")  # 不应该抛出异常

    def test_invalid_state_transitions(self):
        """测试无效状态转换"""
        engine = ContiTextEngine()
        cont = engine.derive()

        # 尝试恢复非挂起状态的续体
        with pytest.raises(ValueError):
            engine.resume(cont.handle)

        # 尝试完成已取消的续体
        engine.cancel(cont.handle)
        with pytest.raises(ConflictError):
            engine.complete(cont.handle, "result")

        # 尝试失败已取消的续体
        cont2 = engine.derive()
        engine.cancel(cont2.handle)
        with pytest.raises(ConflictError):
            engine.fail(cont2.handle, "error")


class TestPerformanceAndLimits:
    """测试性能和限制"""

    def test_large_number_of_continuations(self):
        """测试大量续体"""
        engine = ContiTextEngine()

        # 创建大量续体
        continuations = []
        for i in range(1000):
            cont = engine.derive()
            continuations.append(cont)

        # 验证所有续体都被正确创建
        assert len(continuations) == 1000
        for cont in continuations:
            assert cont.status == Status.RUNNING
            assert cont.handle is not None

    def test_concurrent_operations(self):
        """测试并发操作"""
        import threading
        import time

        engine = ContiTextEngine()
        results = []

        def create_continuations(start_id, count):
            for i in range(count):
                cont = engine.derive()
                results.append(cont.handle)

        # 并发创建续体
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_continuations, args=(i * 100, 100))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 1000
        assert len(set(results)) == 1000  # 所有句柄都唯一
