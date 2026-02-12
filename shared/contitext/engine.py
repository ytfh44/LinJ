"""
ContiText 引擎

实现 LinJ 规范第 19 节定义的基本操作：派生、挂起、恢复、合流、取消
集成 CommitManager 实现决定性变更提交
框架无关的续体执行引擎实现
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import logging

from .continuation import Continuation, Status, ContinuationView, StateManager
from .signal import Signal, WaitCondition, SignalQueue
from .commit_manager import CommitManager, CommitResult

logger = logging.getLogger(__name__)


class JoinResult:
    """合流结果 (19.4 节)"""

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
    """变更集协议，用于框架无关的变更集操作"""

    def is_empty(self) -> bool:
        """检查变更集是否为空"""
        ...

    def intersects_with(self, other: Any) -> bool:
        """检查是否与另一个变更集相交"""
        ...

    def apply_to_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """将变更集应用到状态"""
        ...


class ContiTextEngine:
    """
    ContiText 执行引擎

    管理续体的生命周期：派生、挂起、恢复、合流、取消
    集成决定性变更提交管理
    框架无关的引擎实现
    """

    def __init__(self, state_manager: Optional[StateManager] = None):
        """
        初始化 ContiText 引擎

        Args:
            state_manager: 状态管理器，用于变更集提交
        """
        self._continuations: Dict[str, Continuation] = {}
        self._signal_queue = SignalQueue()
        self._children: Dict[str, List[str]] = {}  # parent -> children

        # 状态管理和变更提交
        if state_manager is None:
            # 创建一个简单的默认状态管理器
            self._state_manager = self._create_default_state_manager()
        else:
            self._state_manager = state_manager

        self._commit_manager = CommitManager(self._state_manager)

        # 待提交的变更集（按 step_id 排序）
        self._pending_changes: Dict[int, Any] = {}

        logger.info("ContiText engine initialized")

    def _create_default_state_manager(self) -> StateManager:
        """创建默认的状态管理器实现"""

        class DefaultStateManager:
            def __init__(self):
                self._state: Dict[str, Any] = {}
                self._revision: int = 0

            def get_full_state(self) -> Dict[str, Any]:
                return self._state.copy()

            def get_revision(self) -> int:
                return self._revision

            def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
                # 简单的变更集应用逻辑
                if hasattr(changeset, "apply_to_state"):
                    self._state = changeset.apply_to_state(self._state)
                else:
                    # 简单的状态更新逻辑
                    if isinstance(changeset, dict):
                        self._state.update(changeset)
                self._revision += 1

        return DefaultStateManager()

    def derive(self, parent: Optional[Continuation] = None) -> Continuation:
        """
        派生子续体 (19.1 节)

        Args:
            parent: 父续体，None 表示创建根续体

        Returns:
            新创建的续体
        """
        cont = Continuation(
            parent_handle=parent.handle if parent else None,
            step_id=parent.step_id if parent else 0,
            base_revision=parent.base_revision if parent else 0,
        )

        self._continuations[cont.handle] = cont

        # 记录父子关系
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
        挂起续体 (19.2 节)

        Args:
            cont: 要挂起的续体
            changeset: 挂起时产生的变更集
            wait_condition: 等待条件（如果有）
        """
        if cont.handle not in self._continuations:
            raise ValueError(f"Continuation {cont.handle} not found")

        cont.status = Status.SUSPENDED
        cont.pending_changeset = changeset

        # 注册等待者
        if wait_condition:
            self._signal_queue.register_waiter(cont.handle, wait_condition)

        logger.debug(f"Suspended continuation: {cont.handle}")

    def resume(self, handle: str, input_data: Optional[Dict] = None) -> Continuation:
        """
        恢复续体 (19.3 节)

        Args:
            handle: 续体句柄
            input_data: 注入的输入数据

        Returns:
            恢复后的续体
        """
        if handle not in self._continuations:
            raise ValueError(f"Continuation {handle} expired or invalid")

        cont = self._continuations[handle]

        if cont.status != Status.SUSPENDED:
            raise ValueError(f"Cannot resume continuation in state: {cont.status}")

        cont.status = Status.RUNNING

        # 注入输入数据
        if input_data:
            cont.local_state.update(input_data)

        # 取消等待注册
        self._signal_queue.unregister_waiter(handle)

        logger.debug(f"Resumed continuation: {handle}")
        return cont

    async def join(self, handles: List[str]) -> List[JoinResult]:
        """
        合流等待 (19.4 节)

        等待所有指定续体到达终态

        Args:
            handles: 续体句柄列表

        Returns:
            每个续体的合流结果
        """
        import asyncio

        results = []

        for handle in handles:
            if handle not in self._continuations:
                raise ValueError(f"Continuation {handle} not found")

            cont = self._continuations[handle]

            # 等待续体到达终态
            while not cont.is_terminal():
                await asyncio.sleep(0.01)  # 简单轮询

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
        取消续体 (19.5 节)

        - 幂等：重复取消不产生新副作用
        - 传播：主续体被取消时，子续体也应被取消
        - 取消后不得提交变更

        Args:
            handle: 要取消的续体句柄
        """
        if handle not in self._continuations:
            return  # 幂等：不存在的句柄静默返回

        cont = self._continuations[handle]

        # 如果已经是终态，不再处理
        if cont.is_terminal():
            return

        # 标记为取消
        cont.status = Status.CANCELLED

        logger.debug(f"Cancelled continuation: {handle}")

        # 递归取消子续体
        if handle in self._children:
            for child_handle in self._children[handle]:
                self.cancel(child_handle)

    def complete(self, handle: str, result: Any) -> None:
        """标记续体成功完成"""
        if handle not in self._continuations:
            raise ValueError(f"Continuation {handle} not found")

        cont = self._continuations[handle]

        if cont.status == Status.CANCELLED:
            raise ValueError("Cannot complete cancelled continuation")

        cont.status = Status.COMPLETED
        cont.result = result

        logger.debug(f"Completed continuation: {handle}")

    def fail(self, handle: str, error: str) -> None:
        """标记续体失败"""
        if handle not in self._continuations:
            raise ValueError(f"Continuation {handle} not found")

        cont = self._continuations[handle]

        if cont.status == Status.CANCELLED:
            raise ValueError("Cannot fail cancelled continuation")

        cont.status = Status.FAILED
        cont.error = error

        logger.debug(f"Failed continuation: {handle}, error: {error}")

    def send_signal(self, signal: Signal) -> None:
        """发送信号"""
        self._signal_queue.send(signal)
        logger.debug(f"Sent signal: {signal.name}")

    def check_signal(self, handle: str, state: Dict) -> Optional[Signal]:
        """检查等待者是否等到了信号"""
        if handle not in self._continuations:
            return None

        cont = self._continuations[handle]
        if cont.status != Status.SUSPENDED:
            return None

        # 查找匹配的信号
        for signal in self._signal_queue._signals:
            # 检查是否有等待条件
            wait_condition = self._signal_queue._waiters.get(handle)
            if wait_condition:
                if wait_condition.matches(signal, state):
                    logger.debug(
                        f"Signal matched for continuation {handle}: {signal.name}"
                    )
                    return signal
            else:
                # 没有等待条件，返回第一个信号
                logger.debug(
                    f"Signal received for continuation {handle}: {signal.name}"
                )
                return signal

        return None

    def get_continuation(self, handle: str) -> Optional[Continuation]:
        """获取续体"""
        return self._continuations.get(handle)

    def submit_changeset(
        self,
        step_id: int,
        changeset: Any,
        handle: str,
        base_revision: Optional[int] = None,
    ) -> CommitResult:
        """
        提交变更集 (20.2 节)

        集成 CommitManager 实现决定性提交

        Args:
            step_id: 步骤 ID
            changeset: 变更集
            handle: 续体句柄
            base_revision: 基准修订版本，若为 None 则使用当前版本

        Returns:
            提交结果
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
        处理待提交的变更集

        Returns:
            本次处理产生的所有结果
        """
        results = self._commit_manager.process_queue()
        logger.debug(f"Processed {len(results)} pending changesets")
        return results

    def create_view(
        self, cont: Continuation, pending_changes: Optional[List[Any]] = None
    ) -> ContinuationView:
        """
        为续体创建视图 (18.2 节)

        Args:
            cont: 续体
            pending_changes: 待提交的变更集列表

        Returns:
            续体视图
        """
        return cont.create_view(self._state_manager, pending_changes)

    def get_state_manager(self) -> StateManager:
        """获取状态管理器"""
        return self._state_manager

    def get_commit_manager(self) -> CommitManager:
        """获取提交管理器"""
        return self._commit_manager
