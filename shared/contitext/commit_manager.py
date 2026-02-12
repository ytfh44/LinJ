"""
CommitManager - 变更提交管理器

实现 LinJ 规范第 20 节、第 24.3 节定义的变更集提交管理：
- 基准规则: 按 step_id 升序串行接受变更集
- 只读优化: 空变更集可立即记录完成
- 非相交优化: 较大 step_id 可提前接受，前提是写入/删除路径与所有未接受的小 step_id 变更集两两不相交
- 基准修订检查: 提交时检查 base_revision 是否匹配当前状态版本
- 冲突产生: 冲突时产生 ConflictError
框架无关的提交管理器实现
"""

import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


@runtime_checkable
class StateManager(Protocol):
    """状态管理器协议，用于框架无关的状态管理"""

    def get_full_state(self) -> Dict[str, Any]:
        """获取完整状态"""
        ...

    def get_revision(self) -> int:
        """获取当前修订版本"""
        ...

    def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
        """应用变更集"""
        ...


class LinJError(Exception):
    """LinJ 规范相关错误基类"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ConflictError(LinJError):
    """冲突错误"""

    pass


class PendingStatus(str, Enum):
    """待提交变更集状态"""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class PendingChangeSet(BaseModel):
    """
    待提交的变更集封装

    包含变更集元数据及提交状态跟踪
    框架无关的实现
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: int
    base_revision: int
    changeset: Any
    continuation_handle: str
    status: PendingStatus = PendingStatus.PENDING


class CommitResult(BaseModel):
    """
    提交结果

    记录变更集提交的完整结果
    框架无关的实现
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    step_id: int
    new_revision: Optional[int] = None
    error: Optional[LinJError] = None


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


class CommitManager:
    """
    变更提交管理器

    实现 24.3 节定义的决定性提交规则：
    - 基准规则：按 step_id 升序串行接受
    - 只读优化：空变更集可立即接受
    - 非相交优化：不相交变更集可提前接受

    线程安全：所有队列操作受锁保护
    框架无关的提交管理器实现
    """

    def __init__(self, state_manager: StateManager):
        """
        初始化提交管理器

        Args:
            state_manager: 状态管理器，用于应用变更集
        """
        self._state_manager = state_manager
        self._pending: Dict[int, PendingChangeSet] = {}  # step_id -> pending
        self._accepted_step_ids: Set[int] = set()  # 已接受的 step_id
        self._next_expected_step: int = 1  # 下一个期望的 step_id
        self._lock = threading.RLock()  # 可重入锁，保护队列操作
        self._results: Dict[int, CommitResult] = {}  # step_id -> result

    def submit(
        self, step_id: int, base_revision: int, changeset: Any, handle: str
    ) -> CommitResult:
        """
        提交变更集

        流程：
        1. 检查基准修订是否匹配
        2. 若变更集为空 → 立即接受（只读优化）
        3. 检查是否与待提交队列中的变更集相交
        4. 若 step_id == next_expected → 直接应用
        5. 若 step_id > next_expected 且不相交 → 非相交优化允许提前接受
        6. 否则 → 加入待处理队列

        Args:
            step_id: 步骤 ID
            base_revision: 基准修订版本
            changeset: 变更集
            handle: 续体句柄

        Returns:
            CommitResult: 提交结果
        """
        with self._lock:
            # 检查是否已提交
            if step_id in self._results:
                return self._results[step_id]

            # 基准修订检查 (24.3 节)
            # 提交时，base_revision 必须等于当前 revision
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

            # 创建待提交记录
            pending = PendingChangeSet(
                step_id=step_id,
                base_revision=base_revision,
                changeset=changeset,
                continuation_handle=handle,
            )

            # 只读优化：空变更集立即接受
            if self._is_changeset_empty(changeset):
                pending.status = PendingStatus.ACCEPTED
                self._pending[step_id] = pending
                self._accepted_step_ids.add(step_id)

                # 更新 next_expected
                if step_id == self._next_expected_step:
                    self._update_next_expected()

                result = CommitResult(
                    success=True,
                    step_id=step_id,
                    new_revision=current_revision,  # 空变更集不改变版本
                )
                self._results[step_id] = result

                # 尝试处理队列
                self._process_queue_internal()
                return result

            # 检查是否可以接受
            if self.can_accept(step_id, changeset):
                # 应用变更集
                try:
                    self._state_manager.apply(changeset, step_id)
                    new_revision = self._state_manager.get_revision()

                    pending.status = PendingStatus.ACCEPTED
                    self._pending[step_id] = pending
                    self._accepted_step_ids.add(step_id)

                    # 更新 next_expected
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

                # 尝试处理队列
                if result.success:
                    self._process_queue_internal()

                return result
            else:
                # 加入待处理队列
                self._pending[step_id] = pending

                result = CommitResult(
                    success=False,
                    step_id=step_id,
                    error=ConflictError(
                        f"Changeset {step_id} cannot be accepted yet, queued for processing",
                        {"step_id": step_id, "next_expected": self._next_expected_step},
                    ),
                )
                # 注意：这里不记录结果，因为还未最终处理
                return result

    def can_accept(self, step_id: int, changeset: Any) -> bool:
        """
        检查变更集是否可以被接受

        规则：
        1. 若 step_id == next_expected → 可以接受
        2. 若 step_id > next_expected 且所有更小的 step_id 都已提交（在 pending 中）
           且与它们都不相交 → 可以接受（非相交优化）

        Args:
            step_id: 步骤 ID
            changeset: 变更集

        Returns:
            bool: 是否可以接受
        """
        with self._lock:
            # 基准规则：按 step_id 升序串行接受
            if step_id == self._next_expected_step:
                return True

            # 非相交优化：检查是否可以提前接受
            if step_id > self._next_expected_step:
                # 检查所有更小的 step_id 是否都已提交（在 pending 中）
                for sid in range(self._next_expected_step, step_id):
                    if sid not in self._pending:
                        # 有更小的 step_id 尚未提交，无法判断是否相交
                        return False

                # 所有更小的 step_id 都已提交，检查是否与它们相交
                if not self._check_intersection_with_pending(step_id, changeset):
                    return True

            return False

    def _is_changeset_empty(self, changeset: Any) -> bool:
        """
        检查变更集是否为空

        框架无关的空检查实现
        """
        # 检查是否有 is_empty 方法
        if hasattr(changeset, "is_empty"):
            return changeset.is_empty()

        # 检查是否是空字典
        if isinstance(changeset, dict):
            return len(changeset) == 0

        # 检查是否是空列表
        if isinstance(changeset, list):
            return len(changeset) == 0

        # 检查是否是 None
        if changeset is None:
            return True

        # 默认情况下认为非空
        return False

    def _changesets_intersect(self, changeset1: Any, changeset2: Any) -> bool:
        """
        检查两个变更集是否相交

        框架无关的相交检查实现
        """
        # 检查是否有 intersects_with 方法
        if hasattr(changeset1, "intersects_with"):
            return changeset1.intersects_with(changeset2)

        # 简单的相交检查：如果都是字典，检查是否有重叠的键
        if isinstance(changeset1, dict) and isinstance(changeset2, dict):
            return bool(set(changeset1.keys()) & set(changeset2.keys()))

        # 默认情况下认为相交（保守策略）
        return True

    def _check_intersection_with_pending(self, step_id: int, changeset: Any) -> bool:
        """
        检查变更集与所有待接受的小 step_id 变更集是否相交

        按 11.4 节规则判定写入/删除路径相交

        Args:
            step_id: 步骤 ID
            changeset: 变更集

        Returns:
            bool: 是否与任意待处理变更集相交（True = 相交）
        """
        # 获取所有 step_id 更小且未接受的待处理变更集
        for other_step_id, pending in self._pending.items():
            if other_step_id < step_id and pending.status == PendingStatus.PENDING:
                if self._changesets_intersect(changeset, pending.changeset):
                    return True

        return False

    def process_queue(self) -> List[CommitResult]:
        """
        处理待提交队列

        按规则处理队列中的变更集，返回所有新处理的结果

        Returns:
            List[CommitResult]: 本次处理产生的所有结果
        """
        with self._lock:
            results = []

            while True:
                # 按 step_id 排序检查所有处于 PENDING 状态的项
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

                    # 检查基准修订是否仍然有效
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
                        # 这种情况通常不应发生，除非状态被回滚
                        continue

                    # 检查是否可以接受
                    if self.can_accept(step_id, pending.changeset):
                        try:
                            if self._is_changeset_empty(pending.changeset):
                                # 空变更集，只更新状态
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
                                # 应用变更集
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

                # 如果没有处理任何变更集，退出循环
                if not processed_any:
                    break

            return results

    def _process_queue_internal(self) -> None:
        """内部方法：处理队列（必须在锁内调用）"""
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

                # 检查基准修订
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

                # 检查是否可以接受
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
        检查变更集是否与自 base_revision 以来已应用的所有变更集相交

        用于确定是否可以将旧版本的 ChangeSet 应用于新版本状态
        """
        # 我们需要找到所有 new_revision > base_revision 的已接受变更集
        # 这些变更集已经存在于 self._results 中
        for step_id, result in self._results.items():
            if (
                result.success
                and result.new_revision
                and result.new_revision > base_revision
            ):
                # 获取该 step_id 对应的变更集
                if step_id in self._pending:
                    other_cs = self._pending[step_id].changeset
                    if self._changesets_intersect(changeset, other_cs):
                        return True
        return False

    def _update_next_expected(self) -> None:
        """更新 next_expected_step（必须在锁内调用）"""
        while self._next_expected_step in self._accepted_step_ids:
            self._next_expected_step += 1

    def get_pending(self) -> List[PendingChangeSet]:
        """
        获取所有待处理的变更集

        Returns:
            List[PendingChangeSet]: 待处理变更集列表（按 step_id 排序）
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
        获取指定 step_id 的提交结果

        Args:
            step_id: 步骤 ID

        Returns:
            Optional[CommitResult]: 提交结果，若未处理则返回 None
        """
        with self._lock:
            return self._results.get(step_id)

    def get_accepted_count(self) -> int:
        """
        获取已接受的变更集数量

        Returns:
            int: 已接受数量
        """
        with self._lock:
            return len(self._accepted_step_ids)

    def get_pending_count(self) -> int:
        """
        获取待处理的变更集数量

        Returns:
            int: 待处理数量
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
        检查所有已提交的变更集是否都被接受

        Returns:
            bool: 是否全部接受
        """
        with self._lock:
            total = len(self._pending)
            accepted = len(self._accepted_step_ids)
            return total > 0 and total == accepted

    def reset(self) -> None:
        """
        重置提交管理器状态

        清除所有待处理和已接受的状态
        """
        with self._lock:
            self._pending.clear()
            self._accepted_step_ids.clear()
            self._results.clear()
            self._next_expected_step = 1
