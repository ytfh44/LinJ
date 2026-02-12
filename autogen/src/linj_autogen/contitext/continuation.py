"""
续体 (Continuation)

实现 17-18 节定义的续体状态和行为
"""

import uuid
from enum import Enum
from typing import Any, Dict, Optional, List
from copy import deepcopy

from pydantic import BaseModel, Field, ConfigDict

from ..core.changeset import ChangeSet
from ..core.state import StateView, StateManager


class Status(str, Enum):
    """续体状态 (18.1 节)"""

    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContinuationView:
    """
    续体视图 (18.2 节)

    为续体提供对主状态对象的受控视图
    """

    def __init__(
        self,
        state_manager: StateManager,
        step_id: int,
        pending_changes: Optional[List[ChangeSet]] = None,
    ):
        """
        创建续体视图

        Args:
            state_manager: 状态管理器
            step_id: 当前步骤 ID
            pending_changes: 待提交的变更集列表
        """
        self._state_view = StateView(state_manager, step_id, pending_changes)
        self._step_id = step_id

    def read(self, path: str) -> Any:
        """读取路径值"""
        return self._state_view.read(path)

    def exists(self, path: str) -> bool:
        """检查路径是否存在且非 null"""
        return self._state_view.exists(path)

    def len(self, path: str) -> int:
        """获取数组长度"""
        return self._state_view.len(path)

    def get_full_state(self) -> Dict[str, Any]:
        """获取完整状态（深拷贝）"""
        return self._state_view.get_full_state()

    def propose_changeset(self, changeset: ChangeSet) -> None:
        """提议变更集（用于挂起时）"""
        self._state_view.propose(changeset)


class Continuation(BaseModel):
    """
    续体
    
    17.1 节：每个续体拥有唯一 handle，可序列化为字符串
    18.1 节：包含 status, local_state, view
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    handle: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Status = Status.RUNNING
    local_state: Dict[str, Any] = Field(default_factory=dict)
    step_id: int = 0
    base_revision: int = 0
    parent_handle: Optional[str] = None

    # 执行结果
    result: Optional[Any] = None
    error: Optional[str] = None

    # 挂起时产生的变更集
    pending_changeset: Optional[ChangeSet] = None

    # 视图（运行时设置，不序列化）
    view: Optional[ContinuationView] = Field(default=None, exclude=True)

    def is_active(self) -> bool:
        """检查续体是否仍处于活动状态"""
        return self.status in (Status.RUNNING, Status.SUSPENDED)

    def is_terminal(self) -> bool:
        """检查续体是否已到达终态"""
        return self.status in (Status.COMPLETED, Status.FAILED, Status.CANCELLED)

    def can_submit_changes(self) -> bool:
        """检查是否可以提交变更 (19.5 节：取消后不得提交)"""
        return self.status != Status.CANCELLED

    def create_view(
        self,
        state_manager: StateManager,
        pending_changes: Optional[List[ChangeSet]] = None,
    ) -> ContinuationView:
        """
        创建或更新续体视图

        Args:
            state_manager: 状态管理器
            pending_changes: 待提交的变更集列表

        Returns:
            续体视图
        """
        self.view = ContinuationView(state_manager, self.step_id, pending_changes)
        return self.view

    def get_logical_snapshot(self) -> Dict[str, Any]:
        """
        获取逻辑快照 (18.2 节)

        返回与 step_id 对应的状态快照
        """
        if self.view:
            return self.view.get_full_state()
        return {}

    def update_step_id(self, new_step_id: int) -> None:
        """
        更新 step_id 并重新创建视图

        Args:
            new_step_id: 新的步骤 ID
        """
        self.step_id = new_step_id
        # 视图会在下次访问时重新创建
