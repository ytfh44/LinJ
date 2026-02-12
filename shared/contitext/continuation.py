"""
Continuation - 续体

实现 LinJ 规范第 17-18 节的续体状态和行为
框架无关的续体实现，支持跨框架使用
"""

import uuid
import time
from enum import Enum
from typing import Any, Dict, Optional, List, Protocol, runtime_checkable
from copy import deepcopy

from pydantic import BaseModel, Field, ConfigDict

# 导入路径解析器
from ..core.path import PathResolver


class HandleExpired(Exception):
    """
    续体句柄过期错误 (17.2 节)

    当尝试访问已过期的续体句柄时抛出
    """

    def __init__(self, handle: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Continuation handle expired: {handle}")
        self.handle = handle
        self.details = details or {}


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
    框架无关的视图实现
    """

    def __init__(
        self,
        state_manager: StateManager,
        step_id: int,
        pending_changes: Optional[List[Any]] = None,
    ):
        """
        创建续体视图

        Args:
            state_manager: 状态管理器
            step_id: 当前步骤 ID
            pending_changes: 待提交的变更集列表
        """
        self._state_manager = state_manager
        self._step_id = step_id
        self._pending_changes = pending_changes or []

    def read(self, path: str) -> Any:
        """
        读取路径值 (5.2 节)

        按 LinJ 规范：读取不存在路径时返回空值（null）
        支持 $.a.b 路径语法和 [n] 数组下标

        Args:
            path: 状态路径，如 "$.a.b" 或 "$.arr[0]"

        Returns:
            路径对应的值，不存在则返回 None
        """
        state = self._state_manager.get_full_state()
        return PathResolver.get(state, path)

    def value(self, path: str) -> Any:
        """
        获取路径值 (14.1 节)

        与 read() 相同，用于条件表达式求值

        Args:
            path: 状态路径

        Returns:
            路径对应的值，不存在则返回 None
        """
        return self.read(path)

    def exists(self, path: str) -> bool:
        """检查路径是否存在且非 null"""
        return self.value(path) is not None

    def len(self, path: str) -> int:
        """
        获取数组长度 (14.1 节)

        返回数组的物理长度（包含尾部 null）
        非数组或不存在则返回 0

        Args:
            path: 状态路径

        Returns:
            数组长度，非数组或不存在返回 0
        """
        value = self.read(path)
        if isinstance(value, list):
            return len(value)
        return 0

    def get_full_state(self) -> Dict[str, Any]:
        """
        获取完整状态（深拷贝）

        应用所有待提交的变更集来构造当前视图状态
        确保视图反映与当前 step_id 对应的逻辑快照

        Returns:
            包含所有待提交变更的完整状态深拷贝
        """
        state = self._state_manager.get_full_state()

        # 应用待提交的变更集来构造当前视图状态
        # 18.2 节：视图必须不包含 step_id 大于等于本次尝试的变更集效果
        for changeset in self._pending_changes:
            if hasattr(changeset, "apply_to"):
                # 使用 ChangeSet 的 apply_to 方法
                changeset.apply_to(state)
            elif hasattr(changeset, "apply_to_state"):
                # 使用协议的 apply_to_state 方法
                state = changeset.apply_to_state(state)
            elif isinstance(changeset, dict):
                # 简单的字典更新（向后兼容）
                if "writes" in changeset:
                    for write in changeset["writes"]:
                        PathResolver.set(state, write["path"], write["value"])
                if "deletes" in changeset:
                    for delete in changeset["deletes"]:
                        PathResolver.delete(state, delete["path"])

        return deepcopy(state)

    def propose_changeset(self, changeset: Any) -> None:
        """提议变更集（用于挂起时）"""
        self._pending_changes.append(changeset)


class Continuation(BaseModel):
    """
    续体

    17.1 节：每个续体拥有唯一 handle，可序列化为字符串
    18.1 节：包含 status, local_state, view
    框架无关的续体实现
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
    pending_changeset: Optional[Any] = None

    # 过期时间戳（毫秒），None 表示永不过期
    expires_at_ms: Optional[int] = None

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
        pending_changes: Optional[List[Any]] = None,
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


class ContinuationRegistry:
    """
    续体注册表

    管理续体的注册和过期处理
    框架无关的注册表实现
    """

    def __init__(self, default_ttl_ms: Optional[int] = None):
        """
        初始化注册表

        Args:
            default_ttl_ms: 默认生存时间（毫秒），None 表示永不过期
        """
        self._continuations: Dict[str, Continuation] = {}
        self._expiry_times: Dict[str, float] = {}  # handle -> expiry timestamp
        self._default_ttl_ms = default_ttl_ms

    def register(
        self,
        continuation: Continuation,
        ttl_ms: Optional[int] = None,
    ) -> None:
        """
        注册续体

        Args:
            continuation: 要注册的续体
            ttl_ms: 生存时间（毫秒），使用默认值如果为 None
        """
        self._continuations[continuation.handle] = continuation

        # 设置过期时间
        if ttl_ms is not None or self._default_ttl_ms is not None:
            effective_ttl = ttl_ms if ttl_ms is not None else self._default_ttl_ms
            expiry_time = time.time() * 1000 + effective_ttl
            self._expiry_times[continuation.handle] = expiry_time

    def get(self, handle: str) -> Continuation:
        """
        获取续体

        Args:
            handle: 续体句柄

        Returns:
            续体对象

        Raises:
            HandleExpired: 句柄不存在或已过期
        """
        # 检查是否存在
        if handle not in self._continuations:
            raise HandleExpired(handle, {"reason": "not_found"})

        # 检查是否过期
        if handle in self._expiry_times:
            if time.time() * 1000 > self._expiry_times[handle]:
                # 清理过期续体
                del self._continuations[handle]
                del self._expiry_times[handle]
                raise HandleExpired(
                    handle,
                    {"reason": "expired", "expired_at": self._expiry_times[handle]},
                )

        return self._continuations[handle]

    def remove(self, handle: str) -> bool:
        """
        移除续体

        Args:
            handle: 续体句柄

        Returns:
            是否成功移除
        """
        if handle in self._continuations:
            del self._continuations[handle]
            if handle in self._expiry_times:
                del self._expiry_times[handle]
            return True
        return False

    def is_expired(self, handle: str) -> bool:
        """检查句柄是否已过期"""
        if handle not in self._continuations:
            return True
        if handle in self._expiry_times:
            return time.time() * 1000 > self._expiry_times[handle]
        return False

    def cleanup_expired(self) -> int:
        """清理所有过期的续体，返回清理数量"""
        now = time.time() * 1000
        expired_handles = [h for h, t in self._expiry_times.items() if t <= now]

        for handle in expired_handles:
            self.remove(handle)

        return len(expired_handles)

    def count(self) -> int:
        """获取注册续体数量"""
        return len(self._continuations)

    def clear(self) -> None:
        """清除所有注册"""
        self._continuations.clear()
        self._expiry_times.clear()
