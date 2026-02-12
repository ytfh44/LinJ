"""
主状态管理

实现规范 3.1 节定义的主状态管理
"""

from typing import Any, Dict, List, Optional
import copy

from .changeset import ChangeSet
from .path import PathResolver


class StateManager:
    """
    主状态管理器

    负责管理主状态对象，应用变更集，创建视图
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._state = {}
        if initial_state:
            for path, value in initial_state.items():
                if path.startswith("$."):
                    PathResolver.set(self._state, path, value)
                else:
                    # 兼容不带 $. 的简单键
                    PathResolver.set(self._state, f"$.{path}", value)

        self._revision = 0
        self._history: List[Dict[str, Any]] = []  # 变更历史（用于调试）

    def get(self, path: str) -> Any:
        """读取路径值"""
        return PathResolver.get(self._state, path)

    def get_full_state(self) -> Dict[str, Any]:
        """获取完整状态（深拷贝）"""
        return copy.deepcopy(self._state)

    def apply(self, changeset: ChangeSet, step_id: Optional[int] = None) -> None:
        """
        应用变更集到状态

        9.2 节：变更集应用必须是原子性的
        """
        if changeset.is_empty():
            return

        # 先应用到临时状态验证
        temp_state = copy.deepcopy(self._state)
        changeset.apply_to(temp_state)

        # 验证通过后应用到实际状态
        changeset.apply_to(self._state)

        # 更新版本
        self._revision += 1

        # 记录历史
        self._history.append(
            {
                "step_id": step_id,
                "revision": self._revision,
                "changeset": changeset.model_dump(),
            }
        )

    def get_revision(self) -> int:
        """获取当前版本号"""
        return self._revision

    def snapshot(self) -> Dict[str, Any]:
        """创建状态快照"""
        return copy.deepcopy(self._state)

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """从快照恢复状态"""
        self._state = copy.deepcopy(snapshot)
        self._revision = 0
        self._history.clear()


class StateView:
    """
    状态视图（逻辑快照）

    18.2 节：为每次节点尝试提供与其 step_id 对应的逻辑快照
    """

    def __init__(
        self,
        state_manager: StateManager,
        step_id: int,
        pending_changes: Optional[List[ChangeSet]] = None,
    ):
        """
        创建状态视图

        Args:
            state_manager: 状态管理器
            step_id: 当前步骤 ID
            pending_changes: 待提交的变更集列表
        """
        self._state_manager = state_manager
        self._step_id = step_id
        self._pending = pending_changes or []

        # 创建逻辑快照
        self._snapshot = state_manager.snapshot()

        # 应用所有小于当前 step_id 的待提交变更
        for cs in self._pending:
            cs.apply_to(self._snapshot)

    def read(self, path: str) -> Any:
        """读取路径值"""
        return PathResolver.get(self._snapshot, path)

    def exists(self, path: str) -> bool:
        """检查路径是否存在且非 null"""
        return self.read(path) is not None

    def len(self, path: str) -> int:
        """获取数组长度"""
        value = self.read(path)
        if isinstance(value, list):
            return len(value)
        return 0

    def get_full_state(self) -> Dict[str, Any]:
        """获取完整状态（深拷贝）"""
        return copy.deepcopy(self._snapshot)

    def propose(self, changeset: ChangeSet) -> None:
        """
        提议变更集

        变更集不会直接应用，而是提交到待处理队列
        """
        self._pending.append(changeset)
