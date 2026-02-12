"""
执行上下文和状态管理接口

定义执行过程中的上下文管理和状态操作接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
import time

from .types import NodeExecution, ExecutionStatus, ExecutionContext
from ..core.path import PathResolver as CorePathResolver

# 为了兼容性保留 PathResolver 引用
PathResolver = CorePathResolver


class StateScope(Enum):
    """状态作用域枚举"""

    GLOBAL = "global"  # 全局作用域
    SESSION = "session"  # 会话作用域
    STEP = "step"  # 步骤作用域
    NODE = "node"  # 节点作用域
    TEMPORARY = "temporary"  # 临时作用域


@dataclass
class StateEntry:
    """状态条目"""

    value: Any
    scope: StateScope
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    read_count: int = 0
    write_count: int = 0

    def mark_read(self) -> None:
        """标记读取"""
        self.read_count += 1

    def mark_write(self) -> None:
        """标记写入"""
        self.write_count += 1
        self.timestamp = time.time()


class StateManager(ABC):
    """
    状态管理器抽象接口

    定义状态操作的统一接口：
    - 状态读写和查询
    - 作用域管理
    - 变更追踪
    - 持久化和恢复
    """

    @abstractmethod
    def get(self, path: str, scope: Optional[StateScope] = None) -> Any:
        """
        获取状态值

        Args:
            path: 状态路径
            scope: 状态作用域

        Returns:
            状态值
        """
        pass

    @abstractmethod
    def set(self, path: str, value: Any, scope: Optional[StateScope] = None) -> None:
        """
        设置状态值

        Args:
            path: 状态路径
            value: 要设置的值
            scope: 状态作用域
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        检查路径是否存在

        Args:
            path: 状态路径

        Returns:
            True 表示存在，False 表示不存在
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """
        删除状态项

        Args:
            path: 状态路径

        Returns:
            True 表示删除成功，False 表示路径不存在
        """
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        列出所有键

        Args:
            prefix: 键前缀过滤

        Returns:
            键列表
        """
        pass

    @abstractmethod
    def clear(self, scope: Optional[StateScope] = None) -> None:
        """
        清空状态

        Args:
            scope: 要清空的作用域，None表示清空所有
        """
        pass

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        创建状态快照

        Returns:
            状态快照字典
        """
        pass

    @abstractmethod
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        恢复状态快照

        Args:
            snapshot: 状态快照
        """
        pass


class BaseStateManager(StateManager):
    """
    基础状态管理器实现

    提供内存中的状态管理功能
    """

    def __init__(self):
        self._state: Dict[str, StateEntry] = {}
        self._watchers: Dict[str, List[Callable]] = {}
        self._change_log: List[Dict[str, Any]] = []

    def get(self, path: str, scope: Optional[StateScope] = None) -> Any:
        """获取状态值"""
        if path not in self._state:
            return None

        entry = self._state[path]
        entry.mark_read()

        # 检查作用域过滤
        if scope and entry.scope != scope:
            return None

        # 触发观察者
        self._notify_watchers(path, "read", entry.value)

        return entry.value

    def set(self, path: str, value: Any, scope: Optional[StateScope] = None) -> None:
        """设置状态值"""
        scope = scope or StateScope.GLOBAL

        old_value = None
        if path in self._state:
            old_entry = self._state[path]
            old_value = old_entry.value
            # 更新现有条目
            old_entry.value = value
            old_entry.mark_write()
        else:
            # 创建新条目
            self._state[path] = StateEntry(value=value, scope=scope)

        # 记录变更
        change = {
            "path": path,
            "old_value": old_value,
            "new_value": value,
            "scope": scope.value,
            "timestamp": time.time(),
        }
        self._change_log.append(change)

        # 触发观察者
        self._notify_watchers(path, "write", value)

    def exists(self, path: str) -> bool:
        """检查路径是否存在"""
        return path in self._state

    def delete(self, path: str) -> bool:
        """删除状态项"""
        if path not in self._state:
            return False

        old_value = self._state[path].value
        del self._state[path]

        # 记录变更
        change = {
            "path": path,
            "old_value": old_value,
            "new_value": None,
            "deleted": True,
            "timestamp": time.time(),
        }
        self._change_log.append(change)

        # 触发观察者
        self._notify_watchers(path, "delete", None)

        return True

    def list_keys(self, prefix: str = "") -> List[str]:
        """列出所有键"""
        if prefix:
            return [key for key in self._state.keys() if key.startswith(prefix)]
        return list(self._state.keys())

    def clear(self, scope: Optional[StateScope] = None) -> None:
        """清空状态"""
        if scope is None:
            self._state.clear()
        else:
            keys_to_delete = [
                key for key, entry in self._state.items() if entry.scope == scope
            ]
            for key in keys_to_delete:
                del self._state[key]

        self._change_log.append(
            {
                "action": "clear",
                "scope": scope.value if scope else "all",
                "timestamp": time.time(),
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        """创建状态快照"""
        return {key: entry.value for key, entry in self._state.items()}

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """恢复状态快照"""
        # 清空当前状态
        self._state.clear()

        # 恢复快照
        for path, value in snapshot.items():
            self._state[path] = StateEntry(value=value, scope=StateScope.GLOBAL)

        # 记录恢复操作
        self._change_log.append(
            {
                "action": "restore",
                "timestamp": time.time(),
            }
        )

    def watch(self, path: str, callback: Callable) -> None:
        """添加状态观察者"""
        if path not in self._watchers:
            self._watchers[path] = []
        self._watchers[path].append(callback)

    def unwatch(self, path: str, callback: Callable) -> None:
        """移除状态观察者"""
        if path in self._watchers:
            try:
                self._watchers[path].remove(callback)
                if not self._watchers[path]:
                    del self._watchers[path]
            except ValueError:
                pass

    def _notify_watchers(self, path: str, action: str, value: Any) -> None:
        """通知观察者"""
        if path in self._watchers:
            for callback in self._watchers[path]:
                try:
                    callback(path, action, value)
                except Exception:
                    # 观察者错误不应该影响主流程
                    pass

    def get_change_log(self) -> List[Dict[str, Any]]:
        """获取变更日志"""
        return self._change_log.copy()

    def clear_change_log(self) -> None:
        """清空变更日志"""
        self._change_log.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取状态统计信息"""
        scope_counts = {}
        for entry in self._state.values():
            scope_name = entry.scope.value
            scope_counts[scope_name] = scope_counts.get(scope_name, 0) + 1

        total_reads = sum(entry.read_count for entry in self._state.values())
        total_writes = sum(entry.write_count for entry in self._state.values())

        return {
            "total_entries": len(self._state),
            "scope_distribution": scope_counts,
            "total_reads": total_reads,
            "total_writes": total_writes,
            "change_log_size": len(self._change_log),
            "active_watchers": len(self._watchers),
        }


class ContextManager:
    """
    上下文管理器

    管理执行上下文的生命周期和状态流转
    """

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self._contexts: Dict[str, ExecutionContext] = {}
        self._active_context: Optional[ExecutionContext] = None

    def create_context(
        self,
        context_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionContext:
        """
        创建执行上下文

        Args:
            context_id: 上下文ID
            initial_state: 初始状态
            metadata: 元数据

        Returns:
            执行上下文
        """
        context = ExecutionContext(
            state=initial_state or {},
            metadata=metadata or {},
            step_counter=0,
            execution_history=[],
        )

        self._contexts[context_id] = context
        return context

    def get_context(self, context_id: str) -> Optional[ExecutionContext]:
        """获取执行上下文"""
        return self._contexts.get(context_id)

    def set_active_context(self, context_id: str) -> bool:
        """设置活动上下文"""
        if context_id in self._contexts:
            self._active_context = self._contexts[context_id]
            return True
        return False

    def get_active_context(self) -> Optional[ExecutionContext]:
        """获取活动上下文"""
        return self._active_context

    def update_context(
        self,
        context_id: str,
        state_updates: Dict[str, Any],
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新上下文"""
        if context_id not in self._contexts:
            return False

        context = self._contexts[context_id]
        context.state.update(state_updates)

        if metadata_updates:
            context.metadata.update(metadata_updates)

        return True

    def delete_context(self, context_id: str) -> bool:
        """删除上下文"""
        if context_id not in self._contexts:
            return False

        if self._active_context == self._contexts[context_id]:
            self._active_context = None

        del self._contexts[context_id]
        return True

    def list_contexts(self) -> List[str]:
        """列出所有上下文ID"""
        return list(self._contexts.keys())

    def clear_all_contexts(self) -> None:
        """清空所有上下文"""
        self._contexts.clear()
        self._active_context = None

    def get_context_stats(self) -> Dict[str, Any]:
        """获取上下文统计信息"""
        return {
            "total_contexts": len(self._contexts),
            "active_context_id": (
                next(
                    (k for k, v in self._contexts.items() if v == self._active_context),
                    None,
                )
            ),
            "total_steps": sum(ctx.step_counter for ctx in self._contexts.values()),
            "total_executions": sum(
                len(ctx.execution_history) for ctx in self._contexts.values()
            ),
        }


# PathResolver is now imported from ..core.path
