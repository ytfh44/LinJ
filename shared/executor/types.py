"""
执行引擎类型定义

定义执行过程中使用的通用数据结构和类型。
"""

from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum


# 用于类型检查的异步函数类型
AsyncCallable = Callable[..., Awaitable[Any]]


class ExecutionStatus(Enum):
    """执行状态枚举"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ToolResult:
    """工具执行结果"""

    success: bool
    data: Any = None
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    """执行结果"""

    success: bool
    data: Any = None
    error: Optional[Exception] = None
    changeset: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None


@dataclass
class NodeExecution:
    """节点执行记录"""

    node_id: str
    step_id: int
    status: ExecutionStatus
    result: Optional[ExecutionResult] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    reads: Optional[List[str]] = None
    writes: Optional[List[str]] = None


@dataclass
class ExecutionContext:
    """执行上下文 (LinJ 统一版本)"""

    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_counter: int = 0
    execution_history: List[NodeExecution] = field(default_factory=list)

    # 新增字段以对齐后端需求
    document: Optional[Any] = None  # LinJDocument
    state_manager: Optional[Any] = None  # StateManager
    current_node: Optional[str] = None

    def get_state_value(self, path: str) -> Any:
        """获取状态值（委托给 state_manager 或直接从字典获取）"""
        if self.state_manager and hasattr(self.state_manager, "get"):
            return self.state_manager.get(path)
        return self.state.get(path)

    def set_state_value(self, path: str, value: Any) -> None:
        """设置状态值（委托给 state_manager 或直接设置）"""
        if self.state_manager and hasattr(self.state_manager, "apply"):
            # 注意：apply 需要 ChangeSet，这里简化处理
            self.state[path] = value
        else:
            self.state[path] = value


# 用于类型检查的函数类型
AsyncCallable = Callable[..., Any]


# 后向兼容的导入
try:
    from ..core.changeset import ChangeSet  # type: ignore
except ImportError:
    # 如果无法导入，使用 Any 类型
    ChangeSet = Any  # type: ignore
