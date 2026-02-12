"""
执行后端抽象接口

定义执行引擎的核心接口，支持多后端扩展。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
import asyncio
import time

from .types import (
    ExecutionResult,
    ExecutionContext,
    ToolResult,
    ExecutionStatus,
    NodeExecution,
    AsyncCallable,
)


class ExecutionBackend(ABC):
    """
    执行后端抽象接口

    定义所有执行引擎必须实现的核心功能，支持多种执行模式：
    - 同步/异步执行
    - 单节点/批量节点执行
    - 状态管理和追踪
    """

    @abstractmethod
    async def execute_node(
        self,
        node: Any,
        context: ExecutionContext,
        tools: Optional[Dict[str, AsyncCallable]] = None,
    ) -> ExecutionResult:
        """
        执行单个节点

        Args:
            node: 要执行的节点对象
            context: 执行上下文
            tools: 可用的工具函数集合

        Returns:
            执行结果
        """
        pass

    @abstractmethod
    async def execute_batch(
        self,
        nodes: List[Any],
        context: ExecutionContext,
        tools: Optional[Dict[str, AsyncCallable]] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[ExecutionResult]:
        """
        批量执行节点

        Args:
            nodes: 要执行的节点列表
            context: 执行上下文
            tools: 可用的工具函数集合
            max_concurrency: 最大并发数限制

        Returns:
            执行结果列表，顺序与输入节点相同
        """
        pass

    @abstractmethod
    def validate_node(self, node: Any, context: ExecutionContext) -> bool:
        """
        验证节点是否可以执行

        Args:
            node: 要验证的节点
            context: 执行上下文

        Returns:
            True 表示可以执行，False 表示不能执行
        """
        pass

    @abstractmethod
    def get_dependencies(self, node: Any) -> List[str]:
        """
        获取节点的依赖节点列表

        Args:
            node: 节点对象

        Returns:
            依赖节点ID列表
        """
        pass

    @abstractmethod
    def get_reads(self, node: Any) -> List[str]:
        """
        获取节点的读取路径列表

        Args:
            node: 节点对象

        Returns:
            读取路径列表
        """
        pass

    @abstractmethod
    def get_writes(self, node: Any) -> List[str]:
        """
        获取节点的写入路径列表

        Args:
            node: 节点对象

        Returns:
            写入路径列表
        """
        pass

    def register_tool(self, name: str, tool: AsyncCallable) -> None:
        """
        注册工具函数

        Args:
            name: 工具名称
            tool: 工具函数
        """
        # 默认实现：子类可以重写以提供自定义工具注册逻辑
        pass

    def unregister_tool(self, name: str) -> None:
        """
        注销工具函数

        Args:
            name: 工具名称
        """
        # 默认实现：子类可以重写以提供自定义工具注销逻辑
        pass

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        获取执行统计信息

        Returns:
            统计信息字典
        """
        return {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
        }

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            True 表示后端正常，False 表示异常
        """
        return True


class BaseExecutionBackend(ExecutionBackend):
    """
    基础执行后端实现

    提供通用的执行逻辑和状态管理功能
    """

    def __init__(self):
        self._tools: Dict[str, AsyncCallable] = {}
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "execution_times": [],
        }
        self._running_tasks: Dict[str, asyncio.Task] = {}

    def register_tool(self, name: str, tool: AsyncCallable) -> None:
        """注册工具函数"""
        self._tools[name] = tool

    def unregister_tool(self, name: str) -> None:
        """注销工具函数"""
        if name in self._tools:
            del self._tools[name]

    def get_tools(self) -> Dict[str, AsyncCallable]:
        """获取所有注册的工具"""
        return self._tools.copy()

    async def _execute_with_tracking(
        self,
        node: Any,
        context: ExecutionContext,
        executor: Callable[[Any, ExecutionContext], Awaitable[ExecutionResult]],
    ) -> ExecutionResult:
        """带追踪的执行包装器"""
        start_time = time.time()
        node_id = getattr(node, "id", "unknown")

        # 创建执行记录
        execution = NodeExecution(
            node_id=node_id,
            step_id=context.step_counter,
            status=ExecutionStatus.RUNNING,
            start_time=start_time,
            reads=self.get_reads(node),
            writes=self.get_writes(node),
        )
        context.execution_history.append(execution)

        try:
            # 执行节点
            result = await executor(node, context)

            # 更新执行记录
            end_time = time.time()
            execution.status = (
                ExecutionStatus.COMPLETED if result.success else ExecutionStatus.FAILED
            )
            execution.end_time = end_time
            execution.result = result

            # 更新统计
            self._stats["total_executions"] += 1
            if result.success:
                self._stats["successful_executions"] += 1
            else:
                self._stats["failed_executions"] += 1
            self._stats["execution_times"].append(end_time - start_time)

            return result

        except Exception as e:
            # 处理异常
            end_time = time.time()
            execution.status = ExecutionStatus.FAILED
            execution.end_time = end_time
            execution.result = ExecutionResult(success=False, error=e)

            self._stats["total_executions"] += 1
            self._stats["failed_executions"] += 1
            self._stats["execution_times"].append(end_time - start_time)

            return ExecutionResult(success=False, error=e)

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        times = self._stats["execution_times"]
        avg_time = sum(times) / len(times) if times else 0.0

        return {
            "total_executions": self._stats["total_executions"],
            "successful_executions": self._stats["successful_executions"],
            "failed_executions": self._stats["failed_executions"],
            "average_execution_time": avg_time,
            "total_execution_time": sum(times),
            "registered_tools": len(self._tools),
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "execution_times": [],
        }


class DummyExecutionBackend(BaseExecutionBackend):
    """
    虚拟执行后端

    用于测试和开发环境，不执行实际的业务逻辑
    """

    async def execute_node(
        self,
        node: Any,
        context: ExecutionContext,
        tools: Optional[Dict[str, AsyncCallable]] = None,
    ) -> ExecutionResult:
        """执行单个节点（虚拟实现）"""
        return await self._execute_with_tracking(node, context, self._dummy_execute)

    async def _dummy_execute(
        self, node: Any, context: ExecutionContext
    ) -> ExecutionResult:
        """虚拟执行逻辑"""
        # 模拟一些执行时间
        await asyncio.sleep(0.001)

        # 返回虚拟成功结果
        return ExecutionResult(
            success=True,
            data={"dummy": True, "node_id": getattr(node, "id", "unknown")},
            metadata={"backend": "dummy"},
        )

    async def execute_batch(
        self,
        nodes: List[Any],
        context: ExecutionContext,
        tools: Optional[Dict[str, AsyncCallable]] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[ExecutionResult]:
        """批量执行节点（虚拟实现）"""
        if max_concurrency is None:
            max_concurrency = len(nodes)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_semaphore(node):
            async with semaphore:
                return await self.execute_node(node, context, tools)

        tasks = [execute_with_semaphore(node) for node in nodes]
        return await asyncio.gather(*tasks)

    def validate_node(self, node: Any, context: ExecutionContext) -> bool:
        """验证节点（虚拟实现）"""
        return True

    def get_dependencies(self, node: Any) -> List[str]:
        """获取依赖（虚拟实现）"""
        return getattr(node, "dependencies", [])

    def get_reads(self, node: Any) -> List[str]:
        """获取读取路径（虚拟实现）"""
        return getattr(node, "reads", [])

    def get_writes(self, node: Any) -> List[str]:
        """获取写入路径（虚拟实现）"""
        return getattr(node, "writes", [])
