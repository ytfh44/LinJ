"""
Execution Backend Abstract Interface

Defines the core interface for execution engines, supporting multi-backend extensions.
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
    Execution Backend Abstract Interface

    Defines the core functionality that all execution engines must implement, supporting multiple execution modes:
    - Synchronous/asynchronous execution
    - Single-node/batch-node execution
    - State management and tracking
    """

    @abstractmethod
    async def execute_node(
        self,
        node: Any,
        context: ExecutionContext,
        tools: Optional[Dict[str, AsyncCallable]] = None,
    ) -> ExecutionResult:
        """
        Execute a single node

        Args:
            node: The node object to execute
            context: The execution context
            tools: Available tool function collection

        Returns:
            The execution result
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
        Execute nodes in batch

        Args:
            nodes: The list of nodes to execute
            context: The execution context
            tools: Available tool function collection
            max_concurrency: Maximum concurrency limit

        Returns:
            List of execution results, in the same order as input nodes
        """
        pass

    @abstractmethod
    def validate_node(self, node: Any, context: ExecutionContext) -> bool:
        """
        Validate if a node can be executed

        Args:
            node: The node to validate
            context: The execution context

        Returns:
            True if can execute, False if cannot execute
        """
        pass

    @abstractmethod
    def get_dependencies(self, node: Any) -> List[str]:
        """
        Get the list of dependent nodes for a node

        Args:
            node: The node object

        Returns:
            List of dependent node IDs
        """
        pass

    @abstractmethod
    def get_reads(self, node: Any) -> List[str]:
        """
        Get the read path list for a node

        Args:
            node: The node object

        Returns:
            List of read paths
        """
        pass

    @abstractmethod
    def get_writes(self, node: Any) -> List[str]:
        """
        Get the write path list for a node

        Args:
            node: The node object

        Returns:
            List of write paths
        """
        pass

    def register_tool(self, name: str, tool: AsyncCallable) -> None:
        """
        Register a tool function

        Args:
            name: Tool name
            tool: Tool function
        """
        # Default implementation: subclasses can override to provide custom tool registration logic
        pass

    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool function

        Args:
            name: Tool name
        """
        # Default implementation: subclasses can override to provide custom tool unregistration logic
        pass

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics

        Returns:
            Statistics dictionary
        """
        return {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
        }

    async def health_check(self) -> bool:
        """
        Health check

        Returns:
            True if backend is healthy, False if there is an issue
        """
        return True


class BaseExecutionBackend(ExecutionBackend):
    """
    Base Execution Backend Implementation

    Provides common execution logic and state management functionality
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
        """Register tool function"""
        self._tools[name] = tool

    def unregister_tool(self, name: str) -> None:
        """Unregister tool function"""
        if name in self._tools:
            del self._tools[name]

    def get_tools(self) -> Dict[str, AsyncCallable]:
        """Get all registered tools"""
        return self._tools.copy()

    async def _execute_with_tracking(
        self,
        node: Any,
        context: ExecutionContext,
        executor: Callable[[Any, ExecutionContext], Awaitable[ExecutionResult]],
    ) -> ExecutionResult:
        """Execution wrapper with tracking"""
        start_time = time.time()
        node_id = getattr(node, "id", "unknown")

        # Create execution record
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
            # Execute node
            result = await executor(node, context)

            # Update execution record
            end_time = time.time()
            execution.status = (
                ExecutionStatus.COMPLETED if result.success else ExecutionStatus.FAILED
            )
            execution.end_time = end_time
            execution.result = result

            # Update statistics
            self._stats["total_executions"] += 1
            if result.success:
                self._stats["successful_executions"] += 1
            else:
                self._stats["failed_executions"] += 1
            self._stats["execution_times"].append(end_time - start_time)

            return result

        except Exception as e:
            # Handle exception
            end_time = time.time()
            execution.status = ExecutionStatus.FAILED
            execution.end_time = end_time
            execution.result = ExecutionResult(success=False, error=e)

            self._stats["total_executions"] += 1
            self._stats["failed_executions"] += 1
            self._stats["execution_times"].append(end_time - start_time)

            return ExecutionResult(success=False, error=e)

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
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
        """Reset statistics"""
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "execution_times": [],
        }


class DummyExecutionBackend(BaseExecutionBackend):
    """
    Dummy Execution Backend

    Used for testing and development environments, does not execute actual business logic
    """

    async def execute_node(
        self,
        node: Any,
        context: ExecutionContext,
        tools: Optional[Dict[str, AsyncCallable]] = None,
    ) -> ExecutionResult:
        """Execute a single node (dummy implementation)"""
        return await self._execute_with_tracking(node, context, self._dummy_execute)

    async def _dummy_execute(
        self, node: Any, context: ExecutionContext
    ) -> ExecutionResult:
        """Dummy execution logic"""
        # Simulate some execution time
        await asyncio.sleep(0.001)

        # Return dummy success result
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
        """Execute nodes in batch (dummy implementation)"""
        if max_concurrency is None:
            max_concurrency = len(nodes)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_semaphore(node):
            async with semaphore:
                return await self.execute_node(node, context, tools)

        tasks = [execute_with_semaphore(node) for node in nodes]
        return await asyncio.gather(*tasks)

    def validate_node(self, node: Any, context: ExecutionContext) -> bool:
        """Validate node (dummy implementation)"""
        return True

    def get_dependencies(self, node: Any) -> List[str]:
        """Get dependencies (dummy implementation)"""
        return getattr(node, "dependencies", [])

    def get_reads(self, node: Any) -> List[str]:
        """Get read paths (dummy implementation)"""
        return getattr(node, "reads", [])

    def get_writes(self, node: Any) -> List[str]:
        """Get write paths (dummy implementation)"""
        return getattr(node, "writes", [])
