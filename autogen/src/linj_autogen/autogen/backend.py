"""
AutoGen 执行后端实现

实现 ExecutionBackend 接口，为 AutoGen 框架提供 LinJ 执行能力
"""

import asyncio
from typing import Any, Dict, List, Optional, Awaitable

# Temporary local imports until shared components are fully integrated
from ..core.errors import ExecutionError


# Temporary type definitions until we import from shared
class ExecutionResult:
    def __init__(
        self,
        success: bool,
        data: Any = None,
        error: Optional[Exception] = None,
        changeset: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.changeset = changeset
        self.metadata = metadata


class ExecutionContext:
    def __init__(
        self,
        state: Dict[str, Any],
        metadata: Dict[str, Any],
        step_counter: int,
        execution_history: List[Any],
    ):
        self.state = state
        self.metadata = metadata
        self.step_counter = step_counter
        self.execution_history = execution_history


AsyncCallable = Awaitable[Any]


class AutoGenExecutionBackend:
    """
    AutoGen 执行后端

    基于 LinJ 执行器的 AutoGen 后端实现，支持：
    - 单节点和批量执行
    - ContiText 并发执行
    - 状态管理和追踪
    - 工具注册和执行
    """

    def __init__(self, enable_contitext: bool = True, enable_parallel: bool = False):
        self.enable_contitext = enable_contitext
        self.enable_parallel = enable_parallel
        self._document = None
        self._tools = {}
        self._mapper = None
        self._executor = None

    def set_document(self, document) -> None:
        """设置要执行的 LinJ 文档"""
        self._document = document

        # 如果启用 ContiText，初始化 mapper
        if self.enable_contitext:
            from ..contitext.mapper import LinJToContiTextMapper, ParallelLinJExecutor

            if self.enable_parallel:
                self._mapper = ParallelLinJExecutor(document)
            else:
                self._mapper = LinJToContiTextMapper(document)
        else:
            from ..executor.runner import LinJExecutor

            self._executor = LinJExecutor(tool_registry=self._tools)

    def register_tool(self, name: str, tool: AsyncCallable) -> None:
        """注册工具函数"""
        self._tools[name] = tool

    async def execute_node(
        self,
        node: Any,
        context: ExecutionContext,
        tools: Optional[Dict[str, AsyncCallable]] = None,
    ) -> ExecutionResult:
        """执行单个节点"""
        if not self._document:
            raise ExecutionError("Document not set. Call set_document() first.")

        # 合并工具
        all_tools = {**self._tools, **(tools or {})}

        # 使用 ContiText 执行
        if self.enable_contitext and self._mapper:
            return await self._execute_with_contitext(node, context, all_tools)

        # 使用传统执行器
        return await self._execute_traditional(node, context, all_tools)

    async def execute_batch(
        self,
        nodes: List[Any],
        context: ExecutionContext,
        tools: Optional[Dict[str, AsyncCallable]] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[ExecutionResult]:
        """批量执行节点"""
        if max_concurrency is None:
            max_concurrency = len(nodes)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_semaphore(node):
            async with semaphore:
                return await self.execute_node(node, context, tools)

        tasks = [execute_with_semaphore(node) for node in nodes]
        return await asyncio.gather(*tasks)

    def validate_node(self, node: Any, context: ExecutionContext) -> bool:
        """验证节点是否可以执行"""
        if not self._document:
            return False

        # 检查节点是否在文档中
        try:
            self._document.get_node(node.id)
            return True
        except:
            return False

    def get_dependencies(self, node: Any) -> List[str]:
        """获取节点的依赖节点列表"""
        if not self._document:
            return []

        graph = self._document.build_dependency_graph()
        deps = graph.get_data_dependencies(node.id) + graph.get_control_dependencies(
            node.id
        )
        return deps

    def get_reads(self, node: Any) -> List[str]:
        """获取节点的读取路径列表"""
        return getattr(node, "reads", [])

    def get_writes(self, node: Any) -> List[str]:
        """获取节点的写入路径列表"""
        writes = getattr(node, "writes", [])
        if hasattr(node, "write_to") and node.write_to:
            writes.append(node.write_to)
        return writes

    async def _execute_with_contitext(
        self,
        node: Any,
        context: ExecutionContext,
        tools: Dict[str, AsyncCallable],
    ) -> ExecutionResult:
        """使用 ContiText 执行节点"""
        try:
            # 准备执行状态
            state = context.state.copy()

            # 注册工具到 mapper
            if hasattr(self._mapper, "executor"):
                for name, tool in tools.items():
                    self._mapper.executor.register_tool(name, tool)

            # 如果是并行执行，使用专门的并行执行器
            if self.enable_parallel:
                result = await self._mapper.execute_parallel(state)
            else:
                result = await self._mapper.execute(state)

            return ExecutionResult(
                success=True,
                data=result,
                metadata={"backend": "autogen_contitext", "node_id": node.id},
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=e,
                metadata={"backend": "autogen_contitext", "node_id": node.id},
            )

    async def _execute_traditional(
        self,
        node: Any,
        context: ExecutionContext,
        tools: Dict[str, AsyncCallable],
    ) -> ExecutionResult:
        """使用传统执行器执行节点"""
        try:
            # 更新执行器的工具
            if self._executor:
                for name, tool in tools.items():
                    self._executor.register_tool(name, tool)

                # 更新状态管理器的状态
                from ..core.state import StateManager

                state_manager = StateManager(context.state)

                # 执行节点
                node_result = await self._executor._execute_node(
                    node, state_manager, self._document
                )

                return ExecutionResult(
                    success=node_result.success,
                    data=node_result.next_nodes if node_result.success else None,
                    changeset=node_result.changeset,
                    error=node_result.error,
                    metadata={"backend": "autogen_traditional", "node_id": node.id},
                )

            # 如果没有执行器，返回错误
            return ExecutionResult(
                success=False,
                error=ExecutionError("Traditional executor not initialized"),
                metadata={"backend": "autogen_traditional", "node_id": node.id},
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=e,
                metadata={"backend": "autogen_traditional", "node_id": node.id},
            )

    async def execute_document(
        self,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行整个文档

        这是 AutoGen 特有的便捷方法，用于完整执行 LinJ 文档
        """
        if not self._document:
            raise ExecutionError("Document not set. Call set_document() first.")

        # 如果启用 ContiText，使用 mapper 执行
        if self.enable_contitext and self._mapper:
            # 注册工具到 mapper
            if hasattr(self._mapper, "executor"):
                for name, tool in self._tools.items():
                    self._mapper.executor.register_tool(name, tool)

            # 执行文档
            if self.enable_parallel:
                return await self._mapper.execute_parallel(initial_state or {})
            else:
                return await self._mapper.execute(initial_state or {})

        # 否则使用传统执行器
        if self._executor:
            return await self._executor.run(self._document, initial_state)

        raise ExecutionError("No executor available")
