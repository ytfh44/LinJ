"""
LinJ Agent

为 Autogen 框架提供 LinJ 执行能力
集成 ContiText 支持并发执行
"""

from typing import Any, Callable, Dict, Optional, Union
import os
import asyncio

# Use local imports for now - will migrate to shared components once path issues resolved
from ..core.document import LinJDocument
from ..contitext.mapper import LinJToContiTextMapper, ParallelLinJExecutor
from .backend import AutoGenExecutionBackend
from .bridge import AutogenBridge
from ..executor.runner import LinJExecutor, load_document


class LinJAgent:
    """
    LinJ Agent

    使用 LinJ 作为执行引擎的 Agent，可作为 Autogen 的 FunctionAgent 使用
    支持 ContiText 并发执行

    Example:
        ```python
        agent = LinJAgent("workflow.yaml", enable_parallel=True)
        agent.register_tool("search", search_func)
        result = await agent.run("user input")
        ```
    """

    def __init__(
        self,
        document: Union[str, LinJDocument],
        name: str = "linj_agent",
        description: Optional[str] = None,
        enable_parallel: bool = False,
        enable_contitext: bool = True,
    ):
        """
        初始化 LinJ Agent

        Args:
            document: LinJ 文档路径或文档对象
            name: Agent 名称
            description: Agent 描述
            enable_parallel: 是否启用并行执行
            enable_contitext: 是否启用 ContiText 支持
        """
        self.name = name
        self.description = description or f"LinJ Agent executing {document}"
        self.enable_parallel = enable_parallel
        self.enable_contitext = enable_contitext

        # 加载文档
        if isinstance(document, str):
            if os.path.isfile(document):
                self.document = load_document(document)
            else:
                # 尝试作为 YAML 字符串解析
                import yaml

                data = yaml.safe_load(document)
                self.document = LinJDocument(**data)
        else:
            self.document = document

        # 初始化执行器
        if enable_contitext:
            if enable_parallel:
                self.mapper = ParallelLinJExecutor(self.document)
            else:
                self.mapper = LinJToContiTextMapper(self.document)
            self.executor = None  # 使用 mapper
        else:
            self.executor = LinJExecutor()
            self.mapper = None

        self.bridge = AutogenBridge()

    def register_tool(self, name: str, executor: Callable) -> "LinJAgent":
        """
        注册工具

        Args:
            name: 工具名称
            executor: 工具执行函数

        Returns:
            self，支持链式调用
        """
        if self.executor:
            self.executor.register_tool(name, executor)
        if self.mapper:
            # 为 mapper 创建临时执行器并注册工具
            if not hasattr(self.mapper, "executor"):
                self.mapper.executor = LinJExecutor()
            self.mapper.executor.register_tool(name, executor)

        self.bridge.register_tool(name, executor)
        return self

    def register_tools(self, tools: Dict[str, Callable]) -> "LinJAgent":
        """
        批量注册工具

        Args:
            tools: 工具名称到执行函数的映射

        Returns:
            self，支持链式调用
        """
        for name, executor in tools.items():
            self.register_tool(name, executor)
        return self

    async def run(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行 LinJ 工作流

        Args:
            message: 用户输入消息
            context: 额外上下文

        Returns:
            最终状态字典
        """
        # 构建初始状态
        initial_state = {
            "input": message,
            "agent_name": self.name,
        }

        if context:
            initial_state.update(context)

        # 执行文档
        if self.enable_contitext and self.mapper:
            # 使用 ContiText mapper
            if self.enable_parallel:
                result = await self.mapper.execute_parallel(initial_state)
            else:
                result = await self.mapper.execute(initial_state)
        else:
            # 使用传统执行器
            result = await self.executor.run(self.document, initial_state)

        return result

    async def run_with_state(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用指定初始状态执行

        Args:
            initial_state: 完整初始状态

        Returns:
            最终状态字典
        """
        if self.enable_contitext and self.mapper:
            # 使用 ContiText mapper
            if self.enable_parallel:
                return await self.mapper.execute_parallel(initial_state)
            else:
                return await self.mapper.execute(initial_state)
        else:
            # 使用传统执行器
            return await self.executor.run(self.document, initial_state)

    def get_tools_for_autogen(self) -> list[Dict[str, Any]]:
        """
        获取 Autogen 格式的工具定义

        返回可用于 autogen.register_function 的工具列表
        """
        return self.bridge.to_autogen_tools()

    def get_system_message(self) -> str:
        """
        获取系统消息

        可用于 Autogen 的 system_message 参数
        """
        lines = [
            f"You are {self.name}.",
            self.description,
            "\nAvailable tools:",
        ]

        for tool_name in self.bridge.list_tools():
            lines.append(f"- {tool_name}")

        return "\n".join(lines)


# 便捷函数
def create_agent(
    document_path: str, tools: Optional[Dict[str, Callable]] = None
) -> LinJAgent:
    """
    便捷创建 LinJAgent

    Args:
        document_path: LinJ 文档路径
        tools: 可选的工具字典

    Returns:
        配置的 LinJAgent
    """
    agent = LinJAgent(document_path)
    if tools:
        agent.register_tools(tools)
    return agent
