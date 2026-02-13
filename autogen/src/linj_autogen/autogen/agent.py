"""
LinJ Agent

Provides LinJ execution capability for the Autogen framework
Integrates ContiText for concurrent execution
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

    Agent that uses LinJ as the execution engine, can be used as an Autogen FunctionAgent
    Supports ContiText concurrent execution

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
        Initialize LinJ Agent

        Args:
            document: LinJ document path or document object
            name: Agent name
            description: Agent description
            enable_parallel: Whether to enable parallel execution
            enable_contitext: Whether to enable ContiText support
        """
        self.name = name
        self.description = description or f"LinJ Agent executing {document}"
        self.enable_parallel = enable_parallel
        self.enable_contitext = enable_contitext

        # Load document
        if isinstance(document, str):
            if os.path.isfile(document):
                self.document = load_document(document)
            else:
                # Try to parse as YAML string
                import yaml

                data = yaml.safe_load(document)
                self.document = LinJDocument(**data)
        else:
            self.document = document

        # Initialize executor
        if enable_contitext:
            if enable_parallel:
                self.mapper = ParallelLinJExecutor(self.document)
            else:
                self.mapper = LinJToContiTextMapper(self.document)
            self.executor = None  # Use mapper
        else:
            self.executor = LinJExecutor()
            self.mapper = None

        self.bridge = AutogenBridge()

    def register_tool(self, name: str, executor: Callable) -> "LinJAgent":
        """
        Register a tool

        Args:
            name: Tool name
            executor: Tool execution function

        Returns:
            self, supports method chaining
        """
        if self.executor:
            self.executor.register_tool(name, executor)
        if self.mapper:
            # Create temporary executor for mapper and register tool
            if not hasattr(self.mapper, "executor"):
                self.mapper.executor = LinJExecutor()
            self.mapper.executor.register_tool(name, executor)

        self.bridge.register_tool(name, executor)
        return self

    def register_tools(self, tools: Dict[str, Callable]) -> "LinJAgent":
        """
        Batch register tools

        Args:
            tools: Mapping of tool name to execution function

        Returns:
            self, supports method chaining
        """
        for name, executor in tools.items():
            self.register_tool(name, executor)
        return self

    async def run(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute LinJ workflow

        Args:
            message: User input message
            context: Additional context

        Returns:
            Final state dictionary
        """
        # Build initial state
        initial_state = {
            "input": message,
            "agent_name": self.name,
        }

        if context:
            initial_state.update(context)

        # Execute document
        if self.enable_contitext and self.mapper:
            # Use ContiText mapper
            if self.enable_parallel:
                result = await self.mapper.execute_parallel(initial_state)
            else:
                result = await self.mapper.execute(initial_state)
        else:
            # Use legacy executor
            result = await self.executor.run(self.document, initial_state)

        return result

    async def run_with_state(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute with specified initial state

        Args:
            initial_state: Complete initial state

        Returns:
            Final state dictionary
        """
        if self.enable_contitext and self.mapper:
            # Use ContiText mapper
            if self.enable_parallel:
                return await self.mapper.execute_parallel(initial_state)
            else:
                return await self.mapper.execute(initial_state)
        else:
            # Use legacy executor
            return await self.executor.run(self.document, initial_state)

    def get_tools_for_autogen(self) -> list[Dict[str, Any]]:
        """
        Get tool definitions in Autogen format

        Returns a list of tools that can be used with autogen.register_function
        """
        return self.bridge.to_autogen_tools()

    def get_system_message(self) -> str:
        """
        Get system message

        Can be used with Autogen's system_message parameter
        """
        lines = [
            f"You are {self.name}.",
            self.description,
            "\nAvailable tools:",
        ]

        for tool_name in self.bridge.list_tools():
            lines.append(f"- {tool_name}")

        return "\n".join(lines)


# Convenience function
def create_agent(
    document_path: str, tools: Optional[Dict[str, Callable]] = None
) -> LinJAgent:
    """
    Convenience function to create LinJAgent

    Args:
        document_path: LinJ document path
        tools: Optional tools dictionary

    Returns:
        Configured LinJAgent
    """
    agent = LinJAgent(document_path)
    if tools:
        agent.register_tools(tools)
    return agent
