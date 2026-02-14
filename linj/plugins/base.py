"""
Base plugin functionality for LinJ framework integration.

This module provides framework-agnostic base classes that can be used
by both AutoGen and LangGraph plugins.
"""

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# LinJ imports - using relative imports from linj package
import sys
from pathlib import Path

# Add autogen src to path for imports
_autogen_src = Path(__file__).parent.parent.parent / "autogen" / "src"
if str(_autogen_src) not in sys.path:
    sys.path.insert(0, str(_autogen_src))

try:
    from linj_autogen.executor.runner import LinJExecutor, load_document
    from linj_autogen.core.document import LinJDocument
    from linj_autogen.autogen.bridge import AutogenBridge
except ImportError as e:
    raise ImportError(
        "LinJ autogen components not found. "
        "Ensure autogen/src is in PYTHONPATH. "
        f"Error: {e}"
    )


class LinJPluginBase:
    """
    Base class for LinJ plugin functionality.

    Provides common functionality that can be mixed into different agent types
    across AutoGen and LangGraph frameworks.

    Example:
        >>> plugin = LinJPluginBase("workflows/my_workflow.yaml")
        >>> plugin.register_tool("search", search_func)
        >>> result = await plugin.run_workflow("user query")
    """

    def __init__(
        self,
        workflow: Union[str, Path, Dict[str, Any], LinJDocument],
        tools: Optional[Dict[str, Callable]] = None,
        initial_state_factory: Optional[Callable[[str], Dict[str, Any]]] = None,
        enable_tracing: bool = True,
    ):
        """
        Initialize LinJ plugin base.

        Args:
            workflow: Path to YAML file, YAML string, dict, or LinJDocument
            tools: Optional dict of tool_name -> callable
            initial_state_factory: Factory function to create initial state from message.
                Signature: (message: str) -> Dict[str, Any]
            enable_tracing: Whether to enable execution tracing
        """
        self._workflow = workflow
        self._tools = tools or {}
        self._initial_state_factory = (
            initial_state_factory or self._default_state_factory
        )
        self._enable_tracing = enable_tracing

        # Initialize executor with tools
        self._executor = LinJExecutor(
            tool_registry=self._tools.copy(), enable_tracing=enable_tracing
        )

        # Load document
        self._document = self._load_document(workflow)

        # Initialize bridge for tool export
        self._bridge = AutogenBridge()
        for name, tool in self._tools.items():
            self._bridge.register_tool(name, tool)

    def _load_document(
        self, workflow: Union[str, Path, Dict[str, Any], LinJDocument]
    ) -> LinJDocument:
        """Load workflow document from various formats."""
        if isinstance(workflow, LinJDocument):
            return workflow
        elif isinstance(workflow, (str, Path)) and Path(workflow).exists():
            return load_document(str(workflow))
        elif isinstance(workflow, str):
            # Assume YAML string
            import yaml

            data = yaml.safe_load(workflow)
            return LinJDocument(**data)
        elif isinstance(workflow, dict):
            return LinJDocument(**workflow)
        else:
            raise ValueError(
                f"Invalid workflow type: {type(workflow)}. "
                "Expected: str (path/YAML), dict, or LinJDocument"
            )

    def _default_state_factory(self, message: str) -> Dict[str, Any]:
        """
        Default state factory - puts message in $.input.message.

        Args:
            message: Input message string

        Returns:
            Initial state dictionary
        """
        return {"input": {"message": message, "raw": message}}

    def register_tool(self, name: str, func: Callable) -> "LinJPluginBase":
        """
        Register a tool for LinJ workflow to use.

        Args:
            name: Tool name
            func: Tool execution function

        Returns:
            self, supports method chaining
        """
        self._tools[name] = func
        self._executor.register_tool(name, func)
        self._bridge.register_tool(name, func)
        return self

    def register_tools(self, tools: Dict[str, Callable]) -> "LinJPluginBase":
        """
        Batch register tools.

        Args:
            tools: Mapping of tool name to execution function

        Returns:
            self, supports method chaining
        """
        for name, func in tools.items():
            self.register_tool(name, func)
        return self

    async def run_workflow(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute LinJ workflow with given message.

        Args:
            message: Input message (typically from user)
            context: Optional additional context to include in initial state

        Returns:
            Final state from workflow execution containing:
                - final_state: The complete state object
                - trace: Execution trace (if tracing enabled)
                - execution_stats: Statistics about execution
        """
        # Build initial state
        initial_state = self._initial_state_factory(message)

        if context:
            if "context" not in initial_state:
                initial_state["context"] = {}
            initial_state["context"].update(context)

        # Execute workflow
        result = await self._executor.run(self._document, initial_state)

        return result

    def run_workflow_sync(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous version of run_workflow.

        Args:
            message: Input message
            context: Optional additional context

        Returns:
            Final state from workflow execution
        """
        return asyncio.run(self.run_workflow(message, context))

    def get_tools_for_autogen(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions in AutoGen format.

        Returns:
            List of tool schemas that can be registered with autogen.register_function
        """
        return self._bridge.to_autogen_tools()

    def get_document(self) -> LinJDocument:
        """Get the loaded LinJ document."""
        return self._document

    def get_executor(self) -> LinJExecutor:
        """Get the LinJ executor instance."""
        return self._executor
