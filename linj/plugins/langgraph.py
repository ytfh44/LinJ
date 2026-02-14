"""
LangGraph plugin wrapper for LinJ.

This module provides wrappers that allow LinJ to be embedded into
existing LangGraph projects as nodes.

Usage:
    from langgraph.graph import StateGraph
    from linj.plugins.langgraph import LinJNode

    # Create LangGraph workflow
    workflow = StateGraph(state_schema=MyState)

    # Add LinJ node
    workflow.add_node("process", LinJNode(
        workflow="workflows/process.yaml",
        input_mapping={"raw_data": "$.input.data"},
        output_mapping={"$.output.result": "processed_data"}
    ))

    # Compile and run
    app = workflow.compile()
    result = app.invoke({"raw_data": [...]})

Note:
    This module requires `langgraph` to be installed.
    Install with: pip install langgraph
"""

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

from .base import LinJPluginBase

if TYPE_CHECKING:
    # Type-only imports
    pass


class StateMapper:
    """
    Maps state between LangGraph and LinJ formats.

    Handles conversion of state paths and values between the two frameworks.
    """

    def __init__(
        self,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize state mapper.

        Args:
            input_mapping: Maps LangGraph paths to LinJ paths.
                Keys: LangGraph state paths (e.g., "raw_data", "config.threshold")
                Values: LinJ paths (e.g., "$.input.data", "$.input.config.threshold")
            output_mapping: Maps LinJ paths to LangGraph paths.
                Keys: LinJ paths (e.g., "$.output.result")
                Values: LangGraph state paths (e.g., "processed_data")
        """
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}

    def map_input(self, langgraph_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert LangGraph state to LinJ initial state.

        Args:
            langgraph_state: Current LangGraph state

        Returns:
            Initial state for LinJ workflow
        """
        linj_state = {}

        for lg_path, linj_path in self.input_mapping.items():
            value = self._get_nested_value(langgraph_state, lg_path)
            self._set_nested_value(linj_state, linj_path, value)

        return linj_state

    def map_output(
        self, linj_final_state: Dict[str, Any], original_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert LinJ final state back to LangGraph state.

        Args:
            linj_final_state: Final state from LinJ workflow
            original_state: Original LangGraph state (to be merged with)

        Returns:
            Updated LangGraph state
        """
        result = original_state.copy()

        for linj_path, lg_path in self.output_mapping.items():
            value = self._get_nested_value(linj_final_state, linj_path)
            self._set_nested_value(result, lg_path, value)

        return result

    def _get_nested_value(self, state: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        if not path:
            return state

        # Handle LinJ paths (start with $.)
        if path.startswith("$."):
            path = path[2:]

        parts = path.split(".")
        current = state

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _set_nested_value(self, state: Dict[str, Any], path: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        if not path:
            return

        # Handle LinJ paths (start with $.)
        if path.startswith("$."):
            path = path[2:]

        parts = path.split(".")
        current = state

        # Navigate to parent of target
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value
        if parts:
            current[parts[-1]] = value


class LinJNode:
    """
    LangGraph node wrapper for LinJ workflow.

    Allows a LinJ workflow to be used as a node in a LangGraph state graph.

    Example:
        >>> from langgraph.graph import StateGraph
        >>> workflow = StateGraph(state_schema=MyState)
        >>>
        >>> # Add LinJ node with mappings
        >>> workflow.add_node("analyze", LinJNode(
        ...     workflow="workflows/analysis.yaml",
        ...     input_mapping={
        ...         "raw_data": "$.input.data",
        ...         "config.threshold": "$.input.threshold"
        ...     },
        ...     output_mapping={
        ...         "$.output.result": "analysis_result",
        ...         "$.output.metrics": "analysis_metrics"
        ...     },
        ...     tools={"calculate": calc_func}
        ... ))
        >>>
        >>> workflow.add_edge("start", "analyze")
        >>> workflow.add_edge("analyze", "end")
        >>>
        >>> app = workflow.compile()
        >>> result = app.invoke({"raw_data": [...]})
    """

    def __init__(
        self,
        workflow: Union[str, Path, Dict[str, Any]],
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        tools: Optional[Dict[str, Callable]] = None,
        enable_tracing: bool = True,
    ):
        """
        Initialize LinJ node.

        Args:
            workflow: LinJ workflow definition (file path, YAML string, or dict)
            input_mapping: Maps LangGraph state paths to LinJ input paths
                Format: {"langgraph.path": "$.linj.path"}
            output_mapping: Maps LinJ output paths to LangGraph state paths
                Format: {"$.linj.path": "langgraph.path"}
            tools: Tools to register with LinJ workflow
            enable_tracing: Whether to enable execution tracing
        """
        self._plugin = LinJPluginBase(
            workflow=workflow, tools=tools, enable_tracing=enable_tracing
        )

        self._mapper = StateMapper(
            input_mapping=input_mapping, output_mapping=output_mapping
        )

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute LinJ workflow as a LangGraph node.

        This method is called by LangGraph during execution.

        Args:
            state: Current LangGraph state

        Returns:
            Updated LangGraph state
        """
        # Map LangGraph state to LinJ initial state
        initial_state = self._mapper.map_input(state)

        # Execute workflow
        try:
            result = asyncio.run(self._execute_workflow(initial_state))

            # Extract final state
            final_state = result.get("final_state", {})

            # Map back to LangGraph state
            updated_state = self._mapper.map_output(final_state, state)

            return updated_state

        except Exception as e:
            # On error, return original state with error info
            error_state = state.copy()
            error_state["__linj_error"] = str(e)
            return error_state

    async def _execute_workflow(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow asynchronously.

        Args:
            initial_state: Initial state for workflow

        Returns:
            Workflow execution result
        """
        # Create a message from the initial state for the plugin
        message = initial_state.get("input", {}).get("message", "")

        return await self._plugin.run_workflow(message, initial_state)

    def register_tool(self, name: str, func: Callable) -> "LinJNode":
        """
        Register a tool for the workflow.

        Args:
            name: Tool name
            func: Tool function

        Returns:
            self, supports method chaining
        """
        self._plugin.register_tool(name, func)
        return self

    def register_tools(self, tools: Dict[str, Callable]) -> "LinJNode":
        """Batch register tools."""
        self._plugin.register_tools(tools)
        return self


class LinJStateGraph:
    """
    Wrapper that exposes LinJ workflow as a LangGraph StateGraph.

    This allows the entire LinJ workflow to be used as a subgraph
    within a larger LangGraph application.

    Example:
        >>> # Create LinJ workflow graph
        >>> linj_graph = LinJStateGraph(
        ...     workflow="workflows/complex.yaml",
        ...     state_schema=MyState
        ... )
        >>>
        >>> # Use in main LangGraph
        >>> main_graph = StateGraph(state_schema=MyState)
        >>> main_graph.add_node("preprocess", preprocess_node)
        >>> main_graph.add_node("linj_workflow", linj_graph.compile())
        >>> main_graph.add_node("postprocess", postprocess_node)
    """

    def __init__(
        self,
        workflow: Union[str, Path, Dict[str, Any]],
        state_schema: Optional[type] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        tools: Optional[Dict[str, Callable]] = None,
        enable_tracing: bool = True,
    ):
        """
        Initialize LinJ state graph.

        Args:
            workflow: LinJ workflow definition
            state_schema: Pydantic model or TypedDict for state validation
            input_mapping: State input mapping
            output_mapping: State output mapping
            tools: Tools for workflow
            enable_tracing: Whether to enable tracing
        """
        self._workflow = workflow
        self._state_schema = state_schema
        self._input_mapping = input_mapping or {}
        self._output_mapping = output_mapping or {}
        self._tools = tools or {}
        self._enable_tracing = enable_tracing

        self._node = None
        self._compiled = None

    def _ensure_node_created(self):
        """Lazy creation of the LinJ node."""
        if self._node is None:
            self._node = LinJNode(
                workflow=self._workflow,
                input_mapping=self._input_mapping,
                output_mapping=self._output_mapping,
                tools=self._tools,
                enable_tracing=self._enable_tracing,
            )

    def compile(self):
        """
        Compile the LinJ workflow as a LangGraph runnable.

        Returns:
            Compiled graph that can be used as a node in another graph
        """
        try:
            from langgraph.graph import StateGraph
        except ImportError:
            raise ImportError(
                "langgraph is required for LinJStateGraph. "
                "Install with: pip install langgraph"
            )

        self._ensure_node_created()

        # Create a simple one-node graph
        if self._state_schema:
            workflow = StateGraph(self._state_schema)
        else:
            workflow = StateGraph(dict)

        workflow.add_node("linj_workflow", self._node)
        workflow.set_entry_point("linj_workflow")
        workflow.set_finish_point("linj_workflow")

        return workflow.compile()

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Directly invoke the workflow (convenience method).

        Args:
            state: Input state

        Returns:
            Output state
        """
        self._ensure_node_created()
        return self._node(state)


# Convenience function
def create_langgraph_node(
    workflow: Union[str, Path, Dict[str, Any]],
    input_mapping: Optional[Dict[str, str]] = None,
    output_mapping: Optional[Dict[str, str]] = None,
    tools: Optional[Dict[str, Callable]] = None,
    **kwargs,
) -> LinJNode:
    """
    Create a LangGraph node that wraps a LinJ workflow.

    Args:
        workflow: Path to YAML file, YAML string, or dict
        input_mapping: Input state mapping
        output_mapping: Output state mapping
        tools: Tools for workflow
        **kwargs: Additional arguments for LinJNode

    Returns:
        Configured LinJ node

    Example:
        >>> node = create_langgraph_node(
        ...     workflow="workflows/my_workflow.yaml",
        ...     input_mapping={"data": "$.input.raw"},
        ...     output_mapping={"$.output.result": "processed"}
        ... )
    """
    return LinJNode(
        workflow=workflow,
        input_mapping=input_mapping,
        output_mapping=output_mapping,
        tools=tools,
        **kwargs,
    )
