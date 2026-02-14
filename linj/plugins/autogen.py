"""
AutoGen plugin wrapper for LinJ.

This module provides wrappers that allow LinJ to be embedded into
existing AutoGen projects as FunctionAgent or UserProxyAgent components.

Usage:

    # As FunctionAgent (workflow becomes a callable function)
    agent = LinJFunctionAgent(
        name="research_agent",
        workflow="workflows/research.yaml",
        llm_config={"config_list": [...]},
        tools={"search": search_func}
    )

    # Add to existing GroupChat
    groupchat = autogen.GroupChat(agents=[user_proxy, agent, assistant], ...)


    # As UserProxyAgent (intercepts and processes messages)
    proxy = LinJUserProxyAgent(
        name="workflow_proxy",
        workflow="workflows/orchestrate.yaml",
        tools={"execute": execute_func}
    )

Note:
    This module requires `pyautogen` to be installed.
    Install with: pip install pyautogen
"""

from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path

from .base import LinJPluginBase

if TYPE_CHECKING:
    # Type-only imports to avoid circular dependencies
    pass


class LinJFunctionAgent:
    """
    FunctionAgent wrapper for LinJ.

    This agent wraps a LinJ workflow as an AutoGen ConversableAgent.
    The workflow becomes a "function" that can be called by other agents.

    The agent will:
    1. Receive messages from the GroupChat
    2. Pass them to the LinJ workflow
    3. Return workflow results as responses

    Example:
        >>> agent = LinJFunctionAgent(
        ...     name="data_processor",
        ...     workflow="workflows/process.yaml",
        ...     llm_config={"config_list": [{"model": "gpt-4", "api_key": "..."}]},
        ...     description="Process data using LinJ workflow"
        ... )
        >>> # Use in GroupChat
        >>> groupchat = autogen.GroupChat(agents=[user_proxy, agent, assistant])
    """

    def __init__(
        self,
        name: str,
        workflow: Union[str, Path, Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[Dict[str, Callable]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: str = "NEVER",
        **kwargs,
    ):
        """
        Initialize LinJ FunctionAgent.

        Args:
            name: Agent name (visible to other agents)
            workflow: LinJ workflow definition (file path, YAML string, or dict)
            llm_config: LLM configuration for AutoGen (required)
            description: Agent description shown to other agents
            system_message: System message for the LLM
            tools: Tools to register with LinJ workflow
            is_termination_msg: Function to check if message signals termination
            max_consecutive_auto_reply: Max auto replies before requiring human
            human_input_mode: "ALWAYS", "TERMINATE", or "NEVER"
            **kwargs: Additional arguments passed to ConversableAgent

        Raises:
            ImportError: If pyautogen is not installed
        """
        self._name = name
        self._description = description or f"LinJ Agent executing {workflow}"
        self._plugin = LinJPluginBase(workflow, tools)

        # Store configuration for lazy agent creation
        self._llm_config = llm_config or {}
        self._system_message = system_message or self._generate_system_message()
        self._is_termination_msg = is_termination_msg
        self._max_consecutive_auto_reply = max_consecutive_auto_reply
        self._human_input_mode = human_input_mode
        self._agent_kwargs = kwargs

        self._agent = None

    def _generate_system_message(self) -> str:
        """Generate default system message for the agent."""
        lines = [
            f"You are {self._name}.",
            self._description,
            "",
            "You execute LinJ workflows to process requests.",
            "Available tools:",
        ]

        for tool_name in self._plugin._tools.keys():
            lines.append(f"- {tool_name}")

        lines.extend(
            [
                "",
                "When you receive a request, it will be processed through your configured workflow.",
                "Return clear, structured responses based on the workflow output.",
            ]
        )

        return "\n".join(lines)

    def _ensure_agent_created(self):
        """Lazy creation of AutoGen agent to avoid import errors."""
        if self._agent is not None:
            return

        try:
            import autogen
        except ImportError:
            raise ImportError(
                "pyautogen is required for LinJFunctionAgent. "
                "Install with: pip install pyautogen"
            )

        # Create the underlying agent
        self._agent = autogen.ConversableAgent(
            name=self._name,
            system_message=self._system_message,
            llm_config=self._llm_config,
            is_termination_msg=self._is_termination_msg,
            max_consecutive_auto_reply=self._max_consecutive_auto_reply,
            human_input_mode=self._human_input_mode,
            **self._agent_kwargs,
        )

        # Set up message handling to route through LinJ workflow
        self._setup_message_handler()

    def _setup_message_handler(self):
        """Configure the agent to use LinJ workflow for processing."""
        import autogen

        # Store reference to plugin
        plugin = self._plugin

        # Override the agent's message generation to use LinJ
        original_generate_reply = self._agent.generate_reply

        def generate_reply_with_linj(messages=None, sender=None, **kwargs):
            """Generate reply by executing LinJ workflow."""
            # Get the last user message
            if messages and len(messages) > 0:
                last_message = messages[-1].get("content", "")
            else:
                last_message = ""

            # Execute workflow
            import asyncio

            try:
                result = asyncio.run(plugin.run_workflow(last_message))

                # Extract response from result
                final_state = result.get("final_state", {})

                # Try common output paths
                if "output" in final_state:
                    return final_state["output"]
                elif "result" in final_state:
                    return final_state["result"]
                elif "response" in final_state:
                    return final_state["response"]
                elif "message" in final_state:
                    return final_state["message"]
                else:
                    # Return formatted state
                    return f"Workflow completed. Result: {final_state}"

            except Exception as e:
                return f"Error executing workflow: {str(e)}"

        # Replace the generate_reply method
        self._agent.generate_reply = generate_reply_with_linj

    @property
    def agent(self):
        """Get or create the underlying AutoGen agent."""
        self._ensure_agent_created()
        return self._agent

    def register_tool(self, name: str, func: Callable) -> "LinJFunctionAgent":
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

    def register_tools(self, tools: Dict[str, Callable]) -> "LinJFunctionAgent":
        """Batch register tools."""
        self._plugin.register_tools(tools)
        return self

    async def run(self, message: str, **kwargs) -> Dict[str, Any]:
        """
        Run the workflow directly (bypassing agent chat).

        Args:
            message: Input message
            **kwargs: Additional context

        Returns:
            Workflow execution result
        """
        return await self._plugin.run_workflow(message, kwargs)

    def run_sync(self, message: str, **kwargs) -> Dict[str, Any]:
        """Synchronous version of run."""
        return self._plugin.run_workflow_sync(message, kwargs)


class LinJUserProxyAgent:
    """
    UserProxyAgent wrapper that routes messages through LinJ workflow.

    This agent acts as a proxy that:
    1. Intercepts incoming messages
    2. Processes them through LinJ workflow
    3. Returns processed results

    Useful for:
    - Pre-processing user inputs
    - Routing messages to appropriate agents
    - Orchestrating complex workflows

    Example:
        >>> proxy = LinJUserProxyAgent(
        ...     name="workflow_manager",
        ...     workflow="workflows/orchestrate.yaml",
        ...     tools={"route": route_func}
        ... )
        >>> groupchat = autogen.GroupChat(agents=[proxy, assistant, coder], ...)
    """

    def __init__(
        self,
        name: str,
        workflow: Union[str, Path, Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        system_message: Optional[str] = None,
        tools: Optional[Dict[str, Callable]] = None,
        human_input_mode: str = "ALWAYS",
        max_consecutive_auto_reply: int = 10,
        **kwargs,
    ):
        """
        Initialize LinJ UserProxyAgent.

        Args:
            name: Agent name
            workflow: LinJ workflow definition
            llm_config: Optional LLM configuration
            description: Agent description
            system_message: System message
            tools: Tools for LinJ workflow
            human_input_mode: When to ask for human input
            max_consecutive_auto_reply: Max auto replies
            **kwargs: Additional arguments for UserProxyAgent
        """
        self._name = name
        self._description = description or f"LinJ Proxy Agent for {workflow}"
        self._plugin = LinJPluginBase(workflow, tools)

        self._llm_config = llm_config
        self._system_message = system_message
        self._human_input_mode = human_input_mode
        self._max_consecutive_auto_reply = max_consecutive_auto_reply
        self._agent_kwargs = kwargs

        self._agent = None

    def _ensure_agent_created(self):
        """Lazy creation of UserProxyAgent."""
        if self._agent is not None:
            return

        try:
            import autogen
        except ImportError:
            raise ImportError(
                "pyautogen is required for LinJUserProxyAgent. "
                "Install with: pip install pyautogen"
            )

        # Store plugin reference for use in method
        plugin = self._plugin

        # Create a custom UserProxyAgent class
        class _LinJUserProxy(autogen.UserProxyAgent):
            """Custom UserProxy that routes through LinJ."""

            def __init__(inner_self, **kwargs):
                super().__init__(**kwargs)
                inner_self._linj_plugin = plugin

            def generate_reply(inner_self, messages=None, sender=None, **kwargs):
                """Process message through LinJ before generating reply."""
                # Get message content
                if messages and len(messages) > 0:
                    last_message = messages[-1].get("content", "")
                else:
                    last_message = ""

                # Run through LinJ workflow
                import asyncio

                try:
                    result = asyncio.run(
                        inner_self._linj_plugin.run_workflow(last_message)
                    )

                    # Extract output from result
                    final_state = result.get("final_state", {})

                    # Look for common output paths
                    if "output" in final_state:
                        return final_state["output"]
                    elif "result" in final_state:
                        return final_state["result"]
                    elif "response" in final_state:
                        return final_state["response"]
                    elif "message" in final_state:
                        return final_state["message"]
                    else:
                        # Return formatted state
                        return f"Workflow result: {final_state}"

                except Exception as e:
                    # Fall back to default behavior on error
                    return super(_LinJUserProxy, inner_self).generate_reply(
                        messages=messages, sender=sender, **kwargs
                    )

        # Create the agent instance
        self._agent = _LinJUserProxy(
            name=self._name,
            system_message=self._system_message,
            llm_config=self._llm_config,
            human_input_mode=self._human_input_mode,
            max_consecutive_auto_reply=self._max_consecutive_auto_reply,
            **self._agent_kwargs,
        )

    @property
    def agent(self):
        """Get or create the underlying UserProxyAgent."""
        self._ensure_agent_created()
        return self._agent

    def register_tool(self, name: str, func: Callable) -> "LinJUserProxyAgent":
        """Register a tool for the workflow."""
        self._plugin.register_tool(name, func)
        return self

    def register_tools(self, tools: Dict[str, Callable]) -> "LinJUserProxyAgent":
        """Batch register tools."""
        self._plugin.register_tools(tools)
        return self


# Convenience function
def create_autogen_agent(
    name: str,
    workflow: Union[str, Path, Dict[str, Any]],
    agent_type: str = "function",
    **kwargs,
) -> Union[LinJFunctionAgent, LinJUserProxyAgent]:
    """
    Create an AutoGen agent that wraps a LinJ workflow.

    Args:
        name: Agent name
        workflow: Path to YAML file, YAML string, or dict
        agent_type: "function" or "user_proxy"
        **kwargs: Additional arguments for agent

    Returns:
        Configured agent wrapper

    Example:
        >>> agent = create_autogen_agent(
        ...     name="my_agent",
        ...     workflow="workflows/my_workflow.yaml",
        ...     agent_type="function",
        ...     llm_config={"config_list": [...]}
        ... )
    """
    if agent_type == "function":
        return LinJFunctionAgent(name=name, workflow=workflow, **kwargs)
    elif agent_type == "user_proxy":
        return LinJUserProxyAgent(name=name, workflow=workflow, **kwargs)
    else:
        raise ValueError(
            f"Unknown agent_type: {agent_type}. Choose from: function, user_proxy"
        )
