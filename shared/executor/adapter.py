"""
Tool Adapter Abstract Interface

Defines the adapter interface for tool functions, supporting integration with different tool frameworks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
import inspect
import asyncio

from .types import ToolResult, AsyncCallable


class ToolAdapter(ABC):
    """
    Tool Adapter Abstract Interface

    Defines the adaptation and conversion logic for tool functions, supporting:
    - Function signature adaptation
    - Parameter conversion and validation
    - Result format standardization
    - Error handling and retry
    """

    @abstractmethod
    def adapt_function(self, func: Callable) -> AsyncCallable:
        """
        Adapt a regular function to an async tool function

        Args:
            func: The original function

        Returns:
            The adapted async function
        """
        pass

    @abstractmethod
    def validate_parameters(self, func: AsyncCallable, args: Dict[str, Any]) -> bool:
        """
        Validate function parameters

        Args:
            func: The tool function
            args: The arguments dictionary

        Returns:
            True if parameters are valid, False otherwise
        """
        pass

    @abstractmethod
    def transform_result(self, result: Any, func: AsyncCallable) -> ToolResult:
        """
        Transform function result to standard format

        Args:
            result: The original return value
            func: The tool function

        Returns:
            The standardized tool result
        """
        pass

    def get_tool_schema(self, func: AsyncCallable) -> Dict[str, Any]:
        """
        Get JSON Schema description for the tool

        Args:
            func: The tool function

        Returns:
            Tool description in JSON Schema format
        """
        sig = inspect.signature(func)
        schema = {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        for param_name, param in sig.parameters.items():
            param_type = "string"  # Default type

            # Simple type inference
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"

            schema["parameters"]["properties"][param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",
            }

            if param.default == inspect.Parameter.empty:
                schema["parameters"]["required"].append(param_name)
            else:
                schema["parameters"]["properties"][param_name]["default"] = (
                    param.default
                )

        return schema


class BaseToolAdapter(ToolAdapter):
    """
    Base Tool Adapter Implementation

    Provides common adaptation logic and default implementations
    """

    def __init__(self):
        self._registered_tools: Dict[str, AsyncCallable] = {}

    def adapt_function(self, func: Callable) -> AsyncCallable:
        """Adapt a regular function to an async tool function"""
        if asyncio.iscoroutinefunction(func):
            return func

        async def async_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return async_wrapper

    def register_tool(self, name: str, func: Callable) -> None:
        """
        Register a tool function

        Args:
            name: Tool name
            func: Tool function
        """
        adapted_func = self.adapt_function(func)
        self._registered_tools[name] = adapted_func

    def unregister_tool(self, name: str) -> None:
        """
        Unregister a tool function

        Args:
            name: Tool name
        """
        if name in self._registered_tools:
            del self._registered_tools[name]

    def get_tool(self, name: str) -> Optional[AsyncCallable]:
        """
        Get a registered tool function

        Args:
            name: Tool name

        Returns:
            The tool function, or None if not found
        """
        return self._registered_tools.get(name)

    def list_tools(self) -> List[str]:
        """
        List all registered tool names

        Returns:
            List of tool names
        """
        return list(self._registered_tools.keys())

    def validate_parameters(self, func: AsyncCallable, args: Dict[str, Any]) -> bool:
        """Validate function parameters"""
        try:
            sig = inspect.signature(func)
            sig.bind(**args)
            return True
        except TypeError:
            return False

    def transform_result(self, result: Any, func: AsyncCallable) -> ToolResult:
        """Transform function result to standard format"""
        if isinstance(result, ToolResult):
            return result

        if isinstance(result, Exception):
            return ToolResult(success=False, error=result)

        return ToolResult(success=True, data=result)

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool function

        Args:
            name: Tool name
            args: Arguments dictionary

        Returns:
            Tool execution result
        """
        if name not in self._registered_tools:
            return ToolResult(success=False, error=KeyError(f"Tool '{name}' not found"))

        func = self._registered_tools[name]

        # Validate parameters
        if not self.validate_parameters(func, args):
            return ToolResult(
                success=False, error=TypeError(f"Invalid parameters for tool '{name}'")
            )

        try:
            # Execute tool
            result = await func(**args)
            return self.transform_result(result, func)
        except Exception as e:
            return ToolResult(success=False, error=e)


class LangChainToolAdapter(BaseToolAdapter):
    """
    LangChain Tool Adapter

    Adapts tool functions for the LangChain framework
    """

    def adapt_function(self, func: Callable) -> AsyncCallable:
        """Adapt LangChain tool function"""
        async_wrapper = super().adapt_function(func)

        async def langchain_wrapper(*args, **kwargs):
            # LangChain-specific parameter handling
            if "kwargs" in kwargs:
                # Handle LangChain's kwargs parameter
                langchain_kwargs = kwargs.pop("kwargs", {})
                kwargs.update(langchain_kwargs)

            return await async_wrapper(*args, **kwargs)

        return langchain_wrapper

    def get_tool_schema(self, func: AsyncCallable) -> Dict[str, Any]:
        """Get tool description in LangChain format"""
        schema = super().get_tool_schema(func)

        # Add LangChain-specific fields
        schema.update(
            {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            }
        )

        return schema


class AutoGenToolAdapter(BaseToolAdapter):
    """
    AutoGen Tool Adapter

    Adapts tool functions for the AutoGen framework
    """

    def adapt_function(self, func: Callable) -> AsyncCallable:
        """Adapt AutoGen tool function"""
        async_wrapper = super().adapt_function(func)

        async def autogen_wrapper(*args, **kwargs):
            # AutoGen-specific parameter handling
            # AutoGen typically uses positional arguments or specific parameter structures
            if len(args) == 1 and isinstance(args[0], str):
                # If there's only one string argument, it might be AutoGen's specific format
                kwargs["message"] = args[0]
                args = ()

            return await async_wrapper(*args, **kwargs)

        return autogen_wrapper

    def transform_result(self, result: Any, func: AsyncCallable) -> ToolResult:
        """Transform result to AutoGen format"""
        base_result = super().transform_result(result, func)

        # AutoGen-specific result format
        if base_result.success and base_result.data is not None:
            # AutoGen typically expects string results
            if not isinstance(base_result.data, str):
                base_result.data = str(base_result.data)

        return base_result


class DummyToolAdapter(BaseToolAdapter):
    """
    Dummy Tool Adapter

    Used for testing and development environments
    """

    def __init__(self):
        super().__init__()
        # Register some dummy tools
        self.register_dummy_tools()

    def register_dummy_tools(self):
        """Register dummy tool functions"""

        def echo(message: str) -> str:
            return f"Echo: {message}"

        def add(a: int, b: int) -> int:
            return a + b

        def get_time() -> str:
            import datetime

            return datetime.datetime.now().isoformat()

        self.register_tool("echo", echo)
        self.register_tool("add", add)
        self.register_tool("get_time", get_time)
