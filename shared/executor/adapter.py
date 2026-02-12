"""
工具适配器抽象接口

定义工具函数的适配器接口，支持不同工具框架的集成。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Awaitable
import inspect
import asyncio

from .types import ToolResult, AsyncCallable


class ToolAdapter(ABC):
    """
    工具适配器抽象接口

    定义工具函数的适配和转换逻辑，支持：
    - 函数签名适配
    - 参数转换和验证
    - 结果格式标准化
    - 错误处理和重试
    """

    @abstractmethod
    def adapt_function(self, func: Callable) -> AsyncCallable:
        """
        将普通函数适配为异步工具函数

        Args:
            func: 原始函数

        Returns:
            适配后的异步函数
        """
        pass

    @abstractmethod
    def validate_parameters(self, func: AsyncCallable, args: Dict[str, Any]) -> bool:
        """
        验证函数参数

        Args:
            func: 工具函数
            args: 参数字典

        Returns:
            True 表示参数有效，False 表示无效
        """
        pass

    @abstractmethod
    def transform_result(self, result: Any, func: AsyncCallable) -> ToolResult:
        """
        转换函数结果为标准格式

        Args:
            result: 原始返回结果
            func: 工具函数

        Returns:
            标准化的工具结果
        """
        pass

    def get_tool_schema(self, func: AsyncCallable) -> Dict[str, Any]:
        """
        获取工具的JSON Schema描述

        Args:
            func: 工具函数

        Returns:
            JSON Schema格式的工具描述
        """
        sig = inspect.signature(func)
        schema = {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        for param_name, param in sig.parameters.items():
            param_type = "string"  # 默认类型

            # 简单的类型推断
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
    基础工具适配器实现

    提供通用的适配逻辑和默认实现
    """

    def __init__(self):
        self._registered_tools: Dict[str, AsyncCallable] = {}

    def adapt_function(self, func: Callable) -> AsyncCallable:
        """将普通函数适配为异步工具函数"""
        if asyncio.iscoroutinefunction(func):
            return func

        async def async_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return async_wrapper

    def register_tool(self, name: str, func: Callable) -> None:
        """
        注册工具函数

        Args:
            name: 工具名称
            func: 工具函数
        """
        adapted_func = self.adapt_function(func)
        self._registered_tools[name] = adapted_func

    def unregister_tool(self, name: str) -> None:
        """
        注销工具函数

        Args:
            name: 工具名称
        """
        if name in self._registered_tools:
            del self._registered_tools[name]

    def get_tool(self, name: str) -> Optional[AsyncCallable]:
        """
        获取注册的工具函数

        Args:
            name: 工具名称

        Returns:
            工具函数，如果不存在则返回None
        """
        return self._registered_tools.get(name)

    def list_tools(self) -> List[str]:
        """
        列出所有注册的工具名称

        Returns:
            工具名称列表
        """
        return list(self._registered_tools.keys())

    def validate_parameters(self, func: AsyncCallable, args: Dict[str, Any]) -> bool:
        """验证函数参数"""
        try:
            sig = inspect.signature(func)
            sig.bind(**args)
            return True
        except TypeError:
            return False

    def transform_result(self, result: Any, func: AsyncCallable) -> ToolResult:
        """转换函数结果为标准格式"""
        if isinstance(result, ToolResult):
            return result

        if isinstance(result, Exception):
            return ToolResult(success=False, error=result)

        return ToolResult(success=True, data=result)

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """
        执行工具函数

        Args:
            name: 工具名称
            args: 参数字典

        Returns:
            工具执行结果
        """
        if name not in self._registered_tools:
            return ToolResult(success=False, error=KeyError(f"Tool '{name}' not found"))

        func = self._registered_tools[name]

        # 验证参数
        if not self.validate_parameters(func, args):
            return ToolResult(
                success=False, error=TypeError(f"Invalid parameters for tool '{name}'")
            )

        try:
            # 执行工具
            result = await func(**args)
            return self.transform_result(result, func)
        except Exception as e:
            return ToolResult(success=False, error=e)


class LangChainToolAdapter(BaseToolAdapter):
    """
    LangChain工具适配器

    适配LangChain框架的工具函数
    """

    def adapt_function(self, func: Callable) -> AsyncCallable:
        """适配LangChain工具函数"""
        async_wrapper = super().adapt_function(func)

        async def langchain_wrapper(*args, **kwargs):
            # LangChain特定的参数处理
            if "kwargs" in kwargs:
                # 处理LangChain的kwargs参数
                langchain_kwargs = kwargs.pop("kwargs", {})
                kwargs.update(langchain_kwargs)

            return await async_wrapper(*args, **kwargs)

        return langchain_wrapper

    def get_tool_schema(self, func: AsyncCallable) -> Dict[str, Any]:
        """获取LangChain格式的工具描述"""
        schema = super().get_tool_schema(func)

        # 添加LangChain特定的字段
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
    AutoGen工具适配器

    适配AutoGen框架的工具函数
    """

    def adapt_function(self, func: Callable) -> AsyncCallable:
        """适配AutoGen工具函数"""
        async_wrapper = super().adapt_function(func)

        async def autogen_wrapper(*args, **kwargs):
            # AutoGen特定的参数处理
            # AutoGen通常使用位置参数或特定的参数结构
            if len(args) == 1 and isinstance(args[0], str):
                # 如果只有一个字符串参数，可能是AutoGen的特定格式
                kwargs["message"] = args[0]
                args = ()

            return await async_wrapper(*args, **kwargs)

        return autogen_wrapper

    def transform_result(self, result: Any, func: AsyncCallable) -> ToolResult:
        """转换结果为AutoGen格式"""
        base_result = super().transform_result(result, func)

        # AutoGen特定的结果格式
        if base_result.success and base_result.data is not None:
            # AutoGen通常期望字符串结果
            if not isinstance(base_result.data, str):
                base_result.data = str(base_result.data)

        return base_result


class DummyToolAdapter(BaseToolAdapter):
    """
    虚拟工具适配器

    用于测试和开发环境
    """

    def __init__(self):
        super().__init__()
        # 注册一些虚拟工具
        self.register_dummy_tools()

    def register_dummy_tools(self):
        """注册虚拟工具函数"""

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
