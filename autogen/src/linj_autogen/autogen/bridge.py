"""
Autogen 桥接器

桥接 LinJ 与 Autogen 的组件
增强 ContiText 状态管理和错误处理
"""

from typing import Any, Callable, Dict, Optional, List
import asyncio
import logging

# Use local imports for now
from ..core.errors import ExecutionError

logger = logging.getLogger(__name__)


class AutogenBridge:
    """
    Autogen 桥接器

    用于将 LinJ 工具注册为 Autogen 可调用工具
    支持 ContiText 集成和共享执行引擎抽象
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._timeout: Optional[float] = None
        self._retry_count: int = 3

    def register_tool(self, name: str, executor: Callable) -> None:
        """注册工具"""
        self._tools[name] = executor

    def get_tool(self, name: str) -> Optional[Callable]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有工具"""
        return list(self._tools.keys())

    async def execute_tool(self, name: str, **kwargs) -> Any:
        """
        执行工具

        增强错误处理和超时支持
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        metadata = self._tool_metadata.get(name, {})

        for attempt in range(self._retry_count):
            try:
                if asyncio.iscoroutinefunction(tool):
                    if self._timeout:
                        return await asyncio.wait_for(
                            tool(**kwargs), timeout=self._timeout
                        )
                    else:
                        return await tool(**kwargs)
                else:
                    return tool(**kwargs)
            except Exception as e:
                if attempt == self._retry_count - 1:
                    # 最后一次尝试失败，记录并抛出
                    logger.error(
                        f"Tool {name} failed after {attempt + 1} attempts: {e}"
                    )
                    raise RuntimeError(f"Tool execution failed: {e}") from e

                logger.warning(
                    f"Tool {name} attempt {attempt + 1} failed: {e}, retrying..."
                )
                await asyncio.sleep(0.1 * (attempt + 1))  # 指数退避

    def to_autogen_tools(self) -> list[Dict[str, Any]]:
        """
        转换为 Autogen 工具格式

        增强参数类型推断和描述生成
        """
        tools = []
        for name, func in self._tools.items():
            import inspect

            sig = inspect.signature(func)

            # 构建参数定义
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                param_info = self._infer_param_type(param)
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
                properties[param_name] = param_info

            # 获取工具描述
            description = func.__doc__ or f"Execute {name}"
            metadata = self._tool_metadata.get(name, {})
            if "description" in metadata:
                description = metadata["description"]

            tool_def = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

            # 添加额外元数据
            if "timeout" in metadata:
                tool_def["function"]["timeout"] = metadata["timeout"]

            tools.append(tool_def)

        return tools

    def _infer_param_type(self, param) -> Dict[str, Any]:
        """
        推断参数类型

        Args:
            param: inspect.Parameter 对象

        Returns:
            参数类型定义
        """
        # 基础类型推断
        if param.annotation == str:
            return {"type": "string"}
        elif param.annotation == int:
            return {"type": "integer"}
        elif param.annotation == float:
            return {"type": "number"}
        elif param.annotation == bool:
            return {"type": "boolean"}
        elif param.annotation == list:
            return {"type": "array", "items": {"type": "string"}}
        elif param.annotation == dict:
            return {"type": "object", "additionalProperties": {"type": "string"}}
        else:
            # 默认为字符串
            return {"type": "string"}

    def set_timeout(self, timeout: float) -> None:
        """设置工具执行超时"""
        self._timeout = timeout

    def set_retry_count(self, count: int) -> None:
        """设置重试次数"""
        self._retry_count = max(1, count)

    def add_tool_metadata(self, name: str, **metadata) -> None:
        """添加工具元数据"""
        if name in self._tools:
            self._tool_metadata[name] = metadata
