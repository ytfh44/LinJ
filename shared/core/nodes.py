"""
LinJ 节点定义

实现规范 6 节、13 节定义的节点类型和语义
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Contract(BaseModel):
    """
    合同（输入/输出约束）

    7.1 节定义的最小合同语言
    """

    type: str  # object/array/string/number/boolean/null
    required: Optional[List[str]] = None
    properties: Optional[Dict[str, "Contract"]] = None
    items: Optional["Contract"] = None


class NodePolicy(BaseModel):
    """节点策略 (10.2 节)"""

    allow_reenter: bool = False


class ValueRef(BaseModel):
    """
    路径引用或常量 (6.2 节)

    - {"$path": "$.x.y"} 表示路径引用
    - {"$const": <value>} 表示常量
    """

    model_config = {"populate_by_name": True}

    path: Optional[str] = Field(default=None, alias="$path")
    const: Optional[Any] = Field(default=None, alias="$const")

    def resolve(self, state: Dict[str, Any]) -> Any:
        """从状态解析值"""
        from .path import PathResolver

        if self.path is not None:
            return PathResolver.get(state, self.path)
        return self.const

    @classmethod
    def from_value(cls, value: Any) -> "ValueRef":
        """
        从值创建 ValueRef

        兼容模式：字符串以 $. 开头视为路径
        """
        if isinstance(value, str) and value.startswith("$."):
            return cls(path=value)
        return cls(const=value)


class Node(BaseModel, ABC):
    """
    节点基类 (6.1 节)

    所有节点必须包含：
    - id: 全局唯一标识
    - type: hint/tool/join/gate 之一
    """

    id: str
    type: str
    title: Optional[str] = None
    description: Optional[str] = None
    reads: Optional[List[str]] = None
    writes: Optional[List[str]] = None
    in_contract: Optional[Contract] = None
    out_contract: Optional[Contract] = None
    policy: Optional[NodePolicy] = None
    rank: Optional[float] = None


class HintNode(Node):
    """
    hint 节点 (13.1 节)

    提示生成节点，用于模板渲染
    """

    type: str = "hint"
    template: str
    vars: Optional[Dict[str, Union[ValueRef, Any]]] = None
    write_to: str

    def render(self, state: Dict[str, Any]) -> str:
        """
        渲染模板

        替换 {{name}} 为对应变量值
        """
        import re

        result = self.template

        if self.vars:
            for name, ref in self.vars.items():
                if isinstance(ref, dict):
                    ref = ValueRef.model_validate(ref)
                elif not isinstance(ref, ValueRef):
                    ref = ValueRef.from_value(ref)

                value = ref.resolve(state)
                if value is None:
                    raise ValueError(f"Variable {name} not found in state")

                # 替换 {{name}}
                placeholder = f"{{{{{name}}}}}"
                str_value = "" if value is None else str(value)
                result = result.replace(placeholder, str_value)

        return result


class ToolCall(BaseModel):
    """工具调用定义"""

    name: str
    args: Optional[Dict[str, Union[ValueRef, Any]]] = None


class Effect(str, Enum):
    """工具效果类型"""

    NONE = "none"
    READ = "read"
    WRITE = "write"


class ToolNode(Node):
    """
    tool 节点 (13.2 节)

    工具调用节点
    """

    type: str = "tool"
    call: ToolCall
    write_to: Optional[str] = None
    effect: Effect = Effect.READ
    repeat_safe: bool = False

    def get_args(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """解析工具参数"""
        if not self.call.args:
            return {}

        result = {}
        for key, ref in self.call.args.items():
            if isinstance(ref, dict):
                ref = ValueRef.model_validate(ref)
            elif not isinstance(ref, ValueRef):
                ref = ValueRef.from_value(ref)

            value = ref.resolve(state)
            result[key] = value

        return result

    def can_retry(self) -> bool:
        """检查是否可以自动重试"""
        return self.effect in (Effect.NONE, Effect.READ) or self.repeat_safe


class GlossaryItem(BaseModel):
    """术语表项 (13.3 节)"""

    prefer: Optional[str] = None
    forbid: Optional[List[str]] = None


class JoinNode(Node):
    """
    join 节点 (13.3 节)

    文本接合节点
    """

    type: str = "join"
    input_from: str
    output_to: str
    language: Optional[str] = None
    style: Optional[str] = None
    glossary: Optional[List[GlossaryItem]] = None

    def validate_forbidden(self, text: str) -> Optional[str]:
        """
        验证是否包含禁止项

        返回 None 表示验证通过；否则返回第一个发现的禁止项
        """
        if not self.glossary:
            return None

        for item in self.glossary:
            if item.forbid:
                for forbidden in item.forbid:
                    if forbidden in text:
                        return forbidden

        return None


class GateNode(Node):
    """
    gate 节点 (13.4 节)

    条件门控节点
    """

    model_config = {"populate_by_name": True}

    type: str = "gate"
    condition: str
    then: List[str]
    else_: List[str] = Field(default_factory=list, alias="else")

    def evaluate(self, state: Dict[str, Any]) -> bool:
        """
        求值条件 (13.4 节, 14.x 节)

        使用条件表达式求值器计算条件结果
        支持比较运算符、逻辑运算符和内置函数

        Args:
            state: 主状态对象

        Returns:
            条件求值结果（True 或 False）

        Raises:
            SyntaxError: 条件表达式语法错误
            TypeError: 类型不匹配
        """
        from .condition import evaluate_condition

        return evaluate_condition(self.condition, state)

    def get_next_nodes(self, state: Dict[str, Any]) -> List[str]:
        """获取下一步节点"""
        if self.evaluate(state):
            return self.then
        return self.else_


# Union type for all node types
NodeType = Union[HintNode, ToolNode, JoinNode, GateNode]


def parse_node(data: Dict[str, Any]) -> Node:
    """从字典解析节点"""
    node_type = data.get("type")

    if node_type == "hint":
        return HintNode(**data)
    elif node_type == "tool":
        return ToolNode(**data)
    elif node_type == "join":
        return JoinNode(**data)
    elif node_type == "gate":
        return GateNode(**data)
    else:
        raise ValueError(f"Unknown node type: {node_type}")
