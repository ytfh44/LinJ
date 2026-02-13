"""
LinJ Node Definitions

Implements node types and semantics defined in Specification Sections 6 and 13
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Contract(BaseModel):
    """
    Contract (input/output constraints)

    Minimal contract language defined in Section 7.1
    """

    type: str  # object/array/string/number/boolean/null
    required: Optional[List[str]] = None
    properties: Optional[Dict[str, "Contract"]] = None
    items: Optional["Contract"] = None


class NodePolicy(BaseModel):
    """Node Policy (Section 10.2)"""

    allow_reenter: bool = False


class ValueRef(BaseModel):
    """
    Path reference or constant (Section 6.2)

    - {"$path": "$.x.y"} represents a path reference
    - {"$const": <value>} represents a constant
    """

    model_config = ConfigDict(populate_by_name=True)

    path: Optional[str] = Field(None, alias="$path")
    const: Optional[Any] = Field(None, alias="$const")

    def resolve(self, state: Dict[str, Any]) -> Any:
        """Resolve value from state"""
        from .path import PathResolver

        if self.path is not None:
            return PathResolver.get(state, self.path)
        return self.const

    @classmethod
    def from_value(cls, value: Any) -> "ValueRef":
        """
        Create ValueRef from value

        Compatibility mode: strings starting with $. are treated as paths
        """
        if isinstance(value, str) and value.startswith("$."):
            return cls(path=value)
        return cls(const=value)


class Node(BaseModel, ABC):
    """
    Node base class (Section 6.1)

    All nodes must include:
    - id: globally unique identifier
    - type: one of hint/tool/join/gate
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
    hint node (Section 13.1)

    Hint generation node for template rendering
    """

    type: str = "hint"
    template: str
    vars: Optional[Dict[str, Union[ValueRef, Any]]] = None
    write_to: str

    def render(self, state: Dict[str, Any]) -> str:
        """
        Render template

        Replace {{name}} with corresponding variable value
        """
        import re

        result = self.template

        if self.vars:
            for name, ref in self.vars.items():
                if isinstance(ref, dict):
                    ref = ValueRef(**ref)
                elif not isinstance(ref, ValueRef):
                    ref = ValueRef.from_value(ref)

                value = ref.resolve(state)
                if value is None:
                    raise ValueError(f"Variable {name} not found in state")

                # Replace {{name}}
                placeholder = f"{{{{{name}}}}}"
                str_value = "" if value is None else str(value)
                result = result.replace(placeholder, str_value)

        return result


class ToolCall(BaseModel):
    """Tool call definition"""

    name: str
    args: Optional[Dict[str, Union[ValueRef, Any]]] = None


class Effect(str, Enum):
    """Tool effect types"""

    NONE = "none"
    READ = "read"
    WRITE = "write"


class ToolNode(Node):
    """
    tool node (Section 13.2)

    Tool call node
    """

    type: str = "tool"
    call: ToolCall
    write_to: Optional[str] = None
    effect: Effect = Effect.READ
    repeat_safe: bool = False

    def get_args(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse tool arguments"""
        if not self.call.args:
            return {}

        result = {}
        for key, ref in self.call.args.items():
            if isinstance(ref, dict):
                ref = ValueRef(**ref)
            elif not isinstance(ref, ValueRef):
                ref = ValueRef.from_value(ref)

            value = ref.resolve(state)
            result[key] = value

        return result

    def can_retry(self) -> bool:
        """Check if automatic retry is allowed"""
        return self.effect in (Effect.NONE, Effect.READ) or self.repeat_safe


class GlossaryItem(BaseModel):
    """Glossary item (Section 13.3)"""

    prefer: Optional[str] = None
    forbid: Optional[List[str]] = None


class JoinNode(Node):
    """
    join node (Section 13.3)

    Text joining node
    """

    type: str = "join"
    input_from: str
    output_to: str
    language: Optional[str] = None
    style: Optional[str] = None
    glossary: Optional[List[GlossaryItem]] = None

    def validate_forbidden(self, text: str) -> Optional[str]:
        """
        Validate if forbidden items are included

        Returns None if validation passes; otherwise returns the first forbidden item found
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
    gate node (Section 13.4)

    Conditional gating node
    """

    type: str = "gate"
    condition: str
    then: List[str]
    else_: List[str] = Field(alias="else", default_factory=list)

    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Evaluate condition"""
        from ..executor.evaluator import evaluate_condition

        return evaluate_condition(self.condition, state)

    def get_next_nodes(self, state: Dict[str, Any]) -> List[str]:
        """Get next nodes"""
        if self.evaluate(state):
            return self.then
        return self.else_


# Union type for all node types
NodeType = Union[HintNode, ToolNode, JoinNode, GateNode]


def parse_node(data: Dict[str, Any]) -> Node:
    """Parse node from dictionary"""
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
