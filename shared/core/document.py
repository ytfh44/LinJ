"""
LinJ 文档模型

实现规范 4 节定义的 LinJ 文档结构
"""

import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from .nodes import Node, parse_node, NodeType
from .edges import Edge, DependencyGraph, EdgeKind
from ..exceptions.errors import (
    ValidationError,
    InvalidRequirements,
    InvalidPlacement,
    ResourceConstraintUnsatisfied,
)


class Policies(BaseModel):
    """
    全局策略 (10.1 节)
    """

    max_steps: Optional[int] = None
    max_rounds: Optional[int] = None
    timeout_ms: Optional[int] = None
    retry: Optional[Dict[str, Any]] = None
    max_array_length: Optional[int] = None
    max_local_state_bytes: Optional[int] = None


class Loop(BaseModel):
    """
    显式循环 (12 节)
    """

    id: str
    entry: str
    members: List[str]
    mode: str = "finite"  # finite 或 infinite
    stop_condition: Optional[str] = None
    max_rounds: Optional[int] = None


class Placement(BaseModel):
    """
    放置声明 (15.2 节)
    """

    target: str  # 节点 id 或 resource_name
    domain: str  # 执行域标签


class Requirements(BaseModel):
    """
    运行要求 (15.1 节)
    """

    allow_parallel: bool = False
    allow_child_units: bool = False
    require_resume: bool = False


class LinJDocument(BaseModel):
    """
    LinJ 文档 (4.1 节)

    必须包含：
    - linj_version: 版本号
    - nodes: 节点数组
    - edges: 依赖边数组
    """

    linj_version: str
    nodes: List[NodeType] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    loops: Optional[List[Loop]] = None
    policies: Optional[Policies] = None
    requirements: Optional[Requirements] = None
    placement: Optional[List[Placement]] = None

    @field_validator("nodes", mode="before")
    @classmethod
    def parse_nodes(cls, v):
        """解析节点列表"""
        if v is None:
            return []
        return [parse_node(node) if isinstance(node, dict) else node for node in v]

    @field_validator("linj_version")
    @classmethod
    def validate_version(cls, v):
        """验证版本号格式"""
        parts = v.split(".")
        if len(parts) != 2:
            raise ValidationError(f"Invalid version format: {v}. Expected: major.minor")
        try:
            int(parts[0])
            int(parts[1])
        except ValueError:
            raise ValidationError(f"Invalid version numbers: {v}")
        return v

    def get_major_version(self) -> int:
        """获取主版本号"""
        return int(self.linj_version.split(".")[0])
