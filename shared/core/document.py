"""
LinJ Document Model

Implements the LinJ document structure defined in Specification Section 4
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
    Global policies (Section 10.1)
    """

    max_steps: Optional[int] = None
    max_rounds: Optional[int] = None
    timeout_ms: Optional[int] = None
    retry: Optional[Dict[str, Any]] = None
    max_array_length: Optional[int] = None
    max_local_state_bytes: Optional[int] = None


class Loop(BaseModel):
    """
    Explicit loop (Section 12)
    """

    id: str
    entry: str
    members: List[str]
    mode: str = "finite"  # finite or infinite
    stop_condition: Optional[str] = None
    max_rounds: Optional[int] = None


class Placement(BaseModel):
    """
    Placement declaration (Section 15.2)
    """

    target: str  # node id or resource_name
    domain: str  # execution domain label


class Requirements(BaseModel):
    """
    Runtime requirements (Section 15.1)
    """

    allow_parallel: bool = False
    allow_child_units: bool = False
    require_resume: bool = False


class LinJDocument(BaseModel):
    """
    LinJ Document (Section 4.1)

    Required fields:
    - linj_version: version number
    - nodes: array of nodes
    - edges: array of dependency edges
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
        """Parse node list"""
        if v is None:
            return []
        return [parse_node(node) if isinstance(node, dict) else node for node in v]

    @field_validator("linj_version")
    @classmethod
    def validate_version(cls, v):
        """Validate version format"""
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
        """Get major version number"""
        return int(self.linj_version.split(".")[0])
