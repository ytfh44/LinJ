"""LinJ Plugin Framework - Base classes and utilities for framework integration.

This package provides plugin wrappers that allow LinJ to be embedded into
existing AutoGen and LangGraph projects.

Usage:
    # AutoGen integration
    from linj.plugins.autogen import LinJFunctionAgent, LinJUserProxyAgent

    # LangGraph integration
    from linj.plugins.langgraph import LinJNode, StateMapper

    # Base functionality
    from linj.plugins.base import LinJPluginBase
"""

__version__ = "0.1.0"

# Base classes
from .base import LinJPluginBase

# AutoGen integration (requires pyautogen)
from .autogen import (
    LinJFunctionAgent,
    LinJUserProxyAgent,
    create_autogen_agent,
)

# LangGraph integration (requires langgraph)
from .langgraph import (
    LinJNode,
    LinJStateGraph,
    StateMapper,
    create_langgraph_node,
)

__all__ = [
    # Base
    "LinJPluginBase",
    # AutoGen
    "LinJFunctionAgent",
    "LinJUserProxyAgent",
    "create_autogen_agent",
    # LangGraph
    "LinJNode",
    "LinJStateGraph",
    "StateMapper",
    "create_langgraph_node",
]
