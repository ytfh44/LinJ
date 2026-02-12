"""
Base Adapter for LangGraph Integration

This module provides the base adapter class for integrating LangGraph
with other system components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..types import ExecutionContext, ExecutionResult


class BaseAdapter(ABC):
    """
    Base class for all LangGraph adapters

    Adapters handle integration between LangGraph workflows and external systems
    or shared components.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter

        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config or {}

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the adapter (set up connections, etc.)"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the adapter"""
        pass

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute the adapter's primary function

        Args:
            context: Execution context with state and configuration

        Returns:
            Execution result with output or error information
        """
        pass

    def validate_context(self, context: ExecutionContext) -> bool:
        """
        Validate that the execution context is suitable for this adapter

        Args:
            context: Execution context to validate

        Returns:
            True if context is valid, False otherwise
        """
        return True

    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get information about this adapter

        Returns:
            Dictionary with adapter metadata
        """
        return {
            "adapter_type": self.__class__.__name__,
            "config": self.config,
        }
