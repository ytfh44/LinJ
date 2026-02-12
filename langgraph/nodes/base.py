"""
Base Node Implementation

This module provides the base class for all LangGraph nodes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

from ..types import ExecutionContext, ExecutionResult, NodeConfig, NodeType


class BaseNode(ABC):
    """
    Base class for all LangGraph nodes

    Provides common functionality and interface for node implementations.
    Each node type should inherit from this class and implement the execute method.
    """

    def __init__(self, node_config: NodeConfig):
        """
        Initialize node with configuration

        Args:
            node_config: Configuration for this node
        """
        self.node_config = node_config
        self.node_id = node_config.node_id
        self.node_type = node_config.node_type
        self.logger = self._get_logger()

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute the node's primary logic

        Args:
            context: Execution context with state and configuration

        Returns:
            Execution result with output or error information
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before execution

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        # Basic validation - check required inputs
        for input_name, state_path in self.node_config.inputs.items():
            if input_name not in input_data:
                self.logger.error(f"Missing required input: {input_name}")
                return False

        return True

    def prepare_output(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare output data according to node configuration

        Args:
            result_data: Raw result data from node execution

        Returns:
            Formatted output data
        """
        output_data = {}

        # Map outputs according to configuration
        for output_name, state_path in self.node_config.outputs.items():
            if output_name in result_data:
                output_data[output_name] = result_data[output_name]

        # If no specific output mapping, use all result data
        if not self.node_config.outputs:
            output_data = result_data

        return output_data

    def should_execute(self, context: ExecutionContext) -> bool:
        """
        Check if node should execute based on conditions

        Args:
            context: Execution context

        Returns:
            True if node should execute, False otherwise
        """
        if not self.node_config.condition:
            return True

        # Simple condition evaluation - in real implementation would be more sophisticated
        try:
            # For now, just check if the condition string exists in state
            condition_result = context.state_view.read(self.node_config.condition)
            return bool(condition_result)
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return not self.node_config.skip_on_condition_false

    def get_dependencies_satisfied(self, context: ExecutionContext) -> bool:
        """
        Check if all dependencies are satisfied

        Args:
            context: Execution context

        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        for dependency in self.node_config.dependencies:
            # Check if dependency exists in state and is not None
            if not context.state_view.exists(dependency):
                self.logger.warning(f"Dependency not satisfied: {dependency}")
                return False

        return True

    def _get_logger(self) -> Any:
        """Get logger for this node"""
        try:
            import structlog

            return structlog.get_logger(f"node.{self.node_id}")
        except ImportError:
            import logging

            return logging.getLogger(f"node.{self.node_id}")

    def _start_execution_timer(self) -> float:
        """Start execution timer and return start time"""
        import time

        return time.time()

    def _end_execution_timer(self, start_time: float) -> float:
        """End execution timer and return duration in milliseconds"""
        import time

        duration_seconds = time.time() - start_time
        return duration_seconds * 1000  # Convert to milliseconds

    def _handle_execution_error(
        self,
        error: Exception,
        context: ExecutionContext,
        start_time: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Handle execution error and create appropriate result

        Args:
            error: Exception that occurred
            context: Execution context
            start_time: Start time for duration calculation

        Returns:
            Execution result with error information
        """
        duration_ms = None
        if start_time is not None:
            duration_ms = self._end_execution_timer(start_time)

        self.logger.error(
            f"Node {self.node_id} execution failed: {error}",
            exc_info=True,
            extra={
                "node_id": self.node_id,
                "workflow_id": context.workflow_id,
                "step_id": context.step_id,
            },
        )

        return ExecutionResult(
            success=False,
            node_id=self.node_id,
            step_id=context.step_id,
            error=str(error),
            error_type=type(error).__name__,
            duration_ms=duration_ms,
            should_retry=self._should_retry(error),
            retry_delay=self._get_retry_delay(context),
        )

    def _should_retry(self, error: Exception) -> bool:
        """
        Determine if execution should be retried based on error type

        Args:
            error: Exception that occurred

        Returns:
            True if should retry, False otherwise
        """
        # Default implementation - retry on network errors and timeouts
        retryable_errors = [
            "TimeoutError",
            "ConnectionError",
            "HTTPError",
            "RequestException",
        ]

        return type(error).__name__ in retryable_errors

    def _get_retry_delay(self, context: ExecutionContext) -> Optional[float]:
        """
        Calculate retry delay based on configuration and attempt number

        Args:
            context: Execution context

        Returns:
            Retry delay in seconds, or None if no retry
        """
        if context.attempt_number >= self.node_config.max_retries:
            return None

        delay = self.node_config.retry_delay

        if self.node_config.retry_policy.value == "exponential_backoff":
            delay *= self.node_config.retry_backoff_factor ** (
                context.attempt_number - 1
            )
        elif self.node_config.retry_policy.value == "linear_backoff":
            delay *= context.attempt_number

        return min(delay, 60.0)  # Cap at 60 seconds

    def get_node_info(self) -> Dict[str, Any]:
        """
        Get information about this node

        Returns:
            Dictionary with node metadata
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "description": self.node_config.description,
            "dependencies": self.node_config.dependencies,
            "inputs": self.node_config.inputs,
            "outputs": self.node_config.outputs,
            "timeout": self.node_config.timeout,
            "max_retries": self.node_config.max_retries,
            "retry_policy": self.node_config.retry_policy.value,
            "tags": self.node_config.tags,
            "metadata": self.node_config.metadata,
        }
