"""
Executor Adapter

This module provides integration between LangGraph workflows and the shared
executor component.
"""

from typing import Any, Dict, Optional, List

from .base import BaseAdapter
from ..types import ExecutionContext, ExecutionResult


class ExecutorAdapter(BaseAdapter):
    """
    Adapter for integrating LangGraph with the shared executor

    This adapter allows LangGraph workflows to use the shared executor
    backend for scheduling, monitoring, and managing task execution.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._executor = None
        self._scheduler = None

    def initialize(self) -> None:
        """Initialize executor adapter"""
        try:
            # Import executor components here to avoid circular imports
            from shared.executor.backend import DummyExecutionBackend
            from shared.executor.autogen_scheduler import (
                AutoGenDeterministicScheduler,
                check_concurrent_safety,
                find_concurrent_groups,
                get_node_path_set,
            )

            # Use dummy backend for now, replace with actual backend when available
            self._executor = DummyExecutionBackend()

            # Use AutoGenDeterministicScheduler for consistent scheduling
            self._scheduler = AutoGenDeterministicScheduler([])

            # Store concurrency safety functions for use in execution
            self._check_concurrent_safety = check_concurrent_safety
            self._find_concurrent_groups = find_concurrent_groups
            self._get_node_path_set = get_node_path_set
        except ImportError as e:
            raise RuntimeError(f"Failed to import executor components: {e}")

    def cleanup(self) -> None:
        """Clean up executor adapter resources"""
        if hasattr(self._executor, "cleanup"):
            self._executor.cleanup()
        if hasattr(self._scheduler, "cleanup"):
            self._scheduler.cleanup()
        self._executor = None
        self._scheduler = None

    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute a task using the shared executor

        Args:
            context: Execution context with task information

        Returns:
            Execution result from the executor
        """
        if not self._executor:
            raise RuntimeError("Executor adapter not initialized")

        # Check concurrent safety if multiple nodes are ready
        if hasattr(context, "ready_nodes") and len(context.ready_nodes) > 1:
            # Use shared concurrency safety checking
            safe_nodes = self._find_concurrent_executables(context.ready_nodes)
            if context.node_id not in [getattr(n, "id", "unknown") for n in safe_nodes]:
                return ExecutionResult(
                    success=False,
                    node_id=context.node_id,
                    step_id=context.step_id,
                    error="Concurrent execution blocked by safety constraints",
                    error_type="ConcurrencySafetyError",
                )

        # Prepare task for executor
        task_config = {
            "task_id": context.node_id,
            "workflow_id": context.workflow_id,
            "step_id": context.step_id,
            "task_type": context.node_config.node_type.value,
            "config": context.node_config.metadata,
            "input_data": context.input_data,
        }

        try:
            # Submit task to executor
            task_future = self._executor.submit(task_config)

            # Wait for result (simplified - in real implementation would be async)
            result = task_future.result(timeout=context.node_config.timeout)

            return ExecutionResult(
                success=True,
                node_id=context.node_id,
                step_id=context.step_id,
                output_data=result.get("output"),
                artifacts=result.get("artifacts", {}),
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                node_id=context.node_id,
                step_id=context.step_id,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _find_concurrent_executables(self, nodes: List[Any]) -> List[Any]:
        """
        Find nodes that can be safely executed concurrently

        Uses shared concurrency safety detection from autogen_scheduler
        """
        if not self._find_concurrent_groups:
            # Fallback: return first node only (sequential execution)
            return nodes[:1] if nodes else []

        # Use shared concurrent grouping function
        groups = self._find_concurrent_groups(nodes)
        return groups[0] if groups else []

    def validate_context(self, context: ExecutionContext) -> bool:
        """Validate executor-specific context"""
        if not super().validate_context(context):
            return False

        # Check for executor-specific requirements
        required_fields = ["node_config", "input_data"]
        for field in required_fields:
            if not hasattr(context, field) or getattr(context, field) is None:
                return False

        return True
