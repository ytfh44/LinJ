"""
Contitext Adapter

This module provides integration between LangGraph workflows and the shared
contitext component for context management and continuation handling.
"""

from typing import Any, Dict, Optional

from .base import BaseAdapter
from ..types import ExecutionContext, ExecutionResult


class ContitextAdapter(BaseAdapter):
    """
    Adapter for integrating LangGraph with shared contitext component

    This adapter allows LangGraph workflows to use the shared contitext
    system for managing execution context, continuations, and state persistence.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._continuation_manager = None
        self._commit_manager = None
        self._mapper = None

    def initialize(self) -> None:
        """Initialize contitext adapter"""
        try:
            # Import contitext components here to avoid circular imports
            from shared.contitext import ContinuationManager, CommitManager, Mapper

            self._continuation_manager = ContinuationManager()
            self._commit_manager = CommitManager()
            self._mapper = Mapper()
        except ImportError as e:
            raise RuntimeError(f"Failed to import contitext components: {e}")

    def cleanup(self) -> None:
        """Clean up contitext adapter resources"""
        if self._continuation_manager:
            self._continuation_manager.cleanup()
        if self._commit_manager:
            self._commit_manager.cleanup()

    def execute(self, context: ExecutionContext) -> ExecutionResult:
        """
        Execute using contitext context management

        Args:
            context: Execution context with workflow information

        Returns:
            Execution result with context-managed state
        """
        if not self._continuation_manager:
            raise RuntimeError("Contitext adapter not initialized")

        try:
            # Create continuation for this execution
            continuation = self._continuation_manager.create_continuation(
                workflow_id=context.workflow_id,
                step_id=context.step_id,
                node_id=context.node_id,
                state=context.state_view.get_full_state(),
            )

            # Map input data through contitext mapper
            mapped_input = self._mapper.map_input(context.input_data, continuation)

            # Execute with continuation context
            result_data = self._execute_with_continuation(
                continuation, mapped_input, context
            )

            # Create commit if successful
            if result_data.get("success", False):
                commit = self._commit_manager.create_commit(
                    continuation_id=continuation.id,
                    changes=result_data.get("changes", {}),
                    metadata=result_data.get("metadata", {}),
                )
                result_data["commit_id"] = commit.id

            return ExecutionResult(
                success=result_data.get("success", False),
                node_id=context.node_id,
                step_id=context.step_id,
                output_data=result_data.get("output"),
                artifacts=result_data.get("artifacts", {}),
                changeset=result_data.get("changeset"),
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                node_id=context.node_id,
                step_id=context.step_id,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _execute_with_continuation(
        self, continuation: Any, input_data: Dict[str, Any], context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Execute the actual logic with continuation context

        This is a placeholder for the actual execution logic that would
        be implemented by specific nodes.
        """
        # For now, return a basic successful result
        return {
            "success": True,
            "output": input_data,
            "artifacts": {
                "continuation_id": continuation.id,
                "execution_time": context.step_id,
            },
            "changes": {},
            "metadata": {
                "adapter": "contitext",
                "node_type": context.node_config.node_type.value,
            },
        }

    def validate_context(self, context: ExecutionContext) -> bool:
        """Validate contitext-specific context"""
        if not super().validate_context(context):
            return False

        # Check for contitext-specific requirements
        if not hasattr(context, "state_view") or context.state_view is None:
            return False

        return True

    def get_continuation_state(self, continuation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get state for a specific continuation

        Args:
            continuation_id: ID of the continuation

        Returns:
            Continuation state if found, None otherwise
        """
        if not self._continuation_manager:
            return None

        continuation = self._continuation_manager.get_continuation(continuation_id)
        return continuation.state if continuation else None

    def create_commit(
        self, continuation_id: str, changes: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create a commit for changes in a continuation

        Args:
            continuation_id: ID of the continuation
            changes: Changes to commit

        Returns:
            Commit ID if successful, None otherwise
        """
        if not self._commit_manager:
            return None

        try:
            commit = self._commit_manager.create_commit(continuation_id, changes)
            return commit.id
        except Exception:
            return None
