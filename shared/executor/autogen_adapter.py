"""
AutoGen Executor Adapter

Provides a unified executor interface compatible interface, using shared components to execute LinJ workflows.
To ensure consistency with the LangGraph backend (meeting the baseline goal of LinJ specifications),
this adapter reuses core components from shared/executor.
"""

import logging
from typing import Any, Dict, Optional, Union

from ..core.document import LinJDocument, Policies
from ..core.state import StateManager
from .context import ExecutionContext
from .backend import BaseExecutionBackend
from .scheduler import DeterministicScheduler
from .evaluator import BaseEvaluator
from .autogen_scheduler import AutoGenDeterministicScheduler
from .autogen_evaluator import AutoGenConditionEvaluator

logger = logging.getLogger(__name__)


class ConcreteAutoGenEvaluator(AutoGenConditionEvaluator):
    """Concrete implementation of AutoGenConditionEvaluator"""

    def _evaluate_node(self, node: Any, context: ExecutionContext) -> Any:
        # Minimal implementation to satisfy abstract base class
        return node

    def tokenize(self, expression: str) -> list:
        return []

    def parse(self, tokens: list) -> Any:
        return None


class AutoGenExecutorAdapter:
    """
    AutoGen Executor Adapter

    Implements a unified execute_workflow interface using AutoGen-compatible scheduling and evaluation logic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.trace_enabled = self.config.get("trace_enabled", False)
        self.max_concurrency = self.config.get("max_concurrency", 4)

    def execute_workflow(
        self,
        document: Union[Dict[str, Any], LinJDocument],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute workflow

        Args:
            document: LinJ document
            initial_state: Initial state

        Returns:
            Execution result dictionary containing final_state, execution_stats, etc.
        """
        # 1. Prepare document object
        if isinstance(document, dict):
            doc = LinJDocument.model_validate(document)
        else:
            doc = document

        # 2. Initialize state manager
        state_manager = StateManager(initial_state or {})

        # 3. Create execution context
        context = ExecutionContext()
        # Inject document and state manager into context (complying with shared.executor.types.ExecutionContext)
        context.document = doc
        context.state_manager = state_manager

        # 4. Create backend components (using AutoGen-compatible versions)
        # Use AutoGen-compatible condition evaluator
        evaluator = ConcreteAutoGenEvaluator(state_manager.get_full_state())

        # Use AutoGen-compatible scheduler
        scheduler = AutoGenDeterministicScheduler(doc.nodes)  # type: ignore

        # 5. Main execution loop
        # Use unified execution logic
        from .runner_utils import execute_nodes_generic

        # Prepare backend-specific node executor (if needed, currently using generic executor)
        # Can be customized for specific backend needs node_executor_fn

        final_state, stats = execute_nodes_generic(
            doc,
            state_manager,
            scheduler,
            evaluator,
            max_steps=(
                doc.policies.max_steps
                if doc.policies and doc.policies.max_steps
                else 100
            ),
        )

        return {
            "success": True,
            "final_state": final_state,
            "execution_stats": stats,
            "trace": [] if not self.trace_enabled else [{"type": "trace_placeholder"}],
        }
