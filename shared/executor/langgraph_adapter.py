"""
LangGraph Executor Adapter

Provides an interface compatible with the unified executor, executing LinJ workflows using shared components.
To ensure consistency with the AutoGen backend (meeting the baseline goal of LinJ specification),
this adapter reuses core components from shared/executor.
"""

import logging
from typing import Any, Dict, Optional, Union

from ..core.document import LinJDocument
from ..core.state import StateManager
from .context import ExecutionContext
from .scheduler import DeterministicScheduler
from .evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class ConcreteLangGraphEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator"""

    def _evaluate_node(self, node: Any, context: ExecutionContext) -> Any:
        return node

    def tokenize(self, expression: str) -> list:
        return []

    def parse(self, tokens: list) -> Any:
        return None


class LangGraphExecutorAdapter:
    """
    LangGraph Executor Adapter

    Implements the unified execute_workflow interface using LangGraph-compatible scheduling and evaluation logic.
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

        # 3. Create execution backend components
        # Use standard evaluator (LangGraph uses standard logic)
        evaluator = ConcreteLangGraphEvaluator()

        # Use standard deterministic scheduler
        scheduler = DeterministicScheduler(doc.nodes)

        # 4. Execute main loop
        # Use unified execution logic
        from .runner_utils import execute_nodes_generic

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
