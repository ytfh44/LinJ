"""
LinJ Workflow Unified Executor

Provides execution interface fully consistent with AutoGen and LangGraph versions
Ensures same inputs produce same outputs (meets LinJ specification baseline goal)
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from ..core.document import LinJDocument
from ..core.state import StateManager
from ..core.changeset import ChangeSet
from ..exceptions.errors import ValidationError, ResourceConstraintUnsatisfied

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Execution configuration"""

    backend: str = "autogen"  # "autogen" or "langgraph"
    enable_parallel: bool = False
    max_concurrency: int = 4
    max_steps: Optional[int] = None
    max_rounds: Optional[int] = None
    timeout_ms: Optional[int] = None
    trace_enabled: bool = False


@dataclass
class ExecutionResult:
    """Execution result"""

    success: bool
    final_state: Dict[str, Any]
    execution_stats: Dict[str, Any]
    error: Optional[str] = None
    trace: Optional[List[Dict[str, Any]]] = None


class LinJExecutor:
    """
    LinJ workflow executor

    Unifies execution behavior across AutoGen and LangGraph versions:
    - Same document parsing
    - Same state management
    - Same scheduling logic
    - Same error handling
    - Same tracing records
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self._setup_backend()

    def _setup_backend(self) -> None:
        """Set up backend executor"""
        if self.config.backend == "langgraph":
            try:
                # Use shared executor adapter for LangGraph
                from .langgraph_adapter import LangGraphExecutorAdapter

                self._adapter = LangGraphExecutorAdapter(
                    {
                        "max_concurrency": self.config.max_concurrency,
                        "trace_enabled": self.config.trace_enabled,
                    }
                )
                logger.info("Using LangGraph backend")
            except ImportError as e:
                logger.error(f"Failed to import LangGraph backend: {e}")
                raise
        elif self.config.backend == "autogen":
            try:
                # Use shared executor adapter for AutoGen
                from .autogen_adapter import AutoGenExecutorAdapter

                self._adapter = AutoGenExecutorAdapter(
                    {
                        "max_concurrency": self.config.max_concurrency,
                        "trace_enabled": self.config.trace_enabled,
                    }
                )
                logger.info("Using AutoGen backend")
            except ImportError as e:
                logger.error(f"Failed to import AutoGen backend: {e}")
                raise
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def execute(
        self,
        document: Union[Dict[str, Any], LinJDocument],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute LinJ workflow

        Args:
            document: LinJ document (dict or LinJDocument object)
            initial_state: Initial state

        Returns:
            Execution result

        Raises:
            ValidationError: Document validation failed
            ResourceConstraintUnsatisfied: Resource constraint not satisfied
        """
        logger.info(f"Executing LinJ workflow with {self.config.backend} backend")

        try:
            # Parse document
            if isinstance(document, dict):
                try:
                    doc = LinJDocument.model_validate(document)
                except Exception as e:
                    raise ValidationError(f"Invalid LinJ document: {e}")
            else:
                doc = document

            # Validate document
            self._validate_document(doc)

            # Apply config overrides
            doc = self._apply_config_overrides(doc)

            # Execute workflow
            backend_result = self._adapter.execute_workflow(doc, initial_state)

            return ExecutionResult(
                success=True,
                final_state=backend_result["final_state"],
                execution_stats=backend_result["execution_stats"],
                trace=backend_result.get("trace"),
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return ExecutionResult(
                success=False,
                final_state=initial_state or {},
                execution_stats={},
                error=str(e),
            )

    def _validate_document(self, doc: LinJDocument) -> None:
        """Validate LinJ document"""
        # Version validation
        major_version = doc.get_major_version()
        if major_version != 0:  # assuming we're implementing 0.x version
            raise ValidationError(f"Unsupported major version: {major_version}")

        # Node ID uniqueness check
        node_ids = [node.id for node in doc.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValidationError("Duplicate node IDs found")

        # Dependency validity check
        node_id_set = set(node_ids)
        for edge in doc.edges:
            if edge.from_ not in node_id_set:
                raise ValidationError(f"Unknown source node: {edge.from_}")
            if edge.to not in node_id_set:
                raise ValidationError(f"Unknown target node: {edge.to}")

        # Policy validation
        if (
            doc.requirements
            and doc.requirements.allow_parallel
            and not self.config.enable_parallel
        ):
            logger.warning("Document requires parallel but config disables it")

        # Loop validation
        if doc.loops:
            for loop in doc.loops:
                if (
                    loop.mode == "finite"
                    and not loop.stop_condition
                    and not loop.max_rounds
                ):
                    raise ValidationError(
                        f"Finite loop {loop.id} missing stop condition or max_rounds"
                    )

    def _apply_config_overrides(self, doc: LinJDocument) -> LinJDocument:
        """Apply config overrides to document"""
        # Create document copy to avoid modifying original
        doc_dict = doc.model_dump()

        # Override policies
        if not doc_dict.get("policies"):
            doc_dict["policies"] = {}

        policies = doc_dict["policies"]
        if self.config.max_steps:
            policies["max_steps"] = self.config.max_steps
        if self.config.max_rounds:
            policies["max_rounds"] = self.config.max_rounds
        if self.config.timeout_ms:
            policies["timeout_ms"] = self.config.timeout_ms

        # Override requirements
        if not doc_dict.get("requirements"):
            doc_dict["requirements"] = {}

        requirements = doc_dict["requirements"]
        requirements["allow_parallel"] = self.config.enable_parallel

        # Reconstruct document object
        return LinJDocument.model_validate(doc_dict)

    def validate_consistency(
        self,
        document: Union[Dict[str, Any], LinJDocument],
        initial_state: Optional[Dict[str, Any]] = None,
        iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Validate execution consistency between backends

        Args:
            document: LinJ document
            initial_state: Initial state
            iterations: Number of test iterations

        Returns:
            Consistency validation result
        """
        logger.info("Validating execution consistency between backends")

        results = {}

        for backend in ["autogen", "langgraph"]:
            backend_results = []

            for i in range(iterations):
                # Create backend-specific config
                config = ExecutionConfig(
                    backend=backend,
                    enable_parallel=self.config.enable_parallel,
                    max_concurrency=self.config.max_concurrency,
                    trace_enabled=True,  # Enable tracing for detailed information
                )

                executor = LinJExecutor(config)
                result = executor.execute(document, initial_state)
                backend_results.append(result)

            results[backend] = backend_results

        # Analyze consistency
        autogen_states = [r.final_state for r in results["autogen"] if r.success]
        langgraph_states = [r.final_state for r in results["langgraph"] if r.success]

        consistency_analysis = self._analyze_state_consistency(
            autogen_states, langgraph_states
        )

        return {
            "consistent": consistency_analysis["all_identical"],
            "analysis": consistency_analysis,
            "backend_results": results,
            "document_version": (
                getattr(document, "linj_version", "unknown")
                if hasattr(document, "linj_version")
                else document.get("linj_version", "unknown")
            ),
        }

    def _analyze_state_consistency(
        self,
        autogen_states: List[Dict[str, Any]],
        langgraph_states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze state consistency"""
        if not autogen_states or not langgraph_states:
            return {"error": "No successful executions to compare"}

        # Check if all executions produce the same state
        autogen_consistent = all(state == autogen_states[0] for state in autogen_states)
        langgraph_consistent = all(
            state == langgraph_states[0] for state in langgraph_states
        )

        cross_backend_consistent = autogen_states[0] == langgraph_states[0]
        all_identical = (
            autogen_consistent and langgraph_consistent and cross_backend_consistent
        )

        return {
            "all_identical": all_identical,
            "autogen_consistent": autogen_consistent,
            "langgraph_consistent": langgraph_consistent,
            "cross_backend_consistent": cross_backend_consistent,
            "autogen_sample": autogen_states[0] if autogen_states else None,
            "langgraph_sample": langgraph_states[0] if langgraph_states else None,
        }


# Convenience functions
def execute_linj(
    document: Union[Dict[str, Any], LinJDocument],
    backend: str = "autogen",
    initial_state: Optional[Dict[str, Any]] = None,
    **config_kwargs,
) -> ExecutionResult:
    """
    Convenience function: Execute LinJ workflow

    Args:
        document: LinJ document
        backend: Backend choice ("autogen" or "langgraph")
        initial_state: Initial state
        **config_kwargs: Extra config parameters

    Returns:
        Execution result
    """
    config = ExecutionConfig(backend=backend, **config_kwargs)
    executor = LinJExecutor(config)
    return executor.execute(document, initial_state)


def validate_consistency(
    document: Union[Dict[str, Any], LinJDocument],
    initial_state: Optional[Dict[str, Any]] = None,
    iterations: int = 3,
    **config_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function: Validate execution consistency

    Args:
        document: LinJ document
        initial_state: Initial state
        iterations: Number of test iterations
        **config_kwargs: Config parameters

    Returns:
        Consistency validation result
    """
    config = ExecutionConfig(**config_kwargs)
    executor = LinJExecutor(config)
    return executor.validate_consistency(document, initial_state, iterations=iterations)
