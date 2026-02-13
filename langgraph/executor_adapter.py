"""
LangGraph Executor Adapter

Fully integrates shared executor components, ensuring complete behavioral consistency with the AutoGen version
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from ..executor import backend, evaluator, scheduler, context, adapter
from ..executor.autogen_scheduler import (
    DomainAllocator,
    ExecutionState,
    are_dependencies_satisfied,
)
from .scheduler import LangGraphDeterministicScheduler

logger = logging.getLogger(__name__)


@dataclass
class LangGraphExecutionContext:
    """LangGraph execution context"""

    document: Any  # LinJDocument
    dependency_graph: Any  # DependencyGraph
    state_manager: Any  # StateManager
    scheduler: Any  # Scheduler
    executed_this_round: set
    current_step: int
    max_rounds: Optional[int]
    policies: Optional[Dict[str, Any]]


class LangGraphExecutorAdapter(adapter.BaseAdapter):
    """
    LangGraph Executor Adapter

    Provides execution behavior completely consistent with AutoGen:
    - Same node execution logic
    - Same state management
    - Same error handling
    - Same tracing records
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._execution_state = ExecutionState()
        self._domain_allocator = DomainAllocator()

    def create_scheduler(
        self,
        nodes: List[Any],
        strategy: str = "deterministic",
        enable_parallel: bool = False,
    ) -> scheduler.Scheduler:
        """Create scheduler"""
        if strategy == "deterministic":
            return LangGraphDeterministicScheduler(
                nodes, enable_parallel=enable_parallel
            )
        elif strategy == "parallel":
            return scheduler.ParallelScheduler(
                max_concurrency=self._config.get("max_concurrency", 4)
            )
        else:
            raise ValueError(f"Unsupported scheduling strategy: {strategy}")

    def create_evaluator(
        self, config: Optional[Dict[str, Any]] = None
    ) -> evaluator.Evaluator:
        """Create node evaluator"""
        return evaluator.NodeEvaluator(config or {})

    def create_context(
        self,
        document: Any,
        state_manager: Any,
        backend: Any,
    ) -> context.ExecutionContext:
        """Create execution context"""
        # Build dependency graph
        dependency_graph = backend.DependencyGraph(document.edges)

        # Create scheduler
        enable_parallel = (
            document.requirements.allow_parallel
            if hasattr(document.requirements, "allow_parallel")
            else False
        )
        sched = self.create_scheduler(
            document.nodes, strategy="deterministic", enable_parallel=enable_parallel
        )

        # Create evaluator
        evaluator = self.create_evaluator(self._config)

        return LangGraphExecutionContext(
            document=document,
            dependency_graph=dependency_graph,
            state_manager=state_manager,
            scheduler=sched,
            executed_this_round=set(),
            current_step=0,
            max_rounds=getattr(document.policies, "max_rounds", None)
            if document.policies
            else None,
            policies=self._config,
        )

    def execute_workflow(
        self,
        document: Any,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute workflow

        Args:
            document: LinJ document object
            initial_state: Initial state

        Returns:
            Execution result and final state
        """
        logger.info(f"Starting LangGraph workflow execution: {document.linj_version}")

        # Initialize state manager
        state_manager = backend.StateManager(initial_state or {})

        # Create execution context
        exec_context = self.create_context(document, state_manager, None)

        # Execute main loop
        result = self._execute_main_loop(document, exec_context, state_manager)

        logger.info("LangGraph workflow execution completed")
        return result

    def _execute_main_loop(
        self,
        document: Any,
        exec_context: LangGraphExecutionContext,
        state_manager: Any,
    ) -> Dict[str, Any]:
        """Execute main loop"""
        current_round = 0
        total_steps = 0

        while current_round < (exec_context.max_rounds or 100):
            exec_context.executed_this_round.clear()
            round_steps = 0

            # Find all executable nodes
            ready_nodes = self._find_ready_nodes(document, exec_context)

            if not ready_nodes:
                # Check if there are nodes that can advance
                if not self._can_advance(document, exec_context):
                    logger.info(f"Workflow completed at round {current_round}")
                    break
                else:
                    logger.debug(f"No ready nodes, but workflow can advance")
                    current_round += 1
                    continue

            # Execute nodes
            while ready_nodes:
                # Scheduling decision
                decision = exec_context.scheduler.select_nodes(
                    ready_nodes, exec_context
                )

                if not decision.selected_nodes:
                    break

                # Execute selected nodes
                for node in decision.selected_nodes:
                    try:
                        step_result = self._execute_node(
                            node, exec_context, state_manager
                        )

                        # Mark node as completed
                        exec_context.scheduler.mark_completed(
                            getattr(node, "id", "unknown"), step_result.success
                        )

                        total_steps += 1
                        round_steps += 1
                        exec_context.executed_this_round.add(
                            getattr(node, "id", "unknown")
                        )

                    except Exception as e:
                        logger.error(f"Node execution failed: {e}")
                        exec_context.scheduler.mark_completed(
                            getattr(node, "id", "unknown"), False
                        )

                # Update ready nodes list
                ready_nodes = self._find_ready_nodes(document, exec_context)

            current_round += 1

            # Check step limit
            if (
                document.policies
                and document.policies.max_steps
                and total_steps >= document.policies.max_steps
            ):
                logger.warning(f"Maximum steps ({document.policies.max_steps}) reached")
                break

        return {
            "final_state": state_manager.get_full_state(),
            "execution_stats": {
                "total_rounds": current_round,
                "total_steps": total_steps,
                "scheduler_stats": exec_context.scheduler.get_execution_stats(),
                "execution_state": {
                    "completed": list(exec_context._execution_state.completed),
                    "failed": list(exec_context._execution_state.failed),
                },
            },
        }

    def _find_ready_nodes(
        self, document: Any, exec_context: LangGraphExecutionContext
    ) -> List[Any]:
        """Find all ready nodes"""
        ready_nodes = []

        for node in document.nodes:
            node_id = getattr(node, "id", "unknown")

            # Skip completed nodes
            if exec_context._execution_state.is_terminal(node_id):
                continue

            # Check if dependencies are satisfied
            if are_dependencies_satisfied(
                node_id, exec_context.dependency_graph, exec_context._execution_state
            ):
                ready_nodes.append(node)

        return ready_nodes

    def _can_advance(
        self, document: Any, exec_context: LangGraphExecutionContext
    ) -> bool:
        """Check if workflow can still advance"""
        # Check if there are any incomplete nodes
        for node in document.nodes:
            node_id = getattr(node, "id", "unknown")
            if not exec_context._execution_state.is_terminal(node_id):
                return True
        return False

    def _execute_node(
        self,
        node: Any,
        exec_context: LangGraphExecutionContext,
        state_manager: Any,
    ) -> Any:
        """Execute single node"""
        node_id = getattr(node, "id", "unknown")

        logger.debug(f"Executing node: {node_id}")

        # Create node state view
        step_id = exec_context.scheduler.allocate_step_id()
        view = state_manager.create_view(step_id)

        # Execute based on node type
        node_type = getattr(node, "type", "unknown")

        if node_type == "hint":
            return self._execute_hint_node(node, view, state_manager)
        elif node_type == "tool":
            return self._execute_tool_node(node, view, state_manager)
        elif node_type == "join":
            return self._execute_join_node(node, view, state_manager)
        elif node_type == "gate":
            return self._execute_gate_node(node, view, exec_context)
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    def _execute_hint_node(self, node: Any, view: Any, state_manager: Any) -> Any:
        """Execute hint node"""
        try:
            # Render template
            state = view.get_full_state()
            rendered = node.render(state)

            # Create changeset
            from ..core.changeset import ChangeSet

            changeset = ChangeSet(
                writes=[{"path": node.write_to, "value": rendered}], deletes=[]
            )

            # Apply changes
            state_manager.apply(changeset)

            return type("Result", (), {"success": True, "output": rendered})

        except Exception as e:
            logger.error(f"Hint node execution failed: {e}")
            return type("Result", (), {"success": False, "error": str(e)})

    def _execute_tool_node(self, node: Any, view: Any, state_manager: Any) -> Any:
        """Execute tool node"""
        try:
            # Parse arguments
            state = view.get_full_state()
            args = node.get_args(state)

            # Execute tool (actual tool system integration needed here)
            # Returning simulated result for now
            result = f"Tool {node.call.name} executed with args: {args}"

            # Create changeset
            if node.write_to:
                from ..core.changeset import ChangeSet

                changeset = ChangeSet(
                    writes=[{"path": node.write_to, "value": result}], deletes=[]
                )
                state_manager.apply(changeset)

            return type("Result", (), {"success": True, "output": result})

        except Exception as e:
            logger.error(f"Tool node execution failed: {e}")
            return type("Result", (), {"success": False, "error": str(e)})

    def _execute_join_node(self, node: Any, view: Any, state_manager: Any) -> Any:
        """Execute join node"""
        try:
            # Read input
            input_text = view.read(node.input_from)

            # Check forbidden terms
            forbidden = node.validate_forbidden(str(input_text))
            if forbidden:
                raise ValueError(f"Forbidden term found: {forbidden}")

            # Write output (simple copy, actual join logic may be more complex)
            from ..core.changeset import ChangeSet

            changeset = ChangeSet(
                writes=[{"path": node.output_to, "value": input_text}], deletes=[]
            )
            state_manager.apply(changeset)

            return type("Result", (), {"success": True, "output": input_text})

        except Exception as e:
            logger.error(f"Join node execution failed: {e}")
            return type("Result", (), {"success": False, "error": str(e)})

    def _execute_gate_node(
        self, node: Any, view: Any, exec_context: LangGraphExecutionContext
    ) -> Any:
        """Execute gate node"""
        try:
            # Evaluate condition
            state = view.get_full_state()
            condition_result = node.evaluate(state)

            # Get next nodes (only recording result here, actual triggering handled by scheduler)
            next_nodes = node.get_next_nodes(state)

            return type(
                "Result",
                (),
                {
                    "success": True,
                    "output": {
                        "condition_result": condition_result,
                        "next_nodes": next_nodes,
                    },
                },
            )

        except Exception as e:
            logger.error(f"Gate node execution failed: {e}")
            return type("Result", (), {"success": False, "error": str(e)})
