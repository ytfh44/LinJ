"""
LinJ to ContiText Mapping Executor

Implements the mappings defined in sections 23-26:
- LinJ execution corresponds to main continuation
- step_id deterministic allocation
- changeset deterministic commit
- Resource domain constraint mapping
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Mapping

from ..core.document import LinJDocument
from ..core.nodes import Node
from ..core.changeset import ChangeSet
from ..core.state import StateManager
from ..core.errors import ExecutionError, ValidationError
from ..executor.runner import LinJExecutor, NodeExecutionResult
from ..executor.scheduler import (
    DeterministicScheduler,
    ExecutionState,
    are_dependencies_satisfied,
)
from .engine import ContiTextEngine
from .continuation import Continuation, Status
from .signal import Signal, WaitCondition

logger = logging.getLogger(__name__)


class LinJToContiTextMapper:
    """
    LinJ to ContiText Mapper

    Implements the mapping rules from sections 23-26, ensuring serial/parallel execution consistency
    """

    def __init__(
        self, document: LinJDocument, state_manager: Optional[StateManager] = None
    ):
        """
        Initialize the mapper

        Args:
            document: LinJ document
            state_manager: State manager
        """
        self.document = document
        self.state_manager = state_manager or StateManager()
        self.contitext_engine = ContiTextEngine(self.state_manager)
        self.scheduler = DeterministicScheduler(document.nodes)
        self.executor = LinJExecutor()

        # Mapping state
        self.execution_state = ExecutionState()
        self.current_round = 0

        logger.info(
            f"Initialized LinJ-ContiText mapper for document version {document.linj_version}"
        )

    async def execute(
        self, initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute LinJ document (using ContiText)

        Implements the core goal: same document + same initial state = consistent final main state

        Args:
            initial_state: Initial state

        Returns:
            Final state
        """
        # Initialize state
        if initial_state:
            for key, value in initial_state.items():
                self.state_manager.apply(ChangeSet.create_write(f"$.{key}", value))

        # Validate resource constraints (section 25)
        self._validate_resource_constraints()

        # Create main continuation (section 23)
        main_continuation = self.contitext_engine.derive()

        # Execute main loop
        await self._execution_loop(main_continuation)

        # Process all pending changesets
        self.contitext_engine.process_pending_changes()

        logger.info("LinJ-ContiText execution completed successfully")
        return self.state_manager.get_full_state()

    async def _execution_loop(self, main_cont: Continuation) -> None:
        """
        Main execution loop

        Executes LinJ nodes using ContiText continuations
        """
        max_rounds = (
            self.document.policies.max_rounds if self.document.policies else 1000
        )
        max_steps = self.document.policies.max_steps if self.document.policies else None
        executed_this_round: Set[str] = set()
        step_count = 0

        while True:
            # Get ready nodes (consider round and allow_reenter)
            ready_nodes = []
            for node in self.document.nodes:
                if self.execution_state.is_terminal(node.id):
                    continue
                graph = self.document.build_dependency_graph()
                if not are_dependencies_satisfied(node.id, graph, self.execution_state):
                    continue

                allow_reenter = node.policy.allow_reenter if node.policy else False
                if node.id in executed_this_round and not allow_reenter:
                    continue

                ready_nodes.append(node)

            if not ready_nodes:
                if executed_this_round:
                    self.current_round += 1
                    executed_this_round.clear()
                    if self.current_round > max_rounds:
                        raise ExecutionError(f"Exceeded max_rounds: {max_rounds}")
                    continue

                if not self._has_active_nodes():
                    break
                await asyncio.sleep(0.01)
                continue

            # Select node in deterministic order (section 11.3)
            node = self.scheduler.select_from_ready(ready_nodes, executed_this_round)
            if not node:
                break

            # Check step limit
            step_count += 1
            if max_steps and step_count > max_steps:
                raise ExecutionError(f"Exceeded max_steps: {max_steps}")

            # Allocate step_id (section 24.3)
            step_id = self.scheduler.allocate_step_id()

            # Create continuation view (section 18.2)
            view = self.contitext_engine.create_view(
                main_cont, self._get_pending_changesets()
            )

            # Execute node (may spawn child continuations)
            await self._execute_node_with_continuation(node, step_id, main_cont, view)

            executed_this_round.add(node.id)

    async def _execute_node_with_continuation(
        self, node: Node, step_id: int, parent_cont: Continuation, view: Any
    ) -> None:
        """
        Execute node with continuation

        Can choose to execute within main continuation, or spawn child continuations and merge
        """
        try:
            # Simplified: execute directly in main continuation
            # Actual implementation can spawn child continuations as needed
            result = await self._execute_node(node, view)

            if result.success:
                # Submit changeset (section 20.2)
                if result.changeset and not result.changeset.is_empty():
                    commit_result = self.contitext_engine.submit_changeset(
                        step_id=step_id,
                        changeset=result.changeset,
                        handle=parent_cont.handle,
                    )

                    if not commit_result.success:
                        raise ExecutionError(
                            f"Changeset commit failed: {commit_result.error}"
                        )

                self.scheduler.mark_completed(node.id)
                self.execution_state.completed.add(node.id)

                logger.debug(f"Node {node.id} executed successfully at step {step_id}")
            else:
                self.scheduler.mark_completed(node.id)
                self.execution_state.failed.add(node.id)
                if result.error:
                    raise result.error

        except Exception as e:
            self.scheduler.mark_completed(node.id)
            self.execution_state.failed.add(node.id)
            self.contitext_engine.fail(parent_cont.handle, str(e))
            raise ExecutionError(f"Node {node.id} execution failed: {e}")

    async def _execute_node(self, node: Node, view: Any) -> NodeExecutionResult:
        """
        Execute single node (using view)

        Args:
            node: Node definition
            view: Continuation view

        Returns:
            Execution result
        """
        # Convert view to state manager format (temporary)
        temp_state_manager = StateManager(view.get_full_state())

        return await self.executor._execute_node(
            node, temp_state_manager, self.document
        )

    def _get_ready_nodes(self) -> List[Node]:
        """Get ready nodes"""
        graph = self.document.build_dependency_graph()

        ready = []
        for node in self.document.nodes:
            if not self.execution_state.is_terminal(
                node.id
            ) and are_dependencies_satisfied(node.id, graph, self.execution_state):
                ready.append(node)

        return ready

    def _has_active_nodes(self) -> bool:
        """Check if there are active nodes"""
        # Simplified: check if there are any incomplete nodes
        for node in self.document.nodes:
            if not self.execution_state.is_terminal(node.id):
                return True
        return False

    def _get_pending_changesets(self) -> List[ChangeSet]:
        """Get pending changeset list"""
        # Get pending changesets from CommitManager
        pending = self.contitext_engine.get_commit_manager().get_pending()
        return [p.changeset for p in pending]

    def _validate_resource_constraints(self) -> None:
        """
        Validate resource domain constraints (section 25)

        Verify that placement declarations and kind=resource dependencies are satisfiable
        """
        from ..core.document import validate_resource_constraints

        errors = validate_resource_constraints(self.document)
        if errors:
            error_messages = [str(e) for e in errors]
            raise ValidationError(
                f"Resource constraints validation failed: {error_messages}"
            )


class ParallelLinJExecutor(LinJToContiTextMapper):
    """
    Parallel LinJ Executor

    Implements true parallel execution based on continuation mechanism
    """

    async def execute_parallel(
        self, initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute LinJ document in parallel

        Implements parallel execution using continuations and concurrency safety guarantees while ensuring consistency

        Args:
            initial_state: Initial state

        Returns:
            Final state
        """
        # Initialize state
        if initial_state:
            for key, value in initial_state.items():
                self.state_manager.apply(ChangeSet.create_write(f"$.{key}", value))

        # Validate resource constraints
        self._validate_resource_constraints()

        # Create main continuation
        main_continuation = self.contitext_engine.derive()

        # Parallel execution loop
        from ..executor.scheduler import DomainAllocator

        domain_allocator = DomainAllocator()
        domain_map = domain_allocator.allocate_domains(self.document)

        await self._parallel_execution_loop(main_continuation, domain_map)

        # Process all pending changesets
        self.contitext_engine.process_pending_changes()

        logger.info("Parallel LinJ-ContiText execution completed successfully")
        return self.state_manager.get_full_state()

    async def _parallel_execution_loop(
        self, main_cont: Continuation, domain_map: Mapping[str, Any]
    ) -> None:
        """
        Parallel execution loop

        1. Identify node groups that can safely execute concurrently (considering data conflicts and domain constraints)
        2. Derive continuations for each group and execute in parallel
        3. Merge and continue to next group
        """
        from ..executor.scheduler import find_concurrent_groups

        max_rounds = (
            self.document.policies.max_rounds if self.document.policies else 1000
        )
        self.current_round = 0

        while True:
            # Get ready nodes
            ready_nodes = self._get_ready_nodes()

            if not ready_nodes:
                if not self._has_active_nodes():
                    break
                await asyncio.sleep(0.01)
                continue

            # Group into safely concurrent groups (section 11.5 & 25)
            concurrent_groups = find_concurrent_groups(ready_nodes, domain_map)

            # Execute each group in parallel
            for group in concurrent_groups:
                await self._execute_concurrent_group(group, main_cont)

            # Round counting
            self.current_round += 1
            if self.current_round > max_rounds:
                raise ExecutionError(
                    f"Exceeded max_rounds in parallel execution: {max_rounds}"
                )

    async def _execute_concurrent_group(
        self, nodes: List[Node], parent_cont: Continuation
    ) -> None:
        """
        Execute a group of nodes in parallel

        Args:
            nodes: Safely concurrent node group
            parent_cont: Parent continuation
        """
        # Spawn child continuation for each node
        child_continuations = []
        tasks = []

        for node in nodes:
            # Allocate step_id (deterministic order)
            step_id = self.scheduler.allocate_step_id()

            # Spawn child continuation
            child_cont = self.contitext_engine.derive(parent_cont)
            child_continuations.append((node, child_cont, step_id))

            # Create view and start task
            view = self.contitext_engine.create_view(
                child_cont, self._get_pending_changesets()
            )
            task = asyncio.create_task(
                self._execute_node_with_continuation(node, step_id, child_cont, view)
            )
            tasks.append(task)

        # Wait for all tasks to complete (merge)
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Concurrent group execution failed: {e}")
            raise

        # Mark all nodes as completed
        for node, child_cont, step_id in child_continuations:
            if not self.execution_state.is_terminal(node.id):
                # Update execution state based on continuation status
                if child_cont.status == Status.COMPLETED:
                    self.scheduler.mark_completed(node.id)
                    self.execution_state.completed.add(node.id)
                else:
                    self.scheduler.mark_completed(node.id)
                    self.execution_state.failed.add(node.id)
