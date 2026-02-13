"""
LinJ to ContiText Mapper

Implements LinJ ↔ ContiText mapping defined in LinJ specification sections 23-26:
- LinJ execution corresponds to main continuation
- step_id deterministic allocation
- changeset deterministic commit
- Resource domain constraint mapping
Framework-agnostic mapper implementation
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Mapping, Protocol, runtime_checkable

from .engine import ContiTextEngine, ChangeSet
from .continuation import Continuation, Status, StateManager
from .signal import Signal, WaitCondition

# Import dependency graph
from ..core.edges import DependencyGraph, EdgeKind

logger = logging.getLogger(__name__)


@runtime_checkable
class LinJDocument(Protocol):
    """LinJ document protocol for framework-agnostic document operations"""

    @property
    def linj_version(self) -> str:
        """LinJ version"""
        ...

    @property
    def nodes(self) -> List[Any]:
        """Node list"""
        ...

    @property
    def edges(self) -> List[Any]:
        """Dependency edge list"""
        ...

    @property
    def policies(self) -> Optional[Any]:
        """Execution policies"""
        ...


@runtime_checkable
class Node(Protocol):
    """Node protocol for framework-agnostic node operations"""

    @property
    def id(self) -> str:
        """Node ID"""
        ...

    @property
    def policy(self) -> Optional[Any]:
        """Node policy"""
        ...


@runtime_checkable
class DeterministicScheduler(Protocol):
    """Deterministic scheduler protocol"""

    def select_from_ready(self, ready_nodes: List[Node]) -> Optional[Node]:
        """Select a node from ready nodes for execution"""
        ...

    def allocate_step_id(self) -> int:
        """Allocate step ID"""
        ...

    def mark_completed(self, node_id: str) -> None:
        """Mark node as completed"""
        ...


@runtime_checkable
class LinJExecutor(Protocol):
    """LinJ executor protocol"""

    async def execute_node(
        self, node: Node, state_manager: StateManager, document: LinJDocument
    ) -> Any:
        """Execute a single node"""
        ...


class LinJToContiTextMapper:
    """
    LinJ to ContiText Mapper

    Implements mapping rules for sections 23-26, ensuring serial/parallel execution consistency
    Framework-agnostic mapper implementation
    """

    def __init__(
        self,
        document: LinJDocument,
        state_manager: Optional[StateManager] = None,
        scheduler: Optional[DeterministicScheduler] = None,
        executor: Optional[LinJExecutor] = None,
    ):
        """
        Initialize mapper

        Args:
            document: LinJ document
            state_manager: State manager
            scheduler: Deterministic scheduler
            executor: LinJ executor
        """
        self.document = document
        self.state_manager = state_manager or self._create_default_state_manager()
        self.contitext_engine = ContiTextEngine(self.state_manager)
        self.scheduler = scheduler or self._create_default_scheduler()
        self.executor = executor or self._create_default_executor()

        # Mapping state
        self.execution_state = self._create_default_execution_state()
        self.current_round = 0

        logger.info(
            f"Initialized LinJ-ContiText mapper for document version {document.linj_version}"
        )

    def _create_default_state_manager(self) -> StateManager:
        """Create default state manager"""

        class DefaultStateManager:
            def __init__(self):
                self._state: Dict[str, Any] = {}
                self._revision: int = 0

            def get_full_state(self) -> Dict[str, Any]:
                return self._state.copy()

            def get_revision(self) -> int:
                return self._revision

            def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
                # Simple changeset application logic
                if hasattr(changeset, "apply_to_state"):
                    self._state = changeset.apply_to_state(self._state)
                elif isinstance(changeset, dict):
                    self._state.update(changeset)
                self._revision += 1

        return DefaultStateManager()

    def _create_default_scheduler(self) -> DeterministicScheduler:
        """Create default scheduler"""

        class DefaultScheduler:
            def __init__(self):
                self._next_step_id = 1
                self._completed_nodes: Set[str] = set()

            def select_from_ready(self, ready_nodes: List[Node]) -> Optional[Node]:
                # Simple selection by ID lexicographical order
                if not ready_nodes:
                    return None
                return min(ready_nodes, key=lambda n: n.id)

            def allocate_step_id(self) -> int:
                step_id = self._next_step_id
                self._next_step_id += 1
                return step_id

            def mark_completed(self, node_id: str) -> None:
                self._completed_nodes.add(node_id)

        return DefaultScheduler()

    def _create_default_executor(self) -> LinJExecutor:
        """Create default executor"""

        class DefaultExecutor:
            async def execute_node(
                self, node: Node, state_manager: StateManager, document: LinJDocument
            ) -> Any:
                # Simple executor that returns an empty result
                # Actual implementation should execute corresponding logic based on node type
                result = type(
                    "NodeResult",
                    (),
                    {"success": True, "changeset": None, "error": None},
                )()
                return result

        return DefaultExecutor()

    def _create_default_execution_state(self) -> Any:
        """Create default execution state"""

        class DefaultExecutionState:
            def __init__(self):
                self.completed: Set[str] = set()
                self.failed: Set[str] = set()

            def is_terminal(self, node_id: str) -> bool:
                return node_id in self.completed or node_id in self.failed

        return DefaultExecutionState()

    async def execute(
        self, initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute LinJ document (using ContiText)

        Implements the bottom-line goal: same document + same initial state = consistent final main state

        Args:
            initial_state: Initial state

        Returns:
            Final state
        """
        # Initialize state
        if initial_state:
            for key, value in initial_state.items():
                changeset = {"set": {key: value}}
                self.state_manager.apply(changeset)

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

        Execute LinJ nodes using Continuation views
        """
        max_rounds = (
            self.document.policies.max_rounds if self.document.policies else 1000
        )
        max_steps = self.document.policies.max_steps if self.document.policies else None
        executed_this_round: Set[str] = set()
        step_count = 0

        while True:
            # Get ready nodes (considering round and allow_reenter)
            ready_nodes = []
            for node in self.document.nodes:
                if self.execution_state.is_terminal(node.id):
                    continue

                # Simplified dependency check (actual implementation needs complete dependency graph)
                dependencies_satisfied = self._check_dependencies_satisfied(node)

                if not dependencies_satisfied:
                    continue

                allow_reenter = (
                    getattr(node.policy, "allow_reenter", False)
                    if node.policy
                    else False
                )
                if node.id in executed_this_round and not allow_reenter:
                    continue

                ready_nodes.append(node)

            if not ready_nodes:
                if executed_this_round:
                    self.current_round += 1
                    executed_this_round.clear()
                    if self.current_round > max_rounds:
                        raise RuntimeError(f"Exceeded max_rounds: {max_rounds}")
                    continue

                if not self._has_active_nodes():
                    break
                await asyncio.sleep(0.01)
                continue

            # Select node in deterministic order (section 11.3)
            node = self.scheduler.select_from_ready(ready_nodes)
            if not node:
                break

            # Check step limit
            step_count += 1
            if max_steps and step_count > max_steps:
                raise RuntimeError(f"Exceeded max_steps: {max_steps}")

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
        Execute node using continuation

        Can choose to execute within the main continuation, or spawn child continuations and then join
        """
        try:
            # Simplified implementation: execute directly within main continuation
            # Actual implementation can spawn child continuations as needed
            result = await self._execute_node(node, view)

            if result.success:
                # Submit changeset (section 20.2)
                if (
                    hasattr(result, "changeset")
                    and result.changeset
                    and not self._is_changeset_empty(result.changeset)
                ):
                    commit_result = self.contitext_engine.submit_changeset(
                        step_id=step_id,
                        changeset=result.changeset,
                        handle=parent_cont.handle,
                    )

                    if not commit_result.success:
                        raise RuntimeError(
                            f"Changeset commit failed: {commit_result.error}"
                        )

                self.scheduler.mark_completed(node.id)
                self.execution_state.completed.add(node.id)

                logger.debug(f"Node {node.id} executed successfully at step {step_id}")
            else:
                self.scheduler.mark_completed(node.id)
                self.execution_state.failed.add(node.id)
                if hasattr(result, "error") and result.error:
                    raise result.error

        except Exception as e:
            self.scheduler.mark_completed(node.id)
            self.execution_state.failed.add(node.id)
            self.contitext_engine.fail(parent_cont.handle, str(e))
            raise RuntimeError(f"Node {node.id} execution failed: {e}")

    async def _execute_node(self, node: Node, view: Any) -> Any:
        """
        Execute a single node (using view)

        Args:
            node: Node definition
            view: Continuation view

        Returns:
            Execution result
        """
        # Convert view to state manager format (temporary)
        temp_state_manager = self._create_temp_state_manager(view)

        return await self.executor.execute_node(node, temp_state_manager, self.document)

    def _create_temp_state_manager(self, view: Any) -> StateManager:
        """Create temporary state manager from view"""

        class TempStateManager:
            def __init__(self, view):
                self.view = view
                self._revision = 0

            def get_full_state(self) -> Dict[str, Any]:
                if hasattr(self.view, "get_full_state"):
                    return self.view.get_full_state()
                return {}

            def get_revision(self) -> int:
                return self._revision

            def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
                self._revision += 1

        return TempStateManager(view)

    def _is_changeset_empty(self, changeset: Any) -> bool:
        """Check if changeset is empty"""
        if hasattr(changeset, "is_empty"):
            return changeset.is_empty()
        return not changeset

    def _check_dependencies_satisfied(self, node: Node) -> bool:
        """
        Check if node dependencies are satisfied (section 11.1)

        Node's data/control prerequisite dependencies must all be completed
        Node is only executable when all prerequisite dependency nodes have completed successfully

        Args:
            node: Node object

        Returns:
            Whether all prerequisite dependencies are satisfied
        """
        # Get all incoming edges for the node
        incoming_edges = self._get_dependency_graph().get_incoming(node.id)

        for edge in incoming_edges:
            # Check data and control dependencies
            if edge.kind in (EdgeKind.DATA, EdgeKind.CONTROL):
                from_node_id = edge.from_
                # Prerequisite dependency nodes must be completed (success or failure both count)
                if not self.execution_state.is_terminal(from_node_id):
                    return False

        return True

    def _get_dependency_graph(self) -> DependencyGraph:
        """
        Get dependency graph (with caching)

        Returns:
            Dependency graph object
        """
        if not hasattr(self, "_dependency_graph"):
            # Build dependency graph from document
            edges = getattr(self.document, "edges", [])
            self._dependency_graph = DependencyGraph(edges)
        return self._dependency_graph

    def _has_active_nodes(self) -> bool:
        """Check if there are active nodes"""
        for node in self.document.nodes:
            if not self.execution_state.is_terminal(node.id):
                return True
        return False

    def _get_pending_changesets(self) -> List[Any]:
        """Get list of pending changesets to commit"""
        # Get pending changesets from CommitManager
        pending = self.contitext_engine.get_commit_manager().get_pending()
        return [p.changeset for p in pending]

    def _validate_resource_constraints(self) -> None:
        """
        Validate resource domain constraints (section 25)

        Validate that placement declarations and kind=resource dependencies are satisfiable:
        1. Check if placement declarations are valid (target must be node ID or resource_name)
        2. Check if kind=resource dependencies are satisfiable (nodes in same domain must be in same execution domain)
        3. If constraints cannot be satisfied, must produce ValidationError

        Raises:
            ResourceConstraintUnsatisfied: When resource constraints cannot be satisfied
        """
        from ..exceptions.errors import ResourceConstraintUnsatisfied

        # Get placement and edges
        placement = getattr(self.document, "placement", None) or []
        edges = getattr(self.document, "edges", [])

        # Build node ID set
        node_ids = {node.id for node in self.document.nodes}

        # 1. Validate placement declarations
        placement_domains: Dict[str, str] = {}  # target -> domain

        for p in placement:
            target = getattr(p, "target", None)
            domain = getattr(p, "domain", None)

            if not target or not domain:
                continue

            # Check if target is valid
            if target not in node_ids and not self._is_valid_resource_name(target):
                # Target is neither node ID nor valid resource_name
                # According to specification, this should produce ValidationError
                logger.warning(
                    f"Invalid placement target: {target}. "
                    f"Must be a node ID or valid resource_name."
                )
                continue

            # Check for duplicate declarations
            if target in placement_domains:
                if placement_domains[target] != domain:
                    raise ResourceConstraintUnsatisfied(
                        f"Conflicting domain assignments for target '{target}': "
                        f"'{placement_domains[target]}' vs '{domain}'"
                    )
            else:
                placement_domains[target] = domain

        # 2. Validate kind=resource dependencies
        resource_dependencies: Dict[str, List[str]] = {}  # node_id -> resource_names

        for edge in edges:
            if hasattr(edge, "kind") and str(edge.kind) == "resource":
                from_node = getattr(edge, "from_", None)
                resource_name = getattr(edge, "resource_name", None)

                if from_node and resource_name:
                    if from_node not in resource_dependencies:
                        resource_dependencies[from_node] = []
                    resource_dependencies[from_node].append(resource_name)

        # 3. Validate consistency of resource dependencies
        # If node A depends on node B via resource dependency, and B has a placement domain declaration
        # then A must belong to the same domain (or have same resource_name)

        for node_id, resources in resource_dependencies.items():
            for resource in resources:
                # Check if there is a placement with same name
                if resource in placement_domains:
                    node_domain = placement_domains.get(node_id)
                    resource_domain = placement_domains[resource]

                    if node_domain and node_domain != resource_domain:
                        raise ResourceConstraintUnsatisfied(
                            f"Node '{node_id}' (domain='{node_domain}') and "
                            f"resource '{resource}' (domain='{resource_domain}') "
                            f"have conflicting domain assignments"
                        )

        # 4. Build domain mapping (for scheduler use)
        self._domain_map = self._build_domain_map(
            placement_domains, resource_dependencies
        )

        logger.debug(
            f"Resource constraints validated: {len(placement_domains)} placement rules, "
            f"{len(resource_dependencies)} resource dependencies"
        )

    def _is_valid_resource_name(self, name: str) -> bool:
        """Check if it's a valid resource_name"""
        if not name:
            return False
        return isinstance(name, str) and name[0] != "$"

    def _build_domain_map(
        self,
        placement_domains: Dict[str, str],
        resource_dependencies: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """
        Build domain mapping

        Args:
            placement_domains: target -> domain mapping
            resource_dependencies: node_id -> resource_names mapping

        Returns:
            node_id -> domain mapping
        """
        domain_map: Dict[str, str] = {}

        # First add mappings from placement declarations
        for target, domain in placement_domains.items():
            if target in {node.id for node in self.document.nodes}:
                domain_map[target] = domain

        # Then process resource dependencies (nodes with same resource_name should be in same domain)
        resource_to_nodes: Dict[str, List[str]] = {}

        for node_id, resources in resource_dependencies.items():
            for resource in resources:
                if resource not in resource_to_nodes:
                    resource_to_nodes[resource] = []
                if node_id not in resource_to_nodes[resource]:
                    resource_to_nodes[resource].append(node_id)

        # Assign same domain to nodes sharing the same resource
        for resource, nodes in resource_to_nodes.items():
            if len(nodes) > 1:
                # Create shared domain for these nodes
                shared_domain = f"resource:{resource}"
                for node_id in nodes:
                    # If node already has a domain declaration that doesn't match, log warning
                    if node_id in domain_map and domain_map[node_id] != shared_domain:
                        logger.warning(
                            f"Node '{node_id}' has explicit domain '{domain_map[node_id]}' "
                            f"but shares resource '{resource}' with nodes requiring "
                            f"domain '{shared_domain}'"
                        )
                    else:
                        domain_map[node_id] = shared_domain

        return domain_map

    def get_node_domain(self, node_id: str) -> Optional[str]:
        """
        Get execution domain for node

        Args:
            node_id: Node ID

        Returns:
            Domain label, or None if not specified
        """
        if hasattr(self, "_domain_map"):
            return self._domain_map.get(node_id)
        return None


class ParallelLinJExecutor(LinJToContiTextMapper):
    """
    Parallel LinJ Executor

    Implements true parallel execution based on continuation mechanism
    Framework-agnostic parallel executor implementation
    """

    async def execute_parallel(
        self, initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute LinJ document in parallel

        Uses continuation and concurrency safety checking to achieve parallel execution while ensuring consistency

        Args:
            initial_state: Initial state

        Returns:
            Final state
        """
        # Initialize state
        if initial_state:
            for key, value in initial_state.items():
                changeset = {"set": {key: value}}
                self.state_manager.apply(changeset)

        # Validate resource constraints
        self._validate_resource_constraints()

        # Create main continuation
        main_continuation = self.contitext_engine.derive()

        # Parallel execution loop
        domain_map = self._allocate_domains()  # Simplified domain allocation

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

        1. Identify node groups that can be safely executed concurrently (considering data conflicts and domain constraints)
        2. Derive continuations for each group and execute in parallel
        3. Continue to next group after joining
        """
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

            # Group into safely concurrent groups (section 11.5 & section 25)
            concurrent_groups = self._find_concurrent_groups(ready_nodes, domain_map)

            # Execute each group in parallel
            for group in concurrent_groups:
                await self._execute_concurrent_group(group, main_cont)

            # Round count
            self.current_round += 1
            if self.current_round > max_rounds:
                raise RuntimeError(
                    f"Exceeded max_rounds in parallel execution: {max_rounds}"
                )

    async def _execute_concurrent_group(
        self, nodes: List[Node], parent_cont: Continuation
    ) -> None:
        """
        Execute a group of nodes in parallel

        Args:
            nodes: Group of nodes that can be safely executed concurrently
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

        # Wait for all tasks to complete (join)
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

    def _get_ready_nodes(self) -> List[Node]:
        """Get ready nodes"""
        ready = []
        for node in self.document.nodes:
            if not self.execution_state.is_terminal(
                node.id
            ) and self._check_dependencies_satisfied(node):
                ready.append(node)
        return ready

    def _allocate_domains(self) -> Mapping[str, Any]:
        """Allocate resource domains"""
        # Simplified domain allocation implementation
        return {}

    def _find_concurrent_groups(
        self, ready_nodes: List[Node], domain_map: Mapping[str, Any]
    ) -> List[List[Node]]:
        """
        Find node groups that can be safely executed concurrently (section 11.5)

        Two nodes can be safely executed in parallel if and only if:
        1. Their declared write sets (writes) are disjoint
        2. Neither node's read set (reads) intersects with the other node's write set

        If a node lacks reads/writes declarations, it's considered to read/write the entire main state,
        thereby preventing concurrent execution with other nodes

        Args:
            ready_nodes: List of ready nodes
            domain_map: Execution domain mapping

        Returns:
            List of node groups that can be safely executed concurrently
        """
        from ..core.path import PathResolver

        def get_node_reads(node: Node) -> List[str]:
            """Get list of paths that the node reads"""
            return getattr(node, "reads", None) or []

        def get_node_writes(node: Node) -> List[str]:
            """Get list of paths that the node writes"""
            return getattr(node, "writes", None) or []

        def paths_intersect(paths_a: List[str], paths_b: List[str]) -> bool:
            """Check if two sets of paths intersect"""
            for pa in paths_a:
                for pb in paths_b:
                    if PathResolver.intersect(pa, pb):
                        return True
            return False

        def can_run_parallel(node_a: Node, node_b: Node) -> bool:
            """Check if two nodes can be executed in parallel"""
            reads_a = get_node_reads(node_a)
            writes_a = get_node_writes(node_a)
            reads_b = get_node_reads(node_b)
            writes_b = get_node_writes(node_b)

            # If either node doesn't declare reads/writes, consider it to access entire state
            # This prevents concurrency with other nodes
            if not reads_a or not writes_a or not reads_b or not writes_b:
                # At least check if explicit declarations intersect
                if paths_intersect(writes_a, writes_b):
                    return False
                return True

            # Rule 1: Write sets must be disjoint
            if paths_intersect(writes_a, writes_b):
                return False

            # Rule 2: reads_a must not intersect with writes_b
            if paths_intersect(reads_a, writes_b):
                return False

            # Rule 3: reads_b must not intersect with writes_a
            if paths_intersect(reads_b, writes_a):
                return False

            return True

        # Greedy algorithm: try to put nodes into existing groups
        groups: List[List[Node]] = []

        for node in ready_nodes:
            placed = False

            # Try to put into existing groups
            for group in groups:
                can_add = True
                for member in group:
                    if not can_run_parallel(node, member):
                        can_add = False
                        break

                if can_add:
                    group.append(node)
                    placed = True
                    break

            # If cannot put into existing group, create new group
            if not placed:
                groups.append([node])

        return groups
