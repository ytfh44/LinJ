"""
Common execution logic utilities

Provides execution loop logic shared between two adapters to ensure behavioral consistency.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Set, Callable

from ..core.document import LinJDocument
from ..core.nodes import Node, HintNode, ToolNode, JoinNode, GateNode
from ..core.state import StateManager
from ..core.path import PathResolver
from ..core.changeset import ChangeSet
from ..core.condition import evaluate_condition
from ..core.edges import DependencyGraph, EdgeKind

logger = logging.getLogger(__name__)


def execute_nodes_generic(
    doc: LinJDocument,
    state_manager: StateManager,
    scheduler: Any,
    evaluator: Any,
    max_steps: int,
    node_executor_fn: Optional[Callable[[Node, StateManager], bool]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generic node execution loop (LinJ specification aligned version)

    Implements core scheduling logic from sections 11.x and 23-26.
    """
    steps = 0
    rounds = 0
    executed_nodes: Set[str] = set()
    executed_this_round: Set[str] = set()

    # Build complete dependency graph
    graph = DependencyGraph(doc.edges)

    # Fix lint: handle None case
    policies = getattr(doc, "policies", None)
    max_rounds = (
        policies.max_rounds if policies and policies.max_rounds is not None else 1000
    )

    while steps < max_steps:
        # 1. Find all ready nodes
        ready_nodes = []
        for node in doc.nodes:
            # Only unexecuted or re-entrant allowed nodes can be ready
            policy = getattr(node, "policy", None)
            allow_reenter = getattr(policy, "allow_reenter", False) if policy else False

            if node.id in executed_this_round and not allow_reenter:
                continue
            if node.id in executed_nodes and not allow_reenter:
                continue

            # Dependency check (section 11.1)
            deps_satisfied = True
            incoming = graph.get_incoming(node.id)
            for edge in incoming:
                if edge.kind in (EdgeKind.DATA, EdgeKind.CONTROL):
                    if edge.from_ not in executed_nodes:
                        deps_satisfied = False
                        break

            if deps_satisfied:
                ready_nodes.append(node)

        # 2. If no ready nodes, try to enter next round
        if not ready_nodes:
            if executed_this_round:
                rounds += 1
                executed_this_round.clear()
                if rounds >= max_rounds:
                    logger.warning(f"Exceeded max_rounds: {max_rounds}")
                    break
                continue
            else:
                break

        # 3. Deterministic scheduling (section 11.3)
        if hasattr(scheduler, "select_from_ready"):
            selected_node = scheduler.select_from_ready(ready_nodes)
        else:
            node_order = {n.id: i for i, n in enumerate(doc.nodes)}

            def sort_key(n):
                rank = getattr(n, "rank", 0) or 0
                order = node_order.get(n.id, float("inf"))
                return (-rank, order, n.id)

            ready_nodes.sort(key=sort_key)
            selected_node = ready_nodes[0]

        if not selected_node:
            break

        # 4. Execute selected node
        try:
            # If external executor provided, prioritize its use
            if node_executor_fn:
                success = node_executor_fn(selected_node, state_manager)
            else:
                success = _execute_single_node(
                    selected_node, state_manager, doc, evaluator
                )

            # Mark as attempted regardless of success, and count steps
            executed_nodes.add(selected_node.id)
            steps += 1

            if success:
                executed_this_round.add(selected_node.id)
            else:
                logger.warning(f"Node execution returned False: {selected_node.id}")

        except Exception as e:
            logger.error(f"Node execution failed: {selected_node.id} - {e}")
            # Mark as attempted even on failure, and count steps
            executed_nodes.add(selected_node.id)
            steps += 1
            break

    return state_manager.get_full_state(), {
        "total_steps": steps,
        "total_rounds": rounds + 1 if executed_this_round else rounds,
    }


def _execute_single_node(
    node: Any, state_manager: StateManager, doc: LinJDocument, evaluator: Any
) -> bool:
    """Execute single node and handle changeset commit"""
    try:
        if node.type == "hint":
            _execute_hint(node, state_manager)
        elif node.type == "tool":
            _execute_tool(node, state_manager)
        elif node.type == "join":
            _execute_join(node, state_manager)
        elif node.type == "gate":
            # Gate nodes currently in runner_utils only decide flow based on conditions,
            # no direct changeset generation. More complex Gate behavior is handled in adapter.
            _execute_gate(node, state_manager)
        return True
    except Exception as e:
        logger.error(f"Execution error at node {node.id}: {e}")
        return False


def _resolve_value(val_ref: Any, state: Dict[str, Any]) -> Any:
    """Parse value reference (aligning with reading semantics from section 5.x)"""
    if val_ref is None:
        return None

    # If Pydantic model with resolve method, use directly
    if hasattr(val_ref, "resolve") and callable(getattr(val_ref, "resolve")):
        return val_ref.resolve(state)

    # Otherwise process as dict
    if hasattr(val_ref, "model_dump"):
        data = val_ref.model_dump(by_alias=True)
    elif isinstance(val_ref, dict):
        data = val_ref
    else:
        return val_ref

    # First priority: resolve path (section 6.2)
    path = data.get("$path") or data.get("path")
    if path:
        return PathResolver.get(state, path)

    # Second priority: resolve constant
    if "$const" in data:
        return data["$const"]
    if "const" in data:
        return data["const"]

    return None


def _execute_hint(node: HintNode, state_manager: StateManager) -> None:
    """Execute Hint node (section 13.1)"""
    template = node.template
    state = state_manager.get_full_state()

    # Render variables
    result = template
    if node.vars:
        for name, ref in node.vars.items():
            val = _resolve_value(ref, state)
            # Simple string replacement, aligning with hint node basic semantics
            result = result.replace(
                f"{{{{{name}}}}}", str(val) if val is not None else ""
            )

    # Commit changes
    if node.write_to:
        cs = ChangeSet.create_write(node.write_to, result)
        state_manager.apply(cs)


def _execute_tool(node: ToolNode, state_manager: StateManager) -> None:
    """Execute Tool node (section 13.2)"""
    state = state_manager.get_full_state()
    tool_name = node.call.name

    # Here should connect to real ToolAdapter. Currently simplified.
    # More complete logic exists in adapter.py.
    logger.info(f"Executing tool: {tool_name}")

    # Temporary mock: if echo tool
    result = None
    if tool_name == "echo":
        args = node.call.args
        if args and "message" in args:
            result = _resolve_value(args["message"], state)

    if node.write_to and result is not None:
        cs = ChangeSet.create_write(node.write_to, result)
        state_manager.apply(cs)


def _execute_join(node: JoinNode, state_manager: StateManager) -> None:
    """Execute Join node (section 13.3)"""
    state = state_manager.get_full_state()
    input_val = PathResolver.get(state, node.input_from)

    # Validation logic (Glossary)
    if hasattr(node, "glossary") and node.glossary:
        for item in node.glossary:
            # Simple forbid validation
            forbid = getattr(item, "forbid", [])
            for word in forbid:
                if input_val and word in str(input_val):
                    logger.warning(f"Join word forbidden: {word}")

    if node.output_to:
        cs = ChangeSet.create_write(node.output_to, input_val)
        state_manager.apply(cs)


def _execute_gate(node: GateNode, state_manager: StateManager) -> None:
    """Execute Gate node (section 13.4)"""
    # Note: Gate nodes in execute_nodes_generic primarily serve as control flow markers
    # This logic here is only for triggering potential state changes (if any)
    state = state_manager.get_full_state()
    result = evaluate_condition(node.condition, state)

    logger.debug(f"Gate {node.id} evaluated to {result}")
    # Routing decisions based on results are typically handled by scheduler/loop logic
    # In the spec, Gate nodes themselves don't necessarily cause state changes unless write_to is defined
