"""
通用执行逻辑工具

提供两个适配器共用的执行循环逻辑，确保行为一致性。
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
    通用的节点执行循环（LinJ 规范对齐版）

    实现 11.x 节和 23-26 节的核心调度逻辑。
    """
    steps = 0
    rounds = 0
    executed_nodes: Set[str] = set()
    executed_this_round: Set[str] = set()

    # 构建完整依赖图
    graph = DependencyGraph(doc.edges)

    # 修复 lint: 处理 None 情况
    policies = getattr(doc, "policies", None)
    max_rounds = (
        policies.max_rounds if policies and policies.max_rounds is not None else 1000
    )

    while steps < max_steps:
        # 1. 找出当前所有就绪节点
        ready_nodes = []
        for node in doc.nodes:
            # 只有未执行或允许重入的节点可以就绪
            policy = getattr(node, "policy", None)
            allow_reenter = getattr(policy, "allow_reenter", False) if policy else False

            if node.id in executed_this_round and not allow_reenter:
                continue
            if node.id in executed_nodes and not allow_reenter:
                continue

            # 依赖检查 (11.1 节)
            deps_satisfied = True
            incoming = graph.get_incoming(node.id)
            for edge in incoming:
                if edge.kind in (EdgeKind.DATA, EdgeKind.CONTROL):
                    if edge.from_ not in executed_nodes:
                        deps_satisfied = False
                        break

            if deps_satisfied:
                ready_nodes.append(node)

        # 2. 如果没有就绪节点，尝试进入下一轮
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

        # 3. 决定性调度 (11.3 节)
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

        # 4. 执行选中的节点
        try:
            # 如果提供了外部执行器，优先使用
            if node_executor_fn:
                success = node_executor_fn(selected_node, state_manager)
            else:
                success = _execute_single_node(
                    selected_node, state_manager, doc, evaluator
                )

            # 无论成功与否，都标记为已尝试执行，并计步
            executed_nodes.add(selected_node.id)
            steps += 1

            if success:
                executed_this_round.add(selected_node.id)
            else:
                logger.warning(f"Node execution returned False: {selected_node.id}")

        except Exception as e:
            logger.error(f"Node execution failed: {selected_node.id} - {e}")
            # 即使失败，也标记为已尝试执行，并计步
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
    """执行单个节点并处理变更集提交"""
    try:
        if node.type == "hint":
            _execute_hint(node, state_manager)
        elif node.type == "tool":
            _execute_tool(node, state_manager)
        elif node.type == "join":
            _execute_join(node, state_manager)
        elif node.type == "gate":
            # Gate 节点目前在 runner_utils 中只是根据条件决定流程，
            # 不产生直接变更集。更复杂的 Gate 行为在 adapter 中处理。
            _execute_gate(node, state_manager)
        return True
    except Exception as e:
        logger.error(f"Execution error at node {node.id}: {e}")
        return False


def _resolve_value(val_ref: Any, state: Dict[str, Any]) -> Any:
    """解析值引用（符合 5.x 节读语义）"""
    if val_ref is None:
        return None

    # 如果是 Pydantic 模型且有 resolve 方法，直接使用
    if hasattr(val_ref, "resolve") and callable(getattr(val_ref, "resolve")):
        return val_ref.resolve(state)

    # 否则按字典处理
    if hasattr(val_ref, "model_dump"):
        data = val_ref.model_dump(by_alias=True)
    elif isinstance(val_ref, dict):
        data = val_ref
    else:
        return val_ref

    # 优先解析路径 (6.2 节)
    path = data.get("$path") or data.get("path")
    if path:
        return PathResolver.get(state, path)

    # 其次解析常量
    if "$const" in data:
        return data["$const"]
    if "const" in data:
        return data["const"]

    return None


def _execute_hint(node: HintNode, state_manager: StateManager) -> None:
    """执行 Hint 节点 (13.1 节)"""
    template = node.template
    state = state_manager.get_full_state()

    # 渲染变量
    result = template
    if node.vars:
        for name, ref in node.vars.items():
            val = _resolve_value(ref, state)
            # 简单字符串替换，符合提示节点基本语义
            result = result.replace(
                f"{{{{{name}}}}}", str(val) if val is not None else ""
            )

    # 提交变更
    if node.write_to:
        cs = ChangeSet.create_write(node.write_to, result)
        state_manager.apply(cs)


def _execute_tool(node: ToolNode, state_manager: StateManager) -> None:
    """执行 Tool 节点 (13.2 节)"""
    state = state_manager.get_full_state()
    tool_name = node.call.name

    # 这里应该对接真实的 ToolAdapter。目前简化处理。
    # 在 adapter.py 中有更完善的逻辑。
    logger.info(f"Executing tool: {tool_name}")

    # 临时模拟：如果是 echo 工具
    result = None
    if tool_name == "echo":
        args = node.call.args
        if args and "message" in args:
            result = _resolve_value(args["message"], state)

    if node.write_to and result is not None:
        cs = ChangeSet.create_write(node.write_to, result)
        state_manager.apply(cs)


def _execute_join(node: JoinNode, state_manager: StateManager) -> None:
    """执行 Join 节点 (13.3 节)"""
    state = state_manager.get_full_state()
    input_val = PathResolver.get(state, node.input_from)

    # 校验逻辑 (Glossary)
    if hasattr(node, "glossary") and node.glossary:
        for item in node.glossary:
            # 简单的 forbid 校验
            forbid = getattr(item, "forbid", [])
            for word in forbid:
                if input_val and word in str(input_val):
                    logger.warning(f"Join word forbidden: {word}")

    if node.output_to:
        cs = ChangeSet.create_write(node.output_to, input_val)
        state_manager.apply(cs)


def _execute_gate(node: GateNode, state_manager: StateManager) -> None:
    """执行 Gate 节点 (13.4 节)"""
    # Note: Gate 节点在 execute_nodes_generic 中主要作为控制流标记
    # 这里的逻辑仅用于触发可能的状态变更（如果有）
    state = state_manager.get_full_state()
    result = evaluate_condition(node.condition, state)

    logger.debug(f"Gate {node.id} evaluated to {result}")
    # 根据结果决定下一步的路由通常由 scheduler/loop 逻辑处理
    # 规范中 Gate 节点本身不一定导致 state 变更，除非定义了 write_to
