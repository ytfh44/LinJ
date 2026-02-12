"""
LinJ 执行器

主执行器，协调调度、节点执行、变更集提交
集成诊断追踪功能
使用共享 contitext 引擎组件
"""

import sys
import os
import copy
from typing import Any, Callable, Dict, List, Optional, Set

# Add shared components path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Use local imports for now to avoid import issues
# TODO: migrate to shared components once import paths are resolved
from ..core.document import LinJDocument
from ..core.nodes import Node, HintNode, ToolNode, JoinNode, GateNode
from ..core.edges import DependencyGraph
from ..core.changeset import ChangeSet, ChangeSetBuilder
from ..core.state import StateManager
from ..core.errors import ExecutionError, ValidationError, ContractViolation
from ..core.contract_validator import ContractValidator
from ..core.tracing import DiagnosticTracer, TraceEntry, TracingMixin

SHARED_AVAILABLE = False  # Temporarily disable shared imports

from .scheduler import (
    DeterministicScheduler,
    ExecutionState,
    are_dependencies_satisfied,
)


class NodeExecutionResult:
    """节点执行结果"""

    def __init__(
        self,
        success: bool,
        changeset: Optional[ChangeSet] = None,
        next_nodes: Optional[List[str]] = None,
        error: Optional[Exception] = None,
    ):
        self.success = success
        self.changeset = changeset or ChangeSet()
        self.next_nodes = next_nodes or []
        self.error = error


class LinJExecutor(TracingMixin):
    """
    LinJ 执行器

    执行 LinJ 文档，管理状态，协调节点执行
    集成诊断追踪功能
    """

    SUPPORTED_MAJOR_VERSION = 0
    SUPPORTED_MINOR_VERSION = 1

    def __init__(
        self,
        tool_registry: Optional[Dict[str, Callable]] = None,
        enable_tracing: bool = True,
        use_shared_contitext: bool = True,
    ):
        TracingMixin.__init__(self)
        self.tool_registry = tool_registry or {}
        self._step_count = 0
        self._round_count = 0
        self._contract_validator = ContractValidator()
        self._use_shared_contitext = use_shared_contitext and SHARED_AVAILABLE

        if enable_tracing:
            self.set_tracer(DiagnosticTracer(enable_detailed_logging=True))

        # Initialize shared contitext engine if available
        if self._use_shared_contitext:
            self._contitext_engine = ContiTextEngine()
        else:
            self._contitext_engine = None

    def register_tool(self, name: str, executor: Callable) -> None:
        """注册工具"""
        self.tool_registry[name] = executor

    async def run(
        self, document: LinJDocument, initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行 LinJ 文档

        Args:
            document: LinJ 文档
            initial_state: 初始状态

        Returns:
            最终状态
        """
        # 版本检查
        document.check_version_compatibility(
            self.SUPPORTED_MAJOR_VERSION, self.SUPPORTED_MINOR_VERSION
        )

        # 验证引用
        ref_errors = document.validate_references()
        if ref_errors:
            raise ValidationError(f"Reference validation failed: {ref_errors}")

        # 初始化
        state_manager = StateManager(initial_state)
        scheduler = DeterministicScheduler(document.nodes)
        exec_state = ExecutionState()
        graph = document.build_dependency_graph()

        # 激活机制 (13.4 节)
        # 初始激活所有无前置依赖的节点
        activated_nodes: Set[str] = set()
        for node in document.nodes:
            # 获取所有前置依赖（data 和 control）
            deps = graph.get_data_dependencies(
                node.id
            ) + graph.get_control_dependencies(node.id)
            if not deps:
                activated_nodes.add(node.id)

        # 执行主循环
        max_steps = document.policies.max_steps if document.policies else None
        max_rounds = document.policies.max_rounds if document.policies else 10

        executed_this_round: Set[str] = set()

        while True:
            # 获取就绪节点
            ready_nodes = []
            for node_id in list(activated_nodes):
                node = document.get_node(node_id)

                # 检查本轮是否已执行 (13.4 节)
                allow_reenter = node.policy.allow_reenter if node.policy else False
                if node_id in executed_this_round and not allow_reenter:
                    continue

                # 如果已进入终态，检查是否允许重复执行（13.4 节 & 12 节）
                if exec_state.is_terminal(node_id) and not allow_reenter:
                    is_in_loop = False
                    if document.loops:
                        for loop in document.loops:
                            if node_id in loop.members:
                                is_in_loop = True
                                break
                    if not is_in_loop:
                        continue

                # 检查是否满足前置依赖
                # 循环入口节点允许 OR 语义（满足其一即可）
                is_loop_entry = False
                if document.loops:
                    for loop in document.loops:
                        if loop.entry == node_id:
                            is_loop_entry = True
                            break

                if not are_dependencies_satisfied(
                    node_id, graph, exec_state, check_all=not is_loop_entry
                ):
                    continue

                ready_nodes.append(node)

            if not ready_nodes:
                # 如果没有就绪节点，且本轮有已执行节点，则尝试开启新轮次
                if executed_this_round:
                    self._round_count += 1
                    executed_this_round.clear()
                    if self.tracer:
                        self.tracer.start_round()

                    if max_rounds and self._round_count > max_rounds:
                        raise ExecutionError(f"Exceeded max_rounds: {max_rounds}")

                    # 重新评估就绪节点
                    continue

                # 真正没有可执行节点了
                break

            # 按决定性顺序选择节点
            node = scheduler.select_from_ready(ready_nodes, executed_this_round)
            if not node:
                # 可能所有就绪节点都在 in_progress
                break

            # 节点选中执行，从激活列表移除 (直到再次被触发)
            activated_nodes.discard(node.id)

            # 步骤计数与限制
            self._step_count += 1
            if max_steps and self._step_count > max_steps:
                raise ExecutionError(f"Exceeded max_steps: {max_steps}")

            # 分配 Step ID
            step_id = scheduler.allocate_step_id()

            # 开始追踪 (27 节)
            trace = self.trace_step_start(node.id, reads=node.reads)

            # 执行节点
            scheduler.mark_executing(node.id)
            result = await self._execute_node(node, state_manager, document)
            scheduler.mark_completed(node.id)

            if result.success:
                # 应用状态变更
                state_manager.apply(result.changeset, step_id)

                # 记录完成
                self.trace_step_complete(trace, "completed", writes=node.writes)

                exec_state.completed.add(node.id)
                executed_this_round.add(node.id)

                # 处理节点激活 (13.4 节)
                neighbors = graph.get_outgoing(node.id)
                if isinstance(node, GateNode):
                    # GateNode 仅激活选中的分支
                    next_nodes = node.get_next_nodes(state_manager.get_full_state())
                    for target_id in next_nodes:
                        activated_nodes.add(target_id)
                else:
                    # 普通节点激活所有后续节点 (仅 data/control)
                    for edge in neighbors:
                        if not edge.is_resource():
                            activated_nodes.add(edge.to)
            else:
                # 节点执行失败
                self.trace_step_complete(trace, "failed", error=result.error)

                exec_state.failed.add(node.id)
                if result.error:
                    raise result.error
                raise ExecutionError(
                    f"Node {node.id} execution failed without explicit error."
                )

        # 导出追踪信息到状态 (27 节)
        if self.tracer:
            # 获取完整状态
            full_state = state_manager.get_full_state()
            # 导出追踪到字典中
            self.tracer.export_to_state(full_state)

            # 将追踪信息注入回状态管理器
            trace_data = full_state.get("$.trace")
            if trace_data:
                state_manager.apply(ChangeSet.create_write("$.trace", trace_data))

        return state_manager.get_full_state()

    async def _execute_node(
        self, node: Node, state_manager: StateManager, document: LinJDocument
    ) -> NodeExecutionResult:
        """执行单个节点（带合同验证）"""
        try:
            # 1. 验证 in_contract
            if node.in_contract:
                in_result = self._contract_validator.validate_in_contract(
                    node, state_manager
                )
                if not in_result.valid:
                    return NodeExecutionResult(
                        success=False,
                        error=ContractViolation(
                            f"in_contract validation failed for node {node.id}",
                            {
                                "node_id": node.id,
                                "violations": [
                                    {
                                        "path": e.path,
                                        "expected": e.expected,
                                        "actual": e.actual,
                                        "message": e.message,
                                    }
                                    for e in in_result.errors
                                ],
                            },
                        ),
                    )

            # 2. 执行节点逻辑
            if isinstance(node, HintNode):
                result = await self._execute_hint(node, state_manager)
            elif isinstance(node, ToolNode):
                result = await self._execute_tool(node, state_manager)
            elif isinstance(node, JoinNode):
                result = await self._execute_join(node, state_manager)
            elif isinstance(node, GateNode):
                result = await self._execute_gate(node, state_manager)
            else:
                raise ExecutionError(f"Unknown node type: {type(node)}")

            # 3. 如果执行失败，直接返回结果
            if not result.success:
                return result

            # 4. 验证 out_contract
            if node.out_contract and result.changeset:
                out_result = self._contract_validator.validate_out_contract(
                    node, result.changeset, state_manager
                )
                if not out_result.valid:
                    return NodeExecutionResult(
                        success=False,
                        error=ContractViolation(
                            f"out_contract validation failed for node {node.id}",
                            {
                                "node_id": node.id,
                                "violations": [
                                    {
                                        "path": e.path,
                                        "expected": e.expected,
                                        "actual": e.actual,
                                        "message": e.message,
                                    }
                                    for e in out_result.errors
                                ],
                            },
                        ),
                    )

            return result
        except Exception as e:
            return NodeExecutionResult(success=False, error=e)

    async def _execute_hint(
        self, node: HintNode, state_manager: StateManager
    ) -> NodeExecutionResult:
        """执行 hint 节点"""
        rendered = node.render(state_manager.get_full_state())

        changeset = ChangeSetBuilder().write(node.write_to, rendered).build()

        return NodeExecutionResult(success=True, changeset=changeset)

    async def _execute_tool(
        self, node: ToolNode, state_manager: StateManager
    ) -> NodeExecutionResult:
        """执行 tool 节点"""
        tool_name = node.call.name

        if tool_name not in self.tool_registry:
            raise ExecutionError(f"Tool not found: {tool_name}")

        tool_executor = self.tool_registry[tool_name]
        args = node.get_args(state_manager.get_full_state())

        # 执行工具
        try:
            result = await tool_executor(**args)
        except Exception as e:
            raise ExecutionError(f"Tool execution failed: {e}")

        # 构建变更集
        builder = ChangeSetBuilder()
        if node.write_to:
            builder.write(node.write_to, result)

        return NodeExecutionResult(success=True, changeset=builder.build())

    async def _execute_join(
        self, node: JoinNode, state_manager: StateManager
    ) -> NodeExecutionResult:
        """执行 join 节点"""
        from ..core.path import PathResolver

        # 读取输入
        input_value = PathResolver.get(state_manager.get_full_state(), node.input_from)
        if input_value is None:
            input_value = ""

        text = str(input_value)

        # 验证禁止项 (13.3 节)
        forbidden_term = node.validate_forbidden(text)
        if forbidden_term:
            raise ValidationError(
                f"Join validation failed for node {node.id}: "
                f"forbidden term '{forbidden_term}' found in output",
                details={
                    "node_id": node.id,
                    "forbidden_term": forbidden_term,
                    "output_snippet": text[:100] + "..." if len(text) > 100 else text,
                },
            )

        # 构建变更集
        changeset = ChangeSetBuilder().write(node.output_to, text).build()

        return NodeExecutionResult(success=True, changeset=changeset)

    async def _execute_gate(
        self, node: GateNode, state_manager: StateManager
    ) -> NodeExecutionResult:
        """执行 gate 节点"""
        # 求值条件
        condition_result = node.evaluate(state_manager.get_full_state())

        # 获取下一步节点
        next_nodes = node.then if condition_result else node.else_

        # gate 节点本身不产生变更集
        return NodeExecutionResult(
            success=True, changeset=ChangeSet(), next_nodes=next_nodes
        )


def load_document(path: str) -> LinJDocument:
    """从文件加载文档"""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return LinJDocument(**data)
