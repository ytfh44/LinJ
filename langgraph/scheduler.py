"""
LangGraph决定性调度器

基于AutoGen实现迁移，确保与LinJ规范第11节完全一致
提供与AutoGen版本完全相同的调度行为
"""

import logging
from typing import Any, Dict, List, Optional, Set, Mapping, Tuple
from dataclasses import dataclass

from ..executor.scheduler import (
    BaseScheduler,
    DeterministicScheduler,
    SchedulingDecision,
    SchedulingStrategy,
)
from ..executor.autogen_scheduler import (
    ExecutionDomain,
    DomainAllocator,
    ExecutionState,
    select_next_node,
    get_node_path_set,
    check_path_intersection,
    check_concurrent_safety,
    find_concurrent_groups,
    are_dependencies_satisfied,
)

logger = logging.getLogger(__name__)


class LangGraphDeterministicScheduler(DeterministicScheduler):
    """
    LangGraph决定性调度器

    与AutoGen版本保持完全一致的调度行为：
    - 相同的决定性排序规则（11.3节）
    - 相同的依赖检查逻辑
    - 相同的并发安全检查
    - 相同的执行域处理
    """

    def __init__(self, nodes: List[Any], enable_parallel: bool = False):
        super().__init__(nodes)
        self.enable_parallel = enable_parallel

        # 复用AutoGen的状态管理和域分配逻辑
        self._execution_state = ExecutionState()
        self._domain_allocator = DomainAllocator()

        # 域映射缓存
        self._domain_map: Optional[Mapping[str, ExecutionDomain]] = None

    def _initialize_domain_map(self, document: Any) -> None:
        """初始化执行域映射"""
        if self._domain_map is None:
            try:
                self._domain_map = self._domain_allocator.allocate_domains(
                    document, document.edges if hasattr(document, "edges") else []
                )
            except Exception as e:
                logger.warning(
                    f"Domain allocation failed: {e}, using default allocation"
                )
                self._domain_map = {}

    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: Any,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """
        选择节点执行（与AutoGen完全一致的逻辑）

        Args:
            ready_nodes: 就绪节点列表
            context: 执行上下文（包含document等）
            max_concurrency: 最大并发数限制

        Returns:
            调度决策结果
        """
        # 初始化域映射（如果需要）
        if hasattr(context, "document"):
            self._initialize_domain_map(context.document)

        # 过滤可执行的节点
        executable_nodes = []
        executed_this_round = getattr(context, "executed_this_round", set())

        for node in ready_nodes:
            node_id = getattr(node, "id", "unknown")

            # 检查是否已在执行中
            if self.is_executing(node_id):
                continue

            # 检查本轮是否已执行
            allow_reenter = getattr(node, "policy", None)
            allow_reenter = (
                getattr(allow_reenter, "allow_reenter", False)
                if allow_reenter
                else False
            )

            if node_id in executed_this_round and not allow_reenter:
                continue

            # 检查是否满足前置依赖
            if not self._are_dependencies_satisfied(node_id, context):
                continue

            executable_nodes.append(node)

        if not executable_nodes:
            return SchedulingDecision(
                selected_nodes=[],
                execution_order=[],
                concurrency_level=0,
                strategy=SchedulingStrategy.DETERMINISTIC,
                metadata={"reason": "no_executable_nodes"},
            )

        # 根据是否启用并行选择执行策略
        if self.enable_parallel and max_concurrency and max_concurrency > 1:
            return self._select_parallel_group(
                executable_nodes, context, max_concurrency
            )
        else:
            return self._select_single_node(executable_nodes, context)

    def _select_single_node(
        self, executable_nodes: List[Any], context: Any
    ) -> SchedulingDecision:
        """选择单个节点执行（串行模式）"""
        # 按决定性规则排序（与AutoGen完全一致）
        sorted_nodes = self._sort_deterministically(executable_nodes)
        selected_node = sorted_nodes[0]

        # 记录批量大小
        self._stats["batch_sizes"].append(1)

        return SchedulingDecision(
            selected_nodes=[selected_node],
            execution_order=[getattr(selected_node, "id", "unknown")],
            concurrency_level=1,
            strategy=SchedulingStrategy.DETERMINISTIC,
            metadata={
                "total_ready": len(executable_nodes),
                "executable": len(executable_nodes),
                "selected_rank": getattr(selected_node, "rank", 0),
                "execution_mode": "serial",
            },
        )

    def _select_parallel_group(
        self, executable_nodes: List[Any], context: Any, max_concurrency: int
    ) -> SchedulingDecision:
        """选择可并行执行的节点组"""
        # 使用与AutoGen相同的并发分组逻辑
        concurrent_groups = find_concurrent_groups(executable_nodes, self._domain_map)

        if not concurrent_groups:
            return self._select_single_node(executable_nodes, context)

        # 选择最大的可并行组（但不超过并发限制）
        selected_group = concurrent_groups[0]
        selected_nodes = selected_group[:max_concurrency]

        # 记录批量大小
        self._stats["batch_sizes"].append(len(selected_nodes))

        return SchedulingDecision(
            selected_nodes=selected_nodes,
            execution_order=[getattr(node, "id", "unknown") for node in selected_nodes],
            concurrency_level=len(selected_nodes),
            strategy=SchedulingStrategy.PARALLEL,
            metadata={
                "total_ready": len(executable_nodes),
                "executable": len(executable_nodes),
                "parallel_groups": len(concurrent_groups),
                "selected_group_size": len(selected_group),
                "max_concurrency": max_concurrency,
                "execution_mode": "parallel",
            },
        )

    def _are_dependencies_satisfied(self, node_id: str, context: Any) -> bool:
        """检查节点的依赖是否已满足（与AutoGen完全一致）"""
        # 获取依赖图和执行状态
        graph = getattr(context, "dependency_graph", None)
        if not graph:
            return True

        # 使用与AutoGen相同的依赖检查逻辑
        return are_dependencies_satisfied(node_id, graph, self._execution_state)

    def get_dependencies(self, node: Any) -> List[str]:
        """获取节点的依赖列表"""
        return getattr(node, "dependencies", [])

    def can_execute(self, node: Any, context: Any) -> bool:
        """
        检查节点是否可以执行（增强版）

        包含域约束检查
        """
        node_id = getattr(node, "id", "unknown")

        # 基础检查
        if not super().can_execute(node, context):
            return False

        # 域约束检查
        if self._domain_map and node_id in self._domain_map:
            node_domain = self._domain_map[node_id]

            # 检查同域内是否有节点正在执行
            for executing_id in self._executing:
                if executing_id in self._domain_map:
                    executing_domain = self._domain_map[executing_id]
                    if executing_domain is node_domain:
                        return False

        return True

    def mark_completed(self, node_id: str, success: bool = True) -> None:
        """标记节点执行完成（更新执行状态）"""
        super().mark_completed(node_id, success)

        # 更新执行状态
        if success:
            self._execution_state.completed.add(node_id)
        else:
            self._execution_state.failed.add(node_id)

    def reset(self) -> None:
        """重置调度器状态"""
        super().reset()
        self._execution_state = ExecutionState()
        self._domain_map = None

    def get_domain_info(self) -> Dict[str, Any]:
        """获取执行域信息（用于调试）"""
        if not self._domain_map:
            return {"domains": "not_initialized"}

        domain_info = {}
        for node_id, domain in self._domain_map.items():
            domain_info[node_id] = {
                "domain_label": domain.domain_label,
                "node_count": len(domain.node_ids),
                "resource_names": list(domain.resource_names),
            }

        return {
            "total_domains": len(
                set(d.domain_label for d in self._domain_map.values() if d.domain_label)
            ),
            "domain_mapping": domain_info,
        }
