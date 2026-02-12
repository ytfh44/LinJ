"""
调度器抽象类

定义节点调度的抽象接口和基础实现，支持多种调度策略。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from .types import ExecutionContext, ExecutionStatus


class SchedulingStrategy(Enum):
    """调度策略枚举"""

    DETERMINISTIC = "deterministic"  # 决定性调度
    PRIORITY = "priority"  # 优先级调度
    ROUND_ROBIN = "round_robin"  # 轮询调度
    PARALLEL = "parallel"  # 并行调度
    ADAPTIVE = "adaptive"  # 自适应调度


@dataclass
class SchedulingDecision:
    """调度决策结果"""

    selected_nodes: List[Any]  # 选中的节点列表
    execution_order: List[str]  # 执行顺序
    concurrency_level: int  # 并发级别
    strategy: SchedulingStrategy
    metadata: Dict[str, Any]  # 额外的调度信息


class Scheduler(ABC):
    """
    调度器抽象基类

    定义节点调度的核心接口：
    - 节点选择和排序
    - 依赖关系分析
    - 并发安全性检查
    - 执行状态管理
    """

    @abstractmethod
    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: ExecutionContext,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """
        从就绪节点中选择要执行的节点

        Args:
            ready_nodes: 就绪节点列表
            context: 执行上下文
            max_concurrency: 最大并发数限制

        Returns:
            调度决策结果
        """
        pass

    @abstractmethod
    def can_execute(self, node: Any, context: ExecutionContext) -> bool:
        """
        检查节点是否可以执行

        Args:
            node: 节点对象
            context: 执行上下文

        Returns:
            True 表示可以执行，False 表示不能执行
        """
        pass

    @abstractmethod
    def get_dependencies(self, node: Any) -> List[str]:
        """
        获取节点的依赖列表

        Args:
            node: 节点对象

        Returns:
            依赖节点ID列表
        """
        pass

    @abstractmethod
    def mark_executing(self, node_id: str) -> None:
        """标记节点开始执行"""
        pass

    @abstractmethod
    def mark_completed(self, node_id: str, success: bool = True) -> None:
        """标记节点执行完成"""
        pass

    def allocate_step_id(self) -> int:
        """分配步骤ID"""
        # 默认实现：简单的递增计数器
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        self._step_counter += 1
        return self._step_counter

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取调度统计信息"""
        return {
            "total_scheduled": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_batch_size": 0.0,
        }


class BaseScheduler(Scheduler):
    """
    基础调度器实现

    提供通用的调度逻辑和状态管理功能
    """

    def __init__(self):
        self._executing: Set[str] = set()
        self._completed: Set[str] = set()
        self._failed: Set[str] = set()
        self._stats = {
            "total_scheduled": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "batch_sizes": [],
        }

    def can_execute(self, node: Any, context: ExecutionContext) -> bool:
        """检查节点是否可以执行"""
        node_id = getattr(node, "id", "unknown")

        # 检查是否已在执行中
        if node_id in self._executing:
            return False

        # 检查是否已完成
        if node_id in self._completed or node_id in self._failed:
            return False

        # 检查依赖是否满足
        dependencies = self.get_dependencies(node)
        for dep_id in dependencies:
            if dep_id not in self._completed:
                return False

        return True

    def mark_executing(self, node_id: str) -> None:
        """标记节点开始执行"""
        self._executing.add(node_id)
        self._stats["total_scheduled"] += 1

    def mark_completed(self, node_id: str, success: bool = True) -> None:
        """标记节点执行完成"""
        self._executing.discard(node_id)

        if success:
            self._completed.add(node_id)
            self._stats["successful_executions"] += 1
        else:
            self._failed.add(node_id)
            self._stats["failed_executions"] += 1

    def is_executing(self, node_id: str) -> bool:
        """检查节点是否正在执行"""
        return node_id in self._executing

    def is_completed(self, node_id: str) -> bool:
        """检查节点是否已完成"""
        return node_id in self._completed

    def is_failed(self, node_id: str) -> bool:
        """检查节点是否执行失败"""
        return node_id in self._failed

    def get_pending_nodes(self) -> Set[str]:
        """获取待执行节点ID"""
        return (self._completed | self._failed) - self._executing

    def reset(self) -> None:
        """重置调度器状态"""
        self._executing.clear()
        self._completed.clear()
        self._failed.clear()
        self._stats = {
            "total_scheduled": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "batch_sizes": [],
        }

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取调度统计信息"""
        batch_sizes = self._stats["batch_sizes"]
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0.0

        return {
            "total_scheduled": self._stats["total_scheduled"],
            "successful_executions": self._stats["successful_executions"],
            "failed_executions": self._stats["failed_executions"],
            "average_batch_size": avg_batch_size,
            "currently_executing": len(self._executing),
            "completed_count": len(self._completed),
            "failed_count": len(self._failed),
        }


class DeterministicScheduler(BaseScheduler):
    """
    决定性调度器

    基于节点优先级和文档顺序的确定性格调度
    """

    def __init__(self, nodes: List[Any]):
        super().__init__()
        self._node_order = {
            getattr(node, "id", str(i)): i for i, node in enumerate(nodes)
        }
        self._nodes = {
            getattr(node, "id", str(i)): node for i, node in enumerate(nodes)
        }

    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: ExecutionContext,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """选择节点进行执行"""
        # 过滤可执行的节点
        executable_nodes = [
            node for node in ready_nodes if self.can_execute(node, context)
        ]

        if not executable_nodes:
            return SchedulingDecision(
                selected_nodes=[],
                execution_order=[],
                concurrency_level=0,
                strategy=SchedulingStrategy.DETERMINISTIC,
                metadata={"reason": "no_executable_nodes"},
            )

        # 按决定性规则排序
        sorted_nodes = self._sort_deterministically(executable_nodes)

        # 选择第一个节点（单线程执行）
        selected_node = sorted_nodes[0]

        # 记录批量大小
        self._stats["batch_sizes"].append(1)

        return SchedulingDecision(
            selected_nodes=[selected_node],
            execution_order=[getattr(selected_node, "id", "unknown")],
            concurrency_level=1,
            strategy=SchedulingStrategy.DETERMINISTIC,
            metadata={
                "total_ready": len(ready_nodes),
                "executable": len(executable_nodes),
                "selected_rank": getattr(selected_node, "rank", 0),
            },
        )

    def _sort_deterministically(self, nodes: List[Any]) -> List[Any]:
        """按决定性规则排序节点"""

        def sort_key(node):
            rank = getattr(node, "rank", 0)
            rank = rank if rank is not None else 0
            order = self._node_order.get(getattr(node, "id", "unknown"), float("inf"))
            node_id = getattr(node, "id", "unknown")
            return (-rank, order, node_id)  # 负rank实现降序

        return sorted(nodes, key=sort_key)

    def get_dependencies(self, node: Any) -> List[str]:
        """获取节点的依赖列表"""
        return getattr(node, "dependencies", [])


class ParallelScheduler(BaseScheduler):
    """
    并行调度器

    支持多节点并行执行，同时保证安全性
    """

    def __init__(self, max_concurrency: int = 4):
        super().__init__()
        self.max_concurrency = max_concurrency

    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: ExecutionContext,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """选择可并行执行的节点"""
        # 确定最大并发数
        actual_max = min(max_concurrency or self.max_concurrency, self.max_concurrency)

        # 过滤可执行的节点
        executable_nodes = [
            node for node in ready_nodes if self.can_execute(node, context)
        ]

        if not executable_nodes:
            return SchedulingDecision(
                selected_nodes=[],
                execution_order=[],
                concurrency_level=0,
                strategy=SchedulingStrategy.PARALLEL,
                metadata={"reason": "no_executable_nodes"},
            )

        # 分组可并行执行的节点
        parallel_groups = self._find_parallel_groups(executable_nodes)

        # 选择第一组（尽可能多地并行执行）
        selected_group = parallel_groups[0] if parallel_groups else executable_nodes[:1]

        # 限制并发数
        selected_nodes = selected_group[:actual_max]

        # 记录批量大小
        self._stats["batch_sizes"].append(len(selected_nodes))

        return SchedulingDecision(
            selected_nodes=selected_nodes,
            execution_order=[getattr(node, "id", "unknown") for node in selected_nodes],
            concurrency_level=len(selected_nodes),
            strategy=SchedulingStrategy.PARALLEL,
            metadata={
                "total_ready": len(ready_nodes),
                "executable": len(executable_nodes),
                "parallel_groups": len(parallel_groups),
                "group_size": len(selected_group),
                "max_concurrency": actual_max,
            },
        )

    def _find_parallel_groups(self, nodes: List[Any]) -> List[List[Any]]:
        """找到可并行执行的节点组"""
        groups = []

        for node in nodes:
            placed = False
            reads = self._get_node_reads(node)
            writes = self._get_node_writes(node)

            for group in groups:
                # 检查是否可以加入该组
                can_join = True

                for member in group:
                    member_reads = self._get_node_reads(member)
                    member_writes = self._get_node_writes(member)

                    # 检查写入冲突
                    if self._has_path_conflict(writes, member_writes):
                        can_join = False
                        break

                    # 检查读写冲突
                    if self._has_path_conflict(
                        writes, member_reads
                    ) or self._has_path_conflict(reads, member_writes):
                        can_join = False
                        break

                if can_join:
                    group.append(node)
                    placed = True
                    break

            if not placed:
                groups.append([node])

        return groups

    def _get_node_reads(self, node: Any) -> List[str]:
        """获取节点读取路径"""
        return getattr(node, "reads", [])

    def _get_node_writes(self, node: Any) -> List[str]:
        """获取节点写入路径"""
        return getattr(node, "writes", [])

    def _has_path_conflict(self, paths_a: List[str], paths_b: List[str]) -> bool:
        """检查路径是否有冲突"""
        # 简化实现：检查是否有完全相同的路径
        # 实际应该实现更复杂的路径相交检查
        set_a = set(paths_a)
        set_b = set(paths_b)
        return bool(set_a & set_b)

    def get_dependencies(self, node: Any) -> List[str]:
        """获取节点的依赖列表"""
        return getattr(node, "dependencies", [])


class PriorityScheduler(BaseScheduler):
    """
    优先级调度器

    基于节点优先级进行调度
    """

    def select_nodes(
        self,
        ready_nodes: List[Any],
        context: ExecutionContext,
        max_concurrency: Optional[int] = None,
    ) -> SchedulingDecision:
        """按优先级选择节点"""
        executable_nodes = [
            node for node in ready_nodes if self.can_execute(node, context)
        ]

        if not executable_nodes:
            return SchedulingDecision(
                selected_nodes=[],
                execution_order=[],
                concurrency_level=0,
                strategy=SchedulingStrategy.PRIORITY,
                metadata={"reason": "no_executable_nodes"},
            )

        # 按优先级排序
        sorted_nodes = sorted(
            executable_nodes,
            key=lambda n: (-getattr(n, "priority", 0), getattr(n, "id", "unknown")),
        )

        # 选择最高优先级的节点
        selected_node = sorted_nodes[0]

        self._stats["batch_sizes"].append(1)

        return SchedulingDecision(
            selected_nodes=[selected_node],
            execution_order=[getattr(selected_node, "id", "unknown")],
            concurrency_level=1,
            strategy=SchedulingStrategy.PRIORITY,
            metadata={
                "total_ready": len(ready_nodes),
                "executable": len(executable_nodes),
                "selected_priority": getattr(selected_node, "priority", 0),
            },
        )

    def get_dependencies(self, node: Any) -> List[str]:
        """获取节点的依赖列表"""
        return getattr(node, "dependencies", [])
