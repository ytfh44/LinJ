"""
LinJ 到 ContiText 的映射执行器

实现 23-26 节定义的 LinJ ↔ ContiText 映射：
- LinJ 运行对应主续体
- step_id 决定性分配
- 变更集决定性提交
- 资源域约束映射
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
    LinJ 到 ContiText 映射器

    实现 23-26 节的映射规则，确保串行/并行执行一致性
    """

    def __init__(
        self, document: LinJDocument, state_manager: Optional[StateManager] = None
    ):
        """
        初始化映射器

        Args:
            document: LinJ 文档
            state_manager: 状态管理器
        """
        self.document = document
        self.state_manager = state_manager or StateManager()
        self.contitext_engine = ContiTextEngine(self.state_manager)
        self.scheduler = DeterministicScheduler(document.nodes)
        self.executor = LinJExecutor()

        # 映射状态
        self.execution_state = ExecutionState()
        self.current_round = 0

        logger.info(
            f"Initialized LinJ-ContiText mapper for document version {document.linj_version}"
        )

    async def execute(
        self, initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行 LinJ 文档（使用 ContiText）

        实现底线目标：同一文档 + 同一初始状态 = 一致的最终主状态

        Args:
            initial_state: 初始状态

        Returns:
            最终状态
        """
        # 初始化状态
        if initial_state:
            for key, value in initial_state.items():
                self.state_manager.apply(ChangeSet.create_write(f"$.{key}", value))

        # 验证资源约束（25 节）
        self._validate_resource_constraints()

        # 创建主续体（23 节）
        main_continuation = self.contitext_engine.derive()

        # 执行主循环
        await self._execution_loop(main_continuation)

        # 处理所有待提交的变更集
        self.contitext_engine.process_pending_changes()

        logger.info("LinJ-ContiText execution completed successfully")
        return self.state_manager.get_full_state()

    async def _execution_loop(self, main_cont: Continuation) -> None:
        """
        主执行循环

        使用 ContiText 续体执行 LinJ 节点
        """
        max_rounds = (
            self.document.policies.max_rounds if self.document.policies else 1000
        )
        max_steps = self.document.policies.max_steps if self.document.policies else None
        executed_this_round: Set[str] = set()
        step_count = 0

        while True:
            # 获取就绪节点（考虑 round 和 allow_reenter）
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

            # 按决定性顺序选择节点（11.3 节）
            node = self.scheduler.select_from_ready(ready_nodes, executed_this_round)
            if not node:
                break

            # 检查步骤限制
            step_count += 1
            if max_steps and step_count > max_steps:
                raise ExecutionError(f"Exceeded max_steps: {max_steps}")

            # 分配 step_id（24.3 节）
            step_id = self.scheduler.allocate_step_id()

            # 创建续体视图（18.2 节）
            view = self.contitext_engine.create_view(
                main_cont, self._get_pending_changesets()
            )

            # 执行节点（可能派生子续体）
            await self._execute_node_with_continuation(node, step_id, main_cont, view)

            executed_this_round.add(node.id)

    async def _execute_node_with_continuation(
        self, node: Node, step_id: int, parent_cont: Continuation, view: Any
    ) -> None:
        """
        使用续体执行节点

        可以选择在主续体内执行，或派生子续体执行后合流
        """
        try:
            # 简化实现：在主续体内直接执行
            # 实际实现可以根据需要派生子续体
            result = await self._execute_node(node, view)

            if result.success:
                # 提交变更集（20.2 节）
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
        执行单个节点（使用视图）

        Args:
            node: 节点定义
            view: 续体视图

        Returns:
            执行结果
        """
        # 将视图转换为状态管理器格式（临时）
        temp_state_manager = StateManager(view.get_full_state())

        return await self.executor._execute_node(
            node, temp_state_manager, self.document
        )

    def _get_ready_nodes(self) -> List[Node]:
        """获取就绪节点"""
        graph = self.document.build_dependency_graph()

        ready = []
        for node in self.document.nodes:
            if not self.execution_state.is_terminal(
                node.id
            ) and are_dependencies_satisfied(node.id, graph, self.execution_state):
                ready.append(node)

        return ready

    def _has_active_nodes(self) -> bool:
        """检查是否有活跃节点"""
        # 简化实现：检查是否还有未完成的节点
        for node in self.document.nodes:
            if not self.execution_state.is_terminal(node.id):
                return True
        return False

    def _get_pending_changesets(self) -> List[ChangeSet]:
        """获取待提交的变更集列表"""
        # 从 CommitManager 获取待处理变更集
        pending = self.contitext_engine.get_commit_manager().get_pending()
        return [p.changeset for p in pending]

    def _validate_resource_constraints(self) -> None:
        """
        验证资源域约束（25 节）

        验证 placement 声明和 kind=resource 依赖是否可满足
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
    并行 LinJ 执行器

    基于续体机制实现真正的并行执行
    """

    async def execute_parallel(
        self, initial_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        并行执行 LinJ 文档

        使用续体和并发安全判定实现并行执行，同时保证一致性

        Args:
            initial_state: 初始状态

        Returns:
            最终状态
        """
        # 初始化状态
        if initial_state:
            for key, value in initial_state.items():
                self.state_manager.apply(ChangeSet.create_write(f"$.{key}", value))

        # 验证资源约束
        self._validate_resource_constraints()

        # 创建主续体
        main_continuation = self.contitext_engine.derive()

        # 并行执行循环
        from ..executor.scheduler import DomainAllocator

        domain_allocator = DomainAllocator()
        domain_map = domain_allocator.allocate_domains(self.document)

        await self._parallel_execution_loop(main_continuation, domain_map)

        # 处理所有待提交的变更集
        self.contitext_engine.process_pending_changes()

        logger.info("Parallel LinJ-ContiText execution completed successfully")
        return self.state_manager.get_full_state()

    async def _parallel_execution_loop(
        self, main_cont: Continuation, domain_map: Mapping[str, Any]
    ) -> None:
        """
        并行执行循环

        1. 识别可安全并发的节点组（考虑数据冲突和域约束）
        2. 为每组派生续体并行执行
        3. 合流后继续下一组
        """
        from ..executor.scheduler import find_concurrent_groups

        max_rounds = (
            self.document.policies.max_rounds if self.document.policies else 1000
        )
        self.current_round = 0

        while True:
            # 获取就绪节点
            ready_nodes = self._get_ready_nodes()

            if not ready_nodes:
                if not self._has_active_nodes():
                    break
                await asyncio.sleep(0.01)
                continue

            # 分组为可安全并发的组 (11.5 节 & 25 节)
            concurrent_groups = find_concurrent_groups(ready_nodes, domain_map)

            # 并行执行每组
            for group in concurrent_groups:
                await self._execute_concurrent_group(group, main_cont)

            # 轮次计数
            self.current_round += 1
            if self.current_round > max_rounds:
                raise ExecutionError(
                    f"Exceeded max_rounds in parallel execution: {max_rounds}"
                )

    async def _execute_concurrent_group(
        self, nodes: List[Node], parent_cont: Continuation
    ) -> None:
        """
        并行执行一组节点

        Args:
            nodes: 可安全并发的节点组
            parent_cont: 父续体
        """
        # 为每个节点派生子续体
        child_continuations = []
        tasks = []

        for node in nodes:
            # 分配 step_id（决定性顺序）
            step_id = self.scheduler.allocate_step_id()

            # 派生子续体
            child_cont = self.contitext_engine.derive(parent_cont)
            child_continuations.append((node, child_cont, step_id))

            # 创建视图并启动任务
            view = self.contitext_engine.create_view(
                child_cont, self._get_pending_changesets()
            )
            task = asyncio.create_task(
                self._execute_node_with_continuation(node, step_id, child_cont, view)
            )
            tasks.append(task)

        # 等待所有任务完成（合流）
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Concurrent group execution failed: {e}")
            raise

        # 标记所有节点为已完成
        for node, child_cont, step_id in child_continuations:
            if not self.execution_state.is_terminal(node.id):
                # 根据续体状态更新执行状态
                if child_cont.status == Status.COMPLETED:
                    self.scheduler.mark_completed(node.id)
                    self.execution_state.completed.add(node.id)
                else:
                    self.scheduler.mark_completed(node.id)
                    self.execution_state.failed.add(node.id)
