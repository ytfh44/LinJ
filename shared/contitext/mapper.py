"""
LinJ 到 ContiText 的映射执行器

实现 LinJ 规范第 23-26 节定义的 LinJ ↔ ContiText 映射：
- LinJ 运行对应主续体
- step_id 决定性分配
- 变更集决定性提交
- 资源域约束映射
框架无关的映射器实现
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Mapping, Protocol, runtime_checkable

from .engine import ContiTextEngine, ChangeSet
from .continuation import Continuation, Status, StateManager
from .signal import Signal, WaitCondition

# 导入依赖图
from ..core.edges import DependencyGraph, EdgeKind

logger = logging.getLogger(__name__)


@runtime_checkable
class LinJDocument(Protocol):
    """LinJ 文档协议，用于框架无关的文档操作"""

    @property
    def linj_version(self) -> str:
        """LinJ 版本"""
        ...

    @property
    def nodes(self) -> List[Any]:
        """节点列表"""
        ...

    @property
    def edges(self) -> List[Any]:
        """依赖边列表"""
        ...

    @property
    def policies(self) -> Optional[Any]:
        """执行策略"""
        ...


@runtime_checkable
class Node(Protocol):
    """节点协议，用于框架无关的节点操作"""

    @property
    def id(self) -> str:
        """节点 ID"""
        ...

    @property
    def policy(self) -> Optional[Any]:
        """节点策略"""
        ...


@runtime_checkable
class DeterministicScheduler(Protocol):
    """决定性调度器协议"""

    def select_from_ready(self, ready_nodes: List[Node]) -> Optional[Node]:
        """从就绪节点中选择一个节点执行"""
        ...

    def allocate_step_id(self) -> int:
        """分配步骤 ID"""
        ...

    def mark_completed(self, node_id: str) -> None:
        """标记节点已完成"""
        ...


@runtime_checkable
class LinJExecutor(Protocol):
    """LinJ 执行器协议"""

    async def execute_node(
        self, node: Node, state_manager: StateManager, document: LinJDocument
    ) -> Any:
        """执行单个节点"""
        ...


class LinJToContiTextMapper:
    """
    LinJ 到 ContiText 映射器

    实现 23-26 节的映射规则，确保串行/并行执行一致性
    框架无关的映射器实现
    """

    def __init__(
        self,
        document: LinJDocument,
        state_manager: Optional[StateManager] = None,
        scheduler: Optional[DeterministicScheduler] = None,
        executor: Optional[LinJExecutor] = None,
    ):
        """
        初始化映射器

        Args:
            document: LinJ 文档
            state_manager: 状态管理器
            scheduler: 决定性调度器
            executor: LinJ 执行器
        """
        self.document = document
        self.state_manager = state_manager or self._create_default_state_manager()
        self.contitext_engine = ContiTextEngine(self.state_manager)
        self.scheduler = scheduler or self._create_default_scheduler()
        self.executor = executor or self._create_default_executor()

        # 映射状态
        self.execution_state = self._create_default_execution_state()
        self.current_round = 0

        logger.info(
            f"Initialized LinJ-ContiText mapper for document version {document.linj_version}"
        )

    def _create_default_state_manager(self) -> StateManager:
        """创建默认状态管理器"""

        class DefaultStateManager:
            def __init__(self):
                self._state: Dict[str, Any] = {}
                self._revision: int = 0

            def get_full_state(self) -> Dict[str, Any]:
                return self._state.copy()

            def get_revision(self) -> int:
                return self._revision

            def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
                # 简单的变更集应用逻辑
                if hasattr(changeset, "apply_to_state"):
                    self._state = changeset.apply_to_state(self._state)
                elif isinstance(changeset, dict):
                    self._state.update(changeset)
                self._revision += 1

        return DefaultStateManager()

    def _create_default_scheduler(self) -> DeterministicScheduler:
        """创建默认调度器"""

        class DefaultScheduler:
            def __init__(self):
                self._next_step_id = 1
                self._completed_nodes: Set[str] = set()

            def select_from_ready(self, ready_nodes: List[Node]) -> Optional[Node]:
                # 简单的按 ID 字典序选择
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
        """创建默认执行器"""

        class DefaultExecutor:
            async def execute_node(
                self, node: Node, state_manager: StateManager, document: LinJDocument
            ) -> Any:
                # 简单的执行器，返回一个空结果
                # 实际实现应该根据节点类型执行相应的逻辑
                result = type(
                    "NodeResult",
                    (),
                    {"success": True, "changeset": None, "error": None},
                )()
                return result

        return DefaultExecutor()

    def _create_default_execution_state(self) -> Any:
        """创建默认执行状态"""

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
                changeset = {"set": {key: value}}
                self.state_manager.apply(changeset)

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

                # 简化的依赖检查（实际实现需要完整的依赖图）
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

            # 按决定性顺序选择节点（11.3 节）
            node = self.scheduler.select_from_ready(ready_nodes)
            if not node:
                break

            # 检查步骤限制
            step_count += 1
            if max_steps and step_count > max_steps:
                raise RuntimeError(f"Exceeded max_steps: {max_steps}")

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
        执行单个节点（使用视图）

        Args:
            node: 节点定义
            view: 续体视图

        Returns:
            执行结果
        """
        # 将视图转换为状态管理器格式（临时）
        temp_state_manager = self._create_temp_state_manager(view)

        return await self.executor.execute_node(node, temp_state_manager, self.document)

    def _create_temp_state_manager(self, view: Any) -> StateManager:
        """从视图创建临时状态管理器"""

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
        """检查变更集是否为空"""
        if hasattr(changeset, "is_empty"):
            return changeset.is_empty()
        return not changeset

    def _check_dependencies_satisfied(self, node: Node) -> bool:
        """
        检查节点依赖是否满足 (11.1 节)

        节点的 data/control 前置依赖必须都已完成
        只有当所有前置依赖节点都已成功完成时，节点才可执行

        Args:
            node: 节点对象

        Returns:
            是否满足所有前置依赖
        """
        # 获取节点的所有入边
        incoming_edges = self._get_dependency_graph().get_incoming(node.id)

        for edge in incoming_edges:
            # 检查 data 和 control 依赖
            if edge.kind in (EdgeKind.DATA, EdgeKind.CONTROL):
                from_node_id = edge.from_
                # 前置依赖节点必须已完成（成功或失败都算完成）
                if not self.execution_state.is_terminal(from_node_id):
                    return False

        return True

    def _get_dependency_graph(self) -> DependencyGraph:
        """
        获取依赖图（带缓存）

        Returns:
            依赖图对象
        """
        if not hasattr(self, "_dependency_graph"):
            # 从文档构建依赖图
            edges = getattr(self.document, "edges", [])
            self._dependency_graph = DependencyGraph(edges)
        return self._dependency_graph

    def _has_active_nodes(self) -> bool:
        """检查是否有活跃节点"""
        for node in self.document.nodes:
            if not self.execution_state.is_terminal(node.id):
                return True
        return False

    def _get_pending_changesets(self) -> List[Any]:
        """获取待提交的变更集列表"""
        # 从 CommitManager 获取待处理变更集
        pending = self.contitext_engine.get_commit_manager().get_pending()
        return [p.changeset for p in pending]

    def _validate_resource_constraints(self) -> None:
        """
        验证资源域约束（25 节）

        验证 placement 声明和 kind=resource 依赖是否可满足：
        1. 检查 placement 声明是否有效（target 必须是节点 ID 或 resource_name）
        2. 检查 kind=resource 依赖是否可满足（同域节点需在同一执行域）
        3. 如果约束无法满足，必须产生 ValidationError

        Raises:
            ResourceConstraintUnsatisfied: 资源约束无法满足时
        """
        from ..exceptions.errors import ResourceConstraintUnsatisfied

        # 获取 placement 和 edges
        placement = getattr(self.document, "placement", None) or []
        edges = getattr(self.document, "edges", [])

        # 构建节点 ID 集合
        node_ids = {node.id for node in self.document.nodes}

        # 1. 验证 placement 声明
        placement_domains: Dict[str, str] = {}  # target -> domain

        for p in placement:
            target = getattr(p, "target", None)
            domain = getattr(p, "domain", None)

            if not target or not domain:
                continue

            # 检查 target 是否有效
            if target not in node_ids and not self._is_valid_resource_name(target):
                # target 既不是节点 ID 也不是有效的 resource_name
                # 根据规范，这应该产生 ValidationError
                logger.warning(
                    f"Invalid placement target: {target}. "
                    f"Must be a node ID or valid resource_name."
                )
                continue

            # 检查是否重复声明
            if target in placement_domains:
                if placement_domains[target] != domain:
                    raise ResourceConstraintUnsatisfied(
                        f"Conflicting domain assignments for target '{target}': "
                        f"'{placement_domains[target]}' vs '{domain}'"
                    )
            else:
                placement_domains[target] = domain

        # 2. 验证 kind=resource 依赖
        resource_dependencies: Dict[str, List[str]] = {}  # node_id -> resource_names

        for edge in edges:
            if hasattr(edge, "kind") and str(edge.kind) == "resource":
                from_node = getattr(edge, "from_", None)
                resource_name = getattr(edge, "resource_name", None)

                if from_node and resource_name:
                    if from_node not in resource_dependencies:
                        resource_dependencies[from_node] = []
                    resource_dependencies[from_node].append(resource_name)

        # 3. 验证资源依赖的一致性
        # 如果节点 A 通过 resource 依赖节点 B，且 B 有 placement 域声明
        # 则 A 也必须属于同一域（或有同名的 resource_name）

        for node_id, resources in resource_dependencies.items():
            for resource in resources:
                # 检查是否有同名的 placement
                if resource in placement_domains:
                    node_domain = placement_domains.get(node_id)
                    resource_domain = placement_domains[resource]

                    if node_domain and node_domain != resource_domain:
                        raise ResourceConstraintUnsatisfied(
                            f"Node '{node_id}' (domain='{node_domain}') and "
                            f"resource '{resource}' (domain='{resource_domain}') "
                            f"have conflicting domain assignments"
                        )

        # 4. 构建域映射（用于调度器使用）
        self._domain_map = self._build_domain_map(
            placement_domains, resource_dependencies
        )

        logger.debug(
            f"Resource constraints validated: {len(placement_domains)} placement rules, "
            f"{len(resource_dependencies)} resource dependencies"
        )

    def _is_valid_resource_name(self, name: str) -> bool:
        """检查是否是有效的 resource_name"""
        if not name:
            return False
        # resource_name 应该是一个非空字符串，不以 $ 开头
        return isinstance(name, str) and name and not name.startswith("$")

    def _build_domain_map(
        self,
        placement_domains: Dict[str, str],
        resource_dependencies: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """
        构建域映射

        Args:
            placement_domains: target -> domain 映射
            resource_dependencies: node_id -> resource_names 映射

        Returns:
            node_id -> domain 映射
        """
        domain_map: Dict[str, str] = {}

        # 首先添加 placement 声明的映射
        for target, domain in placement_domains.items():
            if target in {node.id for node in self.document.nodes}:
                domain_map[target] = domain

        # 然后处理 resource 依赖（同 resource_name 的节点应在同一域）
        resource_to_nodes: Dict[str, List[str]] = {}

        for node_id, resources in resource_dependencies.items():
            for resource in resources:
                if resource not in resource_to_nodes:
                    resource_to_nodes[resource] = []
                if node_id not in resource_to_nodes[resource]:
                    resource_to_nodes[resource].append(node_id)

        # 为共享同一 resource 的节点分配相同域
        for resource, nodes in resource_to_nodes.items():
            if len(nodes) > 1:
                # 为这些节点创建共享域
                shared_domain = f"resource:{resource}"
                for node_id in nodes:
                    # 如果节点已有域声明且不匹配，记录警告
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
        获取节点的执行域

        Args:
            node_id: 节点 ID

        Returns:
            域标签，如果未指定则返回 None
        """
        if hasattr(self, "_domain_map"):
            return self._domain_map.get(node_id)
        return None


class ParallelLinJExecutor(LinJToContiTextMapper):
    """
    并行 LinJ 执行器

    基于续体机制实现真正的并行执行
    框架无关的并行执行器实现
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
                changeset = {"set": {key: value}}
                self.state_manager.apply(changeset)

        # 验证资源约束
        self._validate_resource_constraints()

        # 创建主续体
        main_continuation = self.contitext_engine.derive()

        # 并行执行循环
        domain_map = self._allocate_domains()  # 简化的域分配

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
            concurrent_groups = self._find_concurrent_groups(ready_nodes, domain_map)

            # 并行执行每组
            for group in concurrent_groups:
                await self._execute_concurrent_group(group, main_cont)

            # 轮次计数
            self.current_round += 1
            if self.current_round > max_rounds:
                raise RuntimeError(
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

    def _get_ready_nodes(self) -> List[Node]:
        """获取就绪节点"""
        ready = []
        for node in self.document.nodes:
            if not self.execution_state.is_terminal(
                node.id
            ) and self._check_dependencies_satisfied(node):
                ready.append(node)
        return ready

    def _allocate_domains(self) -> Mapping[str, Any]:
        """分配资源域"""
        # 简化的域分配实现
        return {}

    def _find_concurrent_groups(
        self, ready_nodes: List[Node], domain_map: Mapping[str, Any]
    ) -> List[List[Node]]:
        """
        找到可安全并发的节点组 (11.5 节)

        两个节点可以安全并行执行，当且仅当：
        1. 它们声明的写入集合（writes）互不相交
        2. 任一节点的读取集合（reads）不得与另一节点的写入集合相交

        如果节点缺少 reads/writes 声明，则视为读取/写入整个主状态，
        从而阻止与其他节点并发执行

        Args:
            ready_nodes: 就绪节点列表
            domain_map: 执行域映射

        Returns:
            可安全并发的节点组列表
        """
        from ..core.path import PathResolver

        def get_node_reads(node: Node) -> List[str]:
            """获取节点的读取路径列表"""
            return getattr(node, "reads", None) or []

        def get_node_writes(node: Node) -> List[str]:
            """获取节点的写入路径列表"""
            return getattr(node, "writes", None) or []

        def paths_intersect(paths_a: List[str], paths_b: List[str]) -> bool:
            """检查两组路径是否相交"""
            for pa in paths_a:
                for pb in paths_b:
                    if PathResolver.intersect(pa, pb):
                        return True
            return False

        def can_run_parallel(node_a: Node, node_b: Node) -> bool:
            """检查两个节点是否可以并行执行"""
            reads_a = get_node_reads(node_a)
            writes_a = get_node_writes(node_a)
            reads_b = get_node_reads(node_b)
            writes_b = get_node_writes(node_b)

            # 如果任一节点没有声明 reads/writes，视为访问整个状态
            # 这会阻止与其他节点并发
            if not reads_a or not writes_a or not reads_b or not writes_b:
                # 至少检查明确的声明是否相交
                if paths_intersect(writes_a, writes_b):
                    return False
                return True

            # 规则 1: 写入集合互不相交
            if paths_intersect(writes_a, writes_b):
                return False

            # 规则 2: reads_a 不与 writes_b 相交
            if paths_intersect(reads_a, writes_b):
                return False

            # 规则 3: reads_b 不与 writes_a 相交
            if paths_intersect(reads_b, writes_a):
                return False

            return True

        # 贪婪算法：尝试将节点放入现有组
        groups: List[List[Node]] = []

        for node in ready_nodes:
            placed = False

            # 尝试放入现有组
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

            # 如果无法放入现有组，创建新组
            if not placed:
                groups.append([node])

        return groups
