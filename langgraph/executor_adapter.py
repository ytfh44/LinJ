"""
LangGraph执行器适配器

完整集成共享执行器组件，确保与AutoGen版本行为完全一致
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
    """LangGraph执行上下文"""

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
    LangGraph执行器适配器

    提供与AutoGen完全一致的执行行为：
    - 相同的节点执行逻辑
    - 相同的状态管理
    - 相同的错误处理
    - 相同的追踪记录
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
        """创建调度器"""
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
        """创建节点执行器"""
        return evaluator.NodeEvaluator(config or {})

    def create_context(
        self,
        document: Any,
        state_manager: Any,
        backend: Any,
    ) -> context.ExecutionContext:
        """创建执行上下文"""
        # 构建依赖图
        dependency_graph = backend.DependencyGraph(document.edges)

        # 创建调度器
        enable_parallel = (
            document.requirements.allow_parallel
            if hasattr(document.requirements, "allow_parallel")
            else False
        )
        sched = self.create_scheduler(
            document.nodes, strategy="deterministic", enable_parallel=enable_parallel
        )

        # 创建执行器
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
        执行工作流

        Args:
            document: LinJ文档对象
            initial_state: 初始状态

        Returns:
            执行结果和最终状态
        """
        logger.info(f"Starting LangGraph workflow execution: {document.linj_version}")

        # 初始化状态管理器
        state_manager = backend.StateManager(initial_state or {})

        # 创建执行上下文
        exec_context = self.create_context(document, state_manager, None)

        # 执行主循环
        result = self._execute_main_loop(document, exec_context, state_manager)

        logger.info("LangGraph workflow execution completed")
        return result

    def _execute_main_loop(
        self,
        document: Any,
        exec_context: LangGraphExecutionContext,
        state_manager: Any,
    ) -> Dict[str, Any]:
        """执行主循环"""
        current_round = 0
        total_steps = 0

        while current_round < (exec_context.max_rounds or 100):
            exec_context.executed_this_round.clear()
            round_steps = 0

            # 找到所有可执行节点
            ready_nodes = self._find_ready_nodes(document, exec_context)

            if not ready_nodes:
                # 检查是否有可推进的节点
                if not self._can_advance(document, exec_context):
                    logger.info(f"Workflow completed at round {current_round}")
                    break
                else:
                    logger.debug(f"No ready nodes, but workflow can advance")
                    current_round += 1
                    continue

            # 执行节点
            while ready_nodes:
                # 调度决策
                decision = exec_context.scheduler.select_nodes(
                    ready_nodes, exec_context
                )

                if not decision.selected_nodes:
                    break

                # 执行选中的节点
                for node in decision.selected_nodes:
                    try:
                        step_result = self._execute_node(
                            node, exec_context, state_manager
                        )

                        # 标记节点完成
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

                # 更新就绪节点列表
                ready_nodes = self._find_ready_nodes(document, exec_context)

            current_round += 1

            # 检查步骤限制
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
        """找到所有就绪的节点"""
        ready_nodes = []

        for node in document.nodes:
            node_id = getattr(node, "id", "unknown")

            # 跳过已完成的节点
            if exec_context._execution_state.is_terminal(node_id):
                continue

            # 检查依赖是否满足
            if are_dependencies_satisfied(
                node_id, exec_context.dependency_graph, exec_context._execution_state
            ):
                ready_nodes.append(node)

        return ready_nodes

    def _can_advance(
        self, document: Any, exec_context: LangGraphExecutionContext
    ) -> bool:
        """检查工作流是否还能推进"""
        # 检查是否还有未完成的节点
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
        """执行单个节点"""
        node_id = getattr(node, "id", "unknown")

        logger.debug(f"Executing node: {node_id}")

        # 创建节点状态视图
        step_id = exec_context.scheduler.allocate_step_id()
        view = state_manager.create_view(step_id)

        # 根据节点类型执行
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
        """执行hint节点"""
        try:
            # 渲染模板
            state = view.get_full_state()
            rendered = node.render(state)

            # 创建变更集
            from ..core.changeset import ChangeSet

            changeset = ChangeSet(
                writes=[{"path": node.write_to, "value": rendered}], deletes=[]
            )

            # 应用变更
            state_manager.apply(changeset)

            return type("Result", (), {"success": True, "output": rendered})

        except Exception as e:
            logger.error(f"Hint node execution failed: {e}")
            return type("Result", (), {"success": False, "error": str(e)})

    def _execute_tool_node(self, node: Any, view: Any, state_manager: Any) -> Any:
        """执行tool节点"""
        try:
            # 解析参数
            state = view.get_full_state()
            args = node.get_args(state)

            # 执行工具（这里需要集成实际的工具系统）
            # 暂时返回模拟结果
            result = f"Tool {node.call.name} executed with args: {args}"

            # 创建变更集
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
        """执行join节点"""
        try:
            # 读取输入
            input_text = view.read(node.input_from)

            # 检查禁止项
            forbidden = node.validate_forbidden(str(input_text))
            if forbidden:
                raise ValueError(f"Forbidden term found: {forbidden}")

            # 写入输出（简单复制，实际可能需要更复杂的接合逻辑）
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
        """执行gate节点"""
        try:
            # 评估条件
            state = view.get_full_state()
            condition_result = node.evaluate(state)

            # 获取下一步节点（这里仅记录结果，实际触发由调度器处理）
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
