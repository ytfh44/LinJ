"""
AutoGen执行器适配器

提供与统一执行器兼容的接口，使用共享组件执行LinJ工作流。
为了保证与LangGraph后端的一致性（满足LinJ规范的底线目标），
本适配器复用shared/executor下的核心组件。
"""

import logging
from typing import Any, Dict, Optional, Union

from ..core.document import LinJDocument, Policies
from ..core.state import StateManager
from .context import ExecutionContext
from .backend import BaseExecutionBackend
from .scheduler import DeterministicScheduler
from .evaluator import BaseEvaluator
from .autogen_scheduler import AutoGenDeterministicScheduler
from .autogen_evaluator import AutoGenConditionEvaluator

logger = logging.getLogger(__name__)


class ConcreteAutoGenEvaluator(AutoGenConditionEvaluator):
    """Concrete implementation of AutoGenConditionEvaluator"""

    def _evaluate_node(self, node: Any, context: ExecutionContext) -> Any:
        # Minimal implementation to satisfy abstract base class
        return node

    def tokenize(self, expression: str) -> list:
        return []

    def parse(self, tokens: list) -> Any:
        return None


class AutoGenExecutorAdapter:
    """
    AutoGen执行器适配器

    实现统一的execute_workflow接口，使用AutoGen兼容的调度和求值逻辑。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.trace_enabled = self.config.get("trace_enabled", False)
        self.max_concurrency = self.config.get("max_concurrency", 4)

    def execute_workflow(
        self,
        document: Union[Dict[str, Any], LinJDocument],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行工作流

        Args:
            document: LinJ文档
            initial_state: 初始状态

        Returns:
            执行结果字典，包含final_state, execution_stats等
        """
        # 1. 准备文档对象
        if isinstance(document, dict):
            doc = LinJDocument.model_validate(document)
        else:
            doc = document

        # 2. 初始化状态管理器
        state_manager = StateManager(initial_state or {})

        # 3. 创建执行上下文
        context = ExecutionContext()
        # 将文档和状态管理器注入上下文 (符合 shared.executor.types.ExecutionContext)
        context.document = doc
        context.state_manager = state_manager

        # 4. 创建后端组件（使用AutoGen兼容版本）
        # 使用 AutoGen 兼容的条件求值器
        evaluator = ConcreteAutoGenEvaluator(state_manager.get_full_state())

        # 使用 AutoGen 兼容的调度器
        scheduler = AutoGenDeterministicScheduler(doc.nodes)  # type: ignore

        # 5. 执行主循环
        # 使用统一的执行逻辑
        from .runner_utils import execute_nodes_generic

        # 准备后端特定的节点执行器（如果需要，目前使用通用执行器）
        # 后续可以根据具体后端需求定制 node_executor_fn

        final_state, stats = execute_nodes_generic(
            doc,
            state_manager,
            scheduler,
            evaluator,
            max_steps=(
                doc.policies.max_steps
                if doc.policies and doc.policies.max_steps
                else 100
            ),
        )

        return {
            "success": True,
            "final_state": final_state,
            "execution_stats": stats,
            "trace": [] if not self.trace_enabled else [{"type": "trace_placeholder"}],
        }
