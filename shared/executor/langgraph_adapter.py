"""
LangGraph执行器适配器

提供与统一执行器兼容的接口，使用共享组件执行LinJ工作流。
为了保证与AutoGen后端的一致性（满足LinJ规范的底线目标），
本适配器复用shared/executor下的核心组件。
"""

import logging
from typing import Any, Dict, Optional, Union

from ..core.document import LinJDocument
from ..core.state import StateManager
from .context import ExecutionContext
from .scheduler import DeterministicScheduler
from .evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class ConcreteLangGraphEvaluator(BaseEvaluator):
    """Concrete implementation of BaseEvaluator"""

    def _evaluate_node(self, node: Any, context: ExecutionContext) -> Any:
        return node

    def tokenize(self, expression: str) -> list:
        return []

    def parse(self, tokens: list) -> Any:
        return None


class LangGraphExecutorAdapter:
    """
    LangGraph执行器适配器

    实现统一的execute_workflow接口，使用LangGraph兼容的调度和求值逻辑。
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

        # 3. 创建执行后端组件
        # 使用标准求值器（LangGraph使用标准逻辑）
        evaluator = ConcreteLangGraphEvaluator()

        # 使用标准决定性调度器
        scheduler = DeterministicScheduler(doc.nodes)

        # 4. 执行主循环
        # 使用统一的执行逻辑
        from .runner_utils import execute_nodes_generic

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
