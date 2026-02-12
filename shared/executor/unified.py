"""
LinJ工作流统一执行器

提供与AutoGen和LangGraph版本完全一致的执行接口
确保相同输入产生相同输出（满足LinJ规范底线目标）
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from ..core.document import LinJDocument
from ..core.state import StateManager
from ..core.changeset import ChangeSet
from ..exceptions.errors import ValidationError, ResourceConstraintUnsatisfied

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """执行配置"""

    backend: str = "autogen"  # "autogen" 或 "langgraph"
    enable_parallel: bool = False
    max_concurrency: int = 4
    max_steps: Optional[int] = None
    max_rounds: Optional[int] = None
    timeout_ms: Optional[int] = None
    trace_enabled: bool = False


@dataclass
class ExecutionResult:
    """执行结果"""

    success: bool
    final_state: Dict[str, Any]
    execution_stats: Dict[str, Any]
    error: Optional[str] = None
    trace: Optional[List[Dict[str, Any]]] = None


class LinJExecutor:
    """
    LinJ工作流执行器

    统一AutoGen和LangGraph版本的执行行为：
    - 相同的文档解析
    - 相同的状态管理
    - 相同的调度逻辑
    - 相同的错误处理
    - 相同的追踪记录
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self._setup_backend()

    def _setup_backend(self) -> None:
        """设置后端执行器"""
        if self.config.backend == "langgraph":
            try:
                # Use shared executor adapter for LangGraph
                from .langgraph_adapter import LangGraphExecutorAdapter

                self._adapter = LangGraphExecutorAdapter(
                    {
                        "max_concurrency": self.config.max_concurrency,
                        "trace_enabled": self.config.trace_enabled,
                    }
                )
                logger.info("Using LangGraph backend")
            except ImportError as e:
                logger.error(f"Failed to import LangGraph backend: {e}")
                raise
        elif self.config.backend == "autogen":
            try:
                # Use shared executor adapter for AutoGen
                from .autogen_adapter import AutoGenExecutorAdapter

                self._adapter = AutoGenExecutorAdapter(
                    {
                        "max_concurrency": self.config.max_concurrency,
                        "trace_enabled": self.config.trace_enabled,
                    }
                )
                logger.info("Using AutoGen backend")
            except ImportError as e:
                logger.error(f"Failed to import AutoGen backend: {e}")
                raise
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def execute(
        self,
        document: Union[Dict[str, Any], LinJDocument],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        执行LinJ工作流

        Args:
            document: LinJ文档（字典或LinJDocument对象）
            initial_state: 初始状态

        Returns:
            执行结果

        Raises:
            ValidationError: 文档验证失败
            ResourceConstraintUnsatisfied: 资源约束不满足
        """
        logger.info(f"Executing LinJ workflow with {self.config.backend} backend")

        try:
            # 解析文档
            if isinstance(document, dict):
                try:
                    doc = LinJDocument.model_validate(document)
                except Exception as e:
                    raise ValidationError(f"Invalid LinJ document: {e}")
            else:
                doc = document

            # 验证文档
            self._validate_document(doc)

            # 应用配置覆盖
            doc = self._apply_config_overrides(doc)

            # 执行工作流
            backend_result = self._adapter.execute_workflow(doc, initial_state)

            return ExecutionResult(
                success=True,
                final_state=backend_result["final_state"],
                execution_stats=backend_result["execution_stats"],
                trace=backend_result.get("trace"),
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return ExecutionResult(
                success=False,
                final_state=initial_state or {},
                execution_stats={},
                error=str(e),
            )

    def _validate_document(self, doc: LinJDocument) -> None:
        """验证LinJ文档"""
        # 版本验证
        major_version = doc.get_major_version()
        if major_version != 0:  # 假设我们实现的是0.x版本
            raise ValidationError(f"Unsupported major version: {major_version}")

        # 节点ID唯一性检查
        node_ids = [node.id for node in doc.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValidationError("Duplicate node IDs found")

        # 依赖有效性检查
        node_id_set = set(node_ids)
        for edge in doc.edges:
            if edge.from_ not in node_id_set:
                raise ValidationError(f"Unknown source node: {edge.from_}")
            if edge.to not in node_id_set:
                raise ValidationError(f"Unknown target node: {edge.to}")

        # 策略验证
        if (
            doc.requirements
            and doc.requirements.allow_parallel
            and not self.config.enable_parallel
        ):
            logger.warning("Document requires parallel but config disables it")

        # 循环验证
        if doc.loops:
            for loop in doc.loops:
                if (
                    loop.mode == "finite"
                    and not loop.stop_condition
                    and not loop.max_rounds
                ):
                    raise ValidationError(
                        f"Finite loop {loop.id} missing stop condition or max_rounds"
                    )

    def _apply_config_overrides(self, doc: LinJDocument) -> LinJDocument:
        """应用配置覆盖到文档"""
        # 创建文档副本以避免修改原对象
        doc_dict = doc.model_dump()

        # 覆盖策略
        if not doc_dict.get("policies"):
            doc_dict["policies"] = {}

        policies = doc_dict["policies"]
        if self.config.max_steps:
            policies["max_steps"] = self.config.max_steps
        if self.config.max_rounds:
            policies["max_rounds"] = self.config.max_rounds
        if self.config.timeout_ms:
            policies["timeout_ms"] = self.config.timeout_ms

        # 覆盖需求
        if not doc_dict.get("requirements"):
            doc_dict["requirements"] = {}

        requirements = doc_dict["requirements"]
        requirements["allow_parallel"] = self.config.enable_parallel

        # 重建文档对象
        return LinJDocument.model_validate(doc_dict)

    def validate_consistency(
        self,
        document: Union[Dict[str, Any], LinJDocument],
        initial_state: Optional[Dict[str, Any]] = None,
        iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        验证两个后端的执行一致性

        Args:
            document: LinJ文档
            initial_state: 初始状态
            iterations: 测试迭代次数

        Returns:
            一致性验证结果
        """
        logger.info("Validating execution consistency between backends")

        results = {}

        for backend in ["autogen", "langgraph"]:
            backend_results = []

            for i in range(iterations):
                # 创建后端特定配置
                config = ExecutionConfig(
                    backend=backend,
                    enable_parallel=self.config.enable_parallel,
                    max_concurrency=self.config.max_concurrency,
                    trace_enabled=True,  # 启用追踪以获得详细信息
                )

                executor = LinJExecutor(config)
                result = executor.execute(document, initial_state)
                backend_results.append(result)

            results[backend] = backend_results

        # 分析一致性
        autogen_states = [r.final_state for r in results["autogen"] if r.success]
        langgraph_states = [r.final_state for r in results["langgraph"] if r.success]

        consistency_analysis = self._analyze_state_consistency(
            autogen_states, langgraph_states
        )

        return {
            "consistent": consistency_analysis["all_identical"],
            "analysis": consistency_analysis,
            "backend_results": results,
            "document_version": (
                getattr(document, "linj_version", "unknown")
                if hasattr(document, "linj_version")
                else document.get("linj_version", "unknown")
            ),
        }

    def _analyze_state_consistency(
        self,
        autogen_states: List[Dict[str, Any]],
        langgraph_states: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """分析状态一致性"""
        if not autogen_states or not langgraph_states:
            return {"error": "No successful executions to compare"}

        # 检查所有执行是否产生相同状态
        autogen_consistent = all(state == autogen_states[0] for state in autogen_states)
        langgraph_consistent = all(
            state == langgraph_states[0] for state in langgraph_states
        )

        cross_backend_consistent = autogen_states[0] == langgraph_states[0]
        all_identical = (
            autogen_consistent and langgraph_consistent and cross_backend_consistent
        )

        return {
            "all_identical": all_identical,
            "autogen_consistent": autogen_consistent,
            "langgraph_consistent": langgraph_consistent,
            "cross_backend_consistent": cross_backend_consistent,
            "autogen_sample": autogen_states[0] if autogen_states else None,
            "langgraph_sample": langgraph_states[0] if langgraph_states else None,
        }


# 便利函数
def execute_linj(
    document: Union[Dict[str, Any], LinJDocument],
    backend: str = "autogen",
    initial_state: Optional[Dict[str, Any]] = None,
    **config_kwargs,
) -> ExecutionResult:
    """
    便利函数：执行LinJ工作流

    Args:
        document: LinJ文档
        backend: 后端选择（"autogen"或"langgraph"）
        initial_state: 初始状态
        **config_kwargs: 额外配置参数

    Returns:
        执行结果
    """
    config = ExecutionConfig(backend=backend, **config_kwargs)
    executor = LinJExecutor(config)
    return executor.execute(document, initial_state)


def validate_consistency(
    document: Union[Dict[str, Any], LinJDocument],
    initial_state: Optional[Dict[str, Any]] = None,
    iterations: int = 3,
    **config_kwargs,
) -> Dict[str, Any]:
    """
    便利函数：验证执行一致性

    Args:
        document: LinJ文档
        initial_state: 初始状态
        iterations: 测试迭代次数
        **config_kwargs: 配置参数

    Returns:
        一致性验证结果
    """
    config = ExecutionConfig(**config_kwargs)
    executor = LinJExecutor(config)
    return executor.validate_consistency(document, initial_state, iterations=iterations)
