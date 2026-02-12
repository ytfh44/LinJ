"""
LinJ执行一致性测试套件

验证AutoGen和LangGraph版本的执行行为完全一致
确保满足LinJ规范底线目标：相同输入产生相同最终状态
"""

import pytest
import logging
from typing import Any, Dict, List, Optional

from shared.executor.unified import (
    LinJExecutor,
    ExecutionConfig,
    execute_linj,
    validate_consistency,
)

logger = logging.getLogger(__name__)


class TestConsistencyBase:
    """一致性测试基类"""

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """设置测试日志"""
        logging.basicConfig(level=logging.DEBUG)

    def execute_both_backends(
        self, document: Dict[str, Any], initial_state: Optional[Dict[str, Any]] = None
    ):
        """使用两个后端执行文档并返回结果"""
        autogen_result = execute_linj(document, "autogen", initial_state)
        langgraph_result = execute_linj(document, "langgraph", initial_state)

        return autogen_result, langgraph_result

    def assert_states_identical(self, state1: Dict[str, Any], state2: Dict[str, Any]):
        """断言两个状态完全相同"""
        assert state1 == state2, (
            f"States differ:\nAutogen: {state1}\nLangGraph: {state2}"
        )

    def assert_execution_stats_consistent(
        self, stats1: Dict[str, Any], stats2: Dict[str, Any]
    ):
        """断言执行统计信息一致"""
        # 检查关键统计指标
        key_metrics = ["total_rounds", "total_steps"]

        for metric in key_metrics:
            val1 = stats1.get(metric)
            val2 = stats2.get(metric)
            assert val1 == val2, f"Metric {metric} differs: {val1} vs {val2}"


class TestBasicNodeExecution(TestConsistencyBase):
    """基础节点执行一致性测试"""

    def test_hint_node_execution(self):
        """测试hint节点执行一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "hint1",
                    "type": "hint",
                    "template": "Hello {{name}}!",
                    "vars": {"name": {"$const": "World"}},
                    "write_to": "$.greeting",
                }
            ],
            "edges": [],
        }

        autogen_result, langgraph_result = self.execute_both_backends(document)

        # 验证成功执行
        assert autogen_result.success
        assert langgraph_result.success

        # 验证状态一致性
        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )

        # 验证结果正确性
        expected_state = {"greeting": "Hello World!"}
        assert autogen_result.final_state == expected_state
        assert langgraph_result.final_state == expected_state

    def test_tool_node_execution(self):
        """测试tool节点执行一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "tool1",
                    "type": "tool",
                    "call": {
                        "name": "echo",
                        "args": {"message": {"$const": "test message"}},
                    },
                    "write_to": "$.result",
                }
            ],
            "edges": [],
        }

        autogen_result, langgraph_result = self.execute_both_backends(document)

        assert autogen_result.success
        assert langgraph_result.success

        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )

    def test_join_node_execution(self):
        """测试join节点执行一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "join1",
                    "type": "join",
                    "input_from": "$.input_text",
                    "output_to": "$.output_text",
                }
            ],
            "edges": [],
        }

        initial_state = {"input_text": "sample text"}
        autogen_result, langgraph_result = self.execute_both_backends(
            document, initial_state
        )

        assert autogen_result.success
        assert langgraph_result.success

        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )

        expected_state = {"input_text": "sample text", "output_text": "sample text"}
        assert autogen_result.final_state == expected_state

    def test_gate_node_execution(self):
        """测试gate节点执行一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "gate1",
                    "type": "gate",
                    "condition": "exists($.test_value)",
                    "then": ["hint1"],
                    "else": [],
                },
                {
                    "id": "hint1",
                    "type": "hint",
                    "template": "Condition was true",
                    "write_to": "$.result",
                },
            ],
            "edges": [{"from": "gate1", "to": "hint1", "kind": "control"}],
        }

        # 测试条件为真的情况
        initial_state_true = {"test_value": True}
        autogen_result_true, langgraph_result_true = self.execute_both_backends(
            document, initial_state_true
        )

        assert autogen_result_true.success
        assert langgraph_result_true.success
        self.assert_states_identical(
            autogen_result_true.final_state, langgraph_result_true.final_state
        )


class TestDeterministicScheduling(TestConsistencyBase):
    """决定性调度一致性测试"""

    def test_rank_based_scheduling(self):
        """测试基于rank的决定性调度一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "low_priority",
                    "type": "hint",
                    "template": "Low priority",
                    "write_to": "$.low",
                    "rank": 1,
                },
                {
                    "id": "high_priority",
                    "type": "hint",
                    "template": "High priority",
                    "write_to": "$.high",
                    "rank": 10,
                },
                {
                    "id": "medium_priority",
                    "type": "hint",
                    "template": "Medium priority",
                    "write_to": "$.medium",
                    "rank": 5,
                },
            ],
            "edges": [],
        }

        autogen_result, langgraph_result = self.execute_both_backends(document)

        assert autogen_result.success
        assert langgraph_result.success

        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )
        self.assert_execution_stats_consistent(
            autogen_result.execution_stats, langgraph_result.execution_stats
        )

    def test_node_order_tie_breaking(self):
        """测试节点顺序决定性打破规则一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "node_a",
                    "type": "hint",
                    "template": "Node A",
                    "write_to": "$.a",
                },
                {
                    "id": "node_b",
                    "type": "hint",
                    "template": "Node B",
                    "write_to": "$.b",
                },
                {
                    "id": "node_c",
                    "type": "hint",
                    "template": "Node C",
                    "write_to": "$.c",
                },
            ],
            "edges": [],
        }

        autogen_result, langgraph_result = self.execute_both_backends(document)

        assert autogen_result.success
        assert langgraph_result.success

        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )
        self.assert_execution_stats_consistent(
            autogen_result.execution_stats, langgraph_result.execution_stats
        )


class TestDependencyResolution(TestConsistencyBase):
    """依赖解析一致性测试"""

    def test_data_dependency_resolution(self):
        """测试数据依赖解析一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "producer",
                    "type": "hint",
                    "template": "produced data",
                    "write_to": "$.shared_data",
                },
                {
                    "id": "consumer",
                    "type": "hint",
                    "template": "consumed: {{data}}",
                    "vars": {"data": {"$path": "$.shared_data"}},
                    "write_to": "$.result",
                },
            ],
            "edges": [{"from": "producer", "to": "consumer", "kind": "data"}],
        }

        autogen_result, langgraph_result = self.execute_both_backends(document)

        assert autogen_result.success
        assert langgraph_result.success

        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )

        # 验证依赖解析正确
        expected_state = {
            "shared_data": "produced data",
            "result": "consumed: produced data",
        }
        assert autogen_result.final_state == expected_state

    def test_control_dependency_resolution(self):
        """测试控制依赖解析一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "controller",
                    "type": "hint",
                    "template": "controller done",
                    "write_to": "$.control_signal",
                },
                {
                    "id": "controlled",
                    "type": "hint",
                    "template": "controlled executed",
                    "write_to": "$.controlled_result",
                },
            ],
            "edges": [{"from": "controller", "to": "controlled", "kind": "control"}],
        }

        autogen_result, langgraph_result = self.execute_both_backends(document)

        assert autogen_result.success
        assert langgraph_result.success

        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )


class TestComplexWorkflows(TestConsistencyBase):
    """复杂工作流一致性测试"""

    def test_multi_level_workflow(self):
        """测试多层级工作流一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "level1_a",
                    "type": "hint",
                    "template": "L1-A: {{input}}",
                    "vars": {"input": {"$const": "start"}},
                    "write_to": "$.l1_a",
                },
                {
                    "id": "level1_b",
                    "type": "hint",
                    "template": "L1-B: {{input}}",
                    "vars": {"input": {"$const": "start"}},
                    "write_to": "$.l1_b",
                },
                {
                    "id": "level2_a",
                    "type": "hint",
                    "template": "L2-A: {{input1}} + {{input2}}",
                    "vars": {
                        "input1": {"$path": "$.l1_a"},
                        "input2": {"$path": "$.l1_b"},
                    },
                    "write_to": "$.l2_a",
                },
                {
                    "id": "final",
                    "type": "join",
                    "input_from": "$.l2_a",
                    "output_to": "$.final_result",
                },
            ],
            "edges": [
                {"from": "level1_a", "to": "level2_a", "kind": "data"},
                {"from": "level1_b", "to": "level2_a", "kind": "data"},
                {"from": "level2_a", "to": "final", "kind": "data"},
            ],
        }

        autogen_result, langgraph_result = self.execute_both_backends(document)

        assert autogen_result.success
        assert langgraph_result.success

        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )
        self.assert_execution_stats_consistent(
            autogen_result.execution_stats, langgraph_result.execution_stats
        )

    def test_workflow_with_loops(self):
        """测试带循环的工作流一致性"""
        document = {
            "linj_version": "0.1",
            "loops": [
                {
                    "id": "test_loop",
                    "entry": "counter",
                    "members": ["counter", "check_limit"],
                    "mode": "finite",
                    "max_rounds": 3,
                }
            ],
            "nodes": [
                {
                    "id": "counter",
                    "type": "hint",
                    "template": "{{count}}",
                    "vars": {"count": {"$path": "$.current_count"}},
                    "write_to": "$.output",
                },
                {
                    "id": "check_limit",
                    "type": "gate",
                    "condition": "len($.outputs) < 3",
                    "then": ["counter"],
                    "else": [],
                },
            ],
            "edges": [
                {"from": "counter", "to": "check_limit", "kind": "data"},
                {"from": "check_limit", "to": "counter", "kind": "control"},
            ],
        }

        initial_state = {"current_count": 0, "outputs": []}
        autogen_result, langgraph_result = self.execute_both_backends(
            document, initial_state
        )

        assert autogen_result.success
        assert langgraph_result.success

        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )


class TestErrorHandling(TestConsistencyBase):
    """错误处理一致性测试"""

    def test_validation_error_consistency(self):
        """测试验证错误一致性"""
        invalid_document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "hint1",
                    "type": "hint",
                    "template": "Hello {{name}}!",
                    "vars": {"name": {"$const": "World"}},
                    # 缺少必需的write_to字段
                }
            ],
            "edges": [],
        }

        autogen_result, langgraph_result = self.execute_both_backends(invalid_document)

        # 两个后端都应该失败
        assert not autogen_result.success
        assert not langgraph_result.success

        # 错误类型应该相似
        assert autogen_result.error is not None
        assert langgraph_result.error is not None

    def test_missing_dependency_consistency(self):
        """测试缺失依赖一致性"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "producer",
                    "type": "hint",
                    "template": "data",
                    "write_to": "$.data",
                },
                {
                    "id": "consumer",
                    "type": "hint",
                    "template": "consumed: {{data}}",
                    "vars": {"data": {"$path": "$.missing_data"}},
                    "write_to": "$.result",
                },
            ],
            "edges": [{"from": "producer", "to": "consumer", "kind": "data"}],
        }

        autogen_result, langgraph_result = self.execute_both_backends(document)

        # 行为应该一致：要么都成功，要么都失败
        assert autogen_result.success == langgraph_result.success

        if autogen_result.success and langgraph_result.success:
            self.assert_states_identical(
                autogen_result.final_state, langgraph_result.final_state
            )


class TestUnifiedExecutor:
    """统一执行器测试"""

    def test_consistency_validation_function(self):
        """测试一致性验证函数"""
        document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "test",
                    "type": "hint",
                    "template": "Test result",
                    "write_to": "$.result",
                }
            ],
            "edges": [],
        }

        # 使用验证函数
        consistency_result = validate_consistency(document, iterations=2)

        assert consistency_result is not None
        assert "consistent" in consistency_result
        assert "analysis" in consistency_result
        assert "backend_results" in consistency_result

        # 两个后端都应该有结果
        assert "autogen" in consistency_result["backend_results"]
        assert "langgraph" in consistency_result["backend_results"]


if __name__ == "__main__":
    # 运行一致性测试
    pytest.main([__file__, "-v"])
