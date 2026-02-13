"""
LinJ Execution Consistency Test Suite

Validates that AutoGen and LangGraph backend implementations produce identical execution behavior
Ensures LinJ specification baseline goal is met: same inputs produce same final state
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
    """Base class for consistency tests"""

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Configure test logging"""
        logging.basicConfig(level=logging.DEBUG)

    def execute_both_backends(
        self, document: Dict[str, Any], initial_state: Optional[Dict[str, Any]] = None
    ):
        """Execute document using both backends and return results"""
        autogen_result = execute_linj(document, "autogen", initial_state)
        langgraph_result = execute_linj(document, "langgraph", initial_state)

        return autogen_result, langgraph_result

    def assert_states_identical(self, state1: Dict[str, Any], state2: Dict[str, Any]):
        """Assert that two states are identical"""
        assert state1 == state2, (
            f"States differ:\nAutogen: {state1}\nLangGraph: {state2}"
        )

    def assert_execution_stats_consistent(
        self, stats1: Dict[str, Any], stats2: Dict[str, Any]
    ):
        """Assert execution statistics are consistent"""
        # Check key metrics
        key_metrics = ["total_rounds", "total_steps"]

        for metric in key_metrics:
            val1 = stats1.get(metric)
            val2 = stats2.get(metric)
            assert val1 == val2, f"Metric {metric} differs: {val1} vs {val2}"


class TestBasicNodeExecution(TestConsistencyBase):
    """Basic node execution consistency tests"""

    def test_hint_node_execution(self):
        """Test hint node execution consistency"""
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

        # Verify successful execution
        assert autogen_result.success
        assert langgraph_result.success

        # Verify state consistency
        self.assert_states_identical(
            autogen_result.final_state, langgraph_result.final_state
        )

        # Verify result correctness
        expected_state = {"greeting": "Hello World!"}
        assert autogen_result.final_state == expected_state
        assert langgraph_result.final_state == expected_state

    def test_tool_node_execution(self):
        """Test tool node execution consistency"""
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
        """Test join node execution consistency"""
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
        """Test gate node execution consistency"""
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

        # Test case when condition is true
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
    """Deterministic scheduling consistency tests"""

    def test_rank_based_scheduling(self):
        """Test rank-based deterministic scheduling consistency"""
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
        """Test node order tie-breaking consistency"""
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
    """Dependency resolution consistency tests"""

    def test_data_dependency_resolution(self):
        """Test data dependency resolution consistency"""
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

        # Verify dependency resolution is correct
        expected_state = {
            "shared_data": "produced data",
            "result": "consumed: produced data",
        }
        assert autogen_result.final_state == expected_state

    def test_control_dependency_resolution(self):
        """Test control dependency resolution consistency"""
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
    """Complex workflow consistency tests"""

    def test_multi_level_workflow(self):
        """Test multi-level workflow consistency"""
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
        """Test workflow with loops consistency"""
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
    """Error handling consistency tests"""

    def test_validation_error_consistency(self):
        """Test validation error consistency"""
        invalid_document = {
            "linj_version": "0.1",
            "nodes": [
                {
                    "id": "hint1",
                    "type": "hint",
                    "template": "Hello {{name}}!",
                    "vars": {"name": {"$const": "World"}},
                    # Missing required write_to field
                }
            ],
            "edges": [],
        }

        autogen_result, langgraph_result = self.execute_both_backends(invalid_document)

        # Both backends should fail
        assert not autogen_result.success
        assert not langgraph_result.success

        # Error types should be similar
        assert autogen_result.error is not None
        assert langgraph_result.error is not None

    def test_missing_dependency_consistency(self):
        """Test missing dependency consistency"""
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

        # Behavior should be consistent: both succeed or both fail
        assert autogen_result.success == langgraph_result.success

        if autogen_result.success and langgraph_result.success:
            self.assert_states_identical(
                autogen_result.final_state, langgraph_result.final_state
            )


class TestUnifiedExecutor:
    """Unified executor tests"""

    def test_consistency_validation_function(self):
        """Test consistency validation function"""
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

        # Use validation function
        consistency_result = validate_consistency(document, iterations=2)

        assert consistency_result is not None
        assert "consistent" in consistency_result
        assert "analysis" in consistency_result
        assert "backend_results" in consistency_result

        # Both backends should have results
        assert "autogen" in consistency_result["backend_results"]
        assert "langgraph" in consistency_result["backend_results"]


if __name__ == "__main__":
    # Run consistency tests
    pytest.main([__file__, "-v"])
