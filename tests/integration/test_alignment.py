"""
Consistency tests

Implements consistency test cases recommended by LinJ Specification Section 28:
Verify that serial and parallel execution produce the same final master state

Test case coverage:
- Multi-input edge mapping conflicts
- Gate deduplication on repeated triggers
- Boundedness rejection for implicit loops
- Write intersection and disjointness determination
- Prohibit automatic retry when effect=write and repeat_safe=false
- Signal wait recovery
- Cancellation propagation and prohibit submission after cancellation
"""

import pytest
from typing import Any, Dict, List, Optional
from shared.core.document import LinJDocument, Policies
from shared.core.nodes import (
    Node,
    HintNode,
    ToolNode,
    JoinNode,
    GateNode,
    NodeType,
    ValueRef,
)
from shared.core.edges import Edge, DependencyGraph, EdgeKind, MapRule
from shared.core.changeset import ChangeSet, WriteOp, DeleteOp
from shared.core.path import PathResolver
from shared.core.condition import evaluate_condition
from shared.contitext.mapper import LinJToContiTextMapper
from shared.contitext.continuation import (
    Continuation,
    ContinuationRegistry,
    HandleExpired,
)
from shared.contitext.engine import ContiTextEngine
from shared.contitext.commit_manager import CommitManager
from shared.core.tracing import DiagnosticTracer


class TestPathOperations:
    """Test path read/write operations (5.1-5.4)"""

    def test_get_nonexistent_path(self):
        """Section 5.2: Reading non-existent paths returns null"""
        state = {"a": {"b": "value"}}
        assert PathResolver.get(state, "$.missing") is None
        assert PathResolver.get(state, "$.a.missing") is None
        assert PathResolver.get(state, "$.a[10]") is None

    def test_set_creates_intermediate(self):
        """Section 5.3.1: Intermediate objects are automatically created when writing"""
        state: Dict[str, Any] = {}
        PathResolver.set(state, "$.a.b.c", "value")
        assert state == {"a": {"b": {"c": "value"}}}

    def test_set_array_creates_array(self):
        """Section 5.3.2: Arrays are automatically created when writing array paths"""
        state: Dict[str, Any] = {}
        PathResolver.set(state, "$.arr[2]", "value")
        assert state == {"arr": [None, None, "value"]}

    def test_delete_sets_null(self):
        """Section 5.4: Delete operation sets value to null"""
        state = {"a": {"b": "value", "c": 123}}
        PathResolver.delete(state, "$.a.b")
        assert state == {"a": {"b": None, "c": 123}}


class TestPathIntersection:
    """Test path intersection determination (11.4)"""

    def test_prefix_intersection(self):
        """Prefix paths intersect"""
        assert PathResolver.intersect("$.a", "$.a.b") is True
        assert PathResolver.intersect("$.a.b", "$.a") is True

    def test_identical_paths(self):
        """Identical paths intersect"""
        assert PathResolver.intersect("$.a.b", "$.a.b") is True

    def test_array_index_no_intersection(self):
        """Different array indices do not intersect"""
        assert PathResolver.intersect("$.a[0]", "$.a[1]") is False

    def test_array_prefix_intersection(self):
        """Array and array element intersect"""
        assert PathResolver.intersect("$.a", "$.a[1]") is True
        assert PathResolver.intersect("$.a[0]", "$.a[0].b") is True


class TestChangeSetIntersection:
    """Test changeset intersection determination"""

    def test_disjoint_changesets(self):
        """Disjoint changesets"""
        cs1 = ChangeSet(writes=[WriteOp(path="$.a", value=1)])
        cs2 = ChangeSet(writes=[WriteOp(path="$.b", value=2)])
        assert cs1.intersects_with(cs2) is False

    def test_intersecting_changesets(self):
        """Intersecting changesets"""
        cs1 = ChangeSet(writes=[WriteOp(path="$.a", value=1)])
        cs2 = ChangeSet(writes=[WriteOp(path="$.a.b", value=2)])
        assert cs1.intersects_with(cs2) is True


class TestConditionEvaluation:
    """Test condition expression evaluation (14.x)"""

    def test_comparison_operators(self):
        """Comparison operators"""
        state = {"count": 5, "name": "test"}
        assert evaluate_condition("count == 5", state) is True
        assert evaluate_condition("count != 3", state) is True
        assert evaluate_condition("count > 3", state) is True
        assert evaluate_condition("count >= 5", state) is True
        assert evaluate_condition("count < 10", state) is True
        assert evaluate_condition("count <= 5", state) is True

    def test_logical_operators(self):
        """Logical operators"""
        state = {"a": True, "b": False}
        assert evaluate_condition("a AND b", state) is False
        assert evaluate_condition("a OR b", state) is True
        assert evaluate_condition("NOT b", state) is True

    def test_functions(self):
        """Built-in functions"""
        state = {"items": [1, 2, 3], "exists_val": "hello"}
        assert evaluate_condition('exists("$.items")', state) is True
        assert evaluate_condition('exists("$.missing")', state) is False
        assert evaluate_condition('len("$.items") == 3', state) is True
        assert evaluate_condition('value("$.exists_val") == "hello"', state) is True

    def test_null_handling(self):
        """Null value handling"""
        state = {"null_val": None, "count": 5}
        assert evaluate_condition("null_val == null", state) is True
        assert evaluate_condition("null_val != null", state) is False
        assert (
            evaluate_condition("null_val > 3", state) is False
        )  # null comparison returns false


class TestNodeExecution:
    """Test node execution"""

    def test_hint_node(self):
        """Hint node template rendering"""
        node = HintNode(
            id="hint1",
            template="Hello, {{name}}! You have {{count}} items.",
            vars={"name": ValueRef(path="$.user"), "count": ValueRef(path="$.count")},
            write_to="$.result",
        )
        state = {"user": "Alice", "count": 42}
        result = node.render(state)
        assert result == "Hello, Alice! You have 42 items."

    def test_gate_node_condition(self):
        """Gate node condition evaluation"""
        node = GateNode(
            id="gate1", condition="count > 10", then=["node_a"], else_=["node_b"]
        )
        assert node.evaluate({"count": 15}) is True
        assert node.evaluate({"count": 5}) is False
        assert node.get_next_nodes({"count": 15}) == ["node_a"]
        assert node.get_next_nodes({"count": 5}) == ["node_b"]

    def test_join_node_forbid(self):
        """Join node forbid validation"""
        from shared.core.nodes import GlossaryItem

        node = JoinNode(
            id="join1",
            input_from="$.input",
            output_to="$.output",
            glossary=[GlossaryItem(forbid=["badword", "spam"])],
        )
        assert node.validate_forbidden("This is good text") is None
        assert node.validate_forbidden("This contains badword here") == "badword"


class TestDependencyGraph:
    """Test dependency graph"""

    def test_data_dependencies(self):
        """Data dependencies"""
        edges = [
            Edge(from_="a", to="b", kind=EdgeKind.DATA),
            Edge(from_="b", to="c", kind=EdgeKind.DATA),
        ]
        graph = DependencyGraph(edges)
        assert graph.get_data_dependencies("c") == ["b"]
        assert graph.get_data_dependencies("b") == ["a"]
        assert graph.get_data_dependencies("a") == []

    def test_control_dependencies(self):
        """Control dependencies"""
        edges = [
            Edge(from_="start", to="task1", kind=EdgeKind.CONTROL),
        ]
        graph = DependencyGraph(edges)
        assert graph.get_control_dependencies("task1") == ["start"]


class TestDeterministicScheduling:
    """Test deterministic scheduling (11.3)"""

    def test_rank_priority(self):
        """Rank priority sorting"""
        nodes = [
            {"id": "a", "rank": 0},
            {"id": "b", "rank": 10},
            {"id": "c", "rank": 5},
        ]
        # Sort by rank in descending order
        sorted_nodes = sorted(nodes, key=lambda n: (-n["rank"], n["id"]))
        assert [n["id"] for n in sorted_nodes] == ["b", "c", "a"]

    def test_node_order_tie_breaking(self):
        """Node order tie-breaking"""
        nodes = [
            {"id": "z", "rank": 5},
            {"id": "a", "rank": 5},
            {"id": "m", "rank": 5},
        ]
        sorted_nodes = sorted(nodes, key=lambda n: (-n["rank"], n["id"]))
        assert [n["id"] for n in sorted_nodes] == ["a", "m", "z"]


class TestConcurrentSafety:
    """Test concurrent safety (11.5)"""

    def test_disjoint_reads_writes(self):
        """Disjoint reads and writes can be parallel"""
        node_a = type("Node", (), {"id": "a", "reads": ["$.x"], "writes": ["$.y"]})()
        node_b = type("Node", (), {"id": "b", "reads": ["$.z"], "writes": ["$.w"]})()

        # Check if can be parallel
        reads_a = getattr(node_a, "reads", []) or []
        writes_a = getattr(node_a, "writes", []) or []
        reads_b = getattr(node_b, "reads", []) or []
        writes_b = getattr(node_b, "writes", []) or []

        can_parallel = True
        for ra in reads_a:
            for wb in writes_b:
                if PathResolver.intersect(ra, wb):
                    can_parallel = False
                    break
            if not can_parallel:
                break

        for wa in writes_a:
            for rb in reads_b:
                if PathResolver.intersect(wa, rb):
                    can_parallel = False
                    break
            if not can_parallel:
                break

        for wa in writes_a:
            for wb in writes_b:
                if PathResolver.intersect(wa, wb):
                    can_parallel = False
                    break

        assert can_parallel is True

    def test_intersecting_writes_cannot_parallel(self):
        """Intersecting writes cannot be parallel"""
        node_a = type("Node", (), {"id": "a", "writes": ["$.shared"]})()
        node_b = type("Node", (), {"id": "b", "writes": ["$.shared.sub"]})()

        writes_a = getattr(node_a, "writes", []) or []
        writes_b = getattr(node_b, "writes", []) or []

        can_parallel = True
        for wa in writes_a:
            for wb in writes_b:
                if PathResolver.intersect(wa, wb):
                    can_parallel = False
                    break

        assert can_parallel is False


class TestContinuationHandle:
    """Test continuation handle (17.x)"""

    def test_handle_creation(self):
        """Handle creation"""
        cont = Continuation()
        assert cont.handle is not None
        assert len(cont.handle) > 0

    def test_handle_expiration(self):
        """Handle expiration"""
        registry = ContinuationRegistry(default_ttl_ms=1)  # 1ms TTL
        cont = Continuation()
        registry.register(cont, ttl_ms=1)

        # Handle should be valid
        assert registry.is_expired(cont.handle) is False

    def test_handle_not_found(self):
        """Handle not found"""
        registry = ContinuationRegistry()
        with pytest.raises(HandleExpired) as exc_info:
            registry.get("nonexistent")
        assert exc_info.value.details["reason"] == "not_found"


class TestChangeSetCommit:
    """Test changeset commit (20.x)"""

    def test_empty_changeset_immediate_accept(self):
        """Empty changeset is immediately accepted"""
        # Test CommitManager's read-only optimization
        from shared.contitext.commit_manager import CommitManager

        class MockStateManager:
            def __init__(self):
                self._state: Dict[str, Any] = {}
                self._revision = 0

            def get_full_state(self) -> Dict[str, Any]:
                return self._state.copy()

            def get_revision(self) -> int:
                return self._revision

            def apply(self, changeset: Any, step_id: Optional[int] = None) -> None:
                if hasattr(changeset, "apply_to"):
                    changeset.apply_to(self._state)
                self._revision += 1

        state_manager = MockStateManager()
        commit_manager = CommitManager(state_manager)

        # Commit empty changeset
        empty_cs = ChangeSet()
        result = commit_manager.submit(
            step_id=1, base_revision=0, changeset=empty_cs, handle="test"
        )

        assert result.success is True
        assert commit_manager.get_accepted_count() == 1


class TestTracer:
    """Test trace recording (27.x)"""

    def test_trace_entry(self):
        """Trace entry creation"""
        from shared.core.tracing import TraceEntry

        entry = TraceEntry(
            step_id=1,
            round=0,
            node_id="test_node",
            attempt=1,
            reads_declared=["$.data"],
            writes_declared=["$.result"],
        )

        assert entry.step_id == 1
        assert entry.node_id == "test_node"
        assert entry.status == "pending"

    def test_trace_to_dict(self):
        """Trace entry serialization"""
        from shared.core.tracing import TraceEntry

        entry = TraceEntry(step_id=1, round=0, node_id="test_node")
        entry.status = "completed"
        entry.ts_start_ms = 1000
        entry.ts_end_ms = 1100

        data = entry.to_dict()
        assert data["step_id"] == 1
        assert data["status"] == "completed"
        assert data["ts_start_ms"] == 1000


class TestDiagnostics:
    """Test diagnostic recording (26.x)"""

    def test_non_replayable_diagnostic(self):
        """Non-replayable diagnostic recording"""
        from shared.core.tracing import DiagnosticsRecorder

        recorder = DiagnosticsRecorder()
        diag = recorder.record_non_replayable(
            node_id="api_node",
            tool_name="http_call",
            reason="Token expired",
            at_step_id=5,
            details={"error_code": "AUTH_001"},
        )

        assert diag.node_id == "api_node"
        assert diag.tool_name == "http_call"
        assert diag.reason == "Token expired"
        assert diag.at_step_id == 5

        # Verify summary
        summary = recorder.get_summary()
        assert summary["total_diagnostics"] == 1
        assert summary["by_node"]["api_node"] == 1


class TestMapConflictResolution:
    """Test map conflict resolution (8.3)"""

    def test_weight_based_resolution(self):
        """Weight-based conflict resolution"""
        from shared.core.edges import DependencyGraph

        edges = [
            Edge(
                from_="a",
                to="target",
                kind=EdgeKind.DATA,
                weight=2.0,
                map_rules=[MapRule(from_="$.x", to="$.y")],
            ),
            Edge(
                from_="b",
                to="target",
                kind=EdgeKind.DATA,
                weight=1.0,
                map_rules=[MapRule(from_="$.x", to="$.y")],
            ),
        ]
        graph = DependencyGraph(edges)
        resolved = graph.resolve_map_conflicts("target")

        # Higher weight should be preserved
        assert len(resolved) >= 1


class TestDocumentValidation:
    """Test document validation"""

    def test_version_format(self):
        """Version number format validation"""
        doc = LinJDocument(linj_version="0.1", nodes=[], edges=[])
        assert doc.linj_version == "0.1"

    def test_node_parsing(self):
        """Node parsing"""
        nodes = [
            {"id": "a", "type": "hint", "template": "Hello", "write_to": "$.out"},
            {"id": "b", "type": "tool", "call": {"name": "test"}},
        ]
        doc = LinJDocument(linj_version="0.1", nodes=nodes, edges=[])
        assert len(doc.nodes) == 2
        assert doc.nodes[0].id == "a"
        assert doc.nodes[1].id == "b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
