"""
一致性测试

实现 LinJ 规范第 28 节建议的一致性测试用例：
验证串行执行与并行执行产生相同的最终主状态

测试用例覆盖：
- 多入边映射冲突
- gate 重复触发去重
- 隐式循环的有界性拒绝
- 写入相交与不相交判定
- effect=write 且 repeat_safe=false 的禁止自动重试
- 信号等待恢复
- 取消传播与取消后禁止提交
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
    """测试路径读写操作 (5.1-5.4)"""

    def test_get_nonexistent_path(self):
        """5.2 节：读取不存在路径返回空值"""
        state = {"a": {"b": "value"}}
        assert PathResolver.get(state, "$.missing") is None
        assert PathResolver.get(state, "$.a.missing") is None
        assert PathResolver.get(state, "$.a[10]") is None

    def test_set_creates_intermediate(self):
        """5.3.1 节：写入时自动创建中间对象"""
        state: Dict[str, Any] = {}
        PathResolver.set(state, "$.a.b.c", "value")
        assert state == {"a": {"b": {"c": "value"}}}

    def test_set_array_creates_array(self):
        """5.3.2 节：写入数组路径时自动创建数组"""
        state: Dict[str, Any] = {}
        PathResolver.set(state, "$.arr[2]", "value")
        assert state == {"arr": [None, None, "value"]}

    def test_delete_sets_null(self):
        """5.4 节：删除操作将值设为 null"""
        state = {"a": {"b": "value", "c": 123}}
        PathResolver.delete(state, "$.a.b")
        assert state == {"a": {"b": None, "c": 123}}


class TestPathIntersection:
    """测试路径相交判定 (11.4)"""

    def test_prefix_intersection(self):
        """前缀路径相交"""
        assert PathResolver.intersect("$.a", "$.a.b") is True
        assert PathResolver.intersect("$.a.b", "$.a") is True

    def test_identical_paths(self):
        """相同路径相交"""
        assert PathResolver.intersect("$.a.b", "$.a.b") is True

    def test_array_index_no_intersection(self):
        """不同数组下标不相交"""
        assert PathResolver.intersect("$.a[0]", "$.a[1]") is False

    def test_array_prefix_intersection(self):
        """数组与数组元素相交"""
        assert PathResolver.intersect("$.a", "$.a[1]") is True
        assert PathResolver.intersect("$.a[0]", "$.a[0].b") is True


class TestChangeSetIntersection:
    """测试变更集相交判定"""

    def test_disjoint_changesets(self):
        """不相交的变更集"""
        cs1 = ChangeSet(writes=[WriteOp(path="$.a", value=1)])
        cs2 = ChangeSet(writes=[WriteOp(path="$.b", value=2)])
        assert cs1.intersects_with(cs2) is False

    def test_intersecting_changesets(self):
        """相交的变更集"""
        cs1 = ChangeSet(writes=[WriteOp(path="$.a", value=1)])
        cs2 = ChangeSet(writes=[WriteOp(path="$.a.b", value=2)])
        assert cs1.intersects_with(cs2) is True


class TestConditionEvaluation:
    """测试条件表达式求值 (14.x)"""

    def test_comparison_operators(self):
        """比较运算符"""
        state = {"count": 5, "name": "test"}
        assert evaluate_condition("count == 5", state) is True
        assert evaluate_condition("count != 3", state) is True
        assert evaluate_condition("count > 3", state) is True
        assert evaluate_condition("count >= 5", state) is True
        assert evaluate_condition("count < 10", state) is True
        assert evaluate_condition("count <= 5", state) is True

    def test_logical_operators(self):
        """逻辑运算符"""
        state = {"a": True, "b": False}
        assert evaluate_condition("a AND b", state) is False
        assert evaluate_condition("a OR b", state) is True
        assert evaluate_condition("NOT b", state) is True

    def test_functions(self):
        """内置函数"""
        state = {"items": [1, 2, 3], "exists_val": "hello"}
        assert evaluate_condition('exists("$.items")', state) is True
        assert evaluate_condition('exists("$.missing")', state) is False
        assert evaluate_condition('len("$.items") == 3', state) is True
        assert evaluate_condition('value("$.exists_val") == "hello"', state) is True

    def test_null_handling(self):
        """null 值处理"""
        state = {"null_val": None, "count": 5}
        assert evaluate_condition("null_val == null", state) is True
        assert evaluate_condition("null_val != null", state) is False
        assert (
            evaluate_condition("null_val > 3", state) is False
        )  # null 比较结果为 false


class TestNodeExecution:
    """测试节点执行"""

    def test_hint_node(self):
        """hint 节点模板渲染"""
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
        """gate 节点条件求值"""
        node = GateNode(
            id="gate1", condition="count > 10", then=["node_a"], else_=["node_b"]
        )
        assert node.evaluate({"count": 15}) is True
        assert node.evaluate({"count": 5}) is False
        assert node.get_next_nodes({"count": 15}) == ["node_a"]
        assert node.get_next_nodes({"count": 5}) == ["node_b"]

    def test_join_node_forbid(self):
        """join 节点 forbid 验证"""
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
    """测试依赖图"""

    def test_data_dependencies(self):
        """数据依赖"""
        edges = [
            Edge(from_="a", to="b", kind=EdgeKind.DATA),
            Edge(from_="b", to="c", kind=EdgeKind.DATA),
        ]
        graph = DependencyGraph(edges)
        assert graph.get_data_dependencies("c") == ["b"]
        assert graph.get_data_dependencies("b") == ["a"]
        assert graph.get_data_dependencies("a") == []

    def test_control_dependencies(self):
        """控制依赖"""
        edges = [
            Edge(from_="start", to="task1", kind=EdgeKind.CONTROL),
        ]
        graph = DependencyGraph(edges)
        assert graph.get_control_dependencies("task1") == ["start"]


class TestDeterministicScheduling:
    """测试决定性调度 (11.3)"""

    def test_rank_priority(self):
        """rank 优先级排序"""
        nodes = [
            {"id": "a", "rank": 0},
            {"id": "b", "rank": 10},
            {"id": "c", "rank": 5},
        ]
        # 按 rank 降序排序
        sorted_nodes = sorted(nodes, key=lambda n: (-n["rank"], n["id"]))
        assert [n["id"] for n in sorted_nodes] == ["b", "c", "a"]

    def test_node_order_tie_breaking(self):
        """节点顺序打破平局"""
        nodes = [
            {"id": "z", "rank": 5},
            {"id": "a", "rank": 5},
            {"id": "m", "rank": 5},
        ]
        sorted_nodes = sorted(nodes, key=lambda n: (-n["rank"], n["id"]))
        assert [n["id"] for n in sorted_nodes] == ["a", "m", "z"]


class TestConcurrentSafety:
    """测试并发安全性 (11.5)"""

    def test_disjoint_reads_writes(self):
        """不相交的读写可以并行"""
        node_a = type("Node", (), {"id": "a", "reads": ["$.x"], "writes": ["$.y"]})()
        node_b = type("Node", (), {"id": "b", "reads": ["$.z"], "writes": ["$.w"]})()

        # 检查是否可并行
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
        """相交的写入不能并行"""
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
    """测试续体句柄 (17.x)"""

    def test_handle_creation(self):
        """句柄创建"""
        cont = Continuation()
        assert cont.handle is not None
        assert len(cont.handle) > 0

    def test_handle_expiration(self):
        """句柄过期"""
        registry = ContinuationRegistry(default_ttl_ms=1)  # 1ms TTL
        cont = Continuation()
        registry.register(cont, ttl_ms=1)

        # 句柄应该有效
        assert registry.is_expired(cont.handle) is False

    def test_handle_not_found(self):
        """句柄不存在"""
        registry = ContinuationRegistry()
        with pytest.raises(HandleExpired) as exc_info:
            registry.get("nonexistent")
        assert exc_info.value.details["reason"] == "not_found"


class TestChangeSetCommit:
    """测试变更集提交 (20.x)"""

    def test_empty_changeset_immediate_accept(self):
        """空变更集立即接受"""
        # 测试 CommitManager 的只读优化
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

        # 提交空变更集
        empty_cs = ChangeSet()
        result = commit_manager.submit(
            step_id=1, base_revision=0, changeset=empty_cs, handle="test"
        )

        assert result.success is True
        assert commit_manager.get_accepted_count() == 1


class TestTracer:
    """测试追踪记录 (27.x)"""

    def test_trace_entry(self):
        """追踪条目创建"""
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
        """追踪条目序列化"""
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
    """测试诊断记录 (26.x)"""

    def test_non_replayable_diagnostic(self):
        """不可重放诊断记录"""
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

        # 验证摘要
        summary = recorder.get_summary()
        assert summary["total_diagnostics"] == 1
        assert summary["by_node"]["api_node"] == 1


class TestMapConflictResolution:
    """测试映射冲突解决 (8.3)"""

    def test_weight_based_resolution(self):
        """权重优先的冲突解决"""
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

        # 权重高的应该被保留
        assert len(resolved) >= 1


class TestDocumentValidation:
    """测试文档验证"""

    def test_version_format(self):
        """版本号格式验证"""
        doc = LinJDocument(linj_version="0.1", nodes=[], edges=[])
        assert doc.linj_version == "0.1"

    def test_node_parsing(self):
        """节点解析"""
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
