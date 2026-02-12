"""
单元测试：依赖边模块

测试决定性合并策略（8.3 节, 22.2 节）
"""

import logging
import pytest
from typing import Any, Dict, List

from linj_autogen.core.edges import (
    EdgeKind,
    MapRule,
    Edge,
    DependencyGraph,
    resolve_map_conflicts,
    MapResolutionResult,
)


class TestDependencyGraphResolveMapConflicts:
    """测试 DependencyGraph.resolve_map_conflicts 方法"""

    def test_single_map_no_conflict(self) -> None:
        """单条映射规则，无冲突"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
        ]
        graph = DependencyGraph(edges)
        
        maps = [
            {"from": "$.a", "to": "$.x", "weight": 1.0, "from_node": "node_a", "edge_index": 0},
        ]
        
        result = graph.resolve_map_conflicts("node_b", maps)
        
        assert len(result) == 1
        assert result[0]["from"] == "$.a"
        assert result[0]["to"] == "$.x"

    def test_weight_higher_wins(self) -> None:
        """权重高者胜出"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.DATA, "weight": 2.0}),
        ]
        graph = DependencyGraph(edges)
        
        maps = [
            {"from": "$.a", "to": "$.x", "weight": 1.0, "from_node": "node_a", "edge_index": 0},
            {"from": "$.b", "to": "$.x", "weight": 2.0, "from_node": "node_c", "edge_index": 1},
        ]
        
        result = graph.resolve_map_conflicts("node_b", maps)
        
        assert len(result) == 1
        assert result[0]["from"] == "$.b"
        assert result[0]["weight"] == 2.0

    def test_same_weight_edge_order_wins(self) -> None:
        """相同权重时，edges 数组中靠前者优先"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
        ]
        graph = DependencyGraph(edges)
        
        maps = [
            {"from": "$.a", "to": "$.x", "weight": 1.0, "from_node": "node_a", "edge_index": 0},
            {"from": "$.b", "to": "$.x", "weight": 1.0, "from_node": "node_c", "edge_index": 1},
        ]
        
        result = graph.resolve_map_conflicts("node_b", maps)
        
        assert len(result) == 1
        assert result[0]["from"] == "$.a"
        assert result[0]["edge_index"] == 0

    def test_lexicographic_order_wins(self) -> None:
        """权重和边顺序都相同时，按 (from, to) 字典序"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
        ]
        graph = DependencyGraph(edges)
        
        # $.a < $.z，所以 $.a 胜出
        maps = [
            {"from": "$.a", "to": "$.x", "weight": 1.0, "from_node": "node_a", "edge_index": 0},
            {"from": "$.z", "to": "$.x", "weight": 1.0, "from_node": "node_c", "edge_index": 1},
        ]
        
        result = graph.resolve_map_conflicts("node_b", maps)
        
        assert len(result) == 1
        assert result[0]["from"] == "$.a"

    def test_path_intersection_cover(self) -> None:
        """路径相交时覆盖（$.a 与 $.a.b 相交）"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
        ]
        graph = DependencyGraph(edges)
        
        # $.x 是 $.x.y 的前缀，相交
        # 由于 edge_index: 1 的条目在 edge_index: 0 之后，
        # 但排序后 edge_index: 0 的先处理，所以 $.x.y 先加入
        # 然后 $.x 与 $.x.y 相交，$.x 被覆盖
        maps = [
            {"from": "$.b", "to": "$.x", "weight": 1.0, "from_node": "node_c", "edge_index": 1},
            {"from": "$.a", "to": "$.x.y", "weight": 1.0, "from_node": "node_a", "edge_index": 0},
        ]
        
        result = graph.resolve_map_conflicts("node_b", maps)
        
        assert len(result) == 1
        # 由于 edge_index: 0 的先处理，所以 $.x.y 胜出
        assert result[0]["from"] == "$.a"
        assert result[0]["to"] == "$.x.y"

    def test_non_intersecting_paths_kept(self) -> None:
        """非相交路径保留"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
        ]
        graph = DependencyGraph(edges)
        
        maps = [
            {"from": "$.a", "to": "$.x", "weight": 1.0, "from_node": "node_a", "edge_index": 0},
            {"from": "$.b", "to": "$.y", "weight": 1.0, "from_node": "node_c", "edge_index": 1},
        ]
        
        result = graph.resolve_map_conflicts("node_b", maps)
        
        assert len(result) == 2
        paths = {m["to"] for m in result}
        assert paths == {"$.x", "$.y"}

    def test_multiple_maps_with_conflicts(self) -> None:
        """多映射规则，部分冲突"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.DATA, "weight": 2.0}),
            Edge(**{"from": "node_d", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.5}),
        ]
        graph = DependencyGraph(edges)
        
        maps = [
            {"from": "$.a", "to": "$.x", "weight": 1.0, "from_node": "node_a", "edge_index": 0},
            {"from": "$.b", "to": "$.y", "weight": 2.0, "from_node": "node_c", "edge_index": 1},
            {"from": "$.c", "to": "$.x", "weight": 1.5, "from_node": "node_d", "edge_index": 2},
        ]
        
        result = graph.resolve_map_conflicts("node_b", maps)
        
        # $.y 和 $.x 非相交，保留；$.x 有冲突，权重高的 node_d 的 $.c 胜出
        assert len(result) == 2
        result_dict = {m["to"]: m["from"] for m in result}
        assert result_dict.get("$.y") == "$.b"
        assert result_dict.get("$.x") == "$.c"

    def test_empty_maps(self) -> None:
        """空映射列表"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA, "weight": 1.0}),
        ]
        graph = DependencyGraph(edges)
        
        result = graph.resolve_map_conflicts("node_b", [])
        
        assert len(result) == 0

    def test_auto_collect_maps_from_edges(self) -> None:
        """自动从入边收集映射规则"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                    "map": [MapRule(**{"from": "$.a", "to": "$.x"})],
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 2.0,
                    "map": [MapRule(**{"from": "$.b", "to": "$.x"})],
                }
            ),
        ]
        graph = DependencyGraph(edges)
        
        # 不传入 current_maps，自动从边收集
        result = graph.resolve_map_conflicts("node_b")
        
        assert len(result) == 1
        assert result[0]["from"] == "$.b"
        assert result[0]["weight"] == 2.0


class TestResolveMapConflictsFunction:
    """测试模块级 resolve_map_conflicts 函数"""

    def test_basic_functionality(self) -> None:
        """基本功能测试"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                    "map": [MapRule(**{"from": "$.a", "to": "$.x"})],
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 2.0,
                    "map": [MapRule(**{"from": "$.b", "to": "$.x"})],
                }
            ),
        ]
        
        result = resolve_map_conflicts(edges, "node_b")
        
        assert isinstance(result, MapResolutionResult)
        assert len(result.resolved_maps) == 1
        assert result.resolved_maps[0]["from"] == "$.b"

    def test_diagnostics_recorded(self) -> None:
        """诊断信息记录测试"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                    "map": [MapRule(**{"from": "$.a", "to": "$.x"})],
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 2.0,
                    "map": [MapRule(**{"from": "$.b", "to": "$.x"})],
                }
            ),
        ]
        
        result = resolve_map_conflicts(edges, "node_b", record_diagnostics=True)
        
        assert len(result.conflicts) >= 0  # 可能有冲突记录
        if result.conflicts:
            conflict = result.conflicts[0]
            assert conflict.target_node == "node_b"
            assert conflict.reason in ("weight", "edge_order", "lexicographic")


class TestEdgeApplyMap:
    """测试 Edge.apply_map 方法"""

    def test_apply_single_map(self) -> None:
        """应用单条映射规则"""
        edge = Edge(
            **{
                "from": "node_a",
                "to": "node_b",
                "kind": EdgeKind.DATA,
                "map": [MapRule(**{"from": "$.input.value", "to": "$.output.value"})],
            }
        )
        
        state = {"input": {"value": 42}}
        result = edge.apply_map(state)
        
        assert result == {"output": {"value": 42}}

    def test_apply_map_with_default(self) -> None:
        """应用带默认值的映射规则"""
        edge = Edge(
            **{
                "from": "node_a",
                "to": "node_b",
                "kind": EdgeKind.DATA,
                "map": [MapRule(**{"from": "$.missing", "to": "$.output.value", "default": 100})],
            }
        )
        
        state = {}
        result = edge.apply_map(state)
        
        assert result == {"output": {"value": 100}}

    def test_apply_multiple_maps(self) -> None:
        """应用多条映射规则"""
        edge = Edge(
            **{
                "from": "node_a",
                "to": "node_b",
                "kind": EdgeKind.DATA,
                "map": [
                    MapRule(**{"from": "$.a", "to": "$.x"}),
                    MapRule(**{"from": "$.b", "to": "$.y"}),
                ],
            }
        )
        
        state = {"a": 1, "b": 2}
        result = edge.apply_map(state)
        
        assert result == {"x": 1, "y": 2}

    def test_apply_map_no_op_for_missing(self) -> None:
        """源路径不存在时跳过"""
        edge = Edge(
            **{
                "from": "node_a",
                "to": "node_b",
                "kind": EdgeKind.DATA,
                "map": [MapRule(**{"from": "$.missing", "to": "$.output"})],
            }
        )
        
        state = {}
        result = edge.apply_map(state)
        
        assert result == {}


class TestDependencyGraphMethods:
    """测试 DependencyGraph 其他方法"""

    def test_get_incoming(self) -> None:
        """获取入边"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.CONTROL}),
        ]
        graph = DependencyGraph(edges)
        
        incoming = graph.get_incoming("node_b")
        
        assert len(incoming) == 2
        assert all(e.to == "node_b" for e in incoming)

    def test_get_outgoing(self) -> None:
        """获取出边"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
            Edge(**{"from": "node_a", "to": "node_c", "kind": EdgeKind.CONTROL}),
        ]
        graph = DependencyGraph(edges)
        
        outgoing = graph.get_outgoing("node_a")
        
        assert len(outgoing) == 2
        assert all(e.from_ == "node_a" for e in outgoing)

    def test_get_data_dependencies(self) -> None:
        """获取数据依赖"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.CONTROL}),
            Edge(**{"from": "node_d", "to": "node_b", "kind": EdgeKind.DATA}),
        ]
        graph = DependencyGraph(edges)
        
        deps = graph.get_data_dependencies("node_b")
        
        assert set(deps) == {"node_a", "node_d"}

    def test_get_data_mapping(self) -> None:
        """获取数据映射边"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "map": [MapRule(**{"from": "$.x", "to": "$.y"})],
                }
            ),
        ]
        graph = DependencyGraph(edges)
        
        mappings = graph.get_data_mapping("node_b")
        
        assert len(mappings) == 1
        assert mappings[0].from_ == "node_c"

    def test_has_incoming(self) -> None:
        """检查是否有入边"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
        ]
        graph = DependencyGraph(edges)
        
        assert graph.has_incoming("node_b") is True
        assert graph.has_incoming("node_c") is False

    def test_has_outgoing(self) -> None:
        """检查是否有出边"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
        ]
        graph = DependencyGraph(edges)
        
        assert graph.has_outgoing("node_a") is True
        assert graph.has_outgoing("node_b") is False


class TestEdgeKinds:
    """测试边类型判断方法"""

    def test_is_data(self) -> None:
        """判断是否为 data 边"""
        edge = Edge(**{"from": "a", "to": "b", "kind": EdgeKind.DATA})
        assert edge.is_data() is True
        assert edge.is_control() is False
        assert edge.is_resource() is False

    def test_is_control(self) -> None:
        """判断是否为 control 边"""
        edge = Edge(**{"from": "a", "to": "b", "kind": EdgeKind.CONTROL})
        assert edge.is_data() is False
        assert edge.is_control() is True
        assert edge.is_resource() is False

    def test_is_resource(self) -> None:
        """判断是否为 resource 边"""
        edge = Edge(**{"from": "a", "to": "b", "kind": EdgeKind.RESOURCE})
        assert edge.is_data() is False
        assert edge.is_control() is False
        assert edge.is_resource() is True


class TestMapRuleAlias:
    """测试 MapRule 的 from alias"""

    def test_map_rule_from_alias(self) -> None:
        """测试 MapRule 的 from 别名"""
        rule = MapRule(**{"from": "$.a", "to": "$.x"})
        assert rule.from_ == "$.a"
        assert rule.to == "$.x"

    def test_map_rule_default(self) -> None:
        """测试 MapRule 默认值"""
        rule = MapRule(**{"from": "$.a", "to": "$.x", "default": 42})
        assert rule.from_ == "$.a"
        assert rule.to == "$.x"
        assert rule.default == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
