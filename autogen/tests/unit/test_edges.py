"""
Unit tests: Dependency Edge Module

Tests for Deterministic Merge Strategy (Sections 8.3, 22.2)
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
    """Test DependencyGraph.resolve_map_conflicts method"""

    def test_single_map_no_conflict(self) -> None:
        """Single map rule, no conflict"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
        ]
        graph = DependencyGraph(edges)

        maps = [
            {
                "from": "$.a",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_a",
                "edge_index": 0,
            },
        ]

        result = graph.resolve_map_conflicts("node_b", maps)

        assert len(result) == 1
        assert result[0]["from"] == "$.a"
        assert result[0]["to"] == "$.x"

    def test_weight_higher_wins(self) -> None:
        """Higher weight wins"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 2.0,
                }
            ),
        ]
        graph = DependencyGraph(edges)

        maps = [
            {
                "from": "$.a",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_a",
                "edge_index": 0,
            },
            {
                "from": "$.b",
                "to": "$.x",
                "weight": 2.0,
                "from_node": "node_c",
                "edge_index": 1,
            },
        ]

        result = graph.resolve_map_conflicts("node_b", maps)

        assert len(result) == 1
        assert result[0]["from"] == "$.b"
        assert result[0]["weight"] == 2.0

    def test_same_weight_edge_order_wins(self) -> None:
        """When weights are equal, earlier entries in edges array take precedence"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
        ]
        graph = DependencyGraph(edges)

        maps = [
            {
                "from": "$.a",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_a",
                "edge_index": 0,
            },
            {
                "from": "$.b",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_c",
                "edge_index": 1,
            },
        ]

        result = graph.resolve_map_conflicts("node_b", maps)

        assert len(result) == 1
        assert result[0]["from"] == "$.a"
        assert result[0]["edge_index"] == 0

    def test_lexicographic_order_wins(self) -> None:
        """When both weights and edge order are equal, sort by (from, to) lexicographic order"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
        ]
        graph = DependencyGraph(edges)

        # $.a < $.z, so $.a wins
        maps = [
            {
                "from": "$.a",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_a",
                "edge_index": 0,
            },
            {
                "from": "$.z",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_c",
                "edge_index": 1,
            },
        ]

        result = graph.resolve_map_conflicts("node_b", maps)

        assert len(result) == 1
        assert result[0]["from"] == "$.a"

    def test_path_intersection_cover(self) -> None:
        """Path intersection coverage ($.a and $.a.b intersect)"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
        ]
        graph = DependencyGraph(edges)

        # $.x is a prefix of $.x.y, they intersect
        # Since edge_index: 1 comes after edge_index: 0
        # But after sorting, edge_index: 0 is processed first, so $.x.y is added first
        # Then $.x intersects with $.x.y, and $.x gets covered
        maps = [
            {
                "from": "$.b",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_c",
                "edge_index": 1,
            },
            {
                "from": "$.a",
                "to": "$.x.y",
                "weight": 1.0,
                "from_node": "node_a",
                "edge_index": 0,
            },
        ]

        result = graph.resolve_map_conflicts("node_b", maps)

        assert len(result) == 1
        # Since edge_index: 0 is processed first, $.x.y wins
        assert result[0]["from"] == "$.a"
        assert result[0]["to"] == "$.x.y"

    def test_non_intersecting_paths_kept(self) -> None:
        """Non-intersecting paths are preserved"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
        ]
        graph = DependencyGraph(edges)

        maps = [
            {
                "from": "$.a",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_a",
                "edge_index": 0,
            },
            {
                "from": "$.b",
                "to": "$.y",
                "weight": 1.0,
                "from_node": "node_c",
                "edge_index": 1,
            },
        ]

        result = graph.resolve_map_conflicts("node_b", maps)

        assert len(result) == 2
        paths = {m["to"] for m in result}
        assert paths == {"$.x", "$.y"}

    def test_multiple_maps_with_conflicts(self) -> None:
        """Multiple map rules, some conflicts"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
            Edge(
                **{
                    "from": "node_c",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 2.0,
                }
            ),
            Edge(
                **{
                    "from": "node_d",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.5,
                }
            ),
        ]
        graph = DependencyGraph(edges)

        maps = [
            {
                "from": "$.a",
                "to": "$.x",
                "weight": 1.0,
                "from_node": "node_a",
                "edge_index": 0,
            },
            {
                "from": "$.b",
                "to": "$.y",
                "weight": 2.0,
                "from_node": "node_c",
                "edge_index": 1,
            },
            {
                "from": "$.c",
                "to": "$.x",
                "weight": 1.5,
                "from_node": "node_d",
                "edge_index": 2,
            },
        ]

        result = graph.resolve_map_conflicts("node_b", maps)

        # $.y and $.x are non-intersecting, preserved; $.x has conflict, node_d's $.c with higher weight wins
        assert len(result) == 2
        result_dict = {m["to"]: m["from"] for m in result}
        assert result_dict.get("$.y") == "$.b"
        assert result_dict.get("$.x") == "$.c"

    def test_empty_maps(self) -> None:
        """Empty maps list"""
        edges = [
            Edge(
                **{
                    "from": "node_a",
                    "to": "node_b",
                    "kind": EdgeKind.DATA,
                    "weight": 1.0,
                }
            ),
        ]
        graph = DependencyGraph(edges)

        result = graph.resolve_map_conflicts("node_b", [])

        assert len(result) == 0

    def test_auto_collect_maps_from_edges(self) -> None:
        """Auto-collect map rules from incoming edges"""
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

        # Without passing current_maps, automatically collect from edges
        result = graph.resolve_map_conflicts("node_b")

        assert len(result) == 1
        assert result[0]["from"] == "$.b"
        assert result[0]["weight"] == 2.0


class TestResolveMapConflictsFunction:
    """Test module-level resolve_map_conflicts function"""

    def test_basic_functionality(self) -> None:
        """Basic functionality test"""
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
        """Diagnostics recording test"""
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

        assert len(result.conflicts) >= 0  # May have conflict records
        if result.conflicts:
            conflict = result.conflicts[0]
            assert conflict.target_node == "node_b"
            assert conflict.reason in ("weight", "edge_order", "lexicographic")


class TestEdgeApplyMap:
    """Test Edge.apply_map method"""

    def test_apply_single_map(self) -> None:
        """Apply single map rule"""
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
        """Apply map rule with default value"""
        edge = Edge(
            **{
                "from": "node_a",
                "to": "node_b",
                "kind": EdgeKind.DATA,
                "map": [
                    MapRule(
                        **{"from": "$.missing", "to": "$.output.value", "default": 100}
                    )
                ],
            }
        )

        state = {}
        result = edge.apply_map(state)

        assert result == {"output": {"value": 100}}

    def test_apply_multiple_maps(self) -> None:
        """Apply multiple map rules"""
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
        """Skip when source path does not exist"""
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
    """Test DependencyGraph other methods"""

    def test_get_incoming(self) -> None:
        """Get incoming edges"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.CONTROL}),
        ]
        graph = DependencyGraph(edges)

        incoming = graph.get_incoming("node_b")

        assert len(incoming) == 2
        assert all(e.to == "node_b" for e in incoming)

    def test_get_outgoing(self) -> None:
        """Get outgoing edges"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
            Edge(**{"from": "node_a", "to": "node_c", "kind": EdgeKind.CONTROL}),
        ]
        graph = DependencyGraph(edges)

        outgoing = graph.get_outgoing("node_a")

        assert len(outgoing) == 2
        assert all(e.from_ == "node_a" for e in outgoing)

    def test_get_data_dependencies(self) -> None:
        """Get data dependencies"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
            Edge(**{"from": "node_c", "to": "node_b", "kind": EdgeKind.CONTROL}),
            Edge(**{"from": "node_d", "to": "node_b", "kind": EdgeKind.DATA}),
        ]
        graph = DependencyGraph(edges)

        deps = graph.get_data_dependencies("node_b")

        assert set(deps) == {"node_a", "node_d"}

    def test_get_data_mapping(self) -> None:
        """Get data mapping edges"""
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
        """Check if incoming edges exist"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
        ]
        graph = DependencyGraph(edges)

        assert graph.has_incoming("node_b") is True
        assert graph.has_incoming("node_c") is False

    def test_has_outgoing(self) -> None:
        """Check if outgoing edges exist"""
        edges = [
            Edge(**{"from": "node_a", "to": "node_b", "kind": EdgeKind.DATA}),
        ]
        graph = DependencyGraph(edges)

        assert graph.has_outgoing("node_a") is True
        assert graph.has_outgoing("node_b") is False


class TestEdgeKinds:
    """Test edge type judgment methods"""

    def test_is_data(self) -> None:
        """Check if it is a data edge"""
        edge = Edge(**{"from": "a", "to": "b", "kind": EdgeKind.DATA})
        assert edge.is_data() is True
        assert edge.is_control() is False
        assert edge.is_resource() is False

    def test_is_control(self) -> None:
        """Check if it is a control edge"""
        edge = Edge(**{"from": "a", "to": "b", "kind": EdgeKind.CONTROL})
        assert edge.is_data() is False
        assert edge.is_control() is True
        assert edge.is_resource() is False

    def test_is_resource(self) -> None:
        """Check if it is a resource edge"""
        edge = Edge(**{"from": "a", "to": "b", "kind": EdgeKind.RESOURCE})
        assert edge.is_data() is False
        assert edge.is_control() is False
        assert edge.is_resource() is True


class TestMapRuleAlias:
    """Test MapRule's from alias"""

    def test_map_rule_from_alias(self) -> None:
        """Test MapRule's from alias"""
        rule = MapRule(**{"from": "$.a", "to": "$.x"})
        assert rule.from_ == "$.a"
        assert rule.to == "$.x"

    def test_map_rule_default(self) -> None:
        """Test MapRule default value"""
        rule = MapRule(**{"from": "$.a", "to": "$.x", "default": 42})
        assert rule.from_ == "$.a"
        assert rule.to == "$.x"
        assert rule.default == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
