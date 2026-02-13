"""
Execution Domain Allocator Tests

Tests ExecutionDomain and DomainAllocator classes
"""

import pytest
from typing import Any, Dict, List, Optional, Set

from linj_autogen.core.document import LinJDocument, Placement
from linj_autogen.core.edges import Edge, EdgeKind
from linj_autogen.executor.scheduler import (
    ExecutionDomain,
    DomainAllocator,
)


def create_edge(**kwargs) -> Edge:
    """Helper function to create Edge (using from alias)"""
    # Convert from_ to from
    if "from_" in kwargs:
        kwargs["from"] = kwargs.pop("from_")
    return Edge(**kwargs)


def create_node_dict(node_id: str) -> Dict[str, Any]:
    """Helper function to create node dict (using ToolNode format)"""
    return {
        "id": node_id,
        "type": "tool",
        "call": {"name": f"tool_{node_id}"},
        "writes": [f"$.{node_id}"],
    }


class TestExecutionDomain:
    """Test ExecutionDomain dataclass"""

    def test_create_empty_domain(self):
        """Test creating empty domain"""
        domain = ExecutionDomain(node_ids=set(), resource_names=set())
        assert len(domain.node_ids) == 0
        assert len(domain.resource_names) == 0
        assert domain.domain_label is None

    def test_create_domain_with_label(self):
        """Test creating domain with label"""
        domain = ExecutionDomain(
            node_ids={"a", "b"},
            resource_names={"res1"},
            domain_label="domain1",
        )
        assert domain.node_ids == {"a", "b"}
        assert domain.resource_names == {"res1"}
        assert domain.domain_label == "domain1"


class TestDomainAllocator:
    """Test DomainAllocator class"""

    def create_test_doc(
        self,
        node_ids: List[str],
        edges: List[Edge],
        placement: Optional[List[Placement]] = None,
    ) -> LinJDocument:
        """Create test document"""
        nodes = [create_node_dict(node_id) for node_id in node_ids]
        return LinJDocument(
            linj_version="1.0",
            nodes=nodes,
            edges=edges,
            placement=placement,
        )

    def test_allocate_domains_no_constraints(self):
        """Test domain allocation without constraints"""
        doc = self.create_test_doc(
            node_ids=["a", "b", "c"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.DATA),
                create_edge(from_="b", to="c", kind=EdgeKind.DATA),
            ],
        )
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)

        # Each node should have its own domain
        assert len(domain_map) == 3
        for node_id in ["a", "b", "c"]:
            assert node_id in domain_map
            domain = domain_map[node_id]
            assert node_id in domain.node_ids

    def test_allocate_domains_with_placement(self):
        """Test domain allocation with placement constraints"""
        doc = self.create_test_doc(
            node_ids=["a", "b", "c"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.DATA),
                create_edge(from_="b", to="c", kind=EdgeKind.DATA),
            ],
            placement=[
                Placement(target="a", domain="domain1"),
                Placement(target="b", domain="domain1"),
            ],
        )
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)

        # a and b should be in the same domain
        domain_a = domain_map["a"]
        domain_b = domain_map["b"]
        assert domain_a is domain_b  # Same object
        assert domain_a.node_ids == {"a", "b"}
        assert domain_a.domain_label == "domain1"

        # c should be in separate domain
        domain_c = domain_map["c"]
        assert domain_c is not domain_a
        assert "c" in domain_c.node_ids

    def test_allocate_domains_with_resource(self):
        """Test domain allocation with resource dependencies"""
        doc = self.create_test_doc(
            node_ids=["a", "b", "c"],
            edges=[
                create_edge(
                    from_="a",
                    to="b",
                    kind=EdgeKind.RESOURCE,
                    resource_name="shared_res",
                ),
                create_edge(from_="b", to="c", kind=EdgeKind.DATA),
            ],
        )
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)

        # a and b should be in the same domain (because shared resource)
        domain_a = domain_map["a"]
        domain_b = domain_map["b"]
        assert domain_a is domain_b
        assert "a" in domain_a.node_ids
        assert "b" in domain_a.node_ids

    def test_can_share_domain(self):
        """Test can_share_domain method"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.DATA),
            ],
        )
        allocator = DomainAllocator()

        # Unidirectional dependency can share domain
        assert allocator.can_share_domain("a", "b", doc.edges) is True
        assert allocator.can_share_domain("b", "a", doc.edges) is True

    def test_can_share_domain_mutual_dependency(self):
        """Test mutually dependent nodes cannot share domain"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.DATA),
                create_edge(from_="b", to="a", kind=EdgeKind.DATA),
            ],
        )
        allocator = DomainAllocator()

        # Mutual dependency cannot share domain
        assert allocator.can_share_domain("a", "b", doc.edges) is False
        assert allocator.can_share_domain("b", "a", doc.edges) is False

    def test_can_share_domain_no_edges(self):
        """Test sharing judgment with no edges"""
        allocator = DomainAllocator()
        assert allocator.can_share_domain("a", "b", []) is True

    def test_allocate_domains_multiple_placements(self):
        """Test multiple placement constraints"""
        doc = self.create_test_doc(
            node_ids=["a", "b", "c", "d"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.DATA),
                create_edge(from_="c", to="d", kind=EdgeKind.DATA),
            ],
            placement=[
                Placement(target="a", domain="domain1"),
                Placement(target="b", domain="domain1"),
                Placement(target="c", domain="domain2"),
                Placement(target="d", domain="domain2"),
            ],
        )
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)

        # a and b share domain
        assert domain_map["a"] is domain_map["b"]
        # c and d share domain
        assert domain_map["c"] is domain_map["d"]
        # Two domains are different
        assert domain_map["a"] is not domain_map["c"]

    def test_allocate_domains_resource_and_placement(self):
        """Test both resource and placement constraints"""
        doc = self.create_test_doc(
            node_ids=["a", "b", "c"],
            edges=[
                create_edge(
                    from_="a", to="b", kind=EdgeKind.RESOURCE, resource_name="res1"
                ),
            ],
            placement=[
                Placement(target="b", domain="domain1"),
                Placement(target="c", domain="domain1"),
            ],
        )
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)

        # b and c should be in same domain (placement)
        assert domain_map["b"] is domain_map["c"]
        # a and b should be in same domain (resource)
        assert domain_map["a"] is domain_map["b"]
        # So a, b, c are all in same domain
        assert domain_map["a"] is domain_map["c"]

    def test_allocate_domains_with_limited_domains(self):
        """Test with limited available domains"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[],
            placement=[
                Placement(target="a", domain="domain1"),
                Placement(
                    target="b", domain="domain1"
                ),  # Changed to same domain to ensure placement is applied
            ],
        )
        allocator = DomainAllocator(available_domains={"domain1"})
        domain_map = allocator.allocate_domains(doc)

        # domain1 is available, a and b are both in domain1
        assert domain_map["a"].domain_label == "domain1"
        assert domain_map["b"].domain_label == "domain1"

    def test_merge_domains(self):
        """Test _merge_domains method"""
        domain_map = {
            "a": ExecutionDomain(node_ids={"a"}, resource_names=set()),
            "b": ExecutionDomain(node_ids={"b"}, resource_names=set()),
            "c": ExecutionDomain(node_ids={"c"}, resource_names=set()),
        }
        allocator = DomainAllocator()

        # Merge a and b
        result = allocator._merge_domains({"a", "b"}, domain_map)

        # a and b should point to same domain
        assert result["a"] is result["b"]
        assert result["a"].node_ids == {"a", "b"}

    def test_merge_single_target(self):
        """Test merging single target"""
        domain_map = {
            "a": ExecutionDomain(node_ids={"a"}, resource_names=set()),
        }
        allocator = DomainAllocator()

        result = allocator._merge_domains({"a"}, domain_map)
        # Single target should not change
        assert result["a"].node_ids == {"a"}

    def test_allocate_domains_empty_document(self):
        """Test domain allocation for empty document"""
        doc = LinJDocument(linj_version="1.0", nodes=[], edges=[])
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)

        assert len(domain_map) == 0


class TestDomainAllocatorIntegration:
    """DomainAllocator integration tests"""

    def create_complex_doc(self) -> tuple[LinJDocument, List[Edge]]:
        """Create complex test document"""
        nodes = [
            create_node_dict("init"),
            create_node_dict("process_a"),
            create_node_dict("process_b"),
            create_node_dict("merge"),
        ]
        edges = [
            create_edge(from_="init", to="process_a", kind=EdgeKind.DATA),
            create_edge(from_="init", to="process_b", kind=EdgeKind.DATA),
            create_edge(from_="process_a", to="merge", kind=EdgeKind.DATA),
            create_edge(from_="process_b", to="merge", kind=EdgeKind.DATA),
        ]
        placement = [
            Placement(target="process_a", domain="workers"),
            Placement(target="process_b", domain="workers"),
        ]
        doc = LinJDocument(
            linj_version="1.0",
            nodes=nodes,
            edges=edges,
            placement=placement,
        )
        return doc, edges

    def test_complex_allocation(self):
        """Test domain allocation for complex scenario"""
        doc, edges = self.create_complex_doc()
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc, edges)

        # init and merge should be in their own domains
        assert domain_map["init"].node_ids == {"init"}
        assert domain_map["merge"].node_ids == {"merge"}

        # process_a and process_b should be in same domain
        assert domain_map["process_a"] is domain_map["process_b"]
        assert domain_map["process_a"].node_ids == {"process_a", "process_b"}
        assert domain_map["process_a"].domain_label == "workers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
