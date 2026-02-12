"""
执行域分配器测试

测试 ExecutionDomain 和 DomainAllocator 类
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
    """创建 Edge 的辅助函数（使用 from 别名）"""
    # 将 from_ 转换为 from
    if "from_" in kwargs:
        kwargs["from"] = kwargs.pop("from_")
    return Edge(**kwargs)


def create_node_dict(node_id: str) -> Dict[str, Any]:
    """创建节点字典的辅助函数（使用 ToolNode 格式）"""
    return {
        "id": node_id,
        "type": "tool",
        "call": {"name": f"tool_{node_id}"},
        "writes": [f"$.{node_id}"],
    }


class TestExecutionDomain:
    """测试 ExecutionDomain 数据类"""

    def test_create_empty_domain(self):
        """测试创建空域"""
        domain = ExecutionDomain(node_ids=set(), resource_names=set())
        assert len(domain.node_ids) == 0
        assert len(domain.resource_names) == 0
        assert domain.domain_label is None

    def test_create_domain_with_label(self):
        """测试创建带标签的域"""
        domain = ExecutionDomain(
            node_ids={"a", "b"},
            resource_names={"res1"},
            domain_label="domain1",
        )
        assert domain.node_ids == {"a", "b"}
        assert domain.resource_names == {"res1"}
        assert domain.domain_label == "domain1"


class TestDomainAllocator:
    """测试 DomainAllocator 类"""

    def create_test_doc(
        self,
        node_ids: List[str],
        edges: List[Edge],
        placement: Optional[List[Placement]] = None,
    ) -> LinJDocument:
        """创建测试文档"""
        nodes = [create_node_dict(node_id) for node_id in node_ids]
        return LinJDocument(
            linj_version="1.0",
            nodes=nodes,
            edges=edges,
            placement=placement,
        )

    def test_allocate_domains_no_constraints(self):
        """测试无约束时的域分配"""
        doc = self.create_test_doc(
            node_ids=["a", "b", "c"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.DATA),
                create_edge(from_="b", to="c", kind=EdgeKind.DATA),
            ],
        )
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)
        
        # 每个节点应该有自己的域
        assert len(domain_map) == 3
        for node_id in ["a", "b", "c"]:
            assert node_id in domain_map
            domain = domain_map[node_id]
            assert node_id in domain.node_ids

    def test_allocate_domains_with_placement(self):
        """测试带 placement 约束的域分配"""
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
        
        # a 和 b 应该在同一域
        domain_a = domain_map["a"]
        domain_b = domain_map["b"]
        assert domain_a is domain_b  # 同一个对象
        assert domain_a.node_ids == {"a", "b"}
        assert domain_a.domain_label == "domain1"
        
        # c 应该在单独域
        domain_c = domain_map["c"]
        assert domain_c is not domain_a
        assert "c" in domain_c.node_ids

    def test_allocate_domains_with_resource(self):
        """测试带 resource 依赖的域分配"""
        doc = self.create_test_doc(
            node_ids=["a", "b", "c"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.RESOURCE, resource_name="shared_res"),
                create_edge(from_="b", to="c", kind=EdgeKind.DATA),
            ],
        )
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)
        
        # a 和 b 应该在同一域（因为共享 resource）
        domain_a = domain_map["a"]
        domain_b = domain_map["b"]
        assert domain_a is domain_b
        assert "a" in domain_a.node_ids
        assert "b" in domain_a.node_ids

    def test_can_share_domain(self):
        """测试 can_share_domain 方法"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.DATA),
            ],
        )
        allocator = DomainAllocator()
        
        # 单向依赖可以同域
        assert allocator.can_share_domain("a", "b", doc.edges) is True
        assert allocator.can_share_domain("b", "a", doc.edges) is True

    def test_can_share_domain_mutual_dependency(self):
        """测试相互依赖的节点不能共享域"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.DATA),
                create_edge(from_="b", to="a", kind=EdgeKind.DATA),
            ],
        )
        allocator = DomainAllocator()
        
        # 相互依赖不能同域
        assert allocator.can_share_domain("a", "b", doc.edges) is False
        assert allocator.can_share_domain("b", "a", doc.edges) is False

    def test_can_share_domain_no_edges(self):
        """测试无边时的共享判断"""
        allocator = DomainAllocator()
        assert allocator.can_share_domain("a", "b", []) is True

    def test_allocate_domains_multiple_placements(self):
        """测试多个 placement 约束"""
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
        
        # a 和 b 同域
        assert domain_map["a"] is domain_map["b"]
        # c 和 d 同域
        assert domain_map["c"] is domain_map["d"]
        # 两个域不同
        assert domain_map["a"] is not domain_map["c"]

    def test_allocate_domains_resource_and_placement(self):
        """测试同时有 resource 和 placement 约束"""
        doc = self.create_test_doc(
            node_ids=["a", "b", "c"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.RESOURCE, resource_name="res1"),
            ],
            placement=[
                Placement(target="b", domain="domain1"),
                Placement(target="c", domain="domain1"),
            ],
        )
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)
        
        # b 和 c 应该在同一域（placement）
        assert domain_map["b"] is domain_map["c"]
        # a 和 b 应该在同一域（resource）
        assert domain_map["a"] is domain_map["b"]
        # 所以 a, b, c 都在同一域
        assert domain_map["a"] is domain_map["c"]

    def test_allocate_domains_with_limited_domains(self):
        """测试有限可用域的情况"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[],
            placement=[
                Placement(target="a", domain="domain1"),
                Placement(target="b", domain="domain1"),  # 改为同域以确保 placement 被应用
            ],
        )
        allocator = DomainAllocator(available_domains={"domain1"})
        domain_map = allocator.allocate_domains(doc)
        
        # domain1 可用，a 和 b 都在 domain1
        assert domain_map["a"].domain_label == "domain1"
        assert domain_map["b"].domain_label == "domain1"

    def test_merge_domains(self):
        """测试 _merge_domains 方法"""
        domain_map = {
            "a": ExecutionDomain(node_ids={"a"}, resource_names=set()),
            "b": ExecutionDomain(node_ids={"b"}, resource_names=set()),
            "c": ExecutionDomain(node_ids={"c"}, resource_names=set()),
        }
        allocator = DomainAllocator()
        
        # 合并 a 和 b
        result = allocator._merge_domains({"a", "b"}, domain_map)
        
        # a 和 b 应该指向同一个域
        assert result["a"] is result["b"]
        assert result["a"].node_ids == {"a", "b"}

    def test_merge_single_target(self):
        """测试合并单个目标"""
        domain_map = {
            "a": ExecutionDomain(node_ids={"a"}, resource_names=set()),
        }
        allocator = DomainAllocator()
        
        result = allocator._merge_domains({"a"}, domain_map)
        # 单个目标不应该改变
        assert result["a"].node_ids == {"a"}

    def test_allocate_domains_empty_document(self):
        """测试空文档的域分配"""
        doc = LinJDocument(linj_version="1.0", nodes=[], edges=[])
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc)
        
        assert len(domain_map) == 0


class TestDomainAllocatorIntegration:
    """域分配器集成测试"""

    def create_complex_doc(self) -> tuple[LinJDocument, List[Edge]]:
        """创建复杂测试文档"""
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
        """测试复杂场景的域分配"""
        doc, edges = self.create_complex_doc()
        allocator = DomainAllocator()
        domain_map = allocator.allocate_domains(doc, edges)
        
        # init 和 merge 应该在各自的域
        assert domain_map["init"].node_ids == {"init"}
        assert domain_map["merge"].node_ids == {"merge"}
        
        # process_a 和 process_b 应该在同一域
        assert domain_map["process_a"] is domain_map["process_b"]
        assert domain_map["process_a"].node_ids == {"process_a", "process_b"}
        assert domain_map["process_a"].domain_label == "workers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
