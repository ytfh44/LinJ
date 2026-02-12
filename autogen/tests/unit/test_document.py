"""
资源域约束验证测试

测试 validate_resource_constraints 函数及其辅助函数
"""

import pytest
from typing import Any, Dict, List, Optional, Set

from linj_autogen.core.document import (
    LinJDocument,
    Placement,
    Requirements,
    validate_resource_constraints,
    _validate_requirements,
    _validate_placement,
    _validate_resource_dependencies,
    _is_valid_resource_name,
    _check_domain_conflicts,
)
from linj_autogen.core.edges import Edge, EdgeKind
from linj_autogen.core.errors import (
    InvalidRequirements,
    InvalidPlacement,
    ResourceConstraintUnsatisfied,
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


class TestValidateRequirements:
    """测试 requirements 字段验证"""

    def test_valid_boolean_requirements(self):
        """测试有效的布尔值 requirements"""
        req = Requirements(allow_parallel=True, allow_child_units=False, require_resume=True)
        errors = _validate_requirements(req)
        assert len(errors) == 0

    def test_default_values_are_valid(self):
        """测试默认值都是有效的"""
        req = Requirements()
        errors = _validate_requirements(req)
        assert len(errors) == 0


class TestValidatePlacement:
    """测试 placement 声明验证"""

    def create_test_doc(self, node_ids: List[str]) -> LinJDocument:
        """创建测试文档"""
        nodes = [create_node_dict(node_id) for node_id in node_ids]
        return LinJDocument(
            linj_version="1.0",
            nodes=nodes,
            edges=[],
        )

    def test_valid_placement(self):
        """测试有效的 placement 声明"""
        doc = self.create_test_doc(["a", "b", "c"])
        placement = [
            Placement(target="a", domain="domain1"),
            Placement(target="b", domain="domain1"),
        ]
        errors = _validate_placement(doc, placement, [])
        assert len(errors) == 0

    def test_placement_no_conflict(self):
        """测试无冲突的 placement"""
        doc = self.create_test_doc(["a", "b"])
        placement = [
            Placement(target="a", domain="domain1"),
            Placement(target="b", domain="domain2"),
        ]
        edges = [
            create_edge(from_="a", to="b", kind=EdgeKind.DATA),
        ]
        errors = _validate_placement(doc, placement, edges)
        assert len(errors) == 0

    def test_placement_mutual_dependency_conflict(self):
        """测试相互依赖的 placement 冲突"""
        doc = self.create_test_doc(["a", "b"])
        placement = [
            Placement(target="a", domain="domain1"),
            Placement(target="b", domain="domain1"),
        ]
        # 相互依赖
        edges = [
            create_edge(from_="a", to="b", kind=EdgeKind.DATA),
            create_edge(from_="b", to="a", kind=EdgeKind.DATA),
        ]
        errors = _validate_placement(doc, placement, edges)
        assert len(errors) == 1
        assert isinstance(errors[0], InvalidPlacement)
        assert "mutual dependency" in errors[0].message.lower()


class TestValidateResourceDependencies:
    """测试 kind=resource 依赖验证"""

    def create_test_doc(self, node_ids: List[str]) -> LinJDocument:
        """创建测试文档"""
        nodes = [create_node_dict(node_id) for node_id in node_ids]
        return LinJDocument(
            linj_version="1.0",
            nodes=nodes,
            edges=[],
        )

    def test_no_resource_edges(self):
        """测试没有 resource 边的情况"""
        doc = self.create_test_doc(["a", "b"])
        edges = [
            create_edge(from_="a", to="b", kind=EdgeKind.DATA),
        ]
        errors = _validate_resource_dependencies(doc, edges)
        assert len(errors) == 0

    def test_single_resource_dependency(self):
        """测试单个 resource 依赖"""
        doc = self.create_test_doc(["a", "b"])
        edges = [
            create_edge(from_="a", to="b", kind=EdgeKind.RESOURCE, resource_name="shared_resource"),
        ]
        errors = _validate_resource_dependencies(doc, edges)
        assert len(errors) == 0

    def test_resource_dependency_with_conflict(self):
        """测试有冲突的 resource 依赖"""
        doc = self.create_test_doc(["a", "b"])
        edges = [
            create_edge(from_="a", to="b", kind=EdgeKind.RESOURCE, resource_name="shared_resource"),
            create_edge(from_="b", to="a", kind=EdgeKind.DATA),  # 相互依赖
        ]
        errors = _validate_resource_dependencies(doc, edges)
        assert len(errors) == 1
        assert isinstance(errors[0], ResourceConstraintUnsatisfied)

    def test_multiple_resources_same_nodes(self):
        """测试多个 resource 依赖同一组节点"""
        doc = self.create_test_doc(["a", "b", "c"])
        edges = [
            create_edge(from_="a", to="b", kind=EdgeKind.RESOURCE, resource_name="res1"),
            create_edge(from_="b", to="c", kind=EdgeKind.RESOURCE, resource_name="res2"),
        ]
        # a, b, c 通过 resource 链连接，应该检查是否能同域
        errors = _validate_resource_dependencies(doc, edges)
        # 没有相互依赖，应该通过
        assert len(errors) == 0


class TestIsValidResourceName:
    """测试 resource_name 有效性检查"""

    def test_valid_resource_names(self):
        """测试有效的 resource_name"""
        assert _is_valid_resource_name("my_resource") is True
        assert _is_valid_resource_name("resource123") is True
        assert _is_valid_resource_name("MyResource") is True
        assert _is_valid_resource_name("a") is True

    def test_invalid_resource_names(self):
        """测试无效的 resource_name"""
        assert _is_valid_resource_name("") is False
        assert _is_valid_resource_name("123resource") is False  # 以数字开头
        assert _is_valid_resource_name("_resource") is False  # 以下划线开头
        assert _is_valid_resource_name(None) is False
        assert _is_valid_resource_name(123) is False


class TestValidateResourceConstraintsIntegration:
    """集成测试 validate_resource_constraints 函数"""

    def create_test_doc(
        self,
        node_ids: List[str],
        edges: List[Edge],
        requirements: Optional[Requirements] = None,
        placement: Optional[List[Placement]] = None,
    ) -> LinJDocument:
        """创建测试文档"""
        nodes = [create_node_dict(node_id) for node_id in node_ids]
        return LinJDocument(
            linj_version="1.0",
            nodes=nodes,
            edges=edges,
            requirements=requirements,
            placement=placement,
        )

    def test_empty_document(self):
        """测试空文档"""
        doc = self.create_test_doc([], [])
        errors = validate_resource_constraints(doc)
        assert len(errors) == 0

    def test_valid_document(self):
        """测试有效文档"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[create_edge(from_="a", to="b", kind=EdgeKind.DATA)],
        )
        errors = validate_resource_constraints(doc)
        assert len(errors) == 0

    def test_resource_constraint_unsatisfied(self):
        """测试无法满足的资源约束"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[
                create_edge(from_="a", to="b", kind=EdgeKind.RESOURCE, resource_name="res"),
                create_edge(from_="b", to="a", kind=EdgeKind.DATA),
            ],
        )
        errors = validate_resource_constraints(doc)
        assert len(errors) == 1
        assert isinstance(errors[0], ResourceConstraintUnsatisfied)

    def test_with_available_domains(self):
        """测试可用域限制"""
        doc = self.create_test_doc(
            node_ids=["a", "b"],
            edges=[create_edge(from_="a", to="b", kind=EdgeKind.DATA)],
            placement=[Placement(target="a", domain="domain1")],
        )
        # 限制可用域
        available_domains = {"domain1"}
        errors = validate_resource_constraints(doc, available_domains=available_domains)
        assert len(errors) == 0

    def test_valid_requirements(self):
        """测试有效 requirements"""
        doc = self.create_test_doc(
            node_ids=["a"],
            edges=[],
            requirements=Requirements(allow_parallel=True),
        )
        errors = validate_resource_constraints(doc)
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
