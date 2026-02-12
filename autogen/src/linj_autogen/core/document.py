"""
LinJ 文档模型

实现规范 4 节定义的 LinJ 文档结构
"""

import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from .nodes import Node, parse_node, NodeType
from .edges import Edge, DependencyGraph, EdgeKind
from .errors import ValidationError, InvalidRequirements, InvalidPlacement, ResourceConstraintUnsatisfied


class Policies(BaseModel):
    """
    全局策略 (10.1 节)
    """
    max_steps: Optional[int] = None
    max_rounds: Optional[int] = None
    timeout_ms: Optional[int] = None
    retry: Optional[Dict[str, Any]] = None
    max_array_length: Optional[int] = None
    max_local_state_bytes: Optional[int] = None


class Loop(BaseModel):
    """
    显式循环 (12 节)
    """
    id: str
    entry: str
    members: List[str]
    mode: str = "finite"  # finite 或 infinite
    stop_condition: Optional[str] = None
    max_rounds: Optional[int] = None


class Placement(BaseModel):
    """
    放置声明 (15.2 节)
    """
    target: str  # 节点 id 或 resource_name
    domain: str  # 执行域标签


class Requirements(BaseModel):
    """
    运行要求 (15.1 节)
    """
    allow_parallel: bool = False
    allow_child_units: bool = False
    require_resume: bool = False


class LinJDocument(BaseModel):
    """
    LinJ 文档 (4.1 节)
    
    必须包含：
    - linj_version: 版本号
    - nodes: 节点数组
    - edges: 依赖边数组
    """
    linj_version: str
    nodes: List[NodeType] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    loops: Optional[List[Loop]] = None
    policies: Optional[Policies] = None
    requirements: Optional[Requirements] = None
    placement: Optional[List[Placement]] = None
    
    @field_validator("nodes", mode="before")
    @classmethod
    def parse_nodes(cls, v):
        """解析节点列表"""
        if v is None:
            return []
        return [parse_node(node) if isinstance(node, dict) else node for node in v]
    
    @field_validator("linj_version")
    @classmethod
    def validate_version(cls, v):
        """验证版本号格式"""
        parts = v.split(".")
        if len(parts) != 2:
            raise ValidationError(f"Invalid version format: {v}. Expected: major.minor")
        try:
            int(parts[0])
            int(parts[1])
        except ValueError:
            raise ValidationError(f"Invalid version numbers: {v}")
        return v
    
    def get_major_version(self) -> int:
        """获取主版本号"""
        return int(self.linj_version.split(".")[0])
    
    def get_minor_version(self) -> int:
        """获取次版本号"""
        return int(self.linj_version.split(".")[1])
    
    def check_version_compatibility(self, supported_major: int, supported_minor: int) -> bool:
        """
        检查版本兼容性 (4.2 节)
        
        - 主版本不匹配：必须拒绝运行
        - 主版本匹配但次版本更高：可以运行，但忽略不识别的字段
        """
        doc_major = self.get_major_version()
        doc_minor = self.get_minor_version()
        
        if doc_major != supported_major:
            raise ValidationError(
                f"Version mismatch: document requires {self.linj_version}, "
                f"but runtime supports {supported_major}.{supported_minor}"
            )
        
        # 次版本更高时发出警告但不阻止运行
        return doc_minor <= supported_minor
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """根据 id 获取节点"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_node_ids(self) -> List[str]:
        """获取所有节点 id"""
        return [node.id for node in self.nodes]
    
    def build_dependency_graph(self) -> DependencyGraph:
        """构建依赖图"""
        return DependencyGraph(self.edges)
    
    def validate_references(self) -> List[str]:
        """
        验证引用有效性
        
        检查所有边引用的节点是否存在
        """
        errors = []
        node_ids = set(self.get_node_ids())
        
        for edge in self.edges:
            if edge.from_ not in node_ids:
                errors.append(f"Edge references unknown source node: {edge.from_}")
            if edge.to not in node_ids:
                errors.append(f"Edge references unknown target node: {edge.to}")
        
        return errors
    
    def validate_loop_constraints(self) -> List[str]:
        """
        验证循环约束 (11.2 节)
        
        检查隐式循环（未在 loops 中声明的循环）是否有 max_rounds
        """
        errors = []
        
        # 获取显式循环及其成员
        explicit_loops = self.loops or []
        loop_members = set()
        for loop in explicit_loops:
            loop_members.update(loop.members)
        
        # 构建依赖图
        graph = self.build_dependency_graph()
        node_ids = self.get_node_ids()
        
        # 使用 DFS 检测循环
        visited = set()
        stack = []  # 当前路径上的节点
        
        def find_cycles(u: str):
            visited.add(u)
            stack.append(u)
            
            # 获取所有类型的前置依赖（入边即为依赖）
            # 注意：在我们的依赖图中，graph.get_incoming(u) 返回的是指向 u 的边
            # 但为了检测循环，我们需要看 node a -> node b -> node a
            # 这里的 get_outgoing(u) 返回的是从 u 指出的边
            for edge in graph.get_outgoing(u):
                v = edge.to
                if v in stack:
                    # 发现循环：v 到 u 再到 v
                    cycle_nodes = stack[stack.index(v):]
                    
                    # 检查该循环是否已被某个显式循环覆盖
                    is_covered = False
                    for loop in explicit_loops:
                        # 如果循环内的所有节点都在显式循环的 members 中，则视为覆盖
                        if all(node in loop.members for node in cycle_nodes):
                            is_covered = True
                            break
                    
                    if not is_covered:
                        # 隐式循环，必须有 max_rounds
                        if not self.policies or not self.policies.max_rounds:
                            cycle_str = " -> ".join(cycle_nodes + [v])
                            errors.append(
                                f"Implicit cycle detected: {cycle_str}. "
                                f"Implicit cycles must have max_rounds policy."
                            )
                elif v not in visited:
                    find_cycles(v)
            
            stack.pop()
        
        for node_id in node_ids:
            if node_id not in visited:
                find_cycles(node_id)
        
        return errors
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinJDocument":
        """从字典创建文档"""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump(by_alias=True, exclude_none=True)


logger = logging.getLogger(__name__)


def validate_resource_constraints(
    doc: "LinJDocument",
    edges: Optional[List[Edge]] = None,
    available_domains: Optional[Set[str]] = None
) -> List[ValidationError]:
    """
    验证资源域约束是否满足（15 节、25 节）
    
    验证内容：
    1. requirements 字段是否为布尔值
    2. placement 声明的同域约束是否可满足
    3. kind=resource 依赖的节点是否可调度到同一域
    
    Args:
        doc: LinJ 文档对象
        edges: 依赖边列表（可选，默认使用 doc.edges）
        available_domains: 可用的执行域集合（可选，用于模拟测试环境）
    
    Returns:
        验证错误列表（空列表表示验证通过）
    """
    errors: List[ValidationError] = []
    
    if edges is None:
        edges = doc.edges
    
    # 1. 验证 requirements 字段（15.1 节）
    if doc.requirements:
        req_errors = _validate_requirements(doc.requirements)
        errors.extend(req_errors)
    
    # 2. 验证 placement 声明（15.2 节）
    if doc.placement:
        placement_errors = _validate_placement(doc, doc.placement, edges, available_domains)
        errors.extend(placement_errors)
    
    # 3. 验证 kind=resource 依赖（25 节）
    resource_errors = _validate_resource_dependencies(doc, edges, available_domains)
    errors.extend(resource_errors)
    
    return errors


def _validate_requirements(req: Requirements) -> List[InvalidRequirements]:
    """
    验证 requirements 字段是否为布尔值
    
    Args:
        req: Requirements 对象
        
    Returns:
        错误列表
    """
    errors: List[InvalidRequirements] = []
    
    # 检查标准字段
    standard_fields = ["allow_parallel", "allow_child_units", "require_resume"]
    for field in standard_fields:
        value = getattr(req, field, None)
        if value is not None and not isinstance(value, bool):
            errors.append(
                InvalidRequirements(
                    f"requirements.{field} must be boolean, got {type(value).__name__}",
                    details={"field": field, "value": value, "expected_type": "bool"}
                )
            )
    
    return errors


def _validate_placement(
    doc: "LinJDocument",
    placement: List[Placement],
    edges: List[Edge],
    available_domains: Optional[Set[str]] = None
) -> List[InvalidPlacement]:
    """
    验证 placement 声明的同域约束
    
    Args:
        doc: LinJ 文档对象
        placement: 放置声明列表
        edges: 依赖边列表
        available_domains: 可用的执行域集合
        
    Returns:
        错误列表
    """
    errors: List[InvalidPlacement] = []
    node_ids = set(doc.get_node_ids())
    
    # 按 domain 分组 placement 条目
    domain_targets: Dict[str, Set[str]] = {}
    
    for p in placement:
        # 验证 target 是否有效
        if p.target not in node_ids and not _is_valid_resource_name(p.target):
            errors.append(
                InvalidPlacement(
                    f"Invalid placement target: {p.target}. Must be a node id or resource_name",
                    details={"target": p.target, "domain": p.domain}
                )
            )
            continue
        
        # 收集同一 domain 的所有 target
        if p.domain not in domain_targets:
            domain_targets[p.domain] = set()
        domain_targets[p.domain].add(p.target)
    
    # 验证同一 domain 的 targets 是否都可以在同一执行域运行
    for domain, targets in domain_targets.items():
        # 检查是否有节点无法与同域其他节点共存
        conflict_info = _check_domain_conflicts(targets, edges)
        if conflict_info:
            errors.append(
                InvalidPlacement(
                    f"Placement conflict in domain '{domain}': {conflict_info}",
                    details={"domain": domain, "targets": list(targets), "conflict": conflict_info}
                )
            )
    
    return errors


def _validate_resource_dependencies(
    doc: "LinJDocument",
    edges: List[Edge],
    available_domains: Optional[Set[str]] = None
) -> List[ResourceConstraintUnsatisfied]:
    """
    验证 kind=resource 依赖的节点是否可调度到同一域（25 节）
    
    Args:
        doc: LinJ 文档对象
        edges: 依赖边列表
        available_domains: 可用的执行域集合
        
    Returns:
        错误列表
    """
    errors: List[ResourceConstraintUnsatisfied] = []
    node_ids = set(doc.get_node_ids())
    
    # 按 resource_name 分组边
    resource_edges: Dict[str, List[Edge]] = {}
    for edge in edges:
        if edge.is_resource() and edge.resource_name:
            if edge.resource_name not in resource_edges:
                resource_edges[edge.resource_name] = []
            resource_edges[edge.resource_name].append(edge)
    
    # 检查每个 resource 的依赖节点是否可共存
    for resource_name, resource_deps in resource_edges.items():
        # 收集使用同一 resource 的所有节点
        nodes_in_resource: Set[str] = set()
        for edge in resource_deps:
            if edge.from_ in node_ids:
                nodes_in_resource.add(edge.from_)
            if edge.to in node_ids:
                nodes_in_resource.add(edge.to)
        
        # 检查这些节点是否有冲突（无法在同一执行域运行）
        if len(nodes_in_resource) > 1:
            # 直接检查这些节点之间是否有相互依赖
            conflict_info = _check_resource_conflicts(nodes_in_resource, edges)
            if conflict_info:
                errors.append(
                    ResourceConstraintUnsatisfied(
                        f"Resource '{resource_name}' depends on nodes that cannot share execution domain: {conflict_info}",
                        details={"resource_name": resource_name, "nodes": list(nodes_in_resource), "conflict": conflict_info}
                    )
                )
    
    return errors


def _check_resource_conflicts(
    targets: Set[str],
    edges: List[Edge]
) -> Optional[str]:
    """
    检查 resource 依赖的节点之间是否存在冲突（相互依赖）
    
    Args:
        targets: 目标集合
        edges: 依赖边列表
        
    Returns:
        冲突描述，如果无冲突返回 None
    """
    if len(targets) <= 1:
        return None
    
    # 构建依赖图
    graph = DependencyGraph(edges)
    
    targets_list = list(targets)
    for i, target_a in enumerate(targets_list):
        for target_b in targets_list[i+1:]:
            # 检查两个目标之间是否有相互依赖（循环）
            if _has_mutual_dependency(graph, target_a, target_b):
                return f"mutual dependency between {target_a} and {target_b}"
    
    return None


def _is_valid_resource_name(name: str) -> bool:
    """
    检查是否为有效的 resource_name
    
    Args:
        name: 要检查的名称
        
    Returns:
        是否为有效的 resource_name
    """
    # resource_name 应该以字母开头，可包含字母、数字、下划线、连字符
    if not name or not isinstance(name, str):
        return False
    return len(name) > 0 and name[0].isalpha()


def _check_domain_conflicts(
    targets: Set[str],
    edges: List[Edge]
) -> Optional[str]:
    """
    检查一组 target 是否存在无法共存的冲突
    
    Args:
        targets: 目标集合（节点 id 或 resource_name）
        edges: 依赖边列表
        
    Returns:
        冲突描述，如果无冲突返回 None
    """
    if len(targets) <= 1:
        return None
    
    # 构建依赖图
    graph = DependencyGraph(edges)
    
    # 检查循环依赖（同一执行域内的循环会导致问题）
    targets_list = list(targets)
    for i, target_a in enumerate(targets_list):
        for target_b in targets_list[i+1:]:
            # 检查两个目标之间是否有相互依赖（循环）
            if _has_mutual_dependency(graph, target_a, target_b):
                return f"mutual dependency between {target_a} and {target_b}"
    
    return None


def _has_mutual_dependency(graph: DependencyGraph, node_a: str, node_b: str) -> bool:
    """
    检查两个节点之间是否存在相互依赖（循环）
    
    包括 data、control 和 resource 依赖
    
    Args:
        graph: 依赖图
        node_a: 第一个节点
        node_b: 第二个节点
        
    Returns:
        是否存在相互依赖
    """
    # 获取 node_a 的所有入边源节点（包括所有类型）
    incoming_a = [edge.from_ for edge in graph.get_incoming(node_a)]
    # 获取 node_b 的所有入边源节点（包括所有类型）
    incoming_b = [edge.from_ for edge in graph.get_incoming(node_b)]
    
    # 检查是否有相互依赖：a 依赖 b 且 b 依赖 a
    # a 依赖 b 意味着 b 在 a 的入边源中
    # b 依赖 a 意味着 a 在 b 的入边源中
    return (node_b in incoming_a) and (node_a in incoming_b)
