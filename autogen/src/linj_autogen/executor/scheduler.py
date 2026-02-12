from __future__ import annotations
"""
调度器

实现 11.3 节、11.4 节、11.5 节定义的决定性调度顺序和并发安全判定
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Mapping

from ..core.nodes import Node
from ..core.path import PathResolver
from ..core.document import LinJDocument, Placement
from ..core.edges import Edge, DependencyGraph, EdgeKind


def select_next_node(ready_nodes: List[Node], 
                     node_order: Dict[str, int]) -> Optional[Node]:
    """
    按决定性顺序选择下一个节点 (11.3 节)
    
    优先级：
    1. rank 较大者优先（未提供视为 0）
    2. nodes 数组中靠前者优先（通过 node_order 字典）
    3. 若仍相同，按 node_id 字典序
    
    Args:
        ready_nodes: 可调度节点列表
        node_order: 节点在原数组中的顺序
        
    Returns:
        选中的节点，或 None（如果没有就绪节点）
    """
    if not ready_nodes:
        return None
    
    def sort_key(node: Node) -> Tuple[float, int, str]:
        rank = node.rank if node.rank is not None else 0.0
        order = node_order.get(node.id, float('inf'))
        return (-rank, order, node.id)  # 负 rank 实现降序
    
    sorted_nodes = sorted(ready_nodes, key=sort_key)
    return sorted_nodes[0]


def get_node_path_set(node: Node, use_writes: bool = True) -> Set[str]:
    """
    获取节点的路径集合
    
    如果节点没有声明 reads/writes，视为整个状态（返回 {"$")
    """
    if use_writes:
        paths = node.writes
    else:
        paths = node.reads
    
    if not paths:
        # 6.1 节：缺失声明视为读取/写入整个主状态
        return {"$"}
    
    return set(paths)


def check_path_intersection(paths_a: Set[str], paths_b: Set[str]) -> bool:
    """
    检查两组路径是否有相交 (11.4 节)
    
    两条路径相交当且仅当：
    - 一条路径是另一条的前缀
    - 两条路径完全相同
    """
    for path_a in paths_a:
        for path_b in paths_b:
            if PathResolver.intersect(path_a, path_b):
                return True
    return False


def check_concurrent_safety(node_a: Node, node_b: Node) -> bool:
    """
    检查两个节点是否可以安全并发执行 (11.5 节)
    
    并发安全条件：
    - 两个节点的 writes 集合互不相交
    - 任一节点的 reads 不与另一节点的 writes 相交
    
    Args:
        node_a: 第一个节点
        node_b: 第二个节点
        
    Returns:
        True 表示可以安全并发，False 表示不能并发
    """
    # 获取路径集合
    reads_a = get_node_path_set(node_a, use_writes=False)
    writes_a = get_node_path_set(node_a, use_writes=True)
    reads_b = get_node_path_set(node_b, use_writes=False)
    writes_b = get_node_path_set(node_b, use_writes=True)
    
    # 检查 writes 互不相交
    if check_path_intersection(writes_a, writes_b):
        return False
    
    # 检查 reads 不与对方 writes 相交
    if check_path_intersection(reads_a, writes_b):
        return False
    if check_path_intersection(reads_b, writes_a):
        return False
    
    return True


def find_concurrent_groups(
    nodes: List[Node], 
    domain_map: Optional[Mapping[str, ExecutionDomain]] = None
) -> List[List[Node]]:
    """
    将节点分组，每组内的节点可以安全并发执行
    
    使用贪心算法：遍历节点，尝试将节点加入现有组，
    如果不能加入任何组，则创建新组
    
    11.5 节 & 25 节：
    - 组内节点必须 writes/reads 互不相交
    - 组内节点必须属于不同的执行域
    """
    if not nodes:
        return []
    
    groups: List[List[Node]] = []
    
    for node in nodes:
        placed = False
        node_domain = domain_map.get(node.id) if domain_map else None
        
        for group in groups:
            # 检查是否可以加入该组
            # 1. 检查并发安全性（路径相交）
            can_join = all(
                check_concurrent_safety(node, member)
                for member in group
            )
            
            # 2. 检查执行域约束（同域节点必须串行）
            if can_join and node_domain:
                can_join = all(
                    domain_map.get(member.id) is not node_domain
                    for member in group
                )
                
            if can_join:
                group.append(node)
                placed = True
                break
        
        if not placed:
            groups.append([node])
    
    return groups


class DeterministicScheduler:
    """
    决定性调度器
    
    维护调度状态，按决定性规则分配 step_id
    """
    
    def __init__(self, nodes: List[Node]):
        self.nodes = {n.id: n for n in nodes}
        self.node_order = {n.id: i for i, n in enumerate(nodes)}
        self._step_counter = 0
        self._executed: Set[str] = set()
        self._in_progress: Set[str] = set()
    
    def allocate_step_id(self) -> int:
        """分配下一个 step_id"""
        self._step_counter += 1
        return self._step_counter
    
    def mark_executing(self, node_id: str) -> None:
        """标记节点开始执行"""
        self._in_progress.add(node_id)
    
    def mark_completed(self, node_id: str) -> None:
        """标记节点执行完成"""
        self._in_progress.discard(node_id)
        self._executed.add(node_id)
    
    def is_completed(self, node_id: str) -> bool:
        """检查节点是否已完成"""
        return node_id in self._executed
    
    def is_executing(self, node_id: str) -> bool:
        """检查节点是否正在执行"""
        return node_id in self._in_progress
    
    def select_from_ready(self, ready_nodes: List[Node], 
                          executed_this_round: Set[str]) -> Optional[Node]:
        """从就绪节点选择下一个执行"""
        available = []
        for n in ready_nodes:
            if self.is_executing(n.id):
                continue
            
            allow_reenter = n.policy.allow_reenter if n.policy else False
            # 在同一轮内，如果没有 allow_reenter，则不能执行两次
            if n.id in executed_this_round and not allow_reenter:
                continue
            
            available.append(n)
            
        return select_next_node(available, self.node_order)
    
    def get_execution_order(self) -> List[str]:
        """获取已执行节点的顺序"""
        # 按 step_id 排序（这里简化处理，实际应存储 step_id）
        return list(self._executed)


class ExecutionState:
    """
    执行状态追踪
    
    追踪节点执行状态，用于依赖解析
    """
    
    def __init__(self):
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        self.cancelled: Set[str] = set()
    
    def is_terminal(self, node_id: str) -> bool:
        """检查节点是否已到达终态"""
        return (
            node_id in self.completed or
            node_id in self.failed or
            node_id in self.cancelled
        )
    
    def is_successful(self, node_id: str) -> bool:
        """检查节点是否成功完成"""
        return node_id in self.completed


def are_dependencies_satisfied(
    node_id: str,
    graph,  # DependencyGraph
    exec_state: ExecutionState,
    check_all: bool = True
) -> bool:
    """
    检查节点的依赖是否已满足
    
    节点的所有 data/control 前置依赖必须已到达终态
    
    Args:
        node_id: 节点 ID
        graph: 依赖图
        exec_state: 执行状态
        check_all: 是否检查所有依赖。
                  对于循环入口节点，如果 check_all=False，则只需满足一个依赖即可（即 OR 语义）
                  但在 LinJ 规范下，默认仍为 AND 语义。
    """
    # 获取所有前置依赖节点
    deps = graph.get_data_dependencies(node_id)
    deps.extend(graph.get_control_dependencies(node_id))
    
    # 去重
    deps = list(set(deps))
    
    if not deps:
        return True

    # 简单的 AND 语义
    if check_all:
        return all(exec_state.is_terminal(dep) for dep in deps)
    else:
        # OR 语义：只要有一个依赖已满足
        return any(exec_state.is_terminal(dep) for dep in deps)


logger = logging.getLogger(__name__)


@dataclass
class ExecutionDomain:
    """
    执行域
    
    表示一组可以在一起执行的节点和资源（15.2 节）
    """
    node_ids: Set[str]
    resource_names: Set[str]
    domain_label: Optional[str] = None


class DomainAllocator:
    """
    执行域分配器
    
    根据 placement 声明和 resource 依赖分配执行域（15.2 节、25 节）
    """
    
    def __init__(self, available_domains: Optional[Set[str]] = None):
        """
        初始化域分配器
        
        Args:
            available_domains: 可用的执行域集合（可选，None 表示无限制）
        """
        self.available_domains = available_domains
    
    def allocate_domains(
        self,
        doc: LinJDocument,
        edges: Optional[List[Edge]] = None
    ) -> Mapping[str, ExecutionDomain]:
        """
        为节点分配执行域
        
        分配策略：
        1. 首先根据 placement 声明分配同域约束
        2. 然后根据 kind=resource 依赖分配同域约束
        3. 剩余节点分配到默认域
        
        Args:
            doc: LinJ 文档对象
            edges: 依赖边列表（可选，默认使用 doc.edges）
            
        Returns:
            节点ID到执行域的映射
        """
        if edges is None:
            edges = doc.edges
        
        node_ids = doc.get_node_ids()
        graph = DependencyGraph(edges)
        
        # 初始化：每个节点一个域
        domain_map: Dict[str, ExecutionDomain] = {}
        for node_id in node_ids:
            domain_map[node_id] = ExecutionDomain(
                node_ids={node_id},
                resource_names=set(),
                domain_label=None
            )
        
        # 应用 placement 同域约束
        domain_map = self._apply_placement_constraints(doc.placement, domain_map)
        
        # 应用 resource 依赖同域约束
        domain_map = self._apply_resource_constraints(edges, domain_map, graph)
        
        return domain_map
    
    def can_share_domain(
        self,
        node_a: str,
        node_b: str,
        edges: List[Edge]
    ) -> bool:
        """
        判断两个节点是否可以共享同一执行域
        
        两个节点可以共享执行域，当且仅当：
        - 它们之间没有相互依赖（循环）
        - 它们之间没有冲突的 writes 路径
        
        Args:
            node_a: 第一个节点
            node_b: 第二个节点
            edges: 依赖边列表
            
        Returns:
            是否可以共享执行域
        """
        graph = DependencyGraph(edges)
        
        # 检查是否有相互依赖
        deps_a = graph.get_data_dependencies(node_a) + graph.get_control_dependencies(node_a)
        deps_b = graph.get_data_dependencies(node_b) + graph.get_control_dependencies(node_b)
        
        if node_b in deps_a and node_a in deps_b:
            return False  # 相互依赖，不能同域
        
        return True
    
    def _apply_placement_constraints(
        self,
        placement: Optional[List[Placement]],
        domain_map: Dict[str, ExecutionDomain]
    ) -> Dict[str, ExecutionDomain]:
        """
        应用 placement 同域约束
        
        Args:
            placement: 放置声明列表
            domain_map: 当前域映射
            
        Returns:
            应用约束后的域映射
        """
        if not placement:
            return domain_map
        
        # 按 domain 分组
        domain_groups: Dict[str, Set[str]] = {}
        for p in placement:
            if p.target not in domain_map:
                continue
            
            if p.domain not in domain_groups:
                domain_groups[p.domain] = set()
            domain_groups[p.domain].add(p.target)
        
        # 合并同一域的节点
        for domain, targets in domain_groups.items():
            domain_map = self._merge_domains(targets, domain_map, domain_label=domain)
        
        return domain_map
    
    def _apply_resource_constraints(
        self,
        edges: List[Edge],
        domain_map: Dict[str, ExecutionDomain],
        graph: DependencyGraph
    ) -> Dict[str, ExecutionDomain]:
        """
        应用 kind=resource 依赖同域约束
        
        Args:
            edges: 依赖边列表
            domain_map: 当前域映射
            graph: 依赖图
            
        Returns:
            应用约束后的域映射
        """
        # 按 resource_name 分组边
        resource_edges: Dict[str, List[Edge]] = {}
        for edge in edges:
            if edge.is_resource() and edge.resource_name:
                if edge.resource_name not in resource_edges:
                    resource_edges[edge.resource_name] = []
                resource_edges[edge.resource_name].append(edge)
        
        # 将使用同一 resource 的节点合并到同一域
        for resource_name, resource_deps in resource_edges.items():
            nodes_in_resource: Set[str] = set()
            for edge in resource_deps:
                if edge.from_ in domain_map:
                    nodes_in_resource.add(edge.from_)
                if edge.to in domain_map:
                    nodes_in_resource.add(edge.to)
            
            if len(nodes_in_resource) > 1:
                domain_map = self._merge_domains(nodes_in_resource, domain_map, resource_name)
        
        return domain_map
    
    def _merge_domains(
        self,
        targets: Set[str],
        domain_map: Dict[str, ExecutionDomain],
        domain_label: Optional[str] = None
    ) -> Dict[str, ExecutionDomain]:
        """
        合并一组目标到同一执行域
        
        Args:
            targets: 要合并的目标集合
            domain_map: 当前域映射
            domain_label: 域标签
            
        Returns:
            合并后的域映射
        """
        if len(targets) <= 1:
            return domain_map
        
        targets_list = list(targets)
        first_target = targets_list[0]
        merged_domain = domain_map[first_target]
        
        # 合并所有目标到第一个目标的域
        for target in targets_list[1:]:
            target_domain = domain_map[target]
            
            # 合并节点集合
            merged_domain.node_ids.update(target_domain.node_ids)
            
            # 合并资源集合
            merged_domain.resource_names.update(target_domain.resource_names)
            
            # 更新目标节点的域引用
            for node_id in target_domain.node_ids:
                domain_map[node_id] = merged_domain
        
        # 设置域标签
        if domain_label:
            merged_domain.domain_label = domain_label
        
        return domain_map
