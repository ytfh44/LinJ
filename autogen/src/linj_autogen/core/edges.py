"""
LinJ 依赖边定义

实现规范 8 节定义的依赖和映射
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EdgeKind(str, Enum):
    """依赖类型 (8.1 节)"""
    DATA = "data"
    CONTROL = "control"
    RESOURCE = "resource"


class MapRule(BaseModel):
    """
    映射规则 (8.2 节)
    
    - from: 源路径
    - to: 目标路径
    - default: 可选默认值
    """
    from_: str = Field(alias="from")
    to: str
    default: Optional[Any] = None


class Edge(BaseModel):
    """
    依赖边 (8.1 节)
    
    必须包含：
    - from: 源节点 id
    - to: 目标节点 id
    - kind: data/control/resource
    """
    from_: str = Field(alias="from")
    to: str
    kind: EdgeKind
    weight: float = 1.0
    map_rules: Optional[List[MapRule]] = Field(default=None, alias="map")
    resource_name: Optional[str] = None
    
    # 兼容性属性
    @property
    def map(self) -> Optional[List[MapRule]]:
        """映射规则列表（兼容性属性）"""
        return self.map_rules
    
    def is_data(self) -> bool:
        return self.kind == EdgeKind.DATA
    
    def is_control(self) -> bool:
        return self.kind == EdgeKind.CONTROL
    
    def is_resource(self) -> bool:
        return self.kind == EdgeKind.RESOURCE
    
    def apply_map(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用映射规则到状态
        
        8.2 节：在目标节点执行前应用映射规则
        """
        from .path import PathResolver
        
        if not self.map:
            return {}
        
        result: Dict[str, Any] = {}
        for rule in self.map:
            source_value = PathResolver.get(state, rule.from_)
            
            if source_value is not None:
                PathResolver.set(result, rule.to, source_value)
            elif rule.default is not None:
                PathResolver.set(result, rule.to, rule.default)
            # 否则该条规则为无效操作
        
        return result


class DependencyGraph:
    """
    依赖图
    
    管理节点间的依赖关系，提供依赖解析功能
    """
    
    def __init__(self, edges: List[Edge]):
        self.edges = edges
        self._build_index()
    
    def _build_index(self):
        """构建索引以加速查询"""
        self._incoming: Dict[str, List[Edge]] = {}
        self._outgoing: Dict[str, List[Edge]] = {}
        
        for edge in self.edges:
            # 入边
            if edge.to not in self._incoming:
                self._incoming[edge.to] = []
            self._incoming[edge.to].append(edge)
            
            # 出边
            if edge.from_ not in self._outgoing:
                self._outgoing[edge.from_] = []
            self._outgoing[edge.from_].append(edge)
    
    def get_incoming(self, node_id: str) -> List[Edge]:
        """获取节点的所有入边"""
        return self._incoming.get(node_id, [])
    
    def get_outgoing(self, node_id: str) -> List[Edge]:
        """获取节点的所有出边"""
        return self._outgoing.get(node_id, [])
    
    def get_data_dependencies(self, node_id: str) -> List[str]:
        """获取节点的数据依赖节点"""
        return [
            edge.from_ for edge in self.get_incoming(node_id)
            if edge.is_data()
        ]
    
    def get_control_dependencies(self, node_id: str) -> List[str]:
        """获取节点的控制依赖节点"""
        return [
            edge.from_ for edge in self.get_incoming(node_id)
            if edge.is_control()
        ]
    
    def get_data_mapping(self, node_id: str) -> List[Edge]:
        """获取节点的数据映射边（含 map 的边）"""
        return [
            edge for edge in self.get_incoming(node_id)
            if edge.is_data() and edge.map
        ]
    
    def has_incoming(self, node_id: str) -> bool:
        """检查节点是否有入边"""
        return node_id in self._incoming and len(self._incoming[node_id]) > 0
    
    def has_outgoing(self, node_id: str) -> bool:
        """检查节点是否有出边"""
        return node_id in self._outgoing and len(self._outgoing[node_id]) > 0
    
    def resolve_map_conflicts(
        self,
        target_node_id: str,
        current_maps: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        解决同一目标节点的多条入边映射冲突 (8.3 节, 22.2 节)
        
        当同一目标节点存在多条入边映射，且对同一路径或相交路径写入时，
        必须按以下决定性优先级应用：
        
        1. weight 大者优先
        2. edges 数组中靠前者优先
        3. 若仍相同，按 (from, to) 字典序优先
        
        Args:
            target_node_id: 目标节点 ID
            current_maps: 当前映射列表，每项包含 'from', 'to', 'default', 'weight', 'from_node', 'edge_index'
                若为 None，则从入边中收集映射规则
        
        Returns:
            解决冲突后的映射列表，每项包含映射信息及覆盖原因
        
        Raises:
            无（采用决定性合并，不产生错误）
        """
        from .path import PathResolver
        
        # 收集所有映射规则
        if current_maps is None:
            current_maps = []
            incoming_edges = self.get_incoming(target_node_id)
            for edge_idx, edge in enumerate(incoming_edges):
                if edge.map:
                    for rule in edge.map:
                        current_maps.append({
                            'from': rule.from_,
                            'to': rule.to,
                            'default': rule.default,
                            'weight': edge.weight,
                            'from_node': edge.from_,
                            'edge_index': edge_idx,
                            'edge_weight': edge.weight,
                        })
        
        if len(current_maps) <= 1:
            return current_maps
        
        # 按优先级排序：weight 降序 -> edge_index 升序 -> (from, to) 字典序
        def sort_key(item: Dict[str, Any]) -> Tuple[float, int, str, str]:
            return (
                -item['weight'],           # weight 降序
                item['edge_index'],         # edge_index 升序
                item['from'],               # from 字典序
                item['to'],                 # to 字典序
            )
        
        sorted_maps = sorted(current_maps, key=sort_key)
        
        # 检测相交路径并解决冲突
        # 策略：已排序的列表中，优先级高的在前
        # 对于相交路径，只保留优先级最高的（第一个）
        resolved_maps: List[Dict[str, Any]] = []
        covered_maps: List[Dict[str, Any]] = []
        
        for i, current in enumerate(sorted_maps):
            is_covered = False
            
            # 检查是否与已解决的路径相交
            for j, existing in enumerate(resolved_maps):
                if PathResolver.intersect(current['to'], existing['to']):
                    # 当前条目优先级低于已解决的条目，被覆盖
                    is_covered = True
                    
                    # 确定覆盖原因
                    if existing['weight'] > current['weight']:
                        cover_reason = f"weight_higher:{existing['weight']}>{current['weight']}"
                    elif existing['weight'] == current['weight']:
                        if existing['edge_index'] < current['edge_index']:
                            cover_reason = f"edge_order:{existing['edge_index']}<{current['edge_index']}"
                        else:
                            cover_reason = f"lexicographic:({existing['from']},{existing['to']})<({current['from']},{current['to']})"
                    else:
                        cover_reason = f"lexicographic:({existing['from']},{existing['to']})<({current['from']},{current['to']})"
                    
                    # 记录被覆盖的映射
                    covered_maps.append({
                        **current,
                        'covered_by': existing['from_node'],
                        'cover_reason': cover_reason,
                        'original_index': i,
                    })
                    break
            
            if not is_covered:
                resolved_maps.append(current)
        
        # 记录诊断信息
        if covered_maps:
            for covered in covered_maps:
                logger.info(
                    "Map rule covered during conflict resolution",
                    extra={
                        "target_node": target_node_id,
                        "covered_from": covered['from'],
                        "covered_to": covered['to'],
                        "covered_by": covered['covered_by'],
                        "cover_reason": covered['cover_reason'],
                    },
                )
        
        return resolved_maps


@dataclass
class MapConflictInfo:
    """
    映射冲突信息
    
    用于记录冲突解决过程中的详细信息
    """
    target_node: str
    covered_map: Dict[str, Any]
    winning_map: Dict[str, Any]
    reason: str  # weight, edge_order, lexicographic


@dataclass
class MapResolutionResult:
    """
    映射冲突解决结果
    """
    resolved_maps: List[Dict[str, Any]]
    conflicts: List[MapConflictInfo]


def resolve_map_conflicts(
    edges: List[Edge],
    target_node_id: Optional[str] = None,
    record_diagnostics: bool = True
) -> MapResolutionResult:
    """
    解决多入边映射冲突 (8.3 节)
    
    默认行为：产生 ConflictError
    若支持决定性覆盖，按以下优先级：
    1. weight 大者优先
    2. edges 数组中靠前者优先
    3. 若仍相同，按 (from, to) 字典序
    
    Args:
        edges: 边列表
        target_node_id: 目标节点 ID（可选）
        record_diagnostics: 是否记录诊断信息
    
    Returns:
        MapResolutionResult: 包含解决后的映射和冲突信息
    """
    # 构建依赖图
    graph = DependencyGraph(edges)
    
    # 收集所有映射规则
    all_maps: List[Dict[str, Any]] = []
    for edge_idx, edge in enumerate(edges):
        if edge.map:
            for rule in edge.map:
                all_maps.append({
                    'from': rule.from_,
                    'to': rule.to,
                    'default': rule.default,
                    'weight': edge.weight,
                    'from_node': edge.from_,
                    'edge_index': edge_idx,
                    'edge_weight': edge.weight,
                })
    
    if target_node_id:
        # 只解决目标节点的冲突
        resolved = graph.resolve_map_conflicts(target_node_id, all_maps)
    else:
        # 解决所有节点的冲突（简化处理）
        resolved = all_maps
    
    # 记录诊断信息
    conflicts: List[MapConflictInfo] = []
    if record_diagnostics:
        # 分析被覆盖的规则
        all_map_set = set((m['from'], m['to']) for m in all_maps)
        resolved_set = set((m['from'], m['to']) for m in resolved)
        
        for om in all_maps:
            if (om['from'], om['to']) not in resolved_set:
                # 找到获胜的规则
                for rm in resolved:
                    if rm['from'] == om['from'] and rm['to'] == om['to']:
                        # 确定覆盖原因
                        if om['weight'] < rm['weight']:
                            reason = "weight"
                        elif om['weight'] == rm['weight'] and om['edge_index'] > rm['edge_index']:
                            reason = "edge_order"
                        else:
                            reason = "lexicographic"
                        
                        conflicts.append(MapConflictInfo(
                            target_node=target_node_id or "",
                            covered_map=om,
                            winning_map=rm,
                            reason=reason,
                        ))
                        break
    
    return MapResolutionResult(resolved_maps=resolved, conflicts=conflicts)
