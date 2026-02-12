"""
变更集 (ChangeSet)

实现 LinJ 规范 9.1、9.2、20 节定义的变更集格式和原子性提交。
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from .path import PathResolver


class WriteOp(BaseModel):
    """写操作"""
    path: str
    value: Any


class DeleteOp(BaseModel):
    """删除操作"""
    path: str


class ChangeSet(BaseModel):
    """
    变更集
    
    按 9.1 节格式：
    - writes: [{path, value}]
    - deletes: [{path}]
    
    按 9.2 节要求：变更集应用必须是原子性的
    """
    writes: List[WriteOp] = Field(default_factory=list)
    deletes: List[DeleteOp] = Field(default_factory=list)
    
    def is_empty(self) -> bool:
        """检查变更集是否为空"""
        return len(self.writes) == 0 and len(self.deletes) == 0
    
    def get_write_paths(self) -> Set[str]:
        """获取所有写入路径"""
        return {op.path for op in self.writes}
    
    def get_delete_paths(self) -> Set[str]:
        """获取所有删除路径"""
        return {op.path for op in self.deletes}
    
    def get_all_modified_paths(self) -> Set[str]:
        """获取所有被修改的路径（写入或删除）"""
        return self.get_write_paths() | self.get_delete_paths()
    
    def intersects_with(self, other: "ChangeSet") -> bool:
        """
        检查是否与另一个变更集有路径相交
        
        按 11.4 节规则判定写入/删除路径相交
        """
        self_paths = self.get_all_modified_paths()
        other_paths = other.get_all_modified_paths()
        
        for path_a in self_paths:
            for path_b in other_paths:
                if PathResolver.intersect(path_a, path_b):
                    return True
        return False
    
    def apply_to(self, state: Dict[str, Any]) -> None:
        """
        将变更集应用到状态
        
        注意：此方法不保证原子性，调用方应在外层处理
        """
        # 先应用删除（5.4 节：设为 null）
        for op in self.deletes:
            PathResolver.delete(state, op.path)
        
        # 再应用写入
        for op in self.writes:
            PathResolver.set(state, op.path, op.value)
    
    @classmethod
    def create_write(cls, path: str, value: Any) -> "ChangeSet":
        """创建单个写入的变更集"""
        return cls(writes=[WriteOp(path=path, value=value)])
    
    @classmethod
    def create_delete(cls, path: str) -> "ChangeSet":
        """创建单个删除的变更集"""
        return cls(deletes=[DeleteOp(path=path)])
    
    def merge(self, other: "ChangeSet") -> "ChangeSet":
        """
        合并两个变更集
        
        注意：合并后的变更集可能包含冲突路径，调用方应检查
        """
        return ChangeSet(
            writes=self.writes + other.writes,
            deletes=self.deletes + other.deletes
        )


class ChangeSetBuilder:
    """
    变更集构建器
    
    用于在节点执行过程中累积变更
    """
    
    def __init__(self):
        self._writes: List[WriteOp] = []
        self._deletes: List[DeleteOp] = []
    
    def write(self, path: str, value: Any) -> "ChangeSetBuilder":
        """添加写入操作"""
        self._writes.append(WriteOp(path=path, value=value))
        return self
    
    def delete(self, path: str) -> "ChangeSetBuilder":
        """添加删除操作"""
        self._deletes.append(DeleteOp(path=path))
        return self
    
    def build(self) -> ChangeSet:
        """构建变更集"""
        return ChangeSet(
            writes=self._writes.copy(),
            deletes=self._deletes.copy()
        )
    
    def is_empty(self) -> bool:
        """检查是否为空"""
        return len(self._writes) == 0 and len(self._deletes) == 0
    
    def clear(self) -> None:
        """清空构建器"""
        self._writes.clear()
        self._deletes.clear()
