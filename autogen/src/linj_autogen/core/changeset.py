"""
ChangeSet

Implements ChangeSet format and atomic commits defined in LinJ specification sections 9.1, 9.2, and 20.
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from .path import PathResolver


class WriteOp(BaseModel):
    """Write operation"""

    path: str
    value: Any


class DeleteOp(BaseModel):
    """Delete operation"""

    path: str


class ChangeSet(BaseModel):
    """
    ChangeSet

    Format per section 9.1:
    - writes: [{path, value}]
    - deletes: [{path}]

    Per section 9.2 requirement: ChangeSet application must be atomic
    """

    writes: List[WriteOp] = Field(default_factory=list)
    deletes: List[DeleteOp] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """Check if ChangeSet is empty"""
        return len(self.writes) == 0 and len(self.deletes) == 0

    def get_write_paths(self) -> Set[str]:
        """Get all write paths"""
        return {op.path for op in self.writes}

    def get_delete_paths(self) -> Set[str]:
        """Get all delete paths"""
        return {op.path for op in self.deletes}

    def get_all_modified_paths(self) -> Set[str]:
        """Get all modified paths (write or delete)"""
        return self.get_write_paths() | self.get_delete_paths()

    def intersects_with(self, other: "ChangeSet") -> bool:
        """
        Check if path intersects with another ChangeSet

        Determine write/delete path intersection per section 11.4 rules
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
        Apply ChangeSet to state

        Note: This method does not guarantee atomicity, caller should handle it externally
        """
        # First apply deletes (section 5.4: set to null)
        for op in self.deletes:
            PathResolver.delete(state, op.path)

        # Then apply writes
        for op in self.writes:
            PathResolver.set(state, op.path, op.value)

    @classmethod
    def create_write(cls, path: str, value: Any) -> "ChangeSet":
        """Create ChangeSet for single write"""
        return cls(writes=[WriteOp(path=path, value=value)])

    @classmethod
    def create_delete(cls, path: str) -> "ChangeSet":
        """Create ChangeSet for single delete"""
        return cls(deletes=[DeleteOp(path=path)])

    def merge(self, other: "ChangeSet") -> "ChangeSet":
        """
        Merge two ChangeSets

        Note: Merged ChangeSet may contain conflicting paths, caller should check
        """
        return ChangeSet(
            writes=self.writes + other.writes, deletes=self.deletes + other.deletes
        )


class ChangeSetBuilder:
    """
    ChangeSet builder

    Used to accumulate changes during node execution
    """

    def __init__(self):
        self._writes: List[WriteOp] = []
        self._deletes: List[DeleteOp] = []

    def write(self, path: str, value: Any) -> "ChangeSetBuilder":
        """Add write operation"""
        self._writes.append(WriteOp(path=path, value=value))
        return self

    def delete(self, path: str) -> "ChangeSetBuilder":
        """Add delete operation"""
        self._deletes.append(DeleteOp(path=path))
        return self

    def build(self) -> ChangeSet:
        """Build ChangeSet"""
        return ChangeSet(writes=self._writes.copy(), deletes=self._deletes.copy())

    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self._writes) == 0 and len(self._deletes) == 0

    def clear(self) -> None:
        """Clear builder"""
        self._writes.clear()
        self._deletes.clear()
