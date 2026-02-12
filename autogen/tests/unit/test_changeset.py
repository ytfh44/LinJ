"""
ChangeSet 单元测试

测试 9.1、9.2、11.4 节定义的变更集格式和相交判定
"""

import pytest
from linj_autogen.core.changeset import ChangeSet, ChangeSetBuilder, WriteOp, DeleteOp


class TestChangeSetCreation:
    """测试变更集创建"""
    
    def test_empty_changeset(self):
        """测试空变更集"""
        cs = ChangeSet()
        assert cs.is_empty()
        assert cs.writes == []
        assert cs.deletes == []
    
    def test_changeset_with_writes(self):
        """测试带写入的变更集"""
        cs = ChangeSet(writes=[
            WriteOp(path="$.a", value=1),
            WriteOp(path="$.b", value=2)
        ])
        assert not cs.is_empty()
        assert len(cs.writes) == 2
        assert cs.get_write_paths() == {"$.a", "$.b"}
    
    def test_changeset_with_deletes(self):
        """测试带删除的变更集"""
        cs = ChangeSet(deletes=[
            DeleteOp(path="$.x"),
            DeleteOp(path="$.y")
        ])
        assert not cs.is_empty()
        assert len(cs.deletes) == 2
        assert cs.get_delete_paths() == {"$.x", "$.y"}
    
    def test_changeset_with_both(self):
        """测试同时有写入和删除的变更集"""
        cs = ChangeSet(
            writes=[WriteOp(path="$.a", value=1)],
            deletes=[DeleteOp(path="$.b")]
        )
        assert not cs.is_empty()
        assert cs.get_all_modified_paths() == {"$.a", "$.b"}


class TestChangeSetBuilder:
    """测试变更集构建器"""
    
    def test_builder_chain(self):
        """测试链式调用"""
        builder = ChangeSetBuilder()
        cs = (builder
              .write("$.a", 1)
              .write("$.b", 2)
              .delete("$.c")
              .build())
        
        assert len(cs.writes) == 2
        assert len(cs.deletes) == 1
    
    def test_builder_empty(self):
        """测试空构建器"""
        builder = ChangeSetBuilder()
        assert builder.is_empty()
        
        cs = builder.build()
        assert cs.is_empty()
    
    def test_builder_clear(self):
        """测试清空构建器"""
        builder = ChangeSetBuilder()
        builder.write("$.a", 1)
        assert not builder.is_empty()
        
        builder.clear()
        assert builder.is_empty()


class TestChangeSetIntersection:
    """测试变更集相交判定 (11.4 节)"""
    
    def test_same_write_path_intersects(self):
        """测试相同写入路径相交"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_write("$.a", 2)
        assert cs1.intersects_with(cs2)
    
    def test_different_write_paths_not_intersect(self):
        """测试不同写入路径不相交"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_write("$.b", 2)
        assert not cs1.intersects_with(cs2)
    
    def test_prefix_path_intersects(self):
        """测试前缀路径相交"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_write("$.a.b", 2)
        assert cs1.intersects_with(cs2)
    
    def test_array_different_index_not_intersect(self):
        """测试不同数组下标不相交"""
        cs1 = ChangeSet.create_write("$.arr[0]", 1)
        cs2 = ChangeSet.create_write("$.arr[1]", 2)
        assert not cs1.intersects_with(cs2)
    
    def test_write_delete_same_path_intersects(self):
        """测试写入和删除相同路径相交"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_delete("$.a")
        assert cs1.intersects_with(cs2)
    
    def test_complex_paths_intersection(self):
        """测试复杂路径相交"""
        cs1 = ChangeSet.create_write("$.a[0].b", 1)
        cs2 = ChangeSet.create_write("$.a[0]", 2)
        assert cs1.intersects_with(cs2)
        
        cs3 = ChangeSet.create_write("$.a[1].b", 1)
        assert not cs1.intersects_with(cs3)


class TestChangeSetApply:
    """测试变更集应用"""
    
    def test_apply_writes(self):
        """测试应用写入"""
        state = {}
        cs = ChangeSet(writes=[
            WriteOp(path="$.a", value=1),
            WriteOp(path="$.b.c", value=2)
        ])
        cs.apply_to(state)
        
        assert state["a"] == 1
        assert state["b"]["c"] == 2
    
    def test_apply_deletes(self):
        """测试应用删除"""
        state = {"a": 1, "b": 2}
        cs = ChangeSet(deletes=[DeleteOp(path="$.a")])
        cs.apply_to(state)
        
        assert "a" not in state
        assert state["b"] == 2
    
    def test_apply_both(self):
        """测试同时应用写入和删除"""
        state = {"old": 1}
        cs = ChangeSet(
            writes=[WriteOp(path="$.new", value=2)],
            deletes=[DeleteOp(path="$.old")]
        )
        cs.apply_to(state)
        
        assert "old" not in state
        assert state["new"] == 2
    
    def test_apply_array_element(self):
        """测试应用数组元素变更"""
        state = {"arr": [1, 2, 3]}
        cs = ChangeSet(writes=[WriteOp(path="$.arr[1]", value=99)])
        cs.apply_to(state)
        
        assert state["arr"] == [1, 99, 3]


class TestChangeSetMerge:
    """测试变更集合并"""
    
    def test_merge_two_changesets(self):
        """测试合并两个变更集"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_write("$.b", 2)
        
        merged = cs1.merge(cs2)
        
        assert len(merged.writes) == 2
        assert merged.get_write_paths() == {"$.a", "$.b"}
    
    def test_merge_with_empty(self):
        """测试与空变更集合并"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet()
        
        merged = cs1.merge(cs2)
        
        assert len(merged.writes) == 1
        assert merged.get_write_paths() == {"$.a"}
