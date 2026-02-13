"""
ChangeSet Unit Tests

Tests for ChangeSet format and intersection logic defined in sections 9.1, 9.2, and 11.4
"""

import pytest
from linj_autogen.core.changeset import ChangeSet, ChangeSetBuilder, WriteOp, DeleteOp


class TestChangeSetCreation:
    """Tests for ChangeSet creation"""

    def test_empty_changeset(self):
        """Test empty ChangeSet"""
        cs = ChangeSet()
        assert cs.is_empty()
        assert cs.writes == []
        assert cs.deletes == []

    def test_changeset_with_writes(self):
        """Test ChangeSet with writes"""
        cs = ChangeSet(
            writes=[WriteOp(path="$.a", value=1), WriteOp(path="$.b", value=2)]
        )
        assert not cs.is_empty()
        assert len(cs.writes) == 2
        assert cs.get_write_paths() == {"$.a", "$.b"}

    def test_changeset_with_deletes(self):
        """Test ChangeSet with deletes"""
        cs = ChangeSet(deletes=[DeleteOp(path="$.x"), DeleteOp(path="$.y")])
        assert not cs.is_empty()
        assert len(cs.deletes) == 2
        assert cs.get_delete_paths() == {"$.x", "$.y"}

    def test_changeset_with_both(self):
        """Test ChangeSet with both writes and deletes"""
        cs = ChangeSet(
            writes=[WriteOp(path="$.a", value=1)], deletes=[DeleteOp(path="$.b")]
        )
        assert not cs.is_empty()
        assert cs.get_all_modified_paths() == {"$.a", "$.b"}


class TestChangeSetBuilder:
    """Tests for ChangeSet builder"""

    def test_builder_chain(self):
        """Test builder chain"""
        builder = ChangeSetBuilder()
        cs = builder.write("$.a", 1).write("$.b", 2).delete("$.c").build()

        assert len(cs.writes) == 2
        assert len(cs.deletes) == 1

    def test_builder_empty(self):
        """Test empty builder"""
        builder = ChangeSetBuilder()
        assert builder.is_empty()

        cs = builder.build()
        assert cs.is_empty()

    def test_builder_clear(self):
        """Test clear builder"""
        builder = ChangeSetBuilder()
        builder.write("$.a", 1)
        assert not builder.is_empty()

        builder.clear()
        assert builder.is_empty()


class TestChangeSetIntersection:
    """Tests for ChangeSet intersection (Section 11.4)"""

    def test_same_write_path_intersects(self):
        """Test same write path intersects"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_write("$.a", 2)
        assert cs1.intersects_with(cs2)

    def test_different_write_paths_not_intersect(self):
        """Test different write paths do not intersect"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_write("$.b", 2)
        assert not cs1.intersects_with(cs2)

    def test_prefix_path_intersects(self):
        """Test prefix path intersects"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_write("$.a.b", 2)
        assert cs1.intersects_with(cs2)

    def test_array_different_index_not_intersect(self):
        """Test different array indices do not intersect"""
        cs1 = ChangeSet.create_write("$.arr[0]", 1)
        cs2 = ChangeSet.create_write("$.arr[1]", 2)
        assert not cs1.intersects_with(cs2)

    def test_write_delete_same_path_intersects(self):
        """Test write and delete on same path intersects"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_delete("$.a")
        assert cs1.intersects_with(cs2)

    def test_complex_paths_intersection(self):
        """Test complex path intersection"""
        cs1 = ChangeSet.create_write("$.a[0].b", 1)
        cs2 = ChangeSet.create_write("$.a[0]", 2)
        assert cs1.intersects_with(cs2)

        cs3 = ChangeSet.create_write("$.a[1].b", 1)
        assert not cs1.intersects_with(cs3)


class TestChangeSetApply:
    """Tests for ChangeSet application"""

    def test_apply_writes(self):
        """Test apply writes"""
        state = {}
        cs = ChangeSet(
            writes=[WriteOp(path="$.a", value=1), WriteOp(path="$.b.c", value=2)]
        )
        cs.apply_to(state)

        assert state["a"] == 1
        assert state["b"]["c"] == 2

    def test_apply_deletes(self):
        """Test apply deletes"""
        state = {"a": 1, "b": 2}
        cs = ChangeSet(deletes=[DeleteOp(path="$.a")])
        cs.apply_to(state)

        assert "a" not in state
        assert state["b"] == 2

    def test_apply_both(self):
        """Test apply both writes and deletes"""
        state = {"old": 1}
        cs = ChangeSet(
            writes=[WriteOp(path="$.new", value=2)], deletes=[DeleteOp(path="$.old")]
        )
        cs.apply_to(state)

        assert "old" not in state
        assert state["new"] == 2

    def test_apply_array_element(self):
        """Test apply array element change"""
        state = {"arr": [1, 2, 3]}
        cs = ChangeSet(writes=[WriteOp(path="$.arr[1]", value=99)])
        cs.apply_to(state)

        assert state["arr"] == [1, 99, 3]


class TestChangeSetMerge:
    """Tests for ChangeSet merge"""

    def test_merge_two_changesets(self):
        """Test merge two ChangeSets"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet.create_write("$.b", 2)

        merged = cs1.merge(cs2)

        assert len(merged.writes) == 2
        assert merged.get_write_paths() == {"$.a", "$.b"}

    def test_merge_with_empty(self):
        """Test merge with empty ChangeSet"""
        cs1 = ChangeSet.create_write("$.a", 1)
        cs2 = ChangeSet()

        merged = cs1.merge(cs2)

        assert len(merged.writes) == 1
        assert merged.get_write_paths() == {"$.a"}
