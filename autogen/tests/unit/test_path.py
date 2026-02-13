"""
PathResolver Unit Tests

Tests path syntax and read/write semantics defined in Sections 5.1-5.4
"""

import pytest
from linj_autogen.core.path import PathResolver, PathSegment
from linj_autogen.core.errors import MappingError


class TestPathParse:
    """Test path parsing"""

    def test_parse_simple_field(self):
        """Test simple field path"""
        segments = PathResolver.parse("$.a")
        assert len(segments) == 1
        assert segments[0].key == "a"
        assert not segments[0].is_array_index()

    def test_parse_nested_field(self):
        """Test nested field path"""
        segments = PathResolver.parse("$.a.b.c")
        assert len(segments) == 3
        assert segments[0].key == "a"
        assert segments[1].key == "b"
        assert segments[2].key == "c"

    def test_parse_array_index(self):
        """Test array index path"""
        segments = PathResolver.parse("$.arr[0]")
        assert len(segments) == 2
        assert segments[0].key == "arr"
        assert segments[1].key == 0
        assert segments[1].is_array_index()

    def test_parse_mixed_path(self):
        """Test mixed path"""
        segments = PathResolver.parse("$.a[0].b[1].c")
        assert len(segments) == 5
        assert segments[0].key == "a"
        assert segments[1].key == 0
        assert segments[2].key == "b"
        assert segments[3].key == 1
        assert segments[4].key == "c"

    def test_parse_root_only(self):
        """Test root path"""
        segments = PathResolver.parse("$")
        assert len(segments) == 0

    def test_parse_invalid_no_dollar(self):
        """Test path not starting with $"""
        with pytest.raises(MappingError):
            PathResolver.parse("a.b.c")

    def test_parse_invalid_syntax(self):
        """Test invalid syntax"""
        with pytest.raises(MappingError):
            PathResolver.parse("$.a..b")


class TestPathGet:
    """Test path read (Section 5.2)"""

    def test_get_simple_field(self):
        """Test reading simple field"""
        obj = {"a": 1}
        assert PathResolver.get(obj, "$.a") == 1

    def test_get_nested_field(self):
        """Test reading nested field"""
        obj = {"a": {"b": {"c": 42}}}
        assert PathResolver.get(obj, "$.a.b.c") == 42

    def test_get_array_element(self):
        """Test reading array element"""
        obj = {"arr": [10, 20, 30]}
        assert PathResolver.get(obj, "$.arr[0]") == 10
        assert PathResolver.get(obj, "$.arr[1]") == 20
        assert PathResolver.get(obj, "$.arr[2]") == 30

    def test_get_nonexistent_field_returns_none(self):
        """Test reading nonexistent field returns None (Section 5.2)"""
        obj = {"a": 1}
        assert PathResolver.get(obj, "$.nonexistent") is None

    def test_get_nonexistent_path_returns_none(self):
        """Test reading nonexistent path returns None"""
        obj = {"a": 1}
        assert PathResolver.get(obj, "$.nonexistent.field") is None

    def test_get_array_out_of_bounds_returns_none(self):
        """Test array out of bounds returns None"""
        obj = {"arr": [1, 2, 3]}
        assert PathResolver.get(obj, "$.arr[10]") is None

    def test_get_from_null_returns_none(self):
        """Test reading from null value returns None"""
        obj = {"a": None}
        assert PathResolver.get(obj, "$.a.b") is None

    def test_get_on_non_object_returns_none(self):
        """Test reading field on non-object returns None"""
        obj = {"a": "string"}
        assert PathResolver.get(obj, "$.a.b") is None


class TestPathSet:
    """Test path write (Section 5.3)"""

    def test_set_simple_field(self):
        """Test setting simple field"""
        obj = {}
        PathResolver.set(obj, "$.a", 42)
        assert obj == {"a": 42}

    def test_set_nested_field_creates_objects(self):
        """Test setting nested field auto-creates objects (Section 5.3.1)"""
        obj = {}
        PathResolver.set(obj, "$.a.b.c", 42)
        assert obj == {"a": {"b": {"c": 42}}}

    def test_set_array_element(self):
        """Test setting array element"""
        obj = {"arr": [1, 2, 3]}
        PathResolver.set(obj, "$.arr[1]", 99)
        assert obj["arr"] == [1, 99, 3]

    def test_set_array_expands_with_null(self):
        """Test array expansion with null fill (Section 5.3.2)"""
        obj = {"arr": [1]}
        PathResolver.set(obj, "$.arr[3]", 42)
        assert obj["arr"] == [1, None, None, 42]

    def test_set_creates_array_when_missing(self):
        """Test auto-create array when missing"""
        obj = {}
        PathResolver.set(obj, "$.arr[0]", 42)
        assert obj == {"arr": [42]}

    def test_set_on_non_object_raises_error(self):
        """Test setting field on non-object raises MappingError (Section 5.3.1)"""
        obj = {"a": "string"}
        with pytest.raises(MappingError):
            PathResolver.set(obj, "$.a.b", 42)

    def test_set_on_non_array_raises_error(self):
        """Test setting element on non-array raises MappingError (Section 5.3.2)"""
        obj = {"arr": "string"}
        with pytest.raises(MappingError):
            PathResolver.set(obj, "$.arr[0]", 42)

    def test_set_root_raises_error(self):
        """Test setting root path raises error"""
        obj = {}
        with pytest.raises(MappingError):
            PathResolver.set(obj, "$", 42)

    def test_set_array_exceeds_max_length(self):
        """Test array length exceeds limit raises error"""
        obj = {"arr": []}
        with pytest.raises(MappingError) as exc_info:
            PathResolver.set(obj, "$.arr[100]", 42, max_array_length=50)
        assert exc_info.value.details["threshold"] == 50


class TestPathDelete:
    """Test path delete (Section 5.4)"""

    def test_delete_field(self):
        """Test deleting field"""
        obj = {"a": 1, "b": 2}
        PathResolver.delete(obj, "$.a")
        assert "a" not in obj
        assert obj["b"] == 2

    def test_delete_nonexistent_field_silently_ignores(self):
        """Test deleting nonexistent field silently ignores (Section 5.4)"""
        obj = {"a": 1}
        PathResolver.delete(obj, "$.nonexistent")
        assert obj == {"a": 1}

    def test_delete_array_element_sets_null(self):
        """Test deleting array element sets null (Section 5.4)"""
        obj = {"arr": [1, 2, 3]}
        PathResolver.delete(obj, "$.arr[1]")
        assert obj["arr"] == [1, None, 3]
        assert len(obj["arr"]) == 3  # Length unchanged

    def test_delete_array_out_of_bounds_silently_ignores(self):
        """Test array out of bounds delete silently ignores (Section 5.4)"""
        obj = {"arr": [1, 2, 3]}
        PathResolver.delete(obj, "$.arr[10]")
        assert obj["arr"] == [1, 2, 3]


class TestPathIntersect:
    """Test path intersection (Section 11.4)"""

    def test_same_path_intersect(self):
        """Test same path intersects"""
        assert PathResolver.intersect("$.a.b", "$.a.b") is True

    def test_prefix_intersect(self):
        """Test prefix path intersects"""
        assert PathResolver.intersect("$.a", "$.a.b") is True
        assert PathResolver.intersect("$.a.b", "$.a") is True

    def test_different_fields_not_intersect(self):
        """Test different fields do not intersect"""
        assert PathResolver.intersect("$.a", "$.b") is False
        assert PathResolver.intersect("$.a.x", "$.a.y") is False

    def test_array_same_index_intersect(self):
        """Test same array index intersects"""
        assert PathResolver.intersect("$.arr[0]", "$.arr[0]") is True
        assert PathResolver.intersect("$.arr[0].x", "$.arr[0]") is True

    def test_array_different_index_not_intersect(self):
        """Test different array indexes do not intersect (Section 11.4)"""
        assert PathResolver.intersect("$.arr[0]", "$.arr[1]") is False
        assert PathResolver.intersect("$.arr[0].x", "$.arr[1].x") is False

    def test_array_and_element_intersect(self):
        """Test array and its element intersect (Section 11.4)"""
        assert PathResolver.intersect("$.arr", "$.arr[0]") is True

    def test_complex_paths(self):
        """Test complex path intersection"""
        assert PathResolver.intersect("$.a[0].b", "$.a[0]") is True
        assert PathResolver.intersect("$.a[0].b", "$.a[1]") is False


class TestPathSegment:
    """Test PathSegment class"""

    def test_field_segment(self):
        """Test field segment"""
        seg = PathSegment("field")
        assert seg.key == "field"
        assert not seg.is_array_index()
        assert repr(seg) == ".field"

    def test_array_segment(self):
        """Test array segment"""
        seg = PathSegment(42)
        assert seg.key == 42
        assert seg.is_array_index()
        assert repr(seg) == "[42]"
