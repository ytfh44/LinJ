"""
PathResolver 单元测试

测试 5.1-5.4 节定义的路径语法和读写语义
"""

import pytest
from linj_autogen.core.path import PathResolver, PathSegment
from linj_autogen.core.errors import MappingError


class TestPathParse:
    """测试路径解析"""
    
    def test_parse_simple_field(self):
        """测试简单字段路径"""
        segments = PathResolver.parse("$.a")
        assert len(segments) == 1
        assert segments[0].key == "a"
        assert not segments[0].is_array_index()
    
    def test_parse_nested_field(self):
        """测试嵌套字段路径"""
        segments = PathResolver.parse("$.a.b.c")
        assert len(segments) == 3
        assert segments[0].key == "a"
        assert segments[1].key == "b"
        assert segments[2].key == "c"
    
    def test_parse_array_index(self):
        """测试数组下标路径"""
        segments = PathResolver.parse("$.arr[0]")
        assert len(segments) == 2
        assert segments[0].key == "arr"
        assert segments[1].key == 0
        assert segments[1].is_array_index()
    
    def test_parse_mixed_path(self):
        """测试混合路径"""
        segments = PathResolver.parse("$.a[0].b[1].c")
        assert len(segments) == 5
        assert segments[0].key == "a"
        assert segments[1].key == 0
        assert segments[2].key == "b"
        assert segments[3].key == 1
        assert segments[4].key == "c"
    
    def test_parse_root_only(self):
        """测试根路径"""
        segments = PathResolver.parse("$")
        assert len(segments) == 0
    
    def test_parse_invalid_no_dollar(self):
        """测试不以 $ 开头的路径"""
        with pytest.raises(MappingError):
            PathResolver.parse("a.b.c")
    
    def test_parse_invalid_syntax(self):
        """测试无效语法"""
        with pytest.raises(MappingError):
            PathResolver.parse("$.a..b")


class TestPathGet:
    """测试路径读取 (5.2 节)"""
    
    def test_get_simple_field(self):
        """测试读取简单字段"""
        obj = {"a": 1}
        assert PathResolver.get(obj, "$.a") == 1
    
    def test_get_nested_field(self):
        """测试读取嵌套字段"""
        obj = {"a": {"b": {"c": 42}}}
        assert PathResolver.get(obj, "$.a.b.c") == 42
    
    def test_get_array_element(self):
        """测试读取数组元素"""
        obj = {"arr": [10, 20, 30]}
        assert PathResolver.get(obj, "$.arr[0]") == 10
        assert PathResolver.get(obj, "$.arr[1]") == 20
        assert PathResolver.get(obj, "$.arr[2]") == 30
    
    def test_get_nonexistent_field_returns_none(self):
        """测试读取不存在字段返回 None (5.2 节)"""
        obj = {"a": 1}
        assert PathResolver.get(obj, "$.nonexistent") is None
    
    def test_get_nonexistent_path_returns_none(self):
        """测试读取不存在路径返回 None"""
        obj = {"a": 1}
        assert PathResolver.get(obj, "$.nonexistent.field") is None
    
    def test_get_array_out_of_bounds_returns_none(self):
        """测试数组越界返回 None"""
        obj = {"arr": [1, 2, 3]}
        assert PathResolver.get(obj, "$.arr[10]") is None
    
    def test_get_from_null_returns_none(self):
        """测试从 null 值读取返回 None"""
        obj = {"a": None}
        assert PathResolver.get(obj, "$.a.b") is None
    
    def test_get_on_non_object_returns_none(self):
        """测试在非对象上读取字段返回 None"""
        obj = {"a": "string"}
        assert PathResolver.get(obj, "$.a.b") is None


class TestPathSet:
    """测试路径写入 (5.3 节)"""
    
    def test_set_simple_field(self):
        """测试设置简单字段"""
        obj = {}
        PathResolver.set(obj, "$.a", 42)
        assert obj == {"a": 42}
    
    def test_set_nested_field_creates_objects(self):
        """测试设置嵌套字段自动创建对象 (5.3.1 节)"""
        obj = {}
        PathResolver.set(obj, "$.a.b.c", 42)
        assert obj == {"a": {"b": {"c": 42}}}
    
    def test_set_array_element(self):
        """测试设置数组元素"""
        obj = {"arr": [1, 2, 3]}
        PathResolver.set(obj, "$.arr[1]", 99)
        assert obj["arr"] == [1, 99, 3]
    
    def test_set_array_expands_with_null(self):
        """测试数组扩容以 null 填充 (5.3.2 节)"""
        obj = {"arr": [1]}
        PathResolver.set(obj, "$.arr[3]", 42)
        assert obj["arr"] == [1, None, None, 42]
    
    def test_set_creates_array_when_missing(self):
        """测试缺失数组时自动创建"""
        obj = {}
        PathResolver.set(obj, "$.arr[0]", 42)
        assert obj == {"arr": [42]}
    
    def test_set_on_non_object_raises_error(self):
        """测试在非对象上设置字段产生 MappingError (5.3.1 节)"""
        obj = {"a": "string"}
        with pytest.raises(MappingError):
            PathResolver.set(obj, "$.a.b", 42)
    
    def test_set_on_non_array_raises_error(self):
        """测试在非数组上设置元素产生 MappingError (5.3.2 节)"""
        obj = {"arr": "string"}
        with pytest.raises(MappingError):
            PathResolver.set(obj, "$.arr[0]", 42)
    
    def test_set_root_raises_error(self):
        """测试设置根路径产生错误"""
        obj = {}
        with pytest.raises(MappingError):
            PathResolver.set(obj, "$", 42)
    
    def test_set_array_exceeds_max_length(self):
        """测试数组长度超过限制产生错误"""
        obj = {"arr": []}
        with pytest.raises(MappingError) as exc_info:
            PathResolver.set(obj, "$.arr[100]", 42, max_array_length=50)
        assert exc_info.value.details["threshold"] == 50


class TestPathDelete:
    """测试路径删除 (5.4 节)"""
    
    def test_delete_field(self):
        """测试删除字段"""
        obj = {"a": 1, "b": 2}
        PathResolver.delete(obj, "$.a")
        assert "a" not in obj
        assert obj["b"] == 2
    
    def test_delete_nonexistent_field_silently_ignores(self):
        """测试删除不存在字段静默忽略 (5.4 节)"""
        obj = {"a": 1}
        PathResolver.delete(obj, "$.nonexistent")
        assert obj == {"a": 1}
    
    def test_delete_array_element_sets_null(self):
        """测试删除数组元素设为 null (5.4 节)"""
        obj = {"arr": [1, 2, 3]}
        PathResolver.delete(obj, "$.arr[1]")
        assert obj["arr"] == [1, None, 3]
        assert len(obj["arr"]) == 3  # 长度不变
    
    def test_delete_array_out_of_bounds_silently_ignores(self):
        """测试数组越界删除静默忽略 (5.4 节)"""
        obj = {"arr": [1, 2, 3]}
        PathResolver.delete(obj, "$.arr[10]")
        assert obj["arr"] == [1, 2, 3]


class TestPathIntersect:
    """测试路径相交判定 (11.4 节)"""
    
    def test_same_path_intersect(self):
        """测试相同路径相交"""
        assert PathResolver.intersect("$.a.b", "$.a.b") is True
    
    def test_prefix_intersect(self):
        """测试前缀路径相交"""
        assert PathResolver.intersect("$.a", "$.a.b") is True
        assert PathResolver.intersect("$.a.b", "$.a") is True
    
    def test_different_fields_not_intersect(self):
        """测试不同字段不相交"""
        assert PathResolver.intersect("$.a", "$.b") is False
        assert PathResolver.intersect("$.a.x", "$.a.y") is False
    
    def test_array_same_index_intersect(self):
        """测试相同数组下标相交"""
        assert PathResolver.intersect("$.arr[0]", "$.arr[0]") is True
        assert PathResolver.intersect("$.arr[0].x", "$.arr[0]") is True
    
    def test_array_different_index_not_intersect(self):
        """测试不同数组下标不相交 (11.4 节)"""
        assert PathResolver.intersect("$.arr[0]", "$.arr[1]") is False
        assert PathResolver.intersect("$.arr[0].x", "$.arr[1].x") is False
    
    def test_array_and_element_intersect(self):
        """测试数组与其元素相交 (11.4 节)"""
        assert PathResolver.intersect("$.arr", "$.arr[0]") is True
    
    def test_complex_paths(self):
        """测试复杂路径相交"""
        assert PathResolver.intersect("$.a[0].b", "$.a[0]") is True
        assert PathResolver.intersect("$.a[0].b", "$.a[1]") is False


class TestPathSegment:
    """测试 PathSegment 类"""
    
    def test_field_segment(self):
        """测试字段段"""
        seg = PathSegment("field")
        assert seg.key == "field"
        assert not seg.is_array_index()
        assert repr(seg) == ".field"
    
    def test_array_segment(self):
        """测试数组段"""
        seg = PathSegment(42)
        assert seg.key == 42
        assert seg.is_array_index()
        assert repr(seg) == "[42]"
