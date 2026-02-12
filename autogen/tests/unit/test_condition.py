"""
ConditionEvaluator 单元测试

测试 14 节定义的条件表达式求值
"""

import pytest
from linj_autogen.executor.evaluator import ConditionEvaluator, evaluate_condition
from linj_autogen.core.errors import ConditionError


class TestBasicComparisons:
    """测试基本比较运算"""
    
    def test_equal(self):
        """测试相等比较"""
        state = {"a": 5}
        assert evaluate_condition("$.a == 5", state) is True
        assert evaluate_condition("$.a == 3", state) is False
    
    def test_not_equal(self):
        """测试不等比较"""
        state = {"a": 5}
        assert evaluate_condition("$.a != 3", state) is True
        assert evaluate_condition("$.a != 5", state) is False
    
    def test_greater_than(self):
        """测试大于"""
        state = {"a": 5}
        assert evaluate_condition("$.a > 3", state) is True
        assert evaluate_condition("$.a > 5", state) is False
    
    def test_greater_equal(self):
        """测试大于等于"""
        state = {"a": 5}
        assert evaluate_condition("$.a >= 5", state) is True
        assert evaluate_condition("$.a >= 6", state) is False
    
    def test_less_than(self):
        """测试小于"""
        state = {"a": 5}
        assert evaluate_condition("$.a < 10", state) is True
        assert evaluate_condition("$.a < 5", state) is False
    
    def test_less_equal(self):
        """测试小于等于"""
        state = {"a": 5}
        assert evaluate_condition("$.a <= 5", state) is True
        assert evaluate_condition("$.a <= 4", state) is False
    
    def test_string_comparison(self):
        """测试字符串比较"""
        state = {"s": "hello"}
        assert evaluate_condition('$.s == "hello"', state) is True
        assert evaluate_condition('$.s != "world"', state) is True


class TestNullHandling:
    """测试 null 处理 (14.2 节)"""
    
    def test_null_equal(self):
        """测试 null 相等比较"""
        state = {"a": None}
        assert evaluate_condition("$.a == null", state) is True
        assert evaluate_condition("$.a != null", state) is False
    
    def test_null_other_comparison_returns_false(self):
        """测试 null 参与其他比较返回 false"""
        state = {"a": None}
        assert evaluate_condition("$.a > 0", state) is False
        assert evaluate_condition("$.a < 0", state) is False
        assert evaluate_condition("$.a >= 0", state) is False
        assert evaluate_condition("$.a <= 0", state) is False
    
    def test_nonexistent_path_is_null(self):
        """测试不存在路径视为 null"""
        state = {}
        assert evaluate_condition("$.nonexistent == null", state) is True


class TestLogicalOperators:
    """测试逻辑运算符（短路求值）"""
    
    def test_and(self):
        """测试 AND"""
        state = {"a": 5, "b": 10}
        assert evaluate_condition("$.a == 5 AND $.b == 10", state) is True
        assert evaluate_condition("$.a == 5 AND $.b == 20", state) is False
        assert evaluate_condition("$.a == 0 AND $.b == 10", state) is False
    
    def test_or(self):
        """测试 OR"""
        state = {"a": 5, "b": 10}
        assert evaluate_condition("$.a == 5 OR $.b == 20", state) is True
        assert evaluate_condition("$.a == 0 OR $.b == 10", state) is True
        assert evaluate_condition("$.a == 0 OR $.b == 0", state) is False
    
    def test_not(self):
        """测试 NOT"""
        state = {"a": 5}
        assert evaluate_condition("NOT $.a == 3", state) is True
        assert evaluate_condition("NOT $.a == 5", state) is False
    
    def test_and_short_circuit(self):
        """测试 AND 短路（左边为假不求右边）"""
        state = {"a": 0}
        # 如果 AND 不短路，会尝试访问 $.nonexistent.x 导致问题
        # 但短路时不会求右边
        result = evaluate_condition("$.a == 1 AND $.nonexistent.x == 1", state)
        assert result is False
    
    def test_or_short_circuit(self):
        """测试 OR 短路（左边为真不求右边）"""
        state = {"a": 1}
        result = evaluate_condition("$.a == 1 OR $.nonexistent.x == 1", state)
        assert result is True
    
    def test_complex_logic(self):
        """测试复杂逻辑组合"""
        state = {"a": 5, "b": 10, "c": 15}
        assert evaluate_condition("$.a < $.b AND $.b < $.c", state) is True
        assert evaluate_condition("$.a > $.b OR $.b < $.c", state) is True
        assert evaluate_condition("NOT ($.a > $.b)", state) is True


class TestFunctions:
    """测试函数"""
    
    def test_exists_true(self):
        """测试 exists 为真"""
        state = {"a": 1}
        assert evaluate_condition("exists($.a)", state) is True
    
    def test_exists_false_for_null(self):
        """测试 exists 对 null 为假"""
        state = {"a": None}
        assert evaluate_condition("exists($.a)", state) is False
    
    def test_exists_false_for_missing(self):
        """测试 exists 对缺失路径为假"""
        state = {}
        assert evaluate_condition("exists($.nonexistent)", state) is False
    
    def test_len_array(self):
        """测试 len 数组长度"""
        state = {"arr": [1, 2, 3]}
        assert evaluate_condition("len($.arr) == 3", state) is True
        assert evaluate_condition("len($.arr) > 0", state) is True
    
    def test_len_not_array(self):
        """测试 len 对非数组返回 0"""
        state = {"a": "string"}
        assert evaluate_condition("len($.a) == 0", state) is True
    
    def test_len_missing(self):
        """测试 len 对缺失路径返回 0"""
        state = {}
        assert evaluate_condition("len($.nonexistent) == 0", state) is True
    
    def test_value_function(self):
        """测试 value 函数"""
        state = {"a": 42}
        assert evaluate_condition("value($.a) == 42", state) is True
        assert evaluate_condition("value($.nonexistent) == null", state) is True


class TestTypeErrors:
    """测试类型错误"""
    
    def test_type_mismatch_raises_error(self):
        """测试类型不匹配产生错误"""
        state = {"a": 5}
        with pytest.raises(ConditionError):
            evaluate_condition('$.a == "string"', state)
    
    def test_string_number_mismatch(self):
        """测试字符串和数字类型不匹配"""
        state = {"s": "5"}
        with pytest.raises(ConditionError):
            evaluate_condition("$.s == 5", state)


class TestNestedPaths:
    """测试嵌套路径"""
    
    def test_nested_object(self):
        """测试嵌套对象访问"""
        state = {"a": {"b": 42}}
        assert evaluate_condition("$.a.b == 42", state) is True
    
    def test_array_element(self):
        """测试数组元素访问"""
        state = {"arr": [10, 20, 30]}
        assert evaluate_condition("$.arr[1] == 20", state) is True
    
    def test_mixed_path(self):
        """测试混合路径"""
        state = {"a": [{"b": 1}, {"b": 2}]}
        assert evaluate_condition("$.a[0].b == 1", state) is True
        assert evaluate_condition("$.a[1].b == 2", state) is True


class TestBooleanLiterals:
    """测试布尔字面量"""
    
    def test_true_literal(self):
        """测试 true 字面量"""
        state = {"flag": True}
        assert evaluate_condition("$.flag == true", state) is True
    
    def test_false_literal(self):
        """测试 false 字面量"""
        state = {"flag": False}
        assert evaluate_condition("$.flag == false", state) is True


class TestParentheses:
    """测试括号优先级"""
    
    def test_parentheses_override_precedence(self):
        """测试括号覆盖优先级"""
        state = {"a": 5, "b": 10}
        # 不加括号：$.a == 1 OR ($.b == 10 AND $.a == 5) -> true
        # 加括号后：($.a == 1 OR $.b == 10) AND $.a == 5 -> true
        assert evaluate_condition("($.a == 1 OR $.b == 10) AND $.a == 5", state) is True
        # 另一种组合
        assert evaluate_condition("$.a == 1 OR ($.b == 10 AND $.a == 0)", state) is False


class TestEmptyCondition:
    """测试空条件"""
    
    def test_empty_string(self):
        """测试空字符串视为真"""
        assert evaluate_condition("", {}) is True
    
    def test_whitespace_only(self):
        """测试仅空白字符视为真"""
        assert evaluate_condition("   ", {}) is True


class TestSingleValue:
    """测试单个值"""
    
    def test_single_true_value(self):
        """测试单个真值"""
        state = {"a": True}
        assert evaluate_condition("$.a", state) is True
    
    def test_single_false_value(self):
        """测试单个假值"""
        state = {"a": False}
        assert evaluate_condition("$.a", state) is False
    
    def test_single_nonzero_number(self):
        """测试非零数字为真"""
        state = {"a": 5}
        assert evaluate_condition("$.a", state) is True
    
    def test_single_zero_is_false(self):
        """测试零为假"""
        state = {"a": 0}
        # 注意：在布尔上下文中，0 是 False
        assert evaluate_condition("$.a", state) is False
