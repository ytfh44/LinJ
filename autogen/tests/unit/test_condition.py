"""
ConditionEvaluator Unit Tests

Tests condition expression evaluation as defined in Section 14
"""

import pytest
from linj_autogen.executor.evaluator import ConditionEvaluator, evaluate_condition
from linj_autogen.core.errors import ConditionError


class TestBasicComparisons:
    """Test basic comparison operations"""

    def test_equal(self):
        """Test equality comparison"""
        state = {"a": 5}
        assert evaluate_condition("$.a == 5", state) is True
        assert evaluate_condition("$.a == 3", state) is False

    def test_not_equal(self):
        """Test inequality comparison"""
        state = {"a": 5}
        assert evaluate_condition("$.a != 3", state) is True
        assert evaluate_condition("$.a != 5", state) is False

    def test_greater_than(self):
        """Test greater than"""
        state = {"a": 5}
        assert evaluate_condition("$.a > 3", state) is True
        assert evaluate_condition("$.a > 5", state) is False

    def test_greater_equal(self):
        """Test greater than or equal"""
        state = {"a": 5}
        assert evaluate_condition("$.a >= 5", state) is True
        assert evaluate_condition("$.a >= 6", state) is False

    def test_less_than(self):
        """Test less than"""
        state = {"a": 5}
        assert evaluate_condition("$.a < 10", state) is True
        assert evaluate_condition("$.a < 5", state) is False

    def test_less_equal(self):
        """Test less than or equal"""
        state = {"a": 5}
        assert evaluate_condition("$.a <= 5", state) is True
        assert evaluate_condition("$.a <= 4", state) is False

    def test_string_comparison(self):
        """Test string comparison"""
        state = {"s": "hello"}
        assert evaluate_condition('$.s == "hello"', state) is True
        assert evaluate_condition('$.s != "world"', state) is True


class TestNullHandling:
    """Test null handling (Section 14.2)"""

    def test_null_equal(self):
        """Test null equality comparison"""
        state = {"a": None}
        assert evaluate_condition("$.a == null", state) is True
        assert evaluate_condition("$.a != null", state) is False

    def test_null_other_comparison_returns_false(self):
        """Test null in other comparisons returns false"""
        state = {"a": None}
        assert evaluate_condition("$.a > 0", state) is False
        assert evaluate_condition("$.a < 0", state) is False
        assert evaluate_condition("$.a >= 0", state) is False
        assert evaluate_condition("$.a <= 0", state) is False

    def test_nonexistent_path_is_null(self):
        """Test nonexistent path is treated as null"""
        state = {}
        assert evaluate_condition("$.nonexistent == null", state) is True


class TestLogicalOperators:
    """Test logical operators (short-circuit evaluation)"""

    def test_and(self):
        """Test AND"""
        state = {"a": 5, "b": 10}
        assert evaluate_condition("$.a == 5 AND $.b == 10", state) is True
        assert evaluate_condition("$.a == 5 AND $.b == 20", state) is False
        assert evaluate_condition("$.a == 0 AND $.b == 10", state) is False

    def test_or(self):
        """Test OR"""
        state = {"a": 5, "b": 10}
        assert evaluate_condition("$.a == 5 OR $.b == 20", state) is True
        assert evaluate_condition("$.a == 0 OR $.b == 10", state) is True
        assert evaluate_condition("$.a == 0 OR $.b == 0", state) is False

    def test_not(self):
        """Test NOT"""
        state = {"a": 5}
        assert evaluate_condition("NOT $.a == 3", state) is True
        assert evaluate_condition("NOT $.a == 5", state) is False

    def test_and_short_circuit(self):
        """Test AND short-circuit (right side not evaluated if left is false)"""
        state = {"a": 0}
        # If AND doesn't short-circuit, it would try to access $.nonexistent.x causing issues
        # But with short-circuit, the right side is not evaluated
        result = evaluate_condition("$.a == 1 AND $.nonexistent.x == 1", state)
        assert result is False

    def test_or_short_circuit(self):
        """Test OR short-circuit (right side not evaluated if left is true)"""
        state = {"a": 1}
        result = evaluate_condition("$.a == 1 OR $.nonexistent.x == 1", state)
        assert result is True

    def test_complex_logic(self):
        """Test complex logic combinations"""
        state = {"a": 5, "b": 10, "c": 15}
        assert evaluate_condition("$.a < $.b AND $.b < $.c", state) is True
        assert evaluate_condition("$.a > $.b OR $.b < $.c", state) is True
        assert evaluate_condition("NOT ($.a > $.b)", state) is True


class TestFunctions:
    """Test functions"""

    def test_exists_true(self):
        """Test exists returns true"""
        state = {"a": 1}
        assert evaluate_condition("exists($.a)", state) is True

    def test_exists_false_for_null(self):
        """Test exists returns false for null"""
        state = {"a": None}
        assert evaluate_condition("exists($.a)", state) is False

    def test_exists_false_for_missing(self):
        """Test exists returns false for missing path"""
        state = {}
        assert evaluate_condition("exists($.nonexistent)", state) is False

    def test_len_array(self):
        """Test len for array length"""
        state = {"arr": [1, 2, 3]}
        assert evaluate_condition("len($.arr) == 3", state) is True
        assert evaluate_condition("len($.arr) > 0", state) is True

    def test_len_not_array(self):
        """Test len returns 0 for non-array"""
        state = {"a": "string"}
        assert evaluate_condition("len($.a) == 0", state) is True

    def test_len_missing(self):
        """Test len returns 0 for missing path"""
        state = {}
        assert evaluate_condition("len($.nonexistent) == 0", state) is True

    def test_value_function(self):
        """Test value function"""
        state = {"a": 42}
        assert evaluate_condition("value($.a) == 42", state) is True
        assert evaluate_condition("value($.nonexistent) == null", state) is True


class TestTypeErrors:
    """Test type errors"""

    def test_type_mismatch_raises_error(self):
        """Test type mismatch raises error"""
        state = {"a": 5}
        with pytest.raises(ConditionError):
            evaluate_condition('$.a == "string"', state)

    def test_string_number_mismatch(self):
        """Test string and number type mismatch"""
        state = {"s": "5"}
        with pytest.raises(ConditionError):
            evaluate_condition("$.s == 5", state)


class TestNestedPaths:
    """Test nested paths"""

    def test_nested_object(self):
        """Test nested object access"""
        state = {"a": {"b": 42}}
        assert evaluate_condition("$.a.b == 42", state) is True

    def test_array_element(self):
        """Test array element access"""
        state = {"arr": [10, 20, 30]}
        assert evaluate_condition("$.arr[1] == 20", state) is True

    def test_mixed_path(self):
        """Test mixed path"""
        state = {"a": [{"b": 1}, {"b": 2}]}
        assert evaluate_condition("$.a[0].b == 1", state) is True
        assert evaluate_condition("$.a[1].b == 2", state) is True


class TestBooleanLiterals:
    """Test boolean literals"""

    def test_true_literal(self):
        """Test true literal"""
        state = {"flag": True}
        assert evaluate_condition("$.flag == true", state) is True

    def test_false_literal(self):
        """Test false literal"""
        state = {"flag": False}
        assert evaluate_condition("$.flag == false", state) is True


class TestParentheses:
    """Test parentheses precedence"""

    def test_parentheses_override_precedence(self):
        """Test parentheses override precedence"""
        state = {"a": 5, "b": 10}
        # Without parentheses: $.a == 1 OR ($.b == 10 AND $.a == 5) -> true
        # With parentheses: ($.a == 1 OR $.b == 10) AND $.a == 5 -> true
        assert evaluate_condition("($.a == 1 OR $.b == 10) AND $.a == 5", state) is True
        # Another combination
        assert (
            evaluate_condition("$.a == 1 OR ($.b == 10 AND $.a == 0)", state) is False
        )


class TestEmptyCondition:
    """Test empty condition"""

    def test_empty_string(self):
        """Test empty string is treated as true"""
        assert evaluate_condition("", {}) is True

    def test_whitespace_only(self):
        """Test whitespace only is treated as true"""
        assert evaluate_condition("   ", {}) is True


class TestSingleValue:
    """Test single value"""

    def test_single_true_value(self):
        """Test single truthy value"""
        state = {"a": True}
        assert evaluate_condition("$.a", state) is True

    def test_single_false_value(self):
        """Test single falsy value"""
        state = {"a": False}
        assert evaluate_condition("$.a", state) is False

    def test_single_nonzero_number(self):
        """Test nonzero number is truthy"""
        state = {"a": 5}
        assert evaluate_condition("$.a", state) is True

    def test_single_zero_is_false(self):
        """Test zero is falsy"""
        state = {"a": 0}
        # Note: In boolean context, 0 is False
        assert evaluate_condition("$.a", state) is False
