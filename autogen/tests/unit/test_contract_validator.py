"""
Unit tests: Contract Validation System

Tests all ContractValidator features:
- Basic type validation
- Object validation (required, properties)
- Array validation (items)
- Unsupported types (warning mode)
- Node integration validation
"""

import pytest
from typing import Any, Dict, List

from linj_autogen.core.contract_validator import (
    ContractValidator,
    ValidationResult,
    ContractValidationError,
)
from linj_autogen.core.nodes import Contract, HintNode, ToolNode, ToolCall
from linj_autogen.core.state import StateManager
from linj_autogen.core.changeset import ChangeSet, ChangeSetBuilder
from linj_autogen.core.errors import ContractViolation


class TestValidationResult:
    """ValidationResult tests"""

    def test_default_valid_result(self):
        """Test default valid result"""
        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_makes_invalid(self):
        """Adding error should make result invalid"""
        result = ValidationResult()
        error = ContractValidationError(
            path="$.field", expected="string", actual="number", message="Type mismatch"
        )
        result.add_error(error)
        assert result.valid is False
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """Adding warning should keep result valid"""
        result = ValidationResult()
        result.add_warning("Unsupported type warning")
        assert result.valid is True
        assert len(result.warnings) == 1

    def test_merge_results(self):
        """Test merging results"""
        result1 = ValidationResult(valid=True)
        result1.add_warning("Warning 1")

        result2 = ValidationResult(valid=False)
        result2.add_error(
            ContractValidationError(
                path="$.x", expected="string", actual="number", message="Error"
            )
        )

        merged = result1.merge(result2)
        assert merged.valid is False
        assert len(merged.errors) == 1
        assert len(merged.warnings) == 1

    def test_raise_if_invalid_raises(self):
        """Invalid result should raise exception"""
        result = ValidationResult(valid=False)
        result.add_error(
            ContractValidationError(
                path="$.x", expected="string", actual="number", message="Error"
            )
        )
        with pytest.raises(ContractViolation):
            result.raise_if_invalid("test_node")

    def test_raise_if_invalid_passes(self):
        """Valid result should not raise exception"""
        result = ValidationResult(valid=True)
        result.raise_if_invalid("test_node")  # Should not raise


class TestBasicTypeValidation:
    """Basic type validation tests"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_string_type_valid(self, validator):
        """Validate string type - valid"""
        contract = Contract(type="string")
        result = validator.validate("hello", contract)
        assert result.valid is True

    def test_string_type_invalid(self, validator):
        """Validate string type - invalid"""
        contract = Contract(type="string")
        result = validator.validate(123, contract)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].expected == "string"

    def test_number_type_valid_int(self, validator):
        """Validate number type - integer valid"""
        contract = Contract(type="number")
        result = validator.validate(42, contract)
        assert result.valid is True

    def test_number_type_valid_float(self, validator):
        """Validate number type - float valid"""
        contract = Contract(type="number")
        result = validator.validate(3.14, contract)
        assert result.valid is True

    def test_number_type_invalid(self, validator):
        """Validate number type - string invalid"""
        contract = Contract(type="number")
        result = validator.validate("123", contract)
        assert result.valid is False

    def test_number_type_excludes_bool(self, validator):
        """Validate number type - boolean invalid (bool is int subclass)"""
        contract = Contract(type="number")
        result = validator.validate(True, contract)
        assert result.valid is False

    def test_boolean_type_valid(self, validator):
        """Validate boolean type - valid"""
        contract = Contract(type="boolean")
        result = validator.validate(True, contract)
        assert result.valid is True
        result = validator.validate(False, contract)
        assert result.valid is True

    def test_boolean_type_invalid(self, validator):
        """Validate boolean type - invalid"""
        contract = Contract(type="boolean")
        result = validator.validate(1, contract)  # int is not bool
        assert result.valid is False

    def test_null_type_valid(self, validator):
        """Validate null type - None valid"""
        contract = Contract(type="null")
        result = validator.validate(None, contract)
        assert result.valid is True

    def test_null_type_invalid(self, validator):
        """Validate null type - non-None invalid"""
        contract = Contract(type="null")
        result = validator.validate("not null", contract)
        assert result.valid is False

    def test_value_null_with_non_null_contract(self, validator):
        """Value is null but contract requires non-null type"""
        contract = Contract(type="string")
        result = validator.validate(None, contract)
        assert result.valid is False
        assert result.errors[0].message == "Expected type 'string', but got null"


class TestObjectValidation:
    """Object type validation tests"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_object_type_valid(self, validator):
        """Validate object type - valid"""
        contract = Contract(type="object")
        result = validator.validate({"a": 1}, contract)
        assert result.valid is True

    def test_object_type_invalid(self, validator):
        """Validate object type - array invalid"""
        contract = Contract(type="object")
        result = validator.validate([1, 2, 3], contract)
        assert result.valid is False

    def test_required_fields_present(self, validator):
        """Validate required fields - all present"""
        contract = Contract(type="object", required=["name", "age"])
        result = validator.validate({"name": "John", "age": 30}, contract)
        assert result.valid is True

    def test_required_field_missing(self, validator):
        """Validate required fields - missing"""
        contract = Contract(type="object", required=["name", "age"])
        result = validator.validate({"name": "John"}, contract)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].path == "$.age"

    def test_nested_properties_valid(self, validator):
        """Validate nested properties - valid"""
        contract = Contract(
            type="object",
            properties={
                "name": Contract(type="string"),
                "age": Contract(type="number"),
            },
        )
        result = validator.validate({"name": "John", "age": 30}, contract)
        assert result.valid is True

    def test_nested_properties_invalid(self, validator):
        """Validate nested properties - invalid"""
        contract = Contract(
            type="object",
            properties={
                "name": Contract(type="string"),
                "age": Contract(type="number"),
            },
        )
        result = validator.validate({"name": "John", "age": "thirty"}, contract)
        assert result.valid is False
        assert result.errors[0].path == "$.age"

    def test_deeply_nested_object(self, validator):
        """Validate deeply nested object"""
        contract = Contract(
            type="object",
            properties={
                "user": Contract(
                    type="object", properties={"name": Contract(type="string")}
                )
            },
        )
        result = validator.validate({"user": {"name": "John"}}, contract)
        assert result.valid is True

    def test_required_and_properties_combined(self, validator):
        """Validate required and properties together"""
        contract = Contract(
            type="object",
            required=["name"],
            properties={
                "name": Contract(type="string"),
                "age": Contract(type="number"),
            },
        )
        # Missing required field
        result = validator.validate({}, contract)
        assert result.valid is False
        assert any(e.path == "$.name" for e in result.errors)

        # Type mismatch
        result = validator.validate({"name": 123}, contract)
        assert result.valid is False
        assert any("$.name" in e.path for e in result.errors)


class TestArrayValidation:
    """Array type validation tests"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_array_type_valid(self, validator):
        """Validate array type - valid"""
        contract = Contract(type="array")
        result = validator.validate([1, 2, 3], contract)
        assert result.valid is True

    def test_array_type_invalid(self, validator):
        """Validate array type - object invalid"""
        contract = Contract(type="array")
        result = validator.validate({"a": 1}, contract)
        assert result.valid is False

    def test_items_string_array(self, validator):
        """Validate items - string array"""
        contract = Contract(type="array", items=Contract(type="string"))
        result = validator.validate(["a", "b", "c"], contract)
        assert result.valid is True

    def test_items_number_array(self, validator):
        """Validate items - number array"""
        contract = Contract(type="array", items=Contract(type="number"))
        result = validator.validate([1, 2, 3], contract)
        assert result.valid is True

    def test_items_invalid_element(self, validator):
        """Validate items - element type mismatch"""
        contract = Contract(type="array", items=Contract(type="string"))
        result = validator.validate(["a", 123, "c"], contract)
        assert result.valid is False
        assert result.errors[0].path == "$[1]"

    def test_nested_array(self, validator):
        """Validate nested array"""
        contract = Contract(
            type="array", items=Contract(type="array", items=Contract(type="number"))
        )
        result = validator.validate([[1, 2], [3, 4]], contract)
        assert result.valid is True

    def test_object_array(self, validator):
        """Validate object array"""
        contract = Contract(
            type="array",
            items=Contract(type="object", properties={"name": Contract(type="string")}),
        )
        result = validator.validate([{"name": "Alice"}, {"name": "Bob"}], contract)
        assert result.valid is True


class TestUnsupportedTypes:
    """Unsupported types tests"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_unsupported_type_generates_warning(self, validator):
        """Unsupported types should generate warnings instead of errors"""
        contract = Contract(type="integer")  # integer not supported, only number
        result = validator.validate(42, contract)
        assert result.valid is True  # Does not block execution
        assert len(result.warnings) == 1
        assert "integer" in result.warnings[0]

    def test_unsupported_type_allows_any_value(self, validator):
        """Unsupported types should allow any value to pass through"""
        contract = Contract(type="custom_type")
        result = validator.validate({"anything": "goes"}, contract)
        assert result.valid is True
        assert len(result.warnings) == 1


class TestNodeIntegration:
    """Node integration validation tests"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    @pytest.fixture
    def state_manager(self) -> StateManager:
        return StateManager(
            {"user": {"name": "John", "age": 30}, "items": ["a", "b", "c"], "count": 42}
        )

    def test_validate_in_contract_simple(self, validator, state_manager):
        """Validate in_contract - simple types"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="output",
            reads=["$.count"],
            in_contract=Contract(type="number"),
        )
        result = validator.validate_in_contract(node, state_manager)
        assert result.valid is True

    def test_validate_in_contract_object(self, validator, state_manager):
        """Validate in_contract - object type"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello {{name}}",
            write_to="output",
            reads=["$.user"],
            in_contract=Contract(
                type="object",
                required=["name"],
                properties={
                    "name": Contract(type="string"),
                    "age": Contract(type="number"),
                },
            ),
        )
        result = validator.validate_in_contract(node, state_manager)
        assert result.valid is True

    def test_validate_in_contract_no_reads(self, validator, state_manager):
        """Validate in_contract - validate entire state when no reads"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="output",
            in_contract=Contract(type="object"),
        )
        result = validator.validate_in_contract(node, state_manager)
        assert result.valid is True

    def test_validate_out_contract(self, validator, state_manager):
        """Validate out_contract"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="$.output",
            writes=["$.output"],
            out_contract=Contract(type="string"),
        )
        changeset = ChangeSetBuilder().write("$.output", "Hello World").build()
        result = validator.validate_out_contract(node, changeset, state_manager)
        assert result.valid is True

    def test_validate_out_contract_type_mismatch(self, validator, state_manager):
        """Validate out_contract - type mismatch"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="$.output",
            writes=["$.output"],
            out_contract=Contract(type="number"),  # Expect number
        )
        changeset = ChangeSetBuilder().write("$.output", "not a number").build()
        result = validator.validate_out_contract(node, changeset, state_manager)
        assert result.valid is False

    def test_validate_no_contract(self, validator, state_manager):
        """No contract should return valid result"""
        node = HintNode(id="test", type="hint", template="Hello", write_to="output")
        result = validator.validate_in_contract(node, state_manager)
        assert result.valid is True
        result = validator.validate_out_contract(node, ChangeSet(), state_manager)
        assert result.valid is True


class TestErrorReporting:
    """Error reporting tests"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_path_in_error(self, validator):
        """Error should contain correct path"""
        contract = Contract(
            type="object",
            properties={
                "level1": Contract(
                    type="object", properties={"level2": Contract(type="string")}
                )
            },
        )
        result = validator.validate({"level1": {"level2": 123}}, contract)
        assert result.valid is False
        assert result.errors[0].path == "$.level1.level2"

    def test_multiple_errors_collected(self, validator):
        """Should collect all errors"""
        contract = Contract(
            type="object",
            required=["a", "b"],
            properties={"c": Contract(type="string")},
        )
        result = validator.validate(
            {"c": 123},  # Missing a, b, and c has type error
            contract,
        )
        assert result.valid is False
        assert len(result.errors) == 3  # Two required + one type error


class TestContractValidationEdgeCases:
    """Edge cases tests"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_empty_object_valid(self, validator):
        """Empty object is valid object type"""
        contract = Contract(type="object")
        result = validator.validate({}, contract)
        assert result.valid is True

    def test_empty_array_valid(self, validator):
        """Empty array is valid array type"""
        contract = Contract(type="array")
        result = validator.validate([], contract)
        assert result.valid is True

    def test_empty_string_valid(self, validator):
        """Empty string is valid string type"""
        contract = Contract(type="string")
        result = validator.validate("", contract)
        assert result.valid is True

    def test_zero_valid_number(self, validator):
        """0 is valid number type"""
        contract = Contract(type="number")
        result = validator.validate(0, contract)
        assert result.valid is True

    def test_negative_number_valid(self, validator):
        """Negative number is valid number type"""
        contract = Contract(type="number")
        result = validator.validate(-42.5, contract)
        assert result.valid is True

    def test_empty_array_with_items(self, validator):
        """Empty array satisfies any items contract"""
        contract = Contract(type="array", items=Contract(type="string"))
        result = validator.validate([], contract)
        assert result.valid is True
