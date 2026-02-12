"""
合同验证系统单元测试

测试 ContractValidator 的所有功能：
- 基本类型验证
- 对象验证（required, properties）
- 数组验证（items）
- 不支持的类型（警告模式）
- 节点集成验证
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
    """ValidationResult 测试"""

    def test_default_valid_result(self):
        """测试默认有效结果"""
        result = ValidationResult()
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error_makes_invalid(self):
        """添加错误应使结果无效"""
        result = ValidationResult()
        error = ContractValidationError(
            path="$.field",
            expected="string",
            actual="number",
            message="Type mismatch"
        )
        result.add_error(error)
        assert result.valid is False
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        """添加警告应保持有效"""
        result = ValidationResult()
        result.add_warning("Unsupported type warning")
        assert result.valid is True
        assert len(result.warnings) == 1

    def test_merge_results(self):
        """测试合并结果"""
        result1 = ValidationResult(valid=True)
        result1.add_warning("Warning 1")

        result2 = ValidationResult(valid=False)
        result2.add_error(ContractValidationError(
            path="$.x", expected="string", actual="number", message="Error"
        ))

        merged = result1.merge(result2)
        assert merged.valid is False
        assert len(merged.errors) == 1
        assert len(merged.warnings) == 1

    def test_raise_if_invalid_raises(self):
        """无效结果应抛出异常"""
        result = ValidationResult(valid=False)
        result.add_error(ContractValidationError(
            path="$.x", expected="string", actual="number", message="Error"
        ))
        with pytest.raises(ContractViolation):
            result.raise_if_invalid("test_node")

    def test_raise_if_invalid_passes(self):
        """有效结果不应抛出异常"""
        result = ValidationResult(valid=True)
        result.raise_if_invalid("test_node")  # 不应抛出


class TestBasicTypeValidation:
    """基本类型验证测试"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_string_type_valid(self, validator):
        """验证字符串类型 - 有效"""
        contract = Contract(type="string")
        result = validator.validate("hello", contract)
        assert result.valid is True

    def test_string_type_invalid(self, validator):
        """验证字符串类型 - 无效"""
        contract = Contract(type="string")
        result = validator.validate(123, contract)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].expected == "string"

    def test_number_type_valid_int(self, validator):
        """验证数字类型 - 整数有效"""
        contract = Contract(type="number")
        result = validator.validate(42, contract)
        assert result.valid is True

    def test_number_type_valid_float(self, validator):
        """验证数字类型 - 浮点数有效"""
        contract = Contract(type="number")
        result = validator.validate(3.14, contract)
        assert result.valid is True

    def test_number_type_invalid(self, validator):
        """验证数字类型 - 字符串无效"""
        contract = Contract(type="number")
        result = validator.validate("123", contract)
        assert result.valid is False

    def test_number_type_excludes_bool(self, validator):
        """验证数字类型 - 布尔值无效（bool 是 int 子类）"""
        contract = Contract(type="number")
        result = validator.validate(True, contract)
        assert result.valid is False

    def test_boolean_type_valid(self, validator):
        """验证布尔类型 - 有效"""
        contract = Contract(type="boolean")
        result = validator.validate(True, contract)
        assert result.valid is True
        result = validator.validate(False, contract)
        assert result.valid is True

    def test_boolean_type_invalid(self, validator):
        """验证布尔类型 - 无效"""
        contract = Contract(type="boolean")
        result = validator.validate(1, contract)  # int 不是 bool
        assert result.valid is False

    def test_null_type_valid(self, validator):
        """验证 null 类型 - None 有效"""
        contract = Contract(type="null")
        result = validator.validate(None, contract)
        assert result.valid is True

    def test_null_type_invalid(self, validator):
        """验证 null 类型 - 非 None 无效"""
        contract = Contract(type="null")
        result = validator.validate("not null", contract)
        assert result.valid is False

    def test_value_null_with_non_null_contract(self, validator):
        """值为 null 但合同要求非 null 类型"""
        contract = Contract(type="string")
        result = validator.validate(None, contract)
        assert result.valid is False
        assert result.errors[0].message == "Expected type 'string', but got null"


class TestObjectValidation:
    """对象类型验证测试"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_object_type_valid(self, validator):
        """验证对象类型 - 有效"""
        contract = Contract(type="object")
        result = validator.validate({"a": 1}, contract)
        assert result.valid is True

    def test_object_type_invalid(self, validator):
        """验证对象类型 - 数组无效"""
        contract = Contract(type="object")
        result = validator.validate([1, 2, 3], contract)
        assert result.valid is False

    def test_required_fields_present(self, validator):
        """验证 required 字段 - 都存在"""
        contract = Contract(
            type="object",
            required=["name", "age"]
        )
        result = validator.validate({"name": "John", "age": 30}, contract)
        assert result.valid is True

    def test_required_field_missing(self, validator):
        """验证 required 字段 - 缺失"""
        contract = Contract(
            type="object",
            required=["name", "age"]
        )
        result = validator.validate({"name": "John"}, contract)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].path == "$.age"

    def test_nested_properties_valid(self, validator):
        """验证嵌套 properties - 有效"""
        contract = Contract(
            type="object",
            properties={
                "name": Contract(type="string"),
                "age": Contract(type="number")
            }
        )
        result = validator.validate({"name": "John", "age": 30}, contract)
        assert result.valid is True

    def test_nested_properties_invalid(self, validator):
        """验证嵌套 properties - 无效"""
        contract = Contract(
            type="object",
            properties={
                "name": Contract(type="string"),
                "age": Contract(type="number")
            }
        )
        result = validator.validate({"name": "John", "age": "thirty"}, contract)
        assert result.valid is False
        assert result.errors[0].path == "$.age"

    def test_deeply_nested_object(self, validator):
        """验证深度嵌套对象"""
        contract = Contract(
            type="object",
            properties={
                "user": Contract(
                    type="object",
                    properties={
                        "name": Contract(type="string")
                    }
                )
            }
        )
        result = validator.validate({"user": {"name": "John"}}, contract)
        assert result.valid is True

    def test_required_and_properties_combined(self, validator):
        """同时验证 required 和 properties"""
        contract = Contract(
            type="object",
            required=["name"],
            properties={
                "name": Contract(type="string"),
                "age": Contract(type="number")
            }
        )
        # 缺失 required 字段
        result = validator.validate({}, contract)
        assert result.valid is False
        assert any(e.path == "$.name" for e in result.errors)

        # 类型不匹配
        result = validator.validate({"name": 123}, contract)
        assert result.valid is False
        assert any("$.name" in e.path for e in result.errors)


class TestArrayValidation:
    """数组类型验证测试"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_array_type_valid(self, validator):
        """验证数组类型 - 有效"""
        contract = Contract(type="array")
        result = validator.validate([1, 2, 3], contract)
        assert result.valid is True

    def test_array_type_invalid(self, validator):
        """验证数组类型 - 对象无效"""
        contract = Contract(type="array")
        result = validator.validate({"a": 1}, contract)
        assert result.valid is False

    def test_items_string_array(self, validator):
        """验证 items - 字符串数组"""
        contract = Contract(
            type="array",
            items=Contract(type="string")
        )
        result = validator.validate(["a", "b", "c"], contract)
        assert result.valid is True

    def test_items_number_array(self, validator):
        """验证 items - 数字数组"""
        contract = Contract(
            type="array",
            items=Contract(type="number")
        )
        result = validator.validate([1, 2, 3], contract)
        assert result.valid is True

    def test_items_invalid_element(self, validator):
        """验证 items - 元素类型不匹配"""
        contract = Contract(
            type="array",
            items=Contract(type="string")
        )
        result = validator.validate(["a", 123, "c"], contract)
        assert result.valid is False
        assert result.errors[0].path == "$[1]"

    def test_nested_array(self, validator):
        """验证嵌套数组"""
        contract = Contract(
            type="array",
            items=Contract(
                type="array",
                items=Contract(type="number")
            )
        )
        result = validator.validate([[1, 2], [3, 4]], contract)
        assert result.valid is True

    def test_object_array(self, validator):
        """验证对象数组"""
        contract = Contract(
            type="array",
            items=Contract(
                type="object",
                properties={
                    "name": Contract(type="string")
                }
            )
        )
        result = validator.validate(
            [{"name": "Alice"}, {"name": "Bob"}],
            contract
        )
        assert result.valid is True


class TestUnsupportedTypes:
    """不支持的类型测试"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_unsupported_type_generates_warning(self, validator):
        """不支持的类型应生成警告而非错误"""
        contract = Contract(type="integer")  # 不支持 integer，只支持 number
        result = validator.validate(42, contract)
        assert result.valid is True  # 不阻塞执行
        assert len(result.warnings) == 1
        assert "integer" in result.warnings[0]

    def test_unsupported_type_allows_any_value(self, validator):
        """不支持的类型应允许任何值通过"""
        contract = Contract(type="custom_type")
        result = validator.validate({"anything": "goes"}, contract)
        assert result.valid is True
        assert len(result.warnings) == 1


class TestNodeIntegration:
    """节点集成验证测试"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    @pytest.fixture
    def state_manager(self) -> StateManager:
        return StateManager({
            "user": {"name": "John", "age": 30},
            "items": ["a", "b", "c"],
            "count": 42
        })

    def test_validate_in_contract_simple(self, validator, state_manager):
        """验证 in_contract - 简单类型"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="output",
            reads=["$.count"],
            in_contract=Contract(type="number")
        )
        result = validator.validate_in_contract(node, state_manager)
        assert result.valid is True

    def test_validate_in_contract_object(self, validator, state_manager):
        """验证 in_contract - 对象类型"""
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
                    "age": Contract(type="number")
                }
            )
        )
        result = validator.validate_in_contract(node, state_manager)
        assert result.valid is True

    def test_validate_in_contract_no_reads(self, validator, state_manager):
        """验证 in_contract - 没有 reads 时验证整个状态"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="output",
            in_contract=Contract(type="object")
        )
        result = validator.validate_in_contract(node, state_manager)
        assert result.valid is True

    def test_validate_out_contract(self, validator, state_manager):
        """验证 out_contract"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="$.output",
            writes=["$.output"],
            out_contract=Contract(type="string")
        )
        changeset = ChangeSetBuilder().write("$.output", "Hello World").build()
        result = validator.validate_out_contract(node, changeset, state_manager)
        assert result.valid is True

    def test_validate_out_contract_type_mismatch(self, validator, state_manager):
        """验证 out_contract - 类型不匹配"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="$.output",
            writes=["$.output"],
            out_contract=Contract(type="number")  # 期望数字
        )
        changeset = ChangeSetBuilder().write("$.output", "not a number").build()
        result = validator.validate_out_contract(node, changeset, state_manager)
        assert result.valid is False

    def test_validate_no_contract(self, validator, state_manager):
        """没有合同时应返回有效结果"""
        node = HintNode(
            id="test",
            type="hint",
            template="Hello",
            write_to="output"
        )
        result = validator.validate_in_contract(node, state_manager)
        assert result.valid is True
        result = validator.validate_out_contract(node, ChangeSet(), state_manager)
        assert result.valid is True


class TestErrorReporting:
    """错误报告测试"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_path_in_error(self, validator):
        """错误应包含正确路径"""
        contract = Contract(
            type="object",
            properties={
                "level1": Contract(
                    type="object",
                    properties={
                        "level2": Contract(type="string")
                    }
                )
            }
        )
        result = validator.validate(
            {"level1": {"level2": 123}},
            contract
        )
        assert result.valid is False
        assert result.errors[0].path == "$.level1.level2"

    def test_multiple_errors_collected(self, validator):
        """应收集所有错误"""
        contract = Contract(
            type="object",
            required=["a", "b"],
            properties={
                "c": Contract(type="string")
            }
        )
        result = validator.validate(
            {"c": 123},  # 缺失 a, b，且 c 类型错误
            contract
        )
        assert result.valid is False
        assert len(result.errors) == 3  # 两个 required + 一个类型错误


class TestContractValidationEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def validator(self) -> ContractValidator:
        return ContractValidator()

    def test_empty_object_valid(self, validator):
        """空对象是有效的 object 类型"""
        contract = Contract(type="object")
        result = validator.validate({}, contract)
        assert result.valid is True

    def test_empty_array_valid(self, validator):
        """空数组是有效的 array 类型"""
        contract = Contract(type="array")
        result = validator.validate([], contract)
        assert result.valid is True

    def test_empty_string_valid(self, validator):
        """空字符串是有效的 string 类型"""
        contract = Contract(type="string")
        result = validator.validate("", contract)
        assert result.valid is True

    def test_zero_valid_number(self, validator):
        """0 是有效的 number 类型"""
        contract = Contract(type="number")
        result = validator.validate(0, contract)
        assert result.valid is True

    def test_negative_number_valid(self, validator):
        """负数是有效的 number 类型"""
        contract = Contract(type="number")
        result = validator.validate(-42.5, contract)
        assert result.valid is True

    def test_empty_array_with_items(self, validator):
        """空数组满足任何 items 合同"""
        contract = Contract(
            type="array",
            items=Contract(type="string")
        )
        result = validator.validate([], contract)
        assert result.valid is True
