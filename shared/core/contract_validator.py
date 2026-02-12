"""
合同验证系统

实现规范 7.1 节定义的最小合同语言验证
支持类型：object/array/string/number/boolean/null
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Set

from .nodes import Contract, Node
from .changeset import ChangeSet
from .state import StateManager
from .path import PathResolver
from ..exceptions.errors import ContractViolation

logger = logging.getLogger(__name__)


class ContractValidationError:
    """合同验证错误详情"""

    def __init__(self, path: str, expected: str, actual: Any, message: str):
        self.path = path
        self.expected = expected
        self.actual = actual
        self.message = message

    def __repr__(self) -> str:
        return (
            f"ContractValidationError("
            f"path={self.path!r}, "
            f"expected={self.expected!r}, "
            f"actual={self.actual!r}, "
            f"message={self.message!r}"
            f")"
        )


class ValidationResult:
    """
    验证结果

    Attributes:
        valid: 验证是否通过
        errors: 验证错误列表
        warnings: 不可验证部分的警告信息
    """

    def __init__(
        self,
        valid: bool = True,
        errors: Optional[List[ContractValidationError]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []

    def add_error(self, error: ContractValidationError) -> None:
        """添加验证错误"""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """添加警告"""
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """合并另一个验证结果到当前结果"""
        self.valid = self.valid and other.valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self

    def raise_if_invalid(self, node_id: str) -> None:
        """如果验证失败，抛出 ContractViolation"""
        if not self.valid:
            error_details = {
                "node_id": node_id,
                "violations": [
                    {
                        "path": e.path,
                        "expected": e.expected,
                        "actual": e.actual,
                        "message": e.message,
                    }
                    for e in self.errors
                ],
            }
            raise ContractViolation(
                f"Contract validation failed for node {node_id}", error_details
            )


class ContractValidator:
    """
    合同验证器

    实现 7.1 节定义的最小合同语言验证
    支持的类型：object, array, string, number, boolean, null
    """

    SUPPORTED_TYPES: Set[str] = {
        "object",
        "array",
        "string",
        "number",
        "boolean",
        "null",
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate(
        self, value: Any, contract: Contract, path: str = "$"
    ) -> ValidationResult:
        """
        验证值是否符合合同

        Args:
            value: 待验证的值
            contract: 合同定义
            path: 当前验证路径（用于错误报告）

        Returns:
            ValidationResult: 验证结果
        """
        # 检查合同类型是否支持
        if contract.type not in self.SUPPORTED_TYPES:
            # 不可验证的部分标记为警告，不阻塞执行
            result = ValidationResult(valid=True)
            result.add_warning(
                f"Unsupported contract type '{contract.type}' at path '{path}', "
                "skipping validation"
            )
            self.logger.warning(
                "Unsupported contract type '%s' at path '%s'", contract.type, path
            )
            return result

        # 处理 null 值
        if value is None:
            if contract.type == "null":
                return ValidationResult(valid=True)
            else:
                result = ValidationResult(valid=False)
                result.add_error(
                    ContractValidationError(
                        path=path,
                        expected=contract.type,
                        actual="null",
                        message=f"Expected type '{contract.type}', but got null",
                    )
                )
                return result

        # 根据类型进行验证
        if contract.type == "object":
            return self._validate_object(value, contract, path)
        elif contract.type == "array":
            return self._validate_array(value, contract, path)
        elif contract.type == "string":
            return self._validate_scalar(value, str, "string", path)
        elif contract.type == "number":
            return self._validate_number(value, path)
        elif contract.type == "boolean":
            return self._validate_scalar(value, bool, "boolean", path)
        elif contract.type == "null":
            # 前面已经处理了 null 的情况，到这里说明值不是 null
            result = ValidationResult(valid=False)
            result.add_error(
                ContractValidationError(
                    path=path,
                    expected="null",
                    actual=type(value).__name__,
                    message=f"Expected null, but got {type(value).__name__}",
                )
            )
            return result

        # 不应该到达这里，但为了类型安全
        return ValidationResult(valid=True)

    def _validate_object(
        self, value: Any, contract: Contract, path: str
    ) -> ValidationResult:
        """验证对象类型"""
        result = ValidationResult()

        # 类型检查
        if not isinstance(value, dict):
            result.add_error(
                ContractValidationError(
                    path=path,
                    expected="object",
                    actual=type(value).__name__,
                    message=f"Expected object, but got {type(value).__name__}",
                )
            )
            return result

        # 检查 required 字段
        if contract.required:
            for field in contract.required:
                if field not in value:
                    result.add_error(
                        ContractValidationError(
                            path=f"{path}.{field}",
                            expected="required field",
                            actual="missing",
                            message=f"Required field '{field}' is missing",
                        )
                    )

        # 递归验证 properties
        if contract.properties:
            for field_name, field_contract in contract.properties.items():
                if field_name in value:
                    field_result = self.validate(
                        value[field_name], field_contract, f"{path}.{field_name}"
                    )
                    result.merge(field_result)

        return result

    def _validate_array(
        self, value: Any, contract: Contract, path: str
    ) -> ValidationResult:
        """验证数组类型"""
        result = ValidationResult()

        # 类型检查
        if not isinstance(value, list):
            result.add_error(
                ContractValidationError(
                    path=path,
                    expected="array",
                    actual=type(value).__name__,
                    message=f"Expected array, but got {type(value).__name__}",
                )
            )
            return result

        # 验证每个元素
        if contract.items:
            for idx, item in enumerate(value):
                item_result = self.validate(item, contract.items, f"{path}[{idx}]")
                result.merge(item_result)

        return result

    def _validate_scalar(
        self, value: Any, expected_type: type, type_name: str, path: str
    ) -> ValidationResult:
        """验证标量类型"""
        if not isinstance(value, expected_type):
            return ValidationResult(
                valid=False,
                errors=[
                    ContractValidationError(
                        path=path,
                        expected=type_name,
                        actual=type(value).__name__,
                        message=f"Expected {type_name}, but got {type(value).__name__}",
                    )
                ],
            )
        return ValidationResult(valid=True)

    def _validate_number(self, value: Any, path: str) -> ValidationResult:
        """验证数字类型（int 或 float）"""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            # bool 是 int 的子类，需要排除
            return ValidationResult(
                valid=False,
                errors=[
                    ContractValidationError(
                        path=path,
                        expected="number",
                        actual=type(value).__name__,
                        message=f"Expected number, but got {type(value).__name__}",
                    )
                ],
            )
        return ValidationResult(valid=True)

    def validate_in_contract(self, node: Node, state: StateManager) -> ValidationResult:
        """
        验证节点的 in_contract

        Args:
            node: 节点定义
            state: 状态管理器

        Returns:
            ValidationResult: 验证结果
        """
        if not node.in_contract:
            return ValidationResult(valid=True)

        self.logger.debug("Validating in_contract for node '%s'", node.id)

        # 从 reads 路径获取输入值进行验证
        # 如果节点没有定义 reads，则验证整个状态
        if node.reads:
            result = ValidationResult()
            for read_path in node.reads:
                value = state.get(read_path)
                path_result = self.validate(value, node.in_contract, read_path)
                result.merge(path_result)
            return result
        else:
            # 没有定义 reads，验证整个状态
            return self.validate(state.get_full_state(), node.in_contract, "$")

    def validate_out_contract(
        self, node: Node, changeset: ChangeSet, state: StateManager
    ) -> ValidationResult:
        """
        验证节点的 out_contract

        在变更集应用后验证相关路径的值

        Args:
            node: 节点定义
            changeset: 变更集
            state: 状态管理器

        Returns:
            ValidationResult: 验证结果
        """
        if not node.out_contract:
            return ValidationResult(valid=True)

        self.logger.debug("Validating out_contract for node '%s'", node.id)

        # 创建临时状态，应用变更集
        temp_state = copy.deepcopy(state.get_full_state())
        changeset.apply_to(temp_state)

        # 从 writes 路径获取输出值进行验证
        # 如果节点没有定义 writes，则验证整个状态
        if node.writes:
            result = ValidationResult()
            for write_path in node.writes:
                value = PathResolver.get(temp_state, write_path)
                path_result = self.validate(value, node.out_contract, write_path)
                result.merge(path_result)
            return result
        else:
            # 没有定义 writes，验证整个状态
            return self.validate(temp_state, node.out_contract, "$")
