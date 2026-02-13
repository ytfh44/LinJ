"""
Contract Validation System

Implements minimum contract language validation as defined in Section 7.1
Supported types: object/array/string/number/boolean/null
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Set

from .nodes import Contract, Node
from .changeset import ChangeSet
from .state import StateManager
from .path import PathResolver
from .errors import ContractViolation

logger = logging.getLogger(__name__)


class ContractValidationError:
    """Contract validation error details"""

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
    Validation result

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        warnings: Warnings for non-validatable parts
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
        """Add a validation error"""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning"""
        self.warnings.append(warning)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into the current result"""
        self.valid = self.valid and other.valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self

    def raise_if_invalid(self, node_id: str) -> None:
        """Raise ContractViolation if validation failed"""
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
    Contract validator

    Implements minimum contract language validation as defined in Section 7.1
    Supported types: object, array, string, number, boolean, null
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
        Validate if a value conforms to the contract

        Args:
            value: The value to validate
            contract: The contract definition
            path: Current validation path (for error reporting)

        Returns:
            ValidationResult: The validation result
        """
        # Check if contract type is supported
        if contract.type not in self.SUPPORTED_TYPES:
            # Non-validatable parts are marked as warnings, don't block execution
            result = ValidationResult(valid=True)
            result.add_warning(
                f"Unsupported contract type '{contract.type}' at path '{path}', "
                "skipping validation"
            )
            self.logger.warning(
                "Unsupported contract type '%s' at path '%s'", contract.type, path
            )
            return result

        # Handle null values
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

        # Validate by type
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
            # null was already handled above, reaching here means value is not null
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

        # Shouldn't reach here, but return safe result for type safety
        return ValidationResult(valid=True)

    def _validate_object(
        self, value: Any, contract: Contract, path: str
    ) -> ValidationResult:
        """Validate object type"""
        result = ValidationResult()

        # Type check
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

        # Check required fields
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

        # Recursively validate properties
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
        """Validate array type"""
        result = ValidationResult()

        # Type check
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

        # Validate each element
        if contract.items:
            for idx, item in enumerate(value):
                item_result = self.validate(item, contract.items, f"{path}[{idx}]")
                result.merge(item_result)

        return result

    def _validate_scalar(
        self, value: Any, expected_type: type, type_name: str, path: str
    ) -> ValidationResult:
        """Validate scalar type"""
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
        """Validate number type (int or float)"""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            # bool is a subclass of int, need to exclude it
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
        Validate node's in_contract

        Args:
            node: Node definition
            state: State manager

        Returns:
            ValidationResult: The validation result
        """
        if not node.in_contract:
            return ValidationResult(valid=True)

        self.logger.debug("Validating in_contract for node '%s'", node.id)

        # Get input values from reads paths for validation
        # If node doesn't define reads, validate entire state
        if node.reads:
            result = ValidationResult()
            for read_path in node.reads:
                value = state.get(read_path)
                path_result = self.validate(value, node.in_contract, read_path)
                result.merge(path_result)
            return result
        else:
            # No reads defined, validate entire state
            return self.validate(state.get_full_state(), node.in_contract, "$")

    def validate_out_contract(
        self, node: Node, changeset: ChangeSet, state: StateManager
    ) -> ValidationResult:
        """
        Validate node's out_contract

        Validate relevant path values after changeset is applied

        Args:
            node: Node definition
            changeset: Change set
            state: State manager

        Returns:
            ValidationResult: The validation result
        """
        if not node.out_contract:
            return ValidationResult(valid=True)

        self.logger.debug("Validating out_contract for node '%s'", node.id)

        # Create temporary state and apply changeset
        temp_state = copy.deepcopy(state.get_full_state())
        changeset.apply_to(temp_state)

        # Get output values from writes paths for validation
        # If node doesn't define writes, validate entire state
        if node.writes:
            result = ValidationResult()
            for write_path in node.writes:
                value = PathResolver.get(temp_state, write_path)
                path_result = self.validate(value, node.out_contract, write_path)
                result.merge(path_result)
            return result
        else:
            # No writes defined, validate entire state
            return self.validate(temp_state, node.out_contract, "$")
