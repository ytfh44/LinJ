"""
LinJ/ContiText Exception Definitions

Defines various error types according to specifications, including ValidationError, ExecutionError, MappingError, ConditionError, etc.
"""

from typing import Any, Dict, Optional


class LinJError(Exception):
    """LinJ base exception"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(LinJError):
    """
    Validation error

    Occurs during document validation, such as version mismatch, missing required fields, unbounded loops, etc.
    """

    pass


class ExecutionError(LinJError):
    """
    Execution error

    Occurs during node execution, such as tool invocation failure, state limit exceeded, etc.
    """

    pass


class MappingError(LinJError):
    """
    State mapping error

    Occurs during path writing, such as intermediate position not being an object, array out of bounds unable to expand, etc.
    """

    pass


class ConditionError(LinJError):
    """
    Condition expression error

    Occurs during condition evaluation, such as type mismatch, syntax error, etc.
    """

    pass


class ConflictError(LinJError):
    """
    Change set conflict error

    Occurs during change set submission, such as intersecting write paths, version mismatch, etc.
    """

    pass


class HandleExpired(LinJError):
    """
    Continuation handle expired error

    Occurs when resuming continuation, handle has expired or is invalid
    """

    pass


class ResourceConstraintUnsatisfied(LinJError):
    """
    Resource constraint unsatisfied error

    When placement or resource dependencies cannot be satisfied
    """

    pass


class InvalidRequirements(LinJError):
    """
    Invalid requirements format error

    When requirements field value is not of boolean type
    """

    pass


class InvalidPlacement(LinJError):
    """
    Invalid placement format error

    When placement declaration is invalid or cannot be satisfied
    """

    pass


class ContractViolation(LinJError):
    """
    Contract violation error

    When output does not satisfy out_contract
    """

    pass
