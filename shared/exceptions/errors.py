"""
LinJ/ContiText Exception Definitions

Define various error types according to specifications, including ValidationError,
ExecutionError, MappingError, ConditionError, etc.
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

    Occurs during document validation phase, such as version mismatch, missing
    required fields, unbounded loops, etc.
    """

    pass


class ExecutionError(LinJError):
    """
    Execution error

    Occurs during node execution phase, such as tool invocation failure,
    state limit exceeded, etc.
    """

    pass


class MappingError(LinJError):
    """
    State mapping error

    Occurs during path writing, such as intermediate position is not an object,
    array out-of-bounds and cannot expand, etc.
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

    Occurs during change set commit, such as overlapping write paths,
    version mismatch, etc.
    """

    pass


class HandleExpired(LinJError):
    """
    Continuation handle expired error

    Occurs during continuation recovery, handle has expired or is invalid
    """

    pass


class ResourceConstraintUnsatisfied(ValidationError):
    """
    Resource constraint unsatisfied error

    When placement or resource dependencies cannot be satisfied
    """

    pass


class InvalidRequirements(ValidationError):
    """
    Invalid requirements format

    When requirements field value is not a boolean type
    """

    pass


class InvalidPlacement(ValidationError):
    """
    Invalid placement format

    When placement declaration is invalid or cannot be satisfied
    """

    pass


class ContractViolation(LinJError):
    """
    Contract violation error

    When output does not satisfy out_contract
    """

    pass
