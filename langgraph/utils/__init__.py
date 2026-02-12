"""
LangGraph Utils Module

This module contains utility functions and helper classes used throughout
the LangGraph implementation. Includes state management helpers, validation
utilities, and common patterns.
"""

from .state_helpers import StateHelper
from .logging import get_logger

# Import validation functions with try/except to avoid circular dependencies
try:
    from .validation import validate_state_transition, validate_node_input
except ImportError:
    # Provide placeholder functions to avoid import errors
    def validate_state_transition(*args, **kwargs):
        return True

    def validate_node_input(*args, **kwargs):
        return []


__all__ = [
    "StateHelper",
    "validate_state_transition",
    "validate_node_input",
    "get_logger",
]
