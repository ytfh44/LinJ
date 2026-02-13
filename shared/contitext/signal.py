"""
Signal and Wait Mechanism

Implements signal structures and wait conditions defined in LinJ Specification Section 21
Framework-agnostic signal handling implementation
"""

from typing import Any, Dict, Optional, Callable, Protocol, runtime_checkable
from pydantic import BaseModel


class Signal(BaseModel):
    """
    Signal (Section 21.1)

    - name: Signal name
    - payload: Payload data
    - correlation: Optional correlation identifier
    Framework-agnostic signal implementation
    """

    name: str
    payload: Optional[Any] = None
    correlation: Optional[str] = None


class WaitCondition(BaseModel):
    """
    Wait Condition (Section 21.2)

    Supports:
    - Exact match by name
    - Exact match by correlation
    - Predicate expressions
    Framework-agnostic wait condition implementation
    """

    name: Optional[str] = None
    correlation: Optional[str] = None
    predicate: Optional[str] = None  # Condition expression
    evaluator: Optional[Callable[[str, Dict[str, Any]], bool]] = None

    def matches(self, signal: Signal, state: Optional[Dict] = None) -> bool:
        """
        Check if signal matches wait condition (Section 21.2)

        Supports:
        - Exact match by name
        - Exact match by correlation
        - Predicate expressions: using condition expression semantics

        Args:
            signal: Signal to check
            state: Current state (for predicate evaluation)

        Returns:
            Whether the signal matches
        """
        # Check name (exact match)
        if self.name and signal.name != self.name:
            return False

        # Check correlation (exact match)
        if self.correlation and signal.correlation != self.correlation:
            return False

        # Check predicate expression
        if self.predicate and state is not None:
            # Inject signal payload into state, access via $.signal.payload per spec
            eval_state = dict(state)
            eval_state["signal"] = {
                "name": signal.name,
                "payload": signal.payload,
                "correlation": signal.correlation,
            }

            try:
                if self.evaluator:
                    # Use custom evaluator
                    return self.evaluator(self.predicate, eval_state)
                else:
                    # Default simple evaluator (basic expressions only)
                    return self._evaluate_condition_simple(self.predicate, eval_state)
            except Exception:
                # Predicate evaluation failure means no match
                return False

        return True

    def _evaluate_condition_simple(self, predicate: str, state: Dict[str, Any]) -> bool:
        """
        Simple condition evaluator

        Supports basic expression evaluation, does not support complex LINJ syntax
        In practice, should be replaced with an appropriate evaluator for specific frameworks
        """
        # Implementing a simple expression evaluator here
        # Only supports basic comparison and logical operations

        try:
            # Replace common path expressions
            expr = predicate.replace("$.signal.payload", "signal['payload']")
            expr = expr.replace("$.signal.name", "signal['name']")
            expr = expr.replace("$.signal.correlation", "signal['correlation']")

            # Limit available built-in functions and variables
            safe_dict = {
                "signal": state.get("signal", {}),
                "state": state,
                "len": len,
                "str": str,
                "int": int,
                "bool": bool,
            }

            # Use eval for simple evaluation (note: a safer implementation is needed in production)
            return bool(eval(expr, {"__builtins__": {}}, safe_dict))
        except Exception:
            return False


class SignalQueue:
    """
    Signal Queue

    Manages signal sending and waiting
    Framework-agnostic signal queue implementation
    """

    def __init__(self):
        self._signals: list[Signal] = []
        self._waiters: dict[str, WaitCondition] = {}  # handle -> condition

    def send(self, signal: Signal) -> None:
        """Send signal"""
        self._signals.append(signal)

    def check_waiter(self, handle: str, signal: Signal, state: Dict) -> bool:
        """Check if waiter has received the signal"""
        if handle not in self._waiters:
            return False

        condition = self._waiters[handle]
        return condition.matches(signal, state)

    def register_waiter(self, handle: str, condition: WaitCondition) -> None:
        """Register a waiter"""
        self._waiters[handle] = condition

    def unregister_waiter(self, handle: str) -> None:
        """Unregister a waiter"""
        self._waiters.pop(handle, None)

    def find_matching_signal(
        self, condition: WaitCondition, state: Dict
    ) -> Optional[Signal]:
        """Find signal matching condition"""
        for signal in self._signals:
            if condition.matches(signal, state):
                return signal
        return None

    def get_signals_for_waiter(self, handle: str, state: Dict) -> list[Signal]:
        """Get all signals matching for waiter"""
        if handle not in self._waiters:
            return []

        condition = self._waiters[handle]
        matching_signals = []

        for signal in self._signals:
            if condition.matches(signal, state):
                matching_signals.append(signal)

        return matching_signals

    def clear_signals(self) -> None:
        """Clear all signals"""
        self._signals.clear()

    def get_pending_waiters(self) -> list[str]:
        """Get list of pending waiter handles"""
        return list(self._waiters.keys())

    def has_waiter(self, handle: str) -> bool:
        """Check if there is a waiter for the specified handle"""
        return handle in self._waiters
