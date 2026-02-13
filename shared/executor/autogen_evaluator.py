"""
Condition expression evaluator implementation

Evaluator implementation migrated and refactored from autogen/executor/evaluator.py,
compatible with existing AutoGen evaluation logic.
"""

import re
from typing import Any, Dict, Optional, List

# Try to import existing module for compatibility
try:
    from ..core.path import PathResolver
    from ..exceptions.errors import ConditionError
except ImportError:
    # Fallback to basic implementation
    PathResolver = Any
    ConditionError = Exception

from .evaluator import (
    BaseEvaluator,
    EvaluationResult,
    TokenType,
    Token,
    EvaluationStrategy,
)


class AutoGenConditionEvaluator(BaseEvaluator):
    """
    AutoGen-compatible condition expression evaluator

    Implements condition expression syntax defined in LinJ Specification Section 14:
    - Comparison: == != > >= < <=
    - Logic: AND OR NOT (short-circuit)
    - Functions: exists(path), len(path), value(path)
    """

    # Lexical analysis pattern
    TOKEN_PATTERN = re.compile(
        r"\s*("  # Optional whitespace
        r"(?P<LPAREN>\()|"  # Left parenthesis
        r"(?P<RPAREN>\))|"  # Right parenthesis
        r"(?P<OP>==|!=|>=|<=|>|<)|"  # Comparison operator
        r"(?P<LOGIC>AND|OR|NOT)|"  # Logic operator
        r"(?P<FUN>exists|len|value)\s*\(|"  # Function call
        r"(?P<PATH>\$\.[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*|\[\d+\])*)|"  # Path
        r'(?P<STRING>"[^"]*")|'  # String
        r"(?P<NUMBER>-?\d+(?:\.\d+)?)|"  # Number
        r"(?P<NULL>null)|"  # null
        r"(?P<BOOL>true|false)"  # Boolean
        r")"
    )

    def __init__(self, state: Dict[str, Any]):
        """
        Initialize the evaluator

        Args:
            state: Main state object, used for path evaluation
        """
        super().__init__()
        self._state = state
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(
        self,
        expression: str,
        context=None,  # Compatible with original interface
        strategy: Optional[EvaluationStrategy] = None,
    ) -> EvaluationResult:
        """
        Evaluate condition expression

        Args:
            expression: Condition expression string
            context: Execution context (compatibility parameter)
            strategy: Evaluation strategy

        Returns:
            Evaluation result

        Raises:
            ConditionError: Expression syntax error or type mismatch
        """
        if not expression or expression.strip() == "":
            return EvaluationResult(
                success=True, value=True
            )  # Empty condition is considered true

        self._tokens = self._tokenize(expression)
        self._pos = 0

        try:
            result = self._parse_or()

            if self._pos < len(self._tokens):
                raise ConditionError(f"Unexpected token: {self._tokens[self._pos]}")

            return EvaluationResult(success=True, value=bool(result))

        except Exception as e:
            return EvaluationResult(success=False, error=e)

    def tokenize(self, expression: str) -> List[Token]:
        """Lexical analysis"""
        return self._tokenize(expression)

    def _tokenize(self, condition: str) -> List[Token]:
        """Lexical analysis"""
        tokens = []
        pos = 0

        while pos < len(condition):
            match = self.TOKEN_PATTERN.match(condition, pos)
            if not match:
                raise ConditionError(
                    f"Invalid character at position {pos}: {condition[pos:]}"
                )

            token = None

            if match.group("LPAREN"):
                token = Token(TokenType.LPAREN, "(")
            elif match.group("RPAREN"):
                token = Token(TokenType.RPAREN, ")")
            elif match.group("OP"):
                token = Token(TokenType.OPERATOR, match.group("OP"))
            elif match.group("LOGIC"):
                token = Token(TokenType.LOGIC, match.group("LOGIC"))
            elif match.group("FUN"):
                # Extract function name and path argument
                fun_name = match.group("FUN")
                # Find matching right parenthesis
                start = match.end()
                depth = 1
                end = start
                while end < len(condition) and depth > 0:
                    if condition[end] == "(":
                        depth += 1
                    elif condition[end] == ")":
                        depth -= 1
                    end += 1

                if depth != 0:
                    raise ConditionError(f"Unclosed function call: {fun_name}")

                path = condition[start : end - 1].strip()
                if (path.startswith('"') and path.endswith('"')) or (
                    path.startswith("'") and path.endswith("'")
                ):
                    path = path[1:-1]

                token = Token(TokenType.FUNCTION, (fun_name, path))
                pos = end
                tokens.append(token)
                continue
            elif match.group("PATH"):
                token = Token(TokenType.PATH, match.group("PATH"))
            elif match.group("STRING"):
                token = Token(
                    TokenType.STRING, match.group("STRING")[1:-1]
                )  # Remove quotes
            elif match.group("NUMBER"):
                num_str = match.group("NUMBER")
                if "." in num_str:
                    token = Token(TokenType.NUMBER, float(num_str))
                else:
                    token = Token(TokenType.NUMBER, int(num_str))
            elif match.group("NULL"):
                token = Token(TokenType.NULL, None)
            elif match.group("BOOL"):
                token = Token(TokenType.BOOLEAN, match.group("BOOL") == "true")

            if token:
                tokens.append(token)
                pos = match.end()

        return tokens

    def parse(self, tokens: List[Token]) -> Any:
        """Syntax analysis (simplified implementation)"""
        self._tokens = tokens
        self._pos = 0
        return self._parse_or()

    def _current(self) -> Optional[Token]:
        """Get current token"""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected_type: Optional[TokenType] = None) -> Token:
        """Consume current token"""
        token = self._current()
        if token is None:
            raise ConditionError("Unexpected end of expression")

        if expected_type and token.type != expected_type:
            raise ConditionError(f"Expected {expected_type}, got {token.type}")

        self._pos += 1
        return token

    def _parse_or(self) -> bool:
        """Parse OR expression (lowest precedence)"""
        left = self._parse_and()

        while (
            self._current()
            and self._current().type == TokenType.LOGIC
            and self._current().value == "OR"
        ):
            self._consume(TokenType.LOGIC)
            if left:  # Short-circuit: if left is true, no need to evaluate right
                # Consume right operand but don't evaluate
                self._parse_and()
                return True
            left = self._parse_and()

        return left

    def _parse_and(self) -> bool:
        """Parse AND expression"""
        left = self._parse_not()

        while (
            self._current()
            and self._current().type == TokenType.LOGIC
            and self._current().value == "AND"
        ):
            self._consume(TokenType.LOGIC)
            if not left:  # Short-circuit: if left is false, no need to evaluate right
                # Consume right operand but don't evaluate
                self._parse_not()
                return False
            left = self._parse_not()

        return left

    def _parse_not(self) -> bool:
        """Parse NOT expression"""
        if (
            self._current()
            and self._current().type == TokenType.LOGIC
            and self._current().value == "NOT"
        ):
            self._consume(TokenType.LOGIC)
            return not self._parse_not()

        return self._parse_comparison()

    def _parse_comparison(self) -> bool:
        """Parse comparison expression"""
        left = self._parse_primary()

        if self._current() and self._current().type == TokenType.OPERATOR:
            op = self._consume(TokenType.OPERATOR).value
            right = self._parse_primary()

            # null handling: only == and != make sense
            if left is None or right is None:
                if op == "==":
                    return left == right
                elif op == "!=":
                    return left != right
                else:
                    # Section 14.2: null participating in other comparisons results in false
                    return False

            # Type checking
            if type(left) != type(right):
                raise ConditionError(
                    f"Type mismatch in comparison: {type(left).__name__} {op} {type(right).__name__}"
                )

            # Perform comparison
            if op == "==":
                return left == right
            elif op == "!=":
                return left != right
            elif op == ">":
                return left > right
            elif op == ">=":
                return left >= right
            elif op == "<":
                return left < right
            elif op == "<=":
                return left <= right

        # Single value is treated as boolean
        return bool(left)

    def _parse_primary(self) -> Any:
        """Parse primary value"""
        token = self._current()

        if token is None:
            raise ConditionError("Unexpected end of expression")

        if token.type == TokenType.LPAREN:
            self._consume(TokenType.LPAREN)
            result = self._parse_or()
            self._consume(TokenType.RPAREN)
            return result

        if token.type == TokenType.FUNCTION:
            return self._eval_function(token.value)

        if token.type == TokenType.PATH:
            self._consume()
            return (
                PathResolver.get(self._state, token.value)
                if hasattr(PathResolver, "get")
                else None
            )

        if token.type in (
            TokenType.STRING,
            TokenType.NUMBER,
            TokenType.NULL,
            TokenType.BOOLEAN,
        ):
            self._consume()
            return token.value

        raise ConditionError(f"Unexpected token: {token}")

    def _eval_function(self, fun_spec: tuple) -> Any:
        """Evaluate function call"""
        fun_name, path = fun_spec
        self._pos += 1  # Consume FUN token

        if fun_name == "exists":
            # exists(path): path exists and value is not null returns true
            if hasattr(PathResolver, "get"):
                value = PathResolver.get(self._state, path)
                return value is not None
            return False

        elif fun_name == "len":
            # len(path): array length; non-array or non-existent returns 0
            if hasattr(PathResolver, "get"):
                value = PathResolver.get(self._state, path)
                if isinstance(value, list):
                    return len(value)
            return 0

        elif fun_name == "value":
            # value(path): get value, returns null if not exists
            if hasattr(PathResolver, "get"):
                return PathResolver.get(self._state, path)
            return None

        raise ConditionError(f"Unknown function: {fun_name}")

    def validate_expression(self, expression: str) -> bool:
        """Validate expression syntax"""
        try:
            tokens = self._tokenize(expression)
            self._tokens = tokens
            self._pos = 0
            self._parse_or()
            return self._pos == len(tokens)
        except Exception:
            return False


class SimplePathResolver:
    """
    Simple path resolver

    Fallback implementation when PathResolver is unavailable
    """

    @staticmethod
    def get(state: Dict[str, Any], path: str) -> Any:
        """Get path value"""
        if not path.startswith("$"):
            return None

        # Simplified implementation: remove $ and split by .
        path_parts = path[1:].split(".")
        current = state

        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    @staticmethod
    def intersect(path_a: str, path_b: str) -> bool:
        """Check if paths intersect"""
        parts_a = path_a[1:].split(".") if path_a.startswith("$") else path_a.split(".")
        parts_b = path_b[1:].split(".") if path_b.startswith("$") else path_b.split(".")

        min_len = min(len(parts_a), len(parts_b))
        for i in range(min_len):
            if parts_a[i] != parts_b[i]:
                return False
        return min_len > 0


# Use simple implementation when PathResolver is unavailable
try:
    from ..core.path import PathResolver
except ImportError:
    PathResolver = SimplePathResolver


def evaluate_condition(condition: str, state: Dict[str, Any]) -> bool:
    """
    Convenient evaluation function

    Args:
        condition: Condition expression
        state: Main state

    Returns:
        Boolean result
    """
    evaluator = AutoGenConditionEvaluator(state)
    result = evaluator.evaluate(condition)
    return result.value if result.success else False
