"""
Abstract Evaluator Class

Defines the abstract interface for conditional expression evaluation, supporting multiple evaluation strategies and expression syntaxes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .types import ExecutionContext


class EvaluationStrategy(Enum):
    """Evaluation strategy enumeration"""

    STRICT = "strict"  # Strict evaluation
    LAZY = "lazy"  # Lazy evaluation
    SHORT_CIRCUIT = "short_circuit"  # Short-circuit evaluation
    PARALLEL = "parallel"  # Parallel evaluation


class TokenType(Enum):
    """Lexical unit type"""

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    NULL = "null"
    PATH = "path"
    OPERATOR = "operator"
    LOGIC = "logic"
    FUNCTION = "function"
    LPAREN = "lparen"
    RPAREN = "rparen"
    EOF = "eof"


@dataclass
class Token:
    """Lexical unit"""

    type: TokenType
    value: Any
    position: int = 0

    def __str__(self):
        return f"Token({self.type.value}, {self.value})"


@dataclass
class EvaluationResult:
    """Evaluation result"""

    success: bool
    value: Any = None
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None


class Evaluator(ABC):
    """
    Abstract Evaluator Base Class

    Defines the core interface for conditional expression evaluation:
    - Lexical analysis and syntax parsing
    - Expression evaluation execution
    - Error handling and recovery
    - Performance optimization and caching
    """

    @abstractmethod
    def evaluate(
        self,
        expression: str,
        context: ExecutionContext,
        strategy: Optional[EvaluationStrategy] = None,
    ) -> EvaluationResult:
        """
        Evaluate expression

        Args:
            expression: Conditional expression string
            context: Execution context
            strategy: Evaluation strategy

        Returns:
            Evaluation result
        """
        pass

    @abstractmethod
    def tokenize(self, expression: str) -> List[Token]:
        """
        Lexical analysis

        Args:
            expression: Expression string

        Returns:
            List of lexical units
        """
        pass

    @abstractmethod
    def parse(self, tokens: List[Token]) -> Any:
        """
        Syntax analysis

        Args:
            tokens: List of lexical units

        Returns:
            Syntax tree (nodes of abstract syntax tree)
        """
        pass

    @abstractmethod
    def validate_expression(self, expression: str) -> bool:
        """
        Validate expression syntax

        Args:
            expression: Expression string

        Returns:
            True if syntax is correct, False if there are errors
        """
        pass

    def get_value(self, path: str, context: ExecutionContext) -> Any:
        """
        Get path value from context

        Args:
            path: State path
            context: Execution context

        Returns:
            Value corresponding to the path
        """
        return context.get_state_value(path)

    def set_value(self, path: str, value: Any, context: ExecutionContext) -> None:
        """
        Set path value in context

        Args:
            path: State path
            value: Value to set
            context: Execution context
        """
        context.set_state_value(path, value)


class BaseEvaluator(Evaluator):
    """
    Base Evaluator Implementation

    Provides common expression evaluation logic and error handling
    """

    def __init__(self):
        self._cache: Dict[str, EvaluationResult] = {}
        self._stats = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "evaluation_errors": 0,
        }

    def evaluate(
        self,
        expression: str,
        context: ExecutionContext,
        strategy: Optional[EvaluationStrategy] = None,
    ) -> EvaluationResult:
        """Evaluate expression"""
        strategy = strategy or EvaluationStrategy.STRICT

        # Check cache
        cache_key = f"{expression}:{id(context)}:{strategy.value}"
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        self._stats["total_evaluations"] += 1

        try:
            # Lexical analysis
            tokens = self.tokenize(expression)

            # Syntax analysis
            ast = self.parse(tokens)

            # Evaluate based on strategy
            if strategy == EvaluationStrategy.SHORT_CIRCUIT:
                result = self._evaluate_short_circuit(ast, context)
            elif strategy == EvaluationStrategy.LAZY:
                result = self._evaluate_lazy(ast, context)
            elif strategy == EvaluationStrategy.PARALLEL:
                result = self._evaluate_parallel(ast, context)
            else:  # STRICT
                result = self._evaluate_strict(ast, context)

            # Cache result
            self._cache[cache_key] = result

            return result

        except Exception as e:
            self._stats["evaluation_errors"] += 1
            return EvaluationResult(success=False, error=e)

    def _evaluate_strict(self, ast: Any, context: ExecutionContext) -> EvaluationResult:
        """Strict evaluation"""
        value = self._evaluate_node(ast, context)
        return EvaluationResult(success=True, value=self._to_boolean(value))

    def _evaluate_short_circuit(
        self, ast: Any, context: ExecutionContext
    ) -> EvaluationResult:
        """Short-circuit evaluation"""
        # Implement AND/OR short-circuit logic
        return self._evaluate_node_with_short_circuit(ast, context)

    def _evaluate_lazy(self, ast: Any, context: ExecutionContext) -> EvaluationResult:
        """Lazy evaluation"""
        # Delay calculation until truly needed
        return self._evaluate_strict(ast, context)

    def _evaluate_parallel(
        self, ast: Any, context: ExecutionContext
    ) -> EvaluationResult:
        """Parallel evaluation"""
        # Compute independent sub-expressions in parallel
        return self._evaluate_strict(ast, context)

    @abstractmethod
    def _evaluate_node(self, node: Any, context: ExecutionContext) -> Any:
        """Evaluate single node"""
        pass

    def _evaluate_node_with_short_circuit(
        self, node: Any, context: ExecutionContext
    ) -> Any:
        """Node evaluation with short-circuit logic"""
        return self._evaluate_node(node, context)

    def _to_boolean(self, value: Any) -> bool:
        """Convert value to boolean type"""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return bool(value)

    def validate_expression(self, expression: str) -> bool:
        """Validate expression syntax"""
        try:
            tokens = self.tokenize(expression)
            self.parse(tokens)
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            "total_evaluations": self._stats["total_evaluations"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["total_evaluations"]
            - self._stats["cache_hits"],
            "cache_hit_rate": self._stats["cache_hits"]
            / max(1, self._stats["total_evaluations"]),
            "evaluation_errors": self._stats["evaluation_errors"],
            "cache_size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear cache"""
        self._cache.clear()

    def reset_stats(self) -> None:
        """Reset statistics"""
        self._stats = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "evaluation_errors": 0,
        }


class SimpleEvaluator(BaseEvaluator):
    """
    Simple Evaluator

    Supports basic conditional expression evaluation
    """

    def __init__(self):
        super().__init__()
        self.operators = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "AND": lambda a, b: a and b,
            "OR": lambda a, b: a or b,
            "NOT": lambda a: not a,
        }
        self.functions = {
            "exists": self._func_exists,
            "len": self._func_len,
            "value": self._func_value,
        }

    def tokenize(self, expression: str) -> List[Token]:
        """Simple lexical analysis"""
        import re

        # Define lexical patterns
        token_patterns = [
            (r"\s+", None),  # Whitespace, skip
            (r"\d+(?:\.\d+)?", TokenType.NUMBER),
            (r'"[^"]*"', TokenType.STRING),
            (r"true|false", TokenType.BOOLEAN),
            (r"null", TokenType.NULL),
            (r"\$[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*", TokenType.PATH),
            (r"==|!=|>=|<=|>|<", TokenType.OPERATOR),
            (r"AND|OR|NOT", TokenType.LOGIC),
            (r"[a-zA-Z_][a-zA-Z0-9_]*\s*\(", TokenType.FUNCTION),
            (r"\(", TokenType.LPAREN),
            (r"\)", TokenType.RPAREN),
        ]

        tokens = []
        i = 0
        pos = 0

        while i < len(expression):
            matched = False

            for pattern, token_type in token_patterns:
                regex = re.compile(pattern)
                match = regex.match(expression, i)

                if match:
                    if token_type:
                        value = match.group(0)

                        # Handle different types of values
                        if token_type == TokenType.NUMBER:
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        elif token_type == TokenType.STRING:
                            value = value[1:-1]  # Remove quotes
                        elif token_type == TokenType.BOOLEAN:
                            value = value == "true"
                        elif token_type == TokenType.NULL:
                            value = None
                        elif token_type == TokenType.FUNCTION:
                            # Extract function name, remove parentheses
                            value = value.replace("(", "")

                        tokens.append(Token(token_type, value, pos))

                    i = match.end()
                    pos = i
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"Invalid character at position {i}: '{expression[i]}'"
                )

        return tokens

    def parse(self, tokens: List[Token]) -> Any:
        """Simple syntax analysis (converted to infix expression list)"""
        # This is a simplified implementation, actual should build a real AST
        return tokens

    def _evaluate_node(self, node: Any, context: ExecutionContext) -> Any:
        """Evaluate node"""
        if isinstance(node, list):
            # Infix expression evaluation
            return self._evaluate_infix(node, context)
        elif isinstance(node, Token):
            return self._evaluate_token(node, context)
        else:
            return node

    def _evaluate_token(self, token: Token, context: ExecutionContext) -> Any:
        """Evaluate lexical unit"""
        if token.type == TokenType.NUMBER:
            return token.value
        elif token.type == TokenType.STRING:
            return token.value
        elif token.type == TokenType.BOOLEAN:
            return token.value
        elif token.type == TokenType.NULL:
            return None
        elif token.type == TokenType.PATH:
            return self.get_value(token.value, context)
        elif token.type == TokenType.FUNCTION:
            return self._evaluate_function(token.value, context)
        else:
            return token.value

    def _evaluate_infix(self, tokens: List[Token], context: ExecutionContext) -> Any:
        """Evaluate infix expression"""
        # This is a very simplified implementation
        # Actual should implement complete expression evaluation logic

        if len(tokens) == 1:
            return self._evaluate_token(tokens[0], context)

        if len(tokens) == 3:
            left = self._evaluate_token(tokens[0], context)
            op = tokens[1].value
            right = self._evaluate_token(tokens[2], context)

            if op in self.operators:
                return self.operators[op](left, right)

        # Default return False
        return False

    def _evaluate_function(self, func_name: str, context: ExecutionContext) -> Any:
        """Evaluate function"""
        if func_name in self.functions:
            return self.functions[func_name](context)
        raise ValueError(f"Unknown function: {func_name}")

    def _func_exists(self, context: ExecutionContext) -> bool:
        """exists function (simplified)"""
        # Simplified implementation: always return True
        return True

    def _func_len(self, context: ExecutionContext) -> int:
        """len function (simplified)"""
        # Simplified implementation: return 0
        return 0

    def _func_value(self, context: ExecutionContext) -> Any:
        """value function (simplified)"""
        # Simplified implementation: return None
        return None


class RegexEvaluator(SimpleEvaluator):
    """
    Advanced Evaluator Based on Regular Expressions

    Supports more complex expression syntax and evaluation logic
    """

    def __init__(self):
        super().__init__()
        # Build more complex lexical patterns
        self.build_advanced_patterns()

    def build_advanced_patterns(self):
        """Build advanced lexical patterns"""
        # Add support for array access, object properties, etc.
        pass

    def tokenize(self, expression: str) -> List[Token]:
        """Advanced lexical analysis"""
        # Use more complex regular expressions
        import re

        # Extended lexical patterns
        token_patterns = [
            (r"\s+", None),
            (r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", TokenType.NUMBER),
            (r'"[^"\\]*(?:\\.[^"\\]*)*"', TokenType.STRING),
            (r"'[^'\\]*(?:\\.[^'\\]*)*'", TokenType.STRING),
            (r"true|false", TokenType.BOOLEAN),
            (r"null", TokenType.NULL),
            (
                r"\$[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*|\[\d+\])*",
                TokenType.PATH,
            ),
            (r"==|!=|>=|<=|>|<", TokenType.OPERATOR),
            (r"AND|OR|NOT", TokenType.LOGIC),
            (r"[a-zA-Z_][a-zA-Z0-9_]*\s*\(", TokenType.FUNCTION),
            (r"\(", TokenType.LPAREN),
            (r"\)", TokenType.RPAREN),
            (r",", None),  # Comma, skip
        ]

        tokens = []
        i = 0
        pos = 0

        while i < len(expression):
            matched = False

            for pattern, token_type in token_patterns:
                regex = re.compile(pattern)
                match = regex.match(expression, i)

                if match:
                    if token_type:
                        value = match.group(0)

                        # Handle escape characters
                        if token_type == TokenType.STRING:
                            value = eval(
                                value
                            )  # Simplified handling, should safely handle escapes in practice

                        # Handle function calls
                        elif token_type == TokenType.FUNCTION:
                            value = value.replace("(", "")

                        tokens.append(Token(token_type, value, pos))

                    i = match.end()
                    pos = i
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"Invalid character at position {i}: '{expression[i]}'"
                )

        return tokens
