"""
Condition Expression Evaluator

Implements condition expression syntax and semantics as defined in LinJ Specification Section 14:
- Comparison operators: ==, !=, >, >=, <, <=
- Logical operators: AND, OR, NOT (must short-circuit)
- Functions: exists(path), len(path), value(path)

Secure implementation: no eval(), using recursive descent parser
"""

import re
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .path import PathResolver


class TokenType(Enum):
    """Lexical token types"""

    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING"
    NULL = "NULL"
    BOOLEAN = "BOOLEAN"

    # Identifiers
    IDENTIFIER = "IDENTIFIER"

    # Comparison operators
    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="

    # Logical operators
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # Parentheses
    LPAREN = "("
    RPAREN = ")"

    # Function calls
    COMMA = ","

    # End
    EOF = "EOF"


@dataclass
class Token:
    """Lexical token"""

    type: TokenType
    value: Any
    position: int


class Lexer:
    """Lexer"""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None

    def advance(self) -> None:
        """Advance to next character"""
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self) -> None:
        """Skip whitespace characters"""
        while self.current_char and self.current_char.isspace():
            self.advance()

    def read_number(self) -> Token:
        """Read number"""
        start = self.pos
        while self.current_char and (
            self.current_char.isdigit() or self.current_char == "."
        ):
            self.advance()

        value = self.text[start : self.pos]
        if "." in value:
            return Token(TokenType.NUMBER, float(value), start)
        return Token(TokenType.NUMBER, int(value), start)

    def read_string(self) -> Token:
        """Read string"""
        start = self.pos
        self.advance()  # Skip opening quote

        while self.current_char and self.current_char != '"':
            if self.current_char == "\\" and self.text[self.pos + 1]:
                self.advance()  # Skip escape character
            self.advance()

        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, self.text[start + 1 : self.pos - 1], start)

    def read_identifier(self) -> Token:
        """Read identifier or keyword"""
        start = self.pos
        while self.current_char and (
            self.current_char.isalnum() or self.current_char == "_"
        ):
            self.advance()

        value = self.text[start : self.pos]

        # Check if keyword
        keyword_map = {
            "AND": TokenType.AND,
            "OR": TokenType.OR,
            "NOT": TokenType.NOT,
            "true": TokenType.BOOLEAN,
            "false": TokenType.BOOLEAN,
            "null": TokenType.NULL,
        }

        if value.lower() in [k.lower() for k in keyword_map.keys()]:
            for key, token_type in keyword_map.items():
                if key.lower() == value.lower():
                    return Token(
                        token_type,
                        key == "true" if token_type == TokenType.BOOLEAN else None,
                        start,
                    )

        return Token(TokenType.IDENTIFIER, value, start)

    def peek(self) -> Optional[str]:
        """Peek at next character"""
        peek_pos = self.pos + 1
        return self.text[peek_pos] if peek_pos < len(self.text) else None

    def next_token(self) -> Token:
        """Get next token"""
        self.skip_whitespace()

        if self.current_char is None:
            return Token(TokenType.EOF, None, self.pos)

        start = self.pos

        # Check two-character operators
        if self.current_char == "=" and self.peek() == "=":
            self.advance()
            self.advance()
            return Token(TokenType.EQ, "==", start)

        if self.current_char == "!" and self.peek() == "=":
            self.advance()
            self.advance()
            return Token(TokenType.NE, "!=", start)

        if self.current_char == ">" and self.peek() == "=":
            self.advance()
            self.advance()
            return Token(TokenType.GTE, ">=", start)

        if self.current_char == "<" and self.peek() == "=":
            self.advance()
            self.advance()
            return Token(TokenType.LTE, "<=", start)

        # Single-character operators
        if self.current_char == ">":
            self.advance()
            return Token(TokenType.GT, ">", start)

        if self.current_char == "<":
            self.advance()
            return Token(TokenType.LT, "<", start)

        # Parentheses and comma
        if self.current_char == "(":
            self.advance()
            return Token(TokenType.LPAREN, "(", start)

        if self.current_char == ")":
            self.advance()
            return Token(TokenType.RPAREN, ")", start)

        if self.current_char == ",":
            self.advance()
            return Token(TokenType.COMMA, ",", start)

        # Path reference (starts with $)
        if self.current_char == "$":
            self.advance()
            path_value = "$"
            # Collect path segments
            while self.current_char and (
                self.current_char.isalnum()
                or self.current_char in "._[]"
                or self.current_char == "$"
            ):
                path_value += self.current_char
                self.advance()
            return Token(TokenType.STRING, path_value, start)

        # Number or string
        if self.current_char.isdigit():
            return self.read_number()

        if self.current_char == '"':
            return self.read_string()

        # Identifier or keyword
        if self.current_char.isalpha() or self.current_char == "_":
            return self.read_identifier()

        raise SyntaxError(
            f"Unexpected character: {self.current_char} at position {self.pos}"
        )


class Parser:
    """Recursive descent parser"""

    def __init__(self, text: str):
        self.lexer = Lexer(text)
        self.current_token = self.lexer.next_token()

    def advance(self) -> None:
        """Advance to next token"""
        self.current_token = self.lexer.next_token()

    def expect(self, token_type: TokenType) -> Token:
        """Expect specific token type"""
        if self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        raise SyntaxError(f"Expected {token_type}, got {self.current_token.type}")

    def parse(self) -> "ASTNode":
        """Parse expression"""
        node = self.parse_or()
        if self.current_token.type != TokenType.EOF:
            raise SyntaxError(f"Unexpected token: {self.current_token.type}")
        return node

    # Precedence: NOT > comparison > AND > OR
    def parse_or(self) -> "ASTNode":
        """Parse OR expression"""
        node = self.parse_and()

        while self.current_token.type == TokenType.OR:
            self.advance()
            right = self.parse_and()
            node = OrNode(node, right)

        return node

    def parse_and(self) -> "ASTNode":
        """Parse AND expression"""
        node = self.parse_not()

        while self.current_token.type == TokenType.AND:
            self.advance()
            right = self.parse_not()
            node = AndNode(node, right)

        return node

    def parse_not(self) -> "ASTNode":
        """Parse NOT expression"""
        if self.current_token.type == TokenType.NOT:
            self.advance()
            operand = self.parse_not()
            return NotNode(operand)

        return self.parse_comparison()

    def parse_comparison(self) -> "ASTNode":
        """Parse comparison expression"""
        left = self.parse_primary()

        # Comparison operators
        comparison_ops = {
            TokenType.EQ: lambda l, r: EqNode(l, r),
            TokenType.NE: lambda l, r: NeNode(l, r),
            TokenType.GT: lambda l, r: GtNode(l, r),
            TokenType.GTE: lambda l, r: GteNode(l, r),
            TokenType.LT: lambda l, r: LtNode(l, r),
            TokenType.LTE: lambda l, r: LteNode(l, r),
        }

        if self.current_token.type in comparison_ops:
            op = self.current_token.type
            self.advance()
            right = self.parse_primary()
            return comparison_ops[op](left, right)

        return left

    def parse_primary(self) -> "ASTNode":
        """Parse primary expression"""
        token = self.current_token

        if token.type == TokenType.LPAREN:
            self.advance()
            node = self.parse_or()
            self.expect(TokenType.RPAREN)
            return node

        if token.type == TokenType.NUMBER:
            self.advance()
            return NumberNode(token.value)

        if token.type == TokenType.STRING:
            self.advance()
            return StringNode(token.value)

        if token.type == TokenType.BOOLEAN:
            self.advance()
            return BooleanNode(token.value)

        if token.type == TokenType.NULL:
            self.advance()
            return NullNode()

        if token.type == TokenType.IDENTIFIER:
            return self.parse_identifier()

        raise SyntaxError(f"Unexpected token: {token.type}")

    def parse_identifier(self) -> "ASTNode":
        """Parse identifier (may be function call or variable)"""
        name = self.current_token.value
        self.advance()

        # Check if function call
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            args = []

            if self.current_token.type != TokenType.RPAREN:
                args.append(self.parse_or())

                while self.current_token.type == TokenType.COMMA:
                    self.advance()
                    args.append(self.parse_or())

            self.expect(TokenType.RPAREN)

            # Built-in functions
            if name == "exists":
                if len(args) != 1:
                    raise SyntaxError("exists() requires exactly 1 argument")
                return ExistsNode(args[0])

            if name == "len":
                if len(args) != 1:
                    raise SyntaxError("len() requires exactly 1 argument")
                return LenNode(args[0])

            if name == "value":
                if len(args) != 1:
                    raise SyntaxError("value() requires exactly 1 argument")
                return ValueNode(args[0])

            # Unknown function call
            return FunctionNode(name, args)

        # Identifier reference
        return IdentifierNode(name)


# AST node classes
class ASTNode:
    """AST node base class"""

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Evaluate"""
        raise NotImplementedError


class NumberNode(ASTNode):
    """Number node"""

    def __init__(self, value: float):
        self.value = value

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self.value


class StringNode(ASTNode):
    """String node"""

    def __init__(self, value: str):
        self.value = value

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self.value


class BooleanNode(ASTNode):
    """Boolean node"""

    def __init__(self, value: bool):
        self.value = value

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self.value


class NullNode(ASTNode):
    """Null node"""

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return None


class IdentifierNode(ASTNode):
    """Identifier node - references value in state"""

    def __init__(self, name: str):
        self.name = name

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        # First check if value is in context (e.g., signal.payload)
        if context and self.name in context:
            return context[self.name]

        # Check if special variable
        if self.name == "true":
            return True
        if self.name == "false":
            return False
        if self.name == "null":
            return None

        # If path reference (starts with $), use path resolver
        if self.name.startswith("$"):
            return PathResolver.get(state, self.name)

        # Otherwise look up as direct key name
        return state.get(self.name)


class ExistsNode(ASTNode):
    """exists() function node"""

    def __init__(self, path_node: ASTNode):
        self.path_node = path_node

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        path = self.path_node.evaluate(state, context)
        if not isinstance(path, str):
            return False
        return PathResolver.get(state, path) is not None


class LenNode(ASTNode):
    """len() function node"""

    def __init__(self, path_node: ASTNode):
        self.path_node = path_node

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        path = self.path_node.evaluate(state, context)
        if not isinstance(path, str):
            return 0
        value = PathResolver.get(state, path)
        if isinstance(value, list):
            return len(value)
        return 0


class ValueNode(ASTNode):
    """value() function node"""

    def __init__(self, path_node: ASTNode):
        self.path_node = path_node

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        path = self.path_node.evaluate(state, context)
        if not isinstance(path, str):
            return None
        return PathResolver.get(state, path)


class FunctionNode(ASTNode):
    """User-defined function node"""

    def __init__(self, name: str, args: list):
        self.name = name
        self.args = args

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        raise NameError(f"Unknown function: {self.name}")


# Comparison operator nodes
class EqNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        left = self.left.evaluate(state, context)
        right = self.right.evaluate(state, context)
        if left is None or right is None:
            return left == right
        return left == right


class NeNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        left = self.left.evaluate(state, context)
        right = self.right.evaluate(state, context)
        if left is None or right is None:
            return left != right
        return left != right


class GtNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        left = self.left.evaluate(state, context)
        right = self.right.evaluate(state, context)
        if left is None or right is None:
            return False
        try:
            return left > right
        except TypeError:
            return False


class GteNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        left = self.left.evaluate(state, context)
        right = self.right.evaluate(state, context)
        if left is None or right is None:
            return False
        try:
            return left >= right
        except TypeError:
            return False


class LtNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        left = self.left.evaluate(state, context)
        right = self.right.evaluate(state, context)
        if left is None or right is None:
            return False
        try:
            return left < right
        except TypeError:
            return False


class LteNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        left = self.left.evaluate(state, context)
        right = self.right.evaluate(state, context)
        if left is None or right is None:
            return False
        try:
            return left <= right
        except TypeError:
            return False


# Logical operator nodes (with short-circuit support)
class AndNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        # Short-circuit: don't evaluate right if left is false
        left = self.left.evaluate(state, context)
        if not left:
            return False
        return self.right.evaluate(state, context)


class OrNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        # Short-circuit: don't evaluate right if left is true
        left = self.left.evaluate(state, context)
        if left:
            return True
        return self.right.evaluate(state, context)


class NotNode(ASTNode):
    def __init__(self, operand: ASTNode):
        self.operand = operand

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        value = self.operand.evaluate(state, context)
        if value is None:
            return True
        return not bool(value)


class ConditionEvaluator:
    """
    Condition Expression Evaluator

    Supports complete condition expression syntax as defined in LinJ 14.x:
    - Comparisons: ==, !=, >, >=, <, <=
    - Logical: AND, OR, NOT (short-circuit)
    - Functions: exists(path), len(path), value(path)
    """

    def __init__(self):
        self._cache: Dict[str, ASTNode] = {}

    def evaluate(
        self,
        expression: str,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate condition expression

        Args:
            expression: Condition expression string
            state: Main state object
            context: Additional context (e.g., signal.payload)

        Returns:
            Boolean result

        Raises:
            SyntaxError: Expression syntax error
            TypeError: Type mismatch
        """
        # Parse and cache AST
        if expression not in self._cache:
            parser = Parser(expression)
            self._cache[expression] = parser.parse()

        ast = self._cache[expression]
        result = ast.evaluate(state, context)

        if result is None:
            return False
        return bool(result)

    def clear_cache(self) -> None:
        """Clear parse cache"""
        self._cache.clear()


# Global evaluator instance
_default_evaluator: Optional[ConditionEvaluator] = None


def get_evaluator() -> ConditionEvaluator:
    """Get global condition evaluator instance"""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = ConditionEvaluator()
    return _default_evaluator


def evaluate_condition(
    expression: str, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Convenience function: evaluate condition expression

    Args:
        expression: Condition expression string
        state: Main state object
        context: Additional context

    Returns:
        Boolean result
    """
    return get_evaluator().evaluate(expression, state, context)


def clear_condition_cache() -> None:
    """Clear condition expression cache"""
    get_evaluator().clear_cache()
