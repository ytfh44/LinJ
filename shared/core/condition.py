"""
条件表达式求值器

实现 LinJ 规范第 14 节定义的条件表达式语法和语义：
- 比较运算符: ==, !=, >, >=, <, <=
- 逻辑运算符: AND, OR, NOT (必须短路)
- 函数: exists(path), len(path), value(path)

安全实现：不使用 eval()，使用递归下降解析器
"""

import re
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .path import PathResolver


class TokenType(Enum):
    """词法标记类型"""

    # 字面量
    NUMBER = "NUMBER"
    STRING = "STRING"
    NULL = "NULL"
    BOOLEAN = "BOOLEAN"

    # 标识符
    IDENTIFIER = "IDENTIFIER"

    # 比较运算符
    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="

    # 逻辑运算符
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # 括号
    LPAREN = "("
    RPAREN = ")"

    # 函数调用
    COMMA = ","

    # 结束
    EOF = "EOF"


@dataclass
class Token:
    """词法标记"""

    type: TokenType
    value: Any
    position: int


class Lexer:
    """词法分析器"""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None

    def advance(self) -> None:
        """前进到下一个字符"""
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self) -> None:
        """跳过空白字符"""
        while self.current_char and self.current_char.isspace():
            self.advance()

    def read_number(self) -> Token:
        """读取数字"""
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
        """读取字符串"""
        start = self.pos
        self.advance()  # 跳过开始的引号

        while self.current_char and self.current_char != '"':
            if self.current_char == "\\" and self.text[self.pos + 1]:
                self.advance()  # 跳过转义字符
            self.advance()

        self.advance()  # 跳过结束的引号
        return Token(TokenType.STRING, self.text[start + 1 : self.pos - 1], start)

    def read_identifier(self) -> Token:
        """读取标识符或关键字"""
        start = self.pos
        while self.current_char and (
            self.current_char.isalnum() or self.current_char == "_"
        ):
            self.advance()

        value = self.text[start : self.pos]

        # 检查是否是关键字
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
        """查看下一个字符"""
        peek_pos = self.pos + 1
        return self.text[peek_pos] if peek_pos < len(self.text) else None

    def next_token(self) -> Token:
        """获取下一个标记"""
        self.skip_whitespace()

        if self.current_char is None:
            return Token(TokenType.EOF, None, self.pos)

        start = self.pos

        # 检查两个字符的运算符
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

        # 单字符运算符
        if self.current_char == ">":
            self.advance()
            return Token(TokenType.GT, ">", start)

        if self.current_char == "<":
            self.advance()
            return Token(TokenType.LT, "<", start)

        # 括号和逗号
        if self.current_char == "(":
            self.advance()
            return Token(TokenType.LPAREN, "(", start)

        if self.current_char == ")":
            self.advance()
            return Token(TokenType.RPAREN, ")", start)

        if self.current_char == ",":
            self.advance()
            return Token(TokenType.COMMA, ",", start)

        # 路径引用 (以 $ 开头)
        if self.current_char == "$":
            self.advance()
            path_value = "$"
            # 收集路径段
            while self.current_char and (
                self.current_char.isalnum()
                or self.current_char in "._[]"
                or self.current_char == "$"
            ):
                path_value += self.current_char
                self.advance()
            return Token(TokenType.STRING, path_value, start)

        # 数字或字符串
        if self.current_char.isdigit():
            return self.read_number()

        if self.current_char == '"':
            return self.read_string()

        # 标识符或关键字
        if self.current_char.isalpha() or self.current_char == "_":
            return self.read_identifier()

        raise SyntaxError(
            f"Unexpected character: {self.current_char} at position {self.pos}"
        )


class Parser:
    """递归下降解析器"""

    def __init__(self, text: str):
        self.lexer = Lexer(text)
        self.current_token = self.lexer.next_token()

    def advance(self) -> None:
        """前进到下一个标记"""
        self.current_token = self.lexer.next_token()

    def expect(self, token_type: TokenType) -> Token:
        """期望特定类型的标记"""
        if self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        raise SyntaxError(f"Expected {token_type}, got {self.current_token.type}")

    def parse(self) -> "ASTNode":
        """解析表达式"""
        node = self.parse_or()
        if self.current_token.type != TokenType.EOF:
            raise SyntaxError(f"Unexpected token: {self.current_token.type}")
        return node

    # 优先级：NOT > 比较 > AND > OR
    def parse_or(self) -> "ASTNode":
        """解析 OR 表达式"""
        node = self.parse_and()

        while self.current_token.type == TokenType.OR:
            self.advance()
            right = self.parse_and()
            node = OrNode(node, right)

        return node

    def parse_and(self) -> "ASTNode":
        """解析 AND 表达式"""
        node = self.parse_not()

        while self.current_token.type == TokenType.AND:
            self.advance()
            right = self.parse_not()
            node = AndNode(node, right)

        return node

    def parse_not(self) -> "ASTNode":
        """解析 NOT 表达式"""
        if self.current_token.type == TokenType.NOT:
            self.advance()
            operand = self.parse_not()
            return NotNode(operand)

        return self.parse_comparison()

    def parse_comparison(self) -> "ASTNode":
        """解析比较表达式"""
        left = self.parse_primary()

        # 比较运算符
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
        """解析基本表达式"""
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
        """解析标识符（可能是函数调用或变量）"""
        name = self.current_token.value
        self.advance()

        # 检查是否是函数调用
        if self.current_token.type == TokenType.LPAREN:
            self.advance()
            args = []

            if self.current_token.type != TokenType.RPAREN:
                args.append(self.parse_or())

                while self.current_token.type == TokenType.COMMA:
                    self.advance()
                    args.append(self.parse_or())

            self.expect(TokenType.RPAREN)

            # 内置函数
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

            # 未知函数调用
            return FunctionNode(name, args)

        # 标识符引用
        return IdentifierNode(name)


# AST 节点类
class ASTNode:
    """AST 节点基类"""

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """求值"""
        raise NotImplementedError


class NumberNode(ASTNode):
    """数字节点"""

    def __init__(self, value: float):
        self.value = value

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self.value


class StringNode(ASTNode):
    """字符串节点"""

    def __init__(self, value: str):
        self.value = value

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self.value


class BooleanNode(ASTNode):
    """布尔节点"""

    def __init__(self, value: bool):
        self.value = value

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return self.value


class NullNode(ASTNode):
    """空值节点"""

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        return None


class IdentifierNode(ASTNode):
    """标识符节点 - 引用状态中的值"""

    def __init__(self, name: str):
        self.name = name

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        # 首先检查是否是上下文中的值（如 signal.payload）
        if context and self.name in context:
            return context[self.name]

        # 检查是否是特殊变量
        if self.name == "true":
            return True
        if self.name == "false":
            return False
        if self.name == "null":
            return None

        # 如果是路径引用（以 $ 开头），使用路径解析器
        if self.name.startswith("$"):
            return PathResolver.get(state, self.name)

        # 否则作为直接键名查找
        return state.get(self.name)


class ExistsNode(ASTNode):
    """exists() 函数节点"""

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
    """len() 函数节点"""

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
    """value() 函数节点"""

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
    """用户定义函数节点"""

    def __init__(self, name: str, args: list):
        self.name = name
        self.args = args

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        raise NameError(f"Unknown function: {self.name}")


# 比较运算符节点
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


# 逻辑运算符节点（支持短路）
class AndNode(ASTNode):
    def __init__(self, left: ASTNode, right: ASTNode):
        self.left = left
        self.right = right

    def evaluate(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        # 短路：左侧为 false 时不求值右侧
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
        # 短路：左侧为 true 时不求值右侧
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
    条件表达式求值器

    支持 LinJ 14.x 节定义的完整条件表达式语法：
    - 比较: ==, !=, >, >=, <, <=
    - 逻辑: AND, OR, NOT (短路)
    - 函数: exists(path), len(path), value(path)
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
        求值条件表达式

        Args:
            expression: 条件表达式字符串
            state: 主状态对象
            context: 额外上下文（如 signal.payload）

        Returns:
            布尔值结果

        Raises:
            SyntaxError: 表达式语法错误
            TypeError: 类型不匹配
        """
        # 解析并缓存 AST
        if expression not in self._cache:
            parser = Parser(expression)
            self._cache[expression] = parser.parse()

        ast = self._cache[expression]
        result = ast.evaluate(state, context)

        if result is None:
            return False
        return bool(result)

    def clear_cache(self) -> None:
        """清除解析缓存"""
        self._cache.clear()


# 全局求值器实例
_default_evaluator: Optional[ConditionEvaluator] = None


def get_evaluator() -> ConditionEvaluator:
    """获取全局条件求值器实例"""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = ConditionEvaluator()
    return _default_evaluator


def evaluate_condition(
    expression: str, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    便捷函数：求值条件表达式

    Args:
        expression: 条件表达式字符串
        state: 主状态对象
        context: 额外上下文

    Returns:
        布尔值结果
    """
    return get_evaluator().evaluate(expression, state, context)


def clear_condition_cache() -> None:
    """清除条件表达式缓存"""
    get_evaluator().clear_cache()
