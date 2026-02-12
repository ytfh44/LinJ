"""
条件表达式求值器实现

从autogen/executor/evaluator.py迁移并重构的求值器实现，兼容现有AutoGen求值逻辑。
"""

import re
from typing import Any, Dict, Optional, List

# 尝试导入现有模块进行兼容
try:
    from ..core.path import PathResolver
    from ..core.errors import ConditionError
except ImportError:
    # 回退到基本实现
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
    AutoGen兼容的条件表达式求值器

    实现 LinJ 规范 14 节定义的条件表达式语法：
    - 比较：== != > >= < <=
    - 逻辑：AND OR NOT（短路）
    - 函数：exists(path), len(path), value(path)
    """

    # 词法分析模式
    TOKEN_PATTERN = re.compile(
        r"\s*("  # 可选空白
        r"(?P<LPAREN>\()|"  # 左括号
        r"(?P<RPAREN>\))|"  # 右括号
        r"(?P<OP>==|!=|>=|<=|>|<)|"  # 比较运算符
        r"(?P<LOGIC>AND|OR|NOT)|"  # 逻辑运算符
        r"(?P<FUN>exists|len|value)\s*\(|"  # 函数调用
        r"(?P<PATH>\$\.[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*|\[\d+\])*)|"  # 路径
        r'(?P<STRING>"[^"]*")|'  # 字符串
        r"(?P<NUMBER>-?\d+(?:\.\d+)?)|"  # 数字
        r"(?P<NULL>null)|"  # null
        r"(?P<BOOL>true|false)"  # 布尔值
        r")"
    )

    def __init__(self, state: Dict[str, Any]):
        """
        初始化求值器

        Args:
            state: 主状态对象，用于路径求值
        """
        super().__init__()
        self._state = state
        self._tokens: List[Token] = []
        self._pos: int = 0

    def evaluate(
        self,
        expression: str,
        context=None,  # 兼容原有接口
        strategy: Optional[EvaluationStrategy] = None,
    ) -> EvaluationResult:
        """
        求值条件表达式

        Args:
            expression: 条件表达式字符串
            context: 执行上下文（兼容性参数）
            strategy: 求值策略

        Returns:
            求值结果

        Raises:
            ConditionError: 表达式语法错误或类型不匹配
        """
        if not expression or expression.strip() == "":
            return EvaluationResult(success=True, value=True)  # 空条件视为真

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
        """词法分析"""
        return self._tokenize(expression)

    def _tokenize(self, condition: str) -> List[Token]:
        """词法分析"""
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
                # 提取函数名和路径参数
                fun_name = match.group("FUN")
                # 找到匹配的右括号
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
                token = Token(TokenType.STRING, match.group("STRING")[1:-1])  # 去掉引号
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
        """语法分析（简化实现）"""
        self._tokens = tokens
        self._pos = 0
        return self._parse_or()

    def _current(self) -> Optional[Token]:
        """获取当前 token"""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _consume(self, expected_type: Optional[TokenType] = None) -> Token:
        """消费当前 token"""
        token = self._current()
        if token is None:
            raise ConditionError("Unexpected end of expression")

        if expected_type and token.type != expected_type:
            raise ConditionError(f"Expected {expected_type}, got {token.type}")

        self._pos += 1
        return token

    def _parse_or(self) -> bool:
        """解析 OR 表达式（最低优先级）"""
        left = self._parse_and()

        while (
            self._current()
            and self._current().type == TokenType.LOGIC
            and self._current().value == "OR"
        ):
            self._consume(TokenType.LOGIC)
            if left:  # 短路：如果左边为真，不需要求右边
                # 消费右操作数但不求值
                self._parse_and()
                return True
            left = self._parse_and()

        return left

    def _parse_and(self) -> bool:
        """解析 AND 表达式"""
        left = self._parse_not()

        while (
            self._current()
            and self._current().type == TokenType.LOGIC
            and self._current().value == "AND"
        ):
            self._consume(TokenType.LOGIC)
            if not left:  # 短路：如果左边为假，不需要求右边
                # 消费右操作数但不求值
                self._parse_not()
                return False
            left = self._parse_not()

        return left

    def _parse_not(self) -> bool:
        """解析 NOT 表达式"""
        if (
            self._current()
            and self._current().type == TokenType.LOGIC
            and self._current().value == "NOT"
        ):
            self._consume(TokenType.LOGIC)
            return not self._parse_not()

        return self._parse_comparison()

    def _parse_comparison(self) -> bool:
        """解析比较表达式"""
        left = self._parse_primary()

        if self._current() and self._current().type == TokenType.OPERATOR:
            op = self._consume(TokenType.OPERATOR).value
            right = self._parse_primary()

            # null 处理：只有 == 和 != 有意义
            if left is None or right is None:
                if op == "==":
                    return left == right
                elif op == "!=":
                    return left != right
                else:
                    # 14.2 节：null 参与其他比较结果为 false
                    return False

            # 类型检查
            if type(left) != type(right):
                raise ConditionError(
                    f"Type mismatch in comparison: {type(left).__name__} {op} {type(right).__name__}"
                )

            # 执行比较
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

        # 单个值视为布尔值
        return bool(left)

    def _parse_primary(self) -> Any:
        """解析基本值"""
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
        """求值函数调用"""
        fun_name, path = fun_spec
        self._pos += 1  # 消费 FUN token

        if fun_name == "exists":
            # exists(path): 路径存在且值非 null 为真
            if hasattr(PathResolver, "get"):
                value = PathResolver.get(self._state, path)
                return value is not None
            return False

        elif fun_name == "len":
            # len(path): 数组长度；非数组或不存在则为 0
            if hasattr(PathResolver, "get"):
                value = PathResolver.get(self._state, path)
                if isinstance(value, list):
                    return len(value)
            return 0

        elif fun_name == "value":
            # value(path): 取值，不存在则为 null
            if hasattr(PathResolver, "get"):
                return PathResolver.get(self._state, path)
            return None

        raise ConditionError(f"Unknown function: {fun_name}")

    def validate_expression(self, expression: str) -> bool:
        """验证表达式语法"""
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
    简单路径解析器

    在PathResolver不可用时的回退实现
    """

    @staticmethod
    def get(state: Dict[str, Any], path: str) -> Any:
        """获取路径值"""
        if not path.startswith("$"):
            return None

        # 简化实现：去掉$然后按.分割
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
        """检查路径是否相交"""
        parts_a = path_a[1:].split(".") if path_a.startswith("$") else path_a.split(".")
        parts_b = path_b[1:].split(".") if path_b.startswith("$") else path_b.split(".")

        min_len = min(len(parts_a), len(parts_b))
        for i in range(min_len):
            if parts_a[i] != parts_b[i]:
                return False
        return min_len > 0


# 在PathResolver不可用时使用简单实现
try:
    from ..core.path import PathResolver
except ImportError:
    PathResolver = SimplePathResolver


def evaluate_condition(condition: str, state: Dict[str, Any]) -> bool:
    """
    便捷的求值函数

    Args:
        condition: 条件表达式
        state: 主状态

    Returns:
        布尔结果
    """
    evaluator = AutoGenConditionEvaluator(state)
    result = evaluator.evaluate(condition)
    return result.value if result.success else False
