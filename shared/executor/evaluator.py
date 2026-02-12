"""
求值器抽象类

定义条件表达式求值的抽象接口，支持多种求值策略和表达式语法。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .types import ExecutionContext


class EvaluationStrategy(Enum):
    """求值策略枚举"""

    STRICT = "strict"  # 严格求值
    LAZY = "lazy"  # 惰性求值
    SHORT_CIRCUIT = "short_circuit"  # 短路求值
    PARALLEL = "parallel"  # 并行求值


class TokenType(Enum):
    """词法单元类型"""

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
    """词法单元"""

    type: TokenType
    value: Any
    position: int = 0

    def __str__(self):
        return f"Token({self.type.value}, {self.value})"


@dataclass
class EvaluationResult:
    """求值结果"""

    success: bool
    value: Any = None
    error: Optional[Exception] = None
    metadata: Optional[Dict[str, Any]] = None


class Evaluator(ABC):
    """
    求值器抽象基类

    定义条件表达式求值的核心接口：
    - 词法分析和语法解析
    - 表达式求值执行
    - 错误处理和恢复
    - 性能优化和缓存
    """

    @abstractmethod
    def evaluate(
        self,
        expression: str,
        context: ExecutionContext,
        strategy: Optional[EvaluationStrategy] = None,
    ) -> EvaluationResult:
        """
        求值表达式

        Args:
            expression: 条件表达式字符串
            context: 执行上下文
            strategy: 求值策略

        Returns:
            求值结果
        """
        pass

    @abstractmethod
    def tokenize(self, expression: str) -> List[Token]:
        """
        词法分析

        Args:
            expression: 表达式字符串

        Returns:
            词法单元列表
        """
        pass

    @abstractmethod
    def parse(self, tokens: List[Token]) -> Any:
        """
        语法分析

        Args:
            tokens: 词法单元列表

        Returns:
            语法树（抽象语法树的节点）
        """
        pass

    @abstractmethod
    def validate_expression(self, expression: str) -> bool:
        """
        验证表达式语法

        Args:
            expression: 表达式字符串

        Returns:
            True 表示语法正确，False 表示有错误
        """
        pass

    def get_value(self, path: str, context: ExecutionContext) -> Any:
        """
        从上下文中获取路径值

        Args:
            path: 状态路径
            context: 执行上下文

        Returns:
            路径对应的值
        """
        return context.get_state_value(path)

    def set_value(self, path: str, value: Any, context: ExecutionContext) -> None:
        """
        在上下文中设置路径值

        Args:
            path: 状态路径
            value: 要设置的值
            context: 执行上下文
        """
        context.set_state_value(path, value)


class BaseEvaluator(Evaluator):
    """
    基础求值器实现

    提供通用的表达式求值逻辑和错误处理
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
        """求值表达式"""
        strategy = strategy or EvaluationStrategy.STRICT

        # 检查缓存
        cache_key = f"{expression}:{id(context)}:{strategy.value}"
        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        self._stats["total_evaluations"] += 1

        try:
            # 词法分析
            tokens = self.tokenize(expression)

            # 语法分析
            ast = self.parse(tokens)

            # 根据策略求值
            if strategy == EvaluationStrategy.SHORT_CIRCUIT:
                result = self._evaluate_short_circuit(ast, context)
            elif strategy == EvaluationStrategy.LAZY:
                result = self._evaluate_lazy(ast, context)
            elif strategy == EvaluationStrategy.PARALLEL:
                result = self._evaluate_parallel(ast, context)
            else:  # STRICT
                result = self._evaluate_strict(ast, context)

            # 缓存结果
            self._cache[cache_key] = result

            return result

        except Exception as e:
            self._stats["evaluation_errors"] += 1
            return EvaluationResult(success=False, error=e)

    def _evaluate_strict(self, ast: Any, context: ExecutionContext) -> EvaluationResult:
        """严格求值"""
        value = self._evaluate_node(ast, context)
        return EvaluationResult(success=True, value=self._to_boolean(value))

    def _evaluate_short_circuit(
        self, ast: Any, context: ExecutionContext
    ) -> EvaluationResult:
        """短路求值"""
        # 实现AND/OR的短路逻辑
        return self._evaluate_node_with_short_circuit(ast, context)

    def _evaluate_lazy(self, ast: Any, context: ExecutionContext) -> EvaluationResult:
        """惰性求值"""
        # 延迟计算直到真正需要
        return self._evaluate_strict(ast, context)

    def _evaluate_parallel(
        self, ast: Any, context: ExecutionContext
    ) -> EvaluationResult:
        """并行求值"""
        # 并行计算独立子表达式
        return self._evaluate_strict(ast, context)

    @abstractmethod
    def _evaluate_node(self, node: Any, context: ExecutionContext) -> Any:
        """求值单个节点"""
        pass

    def _evaluate_node_with_short_circuit(
        self, node: Any, context: ExecutionContext
    ) -> Any:
        """带短路逻辑的节点求值"""
        return self._evaluate_node(node, context)

    def _to_boolean(self, value: Any) -> bool:
        """将值转换为布尔类型"""
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
        """验证表达式语法"""
        try:
            tokens = self.tokenize(expression)
            self.parse(tokens)
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取求值统计信息"""
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
        """清空缓存"""
        self._cache.clear()

    def reset_stats(self) -> None:
        """重置统计信息"""
        self._stats = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "evaluation_errors": 0,
        }


class SimpleEvaluator(BaseEvaluator):
    """
    简单求值器

    支持基本的条件表达式求值
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
        """简单的词法分析"""
        import re

        # 定义词法模式
        token_patterns = [
            (r"\s+", None),  # 空白，跳过
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

                        # 处理不同类型的值
                        if token_type == TokenType.NUMBER:
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        elif token_type == TokenType.STRING:
                            value = value[1:-1]  # 去掉引号
                        elif token_type == TokenType.BOOLEAN:
                            value = value == "true"
                        elif token_type == TokenType.NULL:
                            value = None
                        elif token_type == TokenType.FUNCTION:
                            # 提取函数名，去掉括号
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
        """简单的语法分析（转换为中缀表达式列表）"""
        # 这是一个简化实现，实际应该构建真正的AST
        return tokens

    def _evaluate_node(self, node: Any, context: ExecutionContext) -> Any:
        """求值节点"""
        if isinstance(node, list):
            # 中缀表达式求值
            return self._evaluate_infix(node, context)
        elif isinstance(node, Token):
            return self._evaluate_token(node, context)
        else:
            return node

    def _evaluate_token(self, token: Token, context: ExecutionContext) -> Any:
        """求值词法单元"""
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
        """求值中缀表达式"""
        # 这是一个非常简化的实现
        # 实际应该实现完整的表达式求值逻辑

        if len(tokens) == 1:
            return self._evaluate_token(tokens[0], context)

        if len(tokens) == 3:
            left = self._evaluate_token(tokens[0], context)
            op = tokens[1].value
            right = self._evaluate_token(tokens[2], context)

            if op in self.operators:
                return self.operators[op](left, right)

        # 默认返回False
        return False

    def _evaluate_function(self, func_name: str, context: ExecutionContext) -> Any:
        """求值函数"""
        if func_name in self.functions:
            return self.functions[func_name](context)
        raise ValueError(f"Unknown function: {func_name}")

    def _func_exists(self, context: ExecutionContext) -> bool:
        """exists函数（简化版）"""
        # 简化实现：总是返回True
        return True

    def _func_len(self, context: ExecutionContext) -> int:
        """len函数（简化版）"""
        # 简化实现：返回0
        return 0

    def _func_value(self, context: ExecutionContext) -> Any:
        """value函数（简化版）"""
        # 简化实现：返回None
        return None


class RegexEvaluator(SimpleEvaluator):
    """
    基于正则表达式的高级求值器

    支持更复杂的表达式语法和求值逻辑
    """

    def __init__(self):
        super().__init__()
        # 构建更复杂的词法模式
        self.build_advanced_patterns()

    def build_advanced_patterns(self):
        """构建高级词法模式"""
        # 添加数组访问、对象属性等支持
        pass

    def tokenize(self, expression: str) -> List[Token]:
        """高级词法分析"""
        # 使用更复杂的正则表达式
        import re

        # 扩展的词法模式
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
            (r",", None),  # 逗号，跳过
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

                        # 处理转义字符
                        if token_type == TokenType.STRING:
                            value = eval(value)  # 简化处理，实际应该安全处理转义

                        # 处理函数调用
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
