"""
条件表达式求值器

实现 LinJ 规范 14 节定义的条件表达式语法：
- 比较：== != > >= < <=
- 逻辑：AND OR NOT（短路）
- 函数：exists(path), len(path), value(path)
"""

import re
from typing import Any, Callable, Dict, Optional

from ..core.path import PathResolver
from ..core.errors import ConditionError


class ConditionEvaluator:
    """
    条件表达式求值器
    
    解析并求值条件表达式，支持短路求值
    """
    
    # 词法分析模式
    TOKEN_PATTERN = re.compile(
        r'\s*('                           # 可选空白
        r'(?P<LPAREN>\()|'                # 左括号
        r'(?P<RPAREN>\))|'                # 右括号
        r'(?P<OP>==|!=|>=|<=|>|<)|'       # 比较运算符
        r'(?P<LOGIC>AND|OR|NOT)|'         # 逻辑运算符
        r'(?P<FUN>exists|len|value)\s*\(|'  # 函数调用
        r'(?P<PATH>\$\.[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*|\[\d+\])*)|'  # 路径
        r'(?P<STRING>"[^"]*")|'          # 字符串
        r'(?P<NUMBER>-?\d+(?:\.\d+)?)|'   # 数字
        r'(?P<NULL>null)|'                # null
        r'(?P<BOOL>true|false)'           # 布尔值
        r')'
    )
    
    def __init__(self, state: Dict[str, Any]):
        """
        初始化求值器
        
        Args:
            state: 主状态对象，用于路径求值
        """
        self._state = state
        self._tokens: list = []
        self._pos: int = 0
    
    def evaluate(self, condition: str) -> bool:
        """
        求值条件表达式
        
        Args:
            condition: 条件表达式字符串
            
        Returns:
            布尔结果
            
        Raises:
            ConditionError: 表达式语法错误或类型不匹配
        """
        if not condition or condition.strip() == "":
            return True  # 空条件视为真
        
        self._tokens = self._tokenize(condition)
        self._pos = 0
        
        result = self._parse_or()
        
        if self._pos < len(self._tokens):
            raise ConditionError(f"Unexpected token: {self._tokens[self._pos]}")
        
        return bool(result)
    
    def _tokenize(self, condition: str) -> list:
        """词法分析"""
        tokens = []
        pos = 0
        
        while pos < len(condition):
            match = self.TOKEN_PATTERN.match(condition, pos)
            if not match:
                raise ConditionError(f"Invalid character at position {pos}: {condition[pos:]}")
            
            if match.group('LPAREN'):
                tokens.append(('LPAREN', '('))
            elif match.group('RPAREN'):
                tokens.append(('RPAREN', ')'))
            elif match.group('OP'):
                tokens.append(('OP', match.group('OP')))
            elif match.group('LOGIC'):
                tokens.append(('LOGIC', match.group('LOGIC')))
            elif match.group('FUN'):
                # 提取函数名和路径参数
                fun_name = match.group('FUN')
                # 找到匹配的右括号
                start = match.end()
                depth = 1
                end = start
                while end < len(condition) and depth > 0:
                    if condition[end] == '(':
                        depth += 1
                    elif condition[end] == ')':
                        depth -= 1
                    end += 1
                
                if depth != 0:
                    raise ConditionError(f"Unclosed function call: {fun_name}")
                
                path = condition[start:end-1].strip()
                if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
                    path = path[1:-1]
                
                tokens.append(('FUN', (fun_name, path)))
                pos = end
                continue
            elif match.group('PATH'):
                tokens.append(('PATH', match.group('PATH')))
            elif match.group('STRING'):
                tokens.append(('STRING', match.group('STRING')[1:-1]))  # 去掉引号
            elif match.group('NUMBER'):
                num_str = match.group('NUMBER')
                if '.' in num_str:
                    tokens.append(('NUMBER', float(num_str)))
                else:
                    tokens.append(('NUMBER', int(num_str)))
            elif match.group('NULL'):
                tokens.append(('NULL', None))
            elif match.group('BOOL'):
                tokens.append(('BOOL', match.group('BOOL') == 'true'))
            
            pos = match.end()
        
        return tokens
    
    def _current(self) -> Optional[tuple]:
        """获取当前 token"""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None
    
    def _consume(self, expected_type: Optional[str] = None) -> tuple:
        """消费当前 token"""
        token = self._current()
        if token is None:
            raise ConditionError("Unexpected end of expression")
        
        if expected_type and token[0] != expected_type:
            raise ConditionError(f"Expected {expected_type}, got {token[0]}")
        
        self._pos += 1
        return token
    
    def _parse_or(self) -> bool:
        """解析 OR 表达式（最低优先级）"""
        left = self._parse_and()
        
        while self._current() and self._current()[0] == 'LOGIC' and self._current()[1] == 'OR':
            self._consume('LOGIC')
            if left:  # 短路：如果左边为真，不需要求右边
                # 消费右操作数但不求值
                self._parse_and()
                return True
            left = self._parse_and()
        
        return left
    
    def _parse_and(self) -> bool:
        """解析 AND 表达式"""
        left = self._parse_not()
        
        while self._current() and self._current()[0] == 'LOGIC' and self._current()[1] == 'AND':
            self._consume('LOGIC')
            if not left:  # 短路：如果左边为假，不需要求右边
                # 消费右操作数但不求值
                self._parse_not()
                return False
            left = self._parse_not()
        
        return left
    
    def _parse_not(self) -> bool:
        """解析 NOT 表达式"""
        if self._current() and self._current()[0] == 'LOGIC' and self._current()[1] == 'NOT':
            self._consume('LOGIC')
            return not self._parse_not()
        
        return self._parse_comparison()
    
    def _parse_comparison(self) -> bool:
        """解析比较表达式"""
        left = self._parse_primary()
        
        if self._current() and self._current()[0] == 'OP':
            op = self._consume('OP')[1]
            right = self._parse_primary()
            
            # null 处理：只有 == 和 != 有意义
            if left is None or right is None:
                if op == '==':
                    return left == right
                elif op == '!=':
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
            if op == '==':
                return left == right
            elif op == '!=':
                return left != right
            elif op == '>':
                return left > right
            elif op == '>=':
                return left >= right
            elif op == '<':
                return left < right
            elif op == '<=':
                return left <= right
        
        # 单个值视为布尔值
        return bool(left)
    
    def _parse_primary(self) -> Any:
        """解析基本值"""
        token = self._current()
        
        if token is None:
            raise ConditionError("Unexpected end of expression")
        
        if token[0] == 'LPAREN':
            self._consume('LPAREN')
            result = self._parse_or()
            self._consume('RPAREN')
            return result
        
        if token[0] == 'FUN':
            return self._eval_function(token[1])
        
        if token[0] == 'PATH':
            self._consume()
            return PathResolver.get(self._state, token[1])
        
        if token[0] in ('STRING', 'NUMBER', 'NULL', 'BOOL'):
            self._consume()
            return token[1]
        
        raise ConditionError(f"Unexpected token: {token}")
    
    def _eval_function(self, fun_spec: tuple) -> Any:
        """求值函数调用"""
        fun_name, path = fun_spec
        self._pos += 1  # 消费 FUN token
        
        if fun_name == 'exists':
            # exists(path): 路径存在且值非 null 为真
            value = PathResolver.get(self._state, path)
            return value is not None
        
        elif fun_name == 'len':
            # len(path): 数组长度；非数组或不存在则为 0
            value = PathResolver.get(self._state, path)
            if isinstance(value, list):
                return len(value)
            return 0
        
        elif fun_name == 'value':
            # value(path): 取值，不存在则为 null
            return PathResolver.get(self._state, path)
        
        raise ConditionError(f"Unknown function: {fun_name}")


def evaluate_condition(condition: str, state: Dict[str, Any]) -> bool:
    """
    便捷的求值函数
    
    Args:
        condition: 条件表达式
        state: 主状态
        
    Returns:
        布尔结果
    """
    evaluator = ConditionEvaluator(state)
    return evaluator.evaluate(condition)
