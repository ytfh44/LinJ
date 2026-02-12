"""
LinJ/ContiText 异常定义

按规范定义各类错误，包括 ValidationError, ExecutionError, MappingError, ConditionError 等
"""

from typing import Any, Dict, Optional


class LinJError(Exception):
    """LinJ 基础异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(LinJError):
    """
    验证错误
    
    发生在文档验证阶段，如版本不匹配、必填字段缺失、循环无界等
    """
    pass


class ExecutionError(LinJError):
    """
    执行错误
    
    发生在节点执行阶段，如工具调用失败、状态超限等
    """
    pass


class MappingError(LinJError):
    """
    状态映射错误
    
    发生在路径写入时，如中间位置不是对象、数组越界无法扩容等
    """
    pass


class ConditionError(LinJError):
    """
    条件表达式错误
    
    发生在条件求值时，如类型不匹配、语法错误等
    """
    pass


class ConflictError(LinJError):
    """
    变更集冲突错误
    
    发生在变更集提交时，如写入路径相交、版本不匹配等
    """
    pass


class HandleExpired(LinJError):
    """
    续体句柄过期错误
    
    发生在恢复续体时，句柄已过期或无效
    """
    pass


class ResourceConstraintUnsatisfied(LinJError):
    """
    资源约束未满足错误
    
    当 placement 或 resource 依赖无法满足时
    """
    pass


class InvalidRequirements(LinJError):
    """
    requirements 格式错误
    
    当 requirements 字段值不为布尔类型时
    """
    pass


class InvalidPlacement(LinJError):
    """
    placement 格式错误
    
    当 placement 声明无效或无法满足时
    """
    pass


class ContractViolation(LinJError):
    """
    合同违反错误
    
    当输出不满足 out_contract 时
    """
    pass
