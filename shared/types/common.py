"""
LinJ 共享类型定义

提供跨多个后端实现的通用类型定义
"""

from typing import Any, Dict, List, Optional, Union, Set, Tuple, Callable, Mapping

# 基础类型别名
NodeID = str
PathString = str
StateValue = Any
MappingValue = Any

# 工具相关类型
ToolName = str
ToolArgs = Dict[str, Any]
ToolResult = Any

# 节点相关类型
NodeList = List[Dict[str, Any]]
EdgeList = List[Dict[str, Any]]

# 状态管理类型
StateDict = Dict[str, Any]
StateSnapshot = Dict[str, Any]

# 条件评估类型
ConditionExpression = str
ConditionResult = bool

# 合同验证类型
ContractType = str  # object/array/string/number/boolean/null
RequiredFields = List[str]
ContractProperties = Dict[str, "Contract"]

# 执行相关类型
ExecutionID = str
StepID = int
Revision = int

# 错误相关类型
ErrorMessage = str
ErrorDetails = Dict[str, Any]

# 映射相关类型
SourcePath = str
TargetPath = str
DefaultValue = Any

# 路径段类型
PathKey = Union[str, int]
PathSegmentList = List[Tuple[str, PathKey]]

# 依赖图类型
DependencySet = Set[NodeID]
Weight = float

# 变更集类型
WriteOperations = List[Tuple[PathString, StateValue]]
DeleteOperations = List[PathString]

# 循环相关类型
LoopMode = str  # finite/infinite
StopCondition = Optional[str]
MaxRounds = Optional[int]

# 策略相关类型
MaxSteps = Optional[int]
MaxRoundsPolicy = Optional[int]
TimeoutMs = Optional[int]
RetryPolicy = Optional[Dict[str, Any]]
MaxArrayLength = Optional[int]
MaxLocalStateBytes = Optional[int]

# 放置相关类型
DomainLabel = str
PlacementTarget = str

# 验证结果类型
ValidationErrors = List[str]
ValidationWarnings = List[str]

# 跟踪相关类型
TraceEvent = str
TraceData = Dict[str, Any]
TraceLevel = str
