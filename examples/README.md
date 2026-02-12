# LinJ Examples

本目录包含 LinJ 框架的示例程序，演示如何使用框架创建工作流。

## 示例列表

### 1. Product Name Generator (产品名称生成器)
**目录**: `product_name_generator/`

简单线性工作流示例，演示：
- 关键词提取 (ToolNode)
- 提示词生成 (HintNode)
- LLM 调用 (ToolNode)
- 词汇验证 (JoinNode)
- 质量门控 (GateNode)

**运行命令**:
```bash
# 使用 AutoGen 后端（默认）
python examples/product_name_generator/main.py

# 使用 LangGraph 后端
python examples/product_name_generator/main.py --backend langgraph

# 依次运行两个后端
python examples/product_name_generator/main.py --backend all
```

### 2. Research and Write (研究写作流水线)
**目录**: `research_and_write/`

复杂工作流示例，演示：
- 研究阶段 (ToolNode)
- 草稿生成 (HintNode + ToolNode)
- 内容审计 (ToolNode)
- 质量门控与分支 (GateNode)
- 优化循环 (Loop)
- 术语检查 (JoinNode)
- 发布阶段 (ToolNode)

**运行命令**:
```bash
# 使用 AutoGen 后端（默认）
python examples/research_and_write/main.py

# 使用 LangGraph 后端
python examples/research_and_write/main.py --backend langgraph

# 依次运行两个后端
python examples/research_and_write/main.py --backend all
```

## 后端支持

示例支持两种执行后端：

### AutoGen 后端
- 使用 `linj_autogen.executor.runner.LinJExecutor`
- 完整的 AutoGen 风格执行
- 默认后端

### LangGraph 后端
- 使用 `langgraph` 模块
- 提供统一接口
- 当 LangGraph 完整实现后可用

## 通用接口

所有示例使用统一的接口模式：

```python
import asyncio
from examples.langgraph_backend import create_backend, load_document

# 1. 加载工作流文档
doc = load_document("workflow.yaml")

# 2. 创建后端
backend = create_backend(backend_type="autogen")

# 3. 注册工具
backend.register_tool("tool_name", tool_function)

# 4. 执行工作流
result = await backend.run(doc, initial_state)
```

## 项目结构

```
examples/
├── langgraph_backend.py      # 后端适配层（统一接口）
├── product_name_generator/   # 简单线性工作流示例
│   ├── main.py              # 入口文件
│   └── workflow.yaml        # 工作流定义
└── research_and_write/       # 复杂循环工作流示例
    ├── main.py              # 入口文件
    ├── tools.py             # 工具函数实现
    └── workflow.yaml        # 工作流定义
```

## 要求

- Python 3.11+
- 依赖安装: `pip install -e .`
- Ollama (可选，用于 LLM 调用)

## 注意事项

1. **LangGraph 后端**: 当前 LangGraph Executor 尚未完整实现，使用时会使用模拟执行作为回退方案
2. **httpx**: 示例中使用 httpx 调用本地 Ollama 服务，确保 Ollama 运行在 localhost:11434
3. **路径设置**: 示例会自动处理路径，无需手动配置 PYTHONPATH
