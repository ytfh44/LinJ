"""
研究写作流水线示例

演示如何使用 LinJ 框架创建复杂的工作流：
1. 研究阶段 (ToolNode)
2. 生成草稿 (HintNode + ToolNode)
3. 审计阶段 (ToolNode)
4. 质量门控 (GateNode)
5. 优化循环 (Loop)
6. 术语检查 (JoinNode)
7. 发布阶段 (ToolNode)

支持两种后端：
- AutoGen: 使用 linj_autogen.executor.runner
- LangGraph: 使用 langgraph 模块
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 确保 autogen/src 在路径中
autogen_src = project_root / "autogen" / "src"
if str(autogen_src) not in sys.path:
    sys.path.insert(0, str(autogen_src))

# 导入工具函数
from tools import research_topic, audit_article, generic_llm, publish_result


def get_backend(backend_type: str = "autogen"):
    """
    获取指定类型的后端执行器

    Args:
        backend_type: "autogen" 或 "langgraph"

    Returns:
        后端对象
    """
    if backend_type.lower() == "langgraph":
        try:
            from examples.langgraph_backend import LangGraphBackend

            return LangGraphBackend(enable_tracing=True)
        except ImportError:
            print("Warning: LangGraph backend not available, falling back to AutoGen")
            return get_backend("autogen")
    else:
        from linj_autogen.executor.runner import LinJExecutor

        return LinJExecutor(enable_tracing=True)


def get_load_document_func(backend_type: str):
    """获取适合当前后端的 load_document 函数"""
    if backend_type.lower() == "langgraph":
        try:
            from examples.langgraph_backend import load_document

            return load_document
        except ImportError:
            pass

    from linj_autogen.executor.runner import load_document

    return load_document


async def main(backend_type: str = "autogen"):
    """
    主函数

    Args:
        backend_type: 要使用的后端类型
    """
    print("=== LinJ Complex Demo: Automated Research and Refinement Pipeline ===")
    print(f"Using backend: {backend_type}")
    print(f"Project root: {project_root}")

    # 1. 加载文档
    yaml_path = current_dir / "workflow.yaml"
    load_document = get_load_document_func(backend_type)
    doc = load_document(str(yaml_path))

    # 2. 初始化执行器
    executor = get_backend(backend_type)

    # 3. 注册工具
    executor.register_tool("research_topic", research_topic)
    executor.register_tool("audit_article", audit_article)
    executor.register_tool("llm_call", generic_llm)
    executor.register_tool("publish_result", publish_result)

    # 4. 初始状态
    initial_state = {
        "input": {"topic": "The Future of Artificial Gravity in Space Exploration"}
    }

    # 5. 运行
    print(f"\n>> Starting execution for topic: {initial_state['input']['topic']}")
    try:
        final_state = await executor.run(doc, initial_state)
        print("\n>> Execution Completed Successfully!")

        # 显示结果
        score = final_state.get("audit_report", {}).get("score", "N/A")
        print(f">> Final Audit Score: {score}")
        publish_url = final_state.get("publish_result", {}).get("publish_url", "N/A")
        print(f">> Publish URL: {publish_url}")

        # 显示最终内容预览
        final_content = final_state.get("final_content", "")
        print(
            f"\n>> Final Content Preview (first 200 chars):\n{final_content[:200]}..."
        )

    except Exception as e:
        print(f"\n!! Execution Failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="研究写作流水线示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --backend autogen      # 使用 AutoGen 后端
  python main.py --backend langgraph    # 使用 LangGraph 后端
  python main.py --backend all          # 依次运行两个后端
        """,
    )

    parser.add_argument(
        "--backend",
        choices=["autogen", "langgraph", "all"],
        default="autogen",
        help="选择后端类型 (默认: autogen)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.backend == "all":
        for backend in ["autogen", "langgraph"]:
            print(f"\n\n{'#' * 60}")
            print(f"# Running with {backend.upper()} backend")
            print(f"{'#' * 60}")
            asyncio.run(main(backend_type=backend))
    else:
        asyncio.run(main(backend_type=args.backend))
