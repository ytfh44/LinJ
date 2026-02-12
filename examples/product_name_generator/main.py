"""
产品名称生成器示例

演示如何使用 LinJ 框架创建简单的工作流：
1. 提取关键词 (ToolNode)
2. 生成提示词 (HintNode)
3. 调用 LLM 生成名称 (ToolNode)
4. 验证与输出 (JoinNode)
5. 质量门控 (GateNode)

支持两种后端：
- AutoGen: 使用 linj_autogen.executor.runner
- LangGraph: 使用 langgraph 模块
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 确保 src 目录在路径中
src_path = project_root / "autogen" / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def get_backend(backend_type: str = "autogen"):
    """
    获取指定类型的后端执行器

    Args:
        backend_type: "autogen" 或 "langgraph"

    Returns:
        后端对象
    """
    if backend_type.lower() == "langgraph":
        # 尝试导入 LangGraph 后端
        try:
            from examples.langgraph_backend import LangGraphBackend

            return LangGraphBackend(enable_tracing=True)
        except ImportError:
            print("Warning: LangGraph backend not available, falling back to AutoGen")
            return get_backend("autogen")
    else:
        # 默认使用 AutoGen 后端
        from linj_autogen.executor.runner import LinJExecutor

        return LinJExecutor(enable_tracing=True)


# --- 工具函数 ---


async def extract_keywords(description: str) -> List[str]:
    """
    从产品描述中提取关键词

    Args:
        description: 产品描述文本

    Returns:
        提取的关键词列表
    """
    print(f"[Tool: ExtractKeywords] Input: {description}")

    # 简单逻辑：提取长度大于4的单词
    words = description.split()
    keywords = [w for w in words if len(w) > 4]

    print(f"[Tool: ExtractKeywords] Output: {keywords}")
    return keywords


async def ollama_llm(prompt: str, model: str = "qwen3:0.6b") -> str:
    """
    通过 Ollama 调用 LLM 生成内容

    Args:
        prompt: 提示词
        model: 模型名称

    Returns:
        LLM 生成的内容
    """
    print(f"[Tool: Ollama] Sending prompt to {model}...")

    try:
        import httpx

        url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}

        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json().get("response", "")

            print(f"[Tool: Ollama] Received response ({len(result)} chars)")
            return result.strip()

    except Exception as e:
        print(f"[Tool: Ollama] Error: {e}")
        # 返回模拟结果作为后备
        mock_response = (
            "1. SuperProduct\n"
            "2. MegaTool\n"
            "3. UltraService\n"
            "4. PowerHouse\n"
            "5. PrimeChoice"
        )
        return mock_response


# --- 运行场景 ---


async def run_scenario(
    name: str, input_desc: str, expect_fail: bool = False, backend_type: str = "autogen"
):
    """
    运行指定场景

    Args:
        name: 场景名称
        input_desc: 输入描述
        expect_fail: 是否预期失败
        backend_type: 后端类型 ("autogen" 或 "langgraph")
    """
    print(f"\n{'=' * 60}")
    print(f"--- Running Scenario: {name} (Backend: {backend_type}) ---")
    print(f"{'=' * 60}")

    # 1. 加载文档
    yaml_path = current_dir / "workflow.yaml"

    # 根据后端类型选择导入
    if backend_type.lower() == "langgraph":
        try:
            from examples.langgraph_backend import load_document
        except ImportError:
            from linj_autogen.executor.runner import load_document
    else:
        from linj_autogen.executor.runner import load_document

    doc = load_document(str(yaml_path))

    # 2. 初始化执行器
    executor = get_backend(backend_type)

    # 3. 注册工具
    executor.register_tool("extract_keywords", extract_keywords)
    executor.register_tool("mock_llm", ollama_llm)

    # 4. 初始状态
    initial_state = {"input": {"description": input_desc}}

    try:
        # 5. 执行工作流
        final_state = await executor.run(doc, initial_state)

        print("\n>> Execution Completed Successfully!")
        print(f">> Final Names:\n{final_state.get('final_names')}")

        if expect_fail:
            print(">> Note: Scenario completed without validation error")

    except Exception as e:
        if expect_fail:
            print(f">> Caught EXPECTED error: {type(e).__name__}: {e}")
        else:
            print(f"!! UNEXPECTED Execution Failed: {type(e).__name__}: {e}")
            raise


async def main(backend_type: str = "autogen"):
    """
    主函数

    Args:
        backend_type: 要使用的后端类型
    """
    print(f"=== Product Name Generator Demo ===")
    print(f"Using backend: {backend_type}")
    print(f"Project root: {project_root}")

    # 场景1: 有效输入
    await run_scenario(
        "Valid Product",
        "This is a fantastic machine that makes coffee.",
        expect_fail=False,
        backend_type=backend_type,
    )

    # 场景2: 无效输入（触发词汇检查）
    await run_scenario(
        "Invalid Product (triggers glossary check)",
        "This is a investment scheme with high risk.",
        expect_fail=True,
        backend_type=backend_type,
    )


def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="产品名称生成器示例",
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

    parser.add_argument(
        "--scenario",
        choices=["valid", "invalid", "all"],
        default="all",
        help="选择运行的场景 (默认: all)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.backend == "all":
        # 依次运行两个后端
        for backend in ["autogen", "langgraph"]:
            print(f"\n\n{'#' * 60}")
            print(f"# Running with {backend.upper()} backend")
            print(f"{'#' * 60}")
            asyncio.run(main(backend_type=backend))
    else:
        # 运行指定后端
        asyncio.run(main(backend_type=args.backend))
