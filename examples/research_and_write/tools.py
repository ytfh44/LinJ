import asyncio
import random
from typing import List, Dict, Any

async def research_topic(topic: str) -> Dict[str, Any]:
    """模拟研究阶段，产生一些研究材料"""
    print(f"[Tool: Research] Topic: {topic}")
    await asyncio.sleep(0.5)
    materials = [
        f"Research finding 1 about {topic}",
        f"Statistical data on {topic}",
        f"Expert opinion on {topic}"
    ]
    return {
        "status": "researched",
        "materials": materials,
        "keywords": [topic, "research", "data"]
    }

async def audit_article(content: str) -> Dict[str, Any]:
    """评估文章质量，提供反馈"""
    print(f"[Tool: Audit] Auditing content...")
    await asyncio.sleep(0.5)
    
    # 简单的打分逻辑：字数越多分数越高 (但随机一点)
    score = min(max(len(content) // 10 + random.randint(-5, 5), 0), 100)
    
    feedback = "Good content." if score > 70 else "Needs more depth and detail."
    
    print(f"[Tool: Audit] Score: {score}, Feedback: {feedback}")
    
    return {
        "score": score,
        "feedback": feedback,
        "approved": score > 70
    }

async def generic_llm(prompt: str, model: str = "qwen3:0.6b") -> str:
    """通用 LLM 调用接口"""
    import httpx
    print(f"[Tool: LLM] Model: {model}, Prompt prefix: {prompt[:30]}...")
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json().get("response", "")
            return result.strip()
    except Exception as e:
        print(f"[Tool: LLM] Error calling Ollama: {e}. Falling back to mock.")
        return f"Mocked response for prompt: {prompt[:20]}..."

async def publish_result(title: str, content: str) -> Dict[str, Any]:
    """模拟发布结果"""
    print(f"[Tool: Publish] Publishing article: {title}")
    await asyncio.sleep(0.3)
    url = f"https://example.com/articles/{title.lower().replace(' ', '-')}"
    return {"publish_url": url}
