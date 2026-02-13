import asyncio
import random
from typing import List, Dict, Any


async def research_topic(topic: str) -> Dict[str, Any]:
    """Simulate research phase, generate research materials"""
    print(f"[Tool: Research] Topic: {topic}")
    await asyncio.sleep(0.5)
    materials = [
        f"Research finding 1 about {topic}",
        f"Statistical data on {topic}",
        f"Expert opinion on {topic}",
    ]
    return {
        "status": "researched",
        "materials": materials,
        "keywords": [topic, "research", "data"],
    }


async def audit_article(content: str) -> Dict[str, Any]:
    """Evaluate article quality, provide feedback"""
    print(f"[Tool: Audit] Auditing content...")
    await asyncio.sleep(0.5)

    # Simple scoring logic: longer content gets higher scores (with some randomness)
    score = min(max(len(content) // 10 + random.randint(-5, 5), 0), 100)

    feedback = "Good content." if score > 70 else "Needs more depth and detail."

    print(f"[Tool: Audit] Score: {score}, Feedback: {feedback}")

    return {"score": score, "feedback": feedback, "approved": score > 70}


async def generic_llm(prompt: str, model: str = "qwen3:0.6b") -> str:
    """Generic LLM call interface"""
    import httpx

    print(f"[Tool: LLM] Model: {model}, Prompt prefix: {prompt[:30]}...")

    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

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
    """Simulate publish result"""
    print(f"[Tool: Publish] Publishing article: {title}")
    await asyncio.sleep(0.3)
    url = f"https://example.com/articles/{title.lower().replace(' ', '-')}"
    return {"publish_url": url}
