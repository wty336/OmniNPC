"""
LLM 统一调用客户端 — 封装火山方舟 Ark SDK。

火山方舟提供 OpenAI 兼容的 API 接口，
因此我们使用 volcenginesdkarkruntime 的 Ark 客户端，
其调用方式与 OpenAI SDK 一致。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger

from config.settings import settings


class LLMClient:
    """
    大模型调用客户端。

    基于火山方舟 Ark SDK（OpenAI 兼容接口）。
    支持普通对话、带 Function Calling 的对话、以及结构化 JSON 输出。

    注意: 因为 Ark SDK 本身是同步的，所以 chat() 也设计为同步方法。
    在 async 上下文中可直接调用（内部是 HTTP 请求，不会阻塞 event loop 太久）。
    """

    def __init__(self):
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError:
            raise ImportError(
                "请安装火山方舟 SDK: pip install 'volcengine-python-sdk[ark]'"
            )

        self._client = Ark(api_key=settings.llm.api_key)
        self._model = settings.llm.model_endpoint
        logger.info(f"LLM 客户端初始化完成，模型端点: {self._model}")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """
        发送对话请求（同步）。

        Parameters
        ----------
        messages : 消息列表 [{"role": "...", "content": "..."}]
        tools : 可选的工具定义列表（对应 Function Calling）
        temperature : 采样温度
        max_tokens : 最大生成 token 数
        response_format : 响应格式约束，如 {"type": "json_object"}

        Returns
        -------
        dict : 包含 content, tool_calls 等字段的响应字典
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature or settings.llm.temperature,
            "max_tokens": max_tokens or settings.llm.max_tokens,
            "top_p": settings.llm.top_p,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if response_format:
            kwargs["response_format"] = response_format

        logger.debug(
            f"LLM 请求: model={self._model}, "
            f"messages={len(messages)} 条, "
            f"tools={len(tools) if tools else 0} 个"
        )

        try:
            response = self._client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            result = {
                "content": choice.message.content or "",
                "role": choice.message.role,
                "tool_calls": None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

            # 解析 tool_calls
            if choice.message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]

            logger.debug(
                f"LLM 响应: tokens={result['usage']['total_tokens']}, "
                f"tool_calls={len(result['tool_calls']) if result['tool_calls'] else 0}"
            )
            return result

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        发送对话请求并解析 JSON 输出。

        不使用 response_format 参数（部分模型不支持），
        而是通过 prompt 约束 + 正则提取实现。

        Returns
        -------
        dict : 解析后的 JSON 对象
        """
        result = self.chat(
            messages=messages,
            temperature=temperature or 0.3,
        )
        content = result["content"].strip()

        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试从 ```json ... ``` 代码块中提取
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试提取第一个 { ... } 块
        brace_match = re.search(r'\{.*\}', content, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"JSON 解析失败，原始内容: {content[:200]}")
        return {"raw": content}


# 全局单例（延迟初始化）
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端全局单例。"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
