# 通过 OpenAI 软件开发工具包（SDK）连接 DeepSeek，读取模型配置并声明容器命令执行工具。
# 根据传入模式关闭思考或启用最高档思考，发送完整对话，返回模型消息、停止原因及原始用量等元数据。
# 保留思考内容供后续工具调用回传，关闭 SDK 内部重试并在调用层做显式退避重试，保证每次尝试可观测。

from collections.abc import Callable
import os
import time

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

MODEL_NAME = "deepseek-flash"
MAX_TRANSIENT_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": "在容器工作区执行 Bash 脚本",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "需要执行的 Bash 脚本",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def create_client() -> OpenAI:
    """检查 DeepSeek 配置并创建本次运行的客户端。"""
    # 要求显式配置，防止客户端回退到 OpenAI 的地址或密钥。
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    if not api_key:
        raise ValueError("缺少模型配置 DEEPSEEK_API_KEY")
    if not base_url:
        raise ValueError("缺少模型配置 DEEPSEEK_BASE_URL")
    if os.getenv("DEEPSEEK_MODEL", MODEL_NAME) != MODEL_NAME:
        raise ValueError(
            f"当前预算价目只适用于 {MODEL_NAME}。更换模型前须更新价目"
        )

    # 关闭软件开发工具包（SDK）内部重试，使每轮请求可以明确记账。
    return OpenAI(
        api_key=api_key, base_url=base_url, timeout=120.0, max_retries=0
    )


def _is_retryable_error(error: Exception) -> bool:
    """判断异常是否属于可安全重试的瞬时网络或服务端故障。"""
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in RETRYABLE_STATUS_CODES
    return False


def send_messages(
    client: OpenAI,
    messages: list[dict],
    *,
    thinking_mode: str = "disabled",
    max_retries: int = MAX_TRANSIENT_RETRIES,
    base_delay: float = RETRY_BASE_DELAY_SECONDS,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> tuple[dict, dict]:
    """发送对话和工具定义，遇到瞬时网络故障时显式指数退避重试，返回回复及其元数据。"""
    # 显式传入最高档，避免落到服务端默认的 high。
    extra_body = {"thinking": {"type": "disabled"}}
    if thinking_mode == "max":
        extra_body = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
        }
    elif thinking_mode != "disabled":
        raise ValueError("thinking_mode 只能为 disabled 或 max")

    if max_retries < 0 or base_delay < 0:
        raise ValueError("max_retries 和 base_delay 不能为负数")

    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                tools=TOOLS,
                model=MODEL_NAME,
                messages=messages,
                extra_body=extra_body,
            )
            break
        except Exception as error:
            if attempt >= max_retries or not _is_retryable_error(error):
                raise
            attempt += 1
            delay = base_delay * (2 ** (attempt - 1))
            if on_retry is not None:
                on_retry(attempt, error, delay)
            if delay > 0:
                (sleep or time.sleep)(delay)

    # 保留 reasoning_content，供后续工具调用回传。
    choice = response.choices[0]
    return choice.message.to_dict(), {
        "response_id": response.id,
        "model": response.model,
        "created": response.created,
        "finish_reason": choice.finish_reason,
        "thinking_mode": thinking_mode,
        "usage": response.usage.to_dict()
        if response.usage is not None
        else None,
    }
