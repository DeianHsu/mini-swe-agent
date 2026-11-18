# 通过 OpenAI 软件开发工具包（SDK）连接 DeepSeek，读取模型配置并声明容器命令执行工具。
# 根据传入模式关闭思考或启用最高档思考，发送完整对话，返回模型消息、停止原因及原始用量等元数据。
# 保留思考内容供后续工具调用回传，并关闭内部重试，使每次请求的轮数与消耗可独立记录。

import os

from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "deepseek-flash"
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
        raise ValueError(f"当前预算价目只适用于 {MODEL_NAME}。更换模型前须更新价目")

    # 关闭软件开发工具包（SDK）内部重试，使每轮请求可以明确记账。
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=0)


def send_messages(client: OpenAI, messages: list[dict], *,
                  thinking_mode: str = "disabled") -> tuple[dict, dict]:
    """发送对话和工具定义，返回回复及其元数据。"""
    # 显式传入最高档，避免落到服务端默认的 high。
    extra_body = {"thinking": {"type": "disabled"}}
    if thinking_mode == "max":
        extra_body = {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}
    elif thinking_mode != "disabled":
        raise ValueError("thinking_mode 只能为 disabled 或 max")

    response = client.chat.completions.create(
        tools=TOOLS,
        model=MODEL_NAME,
        messages=messages,
        extra_body=extra_body,
    )

    # 保留 reasoning_content，供后续工具调用回传。
    choice = response.choices[0]
    return choice.message.to_dict(), {
        "response_id": response.id,
        "model": response.model,
        "created": response.created,
        "finish_reason": choice.finish_reason,
        "thinking_mode": thinking_mode,
        "usage": response.usage.to_dict() if response.usage is not None else None,
    }
