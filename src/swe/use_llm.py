# 通过 OpenAI 软件开发工具包（SDK）连接 DeepSeek，读取模型配置并声明容器命令执行工具。
# 根据传入模式关闭思考或启用最高档思考，发送完整对话，返回模型消息、停止原因及原始用量等元数据。
# 保留思考内容供后续工具调用回传，并关闭内部重试，使每次请求的轮数与消耗可独立记录。

import os

from dotenv import load_dotenv
from openai import OpenAI

# 要求显式提供 DeepSeek 配置，防止客户端回退到 OpenAI 的地址或密钥。
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
if not api_key:
    raise ValueError("缺少模型配置 DEEPSEEK_API_KEY")
if not base_url:
    raise ValueError("缺少模型配置 DEEPSEEK_BASE_URL")

# 复用同一个客户端，并设置请求超时。
# 关闭软件开发工具包（SDK）内部重试，使每轮请求可以明确记账。
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=120.0,
    max_retries=0,
)

# 向模型发送对话与工具定义，分别返回回复消息和包含用量的响应元数据。
def use_llm_with_tools(messages: list[dict], *, thinking_mode: str = "disabled") -> tuple[dict, dict]:
    # 当前价目只覆盖 deepseek-flash，防止换用其他模型后仍按旧价计费。
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    if model_name != "deepseek-flash":
        raise ValueError(
            "当前预算价目只适用于 deepseek-flash；更换模型前须更新价目"
        )

    # 默认关闭思考；最高档同时启用思考并显式指定 max，避免落到服务端默认的 high。
    # DeepSeek 扩展参数经 extra_body 原样传入，不依赖 SDK 对思考档位的类型枚举。
    extra_body = {"thinking": {"type": "disabled"}}
    if thinking_mode == "max":
        extra_body = {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}
    elif thinking_mode != "disabled":
        raise ValueError("thinking_mode 只能为 disabled 或 max")

    # 声明 execute 工具及其必需的 command 参数，让模型按约定提交 Bash 脚本。
    tools_schema = [
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
                            "description": "需要执行的 Bash 脚本"
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ]

    # 两种模式使用相同的模型、工具和客户端超时，只切换明确记录的思考参数。
    response = client.chat.completions.create(
        tools=tools_schema,
        model=model_name,
        messages=messages,
        extra_body=extra_body,
    )

    # 对话消息供后续轮次使用，响应标识、停止原因和用量供轨迹与成本统计使用。
    # 完整保留 reasoning_content，供后续工具调用回传；缺失的 usage 保留为 None。
    choice = response.choices[0]
    return choice.message.to_dict(), {
        "response_id": response.id,
        "model": response.model,
        "created": response.created,
        "finish_reason": choice.finish_reason,
        "thinking_mode": thinking_mode,
        "usage": response.usage.to_dict() if response.usage is not None else None,
    }
