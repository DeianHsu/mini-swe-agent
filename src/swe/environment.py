# 在指定的已有 Docker 容器和工作目录中执行 Bash 命令，并限制单条命令的运行时间。
# 将退出码、标准输出和标准错误整理成短文本，供模型读取。

from os import SEEK_END
import subprocess
from tempfile import TemporaryFile
from typing import BinaryIO

MAX_OUTPUT_CHARS = 16000


def limit_output(text: str) -> str:
    """输出过长时只保留首尾，并提示模型缩小读取范围。"""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    notice = "\n[输出已截断。请缩小命令范围继续读取。]\n"
    available = MAX_OUTPUT_CHARS - len(notice)
    head = available // 2
    return text[:head] + notice + text[-(available - head):]


def _read_output(stream: BinaryIO) -> str:
    # 每个 UTF-8 字符最多四字节。两端各读足够的字节供最终截断。
    sample_bytes = MAX_OUTPUT_CHARS * 4
    size = stream.seek(0, SEEK_END)
    stream.seek(0)
    data = stream.read(sample_bytes)
    if size > sample_bytes:
        stream.seek(max(sample_bytes, size - sample_bytes))
        data += stream.read(sample_bytes)
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


# 在指定容器和目录内执行一条命令，返回退出码、标准输出和标准错误。
def execute(command: str, *, container_name: str = "mini-swe-learning",
            workdir: str = "/workspace", timeout: int = 60) -> str:
    # 在容器内用 timeout 限制实际命令的运行时间，超时发送 KILL 信号。
    # Bash 使用非登录模式，沿用镜像原有的工具路径等环境配置。
    args = [
        "docker", "exec", "-w", workdir, container_name,
        "timeout", "--signal=KILL", "--verbose", str(timeout),
        "bash", "-c", command,
    ]

    # 用临时文件接收两路输出，再读取首尾，避免把整份日志读进内存。
    with TemporaryFile() as stdout, TemporaryFile() as stderr:
        try:
            result = subprocess.run(
                args=args, stdout=stdout, stderr=stderr, timeout=timeout + 15, check=False,
            )
        except subprocess.TimeoutExpired as error:
            # 宿主进程额外等待 15 秒。超时后由预算监管流程停止容器。
            raise RuntimeError("Docker 执行通道超时，本题应结束并停止容器") from error

        output = (
            f"Exit code:{result.returncode}\n"
            f"STDOUT:\n{_read_output(stdout)}\n"
            f"STDERR:\n{_read_output(stderr)}\n"
        )
    return limit_output(output)
