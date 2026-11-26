"""独立的单题真实判分校准：空补丁应失败，官方参考补丁应通过。"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib
import traceback
from uuid import uuid4

import verifier


def run_calibration(dataset: Path, instance_id: str, output: Path) -> dict:
    """对指定题目顺序运行两种补丁，独立保存结果，不生成全量评测产物。"""
    task_dir = dataset / "v2/tasks" / instance_id
    case_dir = output.resolve() / instance_id
    case_dir.mkdir(parents=True, exist_ok=False)
    result_path = case_dir / "calibration.json"
    log_path = case_dir / "runner.log"
    result = {
        "instance_id": instance_id,
        "dataset_repository": verifier.DATASET_REPOSITORY,
        "dataset_revision": verifier.DATASET_REVISION,
        "status": "running",
        "passed": False,
        "model_calls": 0,
        "total_tokens": 0,
        "model_cost_usd": "0",
        "checks": [],
        "cleanup_errors": [],
    }
    verifier.save_json(result_path, result)
    started = time.monotonic()
    prefix = "swebench-calibration-" + uuid4().hex[:16]
    probe_name = prefix + "-probe"
    probe_requested = False

    try:
        config = tomllib.loads(
            (task_dir / "task.toml").read_text(encoding="utf-8")
        )
        probe_requested = True
        image_id, workdir, base_commit = verifier.prepare_image(
            config, probe_name, log_path
        )
        result.update(image_id=image_id, base_commit=base_commit)

        # 两次判分使用同一镜像 ID，但容器、补丁和日志各自独立。
        for mode, expected in (("empty", False), ("gold", True)):
            check_dir = case_dir / mode
            check_dir.mkdir()
            check_log = check_dir / "runner.log"
            check_result = {
                "patch_mode": mode,
                "expected_resolved": expected,
                "resolved": None,
                "status": "running",
                "cleanup_errors": [],
            }
            result["checks"].append(check_result)
            verifier.save_json(result_path, result)
            check_started = time.monotonic()
            container = prefix + "-" + mode
            container_requested = False
            try:
                patch_path = check_dir / "model.patch"
                patch_path.write_bytes(
                    b""
                    if mode == "empty"
                    else (task_dir / "solution/gold_patch.diff").read_bytes()
                )
                container_requested = True
                check_result.update(
                    verifier.verify_patch(
                        task_dir,
                        config,
                        image_id,
                        workdir,
                        patch_path,
                        check_dir,
                        container,
                    )
                )
                check_result["status"] = (
                    "passed"
                    if check_result["resolved"] is expected
                    else "mismatch"
                )
            except Exception as error:
                # 判分异常不能充当空补丁的预期失败；仍继续校准另一种补丁。
                check_result.update(
                    status="error",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                with check_log.open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
            finally:
                # 每次判分结束即删除容器，避免两套任务同时占用内存。
                if container_requested:
                    try:
                        verifier.run_command(
                            ["docker", "rm", "-f", container], check_log
                        )
                    except (OSError, subprocess.SubprocessError) as error:
                        check_result["cleanup_errors"].append(str(error))
                check_result["seconds"] = round(
                    time.monotonic() - check_started, 3
                )
                verifier.save_json(
                    check_dir / "calibration_result.json", check_result
                )
                verifier.save_json(result_path, result)

        result["status"] = (
            "passed"
            if all(
                check["status"] == "passed" and not check["cleanup_errors"]
                for check in result["checks"]
            )
            else "failed"
        )
    except Exception as error:
        result.update(
            status="error", error_type=type(error).__name__, error=str(error)
        )
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
    finally:
        if probe_requested:
            try:
                # 探测容器通常已由 --rm 删除，也清理超时后可能残留的容器。
                verifier.run_command(
                    ["docker", "rm", "-f", probe_name], log_path, check=False
                )
            except (OSError, subprocess.SubprocessError) as error:
                result["cleanup_errors"].append(str(error))
                result["status"] = "error"
        result["passed"] = result["status"] == "passed"
        result["seconds"] = round(time.monotonic() - started, 3)
        verifier.save_json(result_path, result)

        # 校准报告使用独立文件名和目录，不会被全量报告计入 pass@1。
        verdict = {None: "未判分", False: "未通过", True: "通过"}
        rows = [
            f'| {check["patch_mode"]} | '
            f'{verdict[check["expected_resolved"]]} | '
            f'{verdict[check["resolved"]]} | '
            f'{check["status"]} | {check["seconds"]} |'
            for check in result["checks"]
            if "seconds" in check
        ]
        (case_dir / "calibration.md").write_text(
            "# SWE-bench Pro V2 判分校准\n\n"
            f'题目：`{instance_id}`\n\n'
            f'校准状态：{result["status"]}；'
            f'总耗时：{result["seconds"]} 秒。\n\n'
            "| 补丁 | 预期判分 | 实际判分 | 校准状态 | 耗时（秒） |\n|---|---|---|---|---|\n"
            + "\n".join(rows)
            + "\n\n模型调用、词元消耗和模型费用均为 0；校准结果不计入全量评测。\n"
              "错误详情、清理状态和镜像版本见 calibration.json；原始日志见 empty/ 和 gold/。\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    """只接受明确指定的一道题；真实容器判分必须在 Linux 主机执行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", action="version", version=verifier.DATASET_REVISION
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"instance_[A-Za-z0-9_.-]+", args.instance_id):
        parser.error("--instance-id 必须是官方单题目录名")
    if sys.platform != "linux":
        parser.error("真实判分校准需在运行 Linux 容器的 Linux 主机上执行")
    result = run_calibration(args.dataset, args.instance_id, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
