"""合并运行结果与独立评测的判分；只读取 JSON，不导入智能体代码。"""

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path
import sys


def write_report(run_dir: Path, verdicts: dict[str, bool | None] | None = None,
                 *, manifest: dict | None = None) -> dict:
    instance_ids = (
        manifest["instances"] if manifest is not None
        else [path.parent.name for path in sorted(run_dir.glob("*/case.json"))]
    )
    if not instance_ids:
        raise ValueError("运行目录中没有题目记录")

    # 新运行记录模式；既有清单未记录时明确显示未知，不替历史结果推定模式。
    thinking_mode = manifest.get("thinking_mode") if manifest is not None else None
    cases = []
    for instance_id in instance_ids:
        result_path = run_dir / instance_id / "agent_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if manifest is not None and result["run_id"] != manifest["run_id"]:
                raise ValueError(f"{instance_id}: 结果不属于本轮评测")
        else:
            # 机器故障、取消或产物上传失败也必须留在本轮分母中。
            result = {
                "status": "missing_result", "resolved": None,
                "seconds": None, "model_calls": None,
                "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                "known_cost_usd": "0", "budget_cost_usd": None, "usage_complete": False,
                "thinking_mode": thinking_mode,
            }
        # 缺少判分表示尚未得到结果，不能当作答错。
        resolved = verdicts.get(instance_id) if verdicts is not None else result["resolved"]
        if resolved is not None and type(resolved) is not bool:
            raise ValueError(f"{instance_id}: 判分必须是 true、false 或 null")

        result["instance_id"] = instance_id
        result["resolved"] = resolved
        cases.append(result)

    known_total_cost_usd = sum(
        (Decimal(case["known_cost_usd"]) for case in cases),
        Decimal("0"),
    )
    usage_complete = all(case["usage_complete"] for case in cases)
    passed = sum(case["resolved"] is True for case in cases)
    failed = sum(case["resolved"] is False for case in cases)
    ungraded = len(cases) - passed - failed
    report = {
        "manifest": manifest,
        "thinking_mode": thinking_mode,
        "cases": cases,
        "task_count": len(cases),
        "passed": passed,
        "failed": failed,
        "ungraded": ungraded,
        "evaluation_complete": ungraded == 0,
        "pass_at_1": passed / len(cases) if ungraded == 0 else None,
        "total_cost_usd": (
            str(known_total_cost_usd) if usage_complete else None
        ),
        "known_total_cost_usd": str(known_total_cost_usd),
        "known_total_tokens": sum(case["total_tokens"] or 0 for case in cases),
        "usage_complete": usage_complete,
    }

    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = ["instance_id", "thinking_mode", "seconds", "prompt_tokens", "completion_tokens", "total_tokens",
              "budget_cost_usd", "known_cost_usd", "model_calls", "resolved", "usage_complete",
              "status", "exit_reason", "stage", "error"]
    with (run_dir / "report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)
    rate = f'{report["pass_at_1"]:.2%}' if report["evaluation_complete"] else "未完成，暂不发布最终通过率"
    cost = (
        f'${known_total_cost_usd}'
        if usage_complete
        else f'未知；已记录 ${known_total_cost_usd}'
    )
    thinking_label = {None: "未记录", "disabled": "关闭", "max": "最高（max）"}[thinking_mode]
    (run_dir / "report.md").write_text(
        "# SWE-bench Pro V2 评测报告\n\n"
        f"| 项目 | 结果 |\n|---|---|\n| 计划题数 | {len(cases)} |\n"
        f"| 思考模式 | {thinking_label} |\n"
        f"| 通过 / 未通过 / 未判分 | {passed} / {failed} / {ungraded} |\n"
        f"| pass@1 | {rate} |\n| 总估算成本（美元） | {cost} |\n"
        f'| 已记录词元数 | {report["known_total_tokens"]} |\n\n'
        "每题耗时、词元、估算成本、轮数和判分见 report.csv；原始数据见 report.json。\n"
        "成本按主线记录的固定价目估算；未返回用量的请求不计作零成本。\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("verdicts", type=Path, nargs="?")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    verdicts = json.loads(args.verdicts.read_text(encoding="utf-8")) if args.verdicts else None
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    report = write_report(args.run_dir, verdicts, manifest=manifest)
    print(json.dumps({key: value for key, value in report.items() if key not in ("cases", "manifest")},
                     ensure_ascii=False, indent=2))
    if manifest is not None and (not report["evaluation_complete"] or not report["usage_complete"]):
        sys.exit(1)
