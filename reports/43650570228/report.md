# SWE-bench Pro V2 全量评测报告：关闭思考模式

## 一、评测结果

| 项目 | 结果 |
|---|---|
| 模型 | `deepseek-flash` |
| 已完成判分 / 总题数 | 642/642 |
| 通过 / 未通过 | 561 / 81 |
| 首次尝试通过率（pass@1） | 87.38% |
| 尝试次数 | 每题 1 次 |
| 并发 | 最多 8 题，3 个批次各 214 题 |
| 代码版本 | `19e8dae03edec0574ec84f98459b5b20811966a3` |
| 数据集版本 | `scaleapi/SWE-bench_Pro-os`，`66f92766bba642462d4bbe5479e83f91f9211862` |

## 二、预算与来源

| 限制项 | 每题预算 | 来源 |
|---|---|---|
| 模型调用轮数 | 250 轮 | [mini-SWE-agent 通用配置](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/config/benchmarks/swebench.yaml)的 `agent.step_limit` |
| 单条命令时限 | 60 秒 | 同一配置的 `environment.timeout` |
| 模型费用阈值 | 3 美元 | 同一配置的 `agent.cost_limit` |
| 解题时限 | 50 分钟 | [SWE-bench Pro V2 锁定评测协议第 3 条](https://github.com/scaleapi/SWE-bench_Pro-os/blob/66f92766bba642462d4bbe5479e83f91f9211862/v2/README.md#the-locked-protocol) |

每轮模型请求前检查累计估算费用，达到费用阈值后结束解题。

## 三、用时与费用

| 统计项 | 完整记录题数 / 总题数 | 累计值 | 平均值 |
|---|---|---|---|
| 解题用时 | 642/642 | 约 42 小时 10 分 11 秒 | 约 3 分 56 秒/题 |
| 单题评测全流程用时 | 642/642 | 约 65 小时 10 分 45 秒 | 约 6 分 5 秒/题 |
| 完整费用样本 | 639/642 | 12.612 美元 | 约 0.02 美元/题 |

整轮实际经过 11 小时 25 分 29 秒，起止时间为 2026-11-15 06:55:11 至 18:20:40（北京时间）。

解题用时涵盖进程启动、模型请求和工具执行；评测全流程另含镜像准备、补丁导出、独立判分和资源清理。

全题已记录估算费用为 12.639 美元。完整总费用和全题平均费用因请求用量缺失而标记为未知。

本轮固定单价为：缓存命中输入 0.006、缓存未命中输入 0.3、输出 1.2，单位为美元/百万词元（token）。

## 四、词元与轮数

| 项目 | 数量 |
|---|---:|
| 已记录输入词元 | 541,996,230 |
| 其中：缓存命中 | 527,595,894 |
| 其中：缓存未命中 | 14,400,336 |
| 已记录输出词元 | 4,294,844 |
| 已记录词元合计 | 546,291,074 |
| 模型请求数（轮数） | 20,553，平均 32.014 次/题 |
| 已获取完整用量的请求数 | 20,550 |

## 五、异常记录

正常结束 629 题，达到轮数上限 8 题、解题时限 2 题，请求异常 3 题。费用阈值触发 0 题。

题号 259、580、588（从 0 开始）发生连接被对端重置（`Connection reset by peer`），各缺少 1 次请求的用量，判分依次为未通过、未通过、通过。

[报告任务](https://github.com/DeianHsu/mini-swe-agent/actions/runs/43650570228/job/132779752980)因用量记录缺失返回退出码 1，GitHub 显示红叉；报告产物已上传。

## 六、结论

本轮评测验证了智能体方案的可行性和当前配置下的解题能力，结果作为项目基线。

最高思考模式全量评测暂缓。后续维护以相关回归测试为主，思考模式对比先通过固定小样本评估收益和成本。

## 七、数据来源

- [GitHub 运行 43650570228](https://github.com/DeianHsu/mini-swe-agent/actions/runs/43650570228)：工作流时间记录；报告产物 `swebench-report`，单题产物 `swebench-case-题号`。
- [report.csv](https://github.com/DeianHsu/mini-swe-agent/blob/master/reports/43650570228/report.csv)：逐题用时、词元、费用、轮数和判分。
- [report.json](https://github.com/DeianHsu/mini-swe-agent/blob/master/reports/43650570228/report.json)：完整统计及异常状态。
- [manifest.json](https://github.com/DeianHsu/mini-swe-agent/blob/master/reports/43650570228/manifest.json)：题目清单、预算和版本。
