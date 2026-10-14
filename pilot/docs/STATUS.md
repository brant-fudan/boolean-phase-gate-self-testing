# 实测状态（2026-10-14）

## 已获证据

初次完整验收提交：`ad4922b92dac80fede643c974556f27fed322dd9`。

[完整通过的 CI](https://github.com/brant-fudan/boolean-phase-gate-self-testing/actions/runs/38506554566)

`evidence/ci_verified_ad4922b.json` 从该运行的实际日志提取，包含源文件 hash、工具版本、公理集合和测试统计。此后新增的非零定理、形式化接口与文档以最新 CI 的结果为准。

| 检查 | 实测结果 |
|---|---|
| Lean 工具链 | 4.19.0，官方 Linux 发行包 SHA256 校验通过 |
| CCZ 三个导数证明 | 编译通过 |
| 一般有限 n 的 Z/X 稳定子 | 编译通过 |
| 公理审计 | 无 sorryAx；只出现 propext、Quot.sound |
| 两个错误命题的 Lean 否定证明 | 通过，无公理依赖 |
| Python 测试 | 初次 CI 26 项；加入陈述门禁测试后本地 27 项通过 |
| 精确实例检查 | 804 组通过 |
| 原 Python 归一化矩阵比对 | 32 个函数通过 |
| Lean/Python CSV | 49,680 行逐项一致 |
| 正确任务自动修复回放 | 5 个任务，均 LEAN_REJECTED → VERIFIED |
| 错误任务回放 | 2 个任务，两轮均 LEAN_REJECTED |
| 真实外部 LLM 调用/成功率/费用 | 未运行；不能从 replay 推断 |
| 复数归一化、张量/矩阵等价桥、自检验 | 未形式化 |

## 环境与修复记录

当前本地容器下载官方 Lean 后，执行 `lean --version` 报 `failed to locate application`。没有把这种基础环境失败解释成证明失败。实际编译转到独立分支的 GitHub Actions Ubuntu 24.04。

真实 Lean 反馈依次暴露并修正：

1. `simp` 已化简目标后 `split` 无法继续：改为按索引相等与否分类讨论。
2. `Bits n` 函数类型缺少可计算相等判定：加入由 `∀ i : Fin n, a i = b i` 构造的可计算 `DecidableEq`。

这些是本次助手根据编译反馈实施的实际开发修复；另有程序内 replay 的自动循环测试。两者都不能替代独立外部模型对照实验。
