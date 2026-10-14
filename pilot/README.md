# 本科生 AI Agent + Lean 4 科研试点

本项目从 CCZ 布尔导数和 Boolean-phase Choi 稳定子出发，训练可迁移的形式化、Agent 工具调用、错误诊断、反例生成及实验评测能力。

**已实际完成 Lean 编译和跨语言验证。当前模型接口可用代码已提供，但没有外部 LLM API 实验数据。默认 replay 是集成测试，不是 AI 成功率。**

实测结果：27 项测试通过；804 组精确检查、32 个矩阵函数、49,680 行跨语言比对通过。详见 [验收记录](docs/STATUS.md) 和 [实际 CI](https://github.com/brant-fudan/boolean-phase-gate-self-testing/actions/runs/38507168563)。

## 1. 快速开始

```bash
git clone --branch pilot/agent-lean-undergraduate https://github.com/brant-fudan/boolean-phase-gate-self-testing.git
cd boolean-phase-gate-self-testing/pilot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# 安装 elan 后，lean-toolchain 会选择 Lean 4.19.0。
# https://github.com/leanprover/elan#installation
lake build
python scripts/verify.py
```

Python 推荐 3.11；Lean 固定 4.19.0，无 Mathlib 或其他 Lean 第三方依赖。`lake-manifest.json` 由 Lake 自动生成，无外部包；不要求学生安装论文仓库的全部作图依赖。Windows 建议 WSL；macOS 可直接使用 elan。

`verify.py` 失败时以非零退出，不将缺少 Lean、超时或导入失败算作“命题被否定”。GitHub Actions 使用固定 Lean 发行包及 SHA256，并保存日志、逐次证明源码、CSV、公理依赖和验收 JSON。无需 API key 即可复现全部验收。

## 2. 已证明什么

| 结论 | Lean 名称 | 量化范围 |
|---|---|---|
| CCZ 对三个变量的导数等于其余两变量之积 | `ccz_derivative0/1/2` | 任意三位布尔输入 |
| cofactor XOR 与翻转定义一致 | `derivative_flip_form` | 任意 `n,f,x,i` |
| 导数不随被求导位翻转改变 | `derivative_flip` | 任意 `n,f,x,i` |
| Z 型稳定子固定 Choi 振幅 | `choi_stabilizerZ` | 任意有限 `n`、任意布尔函数、任意有效位 |
| X 型广义稳定子固定 Choi 振幅 | `choi_stabilizerX` | 同上 |
| Choi 振幅非零 | `choi_nonzero` | 任意 `n,f` |
| “遗漏一个乘积因子”“导数恒零”为假 | `ccz_wrong_omits_factor/zero_derivative` | 有显式反例的否定命题 |

Lean 中 `Bits n := Fin n → Bool`，振幅是 `Int`，态未归一化。这里的 `n` 是比特数；对应空间有 `2^(2n)` 个联合基矢。任意有限 `n` 的理想模型结论**不等于**任意物理维度、未知测量装置的自检验结论。

归一化复 Hilbert 空间、张量积矩阵与振幅算子的形式化同构、Bell 算子谱、鲁棒性、自检验和有限统计均不在已证明范围。Python 的归一化矩阵比对是独立数值验证，不能填补这些形式化缺口。

## 3. 文件入口

| 文件 | 用途 |
|---|---|
| `Pilot/Definitions.lean` | 数据、布尔导数、Choi 振幅与算子定义；不含目标答案 |
| `Pilot/Basic.lean` | CCZ、翻转、符号消去等证明 |
| `Pilot/Stabilizers.lean` | 通用稳定子与非零证明 |
| `Audit.lean` | 打印实际逻辑公理依赖 |
| `ExportRows.lean` | 导出可复核的 Lean 运算结果 |
| `scripts/formalize.py` | 自然语言任务卡、自动形式化接口、陈述匹配门禁 |
| `scripts/agent_loop.py` | 固定目标、模型提案、编译、反馈、有限次数自动修复 |
| `scripts/llm_provider.py` | 可选的 chat-completions-compatible HTTP 适配器 |
| `scripts/cross_validate.py` | 独立前向稀疏算子实现及原 Python 矩阵复核 |
| `scripts/verify.py` | 一键验收，任何环节失败即失败 |
| `benchmarks/cases.json` | 16 个正确、错误、语义陷阱和越界声明案例 |
| `docs/TECHNICAL.md` | 数学对应、信任边界、Agent 协议 |
| `docs/STUDENT_PLAN.md` | 分级任务、六周安排、量化验收 |
| `docs/RESEARCH.md` | 独立论文方向、基线、数据划分和停止条件 |
| `evidence/` | 版本化的实测摘要；完整运行轨迹另见 CI artifacts |

## 4. Agent 使用

先看自然语言任务与形式化约定：

```bash
python scripts/formalize.py --task ccz_d0
```

外部模型返回 `import`、`binders`、`statement` 三个字段；保存为 proposal.json 后：

```bash
python scripts/formalize.py --task ccz_d0 --proposal proposal.json
```

完全匹配参考陈述才给出 `REFERENCE_MATCH`。等价但不同写法返回 `REVIEW_REQUIRED`，不能由这一结果断言其错误。新任务须由教师或另一位同学逐项核对自然语言、量词、假设、定义、结论，随后登记固定目标。证明阶段不能修改它。

无 API 的修复流程演示：

```bash
python scripts/agent_loop.py --task ccz_d0 --max-attempts 2
```

默认 replay 第一次故意使用不存在的引理，第二次给出预设证明；每次均真实调用 Lean 并记录反馈。这只检验控制流、目标固定、编译拒绝与修复验收。

使用真实外部模型时，在本地设置 `LLM_ENDPOINT`（完整 HTTPS endpoint）、`LLM_MODEL`、`LLM_API_KEY`。不要把 key 写入仓库或 CI。然后执行：

```bash
python scripts/formalize.py --task ccz_d0 \
  --provider-cmd-json '["python", "scripts/llm_provider.py"]'
python scripts/agent_loop.py --task ccz_d0 --max-attempts 3 \
  --provider-cmd-json '["python", "scripts/llm_provider.py"]' \
  --output evidence/live_ccz_d0.json
```

这个适配器要求服务提供兼容的 JSON 协议；不同服务的接口字段需要先做联通测试。也可用任意程序代替适配器：stdin 读取请求 JSON，stdout 只输出提案 JSON。运行一个证明任务最多三次模型请求，每次最多 2048 输出 token；不默认启用计费调用。可用 `LLM_USAGE_LOG` 单独记录模型名、response id 和 token usage，价格由实际服务计费表另行计算。

这是受限证明体 Agent。只允许已审查的 Lean 标识符和标点，不能自由写入项目或执行模型生成的 shell。扩大证明语言需同步扩展策略测试；该词汇门禁不应当被宣传为通用不可信代码沙箱。

## 5. 学生第一次交付

运行 `python scripts/verify.py`，解释一个真实 Lean 错误，手动证明一个导数等式，构造一个错误命题的反例，并指出“Lean 检查通过”和“原自然语言命题正确形式化”的区别。具体评分见学生计划。
