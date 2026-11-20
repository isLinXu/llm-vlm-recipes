# Agent 交接契约（人类可读视图）

> 由 `recipe-lab handoff --report` 生成；单一事实源是 `registry/agent_contract.toml`。机器可读版本见 [AGENT-HANDOFF.json](AGENT-HANDOFF.json)。请勿手工编辑本文件。

## 这份文件解决什么

仓库同时服务两类读者：人需要快速理解现状与边界；Agent 需要确定性命令、事实源和可校验的重建结果。本页是人的入口，JSON 是机器的入口，两者都由同一份契约和真实审计器生成。

这不是第六层审计，也不认证 GPU 执行。静态 JSON、hash、测试和 dry-run 只证明内部一致性；GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14 仍独立判定。

## 当前事实

- 配方：**14 篇**；五维已实现 **9 篇**；GPU verified **0**；eligible **0**。
- 方法论：**20/23** 篇论文一手核实。
- 后训练方向：**5/5** 有五维实现支撑，缺口 **0** 个。
- 原理性阻塞：`vlm/sft/qwen2-vl-7b-chartqa`, `vlm/sft/smolvlm-chartqa`。

## 标准工作流

1. 先读 `AGENTS.md`，再看本页与 `AGENT-HANDOFF.json`。
2. 只修改事实源：`registry/`、`recipes/`、`src/`、`tests/`；不要手改 `docs/` 下生成报告。
3. 运行 `make report` 重建五份核心审计生成物与两份 Agent 交接视图，再运行 `recipe-lab handoff --check` 验证可重建性。
4. 运行 `make ci`；提升目标 `gate-*` 必须继续失败关闭。
5. 没有用户明确要求，不要 push 或对外发布。

## 事实源

| path | purpose |
| --- | --- |
| `recipes/**/recipe.toml` | 14 篇配方的单一事实源；训练/评估协议与内容身份只取任务专用段 |
| `registry/taxonomy.toml` | family / trainer / 兼容矩阵与仓库根标记 |
| `registry/reproducibility.toml` | 上游可复现性 L0–L4 判定 |
| `registry/baseline.toml` | B0–B3 就绪规则与退出条件 |
| `registry/capabilities.toml` | 五维能力 implemented / missing 登记；verified 不由此推出 |
| `registry/feasibility.toml` | 实验目标是否成立；两类原理性阻塞不可被证据绕过 |
| `registry/methodology.toml` | 设计决策的一手论文依据与边界；论文分数不是本仓 expected |
| `registry/mllm_post_training.toml` | MLLM 后训练五方向、来源边界与 14 篇映射 |
| `src/recipe_lab/*.py` | 审计器、CLI 与任务适配代码；生成报告必须由这些代码重建 |
| `tests/test_*.py` | 离线正反向测试；通过不等于 GPU 验证 |

## 生成物与当前哈希

| path | 生成器 | sha256 | purpose |
| --- | --- | --- | --- |
| `docs/MATRIX.md` | `matrix --out {out}` | `d6d3f0b5bc99` | 全量 14 篇索引与五维矩阵 |
| `docs/REPRODUCIBILITY.md` | `repro --report {out}` | `36df533d1b19` | 上游可复现性历史审计 |
| `docs/BASELINE-READINESS.md` | `baseline --report {out}` | `73aee2dab143` | B0–B3 就绪、能力矩阵与缺口 |
| `docs/METHODOLOGY.md` | `methodology --report {out}` | `b3b5a3f4f86f` | 方法论依据、一手论文核查与编号更正 |
| `docs/MLLM-POST-TRAINING.md` | `post-training --report {out}` | `32b433c0dad5` | MLLM 后训练五方向与逐篇映射 |

## 命令契约

| id | expectation | argv | purpose |
| --- | --- | --- | --- |
| `doctor` | pass | `doctor` | 仓库根、tomllib、taxonomy 与 recipe.toml 可解析 |
| `validate-l6` | pass | `validate --level l6` | L0–L6 契约校验；warning 是登记的待办风险，不阻塞 CI |
| `ci` | pass | `make ci` | doctor + L0–L6 + 单测 + 五层审计 + dry-run 冒烟 |
| `report` | pass | `make report` | 重建五份核心审计生成物与两份 Agent 交接视图；随后必须检查 git diff --exit-code |
| `handoff-check` | pass | `handoff --check` | 校验交接契约、生成物可重建性与门禁登记未漂移 |
| `gate-b2` | fail_closed | `baseline --check --min-state B2` | 当前不得宣称任何篇达到 research-ready |
| `gate-b3` | fail_closed | `baseline --check --min-state B3` | 当前不得宣称任何篇达到 business-ready |
| `gate-l2` | fail_closed | `repro --check --min-level L2` | 当前不得宣称上游实验完全锁定 |
| `gate-feasibility-b2` | fail_closed | `feasibility --check --min-state B2` | 当前不得把任何篇判为可直接晋级 |
| `gate-methodology` | fail_closed | `methodology --check --min-verified-papers 99` | 一手核实论文数量提升目标；当前 20/23 |
| `gate-post-training` | fail_closed | `post-training --check --min-fivedim-directions 99` | 五维方向覆盖提升目标；当前 5/5 已有五维实现支撑，下一档仍失败关闭 |

## Agent Guardrails

- **source-of-truth**：先改 registry / src / recipes / tests，再运行 make report；不要手改 docs 下的生成报告。
- **no-static-verification**：静态 JSON、hash、测试通过和 dry-run 只证明内部一致性，不认证 GPU 执行，也不解锁 B2/B3。
- **no-blocked-override**：blocked_zero_train 与 contaminated_pretraining 是原理性阻塞；不得靠补证据或论文背书绕过。
- **no-hidden-promotion**：提升目标必须失败关闭；不得把 --min-* 目标改成当前值来制造绿灯。
- **no-unasked-push**：不要自行创建远端、push 或对外发布；只有用户明确要求时执行。
- **source-boundary**：MLLM 五方向只能引用 preprints 202607.1494 摘要级框架；arXiv 2503.06072 与 Awesome-MLLM-Tuning 只作交叉印证 / 方法索引。

## 校验状态

- 交接契约：通过；生成物可重建性：未校验（仅静态检查）。