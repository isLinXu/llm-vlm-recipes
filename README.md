# recipe-lab

面向 LLM / VLM 的研究与业务训练基线仓库。当前交付是**更严格的维护框架 + 八篇五维参考适配**，不是「14 篇已完成 baseline」。

[输入调研](docs/llm-vlm-training-cookbook-research.html) · [来源与修正](docs/SOURCE-MAP.md) · [设计决策](docs/DESIGN.md) · [验证规则](docs/VERIFICATION.md) · [Agent 交接](AGENTS.md)

> 最近一次本地验证（2026-11-17）：`make ci` 0 error / 1160 passed；五份核心审计生成物与两份交接视图哈希稳定。
> 静态检查只证明仓库内部一致性，**不认证 GPU 执行**。

## 阅读路径

| 目标 | 入口 |
| --- | --- |
| 5 分钟判断仓库能做什么、不能做什么 | [当前状态](#当前状态) |
| 本地复现验证门禁 | [快速验证](#快速验证无-gpu) |
| 作为 Agent 接手并避免误改生成物 | [Agent 可复现入口](#agent-可复现入口) |
| 看每篇五维缺什么、波次怎么排 | [BASELINE-READINESS.md](docs/BASELINE-READINESS.md) |
| 看为什么这么设计、依据哪篇论文 | [METHODOLOGY.md](docs/METHODOLOGY.md) |
| 看后训练方向覆盖与缺口 | [MLLM-POST-TRAINING.md](docs/MLLM-POST-TRAINING.md) |

## 当前状态

| 维度 | 当前事实 | 判定 |
| --- | --- | --- |
| 配方登记 | 14 篇，逐篇一手材料核查 | 登记不等于逐字节复现上游 |
| 五维代码 | 8 篇具备 `data/train/eval/comparison/export`；70 格中 40 implemented / 30 missing | `implemented` ≠ `verified` |
| CUDA GPU 实跑 | 0 篇；未下载真实模型/数据，未确认框架兼容、显存或训练效果 | 不能宣称任何篇为已验证 baseline |
| 就绪度 | B2+ 0、B3 0；eligible 0/14 | 8 pending_freeze、4 not_implemented、2 原理性阻塞 |
| 上游可复现性 | L0 8 / L1 6 / L2+ 0 | 没有一篇确认完全锁定 |
| 方法论依据 | 14 篇逐篇绑定；20/23 篇论文一手核实，10 条编号更正 | 论文不背书 GPU 晋级 |
| MLLM 后训练方向 | 五方向映射 14 篇；5/5 有五维实现支撑 | 方向覆盖不等于 DPO / MPO 已有 GPU baseline |
| Agent 交接契约 | `AGENTS.md` + `AGENT-HANDOFF.md/json`；契约登记的命令可复现 | 不新增认证能力，不解锁 B2/B3 |
| CI | `make ci` 0 error / 1160 passed（最近一次） | 只证明内部一致性与离线测试通过 |

> **两条铁律**：B1 只是命令、入口、锁模板、种子与划分意图的声明；`implemented` 只表示任务代码和离线测试已交付。
> `verified` 必须另有冻结环境下的真实运行证据；静态 JSON、hash、测试和 dry-run 都不能替代它。

代码实装、GPU 证据和**实验目标是否成立**是三件事。逐篇可行性判定见 `registry/feasibility.toml`：

```bash
make feasibility-check
PYTHONPATH=src python3 -m recipe_lab feasibility --min-state B2  # 提升目标：当前应失败
```

其中 `vlm/sft/qwen2-vl-7b-chartqa` 的 `decision.gate=blocked`（上游自承零训练基线即可达标），
`vlm/sft/smolvlm-chartqa` 的基座本就在 ChartQA 上训练过。两篇即使补齐五维代码与 CUDA 证据也不晋级，
必须先改写实验目标 / 换未污染数据，并重新核一手材料。登记内自设 override 一律拒绝。

## 事实源与生成物

| 类别 | 文件 | 维护规则 |
| --- | --- | --- |
| 事实源 | `recipes/**/recipe.toml`、`registry/*.toml`、`src/recipe_lab/*.py`、`tests/test_*.py` | 只改这里；所有结论由这些文件重建 |
| 静态入口 | `AGENTS.md` | Agent 可发现入口；可发现标记由契约强校验 |
| 核心审计生成物（5） | `docs/MATRIX.md`、`REPRODUCIBILITY.md`、`BASELINE-READINESS.md`、`METHODOLOGY.md`、`MLLM-POST-TRAINING.md` | `make report` 重建；`handoff --check` 逐字节重建校验 |
| 交接视图（2） | `docs/AGENT-HANDOFF.md`、`docs/AGENT-HANDOFF.json` | `handoff --report` 同步生成；不要手改 |

`handoff --check` 校验的是 `registry/agent_contract.toml` 中登记的五份核心生成物；
两份 Agent 交接视图由同一命令族生成，不进入该重建清单。任一侧漂移、事实源缺失、
命令枚举错误或 Guardrails 漂移都会失败关闭。

## 快速验证（无 GPU）

需要 Python ≥3.11；核心审计模块只用标准库，pytest 仅用于测试。在仓库根运行：

```bash
make ci                      # doctor + L0-L6 + 单测 + 全部审计 --check + dry-run 冒烟
make report                  # 重建五份核心审计生成物 + 两份 Agent 交接视图
make agent-check             # 交接契约与生成物可重建性
git diff --exit-code -- docs/MATRIX.md docs/REPRODUCIBILITY.md docs/BASELINE-READINESS.md \
  docs/METHODOLOGY.md docs/MLLM-POST-TRAINING.md docs/AGENT-HANDOFF.md docs/AGENT-HANDOFF.json
```

单项审计：

```bash
make capabilities-check      # 14 篇能力登记、文件证据与合法状态
make feasibility-check       # 实验目标是否成立，原理性阻塞不可被证据绕过
make methodology-check       # 论文一手字段、引用闭合、配方回填一致
make post-training-check     # 五方向映射、来源边界与能力/可行性交叉
make baseline-dry-run        # 输出 14 篇声明；不运行训练，不产生实跑证据
PYTHONPATH=src python3 -m recipe_lab capabilities --json
```

发布 / 提升目标是 **fail_closed**，当前必须失败：

```bash
PYTHONPATH=src python3 -m recipe_lab baseline --check --min-state B2
PYTHONPATH=src python3 -m recipe_lab baseline --check --min-state B3
PYTHONPATH=src python3 -m recipe_lab repro --check --min-level L2
PYTHONPATH=src python3 -m recipe_lab feasibility --check --min-state B2
PYTHONPATH=src python3 -m recipe_lab methodology --check --min-verified-papers 99
PYTHONPATH=src python3 -m recipe_lab post-training --check --min-fivedim-directions 5
```

## Agent 可复现入口

人类读 [AGENTS.md](AGENTS.md) 与 [AGENT-HANDOFF.md](docs/AGENT-HANDOFF.md)；
Agent 读 [AGENT-HANDOFF.json](docs/AGENT-HANDOFF.json)。契约版本由 `registry/agent_contract.toml` 决定，
不要凭记忆改计数或手改生成报告。

```bash
PYTHONPATH=src python3 -m recipe_lab handoff --check   # 校验契约与重建能力
PYTHONPATH=src python3 -m recipe_lab handoff --report  # 生成两份交接视图
```

硬 Guardrails：

- 只修改事实源，再运行 `make report`；不要手改 `docs/` 下生成物。
- 静态 JSON、hash、离线测试与 dry-run 不认证 GPU，也不解锁 B2/B3。
- `blocked_zero_train` 与 `contaminated_pretraining` 是原理性阻塞，不能靠证据或论文背书绕过。
- 提升目标必须失败关闭；不得把 `--min-*` 改成当前值制造绿灯。
- 不自行创建远端、push 或对外发布；只有用户明确要求时执行。
- MLLM 五方向只以 preprints 202607.1494 摘要级框架为原始出处；arXiv 2503.06072 与 `Awesome-MLLM-Tuning` 只作交叉印证 / 方法索引。

## 审计轴与成熟度

| 轴 | 回答的问题 | 权威来源 | 入口 |
| --- | --- | --- | --- |
| 验证层级 L0–L6 | 执行哪些结构 / 契约 / 证据规则？ | `rules.py` | `make validate-l6` |
| 可复现性 L0–L4 | 能否重放上游实验？ | `reproducibility.toml` | `make repro-check` |
| B0–B3 | 能否成为自建研究 / 业务基线？ | `baseline.toml` | `make baseline-check` |
| 五维能力 | 哪些任务逻辑已实现、已实跑？ | `capabilities.toml` | `make capabilities-check` |
| 可行性判定 | 实验目标是否成立、数据是否污染？ | `feasibility.toml` | `make feasibility-check` |
| 方法论依据 | 这么设计依据哪篇论文的一手结论？ | `methodology.toml` | `make methodology-check` |
| MLLM 后训练方向 | 训练工作覆盖哪个方向、缺口在哪？ | `mllm_post_training.toml` | `make post-training-check` |
| Agent 交接契约 | 人与 Agent 如何从同一事实源复现？ | `agent_contract.toml` | `make agent-check` |

`status` 的 cataloged → ported → smoke-passed → benchmarked 只记录配方维护进展，不替代上述审计。
复现上游实验与自建 baseline 可以不同，但后者必须公开声明差异并冻结、实测。

## 八篇已实装参考适配

八篇均按 `task_router` 精确分派 `task` / `suite` / `export`，未知或冲突任务失败关闭，互不借用协议。
它们都只有离线合成证据；无 CUDA 实测、无冻结 revision；真实配方未预声明 `export_tolerance` 时 `--full` 导出失败关闭。
DPO 与 MPO 共享偏好 split 算法，但基座不同，`cross_backbone_attribution_valid=false`。

| 任务 | recipe_id / task_id | 自建协议一句话 | 发布级对照 |
| --- | --- | --- | --- |
| 数学 GRPO | `llm/grpo/qwen2-0-5b-numinamath` / `math-grpo-v1` | 500/500/500 固定划分，gold 解析失败即停 | 3 seed + Student-t95 + 完整模型 offline 复载 |
| 代码 FIM | `llm/sft/starcoder1b-fim` / `code-fim-v1` | repo-hash 80/10/10 + 文件内 padding + PSM 评估 | 3 seed + adapter-only offline 复载 |
| xLAM FC | `llm/sft/llama3-8b-xlam-function-calling` / `xlam-fc-v1` | completion-only JSON calls，给全 tools，绝不执行调用 | 3 seed 四指标 + adapter-only offline 复载 |
| RapidFire | `llm/sft/rapidfire-multiconfig` / `rapidfire-sft-v1` | 4 配置顺序执行，规范化问题 hash 80/10/10 | 4 配置 × 3 seed，validation 选配置 / test 验收一次 |
| Granite Vision | `vlm/sft/granite-vision-line-comparison` / `granite-linecomparison-v1` | 图像字节 hash 隔离 + 显式 `image_token_id` + 双层 mask | 3 seed 逐题重算 + DoRA/4bit adapter-only |
| PaliGemma2 | `vlm/sft/paligemma2-refcoco-grounding` / `paligemma-grounding-v1` | COCO `xywh`→`xyxy` + 1024 桶 `<loc####>` round-trip | 3 seed 四指标 + 普通 LoRA adapter-only |
| SmolVLM DPO | `vlm/dpo/smolvlm-rlaif-v` / `smolvlm-dpo-v1` | 图像 hash 隔离 + 显式 beta sigmoid + 冻结 ref | 3 seed 隐式奖励 pairwise + adapter-only（不打包 ref） |
| Qwen2.5-VL MPO | `vlm/mpo/qwen2-5-vl-3b-mpo` / `qwen25vl-mpo-v1` | 共享偏好 split + sigmoid/bco_pair/sft 三损失消融 | 3 消融 × 3 seed；不得与 DPO 篇跨 backbone 归因 |

<details>
<summary>数学 GRPO：固定划分、格式奖励、三 seed 套件</summary>

- Qwen2-0.5B-Instruct + NuminaMath-TIR + TRL 0.14.0；train `[0:500]` / validation `[500:1000]` / test `[1000:1500]`。
- 统一 `<think>...</think><answer>...</answer>`；空输出、格式错误、gold 解析失败或跨 split 重复直接停止，不给满分。
- eval 只报告单 checkpoint；`--runs > 1` 拒绝，重复 greedy 解码不等于多训练 seed 方差。
- 套件执行一次共享基座评估 + 三个独立训练/评估子进程，输出配对增益均值、方差、SEM 与 Student-t 95% 区间。

```bash
make suite-dry-run
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run --json
python3 runner/eval.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run
python3 runner/zero_shot.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run
```

有 CUDA 且输入冻结后才可 `--full`；本机 macOS ARM 无 torch，未执行过。

</details>

<details>
<summary>代码 FIM：仓库隔离与 PSM exact-match</summary>

- 数据卡字段 `repo_id/file_path/content`；repo 哈希 80/10/10，全局源码去重，空 split 不随机补救。
- 文件内独立分块与 padding，真实 EOS 保留标签、padding 屏蔽；不跨文件 attention。
- 文本 / token exact-match 保留空白；不声称 pass@1、编译率，不执行生成代码。
- `fim_export.py` 打包 adapter-only 并在 HF offline 新进程复载；eager 协议明确标注为自建迁移，不是上游复现。

```bash
make fim-dry-run
make fim-suite-dry-run
make fim-export-dry-run
```

</details>

<details>
<summary>xLAM function-calling：completion-only JSON calls</summary>

- 只对 `<calls>` 严格 JSON 数组 `</calls>` + EOS 计算 loss，prompt 部分 label=-100；gold 不进生成输入。
- 给全 tools；评估只用 `json.loads`，绝不 `eval` 或执行生成调用。
- 四指标 `json_parse_rate` / `schema_valid_rate` / `tool_call_exact_match` / `argument_f1` 是格式协议指标，
  不等于真实 API 执行正确率、pass@1 或业务安全。
- 数据卡 gated + research-only，商业许可未确认；无 GPU 实跑。

```bash
make xlam-dry-run
make xlam-suite-dry-run
make xlam-export-dry-run
```

</details>

<details>
<summary>RapidFire：4 配置 × 3 seed 顺序对照</summary>

- 上游 `train[:64]` / `eval[50:60]` 泄漏；本仓规范化问题 hash 80/10/10，同题语言变体同 split，全局去重。
- 4 配置按 `[[configs]]` 精确顺序执行，不冒充并发 / 16–24× 吞吐；`config_id` 进入 fingerprint。
- 纯 Python 无 stemmer ROUGE-L + `exact_match`，与上游 `use_stemmer=True` 的 rougeL 不可比。
- validation paired 均值选配置，test 仅在选中配置验收一次；`upstream_throughput_comparable=false`。

```bash
make rapidfire-dry-run
make rapidfire-suite-dry-run
make rapidfire-export-dry-run
```

</details>

<details>
<summary>Granite Vision：图像隔离与双层 mask</summary>

- 上游只有 50/50、无 validation、无 seed / revision / compute_metrics，评估仅 `test_idx=20` 定性；本仓是自建协议。
- 图像 `mode + tobytes()` SHA256 做 80/10/10，同图问答变体同 split；prompt/full 两次模板化并做 token 级前缀校验。
- 显式 `image_token_id` + assistant span 外双层 `label=-100`；nf4 / DoRA，冻结 processor identity。
- 逐题 `exact_match` / `normalized_exact_match` 只衡量文本一致性，不解释几何语义、不等于人工判分或业务验收。

```bash
make granite-dry-run
make granite-suite-dry-run
make granite-export-dry-run
```

</details>

<details>
<summary>PaliGemma2：定位 token round-trip 与四指标</summary>

- 上游 `format="coco"` 按 xywh 输入却按 xyxy 可视化，训练/推理 prompt 不一致，且 prompt 也计 loss；本仓逐项修正。
- COCO `xywh`→`xyxy` 严格校验，`y1,x1,y2,x2` 顺序 × 1024 桶，单框 encode/decode，`round_trip_min_iou=0.98`。
- 统一 prompt、显式 image token + completion-only 双层 mask；普通 Trainer + 普通 LoRA，不用 4bit / DoRA。
- 四指标 `bbox_iou` / `iou_at_0_5` / `parse_rate` / `exact_match` 从 raw bbox 重建 gold 重算；
  不等于 mAP/AP 或业务验收，不执行代码。Flickr URL 图像必须先冻结为静态副本。

```bash
make paligemma-dry-run
make paligemma-suite-dry-run
make paligemma-export-dry-run
```

</details>

<details>
<summary>SmolVLM DPO：显式 beta 与冻结 ref</summary>

- 上游 DPOConfig 未给 beta/loss/seed/revision/compute_metrics，评估仅定性；本仓自建 sigmoid DPO。
- 图像字节 hash 80/10/10，completion-only 双层 mask，policy/ref 参数空间互斥。
- `win_rate` / `reward_accuracy` 是隐式奖励 pairwise，不是外部 RM 或人工偏好。
- 与 MPO 共享偏好 split 算法，但基座不同，不能归因算法差异。

```bash
make dpo-dry-run
make dpo-suite-dry-run
make dpo-export-dry-run
```

</details>

<details>
<summary>Qwen2.5-VL MPO：三损失消融</summary>

- 上游用未 pin 的 TRL 源码安装，显式 `bf16=False` 与量化 `bfloat16` 矛盾；本仓如实登记，不静默修正。
- `sigmoid` / `bco_pair` / `sft` 权重 0.8 / 0.2 / 1.0；`full` / `sigmoid_only` / `no_sft` 只在本篇 Qwen backbone 内消融。
- 与 DPO 共享偏好 split；`cross_backbone_attribution_valid=false`。
- 一手 MPO 论文仍 unverified，只作补充。

```bash
make mpo-dry-run
make mpo-suite-dry-run
make mpo-export-dry-run
```

</details>

其余 6 篇为 4 篇 `not_implemented` + 2 篇原理性阻塞；不可用统一入口文件的存在冒充实现。

## 完整方法覆盖与实施队列

当前偏向视觉 SFT 与 GRPO 示例，**不是全生态覆盖**。`registry/capabilities.toml` 区分：

- taxonomy 已登记但无 recipe 的 family（运行时自动计算）；
- CPT/DAPT、音频、视频等尚未纳入 taxonomy 的独立模块；
- 已有视觉 DPO/MPO 不等于文本偏好优化已实现；数据、评估、导出是跨方法管线，不是 trainer。

| 波次 | 内容 | 晋级依据 |
| --- | --- | --- |
| wave0 | 失败关闭的状态、输入、环境、证据门禁 | 正反向测试，不靠非空字段或假文件 |
| wave1 | 数学样板 → 独立训练 seed 聚合 → 导出复载 | 真实 zero-shot / smoke / full，至少 3 seed，同协议评估与产物验证 |
| wave2 | 文本 SFT：xLAM、FIM、RapidFire | 数据 / label mask 隔离、任务专用指标、无泄漏与安全执行 |
| wave3 | 视觉 SFT：ChartQA、Granite、PaliGemma | 图像 token mask、图像去重、bbox / loc、污染声明 |
| wave4 | DPO / MPO | shared backbone、ref 模型冻结、三损失消融；不同模型不可直接归因算法 |
| wave5 | vLLM、在线 agent RL、VLM GRPO | 权重同步、同配置开关对照、环境回收与独立对局 |
| wave6 | 偏好 / RM / RLOO / 蒸馏 / CPT 与模态扩展 | 每种方法独立实现、测试、锁、证据与回归 |

波次登记不代表波次已完成；逐篇缺口以能力登记与生成报告为准。

## B2 / B3 的证据要求

- **B2（research-ready）**：真实输入 commit、完整实际运行环境、基座评估、至少 3 个独立训练 seed 的配对评估。
  内容校验包括代码快照、split 去重与 hash、实际环境锁、训练预算 / 参数、checkpoint 身份、逐题预测重算与均值容差。
- **B3（business-ready）**：B2 之上补许可确认、资源实测、业务验收、owner、导出复载及回滚。
  人工 `confirmed` 字段不是自动合规或 SLO 认证。
- 静态 JSON / hash 门禁只证明提交证据**内部一致性**，不能认证 GPU 执行真实性；保留原始日志与产物，由独立运行流程复核。
- 仓库采用 MIT License（见 [LICENSE](LICENSE)）；模型 / 数据许可仍需逐项确认，仓库许可不等于上游资源许可。

## 关键目录

```text
registry/                     taxonomy、reproducibility、baseline、capabilities、feasibility、
                              methodology、mllm_post_training、agent_contract、一手 evidence
recipes/**/recipe.toml        14 篇配方；来源与自建差异显式登记
src/recipe_lab/               审计器与 CLI：plan、math_task、capabilities、feasibility、
                              methodology、post_training、agent_contract、task_router
runner/                       train / eval / zero_shot / lock / suite / export
runner/locks/                 14 份输入 / 环境锁模板；不是实测 freeze
runs/                         本地执行产物（默认忽略，不提交大权重）
tests/                        纯函数、mock、结构集成、正反向证据门禁测试
scripts/                      一次性迁移与来源漂移辅助；重跑不覆盖已有配置 / 锁
docs/BASELINE-READINESS.md    就绪、能力、缺口、依赖与验收（生成）
docs/METHODOLOGY.md           方法论依据、一手论文核查与编号更正（生成）
docs/MLLM-POST-TRAINING.md    MLLM 后训练五方向与 14 篇映射（生成）
docs/AGENT-HANDOFF.md         Agent 交接契约人类可读视图（生成）
docs/AGENT-HANDOFF.json       Agent 交接契约机器可读清单（生成）
docs/REPRODUCIBILITY.md       历史上游可复现性审计（生成）
docs/MATRIX.md                全量配方索引（生成）
docs/DESIGN.md                架构决策与证据检查边界
docs/VERIFICATION.md          规则编号与维护 / 发布门禁区分
AGENTS.md                     Agent 可发现入口与硬边界（静态）
```

新增 recipe 必须同时补 taxonomy 映射（如需）、一手 evidence、repro、baseline_contract 与五维能力登记；
未实现维度填 `missing` 并描述具体阻塞，不以统一入口文件的存在冒充实现。改完运行 `make ci` 与 `make report`。
