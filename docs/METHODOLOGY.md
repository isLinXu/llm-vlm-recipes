# 方法论依据层（practice papers）

> 由 `recipe-lab methodology` 生成；单一事实源是 `registry/methodology.toml`。本文件每次 `make report` 重建，请勿手工编辑。

## 为什么需要这一层

前三层审计各有分工，但都回答不了「为什么这么设计」：

- `capabilities.toml`：代码 + offline test 是否实装（**能不能跑起来**）；
- `baseline.toml` / `baseline_evidence.py`：证据与就绪度（**有没有真实跑过**）；
- `feasibility.toml`：实验目标是否成立、数据是否污染（**值不值得当 baseline**）；
- 本层：设计决策依据哪篇 practice 论文的一手结论、论文没证明什么（**为什么这么设计**）。

四层互不替代，任何一层通过都不解锁 GPU 晋级或 B2/B3。论文的方法论背书**不等于**本仓实测；本仓的 expected/tolerance 必须来自真实 CUDA 实跑，绝不拿论文 benchmark 数字回填。

每篇 `recipe.toml` 另有一个 `[methodology]` 段，只保存 registry 指针、主张 id、论文 id 与简短立场，方便读配方时直接看到依据。它不是第二份事实源：`methodology --check` 会把它与 `registry/methodology.toml` 的 `principles` / `primary_papers` / `supplement_papers` / `stance` 逐字段比较，任一侧漂移即失败关闭。

## 核查方法与诚信边界

每条 `principles.*` 绑定一篇论文的 `paper_supports` / `we_adapted` / `paper_limited` 关系与一句 `boundary`，明确区分「论文一手结论直接支持」「我们据此修正上游（本仓工程判断）」「论文结论有明确边界或本仓不适用」。所有论文的 arXiv 编号、标题、年份与核心结论均回一手摘要页核对；未核到一手来源的（如 MPO 论文、RefCOCO 原始出处、xLAM 论文）一律 `verified=false`，只能进 `supplement_papers`，不得支撑任何主张。

## 当前结论

当前核查论文 **20 篇一手确认** / 3 篇未核实（仅作补充）；方法论主张 24 条，编号更正 10 条，覆盖 14/14 篇。

所有论文的 arXiv 编号、标题与结论均回一手摘要页核对。核查中实际发现 10 条凭记忆写错的编号（FIM、RLAIF-V、MPO、RefCOCO 等），已逐条登记 `was → now`，作为「一手核查不可省略」的物证；未核到一手来源的一律标 unverified，不得作主张依据，也不把论文 benchmark 数字写成本仓 expected/tolerance。

| recipe | 方法论立场 | 一手主论文 | 主张 |
| --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | GRPO 与 reward 过优化提供在线 RL 动机；按对局隔离与会话回收是本仓工程判断。OpenEnv 轨迹 schema、rollout/桥接、独立 solve-rate 均未实现，HF Jobs/papermill 不算本仓执行层。 | `deepseekmath_grpo`, `reward_overopt` | `grpo_critic_free`, `reward_hacking_needs_independent_metric`, `session_isolation` |
| `llm/grpo/qwen2-0-5b-numinamath` | GRPO 主方法来自 DeepSeekMath 一手摘要；多 seed 与 reward 背离监控是防欠规范/过优化的自建纪律；评估固定 greedy，明确排除 self-consistency 的采样收益。全部只有离线合成证据。 | `deepseekmath_grpo`, `reward_overopt`, `underspecification`, `self_consistency` | `grpo_critic_free`, `reward_hacking_needs_independent_metric`, `multiseed_underspecification`, `self_consistency_not_for_repro` |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | vLLM 收益属 serving 层（PagedAttention 一手摘要），必须单变量 on/off 对照才能归因；本仓五维未实现，未做同步/吞吐/显存实测。 | `vllm_pagedattention`, `deepseekmath_grpo` | `vllm_off_on_ablation`, `grpo_critic_free` |
| `llm/grpo/qwen2-5-3b-advanced-reward` | 四 reward 与 GSM8K 消融的方法论动机是 reward 过优化与欠规范；本仓已实装 data/train/eval 三维（gold 剥离、独立严格 verifier、reward hacking 诊断），四 reward 逐项消融与多 seed 聚合（comparison）及导出复载（export）仍缺。 | `reward_overopt`, `underspecification`, `self_consistency` | `reward_hacking_needs_independent_metric`, `multiseed_underspecification`, `self_consistency_not_for_repro` |
| `llm/sft/llama3-8b-xlam-function-calling` | 工具调用结构依据 Toolformer；严格 JSON、gold 不进 prompt、json.loads 不执行调用是本仓自建；xLAM 数据卡 gated/research-only，一手 arXiv 论文未核到，仅作补充。 | `toolformer`, `datasheets` | `toolformer_api_args`, `strict_json_completion_only`, `datasheets_license_documentation` |
| `llm/sft/rapidfire-multiconfig` | 质量优先与去重有文献依据；无 stemmer ROUGE-L、4 配置顺序执行是自建修正，数值与上游不可比；Bitext 数据卡未给 license。 | `datasheets`, `lima`, `dedup_training_data` | `datasheets_license_documentation`, `lima_quality_over_quantity`, `dedup_reduces_leakage` |
| `llm/sft/starcoder1b-fim` | FIM 数据变换与 repo 级去重有论文依据；PSM 评估、文件内 padding 与 adapter-only 是本仓自建；不执行生成代码，exact-match 不冒充 pass@1。 | `fim_infill`, `dedup_training_data`, `lora` | `fim_augmentation_preserves_l2r`, `dedup_reduces_leakage`, `completion_only_loss`, `peft_lora_reduces_memory` |
| `vlm/dpo/smolvlm-rlaif-v` | DPO 闭式损失与 RLAIF-V 开源反馈各有论文依据；本仓该篇已有自建五维迁移实现（smolvlm-dpo-v1）与隔离离线测试，但与 MPO 篇非 shared backbone，不能把指标差异归因 DPO vs MPO 算法；无 GPU 实跑与业务验收。 | `dpo`, `rlaif_v`, `underspecification` | `dpo_closed_form`, `rlaif_v_open_feedback`, `shared_backbone_ablation` |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | GRPO 的数学 reward 依据可迁移到视觉推理的动机，Qwen2.5-VL 支持 bbox 定位；但本仓五维未实现，视觉 rollout/reward 失败关闭与图像级隔离均缺。 | `deepseekmath_grpo`, `qwen25vl`, `qwen2vl`, `reward_overopt` | `grpo_critic_free`, `single_box_iou_loc`, `reward_hacking_needs_independent_metric` |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | MPO 三损失组合只见二手/TRL release 说明，一手 arXiv 未核到，仅作补充；本仓该篇已有自建五维迁移实现（qwen25vl-mpo-v1）与隔离离线测试，sigmoid/bco_pair/sft 装配与消融已显式声明；与 DPO 篇非 shared backbone，不能归因算法差异；无 GPU 实跑与业务验收。 | `dpo`, `underspecification`, `qlora` | `dpo_closed_form`, `shared_backbone_ablation`, `qualitative_only_gap` |
| `vlm/sft/granite-vision-line-comparison` | DoRA/QLoRA 装配有论文依据；双层 mask、同图 hash 隔离、逐题重算是修正 Cookbook 的自建协议。上游文字写 LoRA 而配置为 use_dora=True，已按一手代码登记。 | `dora`, `qlora`, `lora` | `dora_decomposition`, `qlora_fourbit`, `completion_only_loss`, `image_level_split`, `qualitative_only_gap` |
| `vlm/sft/paligemma2-refcoco-grounding` | PaliGemma/2 论文只支持模型与迁移定位，loc 1024 桶与 COCO xywh 来自 Cookbook 一手代码；本仓统一 prompt、双层 mask、单框 IoU 为自建修正，RefCOCO 一手来源未核到。 | `paligemma`, `paligemma2`, `qwen25vl` | `single_box_iou_loc`, `completion_only_loss`, `image_level_split`, `qualitative_only_gap` |
| `vlm/sft/qwen2-vl-7b-chartqa` | ChartQA 论文定义数据集、Qwen2-VL 是模型技术报告；两者都不支持训练收益。本篇方法论结论是 decision.gate=blocked：零训练基线已达标，原理性拒绝训练预算。 | `chartqa`, `qwen2vl` | `chartqa_zero_train_gate`, `qualitative_only_gap` |
| `vlm/sft/smolvlm-chartqa` | 方法论重点是污染检查：基座已在 ChartQA 上训练过，同集 held-out 无效。论文只提供去重/数据集动机；须换未污染外部集并重判，补代码或证据不解除阻塞。 | `dedup_training_data`, `chartqa` | `chartqa_contamination`, `dedup_reduces_leakage`, `qualitative_only_gap` |

### 一手核查到的核心论文

| id | arXiv | 标题 | 年份 |
| --- | --- | --- | --- |
| `chartqa` | 2203.10244 | ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning | 2022 |
| `datasheets` | 1803.09010 | Datasheets for Datasets | 2018 |
| `dedup_training_data` | 2107.06499 | Deduplicating Training Data Makes Language Models Better | 2021 |
| `deepseekmath_grpo` | 2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | 2024 |
| `dora` | 2402.09353 | DoRA: Weight-Decomposed Low-Rank Adaptation | 2024 |
| `dpo` | 2305.18290 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | 2023 |
| `fim_infill` | 2207.14255 | Efficient Training of Language Models to Fill in the Middle | 2022 |
| `lima` | 2305.11206 | LIMA: Less Is More for Alignment | 2023 |
| `lora` | 2106.09685 | LoRA: Low-Rank Adaptation of Large Language Models | 2021 |
| `paligemma` | 2407.07726 | PaliGemma: A versatile 3B VLM for transfer | 2024 |
| `paligemma2` | 2412.03555 | PaliGemma 2: A Family of Versatile VLMs for Transfer | 2024 |
| `qlora` | 2305.14314 | QLoRA: Efficient Finetuning of Quantized LLMs | 2023 |
| `qwen25vl` | 2502.13923 | Qwen2.5-VL Technical Report | 2025 |
| `qwen2vl` | 2409.12191 | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution | 2024 |
| `reward_overopt` | 2210.10760 | Scaling Laws for Reward Model Overoptimization | 2022 |
| `rlaif_v` | 2405.17220 | RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness | 2024 |
| `self_consistency` | 2203.11171 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | 2022 |
| `toolformer` | 2302.04761 | Toolformer: Language Models Can Teach Themselves to Use Tools | 2023 |
| `underspecification` | 2011.03395 | Underspecification Presents Challenges for Credibility in Modern Machine Learning | 2020 |
| `vllm_pagedattention` | 2309.06180 | Efficient Memory Management for Large Language Model Serving with PagedAttention | 2023 |

### 未核实来源（仅作补充，不支撑主张）

- `mpl_mpo`：多来源指向 MPO 用 DPO(sigmoid)+BCO(bco_pair)+SFT 三损失、权重 0.8/0.2/1.0（见 TRL v0.20.0 release 说明），但未核到一手论文摘要页，故不作为任何 principle 的直接依据；本仓实现以 Cookbook/TRL 一手代码为准。
- `referit_refcoco`：记忆中的 1604.07399 经一手核查是宇宙射线模拟论文，已作废；RefCOCO 数据集出处未取得一手摘要页，故标 unverified，仅作补充引用。
- `xlam_dataset`：一手来源是 HF 数据卡（gated、research-only），非论文；数据卡列 query/tools/answers，2024-07 重生成条目与商业许可均未确认。
