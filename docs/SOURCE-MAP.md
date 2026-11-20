# 调研报告 ↔ 仓库映射

上游调研报告：[`llm-vlm-training-cookbook-research.html`](llm-vlm-training-cookbook-research.html)。
报告共 11 章（00–10），本仓库主要承接第 01、02、03、04、05、08、09 章。

| 报告章节 | 内容 | 落到仓库哪里 |
| --- | --- | --- |
| 00 执行摘要 | 九个关键判断 | `README.md` 的「为什么需要它」+ 全部规则 |
| 01 版图 | Cookbook 76 个文件、训练类清单 | `recipes/` 14 篇；未录入项见下文 |
| 02 方法分类学 | TRL v1 taxonomy、dataset type 映射 | `registry/taxonomy.toml` + `CONTRACT.md` 第 1 节 |
| 03 起点拆解 | Qwen2-VL-7B / ChartQA 全参数 | `recipes/vlm/sft/qwen2-vl-7b-chartqa/recipe.toml` |
| 04 VLM 家族 | 7 篇横向拆解 | `recipes/vlm/**` 7 篇 |
| 05 LLM / 在线 RL | GRPO、vLLM、Agent RL、工程化 SFT | `recipes/llm/**` 6 篇 + `recipes/agent/**` 1 篇 |
| 06 生态横评 | 框架分层（二手来源） | 本仓库不复制；仅在设计原则上参考 TRL 作为算法原语层 |
| 07 可复现实操手册 | 复现步骤 | `CONTRACT.md` + `VERIFICATION.md` + `registry/reproducibility.toml` + `REPRODUCIBILITY.md` 的 P0–P5 补齐路径 |
| 07（延伸） | 「每篇都能当 baseline」 | `registry/baseline.toml` + `runner/` + `src/recipe_lab/plan.py` + `BASELINE-READINESS.md` 的 P0–P4 路径 + `docs/DESIGN.md` ADR-010 / ADR-011 |
| 08 陷阱清单 | 20+ 条具体问题 | `TRAPS.md` + 各篇 `[known_issues]` + 审计新增 F09–F15、基线阶段 F16–F18 |
| 09 选型决策树 | Q0–Q3、四阶段实验 | `README.md` 状态机 + `R302` 决策门 |
| 05 / 07（延伸） | 数据规模阶梯、卡数 / 显存 / 时间规划 | `registry/compute_planning.toml` + `docs/COMPUTE-PLANNING.md` + `tests/test_compute_planning.py`（估算与实测锚点分列，与 `recipe.toml`/`methodology.toml` 强校验一致；非 GPU 证据） |
| 10 参考资源 | 链接索引 | 本文件下表；practice 论文方法论见 `registry/methodology.toml` / `METHODOLOGY.md`（ADR-024）；MLLM 后训练五方向见 `registry/mllm_post_training.toml` / `MLLM-POST-TRAINING.md`（ADR-025） |

---

## 已录入的 14 篇 recipe

| recipe id | 上游页面 | 报告章节 | 覆盖度 |
| --- | --- | --- | --- |
| `vlm/sft/qwen2-vl-7b-chartqa` | [fine_tuning_vlm_trl](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) | 03 / 04 | params |
| `vlm/sft/smolvlm-chartqa` | [fine_tuning_smol_vlm_sft_trl](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl) | 04 | params |
| `vlm/sft/granite-vision-line-comparison` | [fine_tuning_granite_vision_sft_trl](https://huggingface.co/learn/cookbook/fine_tuning_granite_vision_sft_trl) | 04 | catalog |
| `vlm/sft/paligemma2-refcoco-grounding` | [fine_tuning_vlm_object_detection_grounding](https://huggingface.co/learn/cookbook/fine_tuning_vlm_object_detection_grounding) | 04 | catalog |
| `vlm/dpo/smolvlm-rlaif-v` | [fine_tuning_vlm_dpo_smolvlm_instruct](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct) | 04 | catalog |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | [fine_tuning_vlm_mpo](https://huggingface.co/learn/cookbook/fine_tuning_vlm_mpo) | 04 | params |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | [fine_tuning_vlm_grpo_trl](https://huggingface.co/learn/cookbook/fine_tuning_vlm_grpo_trl) | 04 | params |
| `llm/grpo/qwen2-0-5b-numinamath` | [fine_tuning_llm_grpo_trl](https://huggingface.co/learn/cookbook/fine_tuning_llm_grpo_trl) | 05 | params |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | [grpo_vllm_online_training](https://huggingface.co/learn/cookbook/grpo_vllm_online_training) | 05 | params |
| `llm/grpo/qwen2-5-3b-advanced-reward` | [trl_grpo_reasoning_advanced_reward](https://huggingface.co/learn/cookbook/trl_grpo_reasoning_advanced_reward) | 05 | params |
| `llm/sft/llama3-8b-xlam-function-calling` | [function_calling_fine_tuning_llms_on_xlam](https://huggingface.co/learn/cookbook/function_calling_fine_tuning_llms_on_xlam) | 05 | catalog |
| `llm/sft/starcoder1b-fim` | [fine_tuning_code_llm_on_single_gpu](https://huggingface.co/learn/cookbook/fine_tuning_code_llm_on_single_gpu) | 05 | catalog |
| `llm/sft/rapidfire-multiconfig` | [rapidfire_sft_multiconfig_training](https://huggingface.co/learn/cookbook/rapidfire_sft_multiconfig_training) | 05 | catalog |
| `agent/grpo/wordle-openenv-hf-jobs` | [grpo_agent_wordle_hf_jobs](https://huggingface.co/learn/cookbook/grpo_agent_wordle_hf_jobs) | 05 | params |

覆盖度沿用报告口径：`params` = 已逐参数核对，`catalog` = 仅确认目录定位与方法类别。
**`catalog` 篇不会对内部配置作任何推断**，其 `[todo]` 列出了补齐项。

---

## 报告中提到、但本仓库尚未录入

这些在调研中属于「目录级」或「非训练」内容，刻意不假装已核对：

- `fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format`（结构化 JSON 输出 SFT）
- `prompt_tuning_peft`、`finetune_t5_for_search_tag_generation`（PEFT 早期方法）
- `optuna_hpo_with_transformers`、`dspy_gepa`（超参 / 非梯度程序优化）
- `fine_tune_chatbot_docs_synthetic`（合成数据 + SFT）
- `generate_preference_dataset_distilabel`、`clean_dataset_judges_distilabel`（偏好数据构建与清洗，属数据侧而非训练侧）
- `structured_generation_vision_language_models` 与三篇 multimodal RAG
  （ColPali / ColQwen2 / SmolVLM）——**不微调的替代方案**，正是起点 recipe 第 6 节
  所指的方向，后续应作为 P0 基线单列一组，而非混进训练 recipe。

---

## 一手 / 二手来源标注

- **一手**：Hugging Face Cookbook 页面、TRL 官方文档（`dataset_formats`、trainer
  taxonomy）、`trl-vlm-alignment` 博客。各 recipe 的 `source` 字段均指向这些页面。
- **二手**：调研报告第 06 章的框架 stars / 加速倍数 / 版本时点。本仓库**没有**
  把它们写进 recipe 配置，只在 `vllm` 篇的 `[benchmark]` 中保留了官方 recipe
  自身给出的吞吐实测值（0.07 vs 0.04 it/s）。
- **关键**：调研报告本身也只是**线索**。可复现性审计阶段发现，上一轮有配置值
  来自二手总结甚至纯属编造（Agent Wordle 的 `Qwen2.5-0.5B-Instruct`、
  `trl 1.7.0 / transformers 5.3.0`），已在 `registry/evidence.toml` 记录
  `was → now → source` 并修正。
- **论文方法论（ADR-024）**：`registry/methodology.toml` 逐篇把设计决策绑定到 practice
  论文，arXiv 编号 / 标题 / 年份 / 核心结论一律回一手摘要页核对；未核到一手来源的
  （MPO 论文、RefCOCO 出处、xLAM 论文）标 `verified=false`，只作补充、不支撑主张。
  本轮核查实际发现并更正 10 条凭记忆写错的编号（FIM、RLAIF-V、RefCOCO、数据污染等），
  登记在 `[[corrections]]` 的 `was → now`，见 [METHODOLOGY.md](METHODOLOGY.md)。
  论文的 benchmark 数字不写进 recipe 的 `evaluation.expected/tolerance`，也不冒充本仓实测。

- **MLLM 后训练方向（ADR-025）**：`registry/mllm_post_training.toml` 把 14 篇映射到
  指令遵循 / 偏好校准 / 推理增强 / 领域自适应 / 可扩展学习五方向。一手核查确认用户提纲
  来自 preprints 202607.1494 摘要（正文页受防护，只做摘要 + 索引级核实）；
  arXiv 2503.06072 是 LLM 后训练五范式的跨域交叉印证，
  `WenkeHuang/Awesome-MLLM-Tuning` 的 Selective / Additive / Reparameterization 是不同轴的方法索引。
  三者边界分别登记，索引或交叉印证来源不得冒充五方向原始出处或复现证据；
  实现状态从 `capabilities.toml` 重算，原理性阻塞篇标为无效覆盖证据。
  见 [MLLM-POST-TRAINING.md](MLLM-POST-TRAINING.md)。

- **Agent 交接（ADR-026）**：`AGENTS.md`、`registry/agent_contract.toml` 与
  `recipe-lab handoff` 把人类入口 `AGENT-HANDOFF.md` 与机器入口 `AGENT-HANDOFF.json`
  绑定到同一事实源。生成报告必须由 CLI 重建，提升门禁保持失败关闭；
  这不是第六层审计，也不把静态检查升级为 GPU 证据。

`confidence` 字段目前 14 篇全为 `primary`；若后续引入社区 benchmark 数字，
应显式降为 `secondary`。

---

## 可复现性审计的材料覆盖

审计不复用本表上面的「章节映射」，而是单独登记在 `registry/evidence.toml [method]`：

| 材料类型 | 篇数 | 说明 |
| --- | --- | --- |
| 本地 `.ipynb` 原始 source | 3 | `fine_tuning_vlm_trl` / `fine_tuning_vlm_grpo_trl` / `fine_tuning_granite_vision_sft_trl`，逐行核查 |
| 上游页面事实抽取 | 11 | 覆盖剩余 11 篇；**未逐字节校验代码单元格** |
| 合计 | **14/14** | `evidence_ref` 全部可由 `R507` 追溯到上述登记 |

`coverage = 14/14` 的含义是「都回过一手材料」，不是「都逐行核对了源码」。
14 篇 `source_sha` 全为空（F15），上游 notebook 漂移后无法定位改动。
结论与补齐路径见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

---

## 基线阶段的材料边界

最初Baseline声明升级没有新增上游事实核查；后续FIM适配已重新核对Cookbook与数据卡（见registry/evidence.toml的fim_adapter_audit）。自建执行契约不表示输入已冻结：
`runner/`、`registry/baseline.toml`、14 份 `runner/locks/*.lock.toml` 模板、
以及各篇 `[execution]` / `[baseline_contract]` / `[resources]` 段。因此：

- 这些字段是**本仓库的定义**，不是上游原文的复述；`seed_origin` 逐篇标注了
  「本仓库声明」还是「来自上游原文」，`evaluation.split` 中带「本仓库定义」
  字样的划分同理。
- `lock` 文件是「应该锁什么」的模板，不是 `pip freeze` 结果；真实 hash 由
  `runner/lock.py --apply` 在有网环境写入，本机 Hub API 返回 502，故 hash 全为空。
- 基线就绪状态与 `repro.level` 完全独立：前者看我们自己的契约，后者看上游留存。
  完整论证见 [BASELINE-READINESS.md](BASELINE-READINESS.md) 与
  [DESIGN.md](DESIGN.md#adr-010)。
- `--apply` 解析输入，`--capture-environment` 冻结实际运行环境；两个阶段不混同，当前锁仍为模板。

### 当前能力与可行性分账（wave5b 后）

- **能力矩阵**（`registry/capabilities.toml`）：**9 篇**具备五维代码与隔离离线测试（70 格中 **45 implemented / 25 missing**，五个维度各 9/14），**GPU verified 0**；其余 5 篇中 3 篇全维 missing、2 篇原理性阻塞。
- **可行性判定**（`registry/feasibility.toml`）：`eligible 0/14`、`pending_freeze 9`、`partial_implementation 0`、`not_implemented 3`、`blocked_zero_train 1`、`contaminated_pretraining 1`。五维齐备不等于可晋级，仍全数缺冻结 revision、CUDA 多 seed 实跑、预声明容差、导出复载与许可确认。
- **算力与规模规划**（`registry/compute_planning.toml` + [COMPUTE-PLANNING.md](COMPUTE-PLANNING.md)）：把 14 篇的 batch / 步数 / 精度 / PEFT / 序列长 / 发布级 4–16 倍乘法与 S0–S4 数据阶梯、实测锚点、估算显存矩阵分开成表，由 `tests/test_compute_planning.py` 强校验；估算不构成 GPU 实跑证据。

| 任务 | `recipe_id` / `task_id` | 一手审计键（`registry/evidence.toml`） | 自建协议与关键偏差（vs 上游） | ADR 与离线测试 |
| --- | --- | --- | --- | --- |
| 数学 GRPO | `llm/grpo/qwen2-0-5b-numinamath` / `math-grpo-v1` | `findings` | 固定 `500/500/500` 连续划分与跨 split 去重；gold 解析失败直接停止，拒绝重复 greedy 冒充多 seed；3 seed 聚合 + 完整模型 HF offline 复载 | `test_math_task.py` / `test_suite.py` / `test_export.py` |
| 代码 FIM | `llm/sft/starcoder1b-fim` / `code-fim-v1` | `[fim_adapter_audit]` | repo 哈希 80/10/10 隔离、文件内分块 padding、PSM 保留空白 exact-match（不执行代码，不冒充 `pass@1`）；eager 自建迁移 + adapter-only 复载 | `test_fim.py` / `test_fim_release.py` |
| xLAM FC | `llm/sft/llama3-8b-xlam-function-calling` / `xlam-fc-v1` | `[xlam_adapter_audit]` | 修上游全文损失/截断 tools/`str(dict)` 输出：completion-only `<calls>` 严格 JSON 数组、给全 tools、绝不 `eval` 或执行调用；data gated/research-only | `test_xlam.py` / `test_xlam_release.py` |
| RapidFire | `llm/sft/rapidfire-multiconfig` / `rapidfire-sft-v1` | `[rapidfire_adapter_audit]` | 修 `eval⊂train` 泄漏与全文损失：规范化问题 hash 80/10/10、纯 Python 无 stemmer ROUGE-L（与上游不可比）；4 配置 × 3 seed 顺序执行，validation 选配置 / test 验收一次 | [ADR-018](DESIGN.md#adr-018) / ADR-019；`test_rapidfire.py` / `test_rapidfire_release.py` |
| Granite Vision | `vlm/sft/granite-vision-line-comparison` / `granite-linecomparison-v1` | `[granite_vision_audit]` | 修 50/50 无 validation、未显式屏蔽图像 token、正文/代码 ChartQA 错配、未 `merge_and_unload`：图像 bytes SHA256 80/10/10、双层 `-100` mask、普通 `Trainer` + DoRA/4bit、文本 `exact_match` 不解释几何语义 | ADR-020 / ADR-021 / ADR-022；`test_granite.py` / `test_granite_release.py` |
| PaliGemma2 | `vlm/sft/paligemma2-refcoco-grounding` / `paligemma-grounding-v1` | `[paligemma_grounding_audit]` | 修 `coco_xywh`/xyxy 口径冲突、prompt 大小写标点不一致、prompt 计 loss 与 `do_sample=True`：统一 `PROMPT_TEMPLATE`、1024 桶单框 round-trip（`IoU≥0.98`）、普通 LoRA、4 指标不等于 mAP/AP；Flickr URL 需冻结静态图 | ADR-023；`test_paligemma.py` / `test_paligemma_release.py` |
| SmolVLM DPO | `vlm/dpo/smolvlm-rlaif-v` / `smolvlm-dpo-v1` | `[dpo_adapter_audit]` | 补齐缺失 `beta`/loss/seed/定量指标：图像 bytes hash 80/10/10、显式 `beta=0.1` sigmoid、冻结 ref、completion-only 双层 mask、隐式奖励 pairwise；与 MPO 非 shared backbone（`cross_backbone_attribution_valid=false`） | ADR-027；`test_preference.py` |
| Qwen2.5-VL MPO | `vlm/mpo/qwen2-5-vl-3b-mpo` / `qwen25vl-mpo-v1` | `[mpo_adapter_audit]` | 共享偏好 split，显式 `sigmoid/bco_pair/sft`（0.8/0.2/1.0）与 3 消融 × 3 seed；如实登记 `bf16=False` 与量化 `bfloat16` 矛盾；一手 MPO 论文 `unverified`；禁止与 DPO 篇跨 backbone 归因 | ADR-027；`test_preference.py` |
| GSM8K 四 reward | `llm/grpo/qwen2-5-3b-advanced-reward` / `gsm8k-reward-v1` | `[gsm8k_adapter_audit]` | `####` gold Decimal 规范化剥离、官方 train 90/10 hash 切 + 1319 test 冻结、独立严格 verifier 与 `reward_hacking_rate`；5 drop-one × seed + 整模型 safetensors 离线复载；四函数等权未经归一化，消融限同 backbone 内 | ADR-028 / ADR-029；`test_gsm8k.py` / `test_gsm8k_release.py` |

- **发布级分派与内容身份边界**：9 篇均已在 `task_router.py` 精确注册，C306/C307/C405 按任务分派、互不借用协议；真实 recipe 未预声明 `evaluation.export_tolerance` 时导出 `--full` 失败关闭，不回填容差。内容身份按任务语义区隔——RapidFire `_identity_recipe()` 剥离 `expected`/`tolerance`/`execution.evidence`，Granite `identity_recipe()` 保留预声明的 `execution.evidence` 以防训练 fingerprint 失配。
- **原理性阻塞（2 篇）与未实现（3 篇）**：`vlm/sft/qwen2-vl-7b-chartqa`（`blocked_zero_train`，`decision.gate=blocked`）与 `vlm/sft/smolvlm-chartqa`（`contaminated_pretraining`，基座已在 ChartQA 训过）登记在 `[chartqa_baseline_feasibility]`，补代码或证据都不能晋级；`wordle-openenv-hf-jobs`、`qwen2-0-5b-vllm-colocate`、`qwen2-5-vl-3b-visual-reasoning` 仍为五维 `not_implemented`。

