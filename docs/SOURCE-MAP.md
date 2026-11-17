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
- 实装由`registry/capabilities.toml`记录：数学、FIM、xLAM、RapidFire、Granite、PaliGemma 六篇各五维均有offline测试（Granite 三维为`src/recipe_lab/granite_task.py` + `tests/test_granite.py` 58项，发布级两维为`granite_suite.py` / `granite_export.py` + `tests/test_granite_release.py` 44项；PaliGemma 三维为`paligemma_task.py` + `tests/test_paligemma.py` 46项，发布级两维为`paligemma_suite.py` / `paligemma_export.py` + `tests/test_paligemma_release.py` 44项），70格中30 implemented/40 missing（五个维度各6/14），GPU未验证；其余8篇全维missing。Granite自建协议：图像原始 `mode + tobytes()` SHA256 做80/10/10且同图所有问答变体同split、规范化(question,answer)全局去重、prompt/full两次`apply_chat_template`并校验token级前缀、显式`image_token_id`+assistant-only双层`label=-100`、普通`Trainer`自建collator、冻结processor identity、greedy+首个EOS截断后逐题`exact_match`/`normalized_exact_match`重算；不解释几何语义、不等于人工判分，无CUDA实跑与completed证据。发布级按ADR-022补齐：`task_router`已精确注册`granite-linecomparison-v1`，recipe显式声明`execution.task_id`，`granite_suite`做三互异独立训练seed逐题从`eval_manifest`的question/answer重建gold并重算两指标与Student-t95聚合，`granite_export`导出DoRA/4bit adapter-only包并在HF offline新进程复载复评，C306/C307/C405按任务分派、互不借用协议；`identity_recipe()`只剥离`expected`/`tolerance`，不剥离作为预声明位置路径的`execution.evidence`（与RapidFire不同）。R103的按精确`recipe_id`查白名单路径仍保留兼容，见[DESIGN.md](DESIGN.md#adr-021)与[DESIGN.md](DESIGN.md#adr-022)。RapidFire一手事实与自建偏差见`registry/evidence.toml [rapidfire_adapter_audit]`：上游eval[50:60]⊂train[:64]、整段文本损失、无revision/包版本、从未报告任何rougeL数值、数据卡未给license；本仓自建问题hash隔离划分、completion-only loss、纯Python无stemmer ROUGE-L并顺序执行4配置，不冒充上游并发调度复现；发布级按ADR-019做4配置×3 seed的validation选配置/test单次验收与成本对照、选中adapter的HF offline新进程test复载。PaliGemma2 grounding篇一手事实与偏差见`registry/evidence.toml [paligemma_grounding_audit]`：上游`SFTTrainer`+`skip_prepare_dataset`、`labels=input_ids.clone()`只mask pad/`<image>`（prompt也计loss）、`A.BboxParams(format="coco")`与`coco_to_xyxy()`按COCO `xywh`输入而可视化直接当`xyxy`、训练/推理prompt三处不一致、`<loc####>`为1024桶按`y1,x1,y2,x2`排列、`SFTConfig`有`eval_dataset`却无`compute_metrics`/IoU/任何数值、只有`train[20]`/`val[13]`单样本定性、`do_sample=True`无生成seed、无revision/训练seed/版本锁、gated基座与未确认许可、图像由Flickr URL运行时下载；本仓自建`coco_xywh`→`xyxy`统一、1024桶单框`<loc####>` round-trip/IoU（下限0.98）、统一`PROMPT_TEMPLATE`、completion-only双层mask、普通`Trainer`/普通LoRA、held-out四指标逐题重算与三seed从raw bbox重建gold的发布级重算，四指标不等于mAP/AP或业务验收。六篇发布级套件/导出按`task_router.py`精确分派，互不借用协议；真实recipe未预声明`evaluation.export_tolerance`时导出`--full`失败关闭，不回填容差。固定数学500/500/500、FIM repo哈希隔离/独立文件padding/PSM指标、xLAM严格JSON输出协议与completion-only loss都是自建协议，不冒充上游结果或无预训练污染证明。xLAM一手事实与偏差见`registry/evidence.toml [xlam_adapter_audit]`：上游SFTTrainer把gold拼进全文损失、推理prompt截断tools、输出是Python str(dict)、数据卡gated/research-only且曾重生成条目。
- 逐篇「原理上能否作 baseline」另由`registry/feasibility.toml`登记：当前`eligible 0/14`，两篇ChartQA分别因`decision.gate=blocked`（上游自承零训练基线达标）与基座在ChartQA上预训练过而判为原理性阻塞，不因五维代码或completed证据晋级。Granite Vision篇2026-11-14回一手页面核查（`[granite_vision_audit]`）：50/50切分无validation、assistant-only隐式mask无显式image token屏蔽、无revision/训练seed/compute_metrics、评估仅`test_idx=20`单样本定性、文字写ChartQA但代码未加载、说merge却无`merge_and_unload()`。wave3因此按Granite篇、PaliGemma2 grounding篇排序，见[DESIGN.md](DESIGN.md#adr-020)；两篇现在都已补齐五维并精确注册`task_router`，可行性均改判`pending_freeze`（只反映五维代码齐备，仍非eligible）。PaliGemma篇2026-11-16回一手页面核查（`[paligemma_grounding_audit]`）见[DESIGN.md](DESIGN.md#adr-023)。`partial_implementation`枚举与交叉校验保留但真实登记计数为0，两篇ChartQA的原理性阻塞与`eligible 0/14`不变。
- `--apply`解析输入，`--capture-environment`冻结实际运行环境；两个阶段不混同，当前锁仍为模板。
