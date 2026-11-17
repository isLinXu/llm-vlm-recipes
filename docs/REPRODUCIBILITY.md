# 可复现性审计报告

> 评估日期：2026-11-13 · 对象：14 篇 recipe · 生成器：`recipe-lab repro --report`

## 结论（先说最重要的）

**确认结果：14 篇中，0 篇达到「可复现」（L2 及以上）。**

- L0 仅登记：8 篇
- L1 规格完整但未锁定：6 篇
- L2 完全锁定：0 篇
- L3 可执行：0 篇
- L4 已复现验证：0 篇

这里把「可复现」定义为 **L2 完全锁定**：模型/数据 revision、依赖 lock 与 commit、训练种子、上游 commit 全部可重放。仅仅把超参写进 TOML（L1）不算可复现——上游用的是 `pip install -U -q git+https://github.com/huggingface/trl.git`，任何人重跑都会装到不同的 trl；`from_pretrained` 不锁 revision，Hub 权重更新会静默改变结果；训练 config 里没有 seed。

因此上一轮交付的「0 error / 33 tests passed」只证明**配置自洽**，不证明**结果可复现**。这两件事必须分开。

## 判定标准

| 等级 | 含义 | 达标条件 |
| --- | --- | --- |
| L0 仅登记 | 仅登记方法 / 模型 / 数据集名称，无法据此重建实验 | `provenance` ≥ `declared` |
| L1 规格完整 | 规格写清：参数、依赖、数据契约、评估协议、来源齐备，但均未锁定 | `model` ≥ `declared`、`data` ≥ `declared`、`params` ≥ `declared`、`env` ≥ `declared`、`eval` ≥ `declared`、`provenance` ≥ `declared` |
| L2 完全锁定 | 完全锁定：模型/数据 revision、依赖 lock + commit、训练种子、上游 commit | `model` ≥ `pinned`、`data` ≥ `pinned`、`params` ≥ `pinned`、`env` ≥ `pinned`、`seed` ≥ `pinned`、`eval` ≥ `declared`、`provenance` ≥ `pinned` |
| L3 可执行 | 可执行：在 L2 之上有确定入口命令、可构建环境、评估脚本，无需人工补代码 | `model` ≥ `pinned`、`data` ≥ `pinned`、`params` ≥ `pinned`、`env` ≥ `pinned`、`seed` ≥ `pinned`、`code` ≥ `pinned`、`eval` ≥ `pinned`、`provenance` ≥ `pinned` |
| L4 已复现验证 | 已复现验证：在声明硬件上重跑，指标落在容差内，且有可复查 evidence 产物 | `model` ≥ `pinned`、`data` ≥ `pinned`、`params` ≥ `pinned`、`env` ≥ `pinned`、`seed` ≥ `pinned`、`code` ≥ `verified`、`eval` ≥ `verified`、`provenance` ≥ `pinned` |

八个维度：`code` 执行入口、`env` 环境依赖、`model` 基座模型、`data` 数据集、`params` 训练超参、`seed` 确定性、`eval` 评估闭环、`provenance` 上游溯源。
每个维度的状态取 `missing → declared → pinned → verified`，等级由八个维度的**最低达标面**自动计算，`R503` 会用同一函数重算，虚报即报错。

## 逐篇审计

| recipe | 等级 | code | env | model | data | params | seed | eval | prov | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | **L0** | — | — | 声明 | 声明 | — | — | 声明 | 声明 | `webfetch:grpo_agent_wordle_hf_jobs` |
| `llm/grpo/qwen2-0-5b-numinamath` | **L1** | — | 声明 | 声明 | 声明 | 声明 | — | 声明 | 声明 | `webfetch:fine_tuning_llm_grpo_trl` |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | **L1** | — | 声明 | 声明 | 声明 | 声明 | — | 声明 | 声明 | `webfetch:grpo_vllm_online_training` |
| `llm/grpo/qwen2-5-3b-advanced-reward` | **L0** | — | — | 声明 | 声明 | — | — | 声明 | 声明 | `webfetch:trl_grpo_reasoning_advanced_reward` |
| `llm/sft/llama3-8b-xlam-function-calling` | **L0** | — | — | 声明 | 声明 | 声明 | — | — | 声明 | `webfetch:function_calling_fine_tuning_llms_on_xlam` |
| `llm/sft/rapidfire-multiconfig` | **L0** | — | — | 声明 | 声明 | 声明 | 声明 | 声明 | 声明 | `webfetch:rapidfire_sft_multiconfig_training` |
| `llm/sft/starcoder1b-fim` | **L0** | — | — | 声明 | 声明 | 声明 | 声明 | 声明 | 声明 | `webfetch:fine_tuning_code_llm_on_single_gpu` |
| `vlm/dpo/smolvlm-rlaif-v` | **L1** | — | 声明 | 声明 | 声明 | 声明 | — | 声明 | 声明 | `webfetch:fine_tuning_vlm_dpo_smolvlm_instruct` |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | **L0** | — | — | 声明 | 声明 | 声明 | — | 声明 | 声明 | `local-notebook:fine_tuning_vlm_grpo_trl.txt` |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | **L1** | — | 声明 | 声明 | 声明 | 声明 | — | 声明 | 声明 | `webfetch:fine_tuning_vlm_mpo` |
| `vlm/sft/granite-vision-line-comparison` | **L0** | — | — | 声明 | 声明 | 声明 | — | — | 声明 | `local-notebook:fine_tuning_granite_vision_sft_trl.txt` |
| `vlm/sft/paligemma2-refcoco-grounding` | **L0** | — | — | 声明 | 声明 | 声明 | — | — | 声明 | `webfetch:fine_tuning_vlm_object_detection_grounding` |
| `vlm/sft/qwen2-vl-7b-chartqa` | **L1** | — | 声明 | 声明 | 声明 | 声明 | — | 声明 | 声明 | `local-notebook:fine_tuning_vlm_trl.txt` |
| `vlm/sft/smolvlm-chartqa` | **L1** | — | 声明 | 声明 | 声明 | 声明 | — | 声明 | 声明 | `webfetch:fine_tuning_smol_vlm_sft_trl` |

## 阻塞项统计

| 维度 | 未达 pinned 的篇数 | 主要缺口 |
| --- | --- | --- |
| `code` | 14/14 | 上游 14 篇均为 notebook；本仓库目前没有任何训练入口脚本 |
| `env` | 14/14 | 上游为 -U 无 pin 或 git+ 源码无 commit；测试版本仅出现在注释里 |
| `model` | 14/14 | 14 篇 from_pretrained 全部未传 revision，Hub 权重更新会静默改变结果 |
| `data` | 14/14 | 14 篇 load_dataset 全部未传 revision；split 常被简写或省略 test |
| `params` | 14/14 | 本轮已按一手材料补齐 11 篇；仍缺 seed / 部分完整 trainer config |
| `seed` | 14/14 | 12 篇训练 config 完全没有 seed；code LLM 与 RapidFire 两篇有 run-level seed 但无 deterministic 策略 |
| `eval` | 14/14 | 13 篇无 compute_metrics；唯一有 ROUGE-L 的 RapidFire 评估集与训练集重叠 |
| `provenance` | 14/14 | 14 篇 source_sha 全为空，上游 notebook 漂移后无法定位改动 |

### 硬阻塞（来自 registry/reproducibility.toml）

| ID | 维度 | 封顶等级 | 条件 | 一手证据 |
| --- | --- | --- | --- | --- |
| B01 | `code` | L2 | 无执行入口脚本 | 14 篇上游均为 .ipynb，本仓库未提供 runner |
| B02 | `env` | L1 | 依赖为 git+ 无 commit 或 -U 无 pin | 14 篇中多篇为 pip install -U -q git+https://github.com/huggingface/trl.git（无 commit）；其余为 -U 无版本；唯一精确 pin 是 MPO 篇 qwen-vl-utils==0.0.8 |
| B03 | `seed` | L1 | 训练 config 无 seed / data_seed / deterministic | 12 篇训练 config 无 seed；例外是 code LLM 篇 SEED=0 + set_seed 与 RapidFire 篇 run_fit(seed=42)，但两者均无 deterministic / cudnn / tf32 策略 |
| B04 | `model` | L1 | from_pretrained 无 revision | 14 篇全部「未提及」revision（3 篇本地原文 + 11 篇页面抽取） |
| B05 | `data` | L1 | load_dataset 无 revision | 14 篇全部「未提及」revision（3 篇本地原文 + 11 篇页面抽取） |
| B06 | `eval` | L2 | 无 compute_metrics / 评估脚本 / 定量基线 | 13 篇无 compute_metrics；唯一有 ROUGE-L 的 RapidFire 评估集 range(50,60) 落在训练集 range(64) 内；Wordle 篇有 solve rate 统计但无独立评估脚本 |
| B07 | `provenance` | L2 | source_sha 为空 | 仓库 14 篇当前全部为空 |

## 一手核查发现

方法：3 篇本地原始 notebook 逐行核查 + 11 篇上游页面事实抽取，覆盖全部 14 篇（覆盖 14/14）。
规则：只采信一手原文；页面未提及即记为未提及；总结类二手材料只作为线索，不作为证据。

本轮把上一轮只有二手调研描述的 7 篇（SmolVLM SFT、VLM DPO、VLM MPO、grounding、xLAM、code LLM、RapidFire）全部回上游页面做了事实抽取，并据此修正了模型 / 数据集 id、补齐训练配置、纠正了两处「以为没有但实际上有」的判断：code LLM 篇其实有 `SEED=0 + set_seed` 和 4000 条 held-out，RapidFire 篇其实有 `ROUGE-L compute_metrics` 和 `run_fit(seed=42)`。这两篇仍不可复现，但原因从「完全没有」变成了「有但不完整 / 评估集泄漏」。

### F01 · 依赖安装不可重放（blocker）

- 证据：多篇为 !pip install -U -q git+https://github.com/huggingface/trl.git ...（无 commit、带 -U）；其余为 !pip install -U trl ... 或 %pip install rapidfireai（无任何版本约束）。唯一写进安装命令的精确 pin 是 MPO 篇的 qwen-vl-utils==0.0.8
- 后果：注释里的 # Tested with trl==0.14.0 只是作者当时的环境快照，不是可执行的安装指令；任何人重跑都会拿到不同的 trl

### F02 · 14 篇模型与数据集均无 revision 锁定（blocker）

- 证据：全部 14 篇的 from_pretrained / load_dataset 调用均无 revision；上游页面同样标注「未提及」。唯一出现的 commit hash 是 code LLM 篇注释里的 Megatron-LM 参考链接，不是模型或数据集 revision
- 后果：Hub 权重或数据集更新会静默改变结果，且无法回溯到当时的版本

### F03 · 训练侧普遍无随机种子，但存在两个例外（blocker）

- 证据：12 篇训练 config 无 seed / data_seed / deterministic。例外：code LLM 篇 SEED = 0 + set_seed(SEED) + shuffle(seed=SEED) + ConstantLengthDataset(seed=SEED)；RapidFire 篇 run_fit(..., seed=42) + train/eval_dataset.shuffle(seed=42)
- 后果：切分可复现但训练不可复现；配置里写 evaluation.seed=42 属于我们补的协议，不是上游已有事实。两篇例外也没有 deterministic / cudnn / tf32 策略与 TrainingArguments.seed，仍达不到 pinned

### F04 · 上游无执行入口（major）

- 证据：14 篇均为 .ipynb / Colab notebook，依赖内核状态；GitHub 更新链接全部指向 notebooks/en/*.md，而非 .py 脚本
- 后果：即使参数齐全也无法一条命令重跑

### F05 · 评估普遍缺失，且「有 compute_metrics」不等于「评估有效」（major）

- 证据：仅 RapidFire 篇有 sample_compute_metrics（ROUGE-L）；其余 13 篇无 compute_metrics / trainer.evaluate。advanced reward 篇仅有单题 expected_answer='72' 的格式检查；Wordle 篇是唯一有 solve rate / mean reward / reward delta 统计的一篇
- 后果：无法判定训练是否真的带来增益

### F06 · reward 失败路径存在设计缺陷（major）

- 证据：基础 GRPO 篇 accuracy_reward：if len(gold_parsed) != 0: ... except: 0.0 ; else: rewards.append(1.0) —— gold 答案解析失败时直接给满分
- 后果：数据质量问题会被当成模型答对，reward 曲线虚高；原配置写成「解析失败返回 None」与原文相反

### F07 · VLM 图像列与 prompt 结构与我们的契约假设不同（major）

- 证据：VLM GRPO 篇 make_conversation 返回 {"prompt": <apply_chat_template 后的字符串>, "image": example["image"]}；顶层图像列是单数 image
- 后果：原配置写 images（复数）+ conversational 结构与上游不符；同时说明 R105 / R108 对 prompt-only VLM 过严，需在规范中注明例外

### F08 · 版本注释 ≠ 版本约束（note）

- 证据：vLLM 篇：# Tested with trl==0.23.1, peft==0.17.1, math_verify==0.8.0, vllm==0.11.0, trackio==0.5.2 transformers==4.57.0；基础 GRPO 篇：# Tested with transformers==4.47.1, trl==0.14.0, datasets==3.2.0, peft==0.14.0, accelerate==1.2.1, math_verify==0.3.3
- 后果：这些是可补入锁文件的线索，但在补齐前不能声称环境已锁定

### F09 · xLAM 篇连安装命令都是注释，且在运行时动态装包（blocker）

- 证据：安装单元格为 # !uv pip install --upgrade bitsandbytes peft trl python-dotenv（整行被注释）；setup_hardware_config() 内还有 os.system("pip install flash_attn --no-build-isolation")
- 后果：环境完全不可声明，更不可能锁定；页面虽给出 PyTorch 2.8.0+cu128 / H100 NVL 的运行快照，但那只是输出文本

### F10 · code LLM 篇是唯一有真训练种子的一篇，但仍不完整（major）

- 证据：SEED = 0；from transformers import set_seed；set_seed(SEED)；train_data.shuffle(buffer_size=5000, seed=SEED)；ConstantLengthDataset(..., seed=SEED)；并保留前 4000 条为 validation
- 后果：这是 14 篇里最接近可复现的一篇；但 TrainingArguments 未显式传 seed，也没有 deterministic / cudnn / tf32 策略说明，按契约只能算 declared

### F11 · RapidFire 是唯一有 compute_metrics 的一篇，但评估集与训练集完全重叠（major）

- 证据：train_dataset = dataset["train"].select(range(64))；eval_dataset = dataset["train"].select(range(50, 60))；shuffle(seed=42) 只改顺序不改成员；sample_compute_metrics 用 evaluate.load("rouge") 算 rougeL
- 后果：50–59 全部落在 0–63 内，ROUGE-L 数值是训练集内的记忆分数，不是泛化指标；页面也从未给出任何实际 loss / ROUGE-L 数值

### F12 · grounding 篇训练与推理 prompt 模板确实不一致（一手确认）（major）

- 证据：训练 collator：prompts.append(f"<image>Detect {sample['caption']}.") 且额外传 suffix=label_for_paligemma；推理：prompt = f"<image>detect {caption}" 无句号、无 suffix
- 后果：大小写、标点、标签注入三处不同；同时该篇确实有 train_test_split(test_size=0.2, seed=42, shuffle=False) 的 held-out 划分（此前记为「无 held-out」有误），但仍无定量指标

### F13 · MPO 篇是唯一在安装命令中写出精确 pin 的一篇（note）

- 证据：!pip install -U -q git+https://github.com/huggingface/trl.git bitsandbytes qwen-vl-utils==0.0.8；原文自承：需从源码安装 trl，因为 MPO trainer 当时尚未进入 official release
- 后果：qwen-vl-utils 可锁定，但 trl 仍指向 main 分支且无 commit；DPOConfig 里 bf16=False（与其它 VLM 篇相反）

### F14 · VLM DPO 篇 DPOConfig 缺少关键算法参数（major）

- 证据：DPOConfig 全文只给了 batch/epochs/eval/save 等工程项；beta、loss_type、learning_rate、max_length、max_prompt_length 均为「未提及」
- 后果：DPO 的核心超参（beta / loss_type）依赖 TRL 0.12.2 默认值；一旦升级 trl 结果就会变，且无法从 recipe 文本还原

### F15 · 14 篇全部缺 source_sha，无法定位上游 notebook 漂移（blocker）

- 证据：本仓库 14 篇 source_sha 全为空；上游页面「Update on GitHub」链接也不带 commit hash
- 后果：上游 recipe 修改后，我们无法判断本地记录对应的是哪一版，也无法做 diff

## 已纠正的事实错误

上一轮部分配置值来自二手调研总结，未逐篇回上游核对。本次核查后已修正：

| recipe | 字段 | 原值 | 核实后 | 来源 |
| --- | --- | --- | --- | --- |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | `model.base` | Qwen2.5-VL-3B | Qwen/Qwen2.5-VL-3B-Instruct | `local-notebook:fine_tuning_vlm_grpo_trl.txt` |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | `modality_contract.max_prompt_length` | 512 | 2048 | `local-notebook:fine_tuning_vlm_grpo_trl.txt` |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | `modality_contract.images_column / dataset.structure` | images / conversational | image / standard（pre-rendered prompt） | `local-notebook:fine_tuning_vlm_grpo_trl.txt` |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | `[train] num_generations / max_completion_length / learning_rate` | 未记录（曾把 vLLM 篇的 8 / 512 串到此处） | 2 / 1024 / 1e-5 | `local-notebook:fine_tuning_vlm_grpo_trl.txt` |
| `vlm/sft/granite-vision-line-comparison` | `model.base` | granite-vision-3.1-2b | ibm-granite/granite-vision-3.1-2b-preview | `local-notebook:fine_tuning_granite_vision_sft_trl.txt` |
| `vlm/sft/granite-vision-line-comparison` | `dataset.name` | Geoperception / LineComparison | euclid-multimodal/Geoperception | `local-notebook:fine_tuning_granite_vision_sft_trl.txt` |
| `vlm/sft/qwen2-vl-7b-chartqa` | `dataset.split` | train[:10%] | train[:10%] / val[:10%] / test[:10%] | `local-notebook:fine_tuning_vlm_trl.txt` |
| `agent/grpo/wordle-openenv-hf-jobs` | `model.base` | Qwen2.5-0.5B-Instruct | Qwen/Qwen3-0.6B（显存 ≥24GB 时 Qwen/Qwen3-1.7B） | `webfetch:grpo_agent_wordle_hf_jobs` |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | `model.base / dataset.name` | Qwen2-0.5B / NuminaMath-TIR | Qwen/Qwen2-0.5B-Instruct / AI-MO/NuminaMath-TIR | `webfetch:grpo_vllm_online_training` |
| `llm/grpo/qwen2-0-5b-numinamath` | `model.base / dataset.name / GRPOConfig` | Qwen2-0.5B / NuminaMath-TIR / 仅 trl=0.14.0 | Qwen/Qwen2-0.5B-Instruct / AI-MO/NuminaMath-TIR / max_completion_length=64, num_generations=4, grad_accum=16, lr=1e-5 | `webfetch:fine_tuning_llm_grpo_trl` |
| `llm/grpo/qwen2-5-3b-advanced-reward` | `model.base / dataset.name` | Qwen2.5-3B / GSM8K | Qwen/Qwen2.5-3B-Instruct / openai/gsm8k (config=main, split=train) | `webfetch:trl_grpo_reasoning_advanced_reward` |
| `vlm/sft/smolvlm-chartqa` | `peft.lora_alpha / [train] / [dependencies]` | lora_alpha = 16；只记录 bf16、r、alpha；无任何依赖版本 | lora_alpha = 8；SFTConfig: epochs=1, bs=4, grad_accum=4, warmup=50, lr=1e-4, max_length=None；测试版本 trl 0.22.0.dev0 / bnb 0.47.0 / peft 0.17.1 / qwen-vl-utils 0.0.11 / trackio 0.2.8 | `webfetch:fine_tuning_smol_vlm_sft_trl` |
| `vlm/sft/smolvlm-chartqa` | `model.base（撤销推断标记）` | HuggingFaceTB/SmolVLM-Instruct（曾疑为 org 前缀是常识补全） | HuggingFaceTB/SmolVLM-Instruct（一手确认，无误） | `webfetch:fine_tuning_smol_vlm_sft_trl` |
| `vlm/dpo/smolvlm-rlaif-v` | `model.base / [dependencies] / DPOConfig` | SmolVLM-Instruct；仅 trl 0.12.2 + transformers 4.46.3 | HuggingFaceTB/SmolVLM-Instruct；6 个测试版本；DPOConfig: epochs=5, bs=1, grad_accum=32, eval_strategy=steps，beta / loss_type / lr / max_length 上游均未提及 | `webfetch:fine_tuning_vlm_dpo_smolvlm_instruct` |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | `model.base / [train] / [dependencies]` | Qwen2.5-VL-3B；只记录组合损失，无训练超参、无依赖 | Qwen/Qwen2.5-VL-3B-Instruct；DPOConfig: loss_type=['sigmoid','bco_pair','sft'], loss_weights=[0.8,0.2,1.0], bf16=False, epochs=1, bs=4, grad_accum=8；安装命令唯一精确 pin qwen-vl-utils==0.0.8 | `webfetch:fine_tuning_vlm_mpo` |
| `vlm/sft/paligemma2-refcoco-grounding` | `model.base / [train] / known_issues` | PaliGemma2-3b-pt-448；未录入训练配置；称「用训练集样本做首次验证」且无 held-out | google/paligemma2-3b-pt-448；SFTConfig: lr=1e-5, epochs=2, bs=2, grad_accum=4, dataset_kwargs={skip_prepare_dataset:True}；确实有 train_test_split(test_size=0.2, seed=42, shuffle=False) 与 eval_dataset=val_dataset，但无定量指标 | `webfetch:fine_tuning_vlm_object_detection_grounding` |
| `llm/sft/llama3-8b-xlam-function-calling` | `model.base / dataset.name / [train]` | Llama-3-8B / xLAM-60k；未录入训练配置 | meta-llama/Meta-Llama-3-8B-Instruct / Salesforce/xlam-function-calling-60k；TrainingConfig 默认值 lr=1e-4, bs=16, grad_accum=8, max_steps=1000, max_seq_length=2048, lora_r=16, lora_alpha=16；SFTConfig(max_length=2048, optim=adamw_8bit) | `webfetch:function_calling_fine_tuning_llms_on_xlam` |
| `llm/sft/llama3-8b-xlam-function-calling` | `[method] / [train] / [evaluation] / 数据协议` | method.trainer=SFTTrainer；[train] 为 TrainingConfig dataclass 默认值 max_steps=1000/max_seq_length=2048；evaluation 只有 function_call_exact_match + argument_f1；无 held-out 定量评估 | [method] 拆分 trainer=Trainer（本仓执行层）与 upstream_trainer=SFTTrainer（一手事实）；上游默认值移入 [upstream] 元信息段，[train] 为本仓自建 full=150/smoke=5 与 completion-only 参数；[evaluation] 增 max_new_tokens=256 与 json_parse_rate/schema_valid_rate/tool_call_exact_match/argument_f1 四个 held-out 指标；新增 [xlam]/[peft]/[quantization] 自建协议段并声明 gated/research-only 许可边界 | `webfetch:function_calling_fine_tuning_llms_on_xlam + webfetch:Salesforce/xlam-function-calling-60k` |
| `llm/sft/starcoder1b-fim` | `model.base / dataset.name / [train] / seed` | StarCoderBase-1B / hf-stack-v1；未录入训练配置；假定无 seed | bigcode/starcoderbase-1b / smangrul/hf-stack-v1；SEED=0 + set_seed(SEED)；lr=5e-4, bs=16, max_steps=2000, seq_length=2048, FIM_RATE=0.5；上游用的是 transformers.Trainer 而非 SFTTrainer | `webfetch:fine_tuning_code_llm_on_single_gpu` |
| `llm/sft/rapidfire-multiconfig` | `dataset.name / [train] / evaluation / seed` | Bitext 客服数据；未录入配置网格；称「上游未报告 held-out 定量对照」 | bitext/Bitext-customer-support-llm-chatbot-training-dataset；4 组网格 lr=5e-4/2e-4 × r=8/32；有 sample_compute_metrics 计算 ROUGE-L；run_fit(..., seed=42)；但 eval_dataset=range(50,60) 落在 train_dataset=range(64) 内 | `webfetch:rapidfire_sft_multiconfig_training` |

其中最严重的是 `agent/grpo/wordle-openenv-hf-jobs` 的 `model.base`：上一轮写入的 `Qwen2.5-0.5B-Instruct` **没有任何一手依据，属于编造**；上游实际按显存自动选择 `Qwen/Qwen3-0.6B` 或 `Qwen/Qwen3-1.7B`。

## 推断值处置

| recipe | 字段 | 值 | 为什么不可信 | 处置 |
| --- | --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | `dependencies.trl` | `1.7.0` | 上游安装命令 pip install -q trl ... 无任何版本约束；1.7.0 来自调研报告对 TRL v1 的描述，不是该 recipe 的实测版本 | 移除该依赖声明，改列入 todo；同时从 [[compat.matrix]] 删除以该 recipe 为 source_recipe 的伪精确组合 |
| `agent/grpo/wordle-openenv-hf-jobs` | `dependencies.transformers` | `5.3.0` | 上游为 'transformers>=5.3.0' 浮动约束；5.3.0 是下界，从未被核实为实际运行版本 | 移除精确 pin，改列入 todo |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | `vllm.gpu_memory_utilization` | `0.3` | 该值只出现在 GRPOConfig 的注释行 # vllm_gpu_memory_utilization=0.3（默认值说明），未作为显式实参传入 | 保留但改名为 default_note，不再当作本 recipe 的显式配置 |
| `vlm/sft/qwen2-vl-7b-chartqa` | `evaluation.seed` | `42` | 上游 SFTConfig 无 seed；42 是本仓库要求的评估协议，不是上游已有事实 | 保留但改记为 repro.seed_gap 的说明，明确区分「协议」与「已实现」 |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | `rewards.failure_path_note` | `解析失败返回 None 而非 0` | 本地原文 accuracy_reward 用 try/except 令 gold_parsed=[] 并走归一化文本回退分支，完整分支未逐行复核 | 改写为已核实部分 + 明确标注待复核分支 |
| `llm/sft/llama3-8b-xlam-function-calling` | `training_config 默认值` | `lr=1e-4, bs=16, max_steps=1000` | 这些是 dataclass 默认值，页面未展示实际运行时的 create_training_config(...) 调用参数，无法确认是否被覆盖 | 录入为 defaults 并标注 derived-from-dataclass，不声称是实测超参 |
| `vlm/sft/smolvlm-chartqa` | `hardware.vram_gb` | `24` | 上游只写 'tested using a L4 GPU'，24GB 是按 L4 常见规格补全，不是原文数值 | 改为 accelerator = L4 并删除推断的显存数值 |

## 补齐路径

按依赖顺序，每一步都能用 `recipe-lab repro` 复查等级是否真的推进：

| 阶段 | 动作 | 解锁维度 | 目标等级 |
| --- | --- | --- | --- |
| P0 溯源 | 为每篇锁定 `source_sha`，记录抓取时的上游 commit | provenance | — |
| P1 锁定输入 | 记录 `model.revision` / `dataset.revision`，写权重与数据校验和 | model, data | — |
| P2 重建环境 | 把 `-U / git+` 安装改成 `requirements.lock`（含 git commit），声明 Python/CUDA | env | — |
| P3 确定性 | 在 trainer config 显式设 `seed` / `data_seed` / deterministic 策略 | seed | **L2** |
| P4 可执行 | 为每篇写 `train.py` + `eval.py`，给出确定命令与 evidence 产物 | code, eval | L3 |
| P5 实跑 | 在声明硬件上重跑，记录指标与容差，人工确认后方可标 L4 | code, eval | L4 |

建议先拿 `llm/grpo/qwen2-0-5b-numinamath` 做端到端样板：它是 0.5B 模型、参数与测试版本已按上游原文补齐，改动面最小。样板跑通后再把同一套锁文件、入口、评估脚本模板推广到其余 13 篇。

## 本机无法确认的部分

- **没有 GPU**，无法执行任何真实训练，因此不存在 L4 的证据产物。
- 一手材料覆盖 14/14，但只有 3 篇拿到了本地 `.ipynb` 原文；其余 11 篇只有网页事实抽取，未逐字节校验代码单元格，也没有抓取时的上游 commit。
- 无。14 篇全部回一手材料核对；但有 11 篇仅有网页抽取而非本地 .ipynb 原文，仍未逐字节校验代码单元格。
- `baseline.result` 与 `evaluation.metrics` 是声明的协议，不是实测数值；验证器只检查它们与 `status` 自洽，不重新跑实验。
- `# Tested with trl==0.14.0` 这类注释只是作者当时的环境快照，本仓库把它登记为 `dependencies` 里的「测试版本」，但 `repro.env` 仍判为 `declared` / `missing`，因为没有 `requirements.lock`。
- 即使 P0–P4 全部完成，仍需真实硬件重跑才能确认数值层面的可复现性。

---

判定标准：[registry/reproducibility.toml](../registry/reproducibility.toml) · 证据登记：[registry/evidence.toml](../registry/evidence.toml) · 规则：[docs/VERIFICATION.md](VERIFICATION.md) · 纠错脚本：[scripts/apply_repro_audit.py](../scripts/apply_repro_audit.py)
