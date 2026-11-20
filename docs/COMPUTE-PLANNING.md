# 垂类后训练：数据规模阶梯与算力规划

> **性质**：规划表，**不是 GPU 实跑证据**。仓库当前 GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14，本文不改变这一点，也不能当作 B3 业务档验收依据。
> **事实源**：[`registry/compute_planning.toml`](../registry/compute_planning.toml)（手写，非生成物）；与 14 篇 `recipe.toml`、`methodology.toml` 及本页表格的一致性由 `tests/test_compute_planning.py` 强校验。
> **三类口径**：**实测锚点**（上游 recipe / 论文自带数字）、**由 recipe.toml 直接读出**、**估算**（工程经验，均标「估算 / 非实测」）。上线前必须用 5 步 smoke 校准。

## 1. 垂类数据规模阶梯（一步步放大）与 P0–P3 决策门

区间是启发式（不是论文硬阈值）；方向性依据列在最后一列。`rl_prompts` 指 GRPO 训练用到的不同 prompt 数。

| 阶梯 | 目的 | SFT 样本 | 偏好对（DPO/MPO） | RL prompts | 独立 seed | 步数提示 | 依据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 冒烟 | 只验证管线可跑、日志 / checkpoint / 导出链路通；不看指标 | 8–32 | 8–32 | 8–32 | 1 | 5 步（仓库 smoke_steps=5） | 仓库 budget.smoke_steps = 5；Wordle smoke_n_prompts = 30 |
| S1 试点 | 看 loss / reward 是否下降，吞吐校准（得到 s/步、s/rollout） | 100–1k | 500–2k | 200–1k | 1 | 30–150 步 | LIMA 1k 样本对齐（arXiv 2305.11206）；Wordle 校准 40 rollouts / 92.4 s → 2.31 s/rollout |
| S2 最小可用 | 垂类行为稳定出现；做去重 / 留出集 / 污染检查后再看指标 | 1k–10k | 2k–10k | 1k–3k | 1–2 | 150–500 步 | 仓库 GRPO full_steps=150 × 8–16 prompts/step ≈ 1.2k–2.4k prompts；Wordle full_n_prompts = 3000；去重 arXiv 2107.06499 |
| S3 基线级 | 可对外比较：≥3 独立 seed、Student-t95、零训练基线对照 | 10k–50k | 10k–50k | 3k–10k | ≥3 | 500–2000 步 × 3 seed | 仓库发布门禁：≥3 独立训练 seed；欠指定 arXiv 2011.03395；奖励过优化 arXiv 2210.10760 |
| S4 放大 | 覆盖长尾与分布外；需要数据治理、版本冻结与持续评测 | 50k–500k+ | 50k+ | 10k+ | ≥3 | 2000+ 步，常需多卡 | 无统一论文阈值；以 S3 的留出集收益曲线是否仍上升决定是否放大 |

### P0–P3 阶段决策门（先测零训练基线，再逐级加量）

| 阶段 | 做法 | 退出 / 停止门 |
| --- | --- | --- |
| P0 零训练基线 | 固定 held-out 与定量指标，先跑 prompting / RAG / 结构化生成 / 更大基座 | 若 P0 已达标（如 ChartQA 零训练即过），按 R302 终止微调，不烧训练算力 |
| P1 最小 SFT | 1B–3B 小模型 + 约 10% 数据（S1–S2 量级），QLoRA r=8，1 epoch，确认数据契约与 completion-only mask | 相对 P0 有可测净增量且无泄漏；缺格式/领域知识时优先补数据清洗而非加卡 |
| P2 规模化 SFT / 偏好 | 全量数据（S3）、更大基座（7B–32B）、顺序/并发超参网格、≥3 seed 与 Student-t95 | 画出成本/收益曲线；有 chosen/rejected 偏好且 SFT 偏移时再进 DPO/MPO |
| P3 可验证 RL（可选） | 先写 reward 失败路径单测，单卡 smoke 校准 rollout 吞吐与 vLLM colocate，再扩至 server/多卡 | 监控 reward 曲线与独立 verifier 指标（如 reward_hacking_rate）；两者背离立即停止 |

放大纪律：S2→S3 之前必须完成去重、留出集与污染检查；S3 起发布级对比要求 ≥3 个独立训练 seed（Student-t95）。放大前看留出集收益曲线是否仍在上升，而不是按预算直接铺满。

## 2. 14 篇 recipe 的完整规模与超参事实（直接读自 recipe.toml）

### 2a. Batch、步数、数据集规模与硬件声明

有效 batch = `micro_batch × grad_accum × 卡数`（本仓 14 篇均为单卡设计，无多卡执行证据）。
GRPO 的 batch 按 completions 计数：每 step 的 prompt 数 = `有效 batch ÷ num_generations`；`steps` 列 `cap` 表示未设 `max_steps`，取 `budget.full_steps` 预算上限；`max` 表示 `train.max_steps`。缺失字段显示 `—`，不凭记忆补。`params_b` 带 `*` 为估算（recipe 未声明）。

| recipe | 方法 | 参数量(B) | 数据集 | 数据规模 / 切分事实 | micro | accum | 有效 batch | G | prompts/step | steps | 已见样本(有效batch×steps) | 声明硬件 | 时间锚点 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | grpo | 0.6* | `OpenEnv Wordle（在线生成，无静态数据集）` | 在线生成：smoke 30 / full 3000 prompts；eval 固定 20 局 | 1 | 64 | 64 | — | — | 150 (cap) | 9600 | 1× A100 80GB（上游 est_full_gpu_hours 6.2） | 6.2 h（上游预估；2.31 s/rollout 可复算） |
| `llm/grpo/qwen2-0-5b-numinamath` | grpo | 0.5 | `AI-MO/NuminaMath-TIR` | 固定连续切分：500 train / 500 val / 500 test（共 1500） | 4 | 16 | 64 | 4 | 16 | 150 (cap) | 9600 | 待实跑确认（按模型规模与 batch 估） | — |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | grpo | 0.5 | `AI-MO/NuminaMath-TIR` | 与 NuminaMath 篇共享 held-out 索引（上游用 train 10%） | — | — | — | 8 | — | 150 (cap) | — | 待实跑确认（按模型规模与 batch 估） | 0.04–0.07 it/s（512 token，上游） |
| `llm/grpo/qwen2-5-3b-advanced-reward` | grpo | 3.0 | `openai/gsm8k` | 官方 7473 train（90/10 hash 切）+ 1319 test 全量 held-out | 4 | 8 | 32 | 4 | 8 | 150 (cap) | 4800 | 1× A100 80GB（3B GRPO） | — |
| `llm/sft/llama3-8b-xlam-function-calling` | qlora_sft | 8.0 | `Salesforce/xlam-function-calling-60k` | 60k（2024-07 重生成 1896 条）；80/10/10 切，test ≤300 | 16 | 8 | 128 | — | — | 150 (max) | 19200 | 1× H100 80GB（8B 4bit，上游快照 H100 NVL） | — |
| `llm/sft/rapidfire-multiconfig` | lora_bf16 | 0.124 | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | 数据卡 26872 行（上游仅取 64 行）；本仓 80/10/10，test ≤300 | 2 | 2 | 4 | — | — | 64 (max) | 256 | 待实跑确认（gpt2 124M fp16 + LoRA，按 batch 与 max_steps 估） | — |
| `llm/sft/starcoder1b-fim` | qlora_sft | 1.0 | `smangrul/hf-stack-v1` | 数据卡 5905 行（本仓上限 6000，repo hash 80/10/10） | 16 | 1 | 16 | — | — | 2000 (max) | 32000 | 待实跑确认（按模型规模与 batch 估） | — |
| `vlm/dpo/smolvlm-rlaif-v` | dpo_qlora | 2.25* | `HuggingFaceH4/rlaif-v_formatted` | 上游 train[:6%]/test[:1%]；本仓图像 hash 80/10/10 | 1 | 32 | 32 | — | — | 150 (cap) | 4800 | 待实跑确认（按模型规模与 batch 估） | — |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | grpo | 3.8* | `lmms-lab/multimodal-open-r1-8k-verified` | 8k 规模（上游取 5%）；本仓固定末 500 条 held-out | 2 | — | — | 2 | — | 150 (cap) | — | 1× A100 80GB（3B VLM GRPO） | — |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | dpo_qlora | 3.8* | `HuggingFaceH4/rlaif-v_formatted` | 上游 train[:5%]/test[:1%]；与 DPO 共享图像 hash 80/10/10 | 4 | 8 | 32 | — | — | 200 (cap) | 6400 | 1× A100 80GB（3B VLM） | — |
| `vlm/sft/granite-vision-line-comparison` | qlora_sft | 2.0 | `euclid-multimodal/Geoperception` | LineComparison 子集；本仓图像 hash 80/10/10 | 4 | 2 | 8 | — | — | 150 (max) | 1200 | 待实跑确认（上游仅称 tested using a A100 GPU，未给显存数值） | — |
| `vlm/sft/paligemma2-refcoco-grounding` | lora_bf16 | 3.0 | `jxu124/refcoco` | 上游 train[:5%]（Flickr URL）；本仓图像 hash 80/10/10 | 2 | 4 | 8 | — | — | 150 (max) | 1200 | 待实跑确认（上游称 Colab A100，未给显存数值） | — |
| `vlm/sft/qwen2-vl-7b-chartqa` | qlora_sft | 7.0 | `HuggingFaceM4/ChartQA` | 上游 train[:10%]/val[:10%]/test[:10%]（原理阻塞：零训练达标） | 4 | 8 | 32 | — | — | 150 (cap) | 4800 | 1× A100 80GB（7B QLoRA，上游 est_gpu_hours 6.0） | 6.0 h（上游预估，非实测） |
| `vlm/sft/smolvlm-chartqa` | qlora_sft | 2.0 | `HuggingFaceM4/ChartQA` | ChartQA（原理阻塞：基座预训练已接触同集） | 4 | 4 | 16 | — | — | 200 (cap) | 3200 | 待实跑确认（按模型规模与 batch 估） | — |

### 2b. 显存敏感超参（精度 / 量化 / PEFT / 序列长 / 梯度检查点 / 优化器）

这部分字段直接决定同样参数量下「单卡 24GB 能否跑起」还是「必须上 80GB / 多卡」。全部读自 `recipe.toml`，缺失显示 `—`。

| recipe | 精度 | 量化 | PEFT 规格 | 序列 / 长度限制 | grad_ckpt | 优化器 | 学习率 | 发布套件总轮数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | unspecified | unspecified | full / 未声明 | 多轮整局（上游约 1024） | — | — | — | 4（1×3 + 1 zero-shot） |
| `llm/grpo/qwen2-0-5b-numinamath` | unspecified | unspecified | full（无 PEFT） | comp=64 | — | — | 1e-5 | 4（1×3 + 1 zero-shot） |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | unspecified | unspecified | full / 未声明 | prompt=512 / comp=512 | — | — | — | 4（1×3 + 1 zero-shot） |
| `llm/grpo/qwen2-5-3b-advanced-reward` | bf16 | unspecified | full（fp32 权重 + bf16 autocast） | prompt=512 / comp=256 | 否 | — | 5e-6 | 16（5×3 + 1 zero-shot） |
| `llm/sft/llama3-8b-xlam-function-calling` | bf16 | nf4_4bit_double | qlora r=16 / α=16 | seq=2048 | 是 | adamw_8bit | 1e-4 | 4（1×3 + 1 zero-shot） |
| `llm/sft/rapidfire-multiconfig` | fp16 | unspecified | lora r=8/32 / α=16/64（4 配置） | seq=512 | 是 | adamw_torch | 5e-4 | 13（4×3 + 1 zero-shot） |
| `llm/sft/starcoder1b-fim` | bf16 | nf4_4bit_double | qlora r=8 / α=32 | seq=2048 | 是 | adamw_torch | 5e-4 | 4（1×3 + 1 zero-shot） |
| `vlm/dpo/smolvlm-rlaif-v` | bf16 | nf4_4bit_double | qlora-dora r=8 / α=8 | seq=512（mc max_length=none） | 是 | — | 5e-6 | 4（1×3 + 1 zero-shot） |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | unspecified | unspecified | full / 未声明 | prompt=2048 / comp=1024 | — | — | 1e-5 | 4（1×3 + 1 zero-shot） |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | bf16_false_mismatch | nf4_4bit_double | qlora-dora r=8 / α=8 | seq=1024（mc max_length=none） | 是 | — | 5e-6 | 10（3×3 + 1 zero-shot） |
| `vlm/sft/granite-vision-line-comparison` | bf16 | nf4_4bit_double_skip_vision | qlora-dora r=8 / α=8 | seq=512（mc max_length=none） | 是 | adamw_torch_fused | 1e-4 | 4（1×3 + 1 zero-shot） |
| `vlm/sft/paligemma2-refcoco-grounding` | bf16 | none | lora r=8 / α=16 | seq=512（mc max_length=none） | 否 | adamw_torch_fused | 1e-5 | 4（1×3 + 1 zero-shot） |
| `vlm/sft/qwen2-vl-7b-chartqa` | bf16 | nf4_4bit_double | qlora r=8 / α=16 | mc max_length=none | 否 | — | 2e-4 | 4（1×3 + 1 zero-shot） |
| `vlm/sft/smolvlm-chartqa` | bf16 | nf4_4bit_double | qlora-dora r=8 / α=8 | mc max_length=none | — | — | 1e-4 | 4（1×3 + 1 zero-shot） |

读表要点：

- **已见样本 = 有效 batch × steps** 是配置上限，不等于数据集行数；GRPO 单位是 completions，SFT/DPO 单位是样本/偏好对。
- **同 8B 不同命**：`llama3-8b-xlam-function-calling` 用 4bit QLoRA 权重只占 ~6GB，但 `seq=2048`、`micro=16`、`有效 batch=128`，激活显存极大，故声明 `1× H100 80GB`；若在 24GB/48GB 卡跑，须同步压低 `micro` 并补足 `grad_accum`。
- **3B 全参 GRPO**：`qwen2-5-3b-advanced-reward` 为全参 fp32 权重 + bf16 autocast、`gradient_checkpointing=False`，直接声明 `1× A100 80GB`。
- **VLM 显存不可按文本估算**：VLM 均设 `max_length=none`（或 `max_prompt_length=2048`），高分辨率图像会带来额外视觉 token 激活。
- **MPO 精度矛盾**：`qwen2-5-vl-3b-mpo` 上游 `bf16=False` 与 4bit `compute_dtype=bfloat16` 矛盾，本仓如实登记，实跑时需确认实际算子精度。

## 3. 发布级算力乘法表（从单次试跑到 ≥3 seed 对照）

单次能跑通 ≠ 可以出结论。本仓发布级门禁要求 ≥3 个独立训练 seed + 冻结基座 zero-shot，多配置/消融套件会把 GPU 预算放大 3–15 倍训练量（4–16 次完整运行）：

| 套件形态 | 配置/消融数 | 每配置 seed | zero-shot | 总运行次数 | 算力倍数（相对单次训练） | 说明 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 标准单配置发布套件（Math / FIM / xLAM / Granite / PaliGemma / SmolVLM DPO） | 1 | 3 | 1 | 4 | 3× 训练 + 4× 评测 | 1 次冻结基座 zero-shot + 3 个独立训练 seed（含各自 checkpoint held-out 评测） |
| MPO 三损失消融套件（qwen25vl-mpo-v1） | 3 | 3 | 1 | 10 | 9× 训练 + 10× 评测 | full / sigmoid_only / no_sft 3 个消融 × 3 seed + 1 次 zero-shot（同一 Qwen2.5-VL backbone） |
| RapidFire 4 配置网格套件（rapidfire-sft-v1） | 4 | 3 | 1 | 13 | 12× 训练 + 13× 评测（validation 选配置 + test 单次验收） | 顺序执行 4 配置 × 3 seed；不宣称复现 rapidfireai 16–24× 并发吞吐 |
| GSM8K 5 drop-one reward 消融套件（gsm8k-reward-v1） | 5 | 3 | 1 | 16 | 15× 训练 + 16× 评测 | full / no_format_exact / no_format_approx / no_correctness / no_extraction 5 消融 × 3 seed + 1 次 zero-shot |

## 4. 模型规模 × 方法：显存、单卡档位与多卡起点（估算，非实测）

推荐显存 = 权重与状态估算 × headroom 1.4（覆盖激活、KV、碎片）。单卡档位 [24, 40, 48, 80] GB，放不下则给多卡下限 = `ceil(推荐显存 ÷ 80GB)`，假设 ZeRO-3 / FSDP 均分，实际需另加 1 卡留通信缓冲与 AllGather 峰值。
长序列、大 micro-batch、VLM 图像 token、vLLM 引擎显存都会突破 headroom；**估算只作选卡起点，必须 5 步 smoke 复核**。

### 4a. 显存估算系数（估算）

| 方法 id | 说明 | 权重+状态 GB = coef×P(B)+const | 依据 |
| --- | --- | --- | --- |
| `qlora_sft` | QLoRA SFT（4bit 底座 + LoRA/DoRA） | 0.65×P+0.8 | QLoRA 论文（2305.14314）：65B 单卡 48GB 可微调（0.65×65+0.8≈43GB 权重，余量给激活） |
| `lora_bf16` | LoRA SFT（bf16/fp16 2-byte 底座） | 2.0×P+1.0 | 估算 |
| `full_sft` | 全参 SFT（bf16 + AdamW fp32 状态） | 16.0×P+0.0 | 2P 权重 + 2P 梯度 + 12P 优化器（含 fp32 master / m / v） |
| `dpo_qlora` | DPO / MPO（QLoRA，policy + 冻结 ref） | 1.3×P+0.8 | 估算 |
| `grpo_lora` | GRPO（LoRA，bf16；不含 rollout 引擎） | 2.0×P+1.0 | 估算 |
| `grpo_full` | GRPO（全参，bf16 + AdamW） | 18.0×P+0.0 | 全参 SFT 16P + 冻结参考模型 2P（对应 Qwen2.5-3B GSM8K 全参 GRPO 声明 1× A100 80GB） |

### 4b. 模型 × 方法 显存与卡数矩阵（单元格 = 推荐显存→卡数，估算）

| 模型 | qlora_sft | lora_bf16 | full_sft | dpo_qlora | grpo_lora | grpo_full |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5B | 2GB→24GB×1 | 3GB→24GB×1 | 11GB→24GB×1 | 2GB→24GB×1 | 3GB→24GB×1 | 13GB→24GB×1 |
| 1.7B | 3GB→24GB×1 | 6GB→24GB×1 | 38GB→40GB×1 | 4GB→24GB×1 | 6GB→24GB×1 | 43GB→48GB×1 |
| 3B | 4GB→24GB×1 | 10GB→24GB×1 | 67GB→80GB×1 | 7GB→24GB×1 | 10GB→24GB×1 | 76GB→80GB×1 |
| 7B | 7GB→24GB×1 | 21GB→24GB×1 | 157GB→2×80GB | 14GB→24GB×1 | 21GB→24GB×1 | 176GB→3×80GB |
| 8B | 8GB→24GB×1 | 24GB→24GB×1 | 179GB→3×80GB | 16GB→24GB×1 | 24GB→24GB×1 | 202GB→3×80GB |
| 14B | 14GB→24GB×1 | 41GB→48GB×1 | 314GB→4×80GB | 27GB→40GB×1 | 41GB→48GB×1 | 353GB→5×80GB |
| 32B | 30GB→40GB×1 | 91GB→2×80GB | 717GB→9×80GB | 59GB→80GB×1 | 91GB→2×80GB | 806GB→11×80GB |
| 70B | 65GB→80GB×1 | 197GB→3×80GB | 1568GB→20×80GB | 129GB→2×80GB | 197GB→3×80GB | 1764GB→23×80GB |

### 4c. 常见硬件规格起点速查（估算，非实测）

| 硬件规格 | QLoRA SFT 上限 | bf16 LoRA 上限 | 全参 SFT 上限 | GRPO 起跑规模 | 建议 micro_batch 起点 |
| --- | --- | --- | --- | --- | --- |
| 1× 24GB（RTX 4090 / L4 / A10G） | 14B（长序列/VLM 建议 ≤7B） | 7B（紧凑 seq_length） | 1B | 0.5B–1.7B（Wordle 阈值：VRAM ≥24GB 选 Qwen3-1.7B，否则 0.6B） | 1–4（配 grad_accum 8–32） |
| 1× 40GB–48GB（A100 40G / L40S / A6000） | 32B（48GB 曾跑通 QLoRA 65B 论文锚点，micro=1） | 14B | 1.7B–2B | 3B QLoRA / 1.7B 全参 | 2–8 |
| 1× 80GB（A100 80G / H100 80G） | 70B（论文/公式估算可达，大 micro 建议 32B 内） | 24B–32B（偏紧） | 3B（含 3B 全参 GRPO） | 3B 全参 / 7B–14B QLoRA | 4–16（8B QLoRA xlam 用 micro=16） |
| 2× 80GB（160GB 总显存） | 70B（含 DPO/MPO 70B QLoRA，双卡留激活余量） | 32B | 7B（ZeRO-2/3） | 7B–14B（1 卡 vLLM server + 1 卡训练，或 2 卡 ZeRO-2） | 每卡 2–8（减半 grad_accum 保有效 batch） |
| 4× 80GB（320GB 总显存） | 70B（大 batch / 长序列 / VLM 高分辨率） | 70B（ZeRO-3） | 14B（ZeRO-3） | 14B–32B（2 卡 vLLM + 2 卡训练或 ZeRO-3） | 每卡 2–8 |
| 8× 80GB（640GB 总显存） | 70B+（高吞吐大序列） | 70B（含 VLM 高分辨率） | 32B（ZeRO-3；70B 全参需 ~16–20×80GB 起） | 32B（分离 vLLM server 卡池 + 训练卡池） | 每卡 1–4（随 world_size=8 调整 accum） |

### 4d. 多卡并行策略与跨卡 Batch 换算规则（估算）

| 并行策略 | 适用场景 | 显存效果 | 有效 Batch 换算规则 |
| --- | --- | --- | --- |
| DDP（纯数据并行） | 单卡放得下完整模型 + 优化器（如 7B QLoRA 多卡扩吞吐） | 不切分权重与优化器；单卡显存几乎不降，靠减小 per-card grad_accum 换吞吐 | 保持 micro_batch × grad_accum × 卡数 = 目标有效 batch |
| DeepSpeed ZeRO-2 / FSDP SHARD_GRAD_OP | 3B–14B LoRA/全参：梯度与优化器显存成为瓶颈，但单卡能放下权重 | 优化器与梯度按卡数均分，通信开销显著低于 ZeRO-3 | 同步下调 grad_accum；保持有效 batch 与学习率不变 |
| DeepSpeed ZeRO-3 / FSDP FULL_SHARD | 14B–70B 全参或 70B bf16 LoRA（单卡权重+状态已超 80GB） | 权重 / 梯度 / 优化器全分片；下限 = ceil(推荐显存÷80GB)，建议 +1 卡留通信与 AllGather 峰值余量 | 尽量维持 micro_batch ≥ 1 并开启 gradient_checkpointing + bf16 |
| GRPO + vLLM colocate | 单卡或小规模（≤3B–7B QLoRA），生成与训练分时复用同一 GPU | 必须给 vLLM 留出 gpu_memory_utilization（如 0.3）；长 completion 吞吐提升 +75%–+100% | micro_batch 必须整除 num_generations；每步 prompt 数 = 有效 batch ÷ num_generations |
| GRPO + vLLM server（trl vllm-serve） | 多卡 / 7B–70B 在线 RL，生成 GPU 与训练 GPU 物理隔离 | 独立推理卡满载 KV cache，训练卡不再被 vLLM 占显存，减少生成/训练切换空闲 | 注意 vllm_importance_sampling_cap 等异步/策略偏移保护；先单卡 smoke 验证 reward 再切多卡 |

## 5. 论文与 Cookbook 一手实测锚点

### 5a. 上游 recipe / 论文实测与预估锚点

| 锚点 | recipe | 数值 | 推导 / 说明 | 是否实测 |
| --- | --- | --- | --- | --- |
| `wordle_s_per_rollout` | `agent/grpo/wordle-openenv-hf-jobs` | 2.31 s/rollout（40 rollouts / 92.4 s 校准） | 150 步 × 64 rollouts/步 × 2.31 s ≈ 6.16 h，与上游 est_full_gpu_hours 6.2 吻合 | 上游实测 |
| `vllm_colocate_its` | `llm/grpo/qwen2-0-5b-vllm-colocate` | max_completion_length=512：0.04 → 0.07 it/s（+75%）；1024：0.02 → 0.04 it/s（+100%） | 瓶颈是 rollout 生成而非梯度；上游只报 it/s，未报显存 / pass@1 | 上游实测 |
| `chartqa_7b_hours` | `vlm/sft/qwen2-vl-7b-chartqa` | est_gpu_hours = 6.0（1× A100 80GB，7B QLoRA） | 上游预估值，不是本仓实测 | 上游预估，非实测 |
| `qlora_65b_48gb` | — | QLoRA 论文：65B 在单张 48GB 上微调 | 用于校准 qlora_sft 规则 | 上游实测 |

Wordle `max_steps_within(budget, 2.31, grad_accum=64, headroom=0.75)` 超时反推（测试 `test_wordle_budget_projection_table` 校验公式一致）：

| 作业预算（秒） | 上游 Flavor 提示 | 安全系数 | 最大可行步数 | 能否跑完 full 150 步 |
| ---: | --- | ---: | ---: | --- |
| 3600 | a10g-small（smoke 1h） | 0.75 | 18 | 不可（需拆段或扩时长） |
| 10800 | a10g-small（full 3h） | 0.75 | 54 | 不可（需拆段或扩时长） |
| 21600 | a100-large（vLLM 6h） | 0.75 | 109 | 不可（需拆段或扩时长） |
| 43200 | 12h 作业窗口（可完整跑 150 步） | 0.75 | 219 | 可 |

### 5b. vLLM colocate 单卡吞吐对照（Cookbook 原文，`max_prompt_length=128` 固定）

规律：生成越长 vLLM 相对加速越明显；`num_generations` 对 it/s 与显存影响较小，实践常设 `>= 8`；colocate 默认注释参考 `gpu_memory_utilization=0.3`，且与 IPython/Jupyter CUDA context 冲突，须脚本式运行。

| max_completion_length | num_generations | vLLM on (it/s) | vLLM off (it/s) | 相对提升 | recipe.toml 是否登记 |
| ---: | --- | ---: | ---: | --- | --- |
| 64 | 4 / 8 / 16 | 0.14 | 0.12 | +16.7% | 调研报告补充（原文表） |
| 128 | 8 / 16 | 0.13 | 0.09 | +44.4% | 调研报告补充（原文表） |
| 256 | 8 / 16 | 0.10 | 0.06 | +66.7% | 调研报告补充（原文表） |
| 512 | 8 / 16 | 0.07 | 0.04 | +75% | 已登记 |
| 1024 | 16 / 32 | 0.04 | 0.02 | +100% | 已登记 |

### 5c. 已核查论文对数据/显存/算力规划的约束（绑定 `registry/methodology.toml`）

所有 arXiv 编号与 `registry/methodology.toml` 中 `verified=true` 论文交叉校验，不凭记忆引用；论文自身基准分数绝不充当本仓 expected/tolerance。

| 论文 id | arXiv | 规划主题 | 对规模 / 显存 / 预算的实际约束 |
| --- | --- | --- | --- |
| `lima` | `2305.11206` | 最小高质量样本量 | 1,000 条精选指令在 65B 上完成标准 SFT 对齐，支撑 S1/S2 先质量后放大；不证明小模型客服/域任务只需 1k |
| `dedup_training_data` | `2107.06499` | 去重与污染拦截 | 近重复造成记忆与 val >4% 重叠，去重后达同精度所需步数更少；S2→S3 放大前必过按图/题/repo hash 去重 |
| `underspecification` | `2011.03395` | 多 seed 预算乘法 | 同 held-out 精度可对应不同部署表现；S3 起算力必须按 ≥3 个独立训练 seed × 配置数规划 |
| `reward_overopt` | `2210.10760` | RL 步数上限 | 代理 reward 过度优化会损害真实指标；RL 不宜盲目堆 steps，须监控独立 verifier 与 reward 背离 |
| `qlora` | `2305.14314` | 单卡显存锚点 | NF4 + 双重量化 + paged optimizers 支持单张 48GB 微调 65B；同时提示定性聊天基准不可信 |
| `lora` | `2106.09685` | 参数与显存压缩 | 低秩冻结减少优化器状态；但 GPT-3 175B 的 3× 显存降幅不线性套用到 0.5B–8B，微批激活仍可能撑爆显存 |
| `dora` | `2402.09353` | DoRA 稳定性 | 分解幅度与方向改善低秩学习；推理零额外延迟，但训练期相对 LoRA 多少量参数与前向开销 |
| `dpo` | `2305.18290` | 偏好训练双模型显存 | 省去 PPO critic 与在线采样，但训练前向需 policy + 冻结 ref，显存规则系数约为 QLoRA SFT 的 2× |
| `rlaif_v` | `2405.17220` | AI 反馈偏好规模 | 开源 MLLM 反馈可用于偏好构建；但跨基座（SmolVLM vs Qwen2.5-VL）不能把指标差归因到算法 |
| `deepseekmath_grpo` | `2402.03300` | GRPO 无 critic 内存 | 去掉 PPO value/critic 省下显存；实际瓶颈转移到每步 num_generations 次在线自回归生成 |
| `vllm_pagedattention` | `2309.06180` | KV 显存与 rollout 引擎 | PagedAttention 提升长生成吞吐；colocate 下须从训练显存划拨 gpu_memory_utilization |
| `qwen2vl` | `2409.12191` | 动态分辨率视觉 token | VLM 图像 token 数随分辨率变化，SFT/DPO 须设 max_length=None，大图会显著推高激活显存 |
| `qwen25vl` | `2502.13923` | 原生空间分辨率感知 | 同上：视觉推理 max_prompt_length 需给足 2048 防止截断视觉 token，显存估算不能只看文本长度 |

## 6. 训练时间怎么估（5 步 Smoke 校准法）

不同卡型、序列长度、图像分辨率和 KV cache 占用的吞吐差异极大，本表不编造「某模型固定几秒/步」，统一用 S0/S1 实测吞吐线性外推：

1. **RL（GRPO）**：先跑 5–40 次 rollout，测 `sec_per_rollout`（如 Wordle 92.4 s / 40 = 2.31 s），再按 `总时长 ≈ steps × (有效 batch) × sec_per_rollout` 估算；加 `headroom=0.75` 反推作业超时上限。
2. **SFT / DPO / MPO**：跑 5 步 smoke，取稳态 `samples/s` 或 `tokens/s` 与峰值显存；`单次时长 ≈ (有效 batch × steps) ÷ samples/s`。
3. **乘上发布级倍数**：单次时长 × seed 数（≥3）× 配置/消融数（3–5）+ zero-shot + HF offline 导出复载评测时间。
4. **先优化生成再堆卡**：GRPO 生成占 80%+ 时间，单卡优先调优 `vLLM colocate`，多卡优先切 `vLLM server`（`trl vllm-serve`）分离生成与训练卡池。

## 7. GPU 与显存利用率区间（估算，非实测）与排障方向

**没有任何**论文或 recipe 给出 GPU / 显存利用率百分比；下表是工程估算区间，只表示典型瓶颈位置，不得当作验收指标。

| 场景 | GPU 利用率 % | 显存利用率 % | 主要瓶颈 |
| --- | --- | --- | --- |
| SFT / LoRA / QLoRA，micro-batch 填满显存 | 60–90 | 70–95 | 算力 / 数据加载 |
| DPO / MPO（policy + ref 前向） | 55–85 | 65–90 | 算力；长序列 / 图像 token 受显存限制 |
| GRPO，HF generate 在线采样 | 30–60 | 50–85 | rollout 生成（自回归解码） |
| GRPO，vLLM colocate | 45–75 | 70–90 | 生成 / 训练切换；gpu_memory_utilization 需与训练争显存 |

校准与调优顺序：smoke 期间每 5 秒采样 `nvidia-smi` 与 trainer 日志。若 **SFT 显存低且 GPU 利用率低**：优先增大 `micro_batch`（同时下调 `grad_accum` 保持有效 batch）、检查 `dataloader_num_workers` 与图像解码是否在 CPU 卡住；若 **GRPO GPU 利用率低且锯齿波动**：这是自回归 rollout 阶段解码受限于访存带宽的正常现象，不要靠硬拉训练 batch 解决，应上 vLLM `colocate` / `server` 或缩短 `max_completion_length` 冗余。

## 8. 硬边界

- 估算、启发式区间、硬件起点表、利用率区间都**不是 GPU 实跑证据**；静态表、hash 与 dry-run 只证明内部一致，不解锁 B2/B3。
- `blocked_zero_train`（`qwen2-vl-7b-chartqa`）与 `contaminated_pretraining`（`smolvlm-chartqa`）是原理性阻塞，算力再多也不能绕过。
- 数据与基座许可：xLAM 为 gated + research-only，Bitext 未给 license，PaliGemma2 / Llama-3 gated 且 RefCOCO 图像运行时由 Flickr URL 下载；放大数据与投入集群算力前必须先确认商用许可。
- 改动任何 `recipe.toml` 的 batch / steps / 序列长度 / PEFT / 硬件声明后，运行 `.venv/bin/python -m pytest tests/test_compute_planning.py` 会自动校验同步。
