# 数据契约规范

依据 TRL 官方 `dataset_formats` 与 VLM alignment 文档，以及调研报告第 02 / 03 / 04 章。
本文件是规范说明；可执行版本在 `src/recipe_lab/rules.py` 的 `R104–R115`。

---

## 1. Trainer ↔ dataset type

`dataset.type` 必须是下列值之一，且必须落在对应 trainer 允许的范围内（`R104`）：

| Trainer | 允许的 `dataset.type` | 备注 |
| --- | --- | --- |
| `SFTTrainer` | `language-modeling`、`prompt-completion` | 标准 / 对话两种结构均可 |
| `DPOTrainer` | `preference` | 推荐显式 prompt |
| `RewardTrainer` | `preference` | 推荐隐式 prompt |
| `KTOTrainer` | `unpaired-preference`、`preference` | |
| `GRPOTrainer` / `RLOOTrainer` / `OnlineDPOTrainer` | `prompt-only` | 在线生成，不需要参考回答 |
| `DistillationTrainer` | `prompt-completion` | |

`dataset.structure` 取 `conversational` 或 `standard`。
**VLM 默认要求 `conversational`**（`R105`）——但有一个已登记例外，见第 2.1 节。

注意 `preference` 里 MPO 不是独立 trainer：`registry/taxonomy.toml` 把它登记为
`tier = "recipe"`，`trainer = "DPOTrainer"`，本质是
`DPOConfig(loss_type=["sigmoid","bco_pair","sft"], loss_weights=[0.8,0.2,1.0])`。
GSPO 同理是 `tier = "variant"` 的 GRPO 损失变体。

---

## 2. VLM 的四条铁律

调研结论：这四点决定绝大多数 VLM 训练的失败，且与算法选择无关。

| # | 契约 | 规则 | 违反后果 |
| --- | --- | --- | --- |
| 1 | 顶层 `images`（多图）或 `image`（单图）列，与 `messages` 并列 | `R107` | 图像输入被静默丢弃，模型只看到文本 |
| 2 | 消息 `content` 必须是 `list[dict]`，用 `{"type":"image"}` / `{"type":"text"}` | `R108` | processor 无法解析多模态消息 |
| 3 | `processing_class=processor`，不是 tokenizer | `R109` | 只做文本分词，视觉塔收不到输入 |
| 4 | VLM SFT/偏好训练设 `max_length=None` | `R110` | 图像 token 被截断，答案依据丢失 |

### 正确写法（来自起点 recipe）

```python
def format_data(sample):
    return {
        "images": [sample["image"]],          # 顶层列，与 messages 并列
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["query"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["label"][0]},
            ]},
        ],
    }

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(max_length=None, bf16=True, ...),   # 不要写 max_length=1024
    train_dataset=ds,
    peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
    processing_class=processor,                       # 不是 tokenizer
)
trainer.train()
```

### 典型错误写法

```python
# 错 1/2：图像只以占位符出现在纯文本 content 里，没有顶层 images 列
{"messages": [{"role": "user", "content": "describe <image>"}]}

# 错 3：用 tokenizer 当 processing_class
SFTTrainer(..., processing_class=tokenizer)

# 错 4：截断图像 token
SFTConfig(max_length=1024)
```

### 2.1 prompt-only VLM 例外（已登记，F07）

上面的四条铁律描述的是 **SFT / 偏好训练**的 `conversational` 形态。
VLM **在线 RL**（GRPO / RLOO）存在另一种合法形态：上游不把图像塞进
`messages`，而是先用 `processor.apply_chat_template()` 把多模态对话**预渲染成
字符串 `prompt`**，数据集最终只有 `prompt: str` + `image: PIL.Image` 两列。

典型来源是 `vlm/grpo/qwen2-5-vl-3b-visual-reasoning`（本地 `.ipynb` 原文核实）：

```python
def make_conversation(example):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": example["problem"]},
        ]},
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {"prompt": prompt, "image": example["image"]}
```

对应配置：

```toml
[dataset]
structure = "standard"
type = "prompt-only"

[modality_contract]
images_column = "image"           # 单数，仍是顶层并列列
content_list_dict = false         # prompt 已是字符串
pre_rendered_prompt = true        # 显式声明例外，缺了就会被 R105/R108 拦下
prompt_rendered_by = "processor.apply_chat_template"
processing_class = "processor"
max_prompt_length = 2048
```

**为什么这不是规避契约**：`list[dict]` 结构仍然存在，只是发生在
`make_conversation()` 内部、交给 `apply_chat_template` 之前；图像也仍是与
`prompt` 并列的顶层列，没有退化成文本占位符。真正会静默丢视觉输入的是
「只写 `<image>` 字符串、没有顶层图像列」，与本例外无关。

`R105` / `R108` 仅在 `dataset.type = "prompt-only"` **且**
`pre_rendered_prompt = true` 同时成立时，把 error 降为 info；
少声明任何一项都会照常报错（见 `tests/test_rules.py` 的双向用例）。
即便如此，`max_prompt_length` 仍需确认图像 token 未被截断——上游原文为 2048，
本仓库保留 `R110` 的在线 RL 提示。

### 其他 VLM 工程约定

- 图像统一转 `RGB`；用 `process_vision_info(messages)` 抽取视觉输入（Qwen2-VL 系）。
- 视觉塔冻结时跳过量化：`llm_int8_skip_modules=["vision_tower", "lm_head"]`。
- 自定义 `collate_fn` 时，手工把 prompt / image token 的 label 设为 `-100`，
  实现 **assistant-only loss**（Granite Vision 篇）。此路径建议配往返单测。
- 在线 RL 关注 `max_prompt_length`：若设有限值，必须确认图像 token 未被截断（`R110`）。
- `GRPOConfig(remove_unused_columns=False)` 常用于保留 `solution` 列供 reward 使用；
  误删会让 `accuracy_reward` 静默拿不到答案（VLM GRPO 篇）。

---

## 3. 训练-推理一致性（`R111`）

上游多个 recipe 训练时包含 system message，推理函数却固定写 `sample["messages"][1:2]`
把 system 丢掉；PaliGemma grounding 篇训练用 `"<image>Detect {caption}."`、
推理用 `"<image>detect {caption}"`，大小写与标点都不一致。

**约定**：prompt 模板写成单一常量，训练与推理共同引用；
`[consistency]` 中显式声明两侧 system message 策略，不一致直接报错。

评估阶段同样要一致：`do_sample`、seed、是否多次运行必须对齐。上游 grounding 篇
微调后 `do_sample=True`、基线 `do_sample=False`，两者根本不可比。

---

## 4. RL 的 reward 契约（`R115` / `R307`）

GRPO / RLOO 没有 value model，**reward 定义就是算法本身**。所有 RL recipe 必须声明：

```toml
[rewards]
functions = ["format_reward", "accuracy_reward"]
failure_path_tested = true
```

必须覆盖的失败路径：空输出、格式错误、答案错误、解析器抛异常、环境未结束。
典型陷阱来自 Wordle / OpenEnv 篇——invalid move 也可能拿到 partial credit，
所以**不能用 `reward > 0` 判定胜利**：

```python
# 错误
solved = reward > 0

# 正确
solved = bool(env.done and reward >= ENV_SOLVED_REWARD)
```

`accuracy_reward` 用 `math_verify.parse/verify` + `LatexExtractionConfig()`，
解析失败应返回 `None` 而非 `0`，否则「格式错」会被误记成「答案错」。
另外要留意 `completions` 的结构假设：上游有的按字符串列表处理，
有的取 `completion[0]["content"]`，跨篇复用 reward 函数时最容易出错。

---

## 5. 评估纪律（`R304–R306`）

Cookbook 最大的方法论缺口是「只展示 1 个 test sample 的定性生成」。
本仓库要求：

- `evaluation.metrics` 必须是定量指标，禁止 `vibe_check` / `qualitative` 占位；
- `evaluation.held_out = true`（上游 RapidFire 的 eval `range(50,60)` 落在训练
  `range(64)` 内，属数据泄漏）；
- 固定 `seed`；非 `cataloged` 状态要求 `multi_run = true`；
- RL 额外监控 reward 曲线与独立指标是否背离（背离即停止）。

推荐四层评估：零训练基线 → 未见样本定量指标 → 训练-推理一致性 → 成本/效果增量对照。

### 5.1 基线阶段的划分与种子

`[evaluation].split` 是 baseline 的固定划分，由 `runner/eval.py --dry-run` 直接读取。
其中标注「本仓库定义」的划分（如 RapidFire 改为 `train[:40]/val[40:50]/test[50:60]`
三段不重叠）是**我们的修正**，不代表上游用了同样的划分；`train.seed` /
`train.data_seed` / `evaluation.seed` 同理，12 篇的上游训练 config 根本没有 seed，
`seed_origin` 逐篇记录这个区别。

要点：做对照实验时，DPO 篇与 MPO 篇、NuminaMath 篇与 vLLM colocate 篇**共用同一
划分与索引**，否则「MPO 比 DPO 好」「vLLM 提了多少吞吐」这类结论会被切分差异污染。
`evaluation.expected` / `tolerance` 必须首次实跑后回填，未回填时
`runner/eval.py --dry-run` 会把这两项显示为「契约检查未通过」，而不是编一个数填满。
详见 [BASELINE-READINESS.md](BASELINE-READINESS.md)。
