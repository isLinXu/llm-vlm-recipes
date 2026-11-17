# 已知陷阱清单与处置

迁移自调研报告第 08 章。第三列说明本仓库如何处置：**规则**表示验证器会拦截，
**单测/manual** 表示需要代码实现阶段的人工检查（诚实标注，不假装自动化了）。

| 类别 | 上游问题 | 涉及 recipe | 处置 |
| --- | --- | --- | --- |
| 版本与依赖 | `trl` 从 GitHub 源码安装，测试版 `0.22.0.dev0`；VLM GRPO / MPO 因功能未进 release 必须源码安装 | 起点、VLM GRPO、MPO | **规则** `R203` 预警 + 各篇 `todo` 要求锁定 commit |
| 版本与依赖 | `trl` 0.12.2 / 0.14.0 / 0.22.0.dev0 / 0.23.1；`transformers` 4.46.3 → 4.57.0 → ≥5.3.0 | 跨篇复制 | **规则** `R201` 精确 pin、`R202` 组合白名单、`R204` 全局漂移、`R207` 同族污染 |
| 版本与依赖 | ≥5.3.0 这类浮动约束不可复现 | Agent Wordle | **规则** `R201`；一手核实上游只有 `transformers>=5.3.0`，**上一轮写入的 `trl 1.7.0 / transformers 5.3.0` 属编造，已删除**并留注释，待核实后补精确 pin |
| 代码可运行性 | `train_dataset[0][:2]` 对 dict 切片 | 起点 | **manual** 记入 `known_issues`；我们的配方不复制该行 |
| 代码可运行性 | 博客 GSPO 示例 `steps_per_generation` 重复两次，Python 语法报错 | trl-vlm-alignment | **manual** 记入 MPO 篇 `known_issues` |
| 代码可运行性 | 评估用 `processor(...)` 但前面加载的是 `trained_processor` | VLM GRPO | **单测** `upstream_deviations` 要求统一 processor 并补用例 |
| 代码可运行性 | reward 对 `completions` 结构假设不一致（字符串列表 vs `completion[0]["content"]`） | VLM GRPO、对齐博客 | **单测** `upstream_deviations` 要求统一解析入口 + 3 个结构用例 |
| 代码可运行性 | Granite 篇输出 `FlashAttention is not installed`，实际 `_attn_implementation=None` | Granite Vision | **manual** 记入 `known_issues` |
| 训练-推理不一致 | 训练含 system message，推理固定 `messages[1:2]` 丢掉 system | 起点、SmolVLM | **规则** `R111`；我们配方的 `[consistency]` 两侧一致 |
| 训练-推理不一致 | 训练 `"<image>Detect {caption}."` vs 推理 `"<image>detect {caption}"` | PaliGemma grounding | **单测** `upstream_deviations` 要求模板常量化并补一致性测试 |
| 训练-推理不一致 | bbox 在 xyxy / COCO `[x,y,w,h]` / `<loc####>` 间描述横跳 | PaliGemma grounding | **单测** `todo` 要求往返转换测试 |
| 数据泄漏 | `SmolVLM-Instruct` 本就在 ChartQA 上训过，作者自承 educational exercise | SmolVLM | **规则** `R305` + `upstream_deviations` 要求未污染 held-out；否则只能作教学对照 |
| 数据泄漏 | eval `range(50,60)` 落在训练 `range(64)` 内；grounding 用训练集样本首次验证 | RapidFire、grounding | **规则** `R305`；RapidFire `upstream_deviations` 要求三段划分 + 断言测试 |
| 概念表述 | 把 QLoRA 解释成「量化 LoRA adapter 权重」，实际量化 base model | 起点、SmolVLM | **manual** 记入 `known_issues`；文档以代码为准 |
| 概念表述 | 标题写 QLoRA/LoRA，代码 `use_dora=True` | SmolVLM、Granite、VLM DPO、MPO | **规则** `R112`；SmolVLM / MPO 的 tags 显式含 `dora` |
| 概念表述 | 说 "merge the LoRA adapters" 但只 `PeftModel.from_pretrained` 未 merge | SmolVLM、VLM DPO | **manual** 记入 `known_issues` |
| 评估随机性 | 微调后 `do_sample=True`、基线 `do_sample=False`，无种子、无多次评估 | grounding | **规则** `R306` + `CONTRACT.md` 第 3 节 |
| 评估缺口 | 单样本定性「vibe-check」，无 `compute_metrics` / 未见样本指标 | 几乎所有篇 | **规则** `R304–R306`；各篇 `[evaluation]` 补定量协议 |
| 基线缺位 | 起点 recipe 自承零训练基线即可达标 | 起点 | **规则** `R302` 决策门；`decision.gate = "blocked"` |
| 环境陷阱 | OpenEnv 单会话，crash 泄漏 session 直到 `CAPACITY_REACHED` | Agent Wordle | **单测** `todo` 要求 `finally` 释放 + smoke 验证 |
| 环境陷阱 | vLLM 在 IPython 下初始化冲突 | vLLM colocate、Agent Wordle | **manual** `known_issues`；`todo` 要求脚本内核复现 |
| 成本失控 | 长时间 RL 无 SMOKE 闸门、无超时投影 | Agent Wordle | **规则** `R308`；`[budget]` + `[throughput]` 双档显式化 |

### 可复现性审计新增（`registry/evidence.toml` F09–F15）

这七条是本轮逐篇回一手材料核查后才发现的，处置以 `[repro]` 维度缺口 +
L5 规则为主，不靠改配置「修好」：

| ID | 上游问题 | 涉及 recipe | 处置 |
| --- | --- | --- | --- |
| F09 | 安装命令整行被注释，且运行时 `os.system("pip install flash_attn")` 动态装包；页面只有 PyTorch 2.8.0+cu128 / H100 的运行输出快照 | xLAM | **规则** `repro.env=missing` + `R506` 写明 gap；环境不可声明，更不可能锁定 |
| F10 | 唯一有真训练种子的一篇：`SEED=0` + `set_seed` + `shuffle(seed=SEED)` + `ConstantLengthDataset(seed=SEED)` + 4000 条 held-out | code LLM | **纠正判断**：从「完全无 seed」改为 `repro.seed=declared`；仍缺 `TrainingArguments.seed` 与 deterministic / cudnn / tf32 策略，达不到 `pinned` |
| F11 | 唯一有 `compute_metrics`（ROUGE-L）的一篇，但 `eval_dataset=range(50,60)` 完全落在 `train_dataset=range(64)` 内 | RapidFire | **规则** `R305` + `repro.eval`；页面从未给出实际 loss / ROUGE-L 数值，评估结论无效 |
| F12 | 训练 `<image>Detect {caption}.` + suffix 标签注入，推理 `<image>detect {caption}` 无句号无 suffix | PaliGemma grounding | **单测** `R111` + 模板常量化；**纠正判断**：该篇其实有 `train_test_split(test_size=0.2, seed=42)` 的 held-out，此前记为「无 held-out」有误 |
| F13 | 唯一在安装命令里写出精确 pin 的一篇：`qwen-vl-utils==0.0.8`；但 `trl` 仍是 `git+.../trl.git` main 分支无 commit，`DPOConfig(bf16=False)` 与其它 VLM 篇相反 | VLM MPO | **规则** `R201/R203`；`repro.env=declared` 而非 `pinned`，并把 `bf16=False` 记入 `[train]` |
| F14 | `DPOConfig` 只给工程项，`beta` / `loss_type` / `learning_rate` / `max_length` 上游均未提及 | VLM DPO | **规则** `repro.params` 标为未锁定；核心超参依赖 TRL 0.12.2 默认值，升级即结果变化 |
| F15 | 14 篇 `source_sha` 全为空，上游「Update on GitHub」链接不带 commit hash | 全部 14 篇 | **规则** `R206` + `R507`；`repro.provenance=declared`；`scripts/check_source_drift.py` 辅助补齐 |

另有跨篇共性 F01–F08：`pip install -U git+...` 无 commit、14 篇模型/数据均无
`revision`、12 篇训练 config 无 seed、无执行入口、评估普遍缺 `compute_metrics`、
reward 失败路径、`prompt-only` VLM 的 `image` + 预渲染字符串例外、版本注释≠版本约束。
完整证据与 53 处事实纠错见 [`registry/evidence.toml`](../registry/evidence.toml)
与 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

### 基线阶段新增（工程实现层，非上游缺陷）

这三条是把「能跑」落成执行契约时踩到的坑，处置落在代码与单测里：

| ID | 问题 | 涉及 | 处置 |
| --- | --- | --- | --- |
| F16 | 有了 `execution.command` 容易被误读成「今天就能出指标」，但统一 runner 只是骨架 | 全部 14 篇 | `execution.adapter_status` 逐篇写明适配状态；`--full` 缺 revision / PEFT / reward / 指标时**明确拒绝退出**，绝不返回占位分数；`runner/README.md` 与 [BASELINE-READINESS.md](BASELINE-READINESS.md) 双重声明 |
| F17 | `runner/locks/*.lock.toml` 是模板，不是冻结环境；把它当 `pip freeze` 用等于没锁 | 全部 14 篇 | 模板里 `model_revision` / `dataset_revision` / `source_sha` 一律 `"(未锁定)"`；真实值只能由 `runner/lock.py --apply` 在有网环境写入；C201–C203 把 hash 归到 B2，不允许靠填字符串升档 |
| F18 | Wordle 篇**已有** `[execution]`（HF Jobs + papermill），迁移脚本若直接追加同名表，TOML 会解析失败；早期 `upsert_scalar` 的全局 `re.sub` 还会误伤 `[repro.code].state` | Agent Wordle / 全部 14 篇 | 抽出共享的 `src/recipe_lab/toml_edit.py`，按段定位 upsert、保留注释与多行数组；`tests/test_toml_edit.py` 用重复表、行尾注释、`state` 同名键三组用例锁死行为 |

`train.seed` / `data_seed` 同样需要留意：除 code LLM（`SEED=0`）与 RapidFire
（`42`）外，其余 12 篇的 seed 是**本仓库定义 baseline 时声明**的，不是上游原值，
`seed_origin` 已逐篇标注。拿它们去「复现上游数值」是范畴错误。

---

## 尚未被规则覆盖的部分

以下风险**没有**自动化拦截，接入真实训练代码时必须靠单测或 review：

- 自定义 `collate_fn` 的 `-100` label masking 是否真的只算 assistant token；
- 其余RL任务的reward空输出/解析异常实际返回值；数学样板已有离线失败路径测试，仍待真实数据验证；
- FIM 模板、bbox 转换、prompt 模板在训练与推理侧是否引用同一常量；
- 环境 session 释放与 GPU 显存余量；
- `baseline.result` 与 `evaluation.metrics` 的数值本身是否真实；
- **L5 也无法覆盖**：`repro.*.state = pinned` 是否真的能装出同一环境、下到同一权重、
  重放出同一指标。本机无 GPU，未做任何真实训练；11 篇只有网页事实抽取，
  未逐字节校验 `.ipynb` 代码单元格。
- **L6不能认证执行真实性**：当前可从关联JSON/文件hash重算指标均值与容差、校验输入/配置/checkpoint身份，
  但不能仅靠这些内容证明真的训练、显存足够或模型收益。
  B1 的 14 篇全部通过 dry-run，但没有任何一篇产出过 `execution.evidence`；
  `evaluation.expected` / `tolerance` 仍为「待实跑」，`baseline.result` 13 篇仍是
  `unmeasured`。这些正是 C304 / C305 / C306 / C307 归到 B2 的原因。

对应待办已写进各篇 `[todo]`，推进到 `smoke-passed` 前需要逐项消解。
提升 `repro.level` 到 L3/L4 之前，必须先按
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) 的 P0–P5 补齐路径逐项留证据。
