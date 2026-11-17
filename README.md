# recipe-lab

面向 LLM / VLM 的研究与业务训练基线仓库，**建设中，尚不能宣称每篇都可作为 baseline**。
以 Hugging Face Cookbook 为来源，维护修正后的任务实现、输入冻结、对照评估与可核查证据，而非仅复制 notebook。

输入调研：[`llm-vlm-training-cookbook-research.html`](docs/llm-vlm-training-cookbook-research.html)。
来源与修正记录：[SOURCE-MAP](docs/SOURCE-MAP.md)、`registry/evidence.toml`。

## 先看实际能力，不看登记数量

| 事实 | 当前情况 |
| --- | --- |
| 已登记配方 | 14 篇，逐篇一手材料核查；不等于逐字节核对全部源码 |
| 有实际任务代码与离线测试 | 数学 GRPO、StarCoder FIM、xLAM function-calling、RapidFire 多配置 SFT、Granite Vision LineComparison、PaliGemma2 RefCOCO grounding 六篇各五维。均**只有离线测试、未做 GPU 验证** |
| 其余任务执行逻辑 | 剩余 8 篇完全未实现；六篇发布级按 `task_router` 精确分派、互不借用协议；未知或冲突任务拒绝实跑 |
| 五维能力 | 五个维度各 6/14；70 格中 30 已实现、40 缺失；GPU verified 0、B2+/B3 0、L2+ 0 |
| CUDA GPU 实跑证据 | 0 篇；未下载真实模型/数据，未确认框架兼容或训练效果 |
| 原理上可作 baseline | **eligible 0/14**；pending_freeze 6、not_implemented 6、`blocked_zero_train` 1、`contaminated_pretraining` 1（见 `make feasibility-check`） |
| 方法论依据 | 14 篇逐篇绑定 practice 论文一手结论：20 篇一手核实 / 3 篇未核实仅作补充、24 条主张、10 条编号更正（见 `make methodology-check`、`METHODOLOGY.md`）。论文依据不解锁 GPU 晋级 |
| MLLM 后训练方向 | 第五层审计：按指令遵循 / 偏好校准 / 推理增强 / 领域自适应 / 可扩展学习五方向映射 14 篇；4/5 方向有五维实现支撑，偏好校准为缺口（见 `make post-training-check`、`MLLM-POST-TRAINING.md`） |
| Agent 交接契约 | 人类读 `AGENTS.md` / `AGENT-HANDOFF.md`，Agent 读 `AGENT-HANDOFF.json`；生成物可重建且提升门禁失败关闭（见 `make agent-check`） |
| 声明就绪状态 | B1 declared 14 篇；**research-ready B2+ 0 篇、business-ready B3 0 篇** |
| 上游实验可复现性 | L0 8 / L1 6 / L2+ 0；没有确认任何篇达到完全锁定 |

B1 **仅是命令、入口、锁模板、种子与划分意图的声明**，不能证明划分可执行、训练跑通或指标可信。
`implemented` 表示任务代码和离线测试已交付；`verified` 必须另有实跑证据，静态 JSON/hash 检查仍不能认证执行真实性。

代码实装、GPU 证据和**实验目标是否成立**是三件事。逐篇可行性判定在
`registry/feasibility.toml`，由 `recipe-lab feasibility` 审计并写入 `BASELINE-READINESS.md`：

```bash
make feasibility-check                                        # 14 篇覆盖 + 一手事实交叉校验
PYTHONPATH=src python3 -m recipe_lab feasibility --min-state B2  # 当前必须失败关闭
```

其中 `qwen2-vl-7b-chartqa` 的 `decision.gate=blocked`（上游自承零训练基线即可达标），
`smolvlm-chartqa` 的基座本就在 ChartQA 上训练过（仅教学对照）——这两篇即使补齐五维代码
和 CUDA 实跑证据也不晋级，必须先改写实验目标 / 换未污染数据并重新核一手材料。
wave3 视觉任务因此按 `granite-vision-line-comparison`、`paligemma2-refcoco-grounding` 排序推进，
两篇现已各补齐五维并精确注册 `task_router`。Granite 篇是 `granite-linecomparison-v1`：图像原始
字节 hash 的 80/10/10 同图隔离、显式 `image_token_id` + assistant-only 双层 label mask、冻结
processor identity、held-out 逐题 exact_match 重算、`granite_suite` 三独立训练 seed 逐题内容重算
与 `granite_export` 的 DoRA/4bit adapter-only 包 / HF offline 新进程复载。PaliGemma 篇是
`paligemma-grounding-v1`：COCO `xywh`→`xyxy` 统一、1024 桶 `<loc####>` 单框 round-trip/IoU、
统一 prompt、显式 image token + completion-only 双层 mask、图像 hash 隔离、held-out 逐题
`bbox_iou`/`iou_at_0_5`/`parse_rate`/`exact_match` 重算、`paligemma_suite` 从 raw bbox 重建 gold
的三 seed 重算与 `paligemma_export` 普通 LoRA adapter-only 包 / HF offline 新进程复载；四指标
不等于 mAP/AP 或业务验收。两篇都**没有** CUDA 实跑、冻结 revision、预声明容差或许可确认
（上游图像由 Flickr URL 运行时下载，还须冻结静态副本），可行性判定改判 `pending_freeze`
（只表示五维代码齐备），仍不是 eligible。

再上一层是**方法论依据**（第四层审计）：每篇设计决策依据哪篇 practice 论文的一手结论、
论文没证明什么，在 `registry/methodology.toml` 登记；14 篇 `recipe.toml` 各有 `[methodology]`
引用式视图，由 `recipe-lab methodology` 审计并逐字段核对两侧一致性，结果写入 `METHODOLOGY.md`。
所有 arXiv 编号回一手摘要页核对，核查中实际发现并更正 10 条凭记忆写错的编号；
论文 benchmark 数字绝不写成本仓 expected/tolerance，文献背书也不洗白原理性阻塞。

```bash
make methodology-check                                    # 14 篇覆盖 + 一手字段 + 引用闭合 + 配方回填一致
PYTHONPATH=src python3 -m recipe_lab methodology --min-verified-papers 99  # 提升目标示例
```

再上一层是 **MLLM 后训练方向层**（第五层审计）：按行为/能力把训练工作映射到
指令遵循、偏好校准、推理增强、领域自适应、可扩展学习五方向，登记在
`registry/mllm_post_training.toml`。五方向原始出处是 preprints 202607.1494 的摘要级框架；
arXiv 2503.06072 只做跨域交叉印证，`Awesome-MLLM-Tuning` 只做不同轴的方法索引，
二者都不当五方向来源或复现证据。实现状态从 `capabilities.toml` 重算，
`blocked_zero_train` / `contaminated_pretraining` 篇显式标为无效覆盖证据。
当前 4/5 方向有五维实现支撑，**偏好校准是缺口**，不表示 DPO/RLAIF-V 已有 baseline。

```bash
make post-training-check                                        # 五方向闭合 + 14 篇映射 + 来源边界
PYTHONPATH=src python3 -m recipe_lab post-training --min-fivedim-directions 5  # 当前必须失败
```

为了让人类和 Agent 看到同一套事实，仓库另有 **Agent 交接契约**（ADR-026）：
`AGENTS.md` 是短入口，`docs/AGENT-HANDOFF.md` 是人类可读现状与工作流，
`docs/AGENT-HANDOFF.json` 是机器可读的事实源、生成物哈希、命令契约与 Guardrails。
`recipe-lab handoff --check` 会用真实 CLI 把五份生成报告重建到临时目录并逐字节比较；
任一侧漂移就失败关闭。它不认证 GPU 执行，也不解锁 B2/B3。

```bash
make agent-check            # 交接契约 + 生成物可重建性
make agent-report           # 生成 AGENT-HANDOFF.md / .json
```

完整的14篇五维矩阵、各篇具体缺失、方法覆盖边界、wave0–wave6依赖和验收：
**[BASELINE-READINESS.md](docs/BASELINE-READINESS.md)**。
方法论依据与一手论文核查：**[METHODOLOGY.md](docs/METHODOLOGY.md)**。
MLLM 后训练五方向与 14 篇映射：**[MLLM-POST-TRAINING.md](docs/MLLM-POST-TRAINING.md)**。
Agent 交接：[AGENTS.md](AGENTS.md) · [AGENT-HANDOFF.md](docs/AGENT-HANDOFF.md) · [AGENT-HANDOFF.json](docs/AGENT-HANDOFF.json)。
历史一手审计：**[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**。

## 快速验证（无 GPU、无训练依赖）

需要 Python ≥3.11；核心审计模块只用标准库，pytest 仅用于测试。在仓库根运行：

```bash
make ci                      # 结构/契约、审计完整性、能力矩阵、方法论依据与离线测试
make capabilities-check      # 精确14篇覆盖、文件证据与合法能力状态
make methodology-check       # 论文一手字段、引用闭合、边界声明与阻塞篇交叉校验
make post-training-check     # MLLM 后训练五方向映射、来源边界与能力/可行性交叉
make agent-check             # Agent 交接契约与生成物可重建性
make report                  # 重建 MATRIX / REPRODUCIBILITY / BASELINE-READINESS / METHODOLOGY / MLLM-POST-TRAINING / AGENT-HANDOFF
make baseline-dry-run        # 输出14篇声明；不运行训练，不产生实跑证据
PYTHONPATH=src python3 -m recipe_lab capabilities --json
```

`make ci` 通过不代表每篇可用；真实发布门槛与登记维护门槛分开：

```bash
PYTHONPATH=src python3 -m recipe_lab baseline --check --min-state B2  # 当前应失败
PYTHONPATH=src python3 -m recipe_lab repro --check --min-level L2     # 当前应失败
```

## 已实现的受控参考样板

`llm/grpo/qwen2-0-5b-numinamath` / `math-grpo-v1`：

- Qwen2-0.5B-Instruct、NuminaMath-TIR、TRL 0.14.0，纯文本全参数 GRPO；不冒充通用框架。
- 数据源 train 连续索引 `[0:500]` 训练、`[500:1000]` validation、`[1000:1500]` test。
  每段500题，记录内容/题目hash及manifest；gold解析失败或跨split重复直接停止。
- 统一 `<think>...</think><answer>...</answer>` prompt；空输出、格式错误、预测解析/验证异常不得满分。
- 训练参数显式传入并校验 API signature 与实际应用值；固定种子、单进程CUDA、smoke/full预算和输出隔离。
- zero-shot 只加载 base；eval 必须加载带已完成训练来源的本地 checkpoint，显式路径不能绕过来源/hash/配置校验。
- 同一held-out数据与greedy解码协议，保存逐题预测、accuracy/pass@1、格式合规率、实际配置与运行环境。
- eval只报告**单checkpoint**结果；`--runs > 1`拒绝，重复greedy解码不等于多训练seed方差。
- 完成前不写completed证据，不自动晋级、不自动填写预期分数或把上游结论当本仓库实测。

```bash
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run --json
python3 runner/eval.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run
python3 runner/zero_shot.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run
```

### 有网且有 CUDA 的运行环境（本轮没有执行）

先使用隔离环境安装并核对已声明依赖；torch CUDA版本必须选择并实测，现有模板不是安装锁，也没有可安装镜像。
依赖冲突/API不兼容应作为明确的版本迁移记录，不能静默忽略。

```bash
python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath --apply
python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath --capture-environment
python3 runner/zero_shot.py --recipe llm/grpo/qwen2-0-5b-numinamath --full
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --smoke
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --full
python3 runner/eval.py --recipe llm/grpo/qwen2-0-5b-numinamath --full
```

`--apply`只解析缺失输入并保留已锁commit；`--capture-environment`捕获完整实际pip freeze、Python、CUDA、驱动及GPU。
每次实跑前核对 frozen lock 与实际完整环境，不接受模板、漂移版本或未冻结依赖。
训练产物在`runs/<recipe>/<experiment>/smoke|full`；评估在`eval|zero-shot`，不覆盖旧结果。
`execution.evidence`是未来B2套件路径，单次training/evaluation日志不足以满足它。

## 数学三 seed 套件与导出复载

`--train-seed`只临时覆盖训练种子；数据种子与评估种子固定，不修改TOML。
`--output-root`必须为绝对路径，模式输出在其`full/eval/zero-shot`子目录。

```bash
# 默认只读预演；把ABS替换为独立实验的绝对路径
python3 runner/suite.py --recipe llm/grpo/qwen2-0-5b-numinamath --seeds 42 43 44 --output-root /ABS/math-suite --dry-run
# 仅在输入和真实CUDA环境已冻结后执行
python3 runner/suite.py --recipe llm/grpo/qwen2-0-5b-numinamath --seeds 42 43 44 --output-root /ABS/math-suite --full
# 仅验证已完成的输入；不能续训、补造历史快照或覆盖已有suite.json
python3 runner/suite.py --recipe llm/grpo/qwen2-0-5b-numinamath --seeds 42 43 44 --output-root /ABS/math-suite --aggregate-only
python3 runner/export.py --recipe llm/grpo/qwen2-0-5b-numinamath --checkpoint /ABS/math-suite/seed-42/full --evaluation /ABS/math-suite/seed-42/eval/evaluation.json --output /ABS/math-export --dry-run
```

- 套件顺序执行一次共享基座评估，再执行三个独立训练/评估子进程；保存代码快照、锁、日志与关联证据。
- 以训练seed为统计单位，输出训练后与配对增益的均值、样本方差、标准差、标准误、Student-t 95%区间。
  固定test的区间不包含数据抽样不确定性，不能当业务显著性结论。
- 缺`expected/tolerance`仍可记录测量，但验收为`not_evaluated`，不会自动填目标或晋级B2。
- 导出`--full`要求显式完整导出容差（`evaluation.export_tolerance`或`evaluation.tolerance`）；包中复制模型/tokenizer、配方快照、锁与参考预测，在新进程强制HF offline重载复评。
  需要预缓存锁定数据，不是安装镜像；许可pending，回滚演练另行确认。
- 基座、套件与导出发布门禁重读内容并重算，拒绝代码/锁漂移、替换gold、修改权重/分数及不一致统计。
  所有hash只验证内部一致性，不认证GPU执行真实性。本轮仅离线测试，未执行上述实跑命令。

## 第二任务：StarCoder 文件隔离 FIM

`llm/sft/starcoder1b-fim` / `code-fim-v1` 五维代码与离线测试均已实现，三独立训练 seed 套件与 adapter-only 导出复载已有代码；**没有真实 GPU 证据，B2/B3 仍未完成**。

- 数据卡核实字段 `repo_id/file_path/content`；有界完整扫描，repo哈希80/10/10、全局源码去重，不随机补救空划分。
- PSM/SPM token变换、文件内独立分块和padding，损失覆盖所有真实causal token；真实EOS保留标签，padding屏蔽。不跨文件attention。
- `Trainer` + nf4/bf16/doublequant/LoRA；full明确2000步，smoke明确5步。候选`q_attn`可未命中；其余target/base冻结/trainable参数显式核验。
- 基座和本地adapter使用相同量化/tokenizer/PSM窗口/greedy协议；严格文本与token exact-match保留空白，gold不入生成输入，EOS预算另留一位。
- adapter只接受safetensors；键集合、shape、有限性和值与模型比对，拒绝非严格加载。文件hash不能替代张量完整性。
- **自建迁移不是上游复现**：repo隔离、文件内padding、eager/no remote code/no Hub push、PSM评估均已登记；未声称pass@1/编译率，不执行任何生成代码。
- FIM发布级重算器/套件/导出已有代码与离线测试，但**没有真实GPU证据**：`fim_suite.py`重算PSM输入/gold与三seed统计，`fim_export.py`打包adapter-only并在HF offline新进程复载；`task_router.py`精确分派、未知任务失败关闭。缺completed实跑证据时C306/C307/C405仍明确阻断B2/B3。模型需授权，数据卡缺许可；repo隔离不证明无预训练污染。

```bash
make fim-dry-run
python3 runner/train.py --recipe llm/sft/starcoder1b-fim --dry-run --json
python3 runner/eval.py --recipe llm/sft/starcoder1b-fim --dry-run
python3 runner/zero_shot.py --recipe llm/sft/starcoder1b-fim --dry-run
```

实跑前须使用已授权模型访问、锁定输入commit，并捕获完整torch/transformers/datasets/peft/bitsandbytes/safetensors CUDA环境。
上游未给精确版本，本仓库不猜版本；真实API不兼容则停止并声明迁移，不静默忽略参数。

## 第三任务：xLAM function-calling SFT（五维代码，未做 GPU 验收）

`llm/sft/llama3-8b-xlam-function-calling` / `xlam-fc-v1` 五维均有代码与隔离离线测试；比较/导出只有合成证据，不是 B2 验收。

- 一手核查（Cookbook 页 + 数据卡）：列为 `query` / `tools` / `answers`，均为字符串；数据卡 gated、research-only，2024-07 曾重生成 1896 条。
  上游把 gold 拼进 `dataset_text_field="text"` 由 `SFTTrainer` 对**整段文本**做损失，推理 prompt 截断在 `<tools>`（不给工具定义），展示输出是 Python `str(dict)` 单引号而非合法 JSON，且无 seed / revision / held-out / 定量指标。
- 本仓自建协议：`transformers.Trainer` + completion-only collator，**只对 `<calls>` 严格 JSON 数组 `</calls>` + EOS 计算 loss**（prompt 部分 label=-100）；gold 不进生成输入；分别编码避免 BPE 跨边界合并；长度超限拒绝截断。
- 严格 schema：工具重名、answer.name 不在 tools、多余参数、缺 required、类型不符、未知列一律拒绝；规范化内容 hash 80/10/10 确定性划分，全局重复（含跨 split）拒绝，空 split 不偷偷补救；test 为 shared held-out（最多 300 条）。
- 评估只用 `json.loads` 解析 `<calls>`，**绝不 eval / 执行生成的调用**；`json_parse_rate` / `schema_valid_rate` / `tool_call_exact_match` / `argument_f1` 逐题重算。这些是格式与协议指标，**不等于真实 API 执行正确率、pass@1 或业务安全**。
- `task_router.py` 精确注册 `xlam-fc-v1` 的 task / suite / export 三类模块：`xlam_suite.py` 做三互异训练 seed 的独立输出、共享 base/held-out 协议、**逐题从 raw query/tools/answers 重建 prompt 与 gold 并重算四指标**、Student-t95 统计；`xlam_export.py` 打包 adapter-only 权重、冻结 base 引用（`base_weights_bundled=false`）、严格 JSON 输出协议，并在 HF offline 新进程真实调用 `xlam_task.run_evaluation` 复载复评。C306/C307/C405 按任务分派，不借用数学 1500 题或 FIM PSM 协议；未知/冲突任务仍失败关闭。
- 仅隔离离线测试（`tests/test_xlam.py`、`tests/test_xlam_release.py`，合成 tmp_path 夹具）；真实 `recipe.toml` **未预声明 `evaluation.export_tolerance`，`--full` 导出会失败关闭**，不自动回填 expected/tolerance。**无 GPU 实跑、无 completed 证据、未冻结 revision**；基座需授权、数据卡商业许可未确认。B1 不晋级。

```bash
make xlam-dry-run          # train / eval / zero_shot 三个只读声明，不需要训练框架
make xlam-suite-dry-run    # 三独立训练 seed 套件的只读声明
make xlam-export-dry-run   # adapter-only 导出与 offline 复载的只读声明
```

## 第四任务：RapidFire 多配置 SFT（五维代码，未做 GPU 验收）

`llm/sft/rapidfire-multiconfig` / `rapidfire-sft-v1` 五维（`data` / `train` / `eval` / `comparison` / `export`）均已有代码与隔离离线测试；
`task_router.py` 精确注册该任务的 `suite` / `export`，C306/C307/C405 分派到 RapidFire 专用内容校验器，未知或冲突任务仍失败关闭，不借用数学 / FIM / xLAM 协议。

- 一手核查（Cookbook 页 + 数据卡）：`gpt2` + Bitext 客服数据集（26872 条、仅 train split、5 列 `flags/instruction/category/intent/response`，卡片未给 license）。
  上游 `train=select(range(64))`、`eval=select(range(50,60))`，**eval 完全落在 train 内**；`sample_formatting_function` 把 gold 拼进整段 `Question:/Answer:` 文本；`sample_compute_metrics` 直接把 `eval_preds` 的 predictions/labels 传给 `evaluate.load('rouge')`（未见 decode，异常静默返回 `{}`），页面**从未报告任何 rougeL / loss 数值**。
- 上游用未固定版本的 `rapidfireai` 服务做 4 配置并发调度（`num_chunks=4`），无 revision / 包版本 / `set_seed` / deterministic；`16-24×` 是吞吐宣传，不是训练指标。
  本仓**顺序执行** 4 组配置（`[[configs]]` 精确声明：`lr 5e-4/linear`、`2e-4/cosine+warmup10` × `r8/alpha16/c_attn`、`r32/alpha64/c_attn+c_proj`），
  `config_recipe()` 只覆盖 `[train]` / `[peft]` / `execution.config_id` / `experiment_id`，逐配置隔离输出目录与证据；不宣称复现并发调度。
- 自建无泄漏数据协议：规范化问题 hash（NFKC + casefold + 空白折叠）80/10/10，同题所有语言变体落同一 split；规范化内容全局去重并拒绝跨 split 重复；空 split 拒绝，不随机补救。
  `Trainer` + completion-only collator，prompt 只到 `Answer:`，只对 ` {response}` + EOS 计 loss，训练右 padding。
- 评估为单样本 greedy、首个 EOS 截断；纯 Python **无 stemmer** LCS ROUGE-L 与 `exact_match` 逐题重算，不引入未冻结版本的 `evaluate` / `rouge_score`，
  **数值与上游 `use_stemmer=True` 的 rougeL 不可比**；checkpoint 来源、fingerprint（绑定协议段/完整网格/config_id）、adapter 张量、逐题预测与 manifest 强绑定。
- 发布级（ADR-019）：`rapidfire_suite.py` 做 4 配置 × 3 互异独立训练 seed 严格聚合，逐题从 `eval_manifest` 的 raw `instruction/response` 重建 prompt 与 gold 并重算指标；只用 validation 的 paired `rouge_l` 均值选配置（平局按 `config_id` 字典序），test **仅在选中配置上验收一次**；输出均值 / 样本方差 / SEM / Student-t95 与顺序执行时长 / 显存 / 步数成本对照，明确 `upstream_throughput_comparable=false`，不宣称复现并发加速。`rapidfire_export.py` 导出选中配置的 adapter-only fp16 LoRA 包，HF offline 全新进程真实调用 `rapidfire_task.run_evaluation` 在 test 复评；真实 recipe 未预声明 `evaluation.export_tolerance`，`--full` 导出失败关闭，不回填容差。
- 内容身份只含训练 / 评估协议：RapidFire 的 `_identity_recipe()` 剥离实跑后回填的 `expected` / `tolerance` / `execution.evidence`，保留训练前预声明的 `export_tolerance`；`acceptance` 是阈值判定而非证据重算内容，验收由 `suite_problems` 用当前阈值独立判定，不采信 JSON 里自称的 `passed`。Granite 的 `identity_recipe()` 只剥离 `expected` / `tolerance`——它的 `execution.evidence` 是配方预声明的位置路径而非回填指针，剥离会破坏与训练时有效配方的内容身份（ADR-022）。
- 只有 `tests/test_rapidfire.py` 与 `tests/test_rapidfire_release.py` 的 `tmp_path` 合成 fake runtime 证据；**无 GPU 实跑、无真实 completed 证据、未冻结 revision、数据卡许可未确认**，B1 不晋级。ROUGE-L / `exact_match` 只是文本格式指标，不证明客服业务正确或安全。

```bash
make rapidfire-dry-run            # 4 配置 + eval / zero_shot（两种 split）只读声明
make rapidfire-suite-dry-run      # 4 配置 × 3 seed 发布级套件只读预演
make rapidfire-export-dry-run     # 选中配置 adapter-only 导出只读预演
python3 runner/train.py --recipe llm/sft/rapidfire-multiconfig --config-id lr2e-4-r32 --dry-run --json
```

## 第五任务：Granite Vision LineComparison（五维代码，未做 GPU 验收）

`vlm/sft/granite-vision-line-comparison` / `granite-linecomparison-v1` 五维代码与隔离离线测试齐备
（`tests/test_granite.py` 58 项 + `tests/test_granite_release.py` 44 项）；`task_router.py` 精确注册
`granite_suite` / `granite_export`，C306/C307/C405 分派到 Granite 专用校验器，不借用数学 / FIM /
xLAM / RapidFire 协议。**全部只有 macOS ARM 离线合成证据，GPU verified 0。**

- 一手核查（ADR-021）：上游只取 `Geoperception` 的 `predicate=="LineComparison"` 并做 50/50 切分（无 validation），collator 只按 `<|assistant|>` 隐式 mask 且无显式 `image_token_id` 屏蔽，无 revision / 训练 seed / `compute_metrics`，评估仅 `test_idx=20` 单样本定性生成，文字写 ChartQA 但代码从未加载，说 merge 却只有 `PeftModel.from_pretrained`。本仓是**自建协议**，不冒充上游复现。
- 自建数据协议：图像原始 `mode + tobytes()` 的 SHA256 做 80/10/10，同图所有问答变体同 split；规范化 `(question,answer)` 全局去重、跨 split 重复与空 split 拒绝。prompt/full 两次 `apply_chat_template` 并做 token 级前缀校验，gold 绝不进生成 prompt；显式 `image_token_id` + assistant span 外双层 `label=-100`；普通 `Trainer` + 自建 collator（非 `SFTTrainer` / `skip_prepare_dataset`）；冻结 processor identity / seed / revision / 代码与锁字节。
- 评估：held-out 逐题 greedy + 首个 EOS 截断 + `exact_match` / `normalized_exact_match` 逐题重算，空生成不给分；`eval_manifest` 从 raw image/question/answer 重建 prompt 与 gold，zero-shot 只加载 base、训练评估必须有 completed checkpoint。**指标只衡量文本一致性，不解释几何语义、不等于人工判分或业务验收**；不执行生成代码，与上游无可比数值。
- 发布级（ADR-022）：`granite_suite.py` 做三互异独立训练 seed 的 held-out 聚合，逐题从 `eval_manifest` 重建 gold 并用 `score_prediction` 重算两指标（不信任摘要分数），输出均值 / 样本方差 / SEM / Student-t95，不回填 `expected/tolerance`、不自动晋级 B2。`granite_export.py` 导出 DoRA/4bit adapter-only 包（冻结 base / processor identity / chat template、显式双层 mask 与 RGB 输入声明、`base_weights_bundled=false`、虚拟锚点锁、包内快照与训练时有效配方逐字节一致），HF offline 新进程真实调用 `granite_task.run_evaluation` 复载复评。真实 recipe 未预声明 `evaluation.export_tolerance`，`--full` 导出失败关闭。
- 内容身份按任务分派：Granite 的 `identity_recipe()` 只剥离 `expected` / `tolerance`，保留 `export_tolerance` 与作为预声明位置路径的 `execution.evidence`；RapidFire 的 `_identity_recipe()` 才额外剥离 `execution.evidence`。`acceptance` 由 `suite_problems` 用当前阈值独立判定，不采信 JSON 自称的 `passed`。
- 剩余门槛：冻结 model/dataset/source/processor revision 与可安装镜像、授权 Linux CUDA 真实三 seed 实跑、基于实跑预声明容差、HF offline 复载、Geoperception 与基座许可/商业使用/再分发确认、业务验收。可行性判定由 `partial_implementation` 改判 `pending_freeze` 只反映五维代码齐备，**eligible 仍为 0/14**，B1 不晋级。

```bash
make granite-dry-run              # 三个基础入口只读声明
make granite-suite-dry-run        # 三 seed 发布级套件只读预演
make granite-export-dry-run       # adapter-only 导出只读预演
python3 runner/train.py --recipe vlm/sft/granite-vision-line-comparison --dry-run --json
```

## 完整方法覆盖与实施队列

当前偏向视觉SFT与GRPO示例，**不是全生态覆盖**。`registry/capabilities.toml`区分：

- taxonomy已登记但无recipe的family（运行时自动计算）；
- CPT/DAPT、音频、视频等尚未纳入taxonomy的独立模块；
- 已有视觉DPO/MPO不等于文本偏好优化已实现；数据、评估、导出是跨方法管线，不是trainer。

| 波次 | 内容 | 晋级依据 |
| --- | --- | --- |
| wave0 | 失败关闭的状态、输入、环境、证据门禁 | 正反向测试，不靠非空字段或假文件 |
| wave1 | 数学样板 → 独立训练seed聚合 → 导出复载 | 真实zero-shot/smoke/full、至少3seed、同协议评估与产物验证 |
| wave2 | 文本SFT：xLAM、FIM、RapidFire | 数据/label mask隔离、任务专用指标、无泄漏与安全执行 |
| wave3 | 视觉SFT：ChartQA、Granite、PaliGemma | 图像token mask、图像去重、bbox/loc、污染声明 |
| wave4 | DPO/MPO | shared backbone、ref模型冻结、三损失消融；不同模型不可直接归因算法 |
| wave5 | vLLM、在线agent RL、VLM GRPO | 权重同步、同配置开关对照、环境回收与独立对局 |
| wave6 | 偏好/RM/RLOO/蒸馏/CPT与模态扩展 | 每种方法独立实现、测试、锁、证据与回归 |

详细依赖和逐篇缺口以能力登记及生成报告为准，波次登记不代表波次已完成。

## 审计轴与成熟度

| 轴 | 问题 | 权威来源 |
| --- | --- | --- |
| 验证层级L0–L6 | 执行哪些结构/契约/证据规则？ | `rules.py`，`--level` |
| 可复现性L0–L4 | 能否重放上游实验？ | `reproducibility.toml`，`repro` |
| B0–B3 | 能否成为自建研究/业务基线？ | `baseline.toml`，`baseline` |
| 实际五维能力 | 哪些任务逻辑已实现、已实跑？ | `capabilities.toml`，`capabilities` |
| 可行性判定 | 实验目标是否成立、数据是否污染？ | `feasibility.toml`，`feasibility` |
| 方法论依据 | 这么设计依据哪篇论文的一手结论？ | `methodology.toml`，`methodology` |
| MLLM 后训练方向 | 训练工作覆盖哪个能力方向、缺口在哪？ | `mllm_post_training.toml`，`post-training` |
| Agent 交接契约 | 人与 Agent 如何从同一事实源复现？ | `agent_contract.toml`，`handoff` |

`status`的cataloged→ported→smoke-passed→benchmarked只记录配方维护进展，不替代上述审计。
复现上游实验与自建baseline可以不同，但后者必须公开声明差异并冻结、实测。

## 关键目录

```text
registry/                    taxonomy、reproducibility、baseline、capabilities、feasibility、methodology、mllm_post_training、agent_contract、一手evidence
recipes/**/recipe.toml        14篇配方；所有来源与自建差异显式登记
src/recipe_lab/               审计/CLI、plan、math_task、baseline_evidence、capabilities、feasibility、methodology
runner/                      train / eval / zero_shot / lock / suite / export；说明见runner/README.md
runner/locks/                当前14份输入/环境锁模板；不是实测freeze
runs/                        本地执行产物（默认忽略，不把大权重提交进仓库）
tests/                       纯函数、mock、结构集成、正反向证据门禁测试
scripts/                     一次性迁移、来源漂移辅助；重跑不能覆盖已有配置/锁
docs/BASELINE-READINESS.md   就绪、实际能力、缺口、实施依赖及验收（生成）
docs/METHODOLOGY.md          逐篇方法论依据、一手论文核查与编号更正（生成）
docs/MLLM-POST-TRAINING.md   MLLM 后训练五方向、来源边界与 14 篇映射（生成）
docs/AGENT-HANDOFF.md        Agent 交接契约的人类可读视图（生成）
docs/AGENT-HANDOFF.json      Agent 交接契约的机器可读清单（生成）
AGENTS.md                    Agent 可发现入口与硬边界（静态）
docs/REPRODUCIBILITY.md      历史上游可复现性审计（生成）
docs/MATRIX.md               全量配方索引（生成）
docs/DESIGN.md              架构决策、证据检查边界
docs/VERIFICATION.md        规则编号与维护/发布门禁区分
```

新增recipe必须同时补taxonomy映射（如需）、一手evidence、repro、baseline_contract与**五维能力登记**；
未实现维度填missing并描述具体阻塞，不以统一入口文件的存在冒充实现。运行`make ci`及`make report`。

## B2 / B3 的证据要求

- B2：真实输入commit、完整实际运行环境、基座评估、至少3个独立训练seed的配对评估。
  内容校验包括代码快照、split去重与hash、实际环境锁、训练预算/参数、checkpoint身份、逐题预测重算与均值容差。
- B3：B2之上补许可确认、资源实测、业务验收、owner、导出复载及回滚。人工confirmed字段不是自动合规或SLO认证。
- 静态JSON/hash门禁只证明提交证据**内部一致性**，不能认证GPU执行真实性；保留原始日志与产物，由独立运行流程复核。
- 数学、FIM、xLAM、RapidFire、Granite、PaliGemma 六篇的独立训练seed编排/严格聚合、发布级逐题内容重算与导出新进程offline复载均已实装并有隔离离线测试，按`task_router`精确分派、不互相借用协议；RapidFire 额外做 4 配置网格 × 3 seed 的 validation 选配置 / test 单次验收与成本对照，Granite 额外做图像内容 hash 隔离的三 seed 逐题重算与 DoRA/4bit adapter-only 复载，PaliGemma 额外做 1024 桶 `<loc####>` round-trip/IoU 与从 raw COCO bbox 重建 gold 的三 seed 四指标重算。六者同样只有离线合成证据，均无CUDA实测、可安装镜像或完整业务回归；真实 recipe 未预声明 `export_tolerance`，`--full` 导出失败关闭。其余8篇仍完全未实现。
- 已声明版本不代表兼容性已实测；数据去重或gold解析失败必须停止，不能为绿灯放宽门槛。
- 仓库采用 MIT License（见 [LICENSE](LICENSE)）；模型 / 数据许可仍需逐项确认，仓库许可不等于上游资源许可。

**当前交付是更严格的维护框架、完整缺口地图与数学/FIM/xLAM/RapidFire/Granite/PaliGemma 六篇五维适配（均未做GPU验证）；不是14篇已完成的baseline。**
