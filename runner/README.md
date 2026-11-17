# runner — 受控执行层

**已实现代码：数学GRPO、StarCoder FIM、xLAM function-calling、RapidFire 多配置 SFT、Granite Vision LineComparison、PaliGemma2 RefCOCO grounding 六篇各五维（data/train/eval/comparison/export）。GPU实跑验证0个。其余8篇拒绝实跑。**
`plan.py`与三个CLI共享配置解释；`math_task.py`是默认数学路径，`fim_task.py` / `xlam_task.py` / `rapidfire_task.py` / `granite_task.py` / `paligemma_task.py`由三个CLI的`TASK_ADAPTERS`精确`recipe_id`白名单分派，不用数学fallback冒充其他任务。Granite 与 PaliGemma 是两篇 VLM 任务，各自有独立的图像内容 hash 隔离、显式 `image_token_id` 双层 label mask、processor identity 冻结与逐题重算协议；`execution.task_id` 均已显式声明，`registry/taxonomy.toml` 按精确 `recipe_id` 登记 `Trainer` 例外。发布级`suite`/`export`由`task_router.py`按精确`recipe_id`/`execution.task_id`分派到六篇各自的专用模块，C306/C307/C405内容校验互不借用；真实 `--full` 在无冻结环境、无预声明 `export_tolerance` 时失败关闭。未知或冲突任务失败关闭。
模块导入与dry-run只用标准库；训练框架延迟导入。

## 当前路由

- recipe：`llm/grpo/qwen2-0-5b-numinamath`，task：`math-grpo-v1`。
- Qwen2-0.5B-Instruct + NuminaMath-TIR + TRL 0.14.0，纯文本全参数GRPO、单进程单CUDA GPU。
- 数学路由不支持PEFT、量化、VLM或其他trainer；不把同名字段当通用适配器。
- train前500题，validation随后500题，test再随后500题；gold解析与跨split题目/内容重复检查失败即停止。
- 格式/正确性reward、训练/评估共用prompt，固定greedy单checkpoint评估。

## 运行命令

以下dry-run不联网、不下载、不产实跑证据：

```bash
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run --json
python3 runner/eval.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run
python3 runner/zero_shot.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run
python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath
```

有网、隔离安装环境、实际CUDA GPU齐备后（本轮未执行）：

```bash
python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath --apply
python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath --capture-environment
python3 runner/zero_shot.py --recipe llm/grpo/qwen2-0-5b-numinamath --full
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --smoke
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --full
python3 runner/eval.py --recipe llm/grpo/qwen2-0-5b-numinamath --full
```

`--recipe`允许唯一前缀；多个匹配拒绝。训练不指定模式时默认dry-run，评估未传`--full`时同样只打印计划。

## 输入与环境冻结

1. 按已声明依赖安装并核对版本，选择并实测torch CUDA版本；此仓库尚无可安装容器镜像或完整安装锁。
2. `--apply`只解析缺失的模型、静态数据与Cookbook commit，生成`resolved-inputs`，**不是冻结环境**。
   已有commit保留；已有有效frozen lock不重查询main、不覆盖。
3. `--capture-environment`在GPU机器捕获完整pip freeze、Python、CUDA、驱动与GPU，拒绝editable/浮动URL。
4. 每次执行核对frozen lock和实际完整环境；环境改变必须另行声明迁移，不能沿用旧结论。
5. 在线环境不要求不存在的Hub dataset revision，而应冻结环境commit、词表hash与种子；Wordle执行逻辑尚未实现。

## 产物与失败关闭

| 入口 | 产物 | 边界 |
| --- | --- | --- |
| train --smoke/--full | 对应目录权重、tokenizer、manifest、training.json | 完成预算并保存成功后才写completed；非空目录拒绝覆盖 |
| zero_shot --full | zero-shot/逐题预测、manifest、evaluation.json | 只加载base，不是训练证据 |
| eval --full | eval/逐题预测、manifest、evaluation.json | 只加载已完成训练的本地checkpoint，缺失不回退base |
| lock --apply | 输入commit锁 | 不证明框架兼容或完整环境可重放 |
| lock --capture-environment | frozen运行环境锁 | 环境捕获仍不是模型训练验证 |

默认根：`runs/<recipe>/<experiment>/`；smoke/full/eval/zero-shot输出隔离，不覆盖旧日志。
显式`--checkpoint`也要求training.json、权重hash及当前配置/revision匹配；省略时读取full产物。
保存阶段仅做模型结构校验，评估阶段再校验训练来源。未知训练参数/API signature不兼容直接拒绝。
CUDA不可用时不下载；确定性设置必须在框架导入之前生效。

## 指标与晋级

- 数学指标：accuracy/pass@1、format_compliance，逐题预测可重算。
- `--runs`目前仅支持1；重复greedy解码不是独立训练seed方差。
- training/evaluation证据不等于`baseline-suite`，不自动写expected/tolerance或升级B2。
- B2仍需至少3个独立训练seed的配对对照、代码/环境/split/产物绑定与均值容差。
- 数学多seed编排、严格聚合与导出新进程offline复载已实现并通过离线测试；GPU实测和业务回归仍未完成。
- 离线mock只验证装配与失败路径；不验证真实数据可用、API兼容、GPU显存、收敛或指标收益。

## 独立训练套件与导出

- 三个基础入口都接受`--train-seed INT`与`--output-root ABS`；只改内存有效训练配置，不修改TOML，data/evaluation seed固定。
- `suite.py --recipe 完整ID --seeds 42 43 44 --output-root ABS`默认仅预演；`--full`依次运行共享base与每个seed独立full/eval；`--aggregate-only`只聚合已完成的真实输入。
- 套件绑定canonical执行代码快照与完整环境锁字节，子命令前后检查漂移，拒绝非空root；错误/timeout终止子进程并保留磁盘日志，不支持自动resume或覆盖。
- 聚合校验full预算、有效seed配置、严格1500条manifest、prompt/gold内容hash、基座/权重来源、预测及binary指标；发布审计复用强校验，不只相信suite摘要。
- 输出均值、样本方差、标准误和Student-t95区间。训练seed是统计单位，固定test不含数据抽样不确定性；`acceptance`与测量完成分开，不回填expected或自动晋级。
- `export.py --recipe 完整ID --checkpoint ABS --evaluation ABS --output ABS`默认预演，`--full`复制模型/tokenizer/锁/配方/参考预测后启动全新Python进程。
  worker强制HF offline，用实际数学评估重载checkpoint；需缓存锁定数据，缺缓存失败而非联网降级。容差必须显式声明，许可保持pending。
- C405重读完整导出包并复核复载身份、文件/锁/模型hash与指标容差；仍需`export_verified`及单独`rollback_verified/rollback_evidence`人工回滚记录。
- B2证据设置`execution.evidence`指向`suite.json`，`baseline.measurement`指向`base/zero-shot/evaluation.json`；不得以单次training替代套件。路径由维护者确认回填，不自动改配置。

## FIM独立适配（代码验收，GPU未验收）

- `llm/sft/starcoder1b-fim / code-fim-v1`使用同一train/eval/zero_shot入口；`make fim-dry-run`可预演，默认不写文件不下载。
- `repo_id/file_path/content`有限完整扫描、全局源码去重与repo hash split；文件内独立PSM/SPM分块/padding，不跨文件attention，causal-all-nonpadding非assistant-only损失。
- `Trainer`+nf4/bf16/doublequant/LoRA，API未知参数失败关闭；完整预算2000步，smoke5步；q_attn只是候选，不强制增加cross-attention。
- 安全safetensors adapter keys/shape/值与模型比较，训练后保存effective配置、source/transform manifest及tokenizer身份；显式checkpoint也须核来源，不回退base。
- 同量化/base revision/tokenizer/test/PSM/greedy协议精确补全，严格保留空白，EOS预算单独保留；无eval/exec/编译或沙箱执行，不能宣称pass@1。
- eager/no remote code/no Hub push/repo隔离等均为自建变体；数据卡未给许可，模型下载需访问授权，repo隔离不证明无预训练污染。
- FIM不借用数学suite/export：`task_router.py`按精确`recipe_id`/`execution.task_id`分派到`fim_suite.py`（三独立训练seed聚合、PSM输入/gold内容重算）与`fim_export.py`（adapter-only包、冻结base声明、HF offline新进程复载）；未知/冲突任务失败关闭。只有**未注册发布级管线**的recipe才在dry-run显式标注`release_pipeline="unavailable"`，已注册的六篇必须分派到各自协议。
- 上述发布级代码只有离线合成证据与受控worker测试；C306/C307/C405不能靠提交普通JSON放行，真实CUDA三seed、真实GPU复载、许可与业务验收仍缺，B2/B3保持未完成。

## xLAM独立适配（五维代码验收，GPU未验收）

- `llm/sft/llama3-8b-xlam-function-calling / xlam-fc-v1`复用三个基础入口；`make xlam-dry-run`只输出声明，不写文件不下载。
- 一手数据卡字段`query`/`tools`/`answers`严格JSON schema校验；上游把gold拼进`dataset_text_field="text"`由`SFTTrainer`对全文做损失，本仓改用`Trainer`+自建completion-only collator（仅`<calls>`严格JSON数组`</calls>`+EOS计loss，prompt label=-100），并给全tools定义（上游推理prompt截断在`<tools>`）。
- 规范化内容hash 80/10/10 + 全局去重（含跨split）、空split拒绝、test shared held-out最多300条；长度超限拒绝截断，不切掉gold。
- 推理输出只用`json.loads`解析，Python `str(dict)`/注入文本一律拒绝；**绝不eval或执行生成的函数调用**。`json_parse_rate`/`schema_valid_rate`/`tool_call_exact_match`/`argument_f1`是格式与协议指标，不是真实API执行正确率或pass@1。
- checkpoint绑定training来源、recipe/fingerprint/锁字节、LoRA target与safetensors张量值、source/transform manifest、tokenizer身份与artifact hash；base zero-shot只加载base，不回退。
- 发布级不借用数学/FIM协议：`task_router.py`按精确`recipe_id`/`execution.task_id`分派到`xlam_suite.py`（三独立训练seed聚合、逐题从raw query/tools/answers重建prompt与gold并重算四指标、Student-t95统计）与`xlam_export.py`（adapter-only包、冻结base与严格JSON输出协议声明、HF offline新进程真实调用`xlam_task.run_evaluation`复载复评）；包内`recipe-snapshot.json`逐字节保持训练内容身份，锁用虚拟锚点自包含。未知/冲突任务仍失败关闭，通用JSON不能晋级。
- `make xlam-suite-dry-run` / `make xlam-export-dry-run`只输出只读声明；真实`recipe.toml`未预声明`evaluation.export_tolerance`，`--full`导出失败关闭，不自动回填expected/tolerance。
- 数据卡gated且research-only、2024-07曾重生成1896条；基座需授权，商业许可与revision冻结、真实CUDA三seed/真实GPU复载/API兼容/显存全未实测；发布级代码只有隔离离线合成证据，B1不晋级。

## RapidFire独立适配（五维代码验收，GPU未验收）

- `llm/sft/rapidfire-multiconfig / rapidfire-sft-v1`复用三个基础入口，并新增`--config-id`精确选择4组配置之一；`make rapidfire-dry-run`覆盖全部4配置与eval/zero_shot，只输出声明，不写文件不下载。
- 上游`train range(64)`/`eval range(50,60)`泄漏、`%pip install rapidfireai`无版本、无revision/`set_seed`、页面从未报告任何rougeL数值；本仓改用规范化问题hash 80/10/10三段不重叠划分（同题变体同split）+ 全局内容去重 + completion-only loss，并顺序执行4配置，不宣称复现并发调度或16-24×加速。
- `[[configs]]`精确声明2个lr/scheduler × 2个LoRA rank/alpha/target_modules；`config_recipe()`只覆盖`[train]`/`[peft]`/`execution.config_id`/`experiment_id`，逐配置隔离输出目录与证据；`fingerprint()`绑定协议段、完整网格与当前config_id，改lr/rank后旧checkpoint不能冒充当前配置。
- 评估用纯Python无stemmer LCS ROUGE-L与exact_match逐题重算，首个EOS截断，不引入未冻结版本的evaluate/rouge_score；**与上游`use_stemmer=True`数值不可比**。checkpoint来源、权重张量、transform/source/eval manifest与tokenizer身份强绑定，base zero-shot只加载base。
- **发布级已按ADR-019补齐**：`task_router.py`精确注册`rapidfire-sft-v1`的`rapidfire_suite.py`/`rapidfire_export.py`。套件做4配置×3互异独立训练seed严格聚合，逐题从raw instruction/response重建prompt与gold重算指标，只用validation的paired rouge_l均值选配置（平局按config_id字典序），test仅在选中配置上验收一次；输出Student-t95与顺序执行成本对照，明确不与上游并发吞吐可比。导出选中配置adapter-only包，HF offline新进程真实调用`rapidfire_task.run_evaluation`在test复评；真实recipe未预声明`evaluation.export_tolerance`，`--full`失败关闭，不回填容差。C306/C307/C405用RapidFire专用校验器，不借用数学/FIM/xLAM协议。`--eval-split validation|test`绑定`eval_stage`，选择/验收隔离。
- **内容身份边界**：`_identity_recipe()`剥离实跑后回填的`expected`/`tolerance`/`execution.evidence`，保留训练前预声明的`export_tolerance`，避免注册C307时与训练时`effective_recipe.json`失配；`acceptance`是阈值判定而非证据重算内容，验收由`suite_problems`用当前阈值独立判定。
- 数据卡26872条/5列/仅train split且未给license；revision、完整CUDA环境锁、商业许可、GPU实跑与真实completed证据全缺，只有`tests/test_rapidfire.py`与`tests/test_rapidfire_release.py`的tmp_path合成证据；ROUGE-L/exact_match不证明客服业务正确或安全，B1不晋级。

## Granite Vision 独立适配（五维代码验收，GPU未验收）

- `vlm/sft/granite-vision-line-comparison / granite-linecomparison-v1`复用三个基础入口；`make granite-dry-run`只输出声明，不写文件不下载。上游只取`Geoperception`的`predicate=="LineComparison"`并做50/50切分（无validation），collator仅按`<|assistant|>`隐式mask且无显式image token屏蔽，无revision/训练seed/compute_metrics，评估只有`test_idx=20`单样本定性；文字写ChartQA但代码从未加载，说merge却无`merge_and_unload()`。本仓为**自建协议**，不冒充上游复现。
- 图像原始`mode + tobytes()`的SHA256做80/10/10，同图所有问答变体同split，规范化`(question,answer)`全局去重与跨split重复/空split拒绝；`prompt`/`full`两次`apply_chat_template`并做token级前缀校验，gold只在full序列、绝不进生成prompt；显式`image_token_id` + assistant span外双层`label=-100`；普通`Trainer`自建collator（非`SFTTrainer`/`skip_prepare_dataset`）；nf4/DoRA/bf16/eager、视觉塔跳过量化、动态筛`language_model`+`_proj`且命中非空；冻结processor identity / seed / revision / 代码与锁字节。
- 评估是held-out逐题greedy + 首个EOS截断 + `exact_match`/`normalized_exact_match`重算（NFKC+casefold+空白折叠），空生成不给分；`eval_manifest`从raw image/question/answer重建prompt与gold，zero-shot只加载base、训练评估必须有completed checkpoint且无base fallback。**指标只衡量文本一致性，不解释几何语义、不等于人工判分或业务验收**，也不执行生成代码。
- **发布级已按ADR-022补齐**：`task_router.py`精确注册`granite_suite.py`/`granite_export.py`。套件做三互异独立训练seed的held-out聚合，逐题从`eval_manifest`的`question/answer`重建gold并用`score_prediction`重算两指标（不信任摘要分数），输出均值/样本方差/SEM/Student-t95，不回填`expected/tolerance`、不自动晋级B2。导出DoRA/4bit adapter-only包（冻结base/processor identity/chat template、显式双层mask与RGB输入声明、`base_weights_bundled=false`、虚拟锚点锁、包内`recipe-snapshot.json`与训练时有效配方逐字节一致），HF offline新进程真实调用`granite_task.run_evaluation`在held-out复载复评。C306/C307/C405用Granite专用校验器，不借用数学1500题/FIM PSM/xLAM严格JSON/RapidFire ROUGE-L协议。
- **内容身份边界**：`identity_recipe()`只剥离实跑后回填的`expected`/`tolerance`，保留训练前预声明的`export_tolerance`；`execution.evidence`是配方预声明的**位置路径**、不是回填指针，剥离反而会破坏与训练时`effective_recipe.json`的一致性，因此不剥离（与RapidFire的`_identity_recipe()`不同）。`acceptance`是阈值判定结果，验收由`suite_problems`用当前阈值独立判定。
- `make granite-suite-dry-run` / `make granite-export-dry-run`只输出只读声明；真实`recipe.toml`未预声明`evaluation.export_tolerance`，导出`--full`失败关闭，不自动回填容差。
- 全部只有macOS ARM上的`tests/test_granite.py`（58项）与`tests/test_granite_release.py`（44项）隔离合成证据：未加载真实Geoperception或torch/CUDA，model/dataset/processor revision、可安装镜像锁、依赖版本、Geoperception与基座许可/商业使用/再分发权限、真实CUDA三seed与GPU复载、业务验收全未完成；GPU verified 0、B2+/B3 0、eligible 0/14，B1不晋级。

## PaliGemma2 grounding 独立适配（五维代码验收，GPU未验收）

- `vlm/sft/paligemma2-refcoco-grounding / paligemma-grounding-v1`复用三个基础入口；`make paligemma-dry-run`只输出声明，不写文件不下载。上游一手页面用`SFTTrainer`+`skip_prepare_dataset`、`labels=input_ids.clone()`只mask pad与`<image>`（prompt也计loss）；`A.BboxParams(format="coco")`与`coco_to_xyxy()`按COCO`xywh`输入而可视化直接当`xyxy`；训练prompt`<image>Detect {caption}.`与推理`<image>detect {caption}`三处不一致；`SFTConfig`有`eval_dataset`却无`compute_metrics`/IoU/任何数值，只有`train[20]`与`val[13]`单样本定性；`do_sample=True`无生成seed，无revision/训练seed/版本锁。本仓为**自建协议**，不冒充上游复现。
- 输入明确为`coco_xywh`、内部统一`xyxy`并严格拒绝退化/越界/NaN框；`format_location()`按`round(value*1024/max_value)`生成4位`<loc####>`并clamp到1023，`encode_detection()`只产单框，`decode_detection()`严格拒绝多框/`;`/loc数量不为4/桶值越界/反向框；`ROUND_TRIP_MIN_IOU=0.98`（1024桶量化约1%固有误差，但不丢框/不退化/不越界）。统一`PROMPT_TEMPLATE="<image>Detect {caption}."`，`max_new_tokens=96`（4个loc token共36字符+referring expression，64会截断caption使`exact_match`恒为0）。
- 图像原始`mode + tobytes()`的SHA256做80/10/10，同图所有caption变体同split，规范化`(caption,bbox)`全局去重；processor做`prompt`/`prompt+suffix`两次编码并校验token级前缀，gold只在full序列、绝不进生成prompt；显式`image_token_id` + prompt span外token双层`label=-100`；普通`Trainer`自建completion-only collator，普通LoRA不用4bit/DoRA/`trl`/`bitsandbytes`；冻结processor identity / seed / revision / 代码与锁字节。
- 评估是held-out逐题greedy + 首个EOS截断 + `bbox_iou`/`iou_at_0_5`/`parse_rate`/`exact_match`重算，`eval_manifest`保留raw COCO bbox与冻结width/height；zero-shot只加载base、训练评估必须有completed checkpoint且无base fallback。**四指标只衡量单框定位一致性，不等于mAP/AP、多框检测、人工判分或业务验收**，也不执行生成代码。
- **发布级已按ADR-023补齐**：`task_router.py`精确注册`paligemma_suite.py`/`paligemma_export.py`。套件做三互异独立训练seed的held-out聚合，逐题从`eval_manifest`的raw COCO bbox + width/height + caption重建gold detection与xyxy，再对`generated_text`用`score_prediction`重算四指标与parsed bbox（不信任摘要分数），输出均值/样本方差/SEM/Student-t95，不回填`expected/tolerance`、不自动晋级B2。导出普通LoRA adapter-only包（冻结base/processor identity、统一prompt与loc协议、RGB图像输入、单框IoU、`base_weights_bundled=false`、`generated_code_executed=false`、`license pending`、虚拟锚点锁、包内`recipe-snapshot.json`与训练时有效配方逐字节一致），HF offline新进程真实调用`paligemma_task.run_evaluation`在held-out复载复评。C306/C307/C405用PaliGemma专用校验器，不借用Granite的`exact_match`/数学1500题/FIM PSM/xLAM严格JSON/RapidFire ROUGE-L协议。
- `make paligemma-suite-dry-run` / `make paligemma-export-dry-run`只输出只读声明；真实`recipe.toml`未预声明`evaluation.export_tolerance`，导出`--full`失败关闭，不自动回填容差。
- 全部只有macOS ARM上的`tests/test_paligemma.py`（46项）与`tests/test_paligemma_release.py`（44项）隔离合成证据：未加载真实PaliGemma2/RefCOCO或torch/CUDA；上游图像由Flickr URL运行时下载须先冻结静态副本，model/dataset/processor revision、可安装镜像锁、依赖版本、gated基座与RefCOCO/COCO/Flickr许可/商业使用/再分发权限、真实CUDA三seed与GPU复载、业务验收全未完成；GPU verified 0、B2+/B3 0、eligible 0/14，B1不晋级。

逐篇能力与系统缺口：[BASELINE-READINESS.md](../docs/BASELINE-READINESS.md)。
