# Baseline Readiness 与系统能力审计

> 2026-11-13；由 `recipe-lab baseline --report` 生成。禁止手工修改派生报告。

## 当前结论

**14篇：B0 0 / B1 14 / B2 0 / B3 0。**
**可作 research baseline 的B2+：0/14；业务B3：0/14。**

B1仅证明执行声明齐备，不能证明任务实现、真实split、环境兼容或训练跑通。
此前把14篇B1作为进展主指标不够准确：必须同时查看下方五维实际能力矩阵。
复现上游那次实验与维护自建baseline不同；后者仍必须冻结自有代码、数据、环境并实测。

## 状态与证据门槛

| 状态 | 含义 |
| --- | --- |
| B0 | 执行契约未齐 |
| B1 | 仅声明：命令、入口、环境锁模板与种子存在；不代表任务适配已实现、划分可执行或GPU跑通 |
| B2 | 输入和运行环境冻结，基座实测及至少3个独立训练seed的同协议对照可核查；指标在容差内，仍需人工真实性复核 |
| B3 | B2之上，许可确认、实测资源、业务验收、交接与导出重载回滚复核齐全 |

## 逐篇声明就绪度

| recipe | 状态 | 目标 | 下一档阻塞 |
| --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `llm/grpo/qwen2-0-5b-numinamath` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `llm/grpo/qwen2-5-3b-advanced-reward` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `llm/sft/llama3-8b-xlam-function-calling` | B1 | business | C201, C202, C203, C212, C304, C305, C306, C307 |
| `llm/sft/rapidfire-multiconfig` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `llm/sft/starcoder1b-fim` | B1 | business | C201, C202, C203, C212, C304, C305, C306, C307 |
| `vlm/dpo/smolvlm-rlaif-v` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `vlm/sft/granite-vision-line-comparison` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `vlm/sft/paligemma2-refcoco-grounding` | B1 | research | C201, C202, C203, C212, C304, C305, C306, C307 |
| `vlm/sft/qwen2-vl-7b-chartqa` | B1 | business | C201, C202, C203, C212, C304, C305, C306, C307 |
| `vlm/sft/smolvlm-chartqa` | B1 | business | C201, C202, C203, C212, C304, C305, C306, C307 |

## 全量检查项

| ID | 最低状态 | 检查 | 通过/总数 |
| --- | --- | --- | --- |
| C201 | B2 | 来源commit | 0/14 |
| C202 | B2 | 模型revision | 0/14 |
| C203 | B2 | 数据或在线环境revision | 0/14 |
| C212 | B2 | 完整冻结环境 | 0/14 |
| C304 | B2 | 实测指标数值表 | 0/14 |
| C305 | B2 | 容差数值表 | 0/14 |
| C306 | B2 | 基座实际评估 | 0/14 |
| C307 | B2 | 独立训练对照证据集合 | 0/14 |
| C103 | B3 | 模型许可确认 | 0/14 |
| C104 | B3 | 数据许可确认 | 0/14 |
| C401 | B3 | 资源实测复核 | 0/14 |
| C403 | B3 | 业务验收复核 | 0/14 |
| C404 | B3 | 维护归属 | 0/14 |
| C405 | B3 | 导出重载回滚复核 | 0/14 |
| C101 | B1 | 实验标识 | 14/14 |
| C102 | B1 | revision策略 | 14/14 |
| C204 | B1 | 执行命令 | 14/14 |
| C205 | B1 | 入口文件 | 14/14 |
| C206 | B1 | 锁模板文件 | 14/14 |
| C207 | B1 | 训练seed | 14/14 |
| C208 | B1 | 数据seed | 14/14 |
| C209 | B1 | 确定性策略 | 14/14 |
| C210 | B1 | 运行时约束 | 14/14 |
| C211 | B1 | 对照变量原则 | 14/14 |
| C301 | B1 | 评估入口 | 14/14 |
| C302 | B1 | 数据划分声明 | 14/14 |
| C303 | B1 | 独立训练多seed计划 | 14/14 |
| C402 | B3 | smoke/full成本档 | 14/14 |

## 本轮实装修正

- NuminaMath math-grpo-v1已补固定三段split、gold验证、跨split去重、统一prompt、reward、真实GRPO装配、基座与checkpoint评估以及逐样本证据；只在离线纯函数/mock测试中验证，未下载实际数据、未完成GPU运行。
- eval缺checkpoint拒绝，不再加载base冒充训练效果；zero-shot仅加载base。smoke/full隔离输出，预算越界和未知参数拒绝。
- C201/C202拒绝main与非commit字符串；C304/C305拒绝占位文本、NaN及负容差；C306要求基座实测文件而非上游一句passes。
- C212要求完整实际环境锁；C307复用生产聚合器强校验至少三个独立训练seed、代码/锁/有效配置、严格1500条manifest与gold hash、checkpoint及预测/统计/均值容差；C405自动核导出复载包，人工回滚另行确认。
- 修正合法B2被错误要求B3许可/owner的规则；测试不再要求所有revision永远为空、所有配方永远B1。
- 在线环境使用environment revision、词表内容hash与对局seed；不再要求不存在的Hub dataset revision。

## 复查命令

```bash
make ci
PYTHONPATH=src python3 -m recipe_lab capabilities --check
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run
python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath --apply
python3 runner/lock.py --recipe llm/grpo/qwen2-0-5b-numinamath --capture-environment
python3 runner/zero_shot.py --recipe llm/grpo/qwen2-0-5b-numinamath --full
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --smoke
python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --full
python3 runner/eval.py --recipe llm/grpo/qwen2-0-5b-numinamath --full --checkpoint <真实训练产物目录>
```

后三类命令仅供具备冻结依赖、实际数据与CUDA GPU的环境运行；本轮未执行。
数学gold解析或split去重失败会停止，而不是放宽门槛给满分。

### 计划预演（不是训练日志）

```text
$ python3 runner/train.py --recipe llm/grpo/qwen2-0-5b-numinamath --dry-run
recipe     : llm/grpo/qwen2-0-5b-numinamath  [dry-run]
experiment : llm-grpo-qwen2-0-5b-numinamath-v1
trainer    : GRPOTrainer / prompt-only
model      : Qwen/Qwen2-0.5B-Instruct @ (未锁定)
dataset    : AI-MO/NuminaMath-TIR @ (未锁定)
seed       : 42 / data_seed 42
steps      : 5  ->  runs/llm__grpo__qwen2-0-5b-numinamath/llm-grpo-qwen2-0-5b-numinamath-v1/dry-run
eval       : pass@1, format_compliance | protocol=single_checkpoint_evaluation | runs=1
supported  : True（代码路由，不代表实跑通过）
warnings   : 4 条（见 --json）
```

## 本机无法确认的部分与剩余工作

- 当前没有任何实际GPU训练/指标证据；先前Hub API请求失败不等于所有网络通道永久不可用。未编造任何revision或分数。
- execution.adapter_status仅记录不替代代码；数学math-grpo-v1、StarCoder code-fim-v1、xLAM xlam-fc-v1、RapidFire rapidfire-sft-v1、Granite granite-linecomparison-v1、PaliGemma paligemma-grounding-v1、SmolVLM DPO smolvlm-dpo-v1与Qwen2.5-VL MPO qwen25vl-mpo-v1八篇五维均有代码与隔离离线测试（偏好篇见 tests/test_preference.py）。八篇均按精确recipe_id注册task_router，发布级dry-run分派到各自专用协议，--full仍失败关闭；C306/C307/C405按任务分派，互不借用协议。DPO/MPO共享偏好split算法但backbone不同，cross_backbone_attribution_valid=false，不得跨篇归因算法。真实recipe未预声明export_tolerance时--full导出失败关闭。未知或冲突任务失败关闭；GPU、真实三seed、真实GPU复载、许可与业务验收未通过，其余6篇中advanced-reward的GSM8K gsm8k-reward-v1仅data/train/eval三维离线实装（partial_implementation，未注册task_router，发布级套件unavailable），另3篇任务逻辑拒绝执行、2篇原理性阻塞。
- 环境捕获、框架API兼容与实际模型加载未在GPU实测；离线测试不证明训练收敛或性能。
- 八篇的独立训练seed编排/严格聚合、发布级逐题内容重算与导出新进程offline复载均已实装并有隔离离线测试，按task_router精确分派、互不借用协议。真实recipe未预声明export_tolerance时--full导出失败关闭；八篇GPU多seed/复载验收、可安装镜像锁、冻结model/dataset/source/processor revision、业务安全/成本回归未完成。RapidFire顺序执行4配置不得外推并发编排收益；Granite的exact_match只衡量文本一致性；PaliGemma的单框IoU不等于mAP/AP；DPO/MPO的隐式奖励pairwise不等于人类偏好一致率，且不得跨backbone归因算法。
- B3若使用人工confirmed字段，只证明维护者记录了复核，不能自动认证许可授权或业务SLO；后续须接真实验收流程。
- 静态内容与hash检查证明证据内部一致，不是GPU执行真实性证明；需要独立运行流程与人工复核。
- 晋级条件必须是代码与证据齐备，不是增加空recipe、字段或测试数。完整逐篇缺口与实施依赖如下。

## 实际能力与全生态范围（独立于 Baseline Readiness）

> 本地时点：2026-11-18；不联网，不修改 taxonomy。
- recipe 已登记：14；能力条目匹配：14；至少一维代码已实现：9 篇；train+eval GPU verified：0 篇。
- 五维实际计数：missing 27 / implemented 43 / unverified 0 / verified 0。
- implemented = 真实任务代码 + offline test 文件/覆盖；不代表实际数据下载、依赖兼容、GPU 通过或收敛。
- missing = 任务逻辑缺失；unverified = 证据不足；verified = impl/test 齐备且真实 completed JSON kind 相符。
- 配方、family、B1、lock 模板与 dry-run 只是声明；不等于实现、GPU 验证、research-ready 或 business-ready。
- GPU verified 单次运行仍不代表 3 training seeds 的可比基线；本节绝不自动更新 B2/B3。

能力登记结构/文件审计通过；不是 GPU 测试结果，也不是 offline test 实时执行凭据。

### 五维能力计数

| 维度 | missing | implemented | unverified | verified |
| --- | --- | --- | --- | --- |
| data | 5 | 9 | 0 | 0 |
| train | 5 | 9 | 0 | 0 |
| eval | 5 | 9 | 0 | 0 |
| comparison | 6 | 8 | 0 | 0 |
| export | 6 | 8 | 0 | 0 |

### 逐篇能力矩阵

| recipe | data | train | eval | comparison | export |
| --- | --- | --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | missing | missing | missing | missing | missing |
| `llm/grpo/qwen2-0-5b-numinamath` | implemented | implemented | implemented | implemented | implemented |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | missing | missing | missing | missing | missing |
| `llm/grpo/qwen2-5-3b-advanced-reward` | implemented | implemented | implemented | missing | missing |
| `llm/sft/llama3-8b-xlam-function-calling` | implemented | implemented | implemented | implemented | implemented |
| `llm/sft/rapidfire-multiconfig` | implemented | implemented | implemented | implemented | implemented |
| `llm/sft/starcoder1b-fim` | implemented | implemented | implemented | implemented | implemented |
| `vlm/dpo/smolvlm-rlaif-v` | implemented | implemented | implemented | implemented | implemented |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | missing | missing | missing | missing | missing |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | implemented | implemented | implemented | implemented | implemented |
| `vlm/sft/granite-vision-line-comparison` | implemented | implemented | implemented | implemented | implemented |
| `vlm/sft/paligemma2-refcoco-grounding` | implemented | implemented | implemented | implemented | implemented |
| `vlm/sft/qwen2-vl-7b-chartqa` | missing | missing | missing | missing | missing |
| `vlm/sft/smolvlm-chartqa` | missing | missing | missing | missing | missing |

### 每篇具体缺失与实现边界

- `agent/grpo/wordle-openenv-hf-jobs`：**data [missing]** 缺 Wordle 环境和轨迹 schema、答案词/对局级 train/val/test 划分；在线生成不是无泄漏证明。；**train [missing]** 缺 OpenEnv 交互 rollout、reward 与 trainer 桥接、超时/异常会话回收；HF Jobs/papermill 入口不算本仓实现。；**eval [missing]** 缺独立词表/对局种子的 solve-rate、平均步数与非法动作评估。；**comparison [missing]** 缺 zero-shot 同对局协议与 multi-training-seed aggregate、置信区间及成本对照。；**export [missing]** 缺策略/环境版本打包、全新进程复载与对局回归。
- `llm/grpo/qwen2-0-5b-numinamath`：**data [implemented]** prepare_splits 真切分 source train[0:500]/val[500:1000]/test[1000:1500]；内容/问题跨 split 去重与 manifest；offline test 使用本地构造数据，真实数据/GPU 未测。；仓内证据 `src/recipe_lab/math_task.py`, `tests/test_math_task.py`；**train [implemented]** 真实 GRPOConfig/GRPOTrainer、format_reward/accuracy_reward、math_verify；gold 解析错误拒绝执行；只支持纯文本全参数 math-grpo-v1；mock/offline test 不证明 GPU 成功。；仓内证据 `src/recipe_lab/math_task.py`, `runner/train.py`, `tests/test_math_task.py`；**eval [implemented]** 真实单 checkpoint held-out 与 base zero-shot 共用 prompt/test/greedy/seed；缺训练 checkpoint 不回退 base；拒绝用重复 decode 冒充训练多 seed；GPU 未测。；仓内证据 `src/recipe_lab/math_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_math_task.py`；**comparison [implemented]** 至少3互异训练seed、共享基座且固定data/eval seed、隔离checkpoint、严格关联证据与指标重算、均值/样本方差/Student-t95区间；验收目标不自动回填，发布门禁共用强校验；offline通过，真实GPU多seed未测。；仓内证据 `src/recipe_lab/suite_task.py`, `src/recipe_lab/provenance.py`, `runner/suite.py`, `tests/test_suite.py`, `tests/test_provenance.py`, `tests/test_baseline_evidence.py`；**export [implemented]** 完整模型/tokenizer/配方/锁/参考预测打包、固定新进程offline重载复评、文件/身份/容差回归；C405内容复核且回滚单独确认。离线新Python进程测试仅注入依赖，未做GPU复载；需缓存数据，不是安装镜像，许可pending。；仓内证据 `src/recipe_lab/export_task.py`, `runner/export.py`, `tests/test_export.py`
- `llm/grpo/qwen2-0-5b-vllm-colocate`：**data [missing]** 缺与数学样板共享的冻结索引/gold 隔离适配；同数据集名称不代表同配置划分。；**train [missing]** 缺 use_vllm 同配置开关、colocate 装配、LoRA/权重同步与 rollout policy 身份验证；不能借用 math-grpo-v1 fallback。；**eval [missing]** 缺同 checkpoint/test/采样参数的质量指标及吞吐、显存与同步滞后评估。；**comparison [missing]** 缺只改变 vLLM 开关的同配置 on/off 对照与 3 train seeds 聚合，不能把两个配方名当消融。；**export [missing]** 缺 adapter/base 合并或复载策略、vLLM/HF 输出一致性与权重同步回归。
- `llm/grpo/qwen2-5-3b-advanced-reward`：**data [implemented]** 自建 gsm8k-reward-v1（非上游复现）：严格 question/answer 两列 schema；gold 取自恰一个 #### 之后的单个数字并规范化为 Decimal 字符串，缺失/多个分隔符/不可解析一律 DatasetValidationError，推理文本永不进 prompt；官方 train 按 sha256(data_seed:规范化 question) 取模 100 切 90/10 train/validation，官方 test 1319 题整体 held-out，行数与官方声明不符、test 内重复、与 train/validation 题目重叠即拒绝，train/validation 全局去重并登记丢弃索引；manifest 只存 hash。仅离线合成数据验证，未加载真实 GSM8K、revision 与许可未确认。；仓内证据 `src/recipe_lab/gsm8k_task.py`, `tests/test_gsm8k.py`；**train [implemented]** 全参数 GRPO（trl==0.14.0，fp32 权重 + bf16 autocast，use_vllm=false）：match_format_exactly(3.0)、match_format_approximately(逐 token ±0.5，上限 2.0)、check_answer_correctness(3.0/1.5/0.5/-0.5)、check_numbers_extraction(1.5) 四函数完整实装且等权求和，分档条件与标签为本仓选择；gold 不可解析时抛错而非给分；GRPOConfig 21 个超参全部显式冻结（缺任一即拒绝），API signature 与实际应用值、预算 full150/smoke5、prompt 最大 token 不得触发静默截断、frozen lock/代码身份均校验。仅 fake runtime 离线验证，无 GPU 实跑。；仓内证据 `src/recipe_lab/gsm8k_task.py`, `src/recipe_lab/plan.py`, `runner/train.py`, `tests/test_gsm8k.py`；**eval [implemented]** 官方 test 逐题 greedy 单样本；独立严格 Decimal verifier（不依赖 math_verify、不复用 reward 的 0.5 近似档）重算 pass@1 / format_compliance / reward_hacking_rate，并给出近似档占比、soft-format-only 占比与逐 reward 均值诊断；zero-shot 只加载冻结 base、训练评估必须有 completed checkpoint 且无 base fallback，拒绝重复 decode 冒充多 seed。reward_hacking 只是诊断信号，不能单独证明或排除；GPU 未测，0 verified。；仓内证据 `src/recipe_lab/gsm8k_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_gsm8k.py`；**comparison [missing]** 缺四 reward 逐项消融（drop-one × 至少三独立训练 seed）、同预算 zero-shot/GRPO 对照与 multi-training-seed aggregate；四函数等权相加未经消融也未量纲归一化。未注册 task_router，发布级套件 unavailable。；**export [missing]** 缺模型、reward 配置、tokenizer 打包与 HF offline 新进程复载、官方 test 回归；未注册 task_router，发布级导出 unavailable。
- `llm/sft/llama3-8b-xlam-function-calling`：**data [implemented]** 一手数据卡字段 query/tools/answers 均按严格 JSON schema 校验：工具重名、answers.name 不在 tools、多余参数、缺 required、类型不符、未知列一律拒绝；规范化内容 hash 80/10/10 确定性划分，全局重复（含跨 split）拒绝，空划分不偷偷补救；test shared held-out 最多300条。自建协议非上游复现；真实 gated 数据、revision 冻结与 2024-07 重生成条目影响未验收，许可 research-only 且商业使用未确认。；仓内证据 `src/recipe_lab/xlam_task.py`, `tests/test_xlam.py`；**train [implemented]** Trainer + 自建 completion-only collator：只对 <calls> 严格JSON数组 </calls> + EOS 计算 loss，prompt 部分 label=-100，分别编码避免边界合并；上游 SFTTrainer 把 gold 拼进 dataset_text_field 并对全文做损失，差异已登记。nf4/bf16/doublequant QLoRA、7 个 LoRA target 全匹配、base 冻结/trainable 非空、API signature 与实际应用值、预算 full150/smoke5、frozen lock/代码身份/adapter safetensors 张量值复载均校验。仅合成离线测试，无 GPU/真实数据实跑。；仓内证据 `src/recipe_lab/xlam_task.py`, `src/recipe_lab/plan.py`, `runner/train.py`, `tests/test_xlam.py`；**eval [implemented]** held-out 单 checkpoint 与 base zero-shot 共用完整 tools prompt 与 greedy 协议；gold 不进生成输入；只用 json.loads 解析 <calls>，Python str(dict)/eval 注入文本一律拒绝，绝不执行生成的调用。json_parse_rate / schema_valid_rate / tool_call_exact_match / argument_f1 逐题重算；上游只有 3 条 do_sample 定性样例，无 held-out/指标。checkpoint 来源、recipe/fingerprint/锁/权重与逐题预测强绑定；发布级逐题重算器按task_router分派到xlam_suite.py，不用数学或FIM协议。未实现 pass@1 / 真实 API 执行正确率 / 隔离沙箱，GPU 未测，0 verified。；仓内证据 `src/recipe_lab/xlam_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_xlam.py`；**comparison [implemented]** xlam-fc-v1专用三互异训练seed套件：共享冻结base与内容hash隔离held-out test、独立adapter checkpoint、逐题从raw query/tools/answers重建prompt与gold并重算四指标（不信任摘要分数）、均值/样本方差/SEM/Student-t95区间、配对增益；不回填expected/tolerance、不自动晋级B2。C306/C307按task_router精确分派，不借用数学1500题或FIM PSM协议。离线fake runtime证据只验内部一致性；真实CUDA三seed、同预算FIM/严格JSON输出协议消融、失败工具调用成本、许可与业务验收仍缺。；仓内证据 `src/recipe_lab/xlam_suite.py`, `src/recipe_lab/task_router.py`, `runner/suite.py`, `tests/test_xlam_release.py`, `tests/test_task_router.py`；**export [implemented]** adapter-only导出包：冻结base revision/量化/eager/tokenizer identity、严格JSON输出协议与base_weights_bundled=false显式声明；配方快照逐字节保持训练内容身份，锁用虚拟锚点自包含；HF offline新进程真实调用xlam_task.run_evaluation复载复评，文件/身份/逐题输入/容差内容级复核，C405按任务分派。离线测试注入受控worker；无GPU复载，依赖缓存gated base与research-only数据，不是安装镜像，商业许可pending，隔离沙箱下真实API执行正确率仍缺。；仓内证据 `src/recipe_lab/xlam_export.py`, `src/recipe_lab/task_router.py`, `runner/export.py`, `tests/test_xlam_release.py`, `tests/test_task_router.py`
- `llm/sft/rapidfire-multiconfig`：**data [implemented]** 自建 rapidfire-sft-v1：Bitext 5 列严格 schema（flags/instruction/category/intent/response），未知列/空 gold 拒绝；规范化问题 hash 80/10/10 使同题所有生成变体落同一 split，规范化内容全局去重且拒绝跨 split 重复；空 split 拒绝。上游 train range(64)/eval range(50,60) 泄漏只作 [upstream] 元信息，不参与执行。真实 revision 与 26872 行数据未实测，数据卡未给 license。；仓内证据 `src/recipe_lab/rapidfire_task.py`, `tests/test_rapidfire.py`；**train [implemented]** transformers.Trainer + completion-only collator 的 fp16 LoRA SFT（非上游未固定版本的 rapidfireai 并发调度）；4 组 lr×scheduler×rank/alpha/target_modules 由 [[configs]] 精确声明，config_recipe 只覆盖 train/peft/config_id/experiment_id 并逐配置隔离输出目录与证据；API signature/实际应用值、base 冻结、safetensors 张量复载、full=64/smoke=5 预算、锁与代码身份全绑定。顺序执行，不宣称 16-24× 加速复现；仅隔离离线合成证据，GPU 0 verified。；仓内证据 `src/recipe_lab/rapidfire_task.py`, `runner/train.py`, `src/recipe_lab/plan.py`, `tests/test_rapidfire.py`；**eval [implemented]** 同冻结 base/adapter/tokenizer/test 与单样本 greedy 协议，gold 不进 prompt，首个 EOS 截断；纯 Python 无 stemmer LCS ROUGE-L 与 exact_match 逐题重算（不引入未冻结 evaluate/rouge_score，数值与上游 use_stemmer=True 不可比）；checkpoint 来源、fingerprint、config_id、权重/预测/manifest 强绑定；base zero-shot 与训练 checkpoint 角色分离。上游从未报告任何定量数值；validation/test 评估 split 分离且绑定 eval_stage，C306/C307/C405 按 task_router 分派到 rapidfire_suite/rapidfire_export 的专用内容校验器，未知或冲突任务仍失败关闭。；仓内证据 `src/recipe_lab/rapidfire_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_rapidfire.py`；**comparison [implemented]** rapidfire_suite.py 严格聚合 4 配置 × 至少 3 个互异独立训练 seed：逐题从 eval_manifest 的 raw instruction/response 重建 prompt 与 gold，纯 Python 无 stemmer ROUGE-L + exact_match 重算；只用 validation 的 paired 均值选配置（平局按 config_id 字典序），test 仅在选中配置上验收一次；均值/样本方差/SEM/Student-t95 与 sequential 时长/显存/步数成本对照，明确不宣称与上游并发吞吐可比；逐配置 checkpoint 的 recipe/fingerprint/code/lock/seed/config_id/权重/预测强校验。不回填 expected/tolerance、不自动晋级 B2；只有隔离离线合成证据，GPU 0 verified。；仓内证据 `src/recipe_lab/rapidfire_suite.py`, `runner/suite.py`, `tests/test_rapidfire_release.py`；**export [implemented]** 选中配置的 adapter-only fp16 LoRA 包：显式声明冻结 gpt2 base revision / tokenizer identity / eager / 无量化 / completion-only prompt 协议 / 无 stemmer ROUGE-L 不可比性 / base_weights_bundled=false / license pending；recipe-snapshot 与训练时 r.data 逐字节一致，锁用虚拟锚点自包含；HF offline 全新进程真实调用 rapidfire_task.run_evaluation 在 test split 复评，不复制 JSON 冒充复载。真实 recipe 未预声明 evaluation.export_tolerance 时 --full 失败关闭，不回填容差；只有受控 worker 离线证据，无 GPU 复载。；仓内证据 `src/recipe_lab/rapidfire_export.py`, `runner/export.py`, `tests/test_rapidfire_release.py`
- `llm/sft/starcoder1b-fim`：**data [implemented]** repo_id/file_path/content严格schema，全源有限扫描与全局raw/normalized去重；repo哈希80/10/10隔离；PSM/SPM、special IDs、文件内分块padding与可重建mask/manifest。自建协议非上游复现，真实数据/许可/污染尚未验收。；仓内证据 `src/recipe_lab/fim_task.py`, `tests/test_fim.py`；**train [implemented]** 显式Trainer+nf4/bf16/doublequant+LoRA，候选q_attn不强制，base冻结/非空trainable、预算full2000/smoke5及API/执行身份检查；独立文件causal-all-nonpadding，自建eager/no remote code；safetensors张量keys/shape/值复核；仅mock离线验证，无GPU证据。；仓内证据 `src/recipe_lab/fim_task.py`, `runner/train.py`, `tests/test_fim.py`；**eval [implemented]** 同冻结base/adapter/量化/tokenizer/test窗口/PSM/greedy协议，gold不入input，首EOS/严格保留空白的文本与token exact match；completed checkpoint/权重来源校验，安全张量读取与复载值核对。发布级重算器按task_router分派到fim_suite，未知任务失败关闭。未实现pass@k/compile/sandbox隔离执行评估，exact match不执行代码；GPU未测，0 verified。；仓内证据 `src/recipe_lab/fim_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_fim.py`, `tests/test_fim_release.py`；**comparison [implemented]** code-fim-v1专用三互异训练seed套件：共享base/zero-shot、隔离adapter checkpoint、固定repo隔离test与PSM/greedy协议、逐题输入/gold内容重算、均值/样本方差/Student-t95区间、不回填expected/tolerance、不自动晋级B2。离线fake runtime证据只验内部一致性；真实CUDA三seed、FIM on/off消融、许可证/重复代码偏差、GPU与业务验收仍缺。；仓内证据 `src/recipe_lab/fim_suite.py`, `src/recipe_lab/task_router.py`, `runner/suite.py`, `tests/test_fim_release.py`, `tests/test_task_router.py`；**export [implemented]** adapter-only导出包：冻结base revision/量化/eager/tokenizer identity与base_weights_bundled=false显式声明，配方快照逐字节保持训练内容身份，包内锁用虚拟锚点自包含；HF offline新进程真实调用fim_task.run_evaluation复载复评，文件/身份/容差内容级复核，C405按任务分派。离线测试注入受控worker；无GPU复载，依赖缓存base与数据，不是安装镜像，许可pending，隔离代码执行回归仍缺。；仓内证据 `src/recipe_lab/fim_export.py`, `src/recipe_lab/task_router.py`, `runner/export.py`, `tests/test_fim_release.py`, `tests/test_task_router.py`
- `vlm/dpo/smolvlm-rlaif-v`：**data [implemented]** 自建 smolvlm-dpo-v1：严格 prompt/chosen/rejected/images schema；图像原始 mode+tobytes() SHA256 的 80/10/10 同图隔离，规范化内容全局去重与空 split 拒绝；与 MPO 共享偏好 split 算法与 data_seed，但 backbone 不同故 cross_backbone_attribution_valid=false。仅离线合成图像验证，未加载真实 rlaif-v、revision/许可未确认。；仓内证据 `src/recipe_lab/dpo_task.py`, `src/recipe_lab/preference_common.py`, `tests/test_preference.py`；**train [implemented]** 普通 Trainer + 自建显式 beta=0.1 sigmoid DPO loss（不依赖未固定 TRL 源码）；policy/ref 仅在 completion token 上计 logprob，image placeholder 与 padding label=-100；nf4/DoRA、显式冻结 ref、policy/ref 参数空间互斥校验；full=150/smoke=5。仅 fake runtime 离线验证，无 CUDA 实跑。；仓内证据 `src/recipe_lab/dpo_task.py`, `runner/train.py`, `tests/test_preference.py`；**eval [implemented]** held-out 逐题隐式奖励 pairwise：win_rate/reward_accuracy 从 policy/ref logprob 重算，不是外部 RM 或人工偏好；chosen/rejected 绝不进生成 prompt；长度偏置分桶登记。指标不等于人类偏好一致率或业务验收；GPU verified 0。；仓内证据 `src/recipe_lab/dpo_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_preference.py`；**comparison [implemented]** 三互异独立训练 seed 的 held-out 聚合；逐题从 eval_manifest 重建偏好对并重算隐式奖励指标，Student-t95；明确 cross_backbone_attribution_valid=false，不与 MPO 篇做算法消融。不回填 expected/tolerance、不自动晋级 B2。仅离线合成证据。；仓内证据 `src/recipe_lab/dpo_suite.py`, `runner/suite.py`, `tests/test_preference.py`；**export [implemented]** adapter-only 包：冻结 base/processor identity、显式 beta/双层 mask、base_weights_bundled=false、ref_weights_bundled=false、license pending；HF offline 新进程复载复评。真实 recipe 未预声明 export_tolerance，--full 失败关闭。；仓内证据 `src/recipe_lab/dpo_export.py`, `runner/export.py`, `tests/test_preference.py`
- `vlm/grpo/qwen2-5-vl-3b-visual-reasoning`：**data [missing]** 缺单数 image 列与 processor.apply_chat_template 预渲染字符串 prompt 的真实适配；图像/题目级冻结隔离。；**train [missing]** 缺 VLM GRPO 视觉 rollout/PEFT/reward 实装；gold/prediction 解析失败关闭，不可借用纯文本数学路由。；**eval [missing]** 缺图像 held-out 推理正确率、格式合规和失败预测明细。；**comparison [missing]** 缺同图像/prompt/test 的 zero-shot、reward 消融及 3 train seeds aggregate。；**export [missing]** 缺视觉 processor/image 输入协议与 adapter 打包复载、推理回归。
- `vlm/mpo/qwen2-5-vl-3b-mpo`：**data [implemented]** 自建 qwen25vl-mpo-v1：与 DPO 共享偏好 split 算法与 data_seed（图像 hash 80/10/10），schema/去重/空 split 失败关闭相同；cross_backbone_attribution_valid=false。仅离线合成证据，未加载真实 rlaif-v。；仓内证据 `src/recipe_lab/mpo_task.py`, `src/recipe_lab/preference_common.py`, `tests/test_preference.py`；**train [implemented]** 普通 Trainer + 自建 sigmoid/bco_pair/sft 加权和（默认 0.8/0.2/1.0）；config_recipe 派生 full/sigmoid_only/no_sft 三消融且只在同一 Qwen backbone 内；如实登记 bf16=False 与量化 compute dtype=bfloat16 矛盾，不静默修正；冻结 ref、DoRA/nf4。仅 fake runtime，无 CUDA 实跑。；仓内证据 `src/recipe_lab/mpo_task.py`, `runner/train.py`, `tests/test_preference.py`；**eval [implemented]** held-out 隐式奖励 pairwise 与三损失分量日志；chosen/rejected 不进生成 prompt；指标不是外部 RM/人工偏好。GPU verified 0。；仓内证据 `src/recipe_lab/mpo_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_preference.py`；**comparison [implemented]** 3 消融 × 3 独立训练 seed 网格；逐题隐式奖励重算与 Student-t95；不把 DPO 篇当算法对照（cross_backbone_attribution_valid=false）。不回填 expected/tolerance、不自动晋级 B2。；仓内证据 `src/recipe_lab/mpo_suite.py`, `runner/suite.py`, `tests/test_preference.py`；**export [implemented]** 选中消融 config 的 DoRA/4bit adapter-only 包；声明三损失身份、精度矛盾、ref 不打包、license pending；HF offline 复载。真实 recipe 未预声明 export_tolerance，--full 失败关闭。；仓内证据 `src/recipe_lab/mpo_export.py`, `runner/export.py`, `tests/test_preference.py`
- `vlm/sft/granite-vision-line-comparison`：**data [implemented]** 自建 granite-linecomparison-v1 而非上游 notebook 复现：严格 4 列 schema + predicate=LineComparison 过滤；图像原始 mode + tobytes() 的 SHA256 做 80/10/10，同图所有问答变体同 split；规范化 (question,answer) 全局去重且拒绝跨 split 重复、空 split 拒绝、不随机补救；source_manifest 绑定 split_hashes / image_split_owner / data_seed / dataset revision 与内容 hash。上游只有 50/50 无 validation，本仓三段隔离不冒充上游结果。仅限离线合成图像对象验证，未加载真实 Geoperception、revision 与许可未确认。；仓内证据 `src/recipe_lab/granite_task.py`, `tests/test_granite.py`；**train [implemented]** 普通 transformers.Trainer + 自建 collator（不用上游 SFTTrainer/skip_prepare_dataset）；prompt/full 两次 processor.apply_chat_template 编码并做 token 级前缀校验，gold 只在 full 序列、绝不进生成 prompt；显式 image_token_id 与 assistant span 外 token 双层 label=-100，padding/图像 placeholder 一律屏蔽；nf4/DoRA/bf16/eager、视觉塔跳过量化、动态筛 language_model+_proj 且命中非空，base 全冻结、safetensors adapter keys/shape/值严格复核；冻结 processor identity / seed / revision / 代码与锁字节，full=150 smoke=5。仅 fake runtime 离线验证，无 CUDA/GPU 实跑、无 completed 证据。；仓内证据 `src/recipe_lab/granite_task.py`, `runner/train.py`, `tests/test_granite.py`；**eval [implemented]** held-out test 逐题 greedy + 首个 EOS 截断 + strip，exact_match / normalized_exact_match 逐题重算（NFKC+casefold+空白折叠），空生成不给分；eval_manifest 从 raw image/question/answer 重建 prompt 与 gold 并绑 processor identity / source_manifest hash；单 checkpoint greedy 不冒充训练多 seed 方差，zero-shot 只加载 base、训练评估必须有 completed checkpoint 且无 base fallback。指标只衡量文本一致性，不解释几何语义、不等于人工判分或业务验收；上游只有 test_idx=20 单样本定性，无数值可比。GPU 未测，0 verified。；仓内证据 `src/recipe_lab/granite_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_granite.py`；**comparison [implemented]** 三互异独立训练 seed 的 held-out 聚合：逐题从 eval_manifest 的 question/answer 重建 gold 并用 score_prediction 重算 exact_match / normalized_exact_match，不信任摘要分数；共享同图隔离的冻结 test 与 processor identity；均值/样本方差/SEM/Student-t95；不回填 expected/tolerance、不自动晋级 B2。指标只衡量文本一致性，不解释几何语义、不等于人工判分或业务验收。task_router 按精确 recipe_id 分派，不借用数学 1500 题 / FIM PSM / xLAM JSON / RapidFire ROUGE-L 协议。仅离线合成证据，GPU verified 0。；仓内证据 `src/recipe_lab/granite_suite.py`, `runner/suite.py`, `tests/test_granite_release.py`；**export [implemented]** DoRA/4bit adapter-only 包：显式声明冻结 base/processor identity/chat template、显式 image_token_id + assistant-only 双层 mask、RGB 图像输入与 exact_match 协议、base_weights_bundled=false、generated_code_executed=false、license pending；recipe-snapshot.json 与训练时 effective_recipe 逐字节一致，锁用虚拟锚点自包含；HF offline 新进程真实调用 granite_task.run_evaluation 在 held-out 复载复评。真实 recipe 未预声明 export_tolerance，--full 失败关闭；Geoperception 与基座许可未确认。仅离线合成证据，GPU verified 0。；仓内证据 `src/recipe_lab/granite_export.py`, `runner/export.py`, `tests/test_granite_release.py`
- `vlm/sft/paligemma2-refcoco-grounding`：**data [implemented]** 图像原始 mode+tobytes() SHA256 的 80/10/10 同图隔离、规范化(caption,bbox)全局去重与空 split 拒绝；coco_xywh→xyxy→1024 桶 <loc####> round-trip IoU 硬校验，非法/退化/越界框与 caption 注入拒绝；冻结原图 width/height，不做在线 resize/augmentation。仅离线合成证据。；仓内证据 `src/recipe_lab/paligemma_task.py`, `runner/train.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_paligemma.py`；**train [implemented]** 普通 transformers.Trainer + 自建 completion-only collator（非 SFTTrainer）；prompt/prompt+suffix 两次 processor 编码并校验 token 级前缀；显式 image_token_id + suffix-only 双层 -100 mask；普通 LoRA（无 4bit/DoRA），冻结 processor identity、revision 与 8 个 target_modules。GPU verified 0。；仓内证据 `src/recipe_lab/paligemma_task.py`, `runner/train.py`, `tests/test_paligemma.py`；**eval [implemented]** held-out 逐题 greedy + 首个 EOS 截断；严格单框 loc 反解，重算 bbox_iou / iou_at_0_5 / parse_rate / exact_match；gold 绝不进生成 prompt；zero-shot 只加载冻结 base、训练评估必须有 completed checkpoint 且无 base fallback。指标不冒充 mAP/AP 或业务验收，不执行生成代码。；仓内证据 `src/recipe_lab/paligemma_task.py`, `runner/eval.py`, `runner/zero_shot.py`, `tests/test_paligemma.py`；**comparison [implemented]** 三互异独立训练 seed 的 held-out 聚合；逐题从 eval_manifest 的 raw COCO bbox + 冻结宽高 + caption 重建 detection gold 与 xyxy，用 score_prediction 对 generated_text 重算四指标；均值/样本方差/SEM/Student-t95、paired delta；不回填 expected/tolerance、不自动晋级 B2。仅离线合成证据。；仓内证据 `src/recipe_lab/paligemma_suite.py`, `runner/suite.py`, `tests/test_paligemma_release.py`；**export [implemented]** 普通 LoRA adapter-only 包：冻结 base/processor identity、统一 prompt 模板、coco_xywh/xyxy/1024 桶 loc 协议、RGB 图像输入与单框 IoU 声明、base_weights_bundled=false、generated_code_executed=false、license pending；recipe-snapshot.json 与训练时 effective_recipe 逐字节一致，锁用虚拟锚点自包含；HF offline 新进程真实调用 paligemma_task.run_evaluation 复载复评。真实 recipe 未预声明 export_tolerance，--full 失败关闭；RefCOCO/COCO/Flickr 与基座许可未确认。；仓内证据 `src/recipe_lab/paligemma_export.py`, `runner/export.py`, `tests/test_paligemma_release.py`
- `vlm/sft/qwen2-vl-7b-chartqa`：**data [missing]** 缺真实 ChartQA train/val/test 加载、图像级去重/污染检查、视觉 token mask 与数字答案规范化。；**train [missing]** 缺 Qwen2-VL 4bit/LoRA（use_dora=false）与视觉 collator 适配；不能与 SmolVLM DoRA 混同。；**eval [missing]** 缺 ChartQA relaxed accuracy、数字容差与 human/augmented 分组指标、未污染集验证。；**comparison [missing]** 缺同 split/template zero-shot 与 3 train seeds 聚合；和 SmolVLM 的模型差异/污染必须独立说明。；**export [missing]** 缺 Qwen2-VL 4bit/LoRA、processor 打包复载与 ChartQA 回归。
- `vlm/sft/smolvlm-chartqa`：**data [missing]** 缺真实 image/对话适配与切分；SmolVLM-Instruct 已接触 ChartQA 的仓内污染警告必须保留，需未污染外部集。；**train [missing]** 缺 SmolVLM 4bit/DoRA（use_dora=true）装配与 assistant/image token mask；与 Qwen LoRA 分别验收。；**eval [missing]** 缺 relaxed accuracy、human/augmented 分组与未污染集评估；ChartQA 分数只能作带污染声明的教学对照。；**comparison [missing]** 缺 DoRA 消融和 3 train seeds 聚合；不得将已接触 ChartQA 的结果宣称无污染泛化或纯算法胜出。；**export [missing]** 缺 SmolVLM 4bit/DoRA/processor 打包复载与未污染集指标回归。

### 方法缺口：taxonomy 已登记但没有 recipe

- `cpo`：trainer `CPOTrainer`，tier `experimental`；配方数 0，不能计入能力覆盖。
- `distillation`：trainer `DistillationTrainer`，tier `stable`；配方数 0，不能计入能力覆盖。
- `gkd`：trainer `GKDTrainer`，tier `experimental`；配方数 0，不能计入能力覆盖。
- `gspo`：trainer `GRPOTrainer`，tier `variant`；配方数 0，不能计入能力覆盖。
- `kto`：trainer `KTOTrainer`，tier `stable`；配方数 0，不能计入能力覆盖。
- `online-dpo`：trainer `OnlineDPOTrainer`，tier `experimental`；配方数 0，不能计入能力覆盖。
- `orpo`：trainer `ORPOTrainer`，tier `experimental`；配方数 0，不能计入能力覆盖。
- `ppo`：trainer `PPOTrainer`，tier `removed`；配方数 0，不能计入能力覆盖。
- `prompt-tuning`：trainer `Trainer`，tier `external`；配方数 0，不能计入能力覆盖。
- `reward-modeling`：trainer `RewardTrainer`，tier `stable`；配方数 0，不能计入能力覆盖。
- `rloo`：trainer `RLOOTrainer`，tier `stable`；配方数 0，不能计入能力覆盖。
- 上述列表由当前 root 的 families − recipes.method.family 自动计算；不是按实现数计算。
- 未纳入 taxonomy 的独立模块：`audio`、`cpt-dapt`、`video`。
- 数据构建/eval/发布是跨方法管线，不是空缺 trainer family；已有视觉 DPO/MPO 也不等于文本偏好覆盖。

### Scope 模块覆盖范围

| 模块 | 范围状态 | taxonomy 边界 | why |
| --- | --- | --- | --- |
| CPT / DAPT 继续预训练与领域适应 | backlog | 未纳入 taxonomy | 未纳入当前 taxonomy；需语料许可、文档去重、按来源隔离、长文本打包与 perplexity/领域任务回归，不能用现有 SFT 覆盖冒充。 |
| 通用文本 SFT | in_scope | sft | 现有 xLAM/FIM 是专用任务；还需通用对话、assistant-only mask、packing、截断回归及指令跟随评估。 |
| 文本偏好优化 | in_scope | dpo, mpo, kto, orpo, cpo | 现有 DPO/MPO 配方都是 VLM，不代表文本偏好已覆盖；需共享 backbone/数据/预算与独立偏好评估。 |
| RM 奖励建模 | backlog | reward-modeling | family 已登记但无配方；需偏好对、标注质量、reward 校准、泛化与 reward hacking 检查。 |
| online RL 与 agent RL | in_scope | grpo, rloo, online-dpo, gspo | 先验证数学可核验 reward，再支持 rollout 权重同步、环境生命周期和独立对局；现有 GRPO 登记不代表其余算法已实现。 |
| 蒸馏 | backlog | distillation, gkd | family 已登记但无配方；需 teacher revision、采样与成本、生成数据追踪、student 对照和许可。 |
| 多模态图片 | in_scope | sft, dpo, mpo, grpo | 需 image/processor/collator 实装、视觉 token mask、图像级隔离及 ChartQA/grounding 污染检查。 |
| 多模态音频 | backlog | 未纳入 taxonomy | 未纳入当前 taxonomy；需音频采样率、切片、说话人隔离、转写/语音任务指标，图片 VLM 不能代替。 |
| 多模态视频 | backlog | 未纳入 taxonomy | 未纳入当前 taxonomy；需帧采样、时间标注、视频级隔离与时序评估，静态图像流程不能代替。 |
| 数据构建 | in_scope | 跨方法管线 | 跨方法管线而非 trainer family；覆盖 schema、许可、gold 剥离、去重、冻结 manifest、合成数据溯源与 split 泄漏门禁。 |
| eval 评估体系 | in_scope | 跨方法管线 | 跨方法管线；需同协议 zero-shot、held-out、至少 3 个独立训练 seed 的方差/置信区间和失败样本。 |
| 发布与业务交付 | in_scope | 跨方法管线 | 跨方法管线；需打包、全新进程离线复载、指标回归、许可、成本/SLO、安全、owner、监控与回滚，不执行远端发布。 |
| 经典 PPO 全栈 | excluded | ppo | 当前仓内 taxonomy 将 ppo 标为 removed；不扩张本轮主线，未来需求变化时须单独决定，不据此推断上游时点。 |

### 实施顺序：依赖、前置条件与验收

| 波次 | dependencies | 前置条件 | 工作 | 验收 |
| --- | --- | --- | --- | --- |
| wave0 fail-closed 门禁 | 无 | 完整14篇登记、taxonomy 时点快照与代码/测试证据边界已核对。 | 严格校验 coverage、状态、impl/test 路径与 completed JSON；拒绝缺失 checkpoint 回退 base、假分数和 dry-run 晋级。 | 负例单测通过；registered、implemented、GPU verified、research-ready、business-ready 分开计数；缺证据直接阻断晋级。 |
| wave1 数学端到端样板 | wave0 | wave0 门禁验收；真实 model/dataset revision 与依赖锁定，确认 GPU 与预算。 | 复用 math_task.py：train[0:500]/val[500:1000]/test[1000:1500]、format/accuracy reward、math_verify、真实 train/eval/zero-shot；补多训练 seed 聚合及 export reload。 | offline test + 真实 zero-shot/smoke/full；至少 3 个独立 train seeds，各自 held-out 评估，aggregate 与全新进程复载通过后才称 research baseline。 |
| wave2 文本 SFT | wave0, wave1 | wave1 的冻结/训练/评估/evidence 协议可复用且已验收。 | 通用 SFT backbone；xLAM schema masker、FIM 切段打包、RapidFire 无泄漏 4 配置顺序编排（不宣称复现上游并发调度）。 | gold 不进 prompt；assistant/FIM mask 与 token 对齐；源码/对话级 train/val/test 隔离；任务指标与同预算 3 seeds 对照、隔离代码执行评估。 |
| wave3 视觉 SFT | wave1, wave2 | 文本 collator/packing、独立评估与 export reload 已验收，视觉 processor revision 冻结。 | 两篇 ChartQA 因决策门/基座预训练污染判为原理性阻塞、不计入适配进度；Granite granite-linecomparison-v1 五维代码已齐备（图像内容 hash 三段隔离、显式 image_token_id + assistant-only 双层 mask、冻结 processor identity、held-out 逐题重算；granite_suite 三独立训练 seed 逐题内容重算与 Student-t95 聚合；granite_export 的 DoRA/4bit adapter-only 包与 HF offline 新进程复载），task_router 已精确注册，但全部只有 macOS ARM 离线合成证据、无 CUDA 实跑；PaliGemma bbox/loc round-trip/IoU 仍待实装。 | 图像 token 不参与语言损失；bbox round-trip/IoU；图像级去重；ChartQA 污染声明与未污染外部集；每篇独立 held-out 指标及复载回归；Granite 五维须先冻结 revision 与可安装镜像、授权 Linux CUDA 三 seed 实跑、基于实跑预声明容差并完成许可与业务验收，发布级按 task_router 精确分派、不借用其他任务协议。 |
| wave4 DPO / MPO shared backbone | wave2, wave3 | 同一可用 SFT checkpoint、processor、偏好 split、reference 策略与预算冻结。 | smolvlm-dpo-v1 / qwen25vl-mpo-v1 五维代码已齐备：图像偏好 hash 隔离、显式 beta/三损失、冻结 ref、隐式奖励 pairwise 与发布级套件/导出；task_router 已精确注册。两篇基座不同，cross_backbone_attribution_valid=false，不能把指标差归因算法。 | DPO vs MPO 必须 shared backbone 才能算法归因；beta/loss/权重显式配置，3 seeds 配对对照、ref 无梯度且复载等价；当前仅离线合成证据，须冻结 revision 与 CUDA 三 seed 后才称 research baseline。 |
| wave5 vLLM / agent RL | wave1, wave3, wave4 | 数学 reward 与视觉输入协议通过；多 seed 聚合、权重/checkpoint 身份追踪可用。 | 同配置切换 use_vllm，验证 rollout 权重同步；Wordle 环境、会话回收与按词/对局隔离；VLM GRPO 单数 image 与预渲染 prompt。 | vLLM on/off 除开关外同配置、记录同步步数与吞吐/显存/质量；异常/超时无会话泄漏；独立对局 solve-rate，VLM reward 失败关闭。 |
| wave6 方法与模态扩展 | wave1, wave2, wave3, wave4, wave5 | 前序 shared backbone、数据冻结、聚合、复载协议全部验收；范围与许可单独审批。 | GSM8K 4 reward 的 data/train/eval 已由 wave5a 自建 gsm8k-reward-v1 离线实装（partial_implementation），仍补 drop-one 消融 comparison 与 export 发布级两维；按自动 family 缺口推进 KTO/RM/RLOO/online-DPO/GSPO/蒸馏/ORPO/CPO/GKD/prompt-tuning；CPT/DAPT、音视频另立模块。 | 每种方法有独立配方、offline tests、真实完成 evidence、3 seeds 对照与交付回归；未纳入 taxonomy 的模块先定契约，禁止用登记数宣称全生态覆盖。 |

### Pipeline 边界与验收条款

**数据构建/冻结 → zero train（zero-shot）→ smoke → full（至少 3 train seeds）→ held-out → aggregate → export reload → business**。
- 数据/冻结：许可与来源可追踪；model/dataset/processor/依赖真实 revision 锁定；schema、gold 剥离、去重、train/val/test manifest 与 hash 齐全。
- zero train：只加载 base，不是训练证据；与训练后模型共享 prompt、test IDs/hash、解码参数、预算和指标协议；若零训练已满足目标则记录不训练决策。
- smoke：少量真实训练验证 loss/reward/梯度、资源、保存/加载与失败路径；不代表 full，不产出研究结论。
- full：至少 3 个不同 training seeds 各自从同一冻结 base 独立训练，独立输出/日志/checkpoint；固定 data split/data seed；重复 greedy decode 不能代替训练方差。
- held-out：只用 val 选超参/checkpoint，test 不参与选型；各训练 seed 对应自己的 trained checkpoint；缺 checkpoint 必须报错，不回退 base；逐样本预测与指标可追踪。
- aggregate：校验训练/评估 JSON 的 recipe、kind、stage、seed、revision、config/test hash 与 checkpoint 身份；同协议 zero-shot 对照、均值/标准差/置信区间及预先声明的容差。
- export reload：模型/adapter/base、tokenizer/processor、模板、依赖、manifest 与许可打包；全新进程离线复载，检查 artifact hash 和指标容差；save_model 单独不算交付。
- business：research 门禁通过后再验成本、吞吐/显存、SLO、任务质量、安全/许可、owner、监控漂移和回滚；技术 verified 不自动成为业务验收。
- 证据门禁：implemented/verified 必须引用仓内 implementation 与 test 文件；verified 必须读 JSON 对象，stage=completed 且 kind 匹配，train/eval 必须真实 GPU/CUDA runtime；mock/dry-run/smoke/base 不得冒充 full/eval。
- 审计不是密码学执行证明：不能仅靠人工填写 JSON 认证 GPU；保留运行日志/artifact 并由独立流程复查，禁止用测试夹具作为真实实跑证据。
- 数学样板的当前有效切分以 math_task.py 为准（0:500 / 500:1000 / 1000:1500）；旧 recipe 中的声明或通用 fallback 不覆盖此事实。
- 本节只读报告，可由主线程直接拼接 `capabilities.render(root, recipes)`；不实现聚合器、export 工具或远端发布，不估工期。

## 逐篇 baseline 可行性判定（原理层）

> 由 `recipe-lab feasibility` 生成。代码实装、GPU 证据、**实验目标是否成立**是三件不同的事；本节只判定第三件，不因五维 implemented 或 completed JSON 而放行。

**当前 0 篇可直接作为 research / business baseline：eligible 0/14；pending_freeze 8、partial_implementation 1、not_implemented 3、blocked_zero_train 1、contaminated_pretraining 1。**

两类原理性阻塞（`blocked_zero_train` / `contaminated_pretraining`）即使补齐代码与 CUDA 实跑证据也不晋级：前者上游自承零训练基线即可达标，后者基座本就在评估集上训练过。必须通过改写实验目标、换未污染数据、重核一手材料并显式改写 verdict 解除，不能用新增 evidence 或抬高 B 状态绕过。

| verdict | 含义 |
| --- | --- |
| `eligible` | 实验目标、数据与评估协议成立；补齐代码/冻结/实跑/许可后可直接进入 B2 证据门禁。当前 0 篇。 |
| `blocked_zero_train` | 一手材料自承零训练基线即可达标（decision.gate=blocked）；微调不改变业务结论，投入训练预算不成立。即使补齐代码与 GPU 证据也不晋级，除非先改写实验目标并重新判定。 |
| `contaminated_pretraining` | 基座已在评估数据集上训练过，held-out 不构成泛化验证；作者自承只是 educational exercise。必须换未被污染的外部集并重做判定，否则不得用于泛化结论。 |
| `not_implemented` | 任务逻辑完全缺失；既没有可复现的训练/评估实现，也没有可核查的对照协议。 |
| `partial_implementation` | data/train/eval 三维已有自建协议、真实任务代码与隔离离线测试，但 comparison/export 两维未实装、发布级路由未注册；仍无冻结输入/环境、CUDA 多 seed 实跑与预声明容差，不能算 eligible 或 pending_freeze。 |
| `pending_freeze` | 原理上可作为 baseline 且五维代码已实装，但冻结输入/环境、CUDA 多 seed 实跑、预声明容差、导出复载与许可确认全未完成；GPU verified 0。 |

| recipe | verdict | 一手结论 | 首要阻塞 | 转化所需 |
| --- | --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | `not_implemented` | OpenEnv 在线交互 rollout、词表/对局级隔离、独立对局 solve-rate 与会话回收全部缺失；HF Jobs / papermill 入口不等于本仓执行层。 | 无轨迹 schema 与答案词/对局级 train/val/test 划分 | 实现环境与按对局隔离的冻结数据协议; 实现 rollout、reward 与异常回收; 三独立训练 seed 的同词表 held-out 对照与复载 |
| `llm/grpo/qwen2-0-5b-numinamath` | `pending_freeze` | 五维代码与隔离离线测试齐备，训练/评估协议自建且严格；但从未在 CUDA 上实跑，model/dataset/source revision 与可安装环境锁未冻结，expected/tolerance/export_tolerance 未基于实跑预声明。 | GPU verified 0，无真实 completed training/evaluation 证据 | 授权 Linux CUDA 环境与预算，锁定全部 revision 与依赖; 真实 zero-shot/smoke/full + 三 seed 聚合 + 导出复载; 基于首轮实测预声明容差后再独立验收，不用本轮均值自证 |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | `not_implemented` | 缺 use_vllm 同配置开关、colocate 装配、LoRA/权重同步与 rollout policy 身份校验；不能借用数学样板 fallback 冒充开关消融。 | 无与数学样板共享的冻结索引/gold 隔离适配 | 实现只改 use_vllm 的单变量对照; 同 checkpoint/test/采样参数的质量与成本指标; 三 seed 聚合与导出一致性回归 |
| `llm/grpo/qwen2-5-3b-advanced-reward` | `partial_implementation` | wave5a：gsm8k-reward-v1 的 data/train/eval 三维已有自建协议、真实任务代码与隔离离线测试——#### gold 规范化剥离且不进 prompt、官方 test 1319 题整体 held-out 与 question-hash 90/10 train/validation、四 reward 完整实装、21 个 GRPOConfig 超参显式冻结、独立严格 Decimal verifier 与 reward hacking 诊断。但 comparison（四 reward 逐项消融、同预算 zero-shot/GRPO、多 seed 聚合）与 export（打包与 HF offline 复载）未实装，发布级路由未注册；一手页面本轮未能重新取回，分档条件/标签/超参为本仓选择，不冒充上游。全部只有 macOS ARM 离线合成证据：无 CUDA 实跑、无 completed 证据。 | comparison/export 两维未实装，task_router 未注册，发布级套件/导出 unavailable | 实装 comparison：drop-one 四 reward 消融 × 三独立训练 seed、同预算 zero-shot/GRPO 与聚合; 实装 export：模型 + reward 配置 + tokenizer 打包与 HF offline 新进程复载，并精确注册 task_router; 回一手页面核对四 reward 分档/标签/超参并显式改写偏差; 授权 Linux CUDA 环境冻结全部 revision、实跑三 seed，基于实跑预声明容差后独立验收 |
| `llm/sft/llama3-8b-xlam-function-calling` | `pending_freeze` | 五维代码与隔离离线测试齐备，自建严格 JSON/completion-only 协议；但 gated 基座授权与 research-only 数据商业许可未确认，四指标只是格式/协议指标而非真实 API 执行正确率，且无 GPU 实跑。 | gated Llama-3 基座授权与 research-only 数据商业/再分发许可 pending | 确认基座与数据许可及业务范围; CUDA 三 seed 实跑与 HF offline 复载; 补隔离沙箱下的真实调用回归与业务验收 |
| `llm/sft/rapidfire-multiconfig` | `pending_freeze` | 五维代码与隔离离线测试齐备，自建无泄漏 4 配置 × 3 seed 协议；但 Bitext 数据卡未给 license，上游从不报告任何数值，本仓无 stemmer ROUGE-L 与上游 use_stemmer=True 数值不可比，顺序执行不证明并发调度收益。 | Bitext 数据卡无 license，商业使用/再分发未确认 | 确认数据许可与业务验收口径; CUDA 全网格实跑与 validation 选配置/test 单次验收; 预声明 export_tolerance 与真实 offline 复载 |
| `llm/sft/starcoder1b-fim` | `pending_freeze` | 五维代码与隔离离线测试齐备，自建 repo-hash 隔离与 PSM 协议；但 exact-match 不执行代码、不等于 pass@1/编译率/功能正确，许可与污染、FIM on/off 消融、GPU 实跑未完成。 | GPU verified 0，revision/export_tolerance 未冻结 | 确认语料许可与去重/污染证据; CUDA 三 seed 实跑与导出复载; 补编译/沙箱执行类回归（当前明确不执行生成代码） |
| `vlm/dpo/smolvlm-rlaif-v` | `pending_freeze` | wave4 DPO：一手 Cookbook 缺 beta/loss/seed/revision/compute_metrics，评估仅定性生成；smolvlm-dpo-v1 五维自建协议与隔离离线测试现已齐备——图像 hash 80/10/10 同图隔离、显式 beta=0.1 sigmoid、冻结 ref、completion-only 双层 mask、隐式奖励 pairwise、三 seed 套件与 adapter-only 导出，task_router 已精确注册。与 MPO 共享偏好 split 算法但 backbone 不同，cross_backbone_attribution_valid=false。全部只有 macOS ARM 离线合成证据：无 CUDA 实跑、无 completed 证据、revision/镜像/容差/许可全未确认。 | model/dataset/source/processor revision 与可安装镜像锁未冻结 | 授权 Linux CUDA 环境锁定全部 revision 与 frozen environment; 真实三 seed 实跑、基于实跑预声明容差后独立验收、HF offline 复载; 若要做算法消融须换 shared backbone 或显式改写对照目标; 完成许可与业务验收后重新核一手材料并显式改写 verdict |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | `not_implemented` | 单数 image 列与预渲染 prompt、VLM GRPO 视觉 rollout/reward、图像级冻结隔离与失败关闭解析全部缺失；不可借用纯文本数学路由。 | 无视觉输入协议与图像/题目级冻结隔离 | 实现视觉 rollout 与 reward 失败关闭; 同图像/prompt/test 的 zero-shot 与 3 seed 聚合; 视觉 processor 打包复载与推理回归 |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | `pending_freeze` | wave4 MPO：一手 Cookbook 用未 pin 的 TRL 源码安装、显式 bf16=False 与量化 bfloat16 矛盾、无 seed/revision/compute_metrics，评估仅单样本定性；qwen25vl-mpo-v1 五维自建协议与隔离离线测试现已齐备——与 DPO 共享偏好 split、sigmoid/bco_pair/sft 加权与三消融×三 seed、冻结 ref、隐式奖励 pairwise、adapter-only 导出，task_router 已精确注册。精度矛盾如实登记不静默修正；cross_backbone_attribution_valid=false。全部只有 macOS ARM 离线合成证据：无 CUDA 实跑、无 completed 证据、revision/镜像/容差/许可全未确认；一手 MPO 论文仍 unverified。 | model/dataset/source/processor revision 与可安装镜像锁未冻结；上游 TRL 源码无 commit | 授权 Linux CUDA 环境锁定全部 revision 与 frozen environment; 真实三消融×三 seed 实跑、基于实跑预声明容差后独立验收、HF offline 复载; 若要做 DPO vs MPO 算法消融须换 shared backbone 或显式改写对照目标; 完成许可与业务验收后重新核一手材料并显式改写 verdict |
| `vlm/sft/granite-vision-line-comparison` | `pending_freeze` | wave3 首选实装目标：本轮已回一手页面核清 Geoperception/LineComparison 的 50/50 切分、assistant-only 隐式 mask、DoRA/nf4、skip_prepare_dataset、无 revision/训练 seed/指标、文字写 ChartQA 但代码从未加载 ChartQA、说 merge 却只做 PeftModel.from_pretrained。granite-linecomparison-v1 五维自建协议与隔离离线测试现已齐备：图像原始 bytes SHA256 的 80/10/10 同图隔离、规范化内容全局去重、prompt/full 两次编码 token 级前缀校验、显式 image_token_id + assistant-only 双层 -100 mask、普通 Trainer 自建 collator、冻结 processor identity、held-out 逐题 exact_match / normalized_exact_match 重算；granite_suite 三独立训练 seed 逐题内容重算与 Student-t95 聚合、granite_export 的 adapter-only 包与 HF offline 新进程复载，task_router 已精确注册。但全部只有 macOS ARM 离线合成证据：无 CUDA 实跑、无 completed 证据、revision/镜像/容差/许可全未确认；指标只衡量文本一致性，不解释几何语义，也不等于业务验收。 | model/dataset/source revision、processor revision、可安装镜像锁与依赖版本未冻结 | 授权 Linux CUDA 环境锁定全部 revision 与 frozen environment; 真实三 seed 实跑、基于实跑预声明容差后独立验收、HF offline 复载; 完成 Geoperception 与基座许可确认及业务验收; 完成许可与业务验收后重新核一手材料并显式改写 verdict |
| `vlm/sft/paligemma2-refcoco-grounding` | `pending_freeze` | wave3 第二篇 VLM：本轮已回一手页面核清 fine_tuning_vlm_object_detection_grounding 的 SFTTrainer + skip_prepare_dataset、labels=input_ids.clone() 只 mask pad 与 <image>（prompt 也计 loss）、A.BboxParams(format='coco') 与 coco_to_xyxy() 按 COCO xywh 输入而可视化直接当 xyxy、训练 prompt '<image>Detect {caption}.' 与推理 '<image>detect {caption}' 大小写/标点/suffix 三处不一致、<loc####> 为 1024 桶按 y1,x1,y2,x2 排列、SFTConfig 有 eval_dataset 却无 compute_metrics/IoU/任何数值、只有 train[20] 与 val[13] 单样本定性、do_sample=True 且无 generation seed、无 revision/训练 seed/版本锁、gated 基座与未确认许可。paligemma-grounding-v1 五维自建协议与隔离离线测试现已齐备：图像原始 mode+tobytes() SHA256 的 80/10/10 同图隔离、规范化 (caption,bbox) 全局去重、COCO xywh 输入统一转 xyxy、1024 桶 loc 编解码与 0.98 round-trip IoU 下限、统一 PROMPT_TEMPLATE、prompt/full 两次编码 token 级前缀校验、显式 image_token_id + completion-only 双层 -100 mask、普通 Trainer 自建 collator、冻结 processor identity、held-out 逐题 greedy 的 bbox_iou/iou_at_0_5/parse_rate/exact_match 重算；paligemma_suite 三独立训练 seed 从 raw bbox 重建 gold 并逐题重算、paligemma_export 普通 LoRA adapter-only 包与 HF offline 新进程复载，task_router 已精确注册。但全部只有 macOS ARM 隔离合成证据：无 CUDA 实跑、无 completed 证据、revision/镜像/容差/许可全未确认；单框 IoU 不等于 mAP/AP、人工判分或业务验收，且上游图像由 Flickr URL 运行时下载、必须先冻结静态副本。 | model/dataset/source revision、processor revision、可安装镜像锁与依赖版本未冻结；上游图像为 Flickr URL 运行时下载，未冻结静态副本 | 授权 Linux CUDA 环境锁定全部 revision 与 frozen environment，并把 Flickr 图像冻结为静态副本; 真实三 seed 实跑、基于实跑预声明容差后独立验收、HF offline 复载; 完成基座/数据许可确认与业务验收（含 mAP/AP 或人工判分口径）; 完成许可与业务验收后重新核一手材料并显式改写 verdict |
| `vlm/sft/qwen2-vl-7b-chartqa` | `blocked_zero_train` | 上游第 6 节明确称 base model + 一句 system message 是 viable alternative to fine-tuning；微调只让输出格式变短。配方自身 decision.gate=blocked、baseline.result=passes，投入 A100 预算不成立。这是原理性阻塞，不是代码或 GPU 缺口。 | 零训练基线已达标，训练无业务增量（decision.gate=blocked） | 先改写实验目标（例如换未被零训练基线解决的任务或指标）; 用独立未污染外部集重做 zero-shot 与训练对照; 重新核一手材料并显式改写 verdict；不得靠加 evidence 绕过 |
| `vlm/sft/smolvlm-chartqa` | `contaminated_pretraining` | SmolVLM-Instruct 本就在 ChartQA 上训练过，作者自承是 educational exercise；同一数据集的 held-out 不构成泛化验证。配方 upstream_deviations 已写明若无法取得未污染 held-out 集，只能作为教学对照。这是原理性阻塞，不是代码或 GPU 缺口。 | 基座在评估集上预训练过，泛化结论无效 | 取得未污染外部集并记录无交集证据; 在该外部集上重做 zero-shot 与训练对照; 重新核一手材料并显式改写 verdict；不得靠加 evidence 绕过 |

### wave3 视觉任务的实装排序

本轮回一手页面核实 `fine_tuning_granite_vision_sft_trl`：Geoperception 的 `predicate == "LineComparison"` 子集只做 `train_test_split(test_size=0.5, seed=42)`（无 validation）；自定义 collator 仅按 `<|assistant|>` 位置隐式 mask，没有显式 `image_token_id` 屏蔽；无 model/dataset/processor `revision`、无训练 seed、无 `compute_metrics`；评估只有 `test_idx = 20` 的单样本定性生成，正文写 ChartQA 但代码从未加载 ChartQA；文字称 merge adapters，代码只有 `PeftModel.from_pretrained`，无 `merge_and_unload()`；LoraConfig 用动态表达式筛 `language_model`/`_proj` 且 `use_dora=True`（标题却写 LoRA）。这些是本仓 `granite-linecomparison-v1` 自建协议必须显式纠正的一手事实。

按此排序，wave3 两篇视觉任务现已各补齐五维：`vlm/sft/granite-vision-line-comparison`（图像内容 hash 三段隔离、显式 image token + assistant-only 双层 mask、冻结 processor revision、逐题 held-out 重算）与 `vlm/sft/paligemma2-refcoco-grounding`（COCO xywh→xyxy 统一、1024 桶 `<loc####>` round-trip/IoU、单框 completion-only 协议）。wave4 的 `vlm/dpo/smolvlm-rlaif-v` 与 `vlm/mpo/qwen2-5-vl-3b-mpo` 亦已补齐五维并精确注册 task_router（图像偏好 hash 隔离、显式 beta/三损失、冻结 ref、隐式奖励 pairwise），但两篇基座不同，`cross_backbone_attribution_valid=false`。以上改判 pending_freeze **只是代码齐备，不是晋级**：仍无冻结 revision / CUDA 三 seed 实跑 / 预声明容差 / 真实 HF offline 复载 / 许可与业务验收。两篇 ChartQA 保持原理性阻塞，不计入「适配进度」；`llm/grpo/qwen2-5-3b-advanced-reward` 以自建 `gsm8k-reward-v1` 补齐 data/train/eval 三维（partial_implementation，comparison/export 仍缺，未注册 task_router）；其余 3 篇仍为 not_implemented。
