# 设计说明（ADR）

架构决策记录。每条包含决策、理由、代价与被放弃的替代方案。

<a id="adr-001"></a>
## ADR-001 — 配置格式选 TOML，不用 YAML

**决策**：`recipes/**/recipe.toml`、`registry/taxonomy.toml` 全部使用 TOML，
由 Python 3.11+ 内置 `tomllib` 解析。

**理由**：当前环境是 Python 3.13 且**没有安装 PyYAML**。这不是偶然——一个用来验证
别人训练环境的工具，如果自己依赖第三方解析器，就会在 CI、离线机器、最小镜像里
反复出现「验证器装不上，于是跳过验证」。TOML 还有强类型（数组、嵌套表、布尔、
浮点），比 YAML 的隐式类型转换（`no` → `False`、版本号被当数字）更适合承载
`trl = "0.22.0.dev0"` 这类配置。

**代价**：与 HF / Axolotl 生态惯用的 YAML 不一致；`[[array.of.tables]]` 写法比
YAML 列表啰嗦（见四函数 reward 的 `[[rewards.spec]]`）。

**放弃的方案**：
- YAML + PyYAML：生态一致，但引入唯一运行时依赖，且与「零依赖验证」目标冲突。
- 纯 JSON：零依赖但不支持注释，无法在配方里写「为什么这样设」。
- Python dict 作配置：可读性强但无法被非 Python 工具消费，也不便做静态 diff。

`doctor` 会检测遗留的 `recipe.yaml` 并提示迁移，未来若要接 Axolotl，
可在 `dev` extra 里加 PyYAML 做导出适配器，核心链路不变。

<a id="adr-002"></a>
## ADR-002 — 核心工具链零第三方依赖

**决策**：`src/recipe_lab/` 只用标准库（`tomllib` / `argparse` / `dataclasses` /
`re` / `urllib.parse` / `pathlib`）。`pytest`、`ruff` 属于 `dev` extra，不在运行路径上。

**理由**：验证是门禁，门禁必须永远能跑起来。调研中反复出现的故障正是
「因为某个包版本不对，整篇 notebook 跑不下去」——工具链不该重蹈覆辙。

**代价**：没有 pydantic 的自动类型校验，schema 校验是自己写的 `R001–R005`；
CLI 输出也是手写的纯文本表格。

**放弃的方案**：pydantic + rich + typer（开发体验更好，但把验证器变成一个需要
锁依赖的项目）。

<a id="adr-003"></a>
## ADR-003 — 配置驱动，不复制 notebook

**决策**：仓库里没有 `.ipynb`。每篇 recipe 是一份 `recipe.toml` + 可选的执行入口，
训练代码由外部 runner 按配置生成/调用。

**理由**：上游 20+ 篇 notebook 之间大量重复 `BitsAndBytesConfig` / `LoraConfig`
样板，却各自锁着不同版本——这正是调研中「跨 recipe 复制代码高危」的根因。
把超参、数据契约、依赖、评估协议收敛成声明式配置后，差异才可 diff、才可校验。

**代价**：从「打开就能跑」变成「需要 runner」，初期不能直接执行训练。

**放弃的方案**：fork 一份 notebook 逐篇修 bug（维护成本随篇数线性增长，且
notebook 的 JSON diff 几乎不可审）。

<a id="adr-004"></a>
## ADR-004 — 维护「修正版配方」，缺陷与修正双记录

**决策**：每个 recipe 有三段：
- `[known_issues]`：上游 notebook 的缺陷（照抄会踩的坑）
- `[upstream_deviations]`：我们相对上游做的修正
- `[todo]`：尚未补齐的项

`R309` 要求登记了缺陷就必须有对应处置，不允许「知道有问题但没写怎么办」。

**理由**：如果原样录入上游 bug，CI 会红；如果偷偷改掉不记录，调研知识就丢了。
双记录让「这份配方与上游有何不同、为什么不同」成为可审计的事实。

**代价**：字段变多；每篇需要人工判断哪些是缺陷、哪些是设计选择。

<a id="adr-005"></a>
## ADR-005 — 状态机 + 零训练基线决策门

**决策**：`cataloged → ported → smoke-passed → benchmarked`，
且 `[baseline].result = "passes"` 时禁止离开 `cataloged`（`R302`）。

**理由**：调研最反直觉、也最值钱的结论是——起点 recipe 第 6 节自己承认
「base model + 一句 system message，无需训练也能答对」。把这个结论编码成
一条会报错的规则，比写在 README 里有效得多。那篇 recipe 因此保留
`decision.gate = "blocked"`，永远不会被误当成可投入算力的配方。

**代价**：`validate` 的结果会随 `baseline.result` 变化；补基线实验是推进状态的
前置工作，不能靠改配置绕过。

<a id="adr-006"></a>
## ADR-006 — warning 不阻塞 CI，strict 手动跑

**决策**：`make ci` = `validate` + `test`，门禁是 **0 error**。
`validate --strict` 额外把 warning 当失败，用于提升 `status` 前复核。

**理由**：仓库里 11 条 warning 全是真实待办（7 篇未测零训练基线、2 个 dev 版本、
1 个未登记版本组合、1 条全局版本漂移）。如果让它们阻塞 CI，唯一「修好」的办法
就是删掉这些诚实记录，那正好毁掉仓库的价值。

**代价**：CI 绿灯不等于「所有风险已消解」，需要在发布流程里显式加 strict 一步。

<a id="adr-007"></a>
## ADR-007 — taxonomy 与 MATRIX 是派生的单一事实源

**决策**：`registry/taxonomy.toml` 是方法分类与版本组合的唯一事实源；
`docs/MATRIX.md` 由 `recipe-lab matrix` 生成、`docs/REPRODUCIBILITY.md` 由
`recipe-lab repro --report` 生成，两者**禁止手工编辑**，CI 检查 `git diff`。

**理由**：调研发现 MPO 不是新 trainer（`DPOTrainer` 组合损失）、GSPO 不是独立
trainer（GRPO 的 `importance_sampling_level="sequence"` 变体）、PPO 已从 TRL v1
稳定层移除。这类事实如果散落在各篇 README 里，很快就会互相矛盾。

**代价**：新增 family 要多改一处 taxonomy；MATRIX 每次变更 recipe 都要重生成。

<a id="adr-008"></a>
## ADR-008 — 可复现性等级由契约计算，不由人填写

**决策**：新增 L5 验证层与 `[repro]` 审计段。八个维度
（`code/env/model/data/params/seed/eval/provenance`）各取
`missing → declared → pinned → verified`，等级 `L0–L4` 由
`registry/reproducibility.toml` 的 `levels.*.requires` **计算**得出。
`recipe.toml` 里的 `repro.level` 只是声明，`R503` 会用同一个
`compute_level()` 重算，不一致即 error。

**理由**：第一版仓库交付时给出「0 error / 33 tests passed」，很容易被读成
「14 篇 recipe 可复现」。实际上那只检查了**配置自洽**：没有执行代码、
没有模型/数据 `revision`、dev 版本无 commit、训练 `seed` 未核实，
个别配置值甚至来自二手总结或纯属编造（Agent Wordle 的
`Qwen2.5-0.5B-Instruct`）。把「可复现」变成一个会被重算、会被打回的
数值，而不是一句形容词，才能防止后续维护时靠改数字把绿灯买回来。

本轮据此把一手核查从 7/14 补到 **14/14**（3 篇本地 `.ipynb` 原文 +
11 篇上游页面事实抽取），修正 53 处事实，最终结论写入
[REPRODUCIBILITY.md](REPRODUCIBILITY.md)：**14 篇中 0 篇达到 L2「可复现」**。

**关键设计取舍**：`recipe-lab repro --check` 校验的是**审计完整性**
（等级无虚报、`evidence_ref` 可追溯），**不把「0/14 可复现」当作 CI 失败**。
否则唯一能让 CI 变绿的手段就是虚报等级或删掉诚实的 gap 文本——
与审计目的完全相反。需要阶段性提升目标时显式用 `--min-level L2`。

**代价**：
- 每篇 `recipe.toml` 多出约 40 行 `[repro]` 段，且必须与 `evidence.toml` 同步；
- 等级是静态判定，不真的装环境、下权重、跑训练，因此 L5 全绿仍不等于
  数值层面可复现（见 VERIFICATION.md「不覆盖的范围」）；
- 11 篇只有网页事实抽取，未逐字节校验代码单元格，`coverage = 14/14`
  指的是「都回了一手材料」，不是「都逐行核对了源码」。

**放弃的方案**：
- 直接给每篇一个人工打的「可复现 / 不可复现」布尔值：无法解释为什么，也无法驱动补齐；
- 用 `docker build` 或真实训练做门禁：本机无 GPU、成本不可控，且会把
  「配方登记层」变成「训练执行层」，违反 ADR-003；
- 把证据散进各篇 `recipe.toml`：配方会混入「这个值当时从哪查来的」这类
  审计元数据，纠错历史也随之丢失，故另设 `registry/evidence.toml`。

<a id="adr-009"></a>
## ADR-009 — 一手证据与纠错历史独立于配方

**决策**：`registry/evidence.toml` 单独登记 F01–F15 发现、`[[corrections]]`
（`was → now → source`）与 `[[unverified]]` 推断值；`R507` 要求每篇
`repro.evidence_ref` 必须能在该文件中找到对应的一手材料。

**理由**：调研报告是**线索**不是**证据**。上一轮把二手总结里的
`trl 1.7.0 / transformers 5.3.0` 当成事实写进 `[[compat.matrix]]`，
把 Agent Wordle 的模型 id 写成了不存在的 `Qwen2.5-0.5B-Instruct`。
把「原值 / 核实后 / 依据」显式留档，错误才不会在下一轮维护中被重新引入。

**代价**：新增 recipe 时要同步登记证据；`scripts/apply_repro_audit.py`
是一次性迁移脚本，之后靠人工维护这三张表。

<a id="adr-010"></a>
## ADR-010 — baseline 与 repro 分离：复刻不了原实验，仍能定义自己的基线

**决策**：新增 `registry/baseline.toml`、L6 规则（R601–R606）与
`[baseline_contract]` 段。状态 `B0 / B1 / B2 / B3` 由四组能力
（`G1 可引用` / `G2 可重放` / `G3 可对照` / `G4 可交付`）共 28 条检查
**计算**得出，`R603` 用同一 `compute_state()` 重算，虚报即 error。
`recipe-lab baseline --check` 与 `repro --check` 一样只查完整性，
不把「0/14 到 B2」当失败。

**理由**：用户提出「希望每一个 recipe 都是可以作为 research 或者业务训练的
baseline」。若沿用 ADR-008 的 repro 口径，这个目标几乎不可达——上游当时的精确
commit 多半拿不到，而那是重放原实验的硬前提。但两者的**问题不同**：

| | repro | baseline |
| --- | --- | --- |
| 追求 | 重放作者那一次的结果 | 提供一个别人能对照的稳定参照物 |
| 版本 | 上游当时的精确 commit | **我们自己锁定**一套确定的 commit 并公布 |
| 指标 | 要贴近原文数值 | 我们自己实测并给出容差即可 |

于是「把 14 篇都做成 baseline」从不可达变成可达，代价是必须公开声明
「本 baseline 与上游原数值不要求一致」，否则会误导后来者拿两份不可比的数
做对照。B1（契约齐备、可 dry-run、未实跑）被明确定义为本机无 GPU / 无外网
时的最高可达状态；凡「本机拿不到」的项（真实 revision、零训练基线实测、
held-out 指标、多轮 evidence、许可、资源成本）一律归 B2 / B3，
绝不为了好看下调门槛，也绝不填假 hash。

**代价**：又多一套四字母等级（验证层级 L0–L6 / 可复现性 L0–L4 /
基线 B0–B3 / 成熟度状态机），文档必须反复澄清；每篇 `recipe.toml` 多约
20 行执行契约。

**放弃的方案**：
- 用 `repro.level >= L2` 直接当 baseline 门槛：目标不可达，且会让团队把精力
  花在追上游 commit 而不是建立自己的受控基线；
- 提高 `status` 到 `benchmarked` 就算基线：状态机只管配方成熟度，不表达
  「别人能否引用、能否对照、业务能否验收」这三件事；
- 允许 B1 用注释里的测试版本充当 lock：那正是 ADR-008 打掉的伪精确。

<a id="adr-011"></a>
## ADR-011 — 统一 runner 骨架：宁可明确拒绝，也不静默退化

**决策**：新增 `runner/`（`train.py` / `eval.py` / `zero_shot.py` / `lock.py`）
与 `src/recipe_lab/plan.py`。核心链路零第三方依赖，`trl` / `transformers` /
`torch` 的 import 全部延迟到真正执行时；`--dry-run` 只解释 `recipe.toml`，
不需要 GPU 或训练框架。CLI 的 `baseline --dry-run` 与 runner 共用同一个
`build_plan()`，不存在两套解释。

**理由**：B1 要求「一条确定命令、可 dry-run」，但我们不能为了凑齐字段就
伪造一个能跑的训练循环。因此 runner 的**缺省行为是拒绝**：缺
`model.revision` / `dataset.revision`、PEFT / 量化未适配、RL reward 未移植、
指标函数未注册时，`--full` 直接给出可操作报错并退出，绝不返回占位分数、
假 hash 或假 evidence。每篇 `execution.adapter_status` 都写明适配状态；
`agent/grpo/wordle-openenv-hf-jobs` 还单列 `unimplemented`（OpenEnv / Wordle
环境与 reward 未移植，上游走 HF Jobs + papermill）。

`runner/lock.py --apply` 是留给有网环境的锁定工具，`runner/locks/*.lock.toml`
当前只是**模板**，记录「应该锁什么」，不是已冻结的 `pip freeze` 结果。
本机 Bash 访问 HF Hub API 返回 502，因此仓库里没有任何来路不明的 revision hash。

**代价**：`execution.command` 的含义是「本仓库维护的基线入口」，不是
「今天就能复现结果」；要真正升级到 B2，必须有人在有 GPU / 有网环境完成
P1–P3（锁定 → 零训练基线实测 → 首跑与多轮评估）。这部分工作无法由本仓库
替代，报告里也不假装已完成。

**放弃的方案**：
- 每篇复制一份 `train.py`：直接违反 ADR-003，14 份代码会各自漂移；
- 用 `--dry-run` 输出一份「模拟指标」让 CI 变绿：那是最危险的假证据；
- 把框架 import 放在模块顶层：本机连 dry-run 都跑不了，CI 必须装 TRL。

<a id="adr-012"></a>
## ADR-012 — 实际能力独立登记，证据失败关闭，实装任务不等于GPU验证

**纠正**：用户再次指出“不完整、不全面、不系统”。ADR-010/011的B1与统一入口存在性不够，
不能把14篇声明作为baseline完成度。此前骨架阶段的描述仅保留作历史，不再表达当前代码能力。

**决策**：
- 新增`registry/capabilities.toml`与`capabilities.py`，为14篇分别登记data/train/eval/comparison/export。
  missing/implemented/unverified/verified与B0–B3、repro等级分开；严格精确ID覆盖、无orphan。
- 完整范围区分taxonomy已登记无recipe、CPT/DAPT音视频未taxon模块和数据/评估/交付跨方法管线。
  wave0–wave6列出依赖、前置条件及验收，不把scope与队列当成已实现。
- math-grpo-v1只支持明确模型/数据/TRL的纯文本全参数GRPO，固定500/500/500数据段。
  输入schema/gold解析/跨split重复、reward失败路径、框架API、参数应用、预算、checkpoint来源均校验。
  实跑必须核对完整frozen环境；显式checkpoint也需已完成训练证据，不回退base。
- `--apply`仅解析缺失输入，`--capture-environment`才捕获实际GPU完整环境；失败不写部分成功锁。
  重跑迁移只刷新审核状态，不重置种子/划分/任务路由或覆盖真实锁。
- `baseline_evidence.py`验证baseline-suite v1：至少三个独立训练seed、配对评估、文件内容hash、
  代码快照、split去重与内容指纹、环境锁、训练预算/参数、checkpoint身份、数学逐题指标重算、均值容差。
  空文件/说明字符串/未知版本/main/NaN/负容差不能晋级；合法B2不额外要求B3许可与owner。
- B3许可/业务验收/导出回滚的confirmed字段只记录人工复核，不能当自动合规、SLO或业务认证。
- CI新增capabilities检查；发布目标另用`baseline --min-state B2`，不混同维护绿灯。

**已交付与未交付**：数学data/train/eval代码和offline纯函数/mock测试已实现，但GPU实跑0篇。
数学多训练seed编排/聚合和导出复载的代码已按ADR-013补齐；FIM data/train/eval按ADR-014接入，comparison/export与发布级精确路由按ADR-015补齐；xLAM data/train/eval按ADR-016接入，两维发布级管线按ADR-017补齐；RapidFire data/train/eval按ADR-018接入，4配置×3 seed发布级套件与选中adapter导出按ADR-019补齐并在`task_router.py`精确注册，五维齐备。截至该 ADR 时另有 10 篇任务完全缺失；本轮后续 ADR-021 实装 Granite 的 data/train/eval 三维后变为「9 篇完全缺失 + Granite 三维、发布级两维未实现」。四篇文本任务与 Granite 三维都只有离线合成证据，GPU验收、真实三seed、真实GPU复载、安装镜像与业务回归未完成；真实recipe未预声明`evaluation.export_tolerance`时导出`--full`失败关闭。
所有合成正向B2证据只在测试临时目录使用，不作为真实结果或对外基线。

**限制**：文件hash与内部结构不是密码学执行证明；手填JSON仍需独立流程核查原始日志和产物。
声明版本不是GPU兼容确认；固定前1500题若gold解析或去重失败，应修订数据协议版本而非偷偷放宽。

---

## ADR-013 独立训练seed套件与导出复载

- `provenance.effective_recipe`深拷贝仅覆盖train.seed，固定原始data/evaluation seed；三个CLI增加受控`--train-seed/--output-root`，不原地改TOML。
- completed training/evaluation写当前执行代码canonical快照hash和实际环境锁字节hash；运行前捕获，完成时复核，漂移拒绝completed。
- `suite_task`至少3独立seed，共享一次base，对每个seed从冻结base重新训练；不把重复greedy解码当训练方差。非空目录拒绝，子进程超时/失败停止，日志保留，不自动resume。
- 聚合器绑定模型、训练证据、有效配置、预算、严格1500条manifest、逐题prompt/gold/预测、代码/锁和运行环境；B2发布门禁共用只读强校验，直接提交宽松JSON也不能晋级。
- 统计输出mean/sample_variance/std/sem/Student-t95区间，df>30保守用df30系数；固定test不包含数据抽样不确定性，不声称业务显著性。expected/tolerance是独立验收声明，不用本轮均值自动填写。
- manifest增加solution_sha256，防止只校验problem/content ID却让替换的非空gold漏过；格式分数从completion重算，数学正确性仍需原始verifier执行与独立复核。
- `export_task`复制模型/tokenizer、训练证据、配方快照、环境锁和参考预测，固定Popen新进程强制HF offline，调用实际math_task重载复评；缓存锁定数据仍必需，本包不是安装镜像。
- C405重读导出包，核文件清单、worker PID/PPID、锁/代码/模型身份、参考/复载协议和指标容差。许可与回滚人工确认单独保留；未实现的非数学导出直接失败。
- 五维implemented不等于GPU verified；本机macOS ARM无torch，代码与隔离CPU/new-Python测试不验证CUDA兼容、显存、收敛或收益。当前仍B2+/B3/GPU均0。

---

## ADR-014 第二任务FIM：受控变体与显式路由

- 一手Cookbook为Trainer，数据卡明确repo_id/file_path/content；精确recipe/task白名单允许Trainer，不放宽所有SFT映射。
- repo哈希80/10/10、全局raw/normalized去重、文件内PSM/SPM分块/padding、validation PSM、eager attention/no remote code为自建迁移，不冒充上游随机流packing复现。
- causal-all-nonpadding保留真实EOS label，padding=-100；固定data_seed不随train_seed改变。尾巴少于3 tokens显式记录，不算完整训练样本。
- QLoRA LoRA target保留上游候选q_attn但不强制命中；其余target/base冻结/trainable核验。full预算2000，smoke5，不偷偷截为150。
- 本地adapter必须completed来源/配置/manifest/tokenizer/当前artifact/code/锁匹配；只读safetensors，不用pickle；adapter提取keys/shape/有限性/值逐项匹配，拒绝非严格加载遗漏。
- held-out固定PSM gap，同base/adapter量化/模板/词表/窗口/greedy；gold不入生成input，预算为EOS留一位，严格文本/token exact-match不strip空白。不执行生成代码，不声称pass@1/编译率。
- FIM三seed套件、adapter-only导出及发布级PSM输入/gold独立重算已实装（`fim_suite.py`/`fim_export.py`），见ADR-015；C306/C307/C405按`task_router.py`分派到各自内容校验器，未知任务失败关闭，不借用数学或通用JSON虚报。仅有隔离offline测试，无GPU/completed实跑证据，不晋级B2/B3。
- 数据卡许可缺失与模型授权须单独解决，repo split不等于无预训练污染。所有本轮验证是隔离mock，不宣称GPU/API真实兼容。

## ADR-015 发布级精确任务路由与位置/内容身份分离

- 背景：数学的`1500`题manifest、prompt/gold重算、统计聚合与导出协议与FIM的repo隔离PSM逐题输入不兼容。若把数学`suite_task`/`export_task`当通用校验器，任何任务都能靠提交结构相似的JSON「通过」发布门禁。
- 决策：`task_router.py`以精确`recipe_id` + `execution.task_id`白名单分派`task`/`suite`/`export`三类模块；声明与ID冲突、未知task、未知recipe借用别的task一律抛错。`runner/suite.py`、`runner/export.py`、`baseline.py`、`baseline_evidence.py`的C306/C307/C405全部按此分派。worker不接受命令行自称任务名，只从包内`recipe-snapshot.json`推断。
- 只读声明与真实执行分层：无任务实现的recipe在`--dry-run`仍可得到声明计划，但必须显式带`release_pipeline="unavailable"`；`--full`/`--aggregate-only`/worker/证据门禁继续失败关闭，不把声明当B2证据。
- FIM发布级能力：`fim_suite.py`做三互异训练seed的独立输出、共享base/held-out协议、逐题PSM输入/gold内容重算与Student-t95统计，不回填expected/tolerance、不自动晋级；`fim_export.py`打包adapter-only权重、冻结base引用（含`base_weights_bundled=false`）、tokenizer identity，并在HF offline新进程真实调用`fim_task.run_evaluation`复载复评。
- 位置不是内容身份：checkpoint复制到导出包后，`TrainingArguments.output_dir`、`artifact_manifest.path`与`execution.lockfile`都会变化。只对这三类路径字段豁免逐项相等，文件清单/sha256/训练超参/预算/锁字节仍严格绑定。锁文件通过虚拟锚点目录自包含，包内`recipe-snapshot.json`保持与训练时`r.data`逐字节一致，避免`recipe_sha256`/`effective_recipe.json`/`execution_identity`失配或反过来放宽内容校验。
- 边界：以上全部是代码与隔离离线测试（合成证据只写`tmp_path`）；JSON/hash只证明内部一致性，不认证GPU执行真实性。`implemented`不等于`verified`，B2+/B3、L2+仍为0；许可、真实CUDA环境、真实三seed与真实GPU复载、业务验收仍为开放项。

---

## ADR-016 第三任务 xLAM：输出协议自建与发布级失败关闭

- 背景：一手核查（Cookbook xLAM 页 + `Salesforce/xlam-function-calling-60k` 数据卡）发现上游`process_xlam_sample`把`answers`(gold)拼进`dataset_text_field="text"`，`SFTTrainer`对**含 query/tools/gold 的整段文本**做损失；推理 prompt 却截断为`<user>...</user>\n\n<tools>`，不含候选工具定义；展示输出是 Python `str(dict)` 单引号字面量而非合法 JSON；无 seed / revision / `eval_dataset` / `compute_metrics`，只有 3 条`do_sample=True`定性样例。数据卡 gated、research-only、2024-07 曾重生成 1896 条，未冻结 revision 时输入会漂移。若照搬上游，"schema 合法率"既不可解释也会把 gold 泄漏进评估输入。
- 决策：`xlam_task.py`实现自建`xlam-fc-v1`协议——`transformers.Trainer` + completion-only collator，分别编码 prompt/target 后拼接，只对`<calls>`严格JSON数组`</calls>`+EOS计loss（prompt label=-100）；给全 tools 定义；`render_prompt`显式断言 gold 不在 prompt 中；长度超限拒绝截断。数据侧严格校验列集合、工具去重、`answers.name ∈ tools.name`、参数键/required/类型，规范化内容 hash 80/10/10 确定性划分，全局重复（含跨 split）与空 split 一律拒绝，test 为 shared held-out。
- 评估安全：只用`json.loads`解析`<calls>...</calls>`，拒绝 Python repr / 注入文本 / 非数组 / 结构异常；**绝不`eval`或执行生成的函数调用**。`json_parse_rate` / `schema_valid_rate` / `tool_call_exact_match` / `argument_f1`是格式与协议指标，不等于真实 API 执行正确率、pass@1、编译率或业务安全。
- 事实与执行分离：`method.trainer="Trainer"`是本仓执行层，`method.upstream_trainer="SFTTrainer"`保留一手事实；上游 dataclass 默认值移入`[upstream]`元信息段，`[train]`是自建 full=150/smoke=5 参数。`taxonomy.toml`按`families.sft.task_trainers.xlam-fc-v1`登记 Trainer，`evidence.toml [xlam_adapter_audit]`记录全部偏差与许可边界。
- 路由与指纹：三个 CLI 用`TASK_ADAPTERS`精确`recipe_id`白名单分派，不再靠"数学为默认、FIM 为例外"的二选一硬编码。`baseline_evidence.fingerprint()`把`[xlam]`/`[upstream]`与`evaluation.max_new_tokens`纳入内容身份，防止改协议后旧 checkpoint 凭相同 fingerprint 冒充当前实验产物。发布级`suite`/`export`在ADR-016阶段明确失败关闭，按ADR-017补齐独立实现，仍不借用数学 1500 题或 FIM PSM 协议。
- 边界：只有隔离合成夹具（`tests/test_xlam.py` 36 项），无 GPU 实跑、无 completed 证据、未冻结 revision；基座需授权、数据卡商业许可未确认；B1 不晋级，`implemented`不等于`verified`，B2+/B3、L2+仍为0。

---

## ADR-017 第三任务发布级两维：严格 JSON 协议专用套件与 adapter-only 导出

- 背景：数学的1500题prompt/gold重算、FIM的repo隔离PSM输入都不能解释xLAM的输出协议。若让`xlam-fc-v1`沿用任一协议的摘要JSON结构，改一改指标名就能让「合法JSON率」冒充B2证据；而QLoRA checkpoint只是adapter，导出包若不声明冻结base与严格JSON协议，离线复载无法归因。
- 决策：`task_router.py`为`xlam-fc-v1`补齐`suite`/`export`精确分派；`xlam_suite.py`三互异训练seed共享冻结base与内容hash隔离held-out，逐题从`eval_manifest.json`的raw `query/tools/answers`重新`parse_row`/`render_prompt`/`score_prediction`重算四指标（不信任摘要分数），统计均值、样本方差、SEM与Student-t95区间，不回填expected/tolerance、不自动晋级；`xlam_export.py`打包adapter-only权重、冻结base revision/量化/eager/tokenizer identity与`base_weights_bundled=false`、严格JSON输出协议，HF offline新进程真实调用`xlam_task.run_evaluation`复载复评并做逐题输入/gold/容差内容级复核。
- 位置不是内容身份（沿用ADR-015）：`TrainingArguments.output_dir`、`artifact_manifest.path`、`execution.lockfile`只豁免路径，文件清单/sha256/训练超参/completion-only协议/预算/锁字节仍严格绑定。包内`recipe-snapshot.json`与训练时`r.data`逐字节一致，锁用`_lock_layout()`虚拟锚点自包含，避免`recipe_sha256`/`effective_recipe.json`/`execution_identity`失配。
- 开放协议拿不到容差就关闭：`et._tolerance()`把`evaluation.export_tolerance`作为正式入口；真实`recipe.toml`未基于实跑预声明容差表，`--full`导出必须失败关闭（`tolerance_status=not_declared`），绝不从本轮均值回填expected/tolerance。
- 边界：全部是代码与隔离离线测试（`tests/test_xlam_release.py` 31 项，合成证据只写`tmp_path`）；JSON/hash只证明内部一致性，不认证GPU执行真实性。四指标不等于真实API执行正确率或业务安全；gated基座与research-only数据许可pending；`implemented`不等于`verified`，B2+/B3、L2+仍为0。

---

<a id="adr-018"></a>
## ADR-018 — 第四任务三维：无泄漏多配置 SFT、配置身份绑定与发布级失败关闭

- 背景：wave2 最后一篇 `llm/sft/rapidfire-multiconfig` 的上游是 HPO 编排层：`eval_dataset=range(50,60)` 完全落在 `train_dataset=range(64)` 内、`sample_formatting_function` 把 `response` gold 拼进整段 `Question:/Answer:` 文本、`sample_compute_metrics` 直接把未解码的 `eval_preds` 传给 `evaluate.load('rouge')`（异常静默返回 `{}`）、页面从未报告任何数值；`rapidfireai` 用 `%pip install` 无版本并启动本地服务做 4 配置并发调度，model/dataset 无 revision、无 `set_seed`/deterministic。若把「声明了修正切分」当作实现，或让该任务借用数学/FIM/xLAM 的 `suite`/`export` 摘要结构，泄漏数据与不可比指标就能冒充 B2 证据。
- 决策：新建 `rapidfire_task.py` 实现 data/train/eval 三维自建协议 `rapidfire-sft-v1`。Bitext 5 列严格 schema；规范化问题 hash（NFKC+casefold+空白折叠）80/10/10 让同题所有语言变体落同一 split，规范化内容全局去重并拒绝跨 split 重复，空 split 不补救；`Trainer` + completion-only collator 只对 ` {response}`+EOS 计 loss，训练右 padding；单样本 greedy、首个 EOS 截断，纯 Python 无 stemmer LCS ROUGE-L 与 exact_match 逐题重算（明确声明与上游 `use_stemmer=True` 数值不可比）。上游事实只保留在 `[upstream]` 与 `evidence.toml [rapidfire_adapter_audit]`，不参与执行。
- 配置即内容身份：`[[configs]]` 精确声明上游 4 组 lr/scheduler × LoRA rank/alpha/target_modules；`config_recipe()` 只覆盖 `[train]`/`[peft]`/`execution.config_id`/`baseline_contract.experiment_id`，逐配置隔离输出目录与证据，命令显示也随配置更新。`baseline_evidence.fingerprint()` 绑定 `[rapidfire]` 协议段、完整网格与当前 `config_id`，改 lr/rank/输出协议后旧 checkpoint 不能冒充当前配置产物；`frozen_environment()` 按协议段选择 `{torch,transformers,datasets,peft,safetensors}`，不误要求 `trl`。
- 顺序执行不等于并发复现：本仓按顺序跑 4 配置，不调用未固定版本的 `rapidfireai` 服务，不宣称 `num_chunks=4` chunk 调度或 16-24× 吞吐声明；成本/吞吐消融另列为未交付。
- 发布级分阶段交付：`comparison`/`export` 仍为 `missing`，`task_router.py` **不注册** `rapidfire-sft-v1`（`recipe.toml` 也刻意不声明 `execution.task_id`）。`runner/suite.py` 的 dry-run 显式写 `release_pipeline="unavailable"`，`--full`/`--aggregate-only` 失败关闭；`baseline_evidence` 的 C306/C307/C405 对该任务继续失败关闭，绝不借用数学 1500 题、FIM PSM 或 xLAM 严格 JSON 协议。下一阶段须补 4 配置 × 至少 3 独立训练 seed 的严格聚合（只用 validation 选配置、test 只验收一次）与选中 adapter 的 HF offline 新进程复载，并在 `export_tolerance` 未预声明时继续失败关闭。**该分阶段限制已由 ADR-019 解除：两维代码、精确路由与专用 C306/C307/C405 校验器均已补齐，本节保留当时决策记录。**
- 边界：`tests/test_rapidfire.py`（64 项）只用 `tmp_path` 合成 fake runtime 调真实 `run_training`/`run_evaluation`；JSON/hash 仅证明内部一致性。gpt2 base 与 Bitext 数据 revision/许可（数据卡未给 license）未冻结，无 GPU 实跑、无 completed 证据、无真实三 seed、无导出复载；`implemented` 不等于 `verified`，B2+/B3、L2+ 仍为 0。

---

## ADR-019 — 第四任务发布级两维：4 配置网格 × 3 seed 聚合、validation 选配置与选中 adapter 复载

- 背景：ADR-018 只交付了 data/train/eval 三维。RapidFire 的「可比性」不是三 seed 重复 greedy，而是 4 组 LoRA/lr 配置各自至少 3 个独立训练 seed、在同一冻结 held-out 上的对照；若让 `runner/suite.py` 借用数学 1500 题聚合、FIM PSM 或 xLAM 严格 JSON 摘要结构，`config_id`、validation/test 角色与 completion-only 指标就会全部丢失。导出也不能把 4 个 adapter 混为一包或用「默认配置」冒充被选中的配置。
- 决策 —— 评估 split 是先决条件：给 `rapidfire_task.run_evaluation` 增加 `split_name`，`--eval-split validation|test` 绑定 `eval_stage`（`config_selection` / `final_acceptance`），并把 split 写入 `evaluation_manifest`、`predictions.json` 与评估证据。`source_manifest` 的划分身份改为只绑定完整配置网格与 `data_seed`，**不再绑定当前 `config_id`**，否则 4 配置无法共享同一冻结 held-out；`config_id` 仍由 `fingerprint()` 绑定到每个训练的 checkpoint 身份。
- 决策 —— `rapidfire_suite.py`：完整网格 4 配置 × 3 个互异独立训练 seed，逐题从 `eval_manifest` 的 raw `instruction`/`response` 重建 prompt 与 gold，用 `rt.score_prediction` 重算无 stemmer ROUGE-L 与 `exact_match`（不信任摘要分数）；每轮配对同 split 的冻结 base zero-shot，重新核验 checkpoint 的 recipe/fingerprint/code/lock/revision/seed/config_id、adapter 张量与预测 hash。**只用 validation 的 paired `rouge_l` 均值选配置**（平局按 `config_id` 字典序），test 只在选中配置上验收一次；输出均值、样本方差、SEM、Student-t95 与顺序执行的时长/显存/步数成本，显式 `upstream_throughput_comparable=false`，不宣称复现 `rapidfireai` 并发调度或 16-24× 加速。不回填 `expected`/`tolerance`、不自动晋级 B2。
- 决策 —— `rapidfire_export.py`：导出 validation 选中的**单一** `config_id` 的 fp16 adapter-only 包，`config_id` 从 checkpoint 的 `training.json` 精确读取，拒绝默认配置冒充；包内声明冻结 gpt2 base revision / tokenizer identity / eager / 无量化 / completion-only prompt 协议 / 无 stemmer ROUGE-L 不可比性 / `base_weights_bundled=false` / `generated_code_executed=false` / license pending。`recipe-snapshot.json` 必须与训练时 `r.data` 逐字节一致，锁沿用 ADR-015 的 `_lock_layout()` 虚拟锚点；HF offline 全新进程真实调用 `rapidfire_task.run_evaluation` 在 test split 复评，不复制 JSON 冒充复载。真实 `recipe.toml` 未预声明 `evaluation.export_tolerance` 时 `--full` 失败关闭。
- 决策 —— 内容身份只含协议，不含审核回填：注册 C307 会往 recipe 写入 `execution.evidence` 指针，首次实测后还会写入 `evaluation.expected`/`evaluation.tolerance`；这些字段若进入 `sha256_json(r.data)`，就会与训练时 `effective_recipe.json` 永久失配。`_identity_recipe()` 统一剥离这三者，保留训练前预声明的 `export_tolerance`（`be.fingerprint()` 本就不含它们）。`acceptance` 是阈值判定结果而非从证据重算的客观内容，因此只做结构校验；真正的验收由 `suite_problems` 用当前 recipe 的阈值独立判定，绝不采信 JSON 里自称的 `passed`。数学/FIM/xLAM 三个套件同样只写不校验 `acceptance`。
- 路由与门禁：`task_router.py` 新增精确 `RAPIDFIRE = "rapidfire-sft-v1"`，把 `suite`/`export`/C306/C307/C405 分派到 RapidFire 专用校验器；`frozen_environment()` 按 `execution.task_id` 要求 `{torch,transformers,datasets,peft,safetensors}`，不误要求 `trl` 或 `bitsandbytes`。未知、冲突或跨任务借用协议一律失败关闭。
- 边界：`tests/test_rapidfire_release.py`（39 项）与 `tests/test_rapidfire.py`（64 项）全部只写 `tmp_path`，用隔离合成 fake runtime 与受控 worker；JSON/hash 只证明内部一致性，不认证 GPU 来源。仍缺真实 CUDA 三 seed 实跑、真实 GPU 导出复载、冻结 model/dataset/source revision、可安装镜像锁、Bitext 数据许可（数据卡未给 license）与业务验收；ROUGE-L/`exact_match` 不证明客服业务正确或安全。该 ADR 时四篇五维代码为 `20 implemented / 50 missing`；ADR-021 实装 Granite 三维后为 `23 implemented / 47 missing`（`data`/`train`/`eval` 各5/14，`comparison`/`export` 各4/14）。GPU verified 0、B2+/B3 0、L2+ 0 始终未变。

---

## ADR-020 — 把「能不能作为 baseline」从代码/证据里独立出来判定

- 背景：用户的目标是「每一个 recipe 都可以作为 research 或业务训练的 baseline」。此前 ADR 建立的 `capabilities.toml`（代码 + offline test）、`baseline.toml` / `baseline_evidence.py`（证据与就绪度）只回答了「能不能跑起来」和「有没有真实跑过」，没有回答**实验目标本身是否成立**。回一手材料后发现两类用代码和 GPU 都补不回来的阻塞：`vlm/sft/qwen2-vl-7b-chartqa` 的 `decision.gate=blocked`（上游第 6 节自承零训练基线即可达标，微调只让输出格式变短）；`vlm/sft/smolvlm-chartqa` 的基座本就在 ChartQA 上训练过（作者自承是 educational exercise，同一数据集 held-out 不构成泛化验证）。继续把这两篇算进「适配进度」会系统性高估仓库能力。
- 决策：新增 `registry/feasibility.toml` + `src/recipe_lab/feasibility.py`，逐篇给出六个枚举之一：`eligible` / `pending_freeze` / `partial_implementation` / `not_implemented` / `blocked_zero_train` / `contaminated_pretraining`。当前 `eligible 0/14`、`pending_freeze 4`（四篇五维代码齐备但 GPU verified 0 / revision 与容差未冻结）、`partial_implementation 1`（Granite 三维，见 ADR-021）、`not_implemented 7`、两类原理性阻塞各 1。每篇必须写明一手 `reason`、`blockers`、`requires`、`primary_source`。
- 门禁：`recipe-lab feasibility --check` 校验 14 篇精确覆盖、合法 verdict、非空阻塞/条件，并与配方事实交叉核对——`blocked_zero_train` 必须对应 `decision.gate=blocked` 且 `baseline.result=passes`；`contaminated_pretraining` 必须在 `upstream_deviations` 里有预训练污染记录；`pending_freeze` 要求 capabilities 五维全 `implemented`；`partial_implementation` 要求恰为 data/train/eval = `implemented` 且 comparison/export = `missing`（多一维、少一维都失败关闭）；`not_implemented` 要求五维全 `missing`。登记里自设 `override` 一律拒绝。
- 晋级：`baseline --min-state B2/B3` 与 `feasibility --min-state B2/B3` 都调用 `promotion_problems()`，把非 `eligible` 篇目列为失败。两类原理性阻塞**不接受新增 evidence JSON 或抬高 B 状态绕过**，只能改写实验目标 / 换未污染数据 / 重核一手材料后显式改写 verdict。`make ci` 新增 `feasibility-check`；`BASELINE-READINESS.md` 追加独立章节；`tests/test_feasibility.py`（27 项）覆盖正向一致性与篡改登记失败关闭。
- wave3 排序：本轮回一手页面 `fine_tuning_granite_vision_sft_trl` 核清 Geoperception 的 `predicate=="LineComparison"` 子集只做 `train_test_split(test_size=0.5, seed=42)`（无 validation）；自定义 collator 仅按 `<|assistant|>` 位置隐式 mask、无显式 `image_token_id` 屏蔽；无 model/dataset/processor `revision`、无训练 seed、无 `compute_metrics`；评估只有 `test_idx=20` 单样本定性生成，正文写 ChartQA 但代码从未加载 ChartQA；说 merge adapters 却只有 `PeftModel.from_pretrained`；`LoraConfig` 用动态表达式筛 `language_model`/`_proj` 且 `use_dora=True`（标题写 LoRA）。这些是 `granite-linecomparison-v1` 自建协议必须纠正的事实，因此 wave3 首选 `vlm/sft/granite-vision-line-comparison`（图像内容 hash 三段隔离、显式 image token + assistant-only 双层 mask、冻结 processor revision、逐题 held-out 重算），次选 `vlm/sft/paligemma2-refcoco-grounding`（bbox round-trip / IoU 可验证）。两篇 ChartQA 不计入适配进度。
- 边界：本判定是本地一手材料审计，不是新 GPU 证据；它不改动任何 `baseline_contract.state`，也不替代 CUDA 实跑、许可确认与业务验收。`eligible` 只表示「原理上值得进入 B2 证据门禁」，仍须冻结输入/环境、三独立训练 seed、预声明容差与导出复载。

---

## ADR-021 — wave3 首篇 VLM 只做三维自建协议，且不伪装成发布级任务

- 背景：ADR-020 判定两篇 ChartQA 原理性阻塞后，wave3 首选改为 `vlm/sft/granite-vision-line-comparison`。回一手页面确认上游只有 `Geoperception` 的 `predicate == "LineComparison"` 子集 + 50/50 切分（无 validation），collator 只按 `<|assistant|>` 位置隐式 mask 且没有显式 `image_token_id` 屏蔽，无 revision / 训练 seed / `compute_metrics`，评估只有 `test_idx=20` 单样本定性生成，正文写 ChartQA 但代码从未加载，说 merge 却只有 `PeftModel.from_pretrained`，`use_dora=True` 而标题写 LoRA。上游没有可复现的定量基线，因此**不能**把「复现上游」当作实装目标。
- 决策：新增 `src/recipe_lab/granite_task.py` + `recipes/vlm/sft/granite-vision-line-comparison/recipe.toml`，定义 `granite-linecomparison-v1` 自建协议，只实装 `data`/`train`/`eval` 三维：严格 4 列 schema 与 predicate 过滤；图像原始 `mode + tobytes()` 的 SHA256 做 80/10/10，同图所有问答变体同 split，规范化 `(question, answer)` 全局去重、跨 split 重复与空 split 拒绝；`prompt`/`full` 两次 `processor.apply_chat_template` 编码并做 token 级前缀校验，gold 只在 full 序列、绝不进生成 prompt；显式 `image_token_id` 与 assistant span 外 token 双层 `label=-100`；普通 `transformers.Trainer` + 自建 collator（不用上游 `SFTTrainer` / `skip_prepare_dataset`）；冻结 processor identity / seed / revision / 代码与锁字节；held-out 逐题 greedy + 首个 EOS 截断 + `exact_match` / `normalized_exact_match` 重算。指标只衡量文本一致性，不解释几何语义、不等于人工判分或业务验收；未执行生成代码、不冒称 pass@1。
- 路由约束：**不注册** `task_router`、不实装 `comparison`/`export`。为让发布级 dry-run 保持 `unavailable`、`--full` 失败关闭，recipe 按设计**不声明** `execution.task_id`（`resolve()` 遇到「声明了 task_id 但无管线」会直接抛错）。这带来 R103 的连带问题：`task_trainers` 原按 `execution.task_id` 索引，会把本仓的 `Trainer` 误判为与 `sft` family 的 `SFTTrainer` 不符。修法不是在 recipe 里写假 trainer、也不是放宽 R103，而是让 `rules.py` 在 task_id 缺失时按**精确 `recipe_id`** 查找白名单，并在 `registry/taxonomy.toml` 新增 `[families.sft.task_trainers.granite-linecomparison-v1]` 登记 `recipe_id` + `trainer = "Trainer"`；`expected_trainer` 仍要求 `recipe_id` 精确匹配，绝不按 family 猜测。三维任务也不借用数学/FIM/xLAM/RapidFire 的 suite/export 协议。
- 可行性联动：`registry/feasibility.toml` 新增 `partial_implementation` verdict（Promotion blocking），Granite 由 `not_implemented` 改为该值；`feasibility.py` 交叉校验 capabilities 恰为三维 implemented + 两维 missing，缺一维或多一维都失败关闭。`eligible` 仍为 0/14，B2+/B3 与 GPU verified 仍为 0；补齐五维代码也不解除两篇 ChartQA 的原理性阻塞。
- 验证与边界：`tests/test_granite.py`（58 项）只用 `tmp_path` 内的确定性合成图像与 fake runtime，不加载 torch/transformers、不联网、不产 GPU 证据；覆盖一手事实常量、路由漂移 14 组、图文合同缺失、双层 mask、gold 泄漏、checkpoint/权重身份、`baseline_evidence` 的 fingerprint / frozen_environment 分派。JSON/hash 只证明仓内内容一致性，不认证 GPU 执行真实性。剩余门槛：冻结 model/dataset/processor revision 与可安装镜像、授权 Linux CUDA 环境三独立 seed 实跑、预声明容差、补齐 comparison/export 并精确注册 `task_router`、数据/基座许可与商业使用确认、业务验收。
- 后续状态（ADR-022 已取代本 ADR 的发布级部分）：`comparison`/`export` 已补齐并精确注册 `task_router`，recipe 已显式声明 `execution.task_id`，发布级 dry-run 不再 `unavailable`；`tests/test_granite.py` 相关断言已同步更新，`--full` 仍失败关闭。R103 的精确 `recipe_id` 白名单路径保留兼容，但不再是 Granite 的判定依据。本条与 ADR-020 一样只作为历史决策记录保留。

---

## ADR-022 — wave3 首篇 VLM 补齐发布级两维，且只按自身图像协议验收

- 背景：ADR-021 只交付 `data`/`train`/`eval` 三维，刻意不注册 `task_router`、不声明 `execution.task_id`，发布级 dry-run 显式 `unavailable`。这让 Granite 停留在 `partial_implementation`；但「每一个 recipe 都可作 research / business baseline」的目标要求发布级 `comparison`/`export` 也必须有真实任务代码与隔离离线测试。直接复用数学 1500 题套件、FIM PSM、xLAM 严格 JSON 或 RapidFire 4 配置网格会抹掉图像内容 hash 隔离、processor identity、双层 label mask 与 `exact_match` 语义，等于用别的任务协议冒充验收。
- 决策一（精确路由）：`task_router.py` 新增 `GRANITE = "granite-linecomparison-v1"`，`by_id` 加入 `gt.RECIPE_ID`，`_module()` 精确分派 `granite_task` / `granite_suite` / `granite_export`；`recipes/vlm/sft/granite-vision-line-comparison/recipe.toml` 显式声明 `execution.task_id`。未知、`recipe_id` 与 `task_id` 冲突、借用其他任务协议继续抛 `TaskExecutionError`。`baseline_evidence.py` 的 C306/C307/C405 白名单、`frozen_environment()` 按 Granite 依赖集合 `{torch, transformers, datasets, peft, bitsandbytes, safetensors}` 分派；C405 由 `task_router.load(r, "export")` 落到 `granite_export.validate_export_evidence()`。R103 的「未声明 task_id 时按精确 recipe_id 查白名单」兼容路径保留，但不再作为 Granite 的理由。
- 决策二（comparison）：`granite_suite.py` 只接受三个互异独立训练 seed、共享同一图像内容 hash 隔离的冻结 held-out 与 processor identity；逐题从 `eval_manifest` 的 raw `question/answer` 重建 gold，用 `granite_task.score_prediction()` 对生成文本重算 `exact_match` / `normalized_exact_match`（含 `empty` / `prediction` / `gold_sha256`），绝不信任摘要分数。输出均值、样本方差、SEM、Student-t 95% 区间；`validate_suite_payload()` 与生产聚合器共用强校验，检查代码快照、环境锁字节、有效 seed 配置、checkpoint 权重与逐题预测一致性；不回填 `evaluation.expected/tolerance`、不因有代码自动晋级 B2。`suite_problems()` 用当前 recipe 阈值独立判定 `acceptance`，不采信 JSON 自称的 `passed`。
- 决策三（export）：`granite_export.py` 打包 DoRA/4bit adapter-only 权重，显式声明冻结 base revision / processor identity / chat template、显式 `image_token_id` + assistant-only 双层 mask、RGB 图像输入、`exact_match` 协议、`base_weights_bundled=false`、`generated_code_executed=false`、`upstream_quantitative_evaluation=false`、license pending 与离线数据依赖；`recipe-snapshot.json` 必须与训练时有效配方逐字节一致，锁用虚拟锚点 `_lock-anchor` 保持内容身份不变，`_verify_bundle()` 拒绝未绑定的额外文件。HF offline 新进程真实调用 `granite_task.run_evaluation` 在 held-out 复载复评；真实 recipe 未预声明 `evaluation.export_tolerance`，`--full` 失败关闭，不回填容差。
- 决策四（内容身份差异）：`granite_suite.identity_recipe()` 只剥离实跑后回填的 `evaluation.expected/tolerance`，保留训练前预声明的 `export_tolerance`；与 RapidFire `_identity_recipe()` 不同，它**不剥离** `execution.evidence`——该字段在 Granite 配方里是训练时就存在的位置路径，剥离会导致 checkpoint 的 `recipe_sha256` 与训练时 `effective_recipe.json` 永久失配。这不是放宽，而是按任务实际字段语义保持严格内容身份。
- 可行性联动：`registry/feasibility.toml` 把 Granite 由 `partial_implementation` 改判 `pending_freeze`，`blockers` / `requires` 同步重写；`partial_implementation` 枚举、criteria 与「恰三维实装」交叉校验保留（测试在隔离 sandbox 中重建该状态验证正例与两个负例），但真实登记计数归零。本 ADR 落地时为 `eligible 0/14`、`pending_freeze 5`、`not_implemented 7`、两类原理性阻塞各 1（ADR-023 后变为 `pending_freeze 6` / `not_implemented 6`，`eligible 0/14` 不变）；`baseline --min-state B2/B3` 与 `feasibility --min-state` 仍把 14 篇全部列为失败。五维代码齐备**不解除**两篇 ChartQA 的决策门 / 基座预训练污染阻塞。
- 验证与边界：`tests/test_granite_release.py`（44 项）全部在 `tmp_path` 用合成图像、fake runtime 与受控 worker 运行，覆盖精确路由与协议不互借、三 seed 严格聚合与逐题重算、摘要/权重/processor identity 篡改拒绝、导出包与 C405 内容复核、虚拟锚点锁、无预声明容差时 `--full` 失败关闭；`tests/test_granite.py`（58 项）更新为断言 `task_id` 已精确声明、dry-run 分派到 Granite 专用协议且 `--full` 仍失败关闭。这些只证明代码与协议内部一致，**不认证 GPU 执行真实性**：GPU verified 0、B2+/B3 0、L2+ 0 未变。剩余门槛仍是冻结 model/dataset/source/processor revision 与可安装镜像、授权 Linux CUDA 真实三 seed 实跑、基于实跑预声明容差、HF offline 复载、Geoperception 与基座许可确认、业务验收；`exact_match` 只衡量文本一致性，不解释几何语义、不等于人工判分。

---

## ADR-023 — wave3 第二篇 VLM 用可验证几何指标补齐五维，不冒充上游 grounding 复现

- 背景：HF Cookbook `fine_tuning_vlm_object_detection_grounding` 是 14 篇里唯一带可验证几何量（bbox IoU）的一篇，但一手页面有多处口径冲突：`A.BboxParams(format="coco")` 与 `coco_to_xyxy()` 按 COCO `xywh` 输入，正文与可视化却把 bbox 当 `xyxy`；训练 prompt `<image>Detect {caption}.` 与推理 `<image>detect {caption}` 在大小写、句号、suffix 注入三处不一致；`labels=input_ids.clone()` 只 mask `pad_token_id` 与 `<image>` token，**prompt 也计 loss**；`SFTConfig` 写了 `eval_dataset` / `eval_strategy="steps"` 却没有 `compute_metrics`、IoU、accuracy 或任何数值，只有 `train[20]` / `val[13]` 单样本定性；`do_sample=True` 且无生成 seed；无 `revision` / 训练 seed / 包版本锁；gated 基座 + Flickr URL 运行时下载图像，页面未给 model/dataset/COCO/Flickr 许可条款。直接「复现」会把不可比较的口径当成 baseline。
- 决策一（自建协议 `paligemma-grounding-v1`）：IO 明确声明为 `coco_xywh`，内部统一 `xyxy`，`coco_to_xyxy()` 严格拒绝 NaN、越界、零面积/退化/反向框；`format_location()` 用 `int(round(value * 1024 / max_value))` 生成 4 位 `<loc{v:04}>` 并 clamp 到 `1023`，按 `y1,x1,y2,x2` 排列；`encode_detection()` 只产单框，`decode_detection()` 严格拒绝多框、`;`、loc 数量不为 4、桶值越界与反向框。`ROUND_TRIP_MIN_IOU = 0.98` 承认 1024 桶量化约 1% 的固有误差，但仍强制不丢框、不退化、不越界。`PROMPT_TEMPLATE = "<image>Detect {caption}."` 统一训练与推理，`max_new_tokens = 96`（4 个 `<loc####>` 共 36 字符 + referring expression；沿用上游式 64 会截断 caption 让 `exact_match` 恒为 0，这是真实设计缺陷而非放宽测试）。
- 决策二（三维）：图像原始 `mode + tobytes()` SHA256 做 80/10/10 同图隔离、规范化 `(caption,bbox)` 全局去重；processor 对 prompt / prompt+suffix 两次编码并做 token 级前缀校验，gold 只进 full 序列、绝不进生成 prompt；显式 `image_token_id` + prompt span 外 token 双层 `label=-100`；普通 `transformers.Trainer` 自建 completion-only collator，普通 LoRA，不用 `SFTTrainer` / `skip_prepare_dataset` / 4bit / DoRA / `trl` / `bitsandbytes`；冻结 processor identity / seed / revision / 代码与锁字节。held-out 逐题 greedy + 首个 EOS 截断，`score_prediction()` 重算 `bbox_iou` / `iou_at_0_5` / `parse_rate` / `exact_match`；不执行生成代码。四指标只衡量单框定位一致性，**不等于 mAP/AP、多框检测、人工判分或业务验收**。
- 决策三（精确路由 + 发布级两维）：`task_router.py` 新增 `PALIGEMMA = "paligemma-grounding-v1"`，`by_id` 加入 `pt.RECIPE_ID`，`_module()` 精确分派 `paligemma_task` / `paligemma_suite` / `paligemma_export`；recipe 显式声明 `execution.task_id`，`registry/taxonomy.toml` 登记 `Trainer` 例外；`plan.py`、三个 CLI 的 `TASK_ADAPTERS`、`baseline_evidence.py` 的 C306/C307/C405 白名单与 `fingerprint()` / `frozen_environment()`（依赖集合 `{torch, transformers, datasets, peft, safetensors}`，明确不含 `trl` / `bitsandbytes`）全部按精确 `recipe_id` 分派。`paligemma_suite.py` 只接受三个互异独立训练 seed，逐题从 `eval_manifest` 的 raw COCO bbox + 冻结 width/height + caption 重建 gold detection 与 xyxy，再对 `generated_text` 用 `score_prediction()` 重算四指标和 parsed bbox（不信任摘要分数），输出均值 / 样本方差 / SEM / Student-t95，不回填 `expected/tolerance`。`paligemma_export.py` 打包普通 LoRA adapter-only 权重，声明冻结 base / processor identity、统一 prompt、1024 桶 loc 协议、RGB 输入、单框 IoU、`base_weights_bundled=false`、`generated_code_executed=false`、`license pending`、离线数据依赖、虚拟锚点锁与逐字节一致的 `recipe-snapshot.json`；HF offline 新进程真实调用 `paligemma_task.run_evaluation` 在 held-out 复载复评。真实 recipe 未预声明 `evaluation.export_tolerance`，`--full` 失败关闭、不回填容差。协议不借用 Granite 的 `exact_match`、数学 1500 题、FIM PSM、xLAM 严格 JSON 或 RapidFire ROUGE-L。
- 可行性联动：`registry/evidence.toml` 新增 `[paligemma_grounding_audit]` 记录一手事实与偏差；`registry/feasibility.toml` 把该篇由 `not_implemented` 改判 `pending_freeze`，`reason` / `blockers` / `requires` / `primary_source` 全部重写。能力矩阵从 `25 implemented / 45 missing` 变为 `30 implemented / 40 missing`（五个维度各 6/14），`implemented_recipes` 从 5 变 6；`eligible 0/14`、`pending_freeze 6`、`not_implemented 6`、两类原理性阻塞各 1、`partial_implementation 0`。改判只反映五维代码齐备，**不是晋级**；六篇仍全部缺冻结 revision / CUDA 三 seed / 预声明容差 / 真实 HF offline 复载 / 许可与业务验收。
- 验证与边界：`tests/test_paligemma.py`（46 项）覆盖一手常量、协议漂移、bbox/loc round-trip、非法框、caption 注入、图像隔离、processor identity、双层 mask、四指标、计划/路由、fingerprint、证据分派、端到端合成训练/eval/zero-shot、CLI dry-run 与 `--full` 失败关闭；`tests/test_paligemma_release.py`（44 项）在 `tmp_path` 用 fake runtime 与受控 worker 覆盖三 seed 严格聚合、逐题重算、摘要/权重/processor identity 篡改拒绝、导出包与 C405 内容复核、虚拟锚点锁、无预声明容差时 `--full` 失败关闭。全部是 macOS ARM 隔离合成证据，**不认证 GPU 执行真实性**：GPU verified 0、B2+/B3 0、L2+ 0 未变。剩余门槛：授权 Linux CUDA 实跑三 seed、把 Flickr URL 图像冻结为静态副本、锁定 model/dataset/source/processor revision 与可安装镜像、基于实跑预声明容差、确认 gated 基座与 RefCOCO/COCO/Flickr 许可、补 mAP/AP 或人工判分口径的业务验收。

---

## ADR-024 — 新增方法论依据层：设计决策必须绑定一手核查的论文结论

- 背景：仓库此前只回答「能不能跑起来」（capabilities）、「有没有真实跑过」（baseline/evidence）、「值不值得当 baseline」（feasibility）三个问题，**没有任何论文引用**——调研报告 HTML 0 条 arXiv 链接，registry 无 `papers` / `methodology` 登记。Cookbook 类文档普遍只给配置、不给「为什么这么设计」的依据与边界，导致两条风险：把论文 benchmark 数字误当本仓 expected/tolerance；或让文献背书掩盖原理性阻塞（零训练基线已达标 / 基座预训练污染）。
- 决策一（第四层审计）：新增 `registry/methodology.toml` 与 `src/recipe_lab/methodology.py`，逐篇回答「这么设计依据哪篇 practice 论文的一手结论、论文没证明什么」。它与前三层互不替代，任何一层通过都不解锁 GPU 晋级或 B2/B3。`recipe-lab methodology --check` 校验登记完整性（14 篇精确覆盖、引用闭合、一手字段、边界声明），`--min-verified-papers N` 才是提升目标；当前 3 篇 unverified 是审计结论，不是失败。
- 决策二（一手核查强制）：每条 `papers.*.verified=true` 必须有合法 `arxiv_id`（NNNN.NNNNN）、标题、四位年份、`core_claim` 与 `boundary`；`verified=false` 的论文不得填 `core_claim`、只能进 `supplement_papers`、不得作 `paper_supports` 主张依据。每条主张区分 `paper_supports`（论文一手结论直接支持）、`we_adapted`（我们据论文修正上游，本仓工程判断）、`paper_limited`（论文有明确边界或本仓不适用），并各带 `boundary`。核查中**实际发现 10 条凭记忆写错的 arXiv 编号**（如 FIM `2204.14108` 实为数据科学教育论文、正确为 `2207.14255`；RLAIF-V `2404.14367` 实为 on-policy 偏好论文、正确为 `2405.17220`；RefCOCO `1604.07399` 实为宇宙射线模拟），全部登记到 `[[corrections]]` 的 `was → now`，作为「一手核查不可省略」的物证；未核到一手来源的（MPO 论文、RefCOCO 出处、xLAM 论文）标 unverified，不编造 id 或结论。
- 决策三（防夸大类门禁）：`integrity.forbidden_claim_phrases` 禁止在任何 claim/boundary/stance 写「已在本仓 GPU 验证」「达到 B2」「本仓 expected =」「复现了论文分数」等，把论文依据与本仓实测严格分开。与 `feasibility.toml` 交叉校验：`blocked_zero_train` 篇必须持有 `chartqa_zero_train_gate`、`contaminated_pretraining` 篇必须持有 `chartqa_contamination`，防止用论文背书「洗白」原理性阻塞。悬空引用、编号更正 was==now、非法枚举、僵尸主张均失败关闭。
- 决策四（配方本体引用式回填）：14 篇 `recipe.toml` 各新增 `[methodology]`，只保存 `registry = "registry/methodology.toml"`、`principles`、`primary_papers`、`supplement_papers` 与 `stance`，让读配方时直接看到依据；registry 仍是唯一事实源，不在 14 份配方里复制论文全文。`methodology --check` 逐篇强校验该段存在、registry 指针正确且四个字段与登记完全相等，任一侧漂移即失败关闭。该段只做可读性与一致性用途，不进入任何任务 fingerprint / 内容身份。
- 覆盖范围：全部 14 篇逐篇配方法论依据。当前 **20 篇一手确认 / 3 篇未核实**、24 条主张、10 条编号更正。数学 GRPO 绑定 DeepSeekMath（GRPO 是 PPO 变体、省内存）、reward 过优化与欠规范论文；FIM 绑定 Bavarian 等（数据变换保 L2R）；LoRA/QLoRA/DoRA 支撑 PEFT 装配；DPO/RLAIF-V 支撑偏好篇；PaliGemma/2 + Qwen2-VL/2.5-VL 支撑 VLM 与 grounding；ChartQA 数据集论文 + 去重论文支撑两篇 ChartQA 的原理性阻塞论证。
- 验证与边界：`tests/test_methodology.py`（35 项）在 `tmp_path` 覆盖正例与 21 类负例（非法 arxiv_id、缺 core_claim/boundary、未核实论文冒充主依据或支撑主张、禁止宣称、悬空引用、阻塞篇丢失必需主张、覆盖缺口、僵尸主张、配方本体缺段/缺字段/registry 指针错误/字段漂移、registry 单侧漂移等）；CLI `--check` / `--json` / `--report` / `--min-verified-papers` 全测。这些都是离线登记校验，**不认证论文结论的正确性之外的新事实，也不认证 GPU 执行**：GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14 全部未变。论文依据只解释设计动机与边界，不替代真实三 seed 实跑、许可确认与业务验收。

---

## ADR-025 — 新增 MLLM 后训练方向层：按能力方向映射训练工作，而非按参数机制

- 背景：用户给出三个来源与一套「MLLM 后训练五方向」提纲。一手核查发现三者不是同一套 taxonomy：preprints 202607.1494（*A Survey on Post-Training of Multimodal Large Language Models*，Zhang 等，2026-07-21，DOI `10.20944/preprints202607.1494.v1`）摘要明确给出 instruction following / preference calibration / reasoning enhancement / domain adaptation / scalable training（v2 作 scalable learning）；arXiv 2503.06072 是 LLM PoLM 五范式（Fine-tuning / Alignment / Reasoning / Efficiency / Integration and Adaptation），不是 MLLM 五方向原始出处；`WenkeHuang/Awesome-MLLM-Tuning` 的顶层分类是 Selective / Additive / Reparameterization Tuning，和方法/行为方向不同轴，且没有统一可复现配置。若混用会把二手提纲误写成原文事实，也会用方法索引冒充复现证据。
- 决策一（第五层审计）：新增 `registry/mllm_post_training.toml` 与 `src/recipe_lab/post_training.py`，回答「训练工作覆盖 MLLM 后训练哪个能力方向、缺口在哪」。它与前四层互不替代，任何一层通过都不解锁 GPU 晋级或 B2/B3。`post-training --check` 校验来源边界、五方向闭合、14 篇精确映射和能力/可行性交叉；`--min-fivedim-directions N` 是提升目标。
- 决策二（来源分级）：`verification` 分为 `abstract_and_index` / `abstract` / `readme`。MLLM 综述只能作方向原始出处，且必须保留「正文被 Akamai 防护，仅标题/作者/日期/DOI/摘要经索引交叉核实」的边界；LLM 综述只能作 `cross_check`；Awesome repo 只能作 `method_index`。readme 级索引不得作方向出处；任何来源都不得支撑方法细节、benchmark 数值、本仓 expected/tolerance 或 GPU 实测。
- 决策三（映射与缺口）：14 篇逐篇登记 `primary` / `secondary` / `rationale`；primary 必须非空且与 secondary 不重叠；每个方向至少有一篇 primary 映射。实现状态从 `capabilities.toml` 重算，不采信本登记自称；`blocked_zero_train` 与 `contaminated_pretraining` 篇显式标为无效覆盖证据，不能靠数量掩盖原理性阻塞。ADR-027 补齐 DPO/MPO 五维后，五方向均有五维实现支撑；方向覆盖不等于 GPU baseline。
- 验证与边界：`tests/test_post_training.py` 覆盖来源越权、方向缺失、映射缺口、枚举/计数错误、禁止宣称、能力/可行性漂移与 CLI 行为。该层只做覆盖地图，不表示 DPO/RLAIF-V 已有 baseline，不表示论文中的具体方法已在本仓实测。GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14 全部未变。

---

## ADR-026 — 新增 Agent 交接契约：人类可读入口与机器可复现清单同源

- 背景：仓库已有五层审计和大量 Markdown，但新读者与编码 Agent 仍要自己拼出「先读什么、改哪些事实源、跑哪些命令、哪些门禁必须失败、哪些数字不能信」。口头说明会漂移，README 越写越长，Agent 也可能把生成报告当事实源或直接 push。
- 决策一（双视图）：新增静态 `AGENTS.md`（Agent 可发现入口）、生成 `docs/AGENT-HANDOFF.md`（人类可读现状与工作流）和 `docs/AGENT-HANDOFF.json`（机器可读事实源、生成物哈希、命令契约与 Guardrails）。三者都由 `registry/agent_contract.toml` 约束；该注册表不是第六层审计，只聚合已有五层事实。
- 决策二（可重建性优先）：`recipe-lab handoff --check` 用真实 CLI 把五份生成报告重建到临时目录并逐字节比较，任一报告与当前源码/注册表不一致即失败关闭。`docs/AGENT-HANDOFF.md/json` 自身由同一审计器生成，不引入时间戳或绝对路径，重复生成结果稳定。
- 决策三（硬边界）：契约禁止手改 `docs/` 生成物；登记 `pass` 与 `fail_closed` 两类命令，`--min-*` 提升目标必须继续失败；Guardrails 明确不得把静态检查当 GPU 验证、不得绕过原理性阻塞、未得明确要求不得 push。禁止宣称模式扫描所有面向Agent的说明文字。
- 验证与边界：`tests/test_agent_contract.py`（21 项）覆盖事实源存在、生成物哈希、命令枚举、失败关闭门禁登记、禁止宣称、重复 id、缺 `{out}` 占位、以及真实 CLI 重建一致性。`make ci` 接入 `agent-check`；`make report` 接入 `handoff --report`。本契约只提升可发现性与可复现性，不认证 GPU 执行、不解锁 B2/B3，也不替代任何一层审计。

---

## ADR-027 — wave4 偏好篇补齐五维：共享偏好 split，禁止跨 backbone 归因

- 背景：ADR-025 把 `preference_calibration` 标为唯一五维缺口；wave4 要求 DPO/MPO 有真实任务代码与隔离离线测试，且不得把不同基座的两篇当成算法消融。
- 决策一（自建协议）：`smolvlm-dpo-v1` 用普通 Trainer + 显式 beta=0.1 sigmoid、冻结 ref、图像 hash 80/10/10、completion-only 双层 mask、隐式奖励 pairwise；`qwen25vl-mpo-v1` 在同一偏好 split 算法上叠加 sigmoid/bco_pair/sft（0.8/0.2/1.0）与 full/sigmoid_only/no_sft 消融，并如实登记 bf16=False 与量化 bfloat16 矛盾。
- 决策二（精确路由）：`task_router` 注册两篇的 task/suite/export；C306/C307/C405 与 fingerprint/frozen_environment 按任务分派；`cross_backbone_attribution_valid=false` 写入配方、套件计划与导出包。
- 决策三（登记联动）：capabilities 五维改 `implemented`，feasibility 改 `pending_freeze`，evidence 增补 `dpo_adapter_audit` / `mpo_adapter_audit`；后训练方向 5/5 有五维支撑后，`gate-post-training` 提升目标改为 `--min-fivedim-directions 99` 继续失败关闭。
- 边界：全部是 macOS ARM 隔离合成证据；GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14 未变。隐式奖励不等于人类偏好一致率；若要做 DPO vs MPO 算法消融，必须换 shared backbone 或显式改写对照目标。

---

## recipe.toml 字段规范

| 段 | 关键字段 | 必需性 | 相关规则 |
| --- | --- | --- | --- |
| 顶层 | `schema` `id` `name` `status` `coverage` `confidence` `source` | 全部必需 | R001–R005 |
| `[method]` | `family` `trainer` `peft` | `family` 必需 | R101–R103 |
| `[model]` | `base` `modality` `params_b` | `modality` 必需 | R114 |
| `[dataset]` | `name` `structure` `type` `has_images` | `type` / `structure` 必需 | R104–R105 |
| `[modality_contract]` | `images_column` `content_list_dict` `processing_class` `max_length` | VLM 必需 | R106–R110 |
| `[consistency]` | `train_system_message` `inference_system_message` | 建议 | R111 |
| `[quantization]` | `load_in_4bit` `bnb_4bit_compute_dtype` | 4bit 时必需 | R113 |
| `[rapidfire]` / `[[configs]]` | 自建协议段与 4 组配置网格；`execution.config_id` 选择当前配置，`selection_policy` / `eval_stages` 约束 validation 选配置与 test 单次验收 | RapidFire 任务必需 | ADR-018 / ADR-019 |
| `[paligemma]` / `[upstream]` | 自建单框定位协议：`coco_xywh`→`xyxy`、1024 桶 `<loc{v:04}>`、`round_trip_min_iou`、`prompt_format`、双层 mask 与四指标定义；`location_bins` / `location_token_order` 绑定 fingerprint | PaliGemma 任务必需 | ADR-023 |
| `[dependencies]` | 精确版本键值对 | 建议 | R201–R203 |
| `[rewards]` | `functions` `failure_path_tested` | RL 必需 | R115 / R307 |
| `[baseline]` | `zero_train` `result` | 必需 | R301–R303 |
| `[evaluation]` | `metrics` `held_out` `seed` `multi_run` | 建议（ported 必需） | R304–R306 |
| `[budget]` | `smoke_steps` `full_steps` | 建议（ported 必需） | R308 |
| `[known_issues]` / `[upstream_deviations]` / `[todo]` | `items` 数组 | 条件必需 | R309 / R312 |
| `[execution]` | `command` `entrypoint` `lockfile` `eval_script` `zero_shot_script` `evidence` `adapter_status` | **全部 recipe 必需** | R401–R403 / C204–C206 / C301 |
| `[resources]` | `gpu` `note` | B3 需要实测值 | C401 |
| `[baseline_contract]` | `state` `target_tier` `experiment_id` `revision_policy` `determinism` `python_cuda` `single_variable` | **全部 recipe 必需** | R601–R606 |
| `[methodology]` | `registry` `principles` `primary_papers` `supplement_papers` `stance` | **全部 recipe 必需**：仅引用 registry，逐字段一致性门禁 | ADR-024 |
| `[repro]` | `level` `assessed_at` `assessed_by` `evidence_ref` | **全部 recipe 必需** | R501–R503 / R507 |
| `[repro.<dim>]` | `state` `gap`（八个维度各一段） | **全部 recipe 必需** | R504–R506 |

`[baseline_contract]` 的 `state` 不是权威值：由 `registry/baseline.toml` 的
28 条检查重算，`R603` 拦截虚报；B2 / B3 额外要求的 `license_model` /
`license_data` / `acceptance` / `owner` 目前**刻意留空**，等实跑与许可核查后回填。
`train.seed` / `train.data_seed` 是**本仓库定义 baseline 时声明**的种子，
两篇上游确有 seed 的（code LLM `SEED=0`、RapidFire `42`）保留原值，
`seed_origin` 逐篇记录这个区别——它们不是上游原实验的复现参数。

`[repro]` 的 `level` 不是权威值：它由 `registry/reproducibility.toml` 重算，
`R503` 会比对两者的差异。`evidence_ref` 取值必须是
`local-notebook:<文件名>` / `webfetch:<页面名>` / `research-report-only`，
前两者要能在 `registry/evidence.toml [method]` 里找到登记（R507）。

`todo`、`known_issues`、`upstream_deviations`、`notes` 使用 `items` 数组，
是为了让 TOML 保持扁平可读，不需要为每条备注造一个 key。
