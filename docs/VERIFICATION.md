# 分层验证

```bash
recipe-lab validate --level l3          # 默认：跑到 L3
recipe-lab validate --level l5          # 含可复现性审计规则 R501-R507
recipe-lab validate --level l6          # 再加基线就绪规则 R601-R606（CI 用）
recipe-lab validate --strict            # warning 也算失败（退出码 2）
recipe-lab validate --json              # 机器可读，供 CI / 外部工具消费
recipe-lab validate vlm/                # 只验证某个 id 前缀

recipe-lab repro                        # 可复现性审计摘要
recipe-lab repro --check                # 校验审计完整性（不要求「都可复现」）
recipe-lab repro --check --min-level L2 # 显式设提升目标，当前会失败
recipe-lab repro --report               # 重新生成 docs/REPRODUCIBILITY.md

recipe-lab baseline                     # 基线就绪摘要（B0-B3 + 四组能力）
recipe-lab baseline --check             # 校验状态无虚报（不要求「都到 B2」）
recipe-lab baseline --check --min-state B2   # 显式设提升目标，当前会失败
recipe-lab baseline --dry-run           # 打印 14 篇训练计划，零依赖、无需 GPU
recipe-lab baseline --report            # 重新生成 docs/BASELINE-READINESS.md
recipe-lab capabilities --check         # 五维能力登记：精确覆盖、实现/测试路径、实跑JSON
```

退出码：`0` 通过、`1` 有 error、`2` strict 模式下存在 warning、`130` 中断。
`repro --check` / `baseline --check` 同 `0` / `1`，但它们**不把「0/14 可复现」、
「0/14 达到 B2」当作失败**——见下方 L5 / L6 说明。

---

## 分层

| 层 | 关注点 | 是否需要 GPU | 默认 |
| --- | --- | --- | --- |
| **L0 结构** | schema / id / 枚举 / 必填字段 | 否 | ✅ |
| **L1 契约** | trainer 映射、dataset type、VLM 数据契约、训练-推理一致性 | 否 | ✅ |
| **L2 版本** | 精确 pin、组合白名单、来源可信度、跨 recipe 漂移 | 否 | ✅ |
| **L3 纪律** | 零训练基线、评估闭环、reward 失败路径、成本闸门 | 否 | ✅ |
| **L4 执行** | 入口脚本 / evidence 文件存在性 | 否（只做静态检查） | ❌ |
| **L5 可复现性** | `[repro]` 八维度、等级重算防虚报、证据可追溯 | 否（只做一手材料核查） | ❌ |
| **L6 基线就绪** | 四组能力、状态重算，冻结环境与关联证据内容/hash/身份、数学指标重算 | 否（只读静态证据，不重新训练） | ❌ |

L4 刻意**不跑真实训练**：本仓库是配方登记与契约层，GPU 执行交给外部 runner。
`--level l4` 只检查声明的文件是否存在。

**L4 与 L5 是两件事，别混为一谈。**

| | L4 执行 | L5 可复现性 |
| --- | --- | --- |
| 问的问题 | 声称跑过，证据文件在不在？ | 声称能重放，模型/数据/环境/种子真的锁了吗？ |
| 依据 | `execution.entrypoint` / `evidence` 路径 | 一手材料核查（`registry/evidence.toml`） |
| 当前结论 | 14篇均有统一入口；数学/FIM/xLAM/RapidFire/Granite/PaliGemma 六篇五维代码已实现，其余8篇实跑拒绝；实跑证据0篇 | **14 篇中 0 篇达到 L2「可复现」** |
| 失败含义 | 文件缺失 | 等级虚报、维度状态非法、证据引用查无出处 |

L0–L5 全绿只证明**配置自洽 + 审计诚实**，不证明**结果可复现**。
这是本仓库最重要的一条边界：上一轮 `0 error / 33 tests passed` 曾被误读成
「recipe 可复现」，实际它连 L5 都还没跑。

### 可复现性等级（L0–L4）

与验证层级 L0–L5 同名的数字含义不同，注意区分：验证层级是**规则分层**，
下面是 recipe 的**能力等级**。契约在 `registry/reproducibility.toml`。

| 等级 | 含义 | 最短板要求 |
| --- | --- | --- |
| L0 仅登记 | 只有方法 / 模型 / 数据集名称 | `provenance ≥ declared` |
| L1 规格完整 | 参数、依赖、数据契约、评估协议齐备但未锁定 | 6 个维度 `≥ declared` |
| L2 完全锁定 | 模型/数据 revision、依赖 lock + commit、seed、上游 commit | 含 `seed ≥ pinned` |
| L3 可执行 | L2 之上补确定入口与评估脚本 | `code ≥ pinned`、`eval ≥ pinned` |
| L4 已复现验证 | 声明硬件上重跑、指标落在容差内 | `code` / `eval` 均为 `verified` |

八个维度：`code` `env` `model` `data` `params` `seed` `eval` `provenance`，
状态取 `missing → declared → pinned → verified`。等级由八个维度的**最低达标面**
自动算出，`R503` 会用同一函数重算并与 `repro.level` 比对，虚报即 error。
阈值 `reproducible_threshold = "L2"`：**达到 L2 及以上才算可复现。**

`recipe-lab repro --check` 只校验**审计完整性**（等级无虚报、证据引用可追溯）。
把「0/14 可复现」当 CI 失败，只会逼人去改数字而不是补锁文件；
需要阶段性目标时显式用 `--min-level L2`。

### 基线就绪状态（B0–B3）

L6 问的是**另一个问题**：「这篇 recipe 能不能当别人做研究 / 业务训练的参照物？」
它不要求复刻上游原实验，只要求我们自己锁定一套确定的环境、权重、数据、种子，
并把实测指标记下来。契约在 `registry/baseline.toml`，四组能力共 28 条检查：

| 状态 | 含义 | 当前 |
| --- | --- | --- |
| B0 not-ready | B1 的必要条件都没齐 | 0 篇 |
| B1 declared | 执行契约齐备、入口可 dry-run，但**尚未实跑** | **14 篇** |
| B2 research-ready | 额外有真实 revision、零训练基线实测、held-out 指标 + 容差 + 多轮、可复查 evidence | 0 篇 |
| B3 business-ready | B2 之上补许可、资源成本、验收门槛、owner 与回滚 | 0 篇 |

**实验目标是否成立另有一层判定**：代码实装与 GPU 证据都不能替代它。
`registry/feasibility.toml` / `recipe-lab feasibility` 逐篇给出 `eligible 0/14`、
`pending_freeze 6`、`not_implemented 6`、`blocked_zero_train 1`、
`contaminated_pretraining 1`、`partial_implementation 0`；`baseline --min-state B2/B3` 会把非 eligible 篇目列为失败。
`partial_implementation` 枚举与交叉校验仍保留（恰三维 implemented + 发布级两维 missing），
但 ADR-022 / ADR-023 补齐 Granite / PaliGemma 五维并精确注册 `task_router` 后，真实登记已无篇目使用该 verdict，
两篇改判 `pending_freeze`——这只代表五维代码齐备，不是 baseline 晋级。
`qwen2-vl-7b-chartqa` 上游自承零训练基线即可达标（`decision.gate=blocked`），
`smolvlm-chartqa` 基座本就在 ChartQA 上训练过，这两类**补齐代码和 CUDA 实跑也不能晋级**，
须改写实验目标 / 换未污染数据后重核一手材料。Granite Vision 篇的一手核查见
`registry/evidence.toml [granite_vision_audit]` 与 [DESIGN.md](DESIGN.md#adr-020)；
三维自建协议见 [DESIGN.md](DESIGN.md#adr-021) 与 `tests/test_granite.py`（58 项），
发布级两维见 [DESIGN.md](DESIGN.md#adr-022) 与 `tests/test_granite_release.py`（44 项隔离离线测试）。
PaliGemma2 grounding 篇的一手核查见 `registry/evidence.toml [paligemma_grounding_audit]` 与
[DESIGN.md](DESIGN.md#adr-023)：`tests/test_paligemma.py`（46 项）与
`tests/test_paligemma_release.py`（44 项隔离离线测试）覆盖 1024 桶 `<loc####>` round-trip/IoU、
单框 completion-only mask 与从 raw COCO bbox 重建 gold 的三 seed 四指标重算。

**理论依据另有一层**：`registry/methodology.toml` / `recipe-lab methodology`（ADR-024，第四层审计）
逐篇把设计决策绑定到 practice 论文的一手结论，回答「为什么这么设计、论文没证明什么」。
当前 20 篇一手确认 / 3 篇未核实（仅作补充）、24 条主张、10 条编号更正。
`verified=true` 必须有合法 arXiv id / 标题 / 年份 / `core_claim` / `boundary`；
核查中实际发现并更正 10 条凭记忆写错的编号（FIM `2204.14108`→`2207.14255`、
RLAIF-V `2404.14367`→`2405.17220`、RefCOCO `1604.07399` 实为宇宙射线模拟等）。
`methodology --check` 校验引用闭合、一手字段与 14 篇 `recipe.toml [methodology]` 的逐字段一致性；
`--min-verified-papers N` 才是提升目标。配方本体的 `[methodology]` 只是 registry 引用式视图，
缺段、registry 指针错误、任一字段漂移或 registry 单侧改动都失败关闭；
禁止宣称模式不许把「已在本仓 GPU 验证」「本仓 expected =」写入任何主张；
与 feasibility 交叉校验确保两篇 ChartQA 的原理性阻塞不被论文背书洗白。
论文 benchmark 数字绝不回填本仓 expected/tolerance，文献背书不解锁 B2/B3。见
[METHODOLOGY.md](METHODOLOGY.md)（生成）与 `tests/test_methodology.py`（35 项）。

**能力方向另有一层**：`registry/mllm_post_training.toml` / `recipe-lab post-training`（ADR-025，第五层审计）
按行为/能力把 14 篇映射到指令遵循、偏好校准、推理增强、领域自适应、可扩展学习五方向。
五方向原始出处只能是 preprints 202607.1494 的摘要级框架（正文页受防护，仅索引与摘要交叉核实）；
arXiv 2503.06072 只能作 `cross_check`，`Awesome-MLLM-Tuning` 只能作 `method_index`。
`verification=readme` 的索引不得作方向出处，也不得支撑方法细节、benchmark 数值或本仓实测。
`post-training --check` 校验五方向闭合、14 篇精确映射、primary/secondary 不重叠、
实现状态从 `capabilities.toml` 重算，且 `blocked_zero_train` / `contaminated_pretraining`
篇标为无效覆盖证据；`--min-fivedim-directions N` 才是提升目标。
当前 4/5 方向有五维实现支撑，偏好校准为缺口；这不表示 DPO/RLAIF-V 已有 baseline。
见 [MLLM-POST-TRAINING.md](MLLM-POST-TRAINING.md)（生成）与
`tests/test_post_training.py`（27 项）。

`G1 可引用` / `G2 可重放` / `G3 可对照` / `G4 可交付` 四组各带 `min_state`，
`R603` 用同一个 `compute_state()` 重算并与 `baseline_contract.state` 比对，
虚报即 error。B1 只代表「命令与参数确定、可以 dry-run」，**不代表跑通过一次**：
数学math-grpo-v1、FIM code-fim-v1、xLAM xlam-fc-v1、RapidFire rapidfire-sft-v1、Granite granite-linecomparison-v1与PaliGemma paligemma-grounding-v1六篇五维均已有代码与隔离离线测试（Granite 为58+44项、PaliGemma 为46+44项），GPU与真实completed证据0篇；其余8篇任务尚未实现。六篇发布级校验各由`task_router.py`精确分派到本任务`suite`/`export`模块：数学1500题gold/manifest、FIM PSM输入/gold、xLAM严格JSON prompt/tools/gold、RapidFire 4配置×3 seed 的 raw instruction/response 重建 prompt/gold 与无stemmer ROUGE-L/exact_match 逐题重算、Granite 图像内容hash隔离的三seed逐题 question/answer 重建gold与exact_match重算、PaliGemma 图像内容hash隔离的三seed逐题从 raw COCO bbox + caption 重建gold detection/xyxy 与 bbox_iou/iou_at_0_5/parse_rate/exact_match 重算，互不借用。RapidFire 只用validation选配置、test单次验收，导出选中adapter并在HF offline新进程test复评；Granite 导出DoRA/4bit adapter-only包并在HF offline新进程held-out复评。内容身份按任务分派：RapidFire `_identity_recipe()`剥离实跑后回填的阈值与证据指针，Granite `identity_recipe()`只剥离阈值（`execution.evidence`是预声明位置路径，剥离会破坏训练时内容身份）；`acceptance`判定不被JSON自证，真实recipe未预声明`export_tolerance`时`--full`失败关闭。未知或冲突任务失败关闭，C306/C307/C405不靠提交普通JSON放行，合成JSON也不能替代真实运行。
实跑缺冻结输入/环境、任务路由或API不兼容时明确拒绝，绝不产占位分数。
C306重读真实base内容，C307与生产聚合器共用代码/锁/有效seed配置/严格1500条manifest/预测/统计校验；不能直接提交宽松套件JSON绕过。
C405重读`export.json`及完整导出包，核新进程身份、权重/文件/锁、协议与复载容差；`export_verified=true`不能单独放行，回滚仍需单独`rollback_verified/rollback_evidence`人工复核。
详见 [BASELINE-READINESS.md](BASELINE-READINESS.md)。

**L5 与 L6 的结论难看都不算 CI 失败。** 两个 `--check` 只防「改数字冒充进展」；
要设提升目标就显式加 `--min-level L2` / `--min-state B2`。这是 ADR-008 与 ADR-010
共同确立的门禁哲学。

---

## 规则编号表

### L0 — 结构

| ID | 严重度 | 检查 |
| --- | --- | --- |
| R001 | error | 顶层必填字段齐全 |
| R002 | error | `id` 等于 `recipes/` 下相对目录 |
| R003 | error | `status` / `coverage` / `confidence` / `modality` 枚举合法 |
| R004 | error | `schema` 与 taxonomy 版本一致 |
| R005 | error | 声称 `smoke-passed` / `benchmarked` 必须 `coverage = "params"` |

### L1 — 契约

| ID | 严重度 | 检查 |
| --- | --- | --- |
| R101 | error | `method.family` 已在 taxonomy 登记 |
| R102 | warning | family 属于 `removed` 层级（如 PPO） |
| R103 | error | `method.trainer` 与 family 登记值一致（MPO → `DPOTrainer`） |
| R104 | error | `dataset.type` 合法且被该 trainer 接受 |
| R105 | error / info | `dataset.structure` 合法；VLM 必须 `conversational`。**例外**：`prompt-only` + `pre_rendered_prompt=true` 时为 info |
| R106 | error | VLM 必须声明 `[modality_contract]` |
| R107 | error | 顶层图像列必须是 `images` / `image` |
| R108 | error / info | `content_list_dict = true`；prompt-only 预渲染例外时为 info |
| R109 | error | `processing_class = "processor"` |
| R110 | error / warning | VLM SFT/偏好训练 `max_length = "none"`；在线 RL 声明 `max_prompt_length` 策略 |
| R111 | error | 训练与推理 system message 策略一致 |
| R112 | warning | `use_dora=True` 但 tags 仍宣称 LoRA/QLoRA |
| R113 | error | 4bit 量化的 compute dtype 为 bf16/fp16 |
| R114 | error | `modality` 与 `dataset.has_images` 一致 |
| R115 | error | 需可验证奖励的 family 必须声明 `[rewards].functions` |

### L2 — 版本

| ID | 严重度 | 检查 |
| --- | --- | --- |
| R201 | error | 依赖精确 pin，拦截 `>=` / `^` / `latest` / 无 commit 的 git URL |
| R202 | warning | `trl` + `transformers` 组合在 `[[compat.matrix]]` 白名单内 |
| R203 | warning | dev / rc / alpha 预发布版本（通常需源码安装） |
| R204 | warning | **全局**：仓库内存在多个 trl 版本，跨篇复制有风险 |
| R205 | error | `source` 为登记域名的 HTTPS 链接 |
| R206 | info | 未锁定上游 commit（`source_sha`） |
| R207 | info | **全局**：同一 family 下版本不一致会污染对照实验 |

### L3 — 纪律

| ID | 严重度 | 检查 |
| --- | --- | --- |
| R301 | error / warning | 存在零训练基线且 `zero_train = true` |
| R302 | error | 基线已达标却不是 `cataloged` —— **决策门** |
| R303 | warning / info | 基线状态提示（达标仅作对照 / 未测量需补） |
| R304 | error / warning / info | 定量指标，拒绝 `vibe_check` 等占位 |
| R305 | error | `held_out = true` |
| R306 | warning | 固定 seed；非 catalog 状态 `multi_run = true` |
| R307 | error | RL recipe 声明 reward 失败路径测试 |
| R308 | error / warning | `smoke_steps < full_steps`；非 catalog 需成本预算 |
| R309 | warning | 登记了上游缺陷必须有对应处置 |
| R312 | warning | `cataloged` 必须有 `[todo]` |

### L4 — 执行

| ID | 严重度 | 检查 |
| --- | --- | --- |
| R401 | error | `execution.entrypoint` 指向的文件存在 |
| R402 | info | 未声明入口（当前仓库常态） |
| R403 | error / warning | 已跑通状态必须提供可复查的 evidence |

### L5 — 可复现性审计

| ID | 严重度 | 检查 |
| --- | --- | --- |
| R501 | error | `[repro]` 审计段存在 |
| R502 | error / warning | `level` / `assessed_at` / `assessed_by` / `evidence_ref` 必填且合法；`assessed_at` 落后于契约日期则 warning |
| R503 | error | **支点规则**：用 `compute_level()` 重算等级，与声明的 `repro.level` 不符即虚报 |
| R504 | error | 八个维度段齐全，`state` 属于 `missing/declared/pinned/verified` |
| R505 | error | 声称 `pinned/verified` 时必须有 backing 字段；`code` / `env` / `eval` 还要指向真实文件 |
| R506 | warning | 非 pinned 维度必须写清 gap；已 pinned 却仍留 gap 也提示 |
| R507 | error / warning | **全局**：`evidence_ref` 必须能在 `registry/evidence.toml` 找到对应一手材料；`research-report-only` 只有二手材料，给 warning |

L5 一手核查结果见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)：
14/14 篇回上游核对（3 篇本地 `.ipynb` 原文 + 11 篇页面事实抽取），
结论 **L0 8 篇 / L1 6 篇 / L2+ 0 篇**。

### L6 — 基线就绪

| ID | 严重度 | 检查 |
| --- | --- | --- |
| R601 | error | `[baseline_contract]` 基线就绪契约段存在 |
| R602 | error | `state` / `target_tier` / `experiment_id` / `revision_policy` 必填且枚举合法 |
| R603 | error | **支点规则**：用 `compute_state()` 按 28 条检查重算，与声明的 `state` 不符即虚报 |
| R604 | warning | 状态为 B0 却没有 `[todo]` 说明补齐项 |
| R605 | error | 声称 B2 / B3 却未满足对应 `min_state` 的检查（防止只改字符串升级） |
| R606 | info | 目标档位（research→B2 / business→B3）与当前状态的差距 |

L6 全绿只证明**执行契约自洽 + 状态诚实**，不证明训练跑过、指标可信。
逐篇就绪度、五维实际能力与wave0–wave6实施队列见 [BASELINE-READINESS.md](BASELINE-READINESS.md)：
当前 **B1 14 篇 / B2 0 篇 / B3 0 篇**。

---

## 严重度与门禁

- **error**：会让训练跑挂、得出错误结论，或绕开决策门。必须修。
- **warning**：已登记、尚未消解的风险。`make ci` 不阻塞；`--strict` 阻塞。
- **info**：提示与待办入口。

CI（`.github/workflows/validate.yml`）执行：
`doctor` → `validate --level l6`（0 error，含 L5 + L6）→ `pytest` →
`repro --check` + `baseline --check`（审计完整性）+ `capabilities --check`（五维覆盖）+
`feasibility --check`（原理层）+ `methodology --check`（论文一手字段与引用闭合）+
`post-training --check`（方向层映射与来源边界）→
三个 runner 入口的 dry-run 冒烟 → 重建 `docs/MATRIX.md`、
`docs/REPRODUCIBILITY.md`、`docs/BASELINE-READINESS.md`、`docs/METHODOLOGY.md`、
`docs/MLLM-POST-TRAINING.md` 并检查 `git diff --exit-code`，防止有人手改生成物。

`--strict` 留给**提升 `status` 之前**的复核，例如把某篇从 `ported` 推进到
`smoke-passed` 时，先把该篇的 warning 清零：

```bash
recipe-lab validate vlm/sft/smolvlm-chartqa --strict
```

---

## 如何新增规则

1. 在 `src/recipe_lab/rules.py` 对应层的函数里加一段 `yield from _emit(...)`，
   分配一个未使用的编号（层内递增即可，不要求连续，如 `R312`）。
2. 规则只依赖 `Recipe.get/has` 与 `Context`（taxonomy、全部 recipe、root），
   核心结构规则不读文件；证据层允许只读文件内容/hash。均不联网、不导入训练框架。
3. 在 `tests/test_rules.py` 补两条测试：**正确配置不触发**、**故意写错会触发**。
   只写前者等于没测。
4. 更新本文件的规则表与 `docs/TRAPS.md` 的映射列。

---

## 不覆盖的范围

诚实说明边界，避免把「静态校验通过」误读成「配方有效」：

- 不验证超参是否最优、模型是否能收敛、指标是否真实。
- 不执行 notebook、不下载数据集、不检查 GPU 显存是否真的够。
- 不校验上游 notebook 内容是否变化（`source_sha` 目前靠人工记录，
  可用 `scripts/check_source_drift.py` 辅助）。
- 不做数值回归：`baseline.result` 是人工填写的结论，验证器只检查它与
  `status` 是否自洽，不重新跑基线实验。
- **L5 只查「声明与证据是否一致」，不查「锁定是否真的有效」**：
  即使某篇把八个维度全标 `pinned` 且 backing 文件都在，也没有真的装环境、
  下权重、跑训练去验证重放结果。本机没有 GPU，14 篇 `source_sha` 全为空，
  11 篇只有网页抽取而非 `.ipynb` 原文逐字节核对。
- **L6检查证据内部一致性，不认证GPU执行真实性**：要求真实结构化环境锁、关联训练/评估JSON、
  代码与split快照、文件hash、预算/身份/参数及数学逐题指标重算；仍不能仅靠手填JSON证明真实执行。
  B1仅代表声明，data/train/eval的实际代码与GPU验证另由capabilities登记。
  其余8篇拒绝实跑；数学/FIM/xLAM/RapidFire/Granite/PaliGemma六篇五维代码与隔离离线测试齐备，Granite/PaliGemma发布级已按精确recipe_id注册并分派到自身协议；六篇都须先冻结输入与真实CUDA环境（PaliGemma 还须把 Flickr URL 图像冻结为静态副本），任何任务不能借用别的任务发布证据校验器晋级；发布级内容校验按任务分派，未知或冲突任务失败关闭，合成JSON不能替代真实运行；真实recipe未预声明`export_tolerance`时导出`--full`失败关闭，不回填容差。
- 因此 **L0–L6 全绿 ≠ recipe 可复现，也 ≠ 可以作为 research baseline**。
  当前两个结论分别由 `docs/REPRODUCIBILITY.md`（0/14）与
  `docs/BASELINE-READINESS.md`（B2+ 0/14）承载。

数值层面的确认需要外部 GPU runner 与评估流水线接入后，由 L4 的
`execution.evidence` 产物 + 可复现性 L4 `verified` 状态共同承担。
