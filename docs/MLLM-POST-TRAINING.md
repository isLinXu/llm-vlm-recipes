# MLLM 后训练方向覆盖审计

> 由 `recipe-lab post-training` 生成；单一事实源是 `registry/mllm_post_training.toml`。本文件每次 `make report` 重建，请勿手工编辑。

## 为什么需要这一层

前四层回答的是能不能跑、有没有跑过、值不值得当 baseline、设计依据哪篇论文；本层回答**训练工作覆盖 MLLM 后训练的哪个能力方向、缺口在哪**。

五方向来自 *A Survey on Post-Training of Multimodal Large Language Models* 的摘要级框架：指令遵循、偏好校准、推理增强、领域自适应、可扩展与高效学习。arXiv 2503.06072 是 LLM 后训练五范式的跨域交叉印证，Awesome-MLLM-Tuning 是不同轴的方法索引；两者都不是五方向原始出处。

来源边界：preprints 正文页被防护，仅标题/作者/日期/DOI/摘要经索引交叉核实；本层不支撑具体方法细节、benchmark 数值或本仓 GPU 实测。任何一层通过都不解锁 B2/B3，eligible 与 GPU verified 仍独立判定。

当前覆盖 **14/14 篇**、**5 个方向**；五维已实现 recipe 9 篇，有五维实现支撑的方向 5/5，缺口 0 个。

## 来源边界

| id | 类型 | 核实级别 | 用途 | 边界 |
| --- | --- | --- | --- | --- |
| `awesome_mllm_tuning` | index | readme | method_index | 该 taxonomy 与方法/行为五方向不同轴；README 不提供统一可复现配置、超参、数据划分或评测命令。仅作方法索引，不作五方向来源或复现证据。 |
| `llm_post_training_survey` | survey | abstract | cross_check | 这是 LLM 而非 MLLM 五方向的原始出处，仅作跨域交叉印证；不得把其范式名替换本层方向名，也不得据此声称 MLLM 五方向来自本文。 |
| `mllm_post_training_survey` | survey | abstract_and_index | directions | 正文页被 Akamai 防护；仅标题、作者、日期、DOI 与摘要由 preprints 索引及 Google Scholar/Scilit 交叉核实。不得支撑具体方法细节、论文内数值、benchmark 结论或本仓实测。 |

## 五方向与覆盖

| direction | 定义 | 主方向篇数 | 映射篇数 | 五维实现篇数 | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `domain_adaptation`（领域自适应） | 面向医疗、遥感、烹饪、工业检测等垂直领域定制后训练；综述强调垂直领域常见单阶段后训练，与通用模型两阶段范式不同。 | 1 | 3 | 3 | 已有五维实现支撑 |
| `instruction_following`（指令遵循） | 从图像-文本对话 SFT 向高分辨率视觉 token、交错多模态输入与高质量指令数据配比演进，使 MLLM 遵循用户显式指令。 | 6 | 7 | 5 | 已有五维实现支撑 |
| `preference_calibration`（偏好对齐与校准） | 把 DPO / RLAIF 等偏好优化扩展到多模态，处理视觉幻觉、不实描述与人类价值对齐；包含 listwise 多候选排序与多模态 AI 反馈。 | 2 | 3 | 3 | 已有五维实现支撑 |
| `reasoning_enhancement`（推理增强） | 借鉴 CoT 与可验证奖励强化学习，使 MLLM 从看图说话转向多步视觉、空间、数学与文档推理，GRPO 等群组相对优势方法是代表路线。 | 4 | 7 | 3 | 已有五维实现支撑 |
| `scalable_learning`（可扩展与高效学习） | 在参数规模或模态数量扩展时维持训练与部署效率，覆盖离线/在线蒸馏、PEFT、跨模态能力保持与多配置编排。 | 1 | 3 | 1 | 已有五维实现支撑 |

## 逐篇映射

| recipe | 主方向 | 次方向 | 实现状态 | 可行性 | 有效覆盖证据 | 理由 |
| --- | --- | --- | --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | `reasoning_enhancement` | `scalable_learning` | not_implemented | not_implemented | 是 | Wordle 环境交互 + GRPO 可验证奖励，属于推理增强；在线环境采样也触及效率与扩展问题。当前五维尚未实现（not_implemented）。 |
| `llm/grpo/qwen2-0-5b-numinamath` | `reasoning_enhancement` | - | five_dimension_implemented | pending_freeze | 是 | 数学 GRPO 使用 format/accuracy reward 与 math_verify，是文本可验证奖励推理增强样板；五维已实现但无 GPU 证据。 |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | `scalable_learning` | `reasoning_enhancement` | not_implemented | not_implemented | 是 | vLLM colocate 关注 GRPO 训练中的推理引擎共置与资源效率，主方向是可扩展学习，辅方向是推理增强；当前未实现。 |
| `llm/grpo/qwen2-5-3b-advanced-reward` | `reasoning_enhancement` | `preference_calibration` | five_dimension_implemented | pending_freeze | 是 | 高级奖励设计服务多步推理，奖励塑形同时带有偏好校准色彩；wave5b 已完成 gsm8k-reward-v1 五维离线实现（5 drop-one × seed + 整模型离线复载），GPU verified 0。 |
| `llm/sft/llama3-8b-xlam-function-calling` | `instruction_following` | `domain_adaptation` | five_dimension_implemented | pending_freeze | 是 | xLAM function-calling SFT 按严格 JSON 调用协议遵循指令，同时是工具调用垂直领域适配；五维已实现但无 GPU 证据。 |
| `llm/sft/rapidfire-multiconfig` | `instruction_following` | `scalable_learning` | five_dimension_implemented | pending_freeze | 是 | RapidFire 客服问答 SFT 是指令遵循样板；4 配置网格与顺序成本对照触及可扩展学习，不冒充并发加速。 |
| `llm/sft/starcoder1b-fim` | `domain_adaptation` | `instruction_following` | five_dimension_implemented | pending_freeze | 是 | StarCoder FIM 是代码填充垂直领域适配；三段式 FIM 模板仍属指令/输入格式遵循。五维已实现但无 GPU 证据。 |
| `vlm/dpo/smolvlm-rlaif-v` | `preference_calibration` | - | five_dimension_implemented | pending_freeze | 是 | RLAIF-V 直接对应多模态 AI 反馈偏好校准；五维已实现但无 GPU 证据，且与 MPO 篇非 shared backbone，不能归因算法差异。 |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | `reasoning_enhancement` | - | not_implemented | not_implemented | 是 | Qwen2.5-VL 视觉 GRPO 对应多步视觉/空间推理增强；当前未实现。 |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | `preference_calibration` | - | five_dimension_implemented | pending_freeze | 是 | MPO 基于 DPO 多损失组合，属于偏好校准；五维已实现但无 GPU 证据，一手 MPO 论文未核实，且与 DPO 篇非 shared backbone。 |
| `vlm/sft/granite-vision-line-comparison` | `instruction_following` | `reasoning_enhancement` | five_dimension_implemented | pending_freeze | 是 | Granite 线段比较 SFT 遵循固定问答指令，同时要求简单视觉关系判断；指标只衡量文本一致性，不解释几何语义。 |
| `vlm/sft/paligemma2-refcoco-grounding` | `instruction_following` | `domain_adaptation` | five_dimension_implemented | pending_freeze | 是 | PaliGemma2 grounding 按 Detect caption 指令输出定位 token，属于视觉定位任务适配；四指标不是 mAP/AP。 |
| `vlm/sft/qwen2-vl-7b-chartqa` | `instruction_following` | `reasoning_enhancement` | not_implemented | blocked_zero_train | 否 | ChartQA 视觉问答既是指令遵循也涉及图表推理；上游零训练基线已达标，decision.gate=blocked，不能当有效覆盖证据。 |
| `vlm/sft/smolvlm-chartqa` | `instruction_following` | - | not_implemented | contaminated_pretraining | 否 | SmolVLM ChartQA 为图表问答指令遵循；基座在 ChartQA 上训练过，verdict=contaminated_pretraining，不能当泛化覆盖证据。 |

## 边界与缺口

- 五维实现只表示真实任务代码 + 离线测试；GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14 均未变。
- `blocked_zero_train` 与 `contaminated_pretraining` 篇显式标为无效覆盖证据，不能用数量掩盖原理性阻塞。
- 方向层不替代 capabilities / baseline / feasibility / methodology 任一层，也不给论文分数或本仓 expected/tolerance 背书。
