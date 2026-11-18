# Recipe 矩阵

> 由 `recipe-lab matrix` 自动生成，请勿手工编辑。单一事实源是各 `recipes/**/recipe.toml`。

当前登记 **14** 篇 · 验证层级 L0–L3 · **0** error / **12** warning / **24** info。

状态机：`cataloged` → `ported` → `smoke-passed` → `benchmarked`。

## 全量矩阵

| id | family | trainer | modality | model | dataset | status | coverage | repro | ready | zero_shot | trl | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `agent/grpo/wordle-openenv-hf-jobs` | grpo | GRPOTrainer | llm | Qwen/Qwen3-0.6B | OpenEnv Wordle（在线生成，无静态数据集） | ported | params | L0 | B1 | unmeasured | - | 0 |
| `llm/grpo/qwen2-0-5b-numinamath` | grpo | GRPOTrainer | llm | Qwen/Qwen2-0.5B-Instruct | AI-MO/NuminaMath-TIR | ported | params | L1 | B1 | unmeasured | 0.14.0 | 0 |
| `llm/grpo/qwen2-0-5b-vllm-colocate` | grpo | GRPOTrainer | llm | Qwen/Qwen2-0.5B-Instruct | AI-MO/NuminaMath-TIR | ported | params | L1 | B1 | unmeasured | 0.23.1 | 0 |
| `llm/grpo/qwen2-5-3b-advanced-reward` | grpo | GRPOTrainer | llm | Qwen/Qwen2.5-3B-Instruct | openai/gsm8k | ported | params | L0 | B1 | unmeasured | 0.14.0 | 0 |
| `llm/sft/llama3-8b-xlam-function-calling` | sft | Trainer | llm | meta-llama/Meta-Llama-3-8B-Instruct | Salesforce/xlam-function-calling-60k | cataloged | catalog | L0 | B1 | unmeasured | - | 0 |
| `llm/sft/rapidfire-multiconfig` | hpo | Trainer | llm | gpt2 | bitext/Bitext-customer-support-llm-chatbot-training-dataset | cataloged | catalog | L0 | B1 | unmeasured | - | 0 |
| `llm/sft/starcoder1b-fim` | sft | Trainer | llm | bigcode/starcoderbase-1b | smangrul/hf-stack-v1 | cataloged | catalog | L0 | B1 | unmeasured | - | 0 |
| `vlm/dpo/smolvlm-rlaif-v` | dpo | DPOTrainer | vlm | HuggingFaceTB/SmolVLM-Instruct | HuggingFaceH4/rlaif-v_formatted | cataloged | catalog | L1 | B1 | unmeasured | 0.12.2 | 0 |
| `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` | grpo | GRPOTrainer | vlm | Qwen/Qwen2.5-VL-3B-Instruct | lmms-lab/multimodal-open-r1-8k-verified | ported | params | L0 | B1 | unmeasured | - | 0 |
| `vlm/mpo/qwen2-5-vl-3b-mpo` | mpo | DPOTrainer | vlm | Qwen/Qwen2.5-VL-3B-Instruct | HuggingFaceH4/rlaif-v_formatted | ported | params | L1 | B1 | unmeasured | - | 0 |
| `vlm/sft/granite-vision-line-comparison` | sft | Trainer | vlm | ibm-granite/granite-vision-3.1-2b-preview | euclid-multimodal/Geoperception | cataloged | params | L0 | B1 | unmeasured | - | 0 |
| `vlm/sft/paligemma2-refcoco-grounding` | sft | Trainer | vlm | google/paligemma2-3b-pt-448 | jxu124/refcoco | cataloged | params | L0 | B1 | unmeasured | - | 0 |
| `vlm/sft/qwen2-vl-7b-chartqa` | sft | SFTTrainer | vlm | Qwen/Qwen2-VL-7B-Instruct | HuggingFaceM4/ChartQA | cataloged | params | L1 | B1 | passes | 0.22.0.dev0 | 0 |
| `vlm/sft/smolvlm-chartqa` | sft | SFTTrainer | vlm | HuggingFaceTB/SmolVLM-Instruct | HuggingFaceM4/ChartQA | ported | params | L1 | B1 | unmeasured | 0.22.0.dev0 | 0 |

## 分布

| 维度 | 分布 |
| --- | --- |
| method.family | `dpo` 1, `grpo` 5, `hpo` 1, `mpo` 1, `sft` 6 |
| status | `cataloged` 7, `ported` 7 |

## 可复现性分布

| 等级 | 篇数 |
| --- | --- |
| `L0` | 8 |
| `L1` | 6 |

**达到 L2 及以上（可复现）：0/14 篇。** `repro` 列由 `registry/reproducibility.toml` 按八维度重算（`R503` 会拦截虚报），不是配置自洽的同义词。完整结论与补齐路径见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 基线就绪分布

| 状态 | 篇数 |
| --- | --- |
| `B1` | 14 |

**可作为 research baseline（B2 及以上）：0/14 篇。** `ready` 列是 `[baseline_contract].state`，由 `registry/baseline.toml` 的 四组检查项重算（`R603` 拦截虚报）。注意它与 `repro` 是两件事：复刻不了上游原实验，不妨碍我们锁定自己的基线；B1 只代表契约齐备且可 dry-run，**不代表已经实跑**。详见 [BASELINE-READINESS.md](BASELINE-READINESS.md)。

## 版本漂移

| trl | 篇数 | recipe |
| --- | --- | --- |
| `0.12.2` | 1 | `vlm/dpo/smolvlm-rlaif-v` |
| `0.14.0` | 2 | `llm/grpo/qwen2-0-5b-numinamath`, `llm/grpo/qwen2-5-3b-advanced-reward` |
| `0.22.0.dev0` | 2 | `vlm/sft/qwen2-vl-7b-chartqa`, `vlm/sft/smolvlm-chartqa` |
| `0.23.1` | 1 | `llm/grpo/qwen2-0-5b-vllm-colocate` |

> 不同 trl 版本之间不要直接复制训练代码。上游 cookbook 覆盖 0.12.2 / 0.14.0 / 0.22.0.dev0 / 0.23.1，API 差异显著。（曾误记的 1.7.0 已核实为编造并删除，见 evidence.toml。）

## 待办清单（warning）

warning 不代表失败，而是已登记、尚未消解的风险。

- `R202` · `llm/grpo/qwen2-0-5b-numinamath` — trl 0.14.0 + transformers 4.47.1 不在已验证组合白名单
- `R202` · `llm/grpo/qwen2-5-3b-advanced-reward` — trl 0.14.0 + transformers 4.47.1 不在已验证组合白名单
- `R203` · `vlm/sft/qwen2-vl-7b-chartqa` — `trl = 0.22.0.dev0` 是预发布/dev 版本，通常需要从源码安装
- `R203` · `vlm/sft/smolvlm-chartqa` — `trl = 0.22.0.dev0` 是预发布/dev 版本，通常需要从源码安装
- `R204` · `GLOBAL` — 仓库内存在 4 个不同 trl 版本：0.12.2 (1 篇), 0.14.0 (2 篇), 0.22.0.dev0 (2 篇), 0.23.1 (1 篇)
- `R303` · `agent/grpo/wordle-openenv-hf-jobs` — 零训练基线尚未测量
- `R303` · `llm/grpo/qwen2-0-5b-numinamath` — 零训练基线尚未测量
- `R303` · `llm/grpo/qwen2-0-5b-vllm-colocate` — 零训练基线尚未测量
- `R303` · `llm/grpo/qwen2-5-3b-advanced-reward` — 零训练基线尚未测量
- `R303` · `vlm/grpo/qwen2-5-vl-3b-visual-reasoning` — 零训练基线尚未测量
- `R303` · `vlm/mpo/qwen2-5-vl-3b-mpo` — 零训练基线尚未测量
- `R303` · `vlm/sft/smolvlm-chartqa` — 零训练基线尚未测量

---

字段规范见 [docs/DESIGN.md](DESIGN.md)，VLM 数据契约见 [docs/CONTRACT.md](CONTRACT.md)，验证分层见 [docs/VERIFICATION.md](VERIFICATION.md)，可复现性审计见 [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)，基线就绪见 [docs/BASELINE-READINESS.md](BASELINE-READINESS.md)。
