PYTHON ?= python3
export PYTHONPATH := $(CURDIR)/src

.PHONY: help doctor list validate validate-l5 validate-l6 validate-strict \
        repro repro-check baseline baseline-check baseline-dry-run \
        capabilities-check report matrix test suite-dry-run fim-dry-run xlam-dry-run \
        fim-suite-dry-run fim-export-dry-run xlam-suite-dry-run xlam-export-dry-run \
        rapidfire-dry-run rapidfire-suite-dry-run rapidfire-export-dry-run \
        granite-dry-run granite-suite-dry-run granite-export-dry-run \
        paligemma-dry-run paligemma-suite-dry-run paligemma-export-dry-run \
        feasibility-check methodology-check methodology-report ci

help:
	@echo "recipe-lab 常用命令"
	@echo "  make doctor            环境自检（Python / TOML / 依赖）"
	@echo "  make list              列出所有 recipe"
	@echo "  make validate          L0-L3 全量校验"
	@echo "  make validate-l5       L0-L5 全量校验（含可复现性审计规则 R501-R507）"
	@echo "  make validate-l6       L0-L6 全量校验（再加基线就绪规则 R601-R606）"
	@echo "  make validate-strict   严格模式：warning 也算失败"
	@echo "  make repro             可复现性审计摘要 + 完整性检查"
	@echo "  make repro-check       只跑可复现性完整性检查（CI 用）"
	@echo "  make baseline          基线就绪摘要 + 完整性检查"
	@echo "  make baseline-check    只跑基线状态完整性检查（CI 用）"
	@echo "  make feasibility-check 逐篇判定原理上能否作 baseline（14篇覆盖/一手事实一致）"
	@echo "  make methodology-check 方法论依据层审计（论文一手字段/引用闭合/边界声明）"
	@echo "  make methodology-report 重建 docs/METHODOLOGY.md（practice papers 一手核查）"
	@echo "  make baseline-dry-run  打印 14 篇训练计划（零依赖、无需 GPU）"
	@echo "  make report            重建四份生成物（MATRIX / REPRODUCIBILITY / BASELINE-READINESS / METHODOLOGY）"
	@echo "  make matrix            只重建 docs/MATRIX.md"
	@echo "  make test              运行单元测试"
	@echo "  make suite-dry-run     只读三seed套件预演，不运行训练"
	@echo "  make fim-dry-run       FIM 三个基础入口只读预演"
	@echo "  make fim-suite-dry-run FIM 三seed发布级套件只读预演"
	@echo "  make fim-export-dry-run FIM adapter-only 导出只读预演"
	@echo "  make xlam-dry-run      xLAM 三个基础入口只读预演"
	@echo "  make xlam-suite-dry-run  xLAM 三seed发布级套件只读预演"
	@echo "  make xlam-export-dry-run xLAM adapter-only 导出只读预演"
	@echo "  make rapidfire-dry-run  RapidFire 4 配置基础入口只读预演（含切换配置）"
	@echo "  make rapidfire-suite-dry-run  RapidFire 4 配置 × 3 seed 发布级套件只读预演"
	@echo "  make rapidfire-export-dry-run RapidFire 选中配置 adapter-only 导出只读预演"
	@echo "  make granite-dry-run   Granite 三个基础入口只读预演（图像内容 hash 隔离协议）"
	@echo "  make granite-suite-dry-run  Granite 三 seed 发布级套件只读预演"
	@echo "  make granite-export-dry-run Granite DoRA/4bit adapter-only 导出只读预演"
	@echo "  make paligemma-dry-run PaliGemma 三个基础入口只读预演（单框 loc/IoU 协议）"
	@echo "  make paligemma-suite-dry-run  PaliGemma 三 seed 发布级套件只读预演"
	@echo "  make paligemma-export-dry-run PaliGemma 普通 LoRA adapter-only 导出只读预演"
	@echo "  make ci                CI 门禁：doctor + L6 validate + test + 三个 --check + capabilities-check"

doctor:
	$(PYTHON) -m recipe_lab doctor

list:
	$(PYTHON) -m recipe_lab list

validate:
	$(PYTHON) -m recipe_lab validate

validate-l5:
	$(PYTHON) -m recipe_lab validate --level l5

validate-l6:
	$(PYTHON) -m recipe_lab validate --level l6

validate-strict:
	$(PYTHON) -m recipe_lab validate --level l6 --strict

matrix:
	$(PYTHON) -m recipe_lab matrix --out docs/MATRIX.md

repro:
	$(PYTHON) -m recipe_lab repro
	$(PYTHON) -m recipe_lab repro --check

repro-check:
	$(PYTHON) -m recipe_lab repro --check

baseline:
	$(PYTHON) -m recipe_lab baseline
	$(PYTHON) -m recipe_lab baseline --check

baseline-check:
	$(PYTHON) -m recipe_lab baseline --check

capabilities-check:
	$(PYTHON) -m recipe_lab capabilities --check

# 与 capabilities / baseline 分开：本检查判定「实验目标是否成立」，
# 零训练基线已达标、基座预训练污染两类即使补齐代码与 GPU 证据也不晋级。
feasibility-check:
	$(PYTHON) -m recipe_lab feasibility --check

# 方法论依据层：设计决策依据哪篇论文的一手结论。--check 校验登记闭合与
# 一手字段完整性，不因有 3 篇 unverified 而失败（那是审计结论，不是错误）；
# 一手论文数字绝不写成本仓 expected/tolerance，也不解锁 GPU 晋级。
methodology-check:
	$(PYTHON) -m recipe_lab methodology --check

methodology-report:
	$(PYTHON) -m recipe_lab methodology --report docs/METHODOLOGY.md

baseline-dry-run:
	$(PYTHON) -m recipe_lab baseline --dry-run

# 只读套件预演；不创建runs或实跑证据，不需要训练框架。
suite-dry-run:
	$(PYTHON) runner/suite.py --recipe llm/grpo/qwen2-0-5b-numinamath --seeds 42 43 44 --output-root "$(CURDIR)/runs/math-suite-preview" --dry-run

# FIM的三个基础入口同样只读；不把此预演称为实跑。
fim-dry-run:
	$(PYTHON) runner/train.py --recipe llm/sft/starcoder1b-fim --dry-run --json
	$(PYTHON) runner/eval.py --recipe llm/sft/starcoder1b-fim --dry-run
	$(PYTHON) runner/zero_shot.py --recipe llm/sft/starcoder1b-fim --dry-run

# 发布级管线同样只读预演：按 task_router 精确分派到 fim_suite / fim_export，
# 不创建输出目录、不产 completed 证据，也不把声明计划当作 B2/C405 验收。
fim-suite-dry-run:
	$(PYTHON) runner/suite.py --recipe llm/sft/starcoder1b-fim --seeds 42 43 44 --output-root "$(CURDIR)/runs/fim-suite-preview" --dry-run

fim-export-dry-run:
	$(PYTHON) runner/export.py --recipe llm/sft/starcoder1b-fim --checkpoint "$(CURDIR)/runs/absent/fim-full" --evaluation "$(CURDIR)/runs/absent/fim-eval/evaluation.json" --output "$(CURDIR)/runs/fim-export-preview" --dry-run

# 第三任务三个基础入口；只读声明，不产 completed 证据。
xlam-dry-run:
	$(PYTHON) runner/train.py --recipe llm/sft/llama3-8b-xlam-function-calling --dry-run --json
	$(PYTHON) runner/eval.py --recipe llm/sft/llama3-8b-xlam-function-calling --dry-run
	$(PYTHON) runner/zero_shot.py --recipe llm/sft/llama3-8b-xlam-function-calling --dry-run

# xLAM 发布级管线只读预演：按 task_router 精确分派到 xlam_suite / xlam_export。
# 真实 recipe 未预声明 export_tolerance，--full 导出会失败关闭；这里绝不当 B2/C405 证据。
xlam-suite-dry-run:
	$(PYTHON) runner/suite.py --recipe llm/sft/llama3-8b-xlam-function-calling --seeds 42 43 44 --output-root "$(CURDIR)/runs/xlam-suite-preview" --dry-run

xlam-export-dry-run:
	$(PYTHON) runner/export.py --recipe llm/sft/llama3-8b-xlam-function-calling --checkpoint "$(CURDIR)/runs/absent/xlam-full" --evaluation "$(CURDIR)/runs/absent/xlam-eval/evaluation.json" --output "$(CURDIR)/runs/xlam-export-preview" --dry-run

# 第四任务基础入口只读预演：4 配置全部检查，切换后命令与参数都随配置变化；
# 同时预演 validation/test 两种评估 split。这里不创建 runs、不产 completed 证据。
rapidfire-dry-run:
	$(PYTHON) runner/train.py --recipe llm/sft/rapidfire-multiconfig --dry-run --json
	$(PYTHON) runner/train.py --recipe llm/sft/rapidfire-multiconfig --config-id lr5e-4-r32 --dry-run --json
	$(PYTHON) runner/train.py --recipe llm/sft/rapidfire-multiconfig --config-id lr2e-4-r8 --dry-run --json
	$(PYTHON) runner/train.py --recipe llm/sft/rapidfire-multiconfig --config-id lr2e-4-r32 --dry-run --json
	$(PYTHON) runner/eval.py --recipe llm/sft/rapidfire-multiconfig --dry-run
	$(PYTHON) runner/eval.py --recipe llm/sft/rapidfire-multiconfig --eval-split validation --dry-run
	$(PYTHON) runner/zero_shot.py --recipe llm/sft/rapidfire-multiconfig --dry-run
	$(PYTHON) runner/zero_shot.py --recipe llm/sft/rapidfire-multiconfig --eval-split validation --dry-run

# RapidFire 发布级只读预演：按 task_router 精确分派到 rapidfire_suite / rapidfire_export。
# 真实 recipe 未预声明 export_tolerance，--full 导出会失败关闭；这里绝不当 B2/C405 证据。
rapidfire-suite-dry-run:
	$(PYTHON) runner/suite.py --recipe llm/sft/rapidfire-multiconfig --seeds 42 43 44 --output-root "$(CURDIR)/runs/rapidfire-suite-preview" --dry-run

rapidfire-export-dry-run:
	$(PYTHON) runner/export.py --recipe llm/sft/rapidfire-multiconfig --checkpoint "$(CURDIR)/runs/absent/rapidfire-full" --evaluation "$(CURDIR)/runs/absent/rapidfire-eval/evaluation.json" --output "$(CURDIR)/runs/rapidfire-export-preview" --dry-run

# wave3 首篇 VLM 的三个基础入口：检查 Granite 自建协议声明（图像内容 hash 隔离、
# 显式 image_token_id + assistant-only 双层 mask、冻结 processor identity）。
# 只读预演，不产 completed 证据、不当 B2/C405 验收。
granite-dry-run:
	$(PYTHON) runner/train.py --recipe vlm/sft/granite-vision-line-comparison --dry-run --json
	$(PYTHON) runner/eval.py --recipe vlm/sft/granite-vision-line-comparison --dry-run
	$(PYTHON) runner/zero_shot.py --recipe vlm/sft/granite-vision-line-comparison --dry-run

# ADR-022 后发布级已按 task_router 精确分派到 granite_suite / granite_export；
# dry-run 输出 Granite 专用协议而非 release_pipeline=unavailable，--full 仍失败关闭。
# 真实 recipe 未预声明 evaluation.export_tolerance，导出 --full 不得回填容差。
granite-suite-dry-run:
	$(PYTHON) runner/suite.py --recipe vlm/sft/granite-vision-line-comparison --seeds 42 43 44 --output-root "$(CURDIR)/runs/granite-suite-preview" --dry-run

granite-export-dry-run:
	$(PYTHON) runner/export.py --recipe vlm/sft/granite-vision-line-comparison --checkpoint "$(CURDIR)/runs/absent/granite-full" --evaluation "$(CURDIR)/runs/absent/granite-eval/evaluation.json" --output "$(CURDIR)/runs/granite-export-preview" --dry-run

# wave3 第二篇 VLM 的三个基础入口：检查 PaliGemma 自建单框 bbox / <loc####>
# 协议声明（COCO xywh→xyxy、1024 桶 round-trip IoU、统一 prompt、
# completion-only 双层 mask）。只读预演，不产 completed 证据、不当 B2/C405 验收。
paligemma-dry-run:
	$(PYTHON) runner/train.py --recipe vlm/sft/paligemma2-refcoco-grounding --dry-run --json
	$(PYTHON) runner/eval.py --recipe vlm/sft/paligemma2-refcoco-grounding --dry-run
	$(PYTHON) runner/zero_shot.py --recipe vlm/sft/paligemma2-refcoco-grounding --dry-run

# ADR-023 后发布级已按 task_router 精确分派到 paligemma_suite / paligemma_export；
# dry-run 输出 PaliGemma 专用协议而非 release_pipeline=unavailable，--full 仍失败关闭。
# 真实 recipe 未预声明 evaluation.export_tolerance，导出 --full 不得回填容差。
paligemma-suite-dry-run:
	$(PYTHON) runner/suite.py --recipe vlm/sft/paligemma2-refcoco-grounding --seeds 42 43 44 --output-root "$(CURDIR)/runs/paligemma-suite-preview" --dry-run

paligemma-export-dry-run:
	$(PYTHON) runner/export.py --recipe vlm/sft/paligemma2-refcoco-grounding --checkpoint "$(CURDIR)/runs/absent/paligemma-full" --evaluation "$(CURDIR)/runs/absent/paligemma-eval/evaluation.json" --output "$(CURDIR)/runs/paligemma-export-preview" --dry-run

report: matrix
	$(PYTHON) -m recipe_lab repro --report
	$(PYTHON) -m recipe_lab baseline --report
	$(PYTHON) -m recipe_lab methodology --report docs/METHODOLOGY.md

test:
	$(PYTHON) -m pytest -q

# CI 门禁是「0 error」。warning 是已登记的待办风险，会随维护逐步消解，
# 不应让仓库因为诚实记录了风险而永远红灯。发布前单独跑 make validate-strict。
# 两个 --check 校验的都是审计完整性（等级/状态无虚报、证据可追溯），
# 不因为「0/14 可复现」「0/14 到 B2」而失败（ADR-008 / ADR-010）。
ci: doctor validate-l6 test repro-check baseline-check capabilities-check feasibility-check methodology-check suite-dry-run fim-dry-run xlam-dry-run rapidfire-dry-run granite-dry-run paligemma-dry-run fim-suite-dry-run fim-export-dry-run xlam-suite-dry-run xlam-export-dry-run rapidfire-suite-dry-run rapidfire-export-dry-run granite-suite-dry-run granite-export-dry-run paligemma-suite-dry-run paligemma-export-dry-run
	@echo "CI 通过（0 error）。提升 status / repro.level / baseline_contract.state 前请单独运行 make validate-strict 复核 warning。"
