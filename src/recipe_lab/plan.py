"""把一份 recipe.toml 解析成确定的训练计划。

这是 runner 的零依赖内核：不 import trl / transformers / torch，
因此本机可以做 `--dry-run`，完整验证「配置 → 命令 → 参数 → seed → 输出路径」
这条链路，而不需要 GPU 或训练框架。真正的框架 import 全部延迟到 runner/train.py。

单一事实源仍是 recipe.toml；这里只做解释，不新增配置。
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from .model import Recipe

MODES = ("dry-run", "smoke", "full", "zero-shot", "eval")


def build_plan(r: Recipe, mode: str = "dry-run") -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"未知 mode {mode!r}，允许 {MODES}")

    budget = r.get("budget", {}) or {}
    steps = (
        int(budget.get("smoke_steps", 0) or 0) if mode in ("smoke", "dry-run")
        else int(budget.get("full_steps", 0) or 0)
    )
    train = dict(r.get("train", {}) or {})
    contract = dict(r.get("baseline_contract", {}) or {})

    plan: dict[str, Any] = {
        "recipe_id": r.id,
        "mode": mode,
        "experiment_id": contract.get("experiment_id", ""),
        "command": r.get("execution.command", ""),
        "entrypoint": r.get("execution.entrypoint", ""),
        "lockfile": r.get("execution.lockfile", ""),
        "eval_script": r.get("execution.eval_script", ""),
        "evidence": r.get("execution.evidence", ""),
        "method": {
            "family": r.family,
            "trainer": r.get("method.trainer", ""),
            "dataset_type": r.get("dataset.type", ""),
            "structure": r.get("dataset.structure", ""),
        },
        "model": {
            "base": r.get("model.base", ""),
            "revision": r.get("model.revision", "") or "(未锁定)",
        },
        "dataset": {
            "name": r.get("dataset.name", ""),
            "revision": r.get("dataset.revision", "") or "(未锁定)",
            "split": r.get("evaluation.split", ""),
        },
        "determinism": {
            "seed": train.get("seed"),
            "data_seed": train.get("data_seed"),
            "policy": contract.get("determinism", ""),
            "runtime": contract.get("python_cuda", ""),
        },
        "run": {
            "steps": steps,
            "smoke_steps": budget.get("smoke_steps"),
            "full_steps": budget.get("full_steps"),
            "params": {k: v for k, v in train.items() if k not in ("seed", "data_seed")},
            "output_dir": f"runs/{r.id.replace('/', '__')}/{contract.get('experiment_id') or 'exp'}",
        },
        "evaluation": {
            "metrics": r.get("evaluation.metrics", []) or [],
            "held_out": r.get("evaluation.held_out"),
            "multi_run": r.get("evaluation.multi_run"),
            "seed": r.get("evaluation.seed"),
            "expected": r.get("evaluation.expected", "") or "(待实跑)",
            "tolerance": r.get("evaluation.tolerance", "") or "(待实跑)",
        },
        "baseline": {
            "zero_train": r.get("baseline.zero_train"),
            "result": r.get("baseline.result", ""),
            "gate": r.get("decision.gate", ""),
        },
    }

    # 显示当前模式的真实入口，避免dry-run报告里给出误导性的--full命令。
    command = shlex.split(plan["command"])
    if command[:2] == ["python3", "runner/train.py"]:
        command = [part for part in command if part not in ("--dry-run", "--smoke", "--full")]
        if mode in ("eval", "zero-shot"):
            command[1] = "runner/eval.py" if mode == "eval" else "runner/zero_shot.py"
        command.append("--full" if mode in ("full", "eval", "zero-shot") else "--" + mode)
        plan["command"] = shlex.join(command)

    if r.is_vlm:
        plan["modality_contract"] = dict(r.get("modality_contract", {}) or {})
    if r.get("rewards.functions"):
        plan["rewards"] = {
            "functions": r.get("rewards.functions"),
            "failure_path_tested": r.get("rewards.failure_path_tested"),
        }

    plan["task_id"] = r.get("execution.task_id")
    plan["warnings"] = _warnings(r, mode)
    if r.id == "llm/grpo/qwen2-0-5b-numinamath":
        from .math_task import task_plan
        return task_plan(r, plan, mode)
    if r.id == "llm/sft/starcoder1b-fim":
        from .fim_task import task_plan
        return task_plan(r, plan, mode)
    if r.id == "llm/sft/llama3-8b-xlam-function-calling":
        from .xlam_task import task_plan
        return task_plan(r, plan, mode)
    if r.id == "llm/sft/rapidfire-multiconfig":
        from .rapidfire_task import task_plan
        return task_plan(r, plan, mode)
    if r.id == "vlm/sft/granite-vision-line-comparison":
        # wave3 首篇 VLM：分派 granite-linecomparison-v1 的 data/train/eval；
        # 发布级 comparison/export 由 task_router 精确分派到 granite_suite / granite_export。
        from .granite_task import task_plan
        return task_plan(r, plan, mode)
    if r.id == "vlm/sft/paligemma2-refcoco-grounding":
        # wave3 第二篇 VLM：单框 bbox / <loc####> 协议；发布级由 task_router
        # 精确分派到 paligemma_suite / paligemma_export。
        from .paligemma_task import task_plan
        return task_plan(r, plan, mode)
    if r.id == "vlm/dpo/smolvlm-rlaif-v":
        # wave4 DPO：SmolVLM 偏好对图像 hash 隔离 + beta/ref 冻结；发布级由
        # task_router 精确分派到 dpo_suite / dpo_export。
        from .dpo_task import task_plan
        return task_plan(r, plan, mode)
    if r.id == "vlm/mpo/qwen2-5-vl-3b-mpo":
        # wave4 MPO：Qwen2.5-VL 三损失组合 + 消融；发布级由 task_router 精确
        # 分派到 mpo_suite / mpo_export。
        from .mpo_task import task_plan
        return task_plan(r, plan, mode)
    plan["execution_supported"] = False
    plan["unsupported_reason"] = "该任务的数据/训练/评估适配尚未实现；dry-run 仅输出声明"
    return plan


def _warnings(r: Recipe, mode: str) -> list[str]:
    out: list[str] = []
    if not r.get("model.revision"):
        out.append("model.revision 未锁定：在有网环境跑 runner/lock.py --apply")
    if not r.get("dataset.revision"):
        out.append("dataset.revision 未锁定：同上")
    if not r.get("source_sha"):
        out.append("source_sha 为空：上游 notebook 漂移无法定位")
    if r.get("baseline.result") == "unmeasured":
        out.append("零训练基线未实测：先跑 runner/zero_shot.py（R302 决策门）")
    if not r.get("evaluation.expected"):
        out.append("evaluation.expected 未声明：首次运行可用于测量；独立验收目标须另行预先冻结，不用本轮均值自证通过")
    if mode in ("smoke", "full"):
        out.append("本模式会 import trl / transformers / torch，需要按 lock 装好环境与 GPU")
    return out


def render_plan(plan: dict, compact: bool = False) -> str:
    if compact:
        return (
            f"$ {plan['command'] or '(未配置 execution.command)'}\n"
            f"recipe     : {plan['recipe_id']}  [{plan['mode']}]\n"
            f"experiment : {plan['experiment_id'] or '(未命名)'}\n"
            f"trainer    : {plan['method']['trainer']} / {plan['method']['dataset_type']}\n"
            f"model      : {plan['model']['base']} @ {plan['model']['revision']}\n"
            f"dataset    : {plan['dataset']['name']} @ {plan['dataset']['revision']}\n"
            f"seed       : {plan['determinism']['seed']} / data_seed "
            f"{plan['determinism']['data_seed']}\n"
            f"steps      : {plan['run']['steps']}  ->  {plan['run']['output_dir']}\n"
            f"eval       : {', '.join(plan['evaluation']['metrics']) or '(未声明)'} "
            f"| protocol={plan['evaluation'].get('protocol', 'declared')} "
            f"| runs={plan['evaluation'].get('runs', '(仅计划多seed)' if plan['evaluation'].get('multi_run') else '(未声明)')}\n"
            f"supported  : {plan.get('execution_supported', False)}（代码路由，不代表实跑通过）\n"
            f"warnings   : {len(plan['warnings'])} 条（见 --json）"
        )
    return json.dumps(plan, ensure_ascii=False, indent=2)
