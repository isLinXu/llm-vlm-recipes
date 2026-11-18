#!/usr/bin/env python3
"""仅用 base 的 zero-shot；与 checkpoint 评估共享 prompt / test / seed。"""
from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RECIPE_LAB_ROOT", str(CODE_ROOT))).resolve()
sys.path.insert(0, str(CODE_ROOT / "src"))

from recipe_lab.loader import load_recipes
from recipe_lab import dpo_task as dpo_adapter
from recipe_lab import fim_task as fim_adapter
from recipe_lab import granite_task as granite_adapter
from recipe_lab import gsm8k_task as gsm8k_adapter
from recipe_lab import mpo_task as mpo_adapter
from recipe_lab import paligemma_task as paligemma_adapter
from recipe_lab import rapidfire_task as rapidfire_adapter
from recipe_lab import xlam_task as xlam_adapter
from recipe_lab.plan import build_plan
from recipe_lab.provenance import effective_recipe, output_plan
from recipe_lab.math_task import (
    DatasetValidationError, TaskExecutionError, require_supported, run_evaluation, task_plan,
)

TASK_ADAPTERS = {
    dpo_adapter.RECIPE_ID: dpo_adapter,
    fim_adapter.RECIPE_ID: fim_adapter,
    granite_adapter.RECIPE_ID: granite_adapter,
    gsm8k_adapter.RECIPE_ID: gsm8k_adapter,
    mpo_adapter.RECIPE_ID: mpo_adapter,
    paligemma_adapter.RECIPE_ID: paligemma_adapter,
    xlam_adapter.RECIPE_ID: xlam_adapter,
    rapidfire_adapter.RECIPE_ID: rapidfire_adapter,
}


def find_recipe(recipe_id: str):
    matches = [r for r in load_recipes(ROOT) if r.id.startswith(recipe_id)]
    if not matches:
        raise SystemExit(f"未找到 recipe: {recipe_id}")
    if len(matches) > 1:
        raise SystemExit(f"--recipe {recipe_id!r} 匹配 {len(matches)} 篇，请用完整 id")
    return matches[0]


def zero_shot_plan(r, task_id=None, output_root=None, eval_split="test") -> dict:
    adapter = TASK_ADAPTERS.get(r.id)
    planner = adapter.task_plan if adapter else task_plan
    if eval_split not in ("test", "validation"):
        raise TaskExecutionError("--eval-split 只允许 test / validation")
    if eval_split != "test" and r.id != rapidfire_adapter.RECIPE_ID:
        raise TaskExecutionError("--eval-split 只适用于 RapidFire 发布级套件")
    raw = build_plan(r, mode="zero-shot")
    if r.id == rapidfire_adapter.RECIPE_ID:
        raw["evaluation_split"] = eval_split
        raw["eval_stage"] = rapidfire_adapter.EVAL_STAGES[eval_split]
    plan = output_plan(planner(r, raw, "zero-shot", task_id), ROOT, output_root)
    plan["evidence"] = str(Path(plan["run"]["output_dir"]) / "evaluation.json")
    plan["model_role"] = "base"
    plan["protocol"] = plan.get("evaluation", {}).get("protocol", "single_checkpoint_evaluation")
    plan["training_seed_variance_measured"] = False
    plan["baseline_promotion"] = "none"
    return plan


def run_real(r, plan: dict) -> Path:
    adapter = TASK_ADAPTERS.get(r.id)
    if adapter:
        if r.id == rapidfire_adapter.RECIPE_ID:
            return rapidfire_adapter.run_evaluation(
                r, plan, zero_shot=True, split_name=plan["evaluation_split"])
        return adapter.run_evaluation(r, plan, zero_shot=True)
    require_supported(r, plan)
    return run_evaluation(r, plan, zero_shot=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="recipe-lab base zero-shot 评估")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--task-id")
    ap.add_argument("--config-id", help="RapidFire 4 配置或 MPO 三损失消融之一；只覆盖 [train]/[peft]/execution.config_id")
    ap.add_argument("--eval-split", choices=("test", "validation"), default="test",
                    help="RapidFire 发布级：validation 只用于选配置，test 只用于最终验收")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--full", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--train-seed", type=int)
    ap.add_argument("--output-root", help="绝对实验输出根")
    args = ap.parse_args()
    try:
        r = effective_recipe(find_recipe(args.recipe), args.train_seed)
        if args.config_id:
            if r.id == rapidfire_adapter.RECIPE_ID:
                r = rapidfire_adapter.config_recipe(r, args.config_id)
            elif r.id == mpo_adapter.RECIPE_ID:
                r = mpo_adapter.config_recipe(r, args.config_id)
            else:
                raise TaskExecutionError("--config-id 只适用于 RapidFire / MPO 多配置 recipe")
        plan = zero_shot_plan(r, args.task_id, args.output_root, args.eval_split)
    except (ValueError, TaskExecutionError) as exc:
        ap.exit(1, f"参数错误：{exc}\n")
    if not args.full:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    try:
        evidence = run_real(r, plan)
    except (TaskExecutionError, DatasetValidationError, ImportError) as exc:
        ap.exit(1, f"执行失败：{exc}\n请核对 recipe lock 与任务配置；--dry-run 不需要框架。\n")
    print(f"[ok] base zero-shot 完成：{evidence}（不自动更新 baseline 或决策门）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
