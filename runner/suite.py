#!/usr/bin/env python3
"""独立训练 seed 套件；按 execution.task_id 精确分派，默认只读计划。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RECIPE_LAB_ROOT", str(CODE_ROOT))).resolve()
sys.path.insert(0, str(CODE_ROOT / "src"))

from recipe_lab import task_router
from recipe_lab.loader import load_recipes
from recipe_lab.math_task import DatasetValidationError, TaskExecutionError


def find_recipe(recipe_id):
    matches = [r for r in load_recipes(ROOT) if r.id == recipe_id]
    if len(matches) != 1:
        raise TaskExecutionError(f"--recipe 必须唯一匹配完整 id：{recipe_id}")
    return matches[0]


def suite_plan(r, seeds, output_root):
    """只读声明：未实现发布级管线的 recipe 只能得到显式标记不可验收的计划。

    resolve() 返回 None 表示「没有声明任何任务」，此时用通用声明计划器；
    resolve() 抛错（任务冲突 / 借用未实现的 task_id）时绝不吞掉，直接失败关闭。
    """
    if task_router.resolve(r) is None:
        from recipe_lab import suite_task
        plan = suite_task.suite_plan(r, seeds, output_root)
        plan["release_pipeline"] = "unavailable"
        plan["release_pipeline_reason"] = (
            "recipe 没有已实现的发布级任务套件；此计划只描述声明，不是 B2 证据，"
            "--full / --aggregate-only 会拒绝执行")
        return plan
    return task_router.load(r, "suite").suite_plan(r, seeds, output_root)


def run_suite(r, seeds, output_root):
    """真实执行必须精确分派；未知任务不得借用数学或 FIM 协议。"""
    return task_router.load(r, "suite").run_suite(r, seeds, output_root)


def aggregate_suite(r, seeds, output_root):
    """聚合已有输入同样必须精确分派，拒绝通用 JSON 放行。"""
    return task_router.load(r, "suite").aggregate_suite(r, seeds, output_root)


def main():
    ap = argparse.ArgumentParser(description="独立训练 seed 编排与内容聚合（按任务精确分派）")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--output-root", required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--full", action="store_true")
    group.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    try:
        r = find_recipe(args.recipe)
        if args.full:
            result = run_suite(r, args.seeds, args.output_root)
        elif args.aggregate_only:
            result = aggregate_suite(r, args.seeds, args.output_root)
        else:
            result = suite_plan(r, args.seeds, args.output_root)
    except (TaskExecutionError, DatasetValidationError, OSError, ValueError, TypeError,
            KeyError, AttributeError, ImportError, subprocess.SubprocessError) as exc:
        ap.exit(1, f"套件失败：{exc}\n不允许 resume 或覆盖旧 root；仅可显式聚合已完成输入，或新 root 重跑。\n")
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"[ok] 套件证据：{result}（仅内容一致性；验收单独记录，不自动晋级 B2）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
