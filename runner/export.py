#!/usr/bin/env python3
"""导出与离线复载入入口；按包内 recipe 快照精确分派任务，--worker 仅执行受控新进程。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RECIPE_LAB_ROOT", str(CODE_ROOT))).resolve()
sys.path.insert(0, str(CODE_ROOT / "src"))

from recipe_lab import rapidfire_task as rapidfire_adapter
from recipe_lab import task_router
from recipe_lab.loader import load_recipes
from recipe_lab.math_task import DatasetValidationError, TaskExecutionError


def find_recipe(recipe_id):
    matches = [r for r in load_recipes(ROOT) if r.id == recipe_id]
    if len(matches) != 1:
        raise TaskExecutionError(f"--recipe 必须唯一匹配完整 id：{recipe_id}")
    return matches[0]


def effective_recipe(r, checkpoint):
    return task_router.load(r, "export").effective_recipe(r, checkpoint)


def export_bundle(r, checkpoint, evaluation, output, execute=True):
    """只读声明允许无任务实现，但必须显式标注不可验收；真实导出必须精确分派。"""
    if not execute and task_router.resolve(r) is None:
        from recipe_lab import export_task
        plan = export_task.export_bundle(r, checkpoint, evaluation, output, execute=False)
        plan["release_pipeline"] = "unavailable"
        plan["release_pipeline_reason"] = (
            "recipe 没有已实现的发布级导出/复载协议；此计划只描述声明，不是 C405 证据，"
            "--full 会拒绝执行")
        return plan
    return task_router.load(r, "export").export_bundle(r, checkpoint, evaluation, output, execute)


def run_worker(bundle):
    # worker 不信命令行自称的任务名，只从包内 recipe-snapshot.json 推断。
    exporter, _ = task_router.from_bundle(Path(bundle), "export")
    return exporter.run_worker(Path(bundle))


def main():
    ap = argparse.ArgumentParser(description="导出与离线真实复载（按任务分派，不自动晋级）")
    ap.add_argument("--recipe")
    ap.add_argument("--checkpoint")
    ap.add_argument("--evaluation")
    ap.add_argument("--output")
    ap.add_argument("--config-id", help="RapidFire 选中配置；--full 仍会按 checkpoint 内容二次校验")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--full", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--bundle", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        if not args.bundle or any((args.recipe, args.checkpoint, args.evaluation, args.output, args.dry_run, args.full)):
            ap.error("内部 worker 仅接受 --worker --bundle ABS")
    elif args.bundle or not all((args.recipe, args.checkpoint, args.evaluation, args.output)):
        ap.error("需要 --recipe id --checkpoint --evaluation --output；--bundle 仅限内部 worker")
    try:
        if args.worker:
            result = run_worker(Path(args.bundle))
        else:
            r = find_recipe(args.recipe)
            if args.config_id:
                if r.id != rapidfire_adapter.RECIPE_ID:
                    raise TaskExecutionError("--config-id 只适用于 RapidFire 多配置 recipe")
                r = rapidfire_adapter.config_recipe(r, args.config_id)
            # dry-run 仅读声明，不检查 checkpoint，也不读取其 seed / config_id。
            if args.full:
                r = effective_recipe(r, args.checkpoint)
            result = export_bundle(r, args.checkpoint, args.evaluation, args.output, execute=args.full)
    except (TaskExecutionError, DatasetValidationError, OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        ap.exit(1, f"导出 / 离线复载失败：{exc}\n未生成 completed 导出证据；--dry-run 无需训练框架。\n")
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"[ok] 证据：{result}（许可 pending，不自动晋级）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
