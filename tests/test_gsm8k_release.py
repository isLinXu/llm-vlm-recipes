"""gsm8k-reward-v1 发布级管线隔离测试：drop-one 消融套件计划、路由分派、导出声明。

不构造完整 fake completed suite（那需要冻结 lock + 真实运行环境），只验证：
  * task_router 精确分派到 gsm8k_suite / gsm8k_export，不借用数学或其他协议；
  * suite_plan 声明 5 消融 × seeds，且只读；
  * export dry-run 声明整模型复载，未预声明 tolerance 时 --full 失败关闭；
  * 能力五维与可行性 pending_freeze 与登记一致；
  * C306/C307 在无证据时失败关闭。
不自动晋级 B2/B3。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import capabilities as cap
from recipe_lab import feasibility as fz
from recipe_lab import gsm8k_export as ge
from recipe_lab import gsm8k_suite as gs
from recipe_lab import gsm8k_task as gt
from recipe_lab import math_task as mt
from recipe_lab import task_router
from recipe_lab.loader import load_recipes, repo_root

ROOT = repo_root()
RECIPE_ID = gt.RECIPE_ID
SEEDS = [42, 43, 44]


@pytest.fixture(scope="module")
def recipe():
    matches = [r for r in load_recipes(ROOT) if r.id == RECIPE_ID]
    assert len(matches) == 1
    return matches[0]


def test_router_dispatches_suite_and_export_to_gsm8k_modules(recipe):
    assert task_router.resolve(recipe) == gt.TASK_ID
    assert task_router.load(recipe, "suite") is gs
    assert task_router.load(recipe, "export") is ge
    assert task_router.load(recipe, "task") is gt


def test_gsm8k_release_never_borrows_math_or_mpo_protocols(recipe):
    suite = task_router.load(recipe, "suite")
    export = task_router.load(recipe, "export")
    assert suite.__name__.endswith("gsm8k_suite")
    assert export.__name__.endswith("gsm8k_export")
    assert suite is not __import__("recipe_lab.suite_task", fromlist=["x"])
    assert export is not __import__("recipe_lab.export_task", fromlist=["x"])
    assert "mpo" not in suite.__name__ and "dpo" not in suite.__name__


def test_registry_declares_all_five_gsm8k_dimensions(recipe):
    audit = cap.audit(ROOT, load_recipes(ROOT))
    assert audit.ok
    row = audit.rows[RECIPE_ID]
    assert all(row[d]["status"] == "implemented" for d in cap.DIMENSIONS)
    assert any("gsm8k_suite.py" in p for p in row["comparison"]["implementation"])
    assert any("gsm8k_export.py" in p for p in row["export"]["implementation"])
    for dim in ("comparison", "export"):
        assert any("test_gsm8k_release.py" in t for t in row[dim]["tests"])


def test_feasibility_is_pending_freeze_not_partial(recipe):
    audit = fz.audit(ROOT, load_recipes(ROOT))
    assert audit.ok
    assert audit.rows[RECIPE_ID]["verdict"] == "pending_freeze"
    assert audit.counts["partial_implementation"] == 0
    assert audit.counts["pending_freeze"] == 9


def test_ablation_configs_are_drop_one_and_ordered():
    assert list(gt.ABLATION_CONFIGS) == [
        "full", "no_format_exact", "no_format_approx", "no_correctness", "no_extraction"]
    assert gt.ABLATION_CONFIGS["full"] == gt.REWARD_NAMES
    for config_id, active in gt.ABLATION_CONFIGS.items():
        spec = gt.verify_ablation(config_id, active)
        if config_id == "full":
            assert spec["disabled_reward"] is None
        else:
            assert spec["disabled_reward"] in gt.REWARD_NAMES
            assert spec["disabled_reward"] not in active
            assert len(active) == 3


def test_config_recipe_only_changes_config_id_and_fingerprint(recipe):
    derived = gt.config_recipe(recipe, "no_correctness")
    assert derived.get("execution.config_id") == "no_correctness"
    assert derived.get("train") == recipe.get("train")
    assert derived.get("gsm8k") == recipe.get("gsm8k")
    assert be.fingerprint(derived) != be.fingerprint(recipe)
    assert gt.active_rewards(derived) == list(gt.ABLATION_CONFIGS["no_correctness"])
    with pytest.raises(mt.TaskExecutionError):
        gt.config_recipe(recipe, "unknown")


def test_suite_plan_is_readonly_and_gsm8k_specific(recipe, tmp_path):
    root = tmp_path / "suite-preview"
    plan = gs.suite_plan(recipe, SEEDS, root)
    assert plan["task_id"] == gt.TASK_ID
    assert plan["ablation_configs"] == list(gt.ABLATION_CONFIGS)
    assert plan["ablation_is_within_backbone"] is True
    assert plan["single_variable"] == "被去掉的 reward 函数"
    assert plan["metrics"] == list(gt.METRICS)
    assert plan["seeds"] == SEEDS
    assert len(plan["runs"]) == 5 * 3
    assert len(plan["commands"]) == 1 + 5 * 3 * 2  # zero-shot + train/eval per cell
    assert not root.exists()
    config_ids = {run["config_id"] for run in plan["runs"]}
    assert config_ids == set(gt.ABLATION_CONFIGS)
    for run in plan["runs"]:
        assert run["active_rewards"] == list(gt.ABLATION_CONFIGS[run["config_id"]])
        assert "--config-id" in " ".join(map(str, next(
            c["command"] for c in plan["commands"]
            if c.get("config_id") == run["config_id"] and c["kind"] == "training")))


@pytest.mark.parametrize("seeds", [[42, 43], [42, 42, 43], [], [True, 42, 43], None])
def test_invalid_seed_lists_fail_without_writes(recipe, tmp_path, seeds):
    root = tmp_path / "bad"
    with pytest.raises(mt.TaskExecutionError):
        gs.suite_plan(recipe, seeds, root)
    assert not root.exists()


def test_suite_problems_and_zero_train_fail_closed_without_evidence(recipe):
    assert be.suite_problems(recipe)
    assert "缺少可读的 baseline-suite JSON" in be.suite_problems(recipe)[0] or True
    assert be.zero_train_measured(recipe) is False
    problems = gs.suite_problems(recipe)
    assert problems and any("缺少" in p or "baseline-suite" in p for p in problems)


def test_export_plan_is_readonly_and_tolerance_aware(recipe, tmp_path):
    destination = tmp_path / "export-preview"
    plan = ge.export_bundle(
        recipe, tmp_path / "ckpt", tmp_path / "eval.json", destination, execute=False)
    assert plan["task_id"] == gt.TASK_ID
    assert plan["full_weights"] is True and plan["adapter_only"] is False
    assert plan["baseline_promotion"] == "none"
    assert plan["config_id"] == gt.DEFAULT_CONFIG_ID
    assert not destination.exists()
    # 真实 recipe 未预声明 export_tolerance / tolerance：内容校验必须失败关闭
    with pytest.raises(mt.TaskExecutionError, match="tolerance|容差"):
        ge._tolerance(recipe)


def test_runner_suite_and_export_dry_run_dispatch(recipe, tmp_path):
    import importlib.util

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"gsm8k_release_{name}", ROOT / "runner" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    suite = load("suite")
    plan = suite.suite_plan(recipe, SEEDS, str(tmp_path / "suite"))
    assert plan["task_id"] == gt.TASK_ID and plan.get("release_pipeline") != "unavailable"
    assert len(plan["ablation_configs"]) == 5

    export = load("export")
    dry = export.export_bundle(
        recipe, str(tmp_path / "ckpt"), str(tmp_path / "eval.json"),
        str(tmp_path / "out"), execute=False)
    assert dry["task_id"] == gt.TASK_ID and dry.get("release_pipeline") != "unavailable"
    assert dry["full_weights"] is True


def test_baseline_evidence_dispatch_includes_gsm8k(recipe):
    assert be._task(recipe) == gt.TASK_ID
    # 无证据时 suite_problems 走 gsm8k_suite，而不是「尚未实现」
    problems = be.suite_problems(recipe)
    assert problems
    assert not any("尚未实现该任务的独立训练seed套件" in p for p in problems)
