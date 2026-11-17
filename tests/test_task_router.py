"""发布级任务路由与 CLI 只读/失败关闭边界的隔离契约测试。

合成 recipe 与计划只写 tmp_path；不联网、不加载训练框架、不声称 GPU 实测。
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import task_router as tr
from recipe_lab.loader import load_recipes
from recipe_lab.math_task import TaskExecutionError
from recipe_lab.model import Recipe

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
MATH = "llm/grpo/qwen2-0-5b-numinamath"
FIM = "llm/sft/starcoder1b-fim"


def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "runner" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def isolated_recipe(tmp_path, data=None):
    """仅构造内存 Recipe；测试不写仓库目录。"""
    payload = {"id": "isolated/recipe", **(data or {})}
    return Recipe("isolated/recipe", tmp_path / "recipe.toml", payload)


def real_recipes(root):
    return {r.id: r for r in load_recipes(root)}


def cli_process(script, arguments, tmp_path):
    program = '''import importlib.abc, runpy, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trl', 'transformers', 'datasets', 'math_verify', 'peft', 'bitsandbytes'}:
            raise AssertionError('隔离测试禁止框架: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    return subprocess.run(
        [sys.executable, "-B", "-c", program, str(ROOT / "runner" / script), *arguments],
        cwd=tmp_path, env={**os.environ, "RECIPE_LAB_ROOT": str(tmp_path)},
        capture_output=True, text=True, timeout=30, check=False)


def test_resolve_maps_only_implemented_recipe_ids(root, tmp_path):
    recipes = real_recipes(root)
    assert tr.resolve(recipes[MATH]) == "math-grpo-v1"
    assert tr.resolve(recipes[FIM]) == "code-fim-v1"
    assert tr.resolve(isolated_recipe(tmp_path)) is None


def test_load_is_task_specific_and_never_falls_back_to_math(root):
    recipes = real_recipes(root)
    assert tr.load(recipes[MATH], "suite").__name__ == "recipe_lab.suite_task"
    assert tr.load(recipes[MATH], "export").__name__ == "recipe_lab.export_task"
    assert tr.load(recipes[MATH], "task").__name__ == "recipe_lab.math_task"
    assert tr.load(recipes[FIM], "suite").__name__ == "recipe_lab.fim_suite"
    assert tr.load(recipes[FIM], "export").__name__ == "recipe_lab.fim_export"
    assert tr.load(recipes[FIM], "task").__name__ == "recipe_lab.fim_task"


@pytest.mark.parametrize("kind", ["task", "suite", "export"])
def test_unknown_recipe_without_task_fails_closed(tmp_path, kind):
    r = isolated_recipe(tmp_path)
    with pytest.raises(TaskExecutionError, match="没有实现 None"):
        tr.load(r, kind)


def test_unknown_recipe_cannot_borrow_a_pipeline_by_declaring_task_id(tmp_path):
    r = isolated_recipe(tmp_path, {"execution": {"task_id": "math-grpo-v1"}})
    with pytest.raises(TaskExecutionError, match="没有该 recipe 的发布级管线实现"):
        tr.resolve(r)


def test_conflicting_or_unknown_task_id_is_rejected(root, tmp_path):
    recipes = real_recipes(root)
    conflict = Recipe(MATH, recipes[MATH].path, {**recipes[MATH].data,
                                                 "execution": {**recipes[MATH].get("execution", {}),
                                                               "task_id": "code-fim-v1"}})
    with pytest.raises(TaskExecutionError, match="任务路由冲突"):
        tr.resolve(conflict)
    with pytest.raises(TaskExecutionError, match="没有该 recipe 的发布级管线实现"):
        tr.resolve(isolated_recipe(tmp_path, {"execution": {"task_id": "invented-v1"}}))
    with pytest.raises(TaskExecutionError, match="没有实现.*不得用数学协议替代"):
        tr._module("invented-v1", "suite")


def test_from_bundle_infers_task_from_snapshot_not_cli(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    snapshot = {"id": FIM, "execution": {"task_id": "code-fim-v1"}}
    (bundle / "recipe-snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    module, r = tr.from_bundle(bundle, "export")
    assert module.__name__ == "recipe_lab.fim_export" and r.id == FIM
    (bundle / "recipe-snapshot.json").write_text('{"id": 42}\n', encoding="utf-8")
    with pytest.raises(TaskExecutionError, match="缺少有效 recipe-snapshot"):
        tr.from_bundle(bundle, "export")


def test_suite_plan_for_unknown_recipe_is_readonly_declaration_only(tmp_path):
    module = load_cli("suite")
    r = isolated_recipe(tmp_path)
    before = set(tmp_path.rglob("*"))
    plan = module.suite_plan(r, SEEDS, tmp_path / "declared-root")
    assert plan["status"] == "declared" and plan["execution_supported"] is False
    assert plan["release_pipeline"] == "unavailable" and "不是 B2 证据" in plan["release_pipeline_reason"]
    assert plan["seeds"] == SEEDS and plan["baseline_promotion"] == "none"
    assert set(tmp_path.rglob("*")) == before
    with pytest.raises(TaskExecutionError, match="没有实现 None"):
        module.run_suite(r, SEEDS, tmp_path / "full-root")
    with pytest.raises(TaskExecutionError, match="没有实现 None"):
        module.aggregate_suite(r, SEEDS, tmp_path / "aggregate-root")


def test_suite_plan_uses_fim_protocol_for_fim_recipe(root):
    module = load_cli("suite")
    r = real_recipes(root)[FIM]
    plan = module.suite_plan(r, SEEDS, root / "abs-declared-root")
    assert plan["task_id"] == "code-fim-v1" and plan["status"] == "declared"
    assert plan["training_source"]["adapter_only"] is True
    assert plan["training_source"]["base_weights_bundled"] is False
    assert "exact match" in plan["limitations"] and plan.get("release_pipeline") is None


def test_export_plan_for_unknown_recipe_is_readonly_and_execute_fails_closed(tmp_path):
    module = load_cli("export")
    r = isolated_recipe(tmp_path)
    before = set(tmp_path.rglob("*"))
    plan = module.export_bundle(r, "missing", "missing", "unused", execute=False)
    assert plan["stage"] == "declared" and plan["release_pipeline"] == "unavailable"
    assert plan["baseline_promotion"] == "none" and plan["execution_supported"] is False
    assert set(tmp_path.rglob("*")) == before
    with pytest.raises(TaskExecutionError, match="没有实现 None"):
        module.export_bundle(r, "missing", "missing", "unused", execute=True)
    with pytest.raises(TaskExecutionError, match="没有实现 None"):
        module.effective_recipe(r, tmp_path / "missing-checkpoint")


def test_export_plan_uses_fim_protocol_for_fim_recipe(root):
    module = load_cli("export")
    r = real_recipes(root)[FIM]
    plan = module.export_bundle(r, "missing", "missing", "unused", execute=False)
    assert plan["task_id"] == "code-fim-v1" and plan["stage"] == "declared"
    assert plan["base_weights_bundled"] is False and plan["data_dependency"]
    assert plan.get("release_pipeline") is None


def test_cli_dry_run_declares_unavailable_pipeline_without_frameworks(tmp_path):
    directory = tmp_path / "recipes" / "isolated"
    directory.mkdir(parents=True)
    (directory / "recipe.toml").write_text('id = "isolated/recipe"\n', encoding="utf-8")
    suite = cli_process("suite.py", ["--recipe", "isolated/recipe", "--output-root", str(tmp_path / "unused")], tmp_path)
    assert suite.returncode == 0, suite.stderr
    plan = json.loads(suite.stdout)
    assert plan["status"] == "declared" and plan["release_pipeline"] == "unavailable"
    export = cli_process("export.py", ["--recipe", "isolated/recipe", "--checkpoint", "missing",
                                       "--evaluation", "missing", "--output", "unused", "--dry-run"], tmp_path)
    assert export.returncode == 0, export.stderr
    assert json.loads(export.stdout)["release_pipeline"] == "unavailable"
    assert not (tmp_path / "unused").exists() and not list((tmp_path / "unused").rglob("*.json"))


def test_cli_full_modes_fail_closed_for_unknown_recipe(tmp_path):
    directory = tmp_path / "recipes" / "isolated"
    directory.mkdir(parents=True)
    (directory / "recipe.toml").write_text('id = "isolated/recipe"\n', encoding="utf-8")
    suite = cli_process("suite.py", ["--recipe", "isolated/recipe", "--full",
                                     "--output-root", str(tmp_path / "full-root")], tmp_path)
    assert suite.returncode == 1 and "没有实现 None 的 suite 管线" in suite.stderr
    assert not (tmp_path / "full-root").exists()
    export = cli_process("export.py", ["--recipe", "isolated/recipe", "--full", "--checkpoint", "missing",
                                       "--evaluation", "missing", "--output", str(tmp_path / "bundle")], tmp_path)
    assert export.returncode == 1 and "没有实现 None 的 export 管线" in export.stderr
    assert not (tmp_path / "bundle").exists()


def test_baseline_evidence_dispatch_never_uses_math_gold_for_fim(root):
    unknown = isolated_recipe(root / "unused")
    assert any("尚未实现" in problem for problem in be.suite_problems(unknown))
    assert any("尚未实现" in problem for problem in be.evaluation_problems(
        unknown, {"kind": "evaluation"}, root / "unused.json", role="base"))
    fim = real_recipes(root)[FIM]
    problems = be.suite_problems(fim)
    assert problems and all("尚未实现" not in problem for problem in problems)
    assert any("FIM" in problem or "baseline-suite" in problem for problem in problems)
