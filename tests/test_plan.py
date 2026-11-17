"""plan.py：零依赖训练计划内核。

这里验证的是「配置 → 命令 → 参数 → seed → 输出路径」这条链路的确定性，
以及它是否老实报告未锁定的输入。不验证 TRL 兼容性（那是 dry-run 明确不覆盖的）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_recipe

from recipe_lab.loader import load_recipes
from recipe_lab.plan import MODES, build_plan, render_plan
from recipe_lab.rules import Context

SAMPLE = "llm/grpo/qwen2-0-5b-numinamath"


def _sample(ctx: Context):
    return next(r for r in load_recipes(ctx.root) if r.id == SAMPLE)


def test_build_plan_is_deterministic(ctx: Context) -> None:
    r = _sample(ctx)
    assert build_plan(r, "dry-run") == build_plan(r, "dry-run")
    assert render_plan(build_plan(r, "dry-run")) == render_plan(build_plan(r, "dry-run"))


def test_plan_has_all_required_sections(ctx: Context) -> None:
    plan = build_plan(_sample(ctx), "dry-run")
    for key in (
        "recipe_id", "mode", "experiment_id", "command", "entrypoint",
        "lockfile", "method", "model", "dataset", "determinism",
        "run", "evaluation", "baseline", "warnings",
    ):
        assert key in plan, key
    assert plan["determinism"]["seed"] == 42
    assert plan["determinism"]["data_seed"] == 42
    assert plan["run"]["output_dir"].startswith("runs/")
    assert SAMPLE.replace("/", "__") in plan["run"]["output_dir"]


def test_smoke_and_full_use_different_budgets(ctx: Context) -> None:
    r = _sample(ctx)
    smoke = build_plan(r, "smoke")
    full = build_plan(r, "full")
    assert smoke["run"]["steps"] == r.get("budget.smoke_steps")
    assert full["run"]["steps"] == r.get("budget.full_steps")
    assert full["run"]["steps"] > smoke["run"]["steps"]
    assert smoke["run"]["output_dir"] != full["run"]["output_dir"], (
        "smoke / full必须隔离输出，不能让冒烟覆盖完整训练"
    )
    assert smoke["experiment_id"] == full["experiment_id"]


def test_plan_command_matches_mode(ctx: Context) -> None:
    r = _sample(ctx)
    for mode in ("dry-run", "smoke", "full"):
        assert build_plan(r, mode)["command"].endswith("--" + mode)
    assert "runner/eval.py" in build_plan(r, "eval")["command"]
    assert "runner/zero_shot.py" in build_plan(r, "zero-shot")["command"]


def test_unknown_mode_rejected(ctx: Context) -> None:
    with pytest.raises(ValueError, match="未知 mode"):
        build_plan(_sample(ctx), "magic")
    assert "dry-run" in MODES and "eval" in MODES


def test_warnings_are_honest_about_unlocked_inputs(ctx: Context) -> None:
    plan = build_plan(_sample(ctx), "dry-run")
    joined = "\n".join(plan["warnings"])
    assert "model.revision" in joined
    assert "dataset.revision" in joined
    assert "source_sha" in joined
    assert not plan["evaluation"].get("training_seed_variance_measured", False)
    assert plan["model"]["revision"] == "(未锁定)"
    assert plan["dataset"]["revision"] == "(未锁定)"
    assert "expected" not in plan["evaluation"], "单checkpoint计划不得伪造预期分数"


def test_smoke_warns_about_framework_import(ctx: Context) -> None:
    assert any("import trl" in w for w in build_plan(_sample(ctx), "smoke")["warnings"])
    assert not any("import trl" in w for w in build_plan(_sample(ctx), "dry-run")["warnings"])


def test_render_plan_json_is_parseable(ctx: Context) -> None:
    out = render_plan(build_plan(_sample(ctx), "dry-run"))
    assert json.loads(out)["recipe_id"] == SAMPLE


def test_vlm_plan_carries_modality_contract(ctx: Context) -> None:
    r = next(x for x in load_recipes(ctx.root) if x.id == "vlm/sft/qwen2-vl-7b-chartqa")
    plan = build_plan(r, "dry-run")
    assert plan["modality_contract"]["processing_class"] == "processor"
    assert plan["modality_contract"]["images_column"] == "images"


def test_prompt_only_vlm_exception_preserved(ctx: Context) -> None:
    r = next(
        x for x in load_recipes(ctx.root)
        if x.id == "vlm/grpo/qwen2-5-vl-3b-visual-reasoning"
    )
    plan = build_plan(r, "dry-run")
    assert plan["modality_contract"]["pre_rendered_prompt"] is True
    assert plan["modality_contract"]["images_column"] == "image"


def test_plan_needs_no_training_framework(ctx: Context, root: Path) -> None:
    """核心约束：导入并运行 plan 不应拉进 trl / transformers / torch。"""
    import subprocess
    import sys

    code = (
        "import sys; "
        "sys.path.insert(0, 'src'); "
        "from recipe_lab.loader import load_recipes; "
        "from recipe_lab.plan import build_plan; "
        "from pathlib import Path; "
        "r = next(x for x in load_recipes(Path('.')) "
        "if x.id == 'llm/grpo/qwen2-0-5b-numinamath'); "
        "build_plan(r); "
        "print('LOADED=' + ','.join(m for m in ('trl','torch','transformers') if m in sys.modules))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], cwd=root, text=True)
    assert out.strip() == "LOADED="


def test_render_plan_compact_contains_seed(ctx: Context) -> None:
    text = render_plan(build_plan(_sample(ctx), "dry-run"), compact=True)
    assert "seed       : 42 / data_seed 42" in text
    assert "$ python3 runner/train.py" in text


def test_virtual_recipe_still_builds(ctx: Context) -> None:
    """虚拟路径（单测 fixture）也要能出计划，不能依赖文件真实存在。"""
    from conftest import baseline_data

    r = make_recipe(ctx, baseline_data())
    plan = build_plan(r, "dry-run")
    assert plan["recipe_id"] == r.id
    assert plan["command"] == ""
