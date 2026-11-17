"""能力登记离线测试；所有 JSON 夹具均为合成测试对象，不是 GPU 实跑证据。"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe_lab import capabilities as cap
from recipe_lab.loader import load_recipes, load_taxonomy
from recipe_lab.model import Recipe

ROOT = Path(__file__).resolve().parents[1]
MATH_ID = "llm/grpo/qwen2-0-5b-numinamath"
VLLM_ID = "llm/grpo/qwen2-0-5b-vllm-colocate"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    contract = cap.load(ROOT)
    paths = {"registry/taxonomy.toml"}
    for entry in contract["recipes"].values():
        for dimension in cap.DIMENSIONS:
            for field in ("implementation", "tests"):
                paths.update(entry[dimension].get(field, []))
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    recipes = [Recipe(r.id, tmp_path / "recipes" / r.id / "recipe.toml", copy.deepcopy(r.data))
               for r in load_recipes(ROOT)]
    monkeypatch.setattr(cap, "load", lambda root: copy.deepcopy(contract))
    return SimpleNamespace(root=tmp_path, contract=contract, recipes=recipes)


def promote(sandbox, dimension, payload=None):
    cell = sandbox.contract["recipes"][MATH_ID][dimension]
    cell.update(status="verified", evidence=[f"evidence-{dimension}.json"])
    if payload is None:
        payload = {
            "recipe_id": MATH_ID, "stage": "completed", "kind": cap.EVIDENCE_KINDS[dimension],
            "mode": "full" if dimension == "train" else "eval",
            "runtime": {"gpu": "测试夹具设备", "cuda": "测试夹具版本"},
            "model_role": "trained_checkpoint", "metrics": {"accuracy": 0.5},
        }
    (sandbox.root / cell["evidence"][0]).write_text(json.dumps(payload), encoding="utf-8")
    return cell


def test_actual_registry_exact_coverage_and_counts():
    recipes = load_recipes(ROOT)
    contract = cap.load(ROOT)
    assert len(recipes) == len(contract["recipes"]) == 14
    assert set(contract["recipes"]) == {r.id for r in recipes}
    assert contract["dimensions"] == list(cap.DIMENSIONS)
    assert all(set(entry) == {*cap.DIMENSIONS, "wave"} for entry in contract["recipes"].values())
    result = cap.audit(ROOT, recipes)
    assert result.ok, result.errors
    assert result.registered_recipes == result.capability_entries == 14
    # 六篇五维齐备（数学/FIM/xLAM/RapidFire/Granite/PaliGemma）；implemented 只表示
    # 代码 + offline test，GPU 实测与 completed 证据仍为 0，不得据此晋级 B2/B3。
    assert result.implemented_recipes == 6 and result.gpu_verified_recipes == 0
    assert result.totals == {"missing": 40, "implemented": 30, "unverified": 0, "verified": 0}
    per_dimension = {dimension: {"missing": 8, "implemented": 6, "unverified": 0, "verified": 0}
                     for dimension in cap.DIMENSIONS}
    for dimension in cap.DIMENSIONS:
        assert result.counts[dimension] == per_dimension[dimension]
    full = {MATH_ID, "llm/sft/starcoder1b-fim",
            "llm/sft/llama3-8b-xlam-function-calling", "llm/sft/rapidfire-multiconfig",
            "vlm/sft/granite-vision-line-comparison",
            "vlm/sft/paligemma2-refcoco-grounding"}
    for rid, row in result.rows.items():
        if rid in full:
            assert all(cell["status"] == "implemented" for cell in row.values())
        else:
            assert all(cell["status"] == "missing" for cell in row.values())


@pytest.mark.parametrize("dimension", cap.DIMENSIONS)
def test_missing_dimension_is_error_and_closed(sandbox, dimension):
    sandbox.contract["recipes"][MATH_ID].pop(dimension)
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok
    assert any(f"{MATH_ID}/{dimension}: 缺少 dimension" in error for error in result.errors)
    assert result.rows[MATH_ID][dimension]["status"] == "missing"
    assert result.counts[dimension]["verified"] == 0


def test_orphan_and_missing_exact_ids_not_just_equal_counts(sandbox):
    entry = sandbox.contract["recipes"].pop(VLLM_ID)
    orphan = "llm/grpo/qwen2-0-5b-vllm-colocate-typo"
    sandbox.contract["recipes"][orphan] = entry
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok
    assert f"orphan 能力登记: {orphan}" in result.errors
    assert f"缺少 recipe 能力登记: {VLLM_ID}" in result.errors
    assert orphan not in result.rows and result.capability_entries == 13
    assert all(cell["status"] == "missing" for cell in result.rows[VLLM_ID].values())


def test_duplicate_recipe_and_subset_are_errors(sandbox):
    duplicate = cap.audit(sandbox.root, sandbox.recipes + [sandbox.recipes[0]])
    assert "输入 recipes 有重复 id" in duplicate.errors
    subset = cap.audit(sandbox.root, sandbox.recipes[:-1])
    assert any("expected_recipe_count" in error for error in subset.errors)
    assert any("orphan" in error for error in subset.errors)


@pytest.mark.parametrize("field,value", [
    ("implementation", ["src/recipe_lab/does-not-exist.py"]),
    ("tests", ["tests/test_does_not_exist.py"]),
    ("implementation", []), ("tests", []),
    ("implementation", "src/recipe_lab/math_task.py"),
    ("tests", "tests/test_math_task.py"),
    ("implementation", ["src/recipe_lab"]),
    ("implementation", ["registry/taxonomy.toml"]),
    ("tests", ["src/recipe_lab/math_task.py"]),
    ("implementation", ["../outside.py"]),
    ("implementation", [str(ROOT / "src/recipe_lab/math_task.py")]),
])
def test_fake_implemented_file_contract_rejected(sandbox, field, value):
    sandbox.contract["recipes"][MATH_ID]["train"][field] = value
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok and any(f"{MATH_ID}/train:" in error for error in result.errors)
    assert result.rows[MATH_ID]["train"]["status"] == "missing"
    assert result.counts["train"]["implemented"] == 5  # FIM / xLAM / RapidFire / Granite / PaliGemma 不受数学单元拒绝影响
    assert result.counts["train"]["verified"] == 0


def test_symlink_outside_repository_is_not_implementation(sandbox, tmp_path):
    target = tmp_path.parent / f"{tmp_path.name}-outside.py"
    target.write_text("pass\n", encoding="utf-8")
    (sandbox.root / "outside-link.py").symlink_to(target)
    sandbox.contract["recipes"][MATH_ID]["train"]["implementation"] = ["outside-link.py"]
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert any("越出仓库" in error for error in result.errors)


@pytest.mark.parametrize("status", ["registered", "GPUverified", "dry-run", "", True])
def test_illegal_status_cannot_promote(sandbox, status):
    sandbox.contract["recipes"][MATH_ID]["train"]["status"] = status
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok and result.rows[MATH_ID]["train"]["status"] == "missing"


def test_illegal_dimension_and_missing_note_rejected(sandbox):
    entry = sandbox.contract["recipes"][MATH_ID]
    entry["training"] = entry["train"]
    entry["train"]["note"] = ""
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert any("非法 dimension" in error for error in result.errors)
    assert any("note" in error for error in result.errors)


@pytest.mark.parametrize("payload", [
    "stage=completed kind=training", [], {},
    {"stage": "running", "kind": "training", "recipe_id": MATH_ID},
    {"stage": "completed", "kind": "evaluation", "recipe_id": MATH_ID},
    {"stage": "completed", "kind": "training", "recipe_id": VLLM_ID},
    {"stage": "completed", "kind": "training", "recipe_id": MATH_ID},
    {"stage": "completed", "kind": "training", "recipe_id": MATH_ID, "mode": "dry-run"},
])
def test_verified_requires_json_object_completed_kind_identity_and_runtime(sandbox, payload):
    promote(sandbox, "train", payload)
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok and result.rows[MATH_ID]["train"]["status"] == "missing"
    assert result.gpu_verified_recipes == 0


@pytest.mark.parametrize("evidence", ["GPU 已通过", ["GPU 已通过"], ["training.py"], []])
def test_verified_evidence_is_not_a_string_or_missing_file(sandbox, evidence):
    promote(sandbox, "train")
    sandbox.contract["recipes"][MATH_ID]["train"]["evidence"] = evidence
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok and result.counts["train"]["verified"] == 0


@pytest.mark.parametrize("content", ["not JSON", '"completed"', '{"stage":'])
def test_invalid_json_is_closed(sandbox, content):
    cell = promote(sandbox, "train")
    (sandbox.root / cell["evidence"][0]).write_text(content, encoding="utf-8")
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok and result.counts["train"]["verified"] == 0


@pytest.mark.parametrize("update", [
    {"mode": "smoke"}, {"mode": "dry-run"}, {"mock": True}, {"simulated": True},
    {"dry_run": True}, {"runtime": {"gpu": "mock-GPU", "cuda": "12"}},
    {"runtime": {"gpu": "设备", "cuda": "fake"}}, {"runtime": {}},
])
def test_mock_dry_run_smoke_never_gpu_verified(sandbox, update):
    cell = promote(sandbox, "train")
    path = sandbox.root / cell["evidence"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(update)
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok and result.counts["train"]["verified"] == 0


@pytest.mark.parametrize("update", [
    {"model_role": "base"}, {"model_role": "fallback"}, {"metrics": {}},
    {"metrics": {"accuracy": "0.5"}}, {"metrics": {"accuracy": True}},
    {"metrics": {"accuracy": float("nan")}}, {"metrics": {"accuracy": float("inf")}},
])
def test_eval_base_fallback_and_fake_metrics_never_verified(sandbox, update):
    cell = promote(sandbox, "eval")
    path = sandbox.root / cell["evidence"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(update)
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok and result.counts["eval"]["verified"] == 0


def test_valid_evidence_schema_does_not_implicitly_promote_other_dimensions(sandbox):
    # 只验证 schema 门禁，合成夹具不能证明任何实际 GPU 运行。
    promote(sandbox, "data")
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert result.ok and result.counts["data"]["verified"] == 1
    assert result.gpu_verified_recipes == 0
    promote(sandbox, "train")
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert result.ok and result.gpu_verified_recipes == 0
    promote(sandbox, "eval")
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert result.ok and result.gpu_verified_recipes == 1
    assert result.implemented_recipes == 6  # 六篇五维代码齐备（含 Granite / PaliGemma 发布级两维）
    # 六篇的 comparison/export 登记 implemented（只有代码，无 GPU 证据）；
    # 数学单篇的 verified 提升不改变其他篇的代码登记状态。
    assert result.counts["comparison"]["missing"] == result.counts["export"]["missing"] == 8
    assert result.counts["comparison"]["verified"] == result.counts["export"]["verified"] == 0
    assert "不自动更新 B2/B3" in cap.render(sandbox.root, sandbox.recipes)


def test_verified_still_requires_impl_and_offline_test(sandbox):
    promote(sandbox, "train")
    sandbox.contract["recipes"][MATH_ID]["train"].pop("tests")
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok and result.counts["train"]["verified"] == 0


def test_unverified_and_completed_path_do_not_implicitly_promote(sandbox):
    cell = promote(sandbox, "train")
    cell["status"] = "unverified"
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert result.ok and result.counts["train"]["unverified"] == 1
    assert result.counts["train"]["implemented"] == 5  # FIM / xLAM / RapidFire / Granite / PaliGemma 不受数学单元拒绝影响
    assert result.counts["train"]["verified"] == 0
    cell["status"] = "implemented"
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert result.ok and result.counts["train"]["implemented"] == 6
    assert result.counts["train"]["verified"] == 0


def test_missing_registry_fallback_keeps_registered_separate(tmp_path):
    (tmp_path / "registry").mkdir()
    shutil.copyfile(ROOT / "registry/taxonomy.toml", tmp_path / "registry/taxonomy.toml")
    recipes = load_recipes(ROOT)
    for recipe in recipes:
        recipe.data.update(status="verified", coverage="implemented")
        recipe.data["execution"] = {"adapter_status": "implemented", "evidence": "GPU 已通过"}
        recipe.data["baseline_contract"] = {"state": "B3"}
    with pytest.raises(cap.CapabilityError, match="无法加载"):
        cap.load(tmp_path)
    result = cap.audit(tmp_path, recipes)
    assert not result.ok and result.registered_recipes == 14 and result.capability_entries == 0
    assert result.implemented_recipes == result.gpu_verified_recipes == 0
    assert result.totals == {"missing": 70, "implemented": 0, "unverified": 0, "verified": 0}
    report = cap.render(tmp_path, recipes)
    assert "禁止依据该登记晋级" in report and "代码已实现：0 篇" in report
    assert "GPU verified：0 篇" in report
    assert len([line for line in report.splitlines() if line.startswith("| `")]) == 14


def test_malformed_toml_fallback_is_closed(tmp_path):
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry/capabilities.toml").write_text("schema = [", encoding="utf-8")
    result = cap.audit(tmp_path, load_recipes(ROOT))
    assert not result.ok and result.totals["implemented"] == result.totals["verified"] == 0
    assert "能力审计失败" in cap.render(tmp_path, load_recipes(ROOT))


def test_taxonomy_missing_families_dynamic_and_not_scope_backlog(sandbox):
    taxonomy = load_taxonomy(ROOT)
    expected = sorted(set(taxonomy["families"]) - {r.family for r in sandbox.recipes})
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert result.missing_families == expected
    assert result.untaxonomized_modules == ["audio", "cpt-dapt", "video"]
    assert not set(result.untaxonomized_modules) & set(result.missing_families)
    assert "ppo" in result.missing_families and taxonomy["families"]["ppo"]["tier"] == "removed"
    recipe = next(r for r in sandbox.recipes if r.id == VLLM_ID)
    recipe.data["method"]["family"] = "kto"
    changed = cap.audit(sandbox.root, sandbox.recipes)
    assert changed.ok and "kto" not in changed.missing_families
    assert changed.counts["train"]["implemented"] == 6
    with (sandbox.root / "registry/taxonomy.toml").open("a", encoding="utf-8") as handle:
        handle.write('\n[families.local-test]\ntrainer = "LocalTrainer"\ntier = "external"\n')
    assert "local-test" in cap.audit(sandbox.root, sandbox.recipes).missing_families
    assert "local-test" not in cap.audit(ROOT, load_recipes(ROOT)).missing_families


@pytest.mark.parametrize("field,value", [
    ("schema", 2), ("dimensions", ["train"]), ("evidence_kinds", {"train": "dry-run"}),
    ("states", {"registered": "登记"}), ("scope", {}), ("waves", []),
])
def test_invalid_contract_metadata_rejected(sandbox, field, value):
    sandbox.contract[field] = value
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok
    assert "能力审计失败" in cap.render(sandbox.root, sandbox.recipes)


def test_scope_and_wave_dependency_acceptance_validation(sandbox):
    sandbox.contract["scope"]["audio"]["status"] = "implemented"
    sandbox.contract["scope"]["audio"]["families"] = ["audio"]
    sandbox.contract["waves"][1]["dependencies"] = ["wave6"]
    sandbox.contract["waves"][2]["acceptance"] = ""
    result = cap.audit(sandbox.root, sandbox.recipes)
    assert not result.ok
    assert any("scope audio" in error for error in result.errors)
    assert any("wave1" in error for error in result.errors)
    assert any("wave2" in error for error in result.errors)
    assert "能力审计失败" in cap.render(sandbox.root, sandbox.recipes)


def test_report_complete_bounded_deterministic_and_read_only():
    paths = [ROOT / "registry/capabilities.toml", ROOT / "registry/taxonomy.toml"]
    before = [path.read_bytes() for path in paths]
    recipes = load_recipes(ROOT)
    report = cap.render(ROOT, iter(recipes))
    assert report == cap.render(ROOT, reversed(recipes))
    assert [path.read_bytes() for path in paths] == before
    assert len(report.splitlines()) < 200
    assert len([line for line in report.splitlines() if line.startswith("| `")]) == 14
    for phrase in (
        "五维能力计数", "逐篇能力矩阵", "具体缺失", "方法缺口", "Scope", "实施顺序",
        "dependencies", "验收条款", "zero train", "smoke", "3 train seeds", "held-out",
        "aggregate", "export reload", "business", "会话回收", "权重同步", "schema masker",
        "GSM8K 4 reward", "隔离执行评估", "ref model", "预渲染", "三损失", "assistant",
        "bbox", "loc", "4bit/LoRA", "4bit/DoRA", "污染", "0:500", "500:1000", "1000:1500",
    ):
        assert phrase in report, phrase
    assert "src/recipe_lab/math_task.py" in report and "tests/test_math_task.py" in report
    waves = cap.load(ROOT)["waves"]
    assert [wave["id"] for wave in waves] == [f"wave{i}" for i in range(7)]
    seen = set()
    for wave in waves:
        assert set(wave["dependencies"]) <= seen
        assert all(wave[field] for field in ("prerequisites", "work", "acceptance"))
        seen.add(wave["id"])
