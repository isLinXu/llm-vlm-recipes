"""门禁的正反向测试；所有证据仅在tmp_path内合成，绝不作GPU实测交付。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab.baseline import audit_one, check_passed, load_contract
from recipe_lab.loader import load_recipes, load_taxonomy
from recipe_lab.math_task import sha256_json, require_frozen_runtime, TaskExecutionError, artifact_manifest
from recipe_lab.model import Recipe
from recipe_lab.rules import Context, l6_baseline
from test_math_task import runner
from test_export import case  # noqa: F401
from test_suite import freeze, write_base, write_training, write_evaluation
from recipe_lab import suite_task as st
from recipe_lab import provenance as pv

ROOT = Path(__file__).resolve().parents[1]
MATH = "llm/grpo/qwen2-0-5b-numinamath"


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def record(base, path):
    return {"path": str(path.relative_to(base)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def suite(case):
    # 使用完整1500条合成数据和生产聚合器，不能再用1题/非代码文本绕过B2内容门禁。
    original = next(r for r in load_recipes(ROOT) if r.id == MATH)
    data = copy.deepcopy(original.data)
    for key, value in copy.deepcopy(case.r.data).items():
        if key != "dependencies" and isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    folder = case.r.path.parent
    recipe_path = case.r.path
    data["baseline"]["measurement"] = "evidence/base/zero-shot/evaluation.json"
    data["evaluation"].update(expected={n: 1.0 for n in data["evaluation"]["metrics"]},
                              tolerance={n: 0.05 for n in data["evaluation"]["metrics"]})
    data["execution"].update(lockfile="lock.toml", entrypoint="train.py", eval_script="eval.py",
                             evidence="evidence/suite.json")
    data["baseline_contract"]["state"] = "B2"
    for name in ("train.py", "eval.py"):
        (folder / name).write_text("# 隔离测试入口，不是真实训练证据\n", encoding="utf-8")
    lock_path = folder / "lock.toml"
    lock_path.write_bytes(case.lock.read_bytes())
    # 保留原门禁测试的环境字段，仅是夹具设备。
    text = lock_path.read_text().replace('cuda = "12.4"', 'cuda = "12.1"').replace('driver = "test-driver"', 'driver = "535"').replace('gpu = "test-only-GPU"', 'gpu = "fixture-device"')
    lock_path.write_text(text, encoding="utf-8")
    recipe = Recipe(MATH, recipe_path, data)
    case.r, case.lock = recipe, lock_path
    from test_export import revise
    for source in (case.checkpoint / "training.json", case.evaluation):
        revise(source, lambda value: value["runtime"].update(cuda="12.1", gpu="fixture-device"))
    evidence_dir = folder / "evidence"
    freeze(case, evidence_dir)
    write_base(case, evidence_dir)
    for seed in (42, 43, 44):
        effective = pv.effective_recipe(recipe, seed)
        checkpoint = write_training(case, effective, evidence_dir)
        write_evaluation(case, effective, evidence_dir / f"seed-{seed}" / "eval",
                         {**artifact_manifest(checkpoint),
                          "role": "trained_checkpoint"}, 500, "eval", evidence_dir)
    path = st.aggregate_suite(recipe, [42, 43, 44], evidence_dir)
    payload = json.loads(path.read_text())
    return SimpleNamespace(recipe=recipe, payload=payload, path=path, folder=folder, evidence=evidence_dir)


def test_valid_synthetic_b2_not_forced_to_meet_b3(suite):
    r = suite.recipe
    assert be.frozen_environment(r)
    assert be.zero_train_measured(r)
    assert be.suite_problems(r) == []
    result = audit_one(r, load_contract(ROOT))
    assert result.computed == "B2" and result.state_ok
    ctx = Context(recipes=[r], taxonomy=load_taxonomy(ROOT), root=ROOT, level=6)
    assert not [f for f in l6_baseline(r, ctx) if f.rule == "R605"], "合法B2不能被要求满足B3许可或owner"
    assert any(c["min_state"] == "B3" for c in result.gaps)


@pytest.mark.parametrize("revision", ["main", "latest", "unknown", "", "abc", True])
def test_commit_placeholder_rejected(suite, revision):
    suite.recipe.data["model"]["revision"] = revision
    assert not check_passed(suite.recipe, {"kind": "commit", "field": "model.revision"})
    assert not be.frozen_environment(suite.recipe)


@pytest.mark.parametrize("field,value", [("expected", "待实跑"), ("tolerance", "unknown"),
                                          ("expected", {"pass@1": float("nan"), "format_compliance": 1}),
                                          ("tolerance", {"pass@1": -1, "format_compliance": 0}),
                                          ("expected", {"pass@1": True, "format_compliance": 1})])
def test_nonnumeric_or_incomplete_metrics_cannot_promote(suite, field, value):
    suite.recipe.data["evaluation"][field] = value
    assert not check_passed(suite.recipe, {"kind": "numbers", "field": "evaluation." + field})
    assert be.suite_problems(suite.recipe)


@pytest.mark.parametrize("mutation", ["duplicate_seeds", "short_runs", "missing_code", "text_training", "changed_prediction",
                                     "changed_identity", "wrong_metrics", "wrong_protocol", "incomplete_train", "changed_params"])
def test_invalid_linked_suite_closed(suite, mutation):
    p, run = suite.payload, suite.payload["runs"][0]
    if mutation == "duplicate_seeds":
        p["runs"][1]["train_seed"] = run["train_seed"]
    elif mutation == "short_runs":
        p["runs"] = p["runs"][:2]
    elif mutation == "missing_code":
        p.pop("code_snapshot")
    elif mutation == "changed_prediction":
        (suite.evidence / run["predictions"]["path"]).write_text("[]", encoding="utf-8")
    elif mutation in ("wrong_metrics", "wrong_protocol"):
        ep = suite.evidence / run["trained_evaluation"]["path"]
        e = json.loads(ep.read_text())
        if mutation == "wrong_metrics":
            e["metrics"]["pass@1"] = 0.3
            run["trained_metrics"] = e["metrics"]
        else:
            e["config"]["generation"]["do_sample"] = True
            e["config_sha256"] = sha256_json(e["config"])
        put(ep, e)
        run["trained_evaluation"] = record(suite.evidence, ep)
    else:
        tp = suite.evidence / run["training_evidence"]["path"]
        t = json.loads(tp.read_text())
        if mutation == "text_training":
            t = "stage completed"
        elif mutation == "changed_identity":
            t["recipe_id"] = "other-recipe"
        elif mutation == "incomplete_train":
            t["steps"] = 1
        else:
            t["config"]["learning_rate"] *= 2
            t["config_sha256"] = sha256_json(t["config"])
        put(tp, t)
        run["training_evidence"] = record(suite.evidence, tp)
    put(suite.path, p)
    assert be.suite_problems(suite.recipe)
    assert audit_one(suite.recipe, load_contract(ROOT)).computed == "B1"


@pytest.mark.parametrize("payload", ["not-json", [], {"kind": "baseline-suite"}, {"runs": [None, {}, {}]}])
def test_bad_json_shape_never_crashes_or_promotes(suite, payload):
    put(suite.path, payload)
    assert be.suite_problems(suite.recipe)


def test_prediction_score_tamper_and_cross_split_identity_rejected(suite):
    base = suite.evidence / "base/zero-shot/evaluation.json"
    evidence = json.loads(base.read_text())
    ep = base.parent / evidence["predictions_file"]["path"]
    predictions = json.loads(ep.read_text())
    predictions[0]["accuracy"] = 1
    put(ep, predictions)
    evidence["predictions_file"] = record(suite.evidence, ep)
    put(base, evidence)
    assert not be.zero_train_measured(suite.recipe), "不能更改逐样本分数但沿用旧汇总"
    manifest = copy.deepcopy(evidence["split_manifest"])
    manifest["splits"]["test"]["samples"] = manifest["splits"]["train"]["samples"]
    manifest["splits"]["test"]["sha256"] = sha256_json(manifest["splits"]["test"]["samples"])
    manifest["sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    assert not be.valid_manifest(manifest)


def test_fingerprint_tracks_dependencies_budget_and_params(suite):
    original = be.fingerprint(suite.recipe)
    suite.recipe.data["budget"]["full_steps"] += 1
    assert be.fingerprint(suite.recipe) != original
    assert be.suite_problems(suite.recipe)


def test_hash_file_rejects_traversal_and_changed_bytes(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside.json"
    put(outside, {})
    assert not be.hashed_file(inside, {"path": "../outside.json", "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()})
    path = put(inside / "ok.json", {})
    rec = record(inside, path)
    assert be.hashed_file(inside, rec)
    put(path, {"changed": True})
    assert not be.hashed_file(inside, rec)


def test_online_environment_not_hub_dataset(suite):
    r = suite.recipe
    r.data["dataset"].update(kind="online", revision="")
    r.data["environment"] = {"revision": "e" * 40, "corpus_sha256": "f" * 64, "seed": 0}
    assert be.dataset_pinned(r)
    r.data["environment"]["seed"] = True
    assert not be.dataset_pinned(r)


def test_frozen_lock_cannot_be_empty_or_partial(suite):
    path = suite.folder / "lock.toml"
    text = path.read_text()
    assert be.frozen_environment(suite.recipe)
    path.write_text(text.replace('status = "frozen"', 'status = "resolved-inputs"'))
    assert not be.frozen_environment(suite.recipe)
    path.write_text(text.replace('torch==2.5.1', 'torch==latest'))
    assert not be.frozen_environment(suite.recipe)
    path.write_text(text.replace('torch==2.5.1', 'unrelated==2.5.1'))
    assert not be.frozen_environment(suite.recipe)


def test_runtime_rejects_template_before_any_download(suite):
    (suite.folder / "lock.toml").write_text('status = "template"\n')
    with pytest.raises(TaskExecutionError, match="尚未冻结"):
        require_frozen_runtime(suite.recipe, {})


def test_runtime_checks_actual_complete_environment(suite, monkeypatch):
    import importlib.metadata
    import subprocess
    import sys
    path = suite.folder / "lock.toml"
    text = path.read_text().replace('python = "3.11.0"', 'python = "' + sys.version.split()[0] + '"')
    path.write_text(text)
    import tomllib
    lock = tomllib.loads(text)
    dependencies = {"torch": SimpleNamespace(version=SimpleNamespace(cuda="12.1"),
                    cuda=SimpleNamespace(get_device_name=lambda _: "fixture-device"))}
    monkeypatch.setattr(importlib.metadata, "version", lambda name: suite.recipe.get("dependencies." + name))
    def execute(command, **kwargs):
        return "535\n" if command[0] == "nvidia-smi" else "\n".join(lock["requirements"])
    monkeypatch.setattr(subprocess, "check_output", execute)
    require_frozen_runtime(suite.recipe, dependencies)
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "drifted")
    with pytest.raises(TaskExecutionError, match="驱动或完整"):
        require_frozen_runtime(suite.recipe, dependencies)


def test_missing_or_mock_runtime_cannot_promote(suite):
    training_path = suite.evidence / suite.payload["runs"][0]["training_evidence"]["path"]
    data = json.loads(training_path.read_text())
    data["runtime"]["gpu"] = "mock-GPU"
    put(training_path, data)
    suite.payload["runs"][0]["training_evidence"] = record(suite.evidence, training_path)
    put(suite.path, suite.payload)
    assert be.suite_problems(suite.recipe)


def test_lock_preview_and_reapply_do_not_touch_valid_freeze(suite, monkeypatch):
    module = runner("lock")
    monkeypatch.setattr(module, "load_recipes", lambda root: [suite.recipe])
    monkeypatch.setattr(module, "LOCK_DIR", suite.folder)
    monkeypatch.setattr(module, "lock_path", lambda r: suite.folder / "lock.toml")
    def forbidden(*args):
        pytest.fail("已锁或只读预演不应联网查询main")
    monkeypatch.setattr(module, "query", forbidden)
    monkeypatch.setattr(module, "query_source_sha", forbidden)
    files = [suite.recipe.path, suite.folder / "lock.toml"]
    before = [f.read_bytes() for f in files]
    for action in ([], ["--apply"]):
        monkeypatch.setattr(module.sys, "argv", ["lock.py", "--recipe", MATH, *action])
        assert module.main() == 0
        assert [f.read_bytes() for f in files] == before
