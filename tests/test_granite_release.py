"""Granite 发布级 comparison / export 的隔离离线契约测试。

复用 test_granite.py 的确定性 fake runtime 调用真实
granite_task.run_training / run_evaluation 生成三份独立训练 seed 的合成证据，再验证：
  * 三 seed 套件的逐题图像/gold 重建、exact_match 重算与 Student-t95 聚合；
  * C306 / C307 按 task_router 分派到 granite_suite，不借用数学/FIM/xLAM/RapidFire 协议；
  * DoRA/4bit adapter-only 导出包、冻结 processor 声明与 offline 新进程复载（C405）。

所有产物只写 tmp_path；JSON/hash 只证明内部一致性，不构成 GPU 实跑证据，
也不自动晋级 B2/B3。fake runtime 明确是合成夹具，不声明任何真实模型行为。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipe_lab import baseline as bl
from recipe_lab import baseline_evidence as be
from recipe_lab import export_task as et
from recipe_lab import granite_export as ge
from recipe_lab import granite_suite as gs
from recipe_lab import granite_task as gt
from recipe_lab import provenance as pv
from recipe_lab import suite_task as st
from recipe_lab import task_router as tr
from recipe_lab.math_task import TaskExecutionError
from recipe_lab.model import Recipe
from test_export import read
from test_granite import (
    DATA_REV, LOCKFILE, MODEL_REV, SOURCE_SHA,
    _plan, _recipe, fake_runtime, install_fake, make_rows,
)

SEEDS = [42, 43, 44]
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 合成配置与证据

def _release_recipe(tmp_path):
    """三维测试夹具 + 基于「实跑」预声明的导出容差；不预声明 expected/tolerance。

    三维夹具的 frozen lock 把 runtime 全写成占位 "test"；发布级
    be.runtime_consistent 还会逐值比对 evidence.runtime 与 lock.runtime，
    因此这里把 python/cuda/gpu 改成与 fake runtime 实际写出的一致。
    """
    r = _recipe(tmp_path)
    data = copy.deepcopy(r.data)
    data["evaluation"]["export_tolerance"] = {name: 0.0 for name in gt.METRICS}
    lock_path = tmp_path / LOCKFILE
    lock = lock_path.read_text(encoding="utf-8")
    head = lock.split("[runtime]", 1)[0]
    lock_path.write_text(head + "[runtime]\n"
                         + f'python = "{sys.version.split()[0]}"\n'
                         + 'cuda = "test-cuda"\n'
                         + 'driver = "test-driver"\n'
                         + 'gpu = "test-device"\n', encoding="utf-8")
    return Recipe(r.id, r.path, data)


def _freeze(r, root):
    root.mkdir(parents=True, exist_ok=True)
    (root / st.CODE_SNAPSHOT).write_bytes(pv.snapshot_bytes(pv.code_snapshot()))
    (root / st.LOCKFILE).write_bytes((r.path.parent / LOCKFILE).read_bytes())


def _with_measurement(r, fcase):
    data = copy.deepcopy(r.data)
    data["baseline"]["measurement"] = str(fcase.base.parent.relative_to(r.path.parent) / fcase.base.name)
    return Recipe(r.id, r.path, data)


def _install(monkeypatch, r, rows):
    """test_granite.fake_runtime 返回 (runtime, calls, model)；install_fake 只吃 runtime。"""
    runtime, _, _ = fake_runtime(r, rows)
    install_fake(monkeypatch, runtime)


@pytest.fixture
def fcase(tmp_path, monkeypatch):
    """三 seed 合成训练/评估证据 + 冻结 root；只写 tmp_path，不加载真实框架。"""
    r = _release_recipe(tmp_path)
    root = tmp_path / "suite-root"
    _freeze(r, root)
    rows = make_rows()
    _install(monkeypatch, r, rows)
    base = gt.run_evaluation(r, _plan(r, root / "base", "zero-shot"), zero_shot=True)
    checkpoints, evaluations = {}, {}
    for seed in SEEDS:
        effective = pv.effective_recipe(r, seed)
        _install(monkeypatch, effective, rows)
        training = gt.run_training(effective, _plan(effective, root / f"seed-{seed}", "full"))
        _install(monkeypatch, effective, rows)
        evaluation = gt.run_evaluation(
            effective, _plan(effective, root / f"seed-{seed}", "eval"),
            checkpoint=training.parent, zero_shot=False)
        checkpoints[seed] = training.parent
        evaluations[seed] = evaluation
    return SimpleNamespace(r=r, root=root, base=base, checkpoints=checkpoints,
                           evaluations=evaluations, lock=tmp_path / LOCKFILE)


# ---------------------------------------------------------------- 路由与发布门禁

def test_router_and_baseline_evidence_dispatch_to_granite_release_modules(fcase):
    r = fcase.r
    assert tr.resolve(r) == gt.TASK_ID
    assert tr.load(r, "suite") is gs and tr.load(r, "export") is ge
    base = read(fcase.base)
    assert be.evaluation_problems(r, base, fcase.base, role="base") == []
    assert be.zero_train_measured(_with_measurement(r, fcase)) is True
    assert any("baseline-suite" in p for p in be.suite_problems(r))


def test_granite_release_never_borrows_other_task_protocols(fcase, tmp_path):
    r = fcase.r
    for foreign in ("math-grpo-v1", "code-fim-v1", "xlam-fc-v1", "rapidfire-sft-v1", "invented-v1"):
        data = copy.deepcopy(r.data)
        data["execution"] = {**data["execution"], "task_id": foreign}
        with pytest.raises(TaskExecutionError, match="任务路由冲突|未知 execution.task_id"):
            tr.resolve(Recipe(r.id, r.path, data))
    assert "image_token_id" in ge._base_reference(r, read(fcase.base)["processor_identity"])["loss"]


def test_registry_declares_all_five_granite_dimensions(fcase):
    from recipe_lab import capabilities as cap
    result = cap.audit(ROOT, __import__("recipe_lab.loader", fromlist=["load_recipes"]).load_recipes(ROOT))
    assert result.ok, result.errors
    row = result.rows[gt.RECIPE_ID]
    assert all(cell["status"] == "implemented" for cell in row.values())
    assert set(result.rows[gt.RECIPE_ID]) == set(cap.DIMENSIONS)


# ---------------------------------------------------------------- comparison / 套件

def test_suite_plan_is_readonly_and_granite_specific(fcase):
    before = {p: p.stat().st_mtime_ns for p in fcase.root.rglob("*") if p.is_file()}
    plan = gs.suite_plan(fcase.r, SEEDS, fcase.root)
    assert plan["status"] == "declared" and plan["task_id"] == gt.TASK_ID
    assert plan["seeds"] == SEEDS and len(plan["commands"]) == 7
    assert plan["protocol"] == gt.EVAL_PROTOCOL and plan["metrics"] == list(gt.METRICS)
    assert plan["full_steps"] == 150 and plan["training_source"]["adapter_only"] is True
    assert plan["training_source"]["vision_tower_skipped_from_quantization"] is True
    assert plan["shell"] is False and plan["baseline_promotion"] == "none"
    assert not plan["resume_supported"] and "不解释几何语义" in plan["limitations"]
    assert {p: p.stat().st_mtime_ns for p in fcase.root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("seeds", [[], [1, 1, 2], [1, 2, True], [1, 2, 2**32]])
@pytest.mark.parametrize("operation", [gs.suite_plan, gs.aggregate_suite, gs.run_suite])
def test_invalid_seed_lists_fail_without_writes(fcase, seeds, operation):
    root = fcase.root.parent / "unused-root"
    with pytest.raises(TaskExecutionError):
        operation(fcase.r, seeds, root)
    assert not root.exists()


def test_aggregate_suite_recomputes_three_independent_seeds(fcase):
    result = gs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    assert result == fcase.root / gs.FINAL
    data = read(result)
    assert data["kind"] == "baseline-suite" and data["status"] == "completed"
    assert data["task_id"] == gt.TASK_ID and data["baseline_promotion"] == "none"
    assert [run["train_seed"] for run in data["runs"]] == SEEDS
    assert len({run["training_evidence"]["path"] for run in data["runs"]}) == 3
    assert data["statistical_unit"] == "independent_training_seed" and data["fixed_test"] is True
    assert data["training_seed_variance_measured"] is True
    for name in gs.METRICS:
        stats = data["stats"][name]
        assert stats["trained"]["n"] == 3 and stats["paired_delta"]["n"] == 3
        assert stats["trained"]["student_t95_ci"]["df"] == 2
    assert data["acceptance"]["status"] == "not_evaluated"  # 不回填 expected/tolerance
    assert gs.validate_suite_payload(fcase.r, data, result) is None


@pytest.mark.parametrize("change", ["stats", "fingerprint", "runs", "task_id",
                                    "base_metrics", "records", "evaluation_protocol"])
def test_suite_payload_rejects_tampered_summary(fcase, change):
    result = gs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    data = read(result)
    if change == "stats":
        data["stats"]["exact_match"]["trained"]["mean"] = 0.123
    elif change == "fingerprint":
        data["fingerprint"] = "0" * 64
    elif change == "runs":
        data["runs"] = data["runs"][:2]
    elif change == "task_id":
        data["task_id"] = "math-grpo-v1"
    elif change == "base_metrics":
        data["runs"][0]["base_metrics"]["exact_match"] = 0.5
    elif change == "evaluation_protocol":
        data["base_evaluation"] = {**data["base_evaluation"], "sha256": "0" * 64}
    else:
        key = next(iter(data["records"]))
        data["records"][key]["sha256"] = "0" * 64
    with pytest.raises(TaskExecutionError):
        gs.validate_suite_payload(fcase.r, data, result)


def test_suite_payload_rejects_checkpoint_weight_drift(fcase):
    result = gs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    (fcase.checkpoints[43] / "adapter_model.safetensors").write_bytes(b"drifted adapter")
    with pytest.raises(TaskExecutionError):
        gs.validate_suite_payload(fcase.r, read(result), result)


def test_suite_payload_rejects_processor_identity_drift(fcase):
    result = gs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    path = fcase.evaluations[43].parent / gt.PROCESSOR_IDENTITY_FILE
    identity = read(path)
    identity["image_token_id"] = 999
    identity["sha256"] = gt._hash_manifest(identity)["sha256"]
    path.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(TaskExecutionError):
        gs.validate_suite_payload(fcase.r, read(result), result)


def test_evaluation_content_recomputes_exact_match_and_rejects_tampering(fcase):
    base = read(fcase.base)
    assert gs.evaluation_problems(fcase.r, base, fcase.base, role="base") == []
    assert gs.zero_train_validation(fcase.r, base, fcase.base) is None
    # 摘要计数被替换，但生成文本没变：逐题重算必须拒绝。
    swapped = copy.deepcopy(base)
    swapped["metric_counts"]["exact_match"] = 999
    assert gs.evaluation_problems(fcase.r, swapped, fcase.base, role="base")
    # 逐题分数被替换，但 generated_text 没变：score_prediction 重算必须拒绝。
    prediction_path = fcase.base.parent / gt.PREDICTIONS_FILE
    original_bytes = prediction_path.read_bytes()
    document = read(prediction_path)
    document["predictions"][0]["exact_match"] = 0.0
    document["sha256"] = gt.sha256_json(document["predictions"])
    prediction_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    assert gs.evaluation_problems(fcase.r, base, fcase.base, role="base")
    prediction_path.write_bytes(original_bytes)
    assert gs.evaluation_problems(fcase.r, base, fcase.base, role="base") == []
    # eval_manifest 的 gold 被换成别的答案：内容 hash 与逐题重算必须拒绝。
    manifest_path = fcase.base.parent / gt.EVAL_MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    manifest = read(manifest_path)
    manifest["samples"][0]["answer"] = "a completely different answer"
    manifest["samples_sha256"] = gt.sha256_json(manifest["samples"])
    manifest["sha256"] = gt._hash_manifest(manifest)["sha256"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    assert gs.evaluation_problems(fcase.r, base, fcase.base, role="base")
    manifest_path.write_bytes(manifest_bytes)
    assert gs.evaluation_problems(fcase.r, base, fcase.base, role="base") == []


def test_suite_problems_fails_closed_without_completed_evidence(fcase):
    problems = gs.suite_problems(fcase.r)
    assert problems and "baseline-suite" in problems[0]
    assert be.suite_problems(fcase.r) == problems


def test_zero_train_validation_rejects_checkpoint_role(fcase):
    trained = read(fcase.evaluations[42])
    with pytest.raises(TaskExecutionError, match="zero-shot"):
        gs.zero_train_validation(fcase.r, trained, fcase.evaluations[42])


# ---------------------------------------------------------------- export / offline 复载

def fake_reload(r, plan, checkpoint=None, zero_shot=False):
    """测试专用：把包内参考证据落到 worker 固定输出目录，不加载 GPU 框架。"""
    bundle = Path(checkpoint).parent
    output = Path(plan["run"]["output_dir"])
    assert output == bundle / "reload" / "eval"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation.json", "source_manifest.json", gt.PROCESSOR_IDENTITY_FILE,
                 gt.EVAL_MANIFEST_FILE, gt.PREDICTIONS_FILE):
        shutil.copyfile(bundle / "reference" / name, output / name)
    return output / "evaluation.json"


def mock_worker(monkeypatch, destination):
    child_pid, parent_pid = os.getpid() + 10000, os.getpid()
    calls = []
    monkeypatch.setattr(gt, "run_evaluation", fake_reload)

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        process = Mock(pid=child_pid, returncode=0)

        def communicate(timeout):
            assert timeout == et.WORKER_TIMEOUT_SECONDS
            with monkeypatch.context() as child:
                for key, value in kwargs["env"].items():
                    child.setenv(key, value)
                child.setattr(os, "getpid", lambda: child_pid)
                child.setattr(os, "getppid", lambda: parent_pid)
                ge.run_worker(destination)
            return "Granite 测试 worker 输出", ""
        process.communicate.side_effect = communicate
        return process
    monkeypatch.setattr(et.subprocess, "Popen", launch)
    return calls


def test_export_plan_is_readonly_and_tolerance_aware(fcase, tmp_path):
    r = fcase.r
    plan = ge.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42],
                            tmp_path / "dry", execute=False)
    assert plan["kind"] == "export-plan" and plan["task_id"] == gt.TASK_ID
    assert plan["protocol"] == gt.EVAL_PROTOCOL and plan["metrics"] == list(gt.METRICS)
    assert plan["tolerance"] == {name: 0.0 for name in gt.METRICS}
    assert plan["tolerance_status"] == "declared"
    assert plan["base_weights_bundled"] is False and plan["license"]["status"] == "pending"
    assert plan["vision_contract"]["explicit_image_token_mask"] is True
    assert plan["vision_contract"]["assistant_only_loss"] is True
    assert not (tmp_path / "dry").exists()
    missing = copy.deepcopy(r.data)
    missing["evaluation"].pop("export_tolerance")
    undeclared = Recipe(r.id, r.path, missing)
    dry = ge.export_bundle(undeclared, fcase.checkpoints[42], fcase.evaluations[42],
                           tmp_path / "undeclared", execute=False)
    assert dry["tolerance"] is None and "not_declared" in dry["tolerance_status"]


def test_export_full_fails_closed_without_predeclared_tolerance(fcase, tmp_path):
    data = copy.deepcopy(fcase.r.data)
    data["evaluation"].pop("export_tolerance")
    undeclared = Recipe(fcase.r.id, fcase.r.path, data)
    with pytest.raises(TaskExecutionError, match="tolerance"):
        ge.export_bundle(undeclared, fcase.checkpoints[42], fcase.evaluations[42],
                         tmp_path / "bundle")


def test_export_bundle_copies_adapter_and_reloads_in_isolated_worker(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "bundle"
    calls = mock_worker(monkeypatch, destination)
    result = ge.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    evidence = read(result)
    assert result == destination / ge.FINAL
    assert evidence["task_id"] == gt.TASK_ID and evidence["stage"] == "completed"
    assert evidence["seed"] == 42 and evidence["license"]["status"] == "pending"
    assert evidence["base_weights_bundled"] is False and evidence["baseline_promotion"] == "none"
    assert evidence["protocol"] == gt.EVAL_PROTOCOL and evidence["generated_code_executed"] is False
    assert all(value["within_tolerance"] for value in evidence["comparison"].values())
    assert evidence["subprocess"]["pid"] != os.getpid() and evidence["subprocess"]["returncode"] == 0
    assert calls[0][0] == [sys.executable, str(ge.CODE_ROOT / "runner" / "export.py"),
                           "--worker", "--bundle", str(destination)]
    assert ge.validate_export_evidence(r, result) is True
    base = read(destination / ge.BASE_REFERENCE)
    assert base["base_weights_bundled"] is False and base["adapter_only"] is True
    assert base["model_id"] == gt.MODEL and base["revision"] == r.get("model.revision")
    assert base["protocol"] == gt.EVAL_PROTOCOL and base["generated_code_executed"] is False
    assert type(base["image_token_id"]) is int and "image_token_id" in base["loss"]
    assert base["processor_identity_sha256"] and base["chat_template_sha256"]
    assert (destination / "model" / "adapter_model.safetensors").read_bytes() == \
           (fcase.checkpoints[42] / "adapter_model.safetensors").read_bytes()
    assert not list(destination.glob("model/*.bin"))
    # 快照与训练时 effective recipe 逐字节一致，且不得改写 execution.lockfile。
    snapshot = read(destination / ge.SNAPSHOT)
    effective = pv.effective_recipe(r, 42)
    assert snapshot == effective.data and snapshot["train"]["seed"] == 42
    assert snapshot["execution"]["lockfile"] == LOCKFILE
    assert tomllib.loads((destination / "recipe.toml").read_text(
        encoding="utf-8"))["execution"]["lockfile"] == LOCKFILE


def test_c405_dispatch_uses_granite_exporter(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "bundle"
    mock_worker(monkeypatch, destination)
    result = ge.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    data = copy.deepcopy(r.data)
    data["baseline_contract"] = {
        "export_evidence": str(result.parent.relative_to(r.path.parent) / result.name),
        "export_verified": True, "rollback_verified": True,
        "rollback_evidence": "manual/rollback-review.md"}
    registered = Recipe(r.id, r.path, data)
    check = {"kind": "export_reload", "field": "baseline_contract.export_evidence",
             "confirmation_field": "baseline_contract.export_verified"}
    assert bl.check_passed(registered, check) is True
    (destination / "reload" / "eval" / gt.PREDICTIONS_FILE).write_text("{}\n", encoding="utf-8")
    assert bl.check_passed(registered, check) is False


def test_export_evidence_rejects_tampered_reload_or_bundle(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "bundle"
    mock_worker(monkeypatch, destination)
    result = ge.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    (destination / "reload" / "eval" / gt.PREDICTIONS_FILE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(TaskExecutionError):
        ge.validate_export_evidence(r, result)
    shutil.copyfile(destination / "reference" / gt.PREDICTIONS_FILE,
                    destination / "reload" / "eval" / gt.PREDICTIONS_FILE)
    assert ge.validate_export_evidence(r, result) is True
    (destination / "model" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(TaskExecutionError):
        ge.validate_export_evidence(r, result)


def test_verify_bundle_rejects_unbound_extra_file(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "prepared"

    def stop(bundle):
        raise TaskExecutionError("测试停止于 prepared 包")
    with monkeypatch.context() as context:
        context.setattr(et, "_launch_worker", stop)
        with pytest.raises(TaskExecutionError, match="prepared"):
            ge.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    assert (destination / ge.MANIFEST).is_file() and not (destination / ge.FINAL).exists()
    (destination / "model" / "unbound.json").write_text('{}\n', encoding="utf-8")
    with pytest.raises(TaskExecutionError, match="未绑定文件"):
        ge._verify_bundle(destination)


def test_lock_layout_keeps_content_identity_for_repository_relative_lock(tmp_path):
    raw = "../../../../runner/locks/vlm__sft__granite-vision-line-comparison.lock.toml"
    bundle = tmp_path / "bundle"
    lock_bytes = b'schema_version = 1\nstatus = "frozen"\n'
    lock_path = bundle / "runner" / "locks" / "vlm__sft__granite-vision-line-comparison.lock.toml"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(lock_bytes)
    relative, resolved, recipe_path = ge._lock_layout(raw, bundle)
    assert relative == "runner/locks/vlm__sft__granite-vision-line-comparison.lock.toml"
    assert resolved == lock_path
    assert recipe_path.parent == bundle / ge.LOCK_ANCHOR / ge.LOCK_ANCHOR / ge.LOCK_ANCHOR / ge.LOCK_ANCHOR
    assert Path(os.path.abspath(recipe_path.parent / raw)) == lock_path
    data = {"id": gt.RECIPE_ID, "execution": {"task_id": gt.TASK_ID, "lockfile": raw}}
    r = Recipe(gt.RECIPE_ID, recipe_path, data)
    assert r.get("execution.lockfile") == raw
    assert gt.execution_identity(r)["environment_lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest()
    assert be.frozen_environment(r) is False
    relative2, resolved2, _ = ge._lock_layout("../locks/frozen.lock.toml", bundle)
    assert relative2 == "locks/frozen.lock.toml" and resolved2 == bundle / "locks" / "frozen.lock.toml"


@pytest.mark.parametrize("raw", ["model/adapter_model.safetensors", "recipe-snapshot.json",
                                 "reference/evaluation.json", "../../a/../../b.lock.toml",
                                 "/absolute.lock.toml", "..\\windows.lock.toml", ""])
def test_lock_layout_rejects_escaping_or_conflicting_paths(tmp_path, raw):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with pytest.raises(TaskExecutionError):
        ge._lock_layout(raw, bundle)


# ---------------------------------------------------------------- CLI 预演与失败关闭

def test_release_cli_dry_runs_dispatch_to_granite_and_full_fails_closed(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "RECIPE_LAB_ROOT": str(ROOT)}
    suite = subprocess_run([
        "runner/suite.py", "--recipe", gt.RECIPE_ID, "--seeds", "42", "43", "44",
        "--output-root", str(tmp_path / "suite"), "--dry-run"], env)
    assert suite.returncode == 0, suite.stderr
    suite_plan = json.loads(suite.stdout)
    assert suite_plan["task_id"] == gt.TASK_ID and "release_pipeline" not in suite_plan
    export = subprocess_run([
        "runner/export.py", "--recipe", gt.RECIPE_ID, "--checkpoint", "missing",
        "--evaluation", "missing", "--output", str(tmp_path / "bundle"), "--dry-run"], env)
    assert export.returncode == 0, export.stderr
    export_plan = json.loads(export.stdout)
    assert export_plan["task_id"] == gt.TASK_ID and "release_pipeline" not in export_plan
    full = subprocess_run([
        "runner/suite.py", "--recipe", gt.RECIPE_ID, "--seeds", "42", "43", "44",
        "--output-root", str(tmp_path / "full"), "--full"], env)
    assert full.returncode != 0
    assert "baseline" in (full.stdout + full.stderr).lower() or "失败" in (full.stdout + full.stderr) \
        or "Error" in (full.stdout + full.stderr)
    assert not (tmp_path / "full").exists()


def subprocess_run(args, env):
    import subprocess
    return subprocess.run([sys.executable, str(ROOT / args[0]), *args[1:]],
                          env=env, capture_output=True, text=True, check=False, timeout=60)
