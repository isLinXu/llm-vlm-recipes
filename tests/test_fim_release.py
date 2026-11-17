"""FIM 发布级 comparison / export 的离线契约测试。

用隔离 fake runtime 生成三份独立训练 seed 的合成证据，再验证：
  * 三 seed 套件的内容级重算与统计聚合；
  * 发布门禁的 PSM 输入/gold 重算器；
  * adapter-only 导出包、冻结 base 声明与「新进程」离线复载。

所有产物只写 tmp_path。JSON/hash 只证明内部一致性，不构成 GPU 实跑证据，
也不自动晋级 B2/B3。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import export_task as et
from recipe_lab import fim_export as fe
from recipe_lab import fim_suite as fs
from recipe_lab import fim_task as ft
from recipe_lab import math_task as mt
from recipe_lab import provenance as pv
from recipe_lab import suite_task as st
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan
from test_export import read
from test_fim import fake_runtime, install_fake, rows  # noqa: F401

SEEDS = [42, 43, 44]
LOCKFILE = "frozen.lock.toml"


def _recipe(tmp_path):
    data = {
        "id": ft.RECIPE_ID, "source_sha": "c" * 40,
        "method": {"family": "sft", "trainer": "Trainer", "upstream_trainer": "Trainer", "peft": "qlora"},
        "model": {"base": ft.MODEL, "revision": "a" * 40, "modality": "llm"},
        "dataset": {"name": ft.DATASET, "revision": "b" * 40},
        "fim": copy.deepcopy(ft.FIM_SPEC), "peft": copy.deepcopy(ft.PEFT_SPEC),
        "quantization": copy.deepcopy(ft.QUANT_SPEC),
        "train": {
            "seed": 0, "data_seed": 0, "seed_mechanism": "上游 set_seed + shuffle + ConstantLengthDataset",
            "learning_rate": 5e-4, "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1, "max_steps": 2000, "seq_length": 32,
            "lr_scheduler_type": "cosine", "weight_decay": 0.01, "warmup_steps": 30,
            "eval_steps": 100, "bf16": True, "fim_rate": 0.5, "fim_spm_rate": 0.5,
            "lora_r": 8, "lora_alpha": 32, "lora_dropout": 0.0, "held_out_examples": 4000,
            "per_device_eval_batch_size": 16, "save_steps": 100, "logging_steps": 25,
            "gradient_checkpointing": True, "fp16": False, "optim": "adamw_torch",
            "dataloader_num_workers": 0, "remove_unused_columns": False,
        },
        "budget": {"full_steps": 2000, "smoke_steps": 5},
        "evaluation": {"metrics": ["fim_exact_match", "fim_token_exact_match"],
                       "seed": 0, "max_new_tokens": 8, "multi_run": True,
                       "export_tolerance": {"fim_exact_match": 0.0, "fim_token_exact_match": 0.0}},
        "execution": {"task_id": ft.TASK_ID, "lockfile": LOCKFILE},
        "baseline": {"result": "unmeasured"},
    }
    # 合成 frozen lock；runtime.python 必须与当前解释器一致，否则内容级门禁拒绝。
    lock = "\n".join([
        "schema_version = 1", 'status = "frozen"', f'recipe_id = "{ft.RECIPE_ID}"',
        f'model_id = "{ft.MODEL}"', f'dataset_id = "{ft.DATASET}"',
        f'model_revision = "{"a" * 40}"', f'dataset_revision = "{"b" * 40}"',
        f'source_sha = "{"c" * 40}"',
        'requirements = ["torch==test", "transformers==test", "datasets==test", "peft==test", "bitsandbytes==test"]',
        "[runtime]", f'python = "{sys.version.split()[0]}"', 'cuda = "test"',
        'driver = "test-driver"', 'gpu = "test-device"',
    ]) + "\n"
    (tmp_path / LOCKFILE).write_text(lock, encoding="utf-8")
    (tmp_path / "recipe.toml").write_text(f'id = "{ft.RECIPE_ID}"\n', encoding="utf-8")
    return Recipe(ft.RECIPE_ID, tmp_path / "recipe.toml", data)


def _plan(r, root, mode):
    raw = build_plan(r, mode)
    raw["run"]["output_dir"] = str(root)
    plan = ft.task_plan(r, raw, mode)
    assert plan["execution_supported"], plan.get("unsupported_reason")
    return plan


def _freeze(r, root):
    root.mkdir(parents=True, exist_ok=True)
    (root / st.CODE_SNAPSHOT).write_bytes(pv.snapshot_bytes(pv.code_snapshot()))
    (root / st.LOCKFILE).write_bytes((r.path.parent / LOCKFILE).read_bytes())


@pytest.fixture
def fcase(tmp_path, rows, monkeypatch):
    """三 seed 合成训练/评估证据 + 冻结 root；不写仓库、不加载真实框架。"""
    r = _recipe(tmp_path)
    root = tmp_path / "suite-root"
    _freeze(r, root)
    runtime, _ = fake_runtime(r, rows)
    install_fake(monkeypatch, runtime)
    base_path = ft.run_evaluation(r, _plan(r, root / "base", "zero-shot"), zero_shot=True)
    checkpoints, evaluations = {}, {}
    for seed in SEEDS:
        effective = pv.effective_recipe(r, seed)
        train_runtime, _ = fake_runtime(effective, rows)
        install_fake(monkeypatch, train_runtime)
        training = ft.run_training(effective, _plan(effective, root / f"seed-{seed}", "full"))
        eval_runtime, _ = fake_runtime(effective, rows)
        install_fake(monkeypatch, eval_runtime)
        evaluation = ft.run_evaluation(
            effective, _plan(effective, root / f"seed-{seed}", "eval"),
            checkpoint=training.parent, zero_shot=False)
        checkpoints[seed] = training.parent
        evaluations[seed] = evaluation
    return SimpleNamespace(r=r, root=root, base=base_path, checkpoints=checkpoints,
                           evaluations=evaluations, lock=tmp_path / LOCKFILE)


def test_suite_plan_is_readonly_and_fim_specific(fcase):
    before = {p: p.stat().st_mtime_ns for p in fcase.root.rglob("*") if p.is_file()}
    plan = fs.suite_plan(fcase.r, SEEDS, fcase.root)
    assert plan["status"] == "declared" and plan["task_id"] == ft.TASK_ID
    assert plan["seeds"] == SEEDS and len(plan["commands"]) == 7
    assert plan["training_source"]["adapter_only"] is True
    assert plan["shell"] is False and plan["baseline_promotion"] == "none"
    assert not plan["resume_supported"] and "exact match" in plan["limitations"]
    assert {p: p.stat().st_mtime_ns for p in fcase.root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("seeds", [[], [1, 1, 2], [1, 2, True], [1, 2, 2**32]])
@pytest.mark.parametrize("operation", [fs.suite_plan, fs.aggregate_suite, fs.run_suite])
def test_invalid_seed_lists_fail_without_writes(fcase, seeds, operation):
    root = fcase.root.parent / "unused-root"
    with pytest.raises(mt.TaskExecutionError):
        operation(fcase.r, seeds, root)
    assert not root.exists()


def test_aggregate_suite_recomputes_three_independent_seeds(fcase):
    result = fs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    assert result == fcase.root / fs.FINAL
    data = read(result)
    assert data["kind"] == "baseline-suite" and data["status"] == "completed"
    assert data["task_id"] == ft.TASK_ID and data["baseline_promotion"] == "none"
    assert [run["train_seed"] for run in data["runs"]] == SEEDS
    assert len({run["training_evidence"]["path"] for run in data["runs"]}) == 3
    assert data["statistical_unit"] == "independent_training_seed" and data["fixed_test"] is True
    assert data["training_seed_variance_measured"] is True
    for name in fs.METRICS:
        stats = data["stats"][name]
        assert stats["trained"]["n"] == 3 and stats["paired_delta"]["n"] == 3
        assert stats["trained"]["student_t95_ci"]["df"] == 2
    assert data["acceptance"]["status"] == "not_evaluated"  # 不回填 expected/tolerance
    assert fs.validate_suite_payload(fcase.r, data, result) is None


@pytest.mark.parametrize("change", ["stats", "fingerprint", "runs", "task_id", "base_metrics", "records"])
def test_suite_payload_rejects_tampered_summary(fcase, change):
    result = fs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    data = read(result)
    if change == "stats":
        data["stats"]["fim_exact_match"]["trained"]["mean"] = 0.123
    elif change == "fingerprint":
        data["fingerprint"] = "0" * 64
    elif change == "runs":
        data["runs"] = data["runs"][:2]
    elif change == "task_id":
        data["task_id"] = "math-grpo-v1"
    elif change == "base_metrics":
        data["runs"][0]["base_metrics"]["fim_exact_match"] = 0.5
    else:
        key = next(iter(data["records"]))
        data["records"][key]["sha256"] = "0" * 64
    with pytest.raises(mt.TaskExecutionError):
        fs.validate_suite_payload(fcase.r, data, result)


def test_suite_payload_rejects_checkpoint_weight_drift(fcase):
    result = fs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    (fcase.checkpoints[43] / "adapter_model.safetensors").write_bytes(b"drifted adapter")
    with pytest.raises(mt.TaskExecutionError):
        fs.validate_suite_payload(fcase.r, read(result), result)


def test_evaluation_content_gate_accepts_real_psm_evidence_and_rejects_gold_swap(fcase):
    base = read(fcase.base)
    assert fs.evaluation_problems(fcase.r, base, fcase.base, role="base") == []
    assert fs.zero_train_validation(fcase.r, base, fcase.base) is None
    swapped = copy.deepcopy(base)
    swapped["predictions"]["sha256"] = "0" * 64
    problems = fs.evaluation_problems(fcase.r, swapped, fcase.base, role="base")
    assert problems and "FIM 评估内容拒绝" in problems[0]
    wrong_role = copy.deepcopy(base)
    wrong_role["model_role"] = "trained_checkpoint"
    with pytest.raises(mt.TaskExecutionError):
        fs.zero_train_validation(fcase.r, wrong_role, fcase.base)


def test_suite_problems_fails_closed_without_completed_evidence(fcase):
    problems = fs.suite_problems(fcase.r)
    assert problems and "baseline-suite" in problems[0]
    assert be.suite_problems(fcase.r) == problems


def fake_reload(r, plan, checkpoint=None, zero_shot=False):
    """测试专用：把包内参考证据落到 worker 固定输出目录，不加载 GPU 框架。"""
    bundle = Path(checkpoint).parent
    output = Path(plan["run"]["output_dir"])
    assert output == bundle / "reload" / "eval"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation.json", "source_manifest.json", "tokenizer_identity.json",
                 "eval_manifest.json", "predictions.json"):
        shutil.copyfile(bundle / "reference" / name, output / name)
    return output / "evaluation.json"


def mock_worker(monkeypatch, destination):
    """把离线复载子进程替换为同进程受控调用，保留 pid / offline 身份校验。"""
    child_pid, parent_pid = os.getpid() + 10000, os.getpid()
    calls = []
    monkeypatch.setattr(ft, "run_evaluation", fake_reload)
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
                fe.run_worker(destination)
            return "FIM 测试 worker 输出", ""
        process.communicate.side_effect = communicate
        return process
    monkeypatch.setattr(et.subprocess, "Popen", launch)
    return calls


def test_export_bundle_copies_adapter_and_reloads_in_isolated_worker(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "bundle"
    calls = mock_worker(monkeypatch, destination)
    result = fe.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    evidence = read(result)
    assert result == destination / fe.FINAL
    assert evidence["task_id"] == ft.TASK_ID and evidence["stage"] == "completed"
    assert evidence["seed"] == 42 and evidence["license"]["status"] == "pending"
    assert evidence["base_weights_bundled"] is False and evidence["baseline_promotion"] == "none"
    assert evidence["training_seed_variance_measured"] is False
    assert all(value["within_tolerance"] for value in evidence["comparison"].values())
    assert evidence["subprocess"]["pid"] != os.getpid() and evidence["subprocess"]["returncode"] == 0
    assert calls[0][0] == [sys.executable, str(fe.CODE_ROOT / "runner" / "export.py"),
                           "--worker", "--bundle", str(destination)]
    assert Path(calls[0][0][1]).name == "export.py"
    assert fe.validate_export_evidence(r, result) is True
    # adapter-only：包内不携带 base 权重，但显式声明冻结 base 的 revision 与量化协议。
    base = read(destination / fe.BASE_REFERENCE)
    assert base["base_weights_bundled"] is False and base["adapter_only"] is True
    assert base["model_id"] == ft.MODEL and base["revision"] == r.get("model.revision")
    assert "adapter_model.safetensors" in {p.name for p in (destination / "model").iterdir()}
    assert not list(destination.glob("model/*.bin"))
    assert (destination / "model" / "adapter_model.safetensors").read_bytes() == \
           (fcase.checkpoints[42] / "adapter_model.safetensors").read_bytes()


def test_export_evidence_rejects_tampered_reload_or_bundle(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "bundle"
    mock_worker(monkeypatch, destination)
    result = fe.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    (destination / "reload" / "eval" / "predictions.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(mt.TaskExecutionError):
        fe.validate_export_evidence(r, result)
    # 清理篡改后证据必须重新通过；说明校验是内容级而不是缓存布尔值。
    shutil.copyfile(destination / "reference" / "predictions.json",
                    destination / "reload" / "eval" / "predictions.json")
    assert fe.validate_export_evidence(r, result) is True
    (destination / "model" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(mt.TaskExecutionError):
        fe.validate_export_evidence(r, result)


def test_lock_layout_keeps_content_identity_for_repository_relative_lock(tmp_path):
    """真实 recipe 的 lockfile 带 ../../../../ 前缀；包内必须自包含且快照逐字节不变。"""
    raw = "../../../../runner/locks/llm__sft__starcoder1b-fim.lock.toml"
    bundle = tmp_path / "bundle"
    lock_bytes = b'schema_version = 1\nstatus = "frozen"\n'
    lock_path = bundle / "runner" / "locks" / "llm__sft__starcoder1b-fim.lock.toml"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(lock_bytes)
    relative, resolved, recipe_path = fe._lock_layout(raw, bundle)
    assert relative == "runner/locks/llm__sft__starcoder1b-fim.lock.toml"
    assert resolved == lock_path
    assert recipe_path.parent == bundle / fe.LOCK_ANCHOR / fe.LOCK_ANCHOR / fe.LOCK_ANCHOR / fe.LOCK_ANCHOR
    # 所有沿用 r.path.parent / execution.lockfile 的校验都会读到包内锁，而不是仓库锁。
    assert Path(os.path.abspath(recipe_path.parent / raw)) == lock_path
    data = {"id": ft.RECIPE_ID, "execution": {"task_id": ft.TASK_ID, "lockfile": raw}}
    r = Recipe(ft.RECIPE_ID, recipe_path, data)
    assert r.get("execution.lockfile") == raw  # 内容身份没有被重定位改写
    assert ft.execution_identity(r)["environment_lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest()
    assert be.frozen_environment(r) is False  # 合成最小 lock 不是完整 frozen 环境
    # 单个前导 .. 由一层虚拟锚点吸收，解析结果仍必须落在包内。
    relative2, resolved2, _ = fe._lock_layout("../locks/frozen.lock.toml", bundle)
    assert relative2 == "locks/frozen.lock.toml" and resolved2 == bundle / "locks" / "frozen.lock.toml"


def test_lock_layout_rejects_escaping_or_conflicting_paths(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for raw in ("model/adapter_model.safetensors", "recipe-snapshot.json",
                "reference/evaluation.json", "../../a/../../b.lock.toml",
                "/absolute.lock.toml", "..\\windows.lock.toml", ""):
        with pytest.raises(mt.TaskExecutionError):
            fe._lock_layout(raw, bundle)


def test_verify_bundle_rejects_unbound_extra_file(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "prepared"
    def stop(bundle):
        raise mt.TaskExecutionError("测试停止于 prepared 包")
    with monkeypatch.context() as context:
        context.setattr(et, "_launch_worker", stop)
        with pytest.raises(mt.TaskExecutionError, match="prepared"):
            fe.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    assert (destination / fe.MANIFEST).is_file() and not (destination / fe.FINAL).exists()
    (destination / "model" / "unbound.json").write_text('{}\n', encoding="utf-8")
    with pytest.raises(mt.TaskExecutionError, match="未绑定文件"):
        fe._verify_bundle(destination)
