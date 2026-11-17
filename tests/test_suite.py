"""纯标准库/CPU 隔离契约测试；合成证据与权重只写 tmp_path，不声称 GPU 实测。"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import export_task as et
from recipe_lab import math_task as mt
from recipe_lab import provenance as pv
from recipe_lab import suite_task as st
from recipe_lab.model import Recipe
from test_export import case, read, revise  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]


def write_evaluation(case, r, output, model, correct, mode, root):
    data = copy.deepcopy(read(case.evaluation))
    predictions = read(case.evaluation.parent / data["predictions_file"]["path"])
    for index, prediction in enumerate(predictions):
        prediction["accuracy"] = int(index < correct)
        answer = "2" if index < correct else "3"
        prediction["completion"] = f"<think>隔离契约测试</think><answer>{answer}</answer>"
    prediction_path = mt.save_json(output / "predictions.json", predictions)
    data.update(seed=r.get("train.seed"), fingerprint=be.fingerprint(r), recipe_sha256=mt.sha256_json(r.data),
                code_sha256=pv.file_hash(root / st.CODE_SNAPSHOT), environment_lock_sha256=pv.file_hash(case.lock),
                mode=mode, model_artifact=model, model_role=model["role"],
                metrics={"accuracy": correct / 500, "pass@1": correct / 500, "format_compliance": 1.0},
                metric_counts={"correct": correct, "format_compliant": 500, "total": 500},
                predictions={"path": "predictions.json", "sha256": mt.sha256_json(predictions), "count": 500},
                predictions_file={"path": "predictions.json", "sha256": pv.file_hash(prediction_path)})
    data["config"]["generation"]["seed"] = r.get("evaluation.seed", r.get("train.seed"))
    data["config"]["model"] = model
    data["config_sha256"] = mt.sha256_json(data["config"])
    mt.save_json(output / "manifest.json", data["split_manifest"])
    return mt.save_json(output / "evaluation.json", data)


def write_training(case, r, root):
    checkpoint = root / f"seed-{r.get('train.seed')}" / "full"
    checkpoint.mkdir(parents=True)
    for path in case.checkpoint.iterdir():
        if path.name not in ("training.json", "manifest.json"):
            (checkpoint / path.name).write_bytes(path.read_bytes())
    (checkpoint / "model.safetensors").write_bytes(f"隔离权重 seed={r.get('train.seed')}".encode())
    data = copy.deepcopy(read(case.checkpoint / "training.json"))
    data.update(seed=r.get("train.seed"), fingerprint=be.fingerprint(r), recipe_sha256=mt.sha256_json(r.data),
                code_sha256=pv.file_hash(root / st.CODE_SNAPSHOT), environment_lock_sha256=pv.file_hash(case.lock),
                model_artifact=mt.artifact_manifest(checkpoint))
    data["config"] = {**r.get("train"), **mt.config_kwargs(r, {"mode": "full"}, checkpoint)}
    data["config_sha256"] = mt.sha256_json(data["config"])
    mt.save_json(checkpoint / "manifest.json", data["split_manifest"])
    mt.save_json(checkpoint / "training.json", data)
    return checkpoint


def write_base(case, root):
    return write_evaluation(case, case.r, root / "base" / "zero-shot",
                            {"id": mt.MODEL_ID, "revision": case.r.get("model.revision"), "role": "base"},
                            300, "zero-shot", root)


def freeze(case, root):
    root.mkdir()
    (root / st.CODE_SNAPSHOT).write_bytes(pv.snapshot_bytes(pv.code_snapshot()))
    (root / st.LOCKFILE).write_bytes(case.lock.read_bytes())


@pytest.fixture
def suite(case):
    root = case.r.path.parent / "suite"
    freeze(case, root)
    write_base(case, root)
    for index, seed in enumerate(SEEDS):
        effective = pv.effective_recipe(case.r, seed)
        checkpoint = write_training(case, effective, root)
        write_evaluation(case, effective, root / f"seed-{seed}" / "eval",
                         {**mt.artifact_manifest(checkpoint), "role": "trained_checkpoint"},
                         350 + index * 50, "eval", root)
    return SimpleNamespace(case=case, r=case.r, root=root)


def test_plan_is_readonly_and_commands_have_independent_sources(case, monkeypatch):
    before, original = set(case.r.path.parent.rglob("*")), copy.deepcopy(case.r.data)
    monkeypatch.setattr(mt, "load_runtime", Mock(side_effect=AssertionError("禁止框架")))
    monkeypatch.setattr(be, "frozen_environment", Mock(side_effect=AssertionError("计划不能读 lock")))
    monkeypatch.setattr(et, "_json", Mock(side_effect=AssertionError("计划不能读证据")))
    monkeypatch.setattr(pv, "code_snapshot", Mock(side_effect=AssertionError("计划不能读快照")))
    monkeypatch.setattr(st.subprocess, "Popen", Mock(side_effect=AssertionError("禁止启动")))
    root = case.r.path.parent / "absent-root"
    plan = st.suite_plan(case.r, SEEDS, root)
    assert plan["status"] == "declared" and plan["execution_supported"] is True
    assert plan["shell"] is False and plan["fingerprint"] == be.fingerprint(case.r)
    assert len(plan["commands"]) == 7 and len(plan["runs"]) == 3
    assert plan["commands"][0]["command"] == [sys.executable, str(ROOT / "runner/zero_shot.py"),
                                                "--recipe", case.r.id, "--full", "--output-root", str(root / "base")]
    for index, seed in enumerate(SEEDS):
        train, evaluation = plan["commands"][1 + 2 * index:3 + 2 * index]
        assert train["command"] == [sys.executable, str(ROOT / "runner/train.py"), "--recipe", case.r.id,
                                    "--full", "--output-root", str(root / f"seed-{seed}"), "--train-seed", str(seed)]
        assert evaluation["command"][-4:] == ["--train-seed", str(seed), "--checkpoint", str(root / f"seed-{seed}/full")]
        assert evaluation["command"][1] == str(ROOT / "runner/eval.py")
        assert plan["runs"][index]["fingerprint"] == be.fingerprint(pv.effective_recipe(case.r, seed))
    assert plan["data_seed"] == 17 and plan["evaluation_seed"] == 23
    assert case.r.data == original and set(case.r.path.parent.rglob("*")) == before


@pytest.mark.parametrize("seeds", [[], [1], [1, 2], [1, 1, 2], [1, 2, True], [-1, 1, 2],
                                    [1, 2, 2**32], [1, 2, 3.0], None])
@pytest.mark.parametrize("operation", [st.suite_plan, st.aggregate_suite, st.run_suite])
def test_invalid_seed_lists_fail_without_writes(case, seeds, operation):
    root = case.r.path.parent / "unused"
    with pytest.raises(mt.TaskExecutionError):
        operation(case.r, seeds, root)
    assert not root.exists()


def test_seed_boundaries_and_implicit_eval_seed_remain_fixed(case):
    original = copy.deepcopy(case.r.data)
    case.r.data["evaluation"].pop("seed")
    data = copy.deepcopy(case.r.data)
    plan = st.suite_plan(case.r, [0, 1, 2**32 - 1], case.r.path.parent / "unused")
    assert plan["evaluation_seed"] == 42
    for seed in [0, 1, 2**32 - 1]:
        effective = pv.effective_recipe(case.r, seed)
        assert effective.get("train.seed") == seed and effective.get("evaluation.seed") == 42
        assert effective.get("train.data_seed") == 17
    assert case.r.data == data and original["train"] == data["train"]


@pytest.mark.parametrize("operation", [st.suite_plan, st.aggregate_suite, st.run_suite])
def test_output_root_must_be_absolute(case, operation):
    with pytest.raises(mt.TaskExecutionError, match="ABS"):
        operation(case.r, SEEDS, "relative")


def test_aggregate_binds_all_content_and_computes_seed_statistics(suite, monkeypatch):
    before = {p.relative_to(suite.root): p.read_bytes() for p in suite.root.rglob("*") if p.is_file()}
    original = copy.deepcopy(suite.r.data)
    monkeypatch.setattr(mt, "load_runtime", Mock(side_effect=AssertionError("禁止框架")))
    monkeypatch.setattr(st.subprocess, "Popen", Mock(side_effect=AssertionError("聚合禁止启动进程")))
    result = st.aggregate_suite(suite.r, SEEDS, suite.root)
    data = read(result)
    assert result == suite.root / st.FINAL and data["kind"] == "baseline-suite" and data["status"] == "completed"
    assert data["fingerprint"] == be.fingerprint(suite.r) and data["baseline_promotion"] == "none"
    assert data["acceptance"]["status"] == "not_evaluated" and "verified" not in json.dumps(data["acceptance"])
    assert data["code_sha256"] == hashlib.sha256(pv.snapshot_bytes(pv.code_snapshot())).hexdigest()
    assert data["environment_lock_sha256"] == pv.file_hash(suite.case.lock)
    assert data["statistical_unit"] == "independent_training_seed" and data["fixed_test"] is True
    assert "不能作为业务显著性证据" in data["limitations"]
    checkpoints = set()
    for run, seed in zip(data["runs"], SEEDS):
        assert run["train_seed"] == seed and run["kind"] == "paired-evaluation" and run["training_completed"] is True
        assert run["sample_count"] == 500 and run["split_sha256"] == data["split_sha256"]
        for key in ("training_evidence", "base_evaluation", "trained_evaluation", "predictions", "base_predictions"):
            assert be.hashed_file(suite.root, run[key])
        checkpoints.add(run["model_artifact"]["sha256"])
    assert len(checkpoints) == 3
    for record in data["records"].values():
        assert be.hashed_file(suite.root, record)
    stats = data["stats"]["accuracy"]
    for side, mean in (("trained", 0.8), ("paired_delta", 0.2)):
        s = stats[side]
        assert s["mean"] == pytest.approx(mean)
        assert s["sample_variance"] == pytest.approx(0.01)
        assert s["std"] == pytest.approx(0.1) and s["sem"] == pytest.approx(math.sqrt(0.01 / 3))
        ci = s["student_t95_ci"]
        assert ci["df"] == 2 and ci["critical_df"] == 2 and ci["critical_value"] == st.T95[2]
        assert ci["lower"] == pytest.approx(mean - st.T95[2] * s["sem"])
        assert ci["upper"] == pytest.approx(mean + st.T95[2] * s["sem"])
    assert all((suite.root / p).read_bytes() == content for p, content in before.items())
    assert {p.relative_to(suite.root) for p in suite.root.rglob("*") if p.is_file()} == set(before) | {Path(st.FINAL)}
    assert suite.r.data == original
    # 未声明 expected / tolerance 时，即使测量完成仍不能满足 B2。
    audit_data = copy.deepcopy(suite.r.data)
    audit_data["execution"]["evidence"] = str(result)
    audit = Recipe(suite.r.id, suite.r.path, audit_data)
    assert any("期望" in p or "容差" in p for p in be.suite_problems(audit))


def test_implicit_evaluation_seed_is_fixed_in_actual_aggregated_protocol(case):
    case.r.data["evaluation"].pop("seed")
    original = copy.deepcopy(case.r.data)
    root = case.r.path.parent / "implicit-eval-suite"
    freeze(case, root)
    write_base(case, root)
    for index, seed in enumerate(SEEDS):
        effective = pv.effective_recipe(case.r, seed)
        checkpoint = write_training(case, effective, root)
        write_evaluation(case, effective, root / f"seed-{seed}/eval",
                         {**mt.artifact_manifest(checkpoint), "role": "trained_checkpoint"},
                         350 + index * 50, "eval", root)
    data = read(st.aggregate_suite(case.r, SEEDS, root))
    assert data["fingerprint"] == be.fingerprint(case.r)
    for run in data["runs"]:
        training = be.linked_json(root, run["training_evidence"])
        base = be.linked_json(root, run["base_evaluation"])
        trained = be.linked_json(root, run["trained_evaluation"])
        assert training["seed"] == trained["seed"] == run["train_seed"]
        assert base["config"]["generation"]["seed"] == trained["config"]["generation"]["seed"] == 42
        assert training["data_seed"] == trained["data_seed"] == base["data_seed"] == 17
        assert training["fingerprint"] == trained["fingerprint"] == be.fingerprint(pv.effective_recipe(case.r, run["train_seed"]))
    assert case.r.data == original


def test_metrics_recomputed_even_when_task_id_is_implicit(case):
    case.r.data["execution"].pop("task_id")
    root = case.r.path.parent / "implicit-task-suite"
    freeze(case, root)
    path = write_base(case, root)
    revise(path, lambda data: data["metrics"].update(accuracy=0.1))
    with pytest.raises(mt.TaskExecutionError, match="重算"):
        st.aggregate_suite(case.r, SEEDS, root)
    assert not (root / st.FINAL).exists()


def test_stats_large_df_is_conservative_not_normal():
    values = [i / 100 for i in range(40)]
    stats = st._stats(values)
    ci = stats["student_t95_ci"]
    assert ci["df"] == 39 and ci["critical_df"] == 30
    assert ci["critical_value"] == st.T95[30] > 1.96 and ci["conservative_df30"] is True
    assert st._stats([1, 1, 1])["student_t95_ci"]["lower"] == 1
    with pytest.raises(mt.TaskExecutionError):
        st._stats([1, 2])


@pytest.mark.parametrize("expected,status", [(0.8, "passed"), (0.2, "failed")])
def test_acceptance_is_separate_and_never_refills_expected(suite, expected, status):
    suite.r.data["evaluation"]["expected"] = {n: expected if n != "format_compliance" else 1
                                              for n in suite.r.get("evaluation.metrics")}
    suite.r.data["evaluation"]["tolerance"] = {n: 1e-9 for n in suite.r.get("evaluation.metrics")}
    original = copy.deepcopy(suite.r.data)
    data = read(st.aggregate_suite(suite.r, SEEDS, suite.root))
    assert data["status"] == "completed" and data["acceptance"]["status"] == status
    assert data["baseline_promotion"] == "none" and suite.r.data == original
    if status == "passed":
        audit_data = copy.deepcopy(suite.r.data)
        audit_data["execution"]["evidence"] = str(suite.root / st.FINAL)
        assert be.suite_problems(Recipe(suite.r.id, suite.r.path, audit_data)) == []


@pytest.mark.parametrize("change", ["code", "stats", "fingerprint", "records", "same_checkpoint", "nonempty_gold"])
def test_release_gate_does_not_trust_bypassed_aggregator(suite, change):
    suite.r.data["evaluation"].update(expected={"accuracy": 0.8, "pass@1": 0.8, "format_compliance": 1.0},
                                      tolerance={n: 1e-9 for n in suite.r.get("evaluation.metrics")})
    path = st.aggregate_suite(suite.r, SEEDS, suite.root)
    data = read(path)
    suite.r.data["execution"]["evidence"] = str(path)
    assert not be.suite_problems(suite.r)
    if change == "stats":
        data["stats"]["accuracy"]["trained"]["mean"] = 1.0
    elif change == "records":
        data["records"].pop("seed-43/full/model.safetensors")
    elif change == "same_checkpoint":
        data["runs"][1]["training_evidence"] = data["runs"][0]["training_evidence"]
    elif change == "nonempty_gold":
        target = suite.root / data["runs"][0]["trained_evaluation"]["path"]
        update_predictions(target, lambda predictions: predictions[0].update(solution="a nonempty substituted gold"))
        replacement = et._record(target, suite.root)
        data["runs"][0]["trained_evaluation"] = replacement
        data["records"][replacement["path"]] = replacement
    else:
        target = suite.root / data["runs"][0]["training_evidence"]["path"]
        revise(target, lambda value: value.update({"code_sha256" if change == "code" else "fingerprint": "e" * 64}))
        replacement = et._record(target, suite.root)
        data["runs"][0]["training_evidence"] = replacement
        data["records"][replacement["path"]] = replacement
    mt.save_json(path, data)
    assert be.suite_problems(suite.r)


def update_predictions(path, change):
    data = read(path)
    prediction_path = path.parent / data["predictions_file"]["path"]
    predictions = read(prediction_path)
    change(predictions)
    mt.save_json(prediction_path, predictions)
    data["predictions_file"]["sha256"] = pv.file_hash(prediction_path)
    data["predictions"]["sha256"] = mt.sha256_json(predictions)
    mt.save_json(path, data)


@pytest.mark.parametrize("corruption", ["missing_training", "missing_predictions", "weight", "smoke", "mock", "simulated", "dry_run",
    "training_fingerprint", "eval_fingerprint", "base_fingerprint", "training_seed", "data_seed", "revision", "task", "schema",
    "runtime", "steps", "config", "default_config", "seed_config_type", "lock_hash", "code_hash", "base_code_hash", "generation", "extra_generation",
    "base_artifact", "eval_artifact", "mode", "metrics", "prediction_hash", "prediction_content_hash", "gold", "prompt",
    "missing_completion", "base_gold", "manifest_file", "training_manifest", "protocol", "base_generation", "counts"])
def test_content_tampering_and_partial_inputs_are_rejected(suite, corruption):
    training_path = suite.root / "seed-43/full/training.json"
    eval_path = suite.root / "seed-43/eval/evaluation.json"
    base_path = suite.root / "base/zero-shot/evaluation.json"
    if corruption == "missing_training":
        training_path.unlink()
    elif corruption == "missing_predictions":
        (eval_path.parent / "predictions.json").unlink()
    elif corruption == "weight":
        (training_path.parent / "model.safetensors").write_bytes(b"changed")
    elif corruption in ("gold", "prompt", "missing_completion", "base_gold"):
        def change(predictions):
            if corruption in ("gold", "base_gold"):
                predictions[0]["solution"] = "unmatched nonempty gold"
            elif corruption == "prompt":
                predictions[0]["prompt"] = mt.make_prompt("不同的问题")
            else:
                predictions[0].pop("completion")
        update_predictions(base_path if corruption == "base_gold" else eval_path, change)
    elif corruption in ("manifest_file", "training_manifest"):
        path = (training_path if corruption == "training_manifest" else eval_path).parent / "manifest.json"
        revise(path, lambda value: value.update(dataset_revision="x" * 40))
    else:
        training_changes = {"smoke", "mock", "simulated", "dry_run", "training_fingerprint", "training_seed", "data_seed",
                            "revision", "runtime", "steps", "config", "default_config", "seed_config_type", "lock_hash", "code_hash", "task", "schema"}
        path = training_path if corruption in training_changes else base_path if corruption.startswith("base_") else eval_path
        def alter(data):
            if corruption in ("mock", "simulated", "dry_run"):
                data[corruption] = True
            elif "fingerprint" in corruption:
                data["fingerprint"] = "e" * 64
            elif corruption in ("smoke", "mode"):
                data["mode"] = "smoke"
            elif corruption == "training_seed":
                data["seed"] = 42
            elif corruption == "data_seed":
                data["data_seed"] = 18
            elif corruption == "revision":
                data["model_revision"] = "e" * 40
            elif corruption == "runtime":
                data["runtime"]["gpu"] = "mock-GPU"
            elif corruption == "steps":
                data["steps"] = 149
            elif corruption == "task":
                data["task_id"] = "other"
            elif corruption == "schema":
                data["schema_version"] = True
            elif corruption == "seed_config_type":
                data["config"]["seed"] = float(data["config"]["seed"])
                data["config_sha256"] = mt.sha256_json(data["config"])
            elif corruption in ("config", "default_config"):
                data["config"]["learning_rate" if corruption == "config" else "per_device_train_batch_size"] = 8
                data["config_sha256"] = mt.sha256_json(data["config"])
            elif corruption in ("lock_hash", "code_hash", "base_code_hash"):
                data["environment_lock_sha256" if corruption == "lock_hash" else "code_sha256"] = "e" * 64
            elif corruption in ("generation", "base_generation", "extra_generation"):
                if corruption == "extra_generation":
                    data["config"]["generation"]["temperature"] = 0.3
                else:
                    data["config"]["generation"]["seed"] += 1
                data["config_sha256"] = mt.sha256_json(data["config"])
            elif corruption in ("base_artifact", "eval_artifact"):
                data["model_artifact"]["role"] = "other"
            elif corruption == "metrics":
                data["metrics"]["accuracy"] = 0.0
            elif corruption == "prediction_hash":
                data["predictions_file"]["sha256"] = "e" * 64
            elif corruption == "prediction_content_hash":
                data["predictions"]["sha256"] = "e" * 64
            elif corruption == "protocol":
                data["evaluation_protocol"] = "other"
            elif corruption == "counts":
                data["metric_counts"]["correct"] += 1
        revise(path, alter)
    with pytest.raises(mt.TaskExecutionError):
        st.aggregate_suite(suite.r, SEEDS, suite.root)
    assert not (suite.root / st.FINAL).exists()


@pytest.mark.parametrize("corruption", ["snapshot_missing", "snapshot_reformatted", "snapshot_changed", "lock_copy", "current_lock", "current_code"])
def test_aggregate_only_never_fabricates_history(suite, monkeypatch, corruption):
    snapshot = suite.root / st.CODE_SNAPSHOT
    if corruption == "snapshot_missing":
        snapshot.unlink()
    elif corruption == "snapshot_reformatted":
        mt.save_json(snapshot, read(snapshot))
    elif corruption == "snapshot_changed":
        revise(snapshot, lambda value: value.update(kind="different"))
    elif corruption in ("lock_copy", "current_lock"):
        path = suite.root / st.LOCKFILE if corruption == "lock_copy" else suite.case.lock
        path.write_bytes(path.read_bytes() + "\n# 隔离漂移\n".encode("utf-8"))
    else:
        actual = pv.code_snapshot()
        actual["files"][0]["content"] += "\n# 隔离代码漂移\n"
        monkeypatch.setattr(pv, "code_snapshot", lambda: actual)
    before = {p: p.read_bytes() for p in suite.root.rglob("*") if p.is_file()}
    with pytest.raises(mt.TaskExecutionError, match="snapshot|lock|快照"):
        st.aggregate_suite(suite.r, SEEDS, suite.root)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not (suite.root / st.FINAL).exists()


@pytest.mark.parametrize("change", ["ranges", "selection", "indices", "task", "count"])
def test_strict_1500_manifest_protocol(suite, change):
    manifest = copy.deepcopy(read(suite.root / "base/zero-shot/manifest.json"))
    if change == "ranges":
        manifest["ranges"]["test"]["start"] = 999
    elif change == "selection":
        manifest["selection"] = "shuffle"
    elif change == "indices":
        manifest["splits"]["test"]["samples"][0]["source_index"] = 999
        manifest["splits"]["test"]["sha256"] = mt.sha256_json(manifest["splits"]["test"]["samples"])
    elif change == "task":
        manifest["task_id"] = "other"
    else:
        manifest["splits"]["test"]["samples"].pop()
        manifest["splits"]["test"]["count"] = 499
        manifest["splits"]["test"]["sha256"] = mt.sha256_json(manifest["splits"]["test"]["samples"])
    manifest["sha256"] = mt.sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    with pytest.raises(mt.TaskExecutionError):
        st._manifest(suite.r, manifest)


@pytest.mark.parametrize("kind", ["root_symlink", "ancestor_symlink", "weight_symlink", "prediction_escape", "checkpoint_symlink"])
def test_local_path_containment_and_symlinks_are_rejected(suite, kind):
    root = suite.root
    if kind == "root_symlink":
        root = suite.r.path.parent / "linked"
        root.symlink_to(suite.root, target_is_directory=True)
    elif kind == "ancestor_symlink":
        linked = suite.r.path.parent / "linked"
        linked.symlink_to(suite.r.path.parent, target_is_directory=True)
        root = linked / suite.root.name
    elif kind == "weight_symlink":
        weight = root / "seed-43/full/model.safetensors"
        weight.unlink()
        weight.symlink_to(suite.case.checkpoint / "model.safetensors")
    elif kind == "checkpoint_symlink":
        checkpoint = root / "seed-43/full"
        checkpoint.rename(root / "moved-checkpoint")
        checkpoint.symlink_to(root / "moved-checkpoint", target_is_directory=True)
    else:
        revise(root / "seed-43/eval/evaluation.json", lambda data: data["predictions_file"].update(path="../escape.json"))
    with pytest.raises(mt.TaskExecutionError):
        st.aggregate_suite(suite.r, SEEDS, root)
    assert not (suite.root / st.FINAL).exists()


def fake_commands(case, monkeypatch, root, after=None):
    calls = []
    def factory(command, **options):
        calls.append((command, options))
        process = Mock(pid=os.getpid() + 10000 + len(calls), returncode=0)
        process.poll.return_value = 0
        def wait(timeout):
            assert timeout == st.COMMAND_TIMEOUT_SECONDS
            assert options["stdout"].fileno() > 0 and options["stderr"].fileno() > 0
            options["stdout"].write(b"isolated command stdout\n")
            output = Path(command[command.index("--output-root") + 1])
            if Path(command[1]).name == "zero_shot.py":
                write_base(case, root)
            else:
                seed = int(command[command.index("--train-seed") + 1])
                effective = pv.effective_recipe(case.r, seed)
                if Path(command[1]).name == "train.py":
                    write_training(case, effective, root)
                else:
                    checkpoint = Path(command[command.index("--checkpoint") + 1])
                    write_evaluation(case, effective, output / "eval",
                                     {**mt.artifact_manifest(checkpoint), "role": "trained_checkpoint"},
                                     350 + (seed - 42) * 50, "eval", root)
            if after:
                after(len(calls), root)
            return 0
        process.wait.side_effect = wait
        return process
    monkeypatch.setattr(st.subprocess, "Popen", factory)
    return calls


def test_execution_freezes_bytes_runs_once_and_aggregates_only_at_end(case, monkeypatch):
    root = case.r.path.parent / "run"
    before = case.r.path.read_bytes(), copy.deepcopy(case.r.data)
    calls = fake_commands(case, monkeypatch, root)
    aggregate = st.aggregate_suite
    observed = []
    def capture(r, seeds, output):
        assert len(calls) == 7 and read(root / st.STATE)["status"] == "commands_completed"
        observed.append(output)
        return aggregate(r, seeds, output)
    monkeypatch.setattr(st, "aggregate_suite", capture)
    result = st.run_suite(case.r, SEEDS, root)
    assert observed == [root] and read(result)["status"] == "completed"
    assert len(calls) == 7 and sum(Path(c[0][1]).name == "zero_shot.py" for c in calls) == 1
    assert (root / st.CODE_SNAPSHOT).read_bytes() == pv.snapshot_bytes(pv.code_snapshot())
    assert (root / st.LOCKFILE).read_bytes() == case.lock.read_bytes()
    state = read(root / st.STATE)
    assert len(state["subprocesses"]) == 7
    for command, options in calls:
        assert command[0] == sys.executable and Path(command[1]).is_absolute()
        assert options["shell"] is False and options["cwd"] == ROOT
        assert options["stdout"] != subprocess.PIPE and options["stderr"] != subprocess.PIPE
        assert options["stdout"].closed and options["stderr"].closed
        assert options["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert options["env"]["TOKENIZERS_PARALLELISM"] == "false"
        # 训练套件不强制offline，已冻结revision可按需下载；仅export复载强制offline。
        assert options["env"].get("HF_HUB_OFFLINE") == os.environ.get("HF_HUB_OFFLINE")
    assert len(list((root / "logs").glob("*.log"))) == 14
    assert before == (case.r.path.read_bytes(), case.r.data)
    with pytest.raises(mt.TaskExecutionError, match="非空"):
        st.run_suite(case.r, SEEDS, root)


@pytest.mark.parametrize("failure", ["spawn", "timeout", "wait_error", "nonzero", "interrupt"])
def test_failed_subprocess_is_terminated_logs_retained_and_no_completed_suite(case, monkeypatch, failure):
    root = case.r.path.parent / "failed-run"
    process = Mock(pid=12345, returncode=3 if failure == "nonzero" else 0)
    process.poll.return_value = None
    if failure == "spawn":
        factory = Mock(side_effect=OSError("隔离启动失败"))
    else:
        if failure == "timeout":
            process.wait.side_effect = [subprocess.TimeoutExpired("test-only", 1), 0]
        elif failure == "wait_error":
            process.wait.side_effect = [subprocess.SubprocessError("隔离等待失败"), 0]
        elif failure == "interrupt":
            process.wait.side_effect = [KeyboardInterrupt(), 0]
        factory = Mock(return_value=process)
    monkeypatch.setattr(st.subprocess, "Popen", factory)
    with pytest.raises((OSError, subprocess.SubprocessError, mt.TaskExecutionError, KeyboardInterrupt)):
        st.run_suite(case.r, SEEDS, root)
    assert read(root / st.STATE)["status"] == "failed" and not (root / st.FINAL).exists()
    assert (root / "logs/00.stdout.log").exists() and (root / "logs/00.stderr.log").exists()
    if failure in ("timeout", "wait_error", "interrupt"):
        process.kill.assert_called_once()
        assert process.wait.call_count == 2
    count = factory.call_count
    with pytest.raises(mt.TaskExecutionError, match="非空"):
        st.run_suite(case.r, SEEDS, root)
    assert factory.call_count == count


@pytest.mark.parametrize("drift", ["code", "lock", "snapshot", "recipe"])
def test_each_command_postcheck_stops_on_drift(case, monkeypatch, drift):
    root = case.r.path.parent / "drift-run"
    original_snapshot = pv.code_snapshot()
    def after(index, root):
        assert index == 1
        if drift == "code":
            changed = copy.deepcopy(original_snapshot)
            changed["files"][0]["content"] += "\n# 隔离漂移\n"
            monkeypatch.setattr(pv, "code_snapshot", lambda: changed)
        else:
            path = case.lock if drift == "lock" else root / st.CODE_SNAPSHOT if drift == "snapshot" else case.r.path
            path.write_bytes(path.read_bytes() + b"\n")
    calls = fake_commands(case, monkeypatch, root, after)
    with pytest.raises(mt.TaskExecutionError, match="snapshot|lock|Recipe"):
        st.run_suite(case.r, SEEDS, root)
    assert len(calls) == 1 and read(root / st.STATE)["status"] == "failed"
    assert not (root / st.FINAL).exists() and (root / "logs/00.stdout.log").exists()


def test_precheck_happens_before_any_child(case, monkeypatch):
    root = case.r.path.parent / "precheck-run"
    original = st._guard
    checks = []
    def fail(r, output, hashes):
        checks.append(output)
        raise mt.TaskExecutionError("隔离 precheck 漂移")
    monkeypatch.setattr(st, "_guard", fail)
    launch = Mock(side_effect=AssertionError("precheck 前不允许启动"))
    monkeypatch.setattr(st.subprocess, "Popen", launch)
    with pytest.raises(mt.TaskExecutionError, match="precheck"):
        st.run_suite(case.r, SEEDS, root)
    launch.assert_not_called()
    assert checks == [root] and read(root / st.STATE)["status"] == "failed"
    monkeypatch.setattr(st, "_guard", original)


@pytest.mark.parametrize("invalid", ["unsupported", "frozen", "revision", "blocked", "baseline_passes", "budget", "batch", "unknown_train"])
def test_full_preflight_fails_before_root_creation(case, monkeypatch, invalid):
    root = case.r.path.parent / "unused"
    if invalid == "unsupported":
        case.r = Recipe("other/route", case.r.path, case.r.data)
        assert st.suite_plan(case.r, SEEDS, root)["execution_supported"] is False
    elif invalid == "frozen":
        case.lock.write_text('status = "template"\n', encoding="utf-8")
    elif invalid == "revision":
        case.r.data["model"]["revision"] = "main"
    elif invalid == "blocked":
        case.r.data["decision"] = {"gate": "blocked"}
    elif invalid == "baseline_passes":
        case.r.data["baseline"] = {"result": "passes"}
    elif invalid == "budget":
        case.r.data["train"]["max_steps"] = 151
    elif invalid == "batch":
        case.r.data["train"]["per_device_train_batch_size"] = 5
    else:
        case.r.data["train"]["unimplemented"] = 1
    monkeypatch.setattr(st.subprocess, "Popen", Mock(side_effect=AssertionError("禁止启动")))
    with pytest.raises(mt.TaskExecutionError):
        st.run_suite(case.r, SEEDS, root)
    assert not root.exists()


def test_nonempty_root_is_preserved(case, monkeypatch):
    root = case.r.path.parent / "nonempty"
    root.mkdir()
    keep = root / "keep.txt"
    keep.write_bytes(b"preserve")
    monkeypatch.setattr(st.subprocess, "Popen", Mock(side_effect=AssertionError("禁止启动")))
    with pytest.raises(mt.TaskExecutionError, match="非空"):
        st.run_suite(case.r, SEEDS, root)
    assert list(root.iterdir()) == [keep] and keep.read_bytes() == b"preserve"


def test_failed_state_with_all_completed_inputs_allows_explicit_aggregation(suite):
    mt.save_json(suite.root / st.STATE, {"kind": "suite-execution", "status": "failed"})
    suite.r.data["decision"] = {"gate": "blocked"}
    result = st.aggregate_suite(suite.r, SEEDS, suite.root)
    assert read(result)["status"] == "completed"
    assert read(suite.root / st.STATE)["status"] == "failed"
    with pytest.raises(mt.TaskExecutionError, match="已存在"):
        st.aggregate_suite(suite.r, SEEDS, suite.root)


def test_aggregation_failure_after_children_is_failed_not_completed(case, monkeypatch):
    root = case.r.path.parent / "bad-aggregate-run"
    def after(index, root):
        if index == 7:
            (root / "seed-42/full/model.safetensors").write_bytes(b"changed after training")
    calls = fake_commands(case, monkeypatch, root, after)
    with pytest.raises(mt.TaskExecutionError):
        st.run_suite(case.r, SEEDS, root)
    assert len(calls) == 7 and read(root / st.STATE)["status"] == "failed"
    assert not (root / st.FINAL).exists()


def test_aggregate_rechecks_all_bound_inputs_before_write(suite, monkeypatch):
    original = st._guard
    def change(r, root, hashes):
        original(r, root, hashes)
        (root / "seed-42/full/model.safetensors").write_bytes(b"changed during aggregate")
    monkeypatch.setattr(st, "_guard", change)
    with pytest.raises(mt.TaskExecutionError, match="漂移"):
        st.aggregate_suite(suite.r, SEEDS, suite.root)
    assert not (suite.root / st.FINAL).exists()


def test_cli_modes_dispatch_and_defaults_are_readonly(case, monkeypatch):
    spec = importlib.util.spec_from_file_location("suite_cli_test", ROOT / "runner/suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "find_recipe", lambda recipe_id: case.r)
    dry, full, aggregate = Mock(return_value={"status": "declared"}), Mock(return_value=case.r.path), Mock(return_value=case.r.path)
    monkeypatch.setattr(module, "suite_plan", dry)
    monkeypatch.setattr(module, "run_suite", full)
    monkeypatch.setattr(module, "aggregate_suite", aggregate)
    argv = [str(ROOT / "runner/suite.py"), "--recipe", case.r.id, "--output-root", str(case.r.path.parent / "unused")]
    for mode in ([], ["--dry-run"], ["--full"], ["--aggregate-only"]):
        monkeypatch.setattr(sys, "argv", argv + mode)
        assert module.main() == 0
    assert dry.call_count == 2 and full.call_count == 1 and aggregate.call_count == 1
    assert dry.call_args.args[1] == SEEDS
    monkeypatch.setattr(sys, "argv", argv + ["--seeds", "0", "1", "2", "--full", "--aggregate-only"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_cli_in_new_process_is_framework_free_uses_recipe_root_and_writes_nothing(tmp_path):
    directory = tmp_path / "recipes" / "isolated"
    directory.mkdir(parents=True)
    (directory / "recipe.toml").write_text('id = "isolated/recipe"\n', encoding="utf-8")
    program = '''import importlib.abc, runpy, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trl', 'transformers', 'datasets', 'math_verify'}:
            raise AssertionError('隔离测试禁止框架: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
sys.argv = [sys.argv[1], '--recipe', 'isolated/recipe', '--output-root', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    output = tmp_path / "unused"
    result = subprocess.run([sys.executable, "-B", "-c", program, str(ROOT / "runner/suite.py"), str(output)],
                            cwd=tmp_path, env={**os.environ, "RECIPE_LAB_ROOT": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["status"] == "declared" and plan["execution_supported"] is False and plan["seeds"] == SEEDS
    assert all(Path(step["command"][1]).parent == ROOT / "runner" for step in plan["commands"])
    assert not output.exists() and not list(tmp_path.rglob("*.json"))
