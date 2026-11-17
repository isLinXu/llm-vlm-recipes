"""纯 CPU 隔离契约测试；所有合成权重、分数和环境证据仅落 tmp_path。"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
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
from recipe_lab.model import Recipe
from recipe_lab.provenance import code_hash

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def revise(path, change):
    value = read(path)
    change(value)
    mt.save_json(path, value)


@pytest.fixture
def case(tmp_path):
    data = {
        "id": mt.RECIPE_ID, "source_sha": "c" * 40,
        "method": {"family": "grpo", "trainer": "GRPOTrainer"},
        "model": {"base": mt.MODEL_ID, "revision": "a" * 40, "modality": "llm"},
        "dataset": {"name": mt.DATASET_ID, "revision": "b" * 40},
        "dependencies": {"torch": "2.5.1", "transformers": "4.48.1", "datasets": "3.2.0", "trl": "0.14.0"},
        "rewards": {"functions": ["format_reward", "accuracy_reward"]},
        "train": {"learning_rate": 1e-5, "gradient_accumulation_steps": 16, "num_train_epochs": 1,
                  "max_completion_length": 64, "num_generations": 4, "remove_unused_columns": False,
                  "seed": 42, "data_seed": 17},
        "evaluation": {"metrics": ["accuracy", "pass@1", "format_compliance"], "seed": 23,
                       "export_tolerance": {"accuracy": 0.0, "pass@1": 0.0, "format_compliance": 0.0}},
        "budget": {"smoke_steps": 5, "full_steps": 150},
        "execution": {"task_id": mt.TASK_ID, "lockfile": "source.lock.toml"},
        "data": {"seed": 19, "preserve": {"columns": ["problem", "solution"]}},
    }
    r = Recipe(mt.RECIPE_ID, tmp_path / "recipe.toml", data)
    r.path.write_text('id = "' + r.id + '"\n', encoding="utf-8")
    lock = tmp_path / "source.lock.toml"
    lock.write_text('schema_version = 1\nstatus = "frozen"\nrecipe_id = "' + r.id + '"\n'
                    'model_id = "' + mt.MODEL_ID + '"\nmodel_revision = "' + "a" * 40 + '"\n'
                    'dataset_id = "' + mt.DATASET_ID + '"\ndataset_revision = "' + "b" * 40 + '"\n'
                    'source_sha = "' + "c" * 40 + '"\nrequirements = '
                    '["torch==2.5.1", "transformers==4.48.1", "datasets==3.2.0", "trl==0.14.0"]\n'
                    '[runtime]\npython = "3.11.0"\ncuda = "12.4"\ndriver = "test-driver"\ngpu = "test-only-GPU"\n',
                    encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"):
        (checkpoint / name).write_text('{}\n', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"isolated test-only weights")
    (checkpoint / "vocab.txt").write_text("isolated test-only vocabulary\n", encoding="utf-8")
    rows = [{"problem": f"隔离测试问题{i}", "solution": "2"} for i in range(1500)]
    splits, manifest = mt.prepare_splits(rows, parse=lambda text: [text])
    manifest.pop("sha256")
    manifest["dataset_revision"] = r.get("dataset.revision")
    manifest["sha256"] = mt.sha256_json(manifest)
    artifact = mt.artifact_manifest(checkpoint)
    runtime = {"python": "3.11.0", "cuda": "12.4", "gpu": "test-only-GPU", "world_size": 1,
               "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"}
    common = {
        "schema_version": 1, "stage": "completed", "task_id": mt.TASK_ID, "recipe_id": r.id,
        "model_id": mt.MODEL_ID, "dataset_id": mt.DATASET_ID, "model_revision": r.get("model.revision"),
        "dataset_revision": r.get("dataset.revision"), "seed": 42, "data_seed": 17,
        "fingerprint": be.fingerprint(r), "recipe_sha256": mt.sha256_json(r.data),
        "split_manifest": manifest, "runtime": runtime,
        "code_sha256": code_hash(), "environment_lock_sha256": et._hash(lock),
    }
    config = {**r.get("train"), "max_steps": 150, "per_device_train_batch_size": 4}
    training = {**common, "kind": "training", "mode": "full", "model_artifact": artifact,
                "steps": 150, "max_steps": 150, "config": config, "config_sha256": mt.sha256_json(config)}
    mt.save_json(checkpoint / "training.json", training)
    evaluation_dir = tmp_path / "source-evaluation"
    predictions = [{"sample_id": row["sample_id"], "content_sha256": row["content_sha256"],
                    "prompt": row["prompt"], "solution": row["solution"],
                    "completion": "<think>隔离测试</think><answer>2</answer>",
                    "accuracy": 1, "format_compliance": 1} for row in splits["test"]]
    predictions_path = mt.save_json(evaluation_dir / "nested" / "predictions.json", predictions)
    model = {**artifact, "role": "trained_checkpoint"}
    eval_config = {"task_id": mt.TASK_ID, "system_prompt": mt.SYSTEM_PROMPT, "model": model,
                   "dataset_revision": r.get("dataset.revision"), "test_sha256": manifest["splits"]["test"]["sha256"],
                   "generation": {"seed": 23, "do_sample": False, "num_beams": 1, "max_new_tokens": 64,
                                  "pad_token_id": 0, "eos_token_id": 0, "use_cache": True}}
    evaluation = {**common, "kind": "evaluation", "mode": "eval", "config": eval_config,
                  "config_sha256": mt.sha256_json(eval_config), "model_artifact": model,
                  "model_role": "trained_checkpoint", "evaluation_protocol": "single_checkpoint_evaluation",
                  "sample_count": 500, "metrics": {"accuracy": 1.0, "pass@1": 1.0, "format_compliance": 1.0},
                  "predictions": {"path": "nested/predictions.json", "sha256": mt.sha256_json(predictions), "count": 500},
                  "predictions_file": {"path": "nested/predictions.json", "sha256": et._hash(predictions_path)}}
    mt.save_json(evaluation_dir / "manifest.json", manifest)
    evaluation_path = mt.save_json(evaluation_dir / "evaluation.json", evaluation)
    return SimpleNamespace(r=r, checkpoint=checkpoint, evaluation=evaluation_path, lock=lock,
                           destination=tmp_path / "bundle", runtime=runtime)


def fake_evaluation(r, plan, *, checkpoint):
    """仅注入测试的逐题产物；生产代码不存在该替代路径。"""
    checkpoint = Path(checkpoint)
    bundle = checkpoint.parent
    output = Path(plan["run"]["output_dir"])
    assert output == bundle / "reload" / "eval"
    assert (r.path.parent / r.get("execution.lockfile")).is_file()
    evidence = read(bundle / "reference" / "evaluation.json")
    evidence["recipe_sha256"] = mt.sha256_json(r.data)
    evidence["model_artifact"] = {**mt.artifact_manifest(checkpoint), "role": "trained_checkpoint"}
    evidence["config"]["model"] = evidence["model_artifact"]
    evidence["config_sha256"] = mt.sha256_json(evidence["config"])
    relative = evidence["predictions_file"]["path"]
    mt.save_json(output / relative, read(bundle / "reference" / relative))
    mt.save_json(output / "manifest.json", evidence["split_manifest"])
    return mt.save_json(output / "evaluation.json", evidence)


def mock_process(case, monkeypatch, *, after=None):
    child_pid = os.getpid() + 10000
    parent_pid = os.getpid()
    calls = []
    monkeypatch.setattr(mt, "run_evaluation", fake_evaluation)
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        process = Mock(pid=child_pid, returncode=0)
        def communicate(timeout):
            assert timeout == et.WORKER_TIMEOUT_SECONDS
            with monkeypatch.context() as child:
                for key, value in kwargs["env"].items():
                    child.setenv(key, value)
                child.setattr(et.os, "getpid", lambda: child_pid)
                child.setattr(et.os, "getppid", lambda: parent_pid)
                et.run_worker(case.destination)
            if after:
                after(case.destination)
            return "测试 worker 输出", ""
        process.communicate.side_effect = communicate
        return process
    monkeypatch.setattr(et.subprocess, "Popen", launch)
    return calls


def prepare(case, monkeypatch):
    def stop(bundle):
        raise mt.TaskExecutionError("测试停止于 prepared 包")
    with monkeypatch.context() as context:
        context.setattr(et, "_launch_worker", stop)
        with pytest.raises(mt.TaskExecutionError, match="prepared"):
            et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    assert not (case.destination / et.FINAL).exists()
    return case.destination


def test_export_copies_all_files_binds_hashes_and_launches_offline(case, monkeypatch):
    original = copy.deepcopy(case.r.data)
    original_toml = case.r.path.read_bytes()
    calls = mock_process(case, monkeypatch)
    result = et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    evidence = read(result)
    assert result == case.destination / et.FINAL
    assert evidence["schema_version"] == 1 and evidence["kind"] == "export-reload" and evidence["stage"] == "completed"
    assert evidence["runtime"] == case.runtime
    assert evidence["license"] == {"status": "pending", "commercial_use_confirmed": False}
    assert evidence["baseline_promotion"] == "none"
    assert evidence["subprocess"]["pid"] != os.getpid()
    assert evidence["subprocess"]["returncode"] == 0
    assert all(value["within_tolerance"] for value in evidence["comparison"].values())
    assert case.r.data == original and case.r.path.read_bytes() == original_toml
    assert (case.destination / et.LOCKFILE).read_bytes() == case.lock.read_bytes()
    snapshot = read(case.destination / et.SNAPSHOT)
    assert snapshot["data"] == original["data"] and snapshot["train"] == original["train"]
    assert snapshot["execution"]["lockfile"] == et.LOCKFILE
    for path in case.checkpoint.iterdir():
        assert (case.destination / "model" / path.name).read_bytes() == path.read_bytes()
    assert (case.destination / "reference" / "nested" / "predictions.json").read_bytes() == (case.evaluation.parent / "nested" / "predictions.json").read_bytes()
    for record in evidence["files"].values():
        assert be.hashed_file(case.destination, record)
    assert set(evidence["files"]) == {path.relative_to(case.destination).as_posix() for path in case.destination.rglob("*") if path.is_file()} - {et.FINAL}
    command, options = calls[0]
    assert command == [sys.executable, str(ROOT / "runner" / "export.py"), "--worker", "--bundle", str(case.destination)]
    assert options["shell"] is False and options["cwd"] == ROOT
    assert all(options["env"][key] == value for key, value in et.OFFLINE_ENV.items())
    worker = read(case.destination / et.WORKER_EVIDENCE)
    assert worker["pid"] == evidence["subprocess"]["pid"] and worker["ppid"] == os.getpid()
    assert be.hashed_file(case.destination, worker["evaluation"])


def test_dry_plan_only_reads_declarations(case, monkeypatch):
    original_files = set(case.r.path.parent.rglob("*"))
    monkeypatch.setattr(et, "_json", Mock(side_effect=AssertionError("不得读取 checkpoint 或 evaluation")))
    monkeypatch.setattr(mt, "load_runtime", Mock(side_effect=AssertionError("不得导入框架")))
    monkeypatch.setattr(et, "_launch_worker", Mock(side_effect=AssertionError("不得启动进程")))
    plan = et.export_bundle(case.r, case.r.path.parent / "absent", "absent.json", case.destination, execute=False)
    assert plan["kind"] == "export-plan" and plan["stage"] == "declared"
    assert plan["execution_supported"] is True
    assert plan["train_seed"] == 42 and plan["data_seed"] == 17 and plan["evaluation_seed"] == 23
    assert set(case.r.path.parent.rglob("*")) == original_files


def test_unsupported_execute_rejects_but_dry_declaration_allowed(case):
    r = Recipe("other/route", case.r.path, case.r.data)
    assert et.export_bundle(r, "absent", "absent", "absent", False)["execution_supported"] is False
    with pytest.raises(mt.TaskExecutionError, match="执行路由"):
        et.export_bundle(r, "absent", "absent", case.destination)
    assert not case.destination.exists()


@pytest.mark.parametrize("corruption", ["weight", "missing", "smoke", "mock", "simulated", "dry_run", "config", "steps", "fingerprint", "revision", "data_seed", "lock_hash", "code_hash", "environment"])
def test_bad_checkpoint_or_training_rejected_without_output(case, corruption):
    training_path = case.checkpoint / "training.json"
    if corruption == "weight":
        (case.checkpoint / "model.safetensors").write_bytes(b"changed test artifact")
    elif corruption == "missing":
        training_path.unlink()
    elif corruption == "environment":
        revise(training_path, lambda value: value["runtime"].update(gpu="mock-GPU"))
    else:
        def change(value):
            if corruption == "smoke":
                value["mode"] = "smoke"
            elif corruption in ("mock", "simulated", "dry_run"):
                value[corruption] = True
            elif corruption == "config":
                value["config"]["learning_rate"] = 1e-3
            else:
                field = {"steps": "steps", "fingerprint": "fingerprint", "revision": "model_revision",
                         "data_seed": "data_seed", "lock_hash": "environment_lock_sha256", "code_hash": "code_sha256"}[corruption]
                value[field] = 1 if corruption in ("steps", "data_seed") else "invalid"
        revise(training_path, change)
    with pytest.raises(mt.TaskExecutionError):
        et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    assert not case.destination.exists()


@pytest.mark.parametrize("corruption", ["fingerprint", "artifact", "role", "mock", "metrics", "generation", "gold", "prompt", "prediction_hash", "manifest", "lock_hash", "code_hash"])
def test_bad_reference_evidence_rejected(case, corruption):
    if corruption in ("gold", "prompt"):
        prediction_path = case.evaluation.parent / "nested" / "predictions.json"
        predictions = read(prediction_path)
        predictions[0]["solution" if corruption == "gold" else "prompt"] = "" if corruption == "gold" else mt.make_prompt("被替换问题")
        mt.save_json(prediction_path, predictions)
        def change(value):
            value["predictions_file"]["sha256"] = et._hash(prediction_path)
            value["predictions"]["sha256"] = mt.sha256_json(predictions)
        revise(case.evaluation, change)
    elif corruption == "manifest":
        revise(case.evaluation.parent / "manifest.json", lambda value: value.update(dataset_revision="changed"))
    else:
        def change(value):
            if corruption == "artifact":
                value["model_artifact"]["sha256"] = "e" * 64
                value["config_sha256"] = mt.sha256_json(value["config"])
            elif corruption == "role":
                value["model_role"] = "base"
            elif corruption == "mock":
                value["mock"] = True
            elif corruption == "metrics":
                value["metrics"]["accuracy"] = 0.0
            elif corruption == "generation":
                value["config"]["generation"]["do_sample"] = True
                value["config_sha256"] = mt.sha256_json(value["config"])
            elif corruption == "prediction_hash":
                value["predictions_file"]["sha256"] = "e" * 64
            else:
                value[{"fingerprint": "fingerprint", "lock_hash": "environment_lock_sha256", "code_hash": "code_sha256"}[corruption]] = "e" * 64
        revise(case.evaluation, change)
    with pytest.raises(mt.TaskExecutionError):
        et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    assert not case.destination.exists()


@pytest.mark.parametrize("invalid", [None, {}, {"accuracy": 0.0}, {"accuracy": -1, "pass@1": 0, "format_compliance": 0},
                                    {"accuracy": float("inf"), "pass@1": 0, "format_compliance": 0},
                                    {"accuracy": float("nan"), "pass@1": 0, "format_compliance": 0},
                                    {"accuracy": True, "pass@1": 0, "format_compliance": 0}])
def test_tolerance_missing_partial_negative_nonfinite_rejected(case, invalid):
    case.r.data["evaluation"]["export_tolerance"] = invalid
    with pytest.raises(mt.TaskExecutionError, match="tolerance"):
        et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    assert not case.destination.exists()


def test_tolerance_fallback_is_explicit_and_not_defaulted(case, monkeypatch):
    case.r.data["evaluation"]["tolerance"] = case.r.data["evaluation"].pop("export_tolerance")
    # 容差与审核状态不进入 fingerprint；不重写原始评估快照证据。
    mock_process(case, monkeypatch)
    assert read(et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination))["stage"] == "completed"


@pytest.mark.parametrize("kind", ["nonempty", "symlink", "ancestor_symlink", "inside_checkpoint", "contains_checkpoint", "weight_symlink", "prediction_escape"])
def test_unsafe_destination_and_source_paths_rejected(case, kind):
    if kind == "nonempty":
        case.destination.mkdir()
        (case.destination / "keep.txt").write_text("保留用户文件", encoding="utf-8")
    elif kind == "symlink":
        case.destination.symlink_to(case.checkpoint, target_is_directory=True)
    elif kind == "ancestor_symlink":
        parent = case.r.path.parent / "linked"
        parent.symlink_to(case.r.path.parent, target_is_directory=True)
        case.destination = parent / "output"
    elif kind == "inside_checkpoint":
        case.destination = case.checkpoint / "export"
    elif kind == "contains_checkpoint":
        case.destination = case.checkpoint.parent
    elif kind == "weight_symlink":
        weight = case.checkpoint / "model.safetensors"
        backup = case.r.path.parent / "weight-backup"
        weight.rename(backup)
        weight.symlink_to(backup)
    else:
        revise(case.evaluation, lambda value: value["predictions_file"].update(path="../outside.json"))
    with pytest.raises(mt.TaskExecutionError):
        et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    assert not (case.destination / et.FINAL).exists()
    if kind == "nonempty":
        assert (case.destination / "keep.txt").read_text(encoding="utf-8") == "保留用户文件"


@pytest.mark.parametrize("failure", ["spawn", "exception", "timeout", "nonzero"])
def test_subprocess_failures_never_write_completed_export(case, monkeypatch, failure):
    process = Mock(pid=os.getpid() + 10000, returncode=3 if failure == "nonzero" else 0)
    if failure == "spawn":
        factory = Mock(side_effect=OSError("测试启动失败"))
    else:
        process.communicate.side_effect = ([subprocess.TimeoutExpired("test", 1), ("", "")]
                                           if failure == "timeout" else [subprocess.SubprocessError("测试通信失败"), ("", "")]
                                           if failure == "exception" else lambda **kwargs: ("", "测试失败"))
        process.poll.return_value = None
        factory = Mock(return_value=process)
    monkeypatch.setattr(et.subprocess, "Popen", factory)
    with pytest.raises(mt.TaskExecutionError):
        et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    assert not (case.destination / et.FINAL).exists()
    assert read(case.destination / et.MANIFEST)["stage"] == "prepared"
    if failure in ("timeout", "exception"):
        process.kill.assert_called_once()


@pytest.mark.parametrize("corruption", ["weight", "lock", "snapshot", "reference", "manifest", "evaluation", "worker_pid", "worker_ppid", "worker_stage", "worker_runtime"])
def test_post_worker_tampering_never_completes(case, monkeypatch, corruption):
    def alter(bundle):
        paths = {"weight": "model/model.safetensors", "lock": et.LOCKFILE, "snapshot": et.SNAPSHOT,
                 "reference": "reference/evaluation.json", "manifest": et.MANIFEST, "evaluation": "reload/eval/evaluation.json"}
        if corruption in paths:
            with (bundle / paths[corruption]).open("ab") as handle:
                handle.write(b"changed")
        else:
            key = {"worker_pid": "pid", "worker_ppid": "ppid", "worker_stage": "stage", "worker_runtime": "runtime"}[corruption]
            revise(bundle / et.WORKER_EVIDENCE, lambda value: value.update({key: os.getpid() if key == "pid" else -1 if key == "ppid" else "changed"}))
    mock_process(case, monkeypatch, after=alter)
    with pytest.raises(mt.TaskExecutionError):
        et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    assert not (case.destination / et.FINAL).exists()


@pytest.mark.parametrize("key", list(et.OFFLINE_ENV))
def test_worker_rejects_nonoffline_environment(case, monkeypatch, key):
    for name, value in et.OFFLINE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(key, "invalid")
    load = Mock(side_effect=AssertionError("不应触及评估"))
    monkeypatch.setattr(mt, "run_evaluation", load)
    with pytest.raises(mt.TaskExecutionError, match="offline"):
        et.run_worker(case.destination)
    load.assert_not_called()


@pytest.mark.parametrize("corruption", ["weight", "lock", "snapshot", "manifest", "extra", "symlink"])
def test_worker_rejects_tampered_bundle_before_evaluation(case, monkeypatch, corruption):
    bundle = prepare(case, monkeypatch)
    for key, value in et.OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)
    if corruption == "extra":
        (bundle / "unbound.txt").write_text("测试未绑定文件", encoding="utf-8")
    elif corruption == "symlink":
        (bundle / "linked").symlink_to(case.checkpoint, target_is_directory=True)
    else:
        path = bundle / {"weight": "model/model.safetensors", "lock": et.LOCKFILE, "snapshot": et.SNAPSHOT, "manifest": et.MANIFEST}[corruption]
        with path.open("ab") as handle:
            handle.write(b"changed")
    evaluation = Mock(side_effect=AssertionError("不应开始评估"))
    monkeypatch.setattr(mt, "run_evaluation", evaluation)
    with pytest.raises(mt.TaskExecutionError):
        et.run_worker(bundle)
    evaluation.assert_not_called()
    assert not (bundle / et.WORKER_EVIDENCE).exists()


def test_worker_missing_dataset_cache_reports_data_dependency(case, monkeypatch):
    bundle = prepare(case, monkeypatch)
    for key, value in et.OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(mt, "run_evaluation", Mock(side_effect=FileNotFoundError("dataset cache missing")))
    with pytest.raises(mt.TaskExecutionError, match="不是无数据依赖的安装镜像"):
        et.run_worker(bundle)
    assert not (bundle / et.WORKER_EVIDENCE).exists() and not (bundle / et.FINAL).exists()


def test_real_new_python_worker_with_test_only_injection(case, monkeypatch):
    bundle = prepare(case, monkeypatch)
    # 真正创建新 Python 进程；只替换数学评估依赖，不下载、不触及 GPU。
    program = '''import importlib.abc, runpy, sys
from pathlib import Path
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trl', 'transformers', 'datasets', 'math_verify'}:
            raise AssertionError('测试禁止导入框架: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from recipe_lab import math_task
from test_export import fake_evaluation
math_task.run_evaluation = fake_evaluation
sys.argv = [sys.argv[3], '--worker', '--bundle', sys.argv[4]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    process = subprocess.Popen([sys.executable, "-B", "-c", program, str(ROOT / "src"), str(ROOT / "tests"),
                                str(ROOT / "runner" / "export.py"), str(bundle)],
                               env={**os.environ, **et.OFFLINE_ENV}, cwd=case.r.path.parent,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    worker = read(bundle / et.WORKER_EVIDENCE)
    assert worker["pid"] == process.pid != os.getpid() and worker["ppid"] == os.getpid()
    assert worker["stage"] == "completed" and be.hashed_file(bundle, worker["evaluation"])
    assert "许可 pending" in stdout and not (bundle / et.FINAL).exists()


def test_cli_seed_override_is_temporary_and_only_train_seed(case, monkeypatch):
    spec = importlib.util.spec_from_file_location("export_cli_test", ROOT / "runner" / "export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = copy.deepcopy(case.r.data)
    revise(case.checkpoint / "training.json", lambda value: value.update(seed=43))
    monkeypatch.setattr(module, "find_recipe", lambda recipe_id: case.r)
    seen = []
    def capture(r, checkpoint, evaluation, output, execute):
        seen.append((r, execute))
        return {"kind": "test-only-plan"}
    monkeypatch.setattr(module, "export_bundle", capture)
    argv = [str(ROOT / "runner" / "export.py"), "--recipe", case.r.id, "--checkpoint", str(case.checkpoint),
            "--evaluation", str(case.evaluation), "--output", str(case.destination)]
    monkeypatch.setattr(sys, "argv", argv + ["--full"])
    assert module.main() == 0
    effective, execute = seen[0]
    assert execute is True and effective.get("train.seed") == 43
    expected = copy.deepcopy(original)
    expected["train"]["seed"] = 43
    assert effective.data == expected and case.r.data == original
    monkeypatch.setattr(sys, "argv", argv + ["--dry-run"])
    assert module.main() == 0
    assert seen[1] == (case.r, False)


def test_cli_dry_run_in_new_process_is_framework_free(tmp_path):
    directory = tmp_path / "recipes" / "isolated"
    directory.mkdir(parents=True)
    (directory / "recipe.toml").write_text('id = "isolated/recipe"\n', encoding="utf-8")
    program = '''import importlib.abc, runpy, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trl', 'transformers', 'datasets', 'math_verify'}:
            raise AssertionError('测试禁止导入框架: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
sys.argv = [sys.argv[1], '--recipe', 'isolated/recipe', '--checkpoint', 'missing', '--evaluation', 'missing', '--output', 'unused', '--dry-run']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-B", "-c", program, str(ROOT / "runner" / "export.py")],
                            cwd=tmp_path, env={**os.environ, "RECIPE_LAB_ROOT": str(tmp_path)},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["stage"] == "declared"
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("change", ["gold", "generation", "scores"])
def test_reload_input_or_scores_outside_tolerance_never_complete(case, monkeypatch, change):
    mock_process(case, monkeypatch)
    def changed_evaluation(r, plan, *, checkpoint):
        path = fake_evaluation(r, plan, checkpoint=checkpoint)
        evidence = read(path)
        if change == "generation":
            evidence["config"]["generation"]["pad_token_id"] = 1
            evidence["config_sha256"] = mt.sha256_json(evidence["config"])
        else:
            prediction_path = path.parent / evidence["predictions_file"]["path"]
            predictions = read(prediction_path)
            predictions[0]["solution" if change == "gold" else "accuracy"] = "3" if change == "gold" else 0
            mt.save_json(prediction_path, predictions)
            evidence["predictions_file"]["sha256"] = et._hash(prediction_path)
            evidence["predictions"]["sha256"] = mt.sha256_json(predictions)
            if change == "scores":
                evidence["metrics"]["accuracy"] = evidence["metrics"]["pass@1"] = 499 / 500
        return mt.save_json(path, evidence)
    monkeypatch.setattr(mt, "run_evaluation", changed_evaluation)
    with pytest.raises(mt.TaskExecutionError, match="gold|generation|tolerance"):
        et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    assert not (case.destination / et.FINAL).exists()


def test_export_uses_checkpoint_seed_without_changing_data_or_eval_seed(case, monkeypatch):
    original = copy.deepcopy(case.r.data)
    effective = copy.deepcopy(original)
    effective["train"]["seed"] = 43
    effective_r = Recipe(case.r.id, case.r.path, effective)
    for path in (case.checkpoint / "training.json", case.evaluation):
        def change(value):
            value.update(seed=43, fingerprint=be.fingerprint(effective_r), recipe_sha256=mt.sha256_json(effective))
            if value["kind"] == "training":
                value["config"]["seed"] = 43
                value["config_sha256"] = mt.sha256_json(value["config"])
        revise(path, change)
    mock_process(case, monkeypatch)
    evidence = read(et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination))
    snapshot = read(case.destination / et.SNAPSHOT)
    assert evidence["seed"] == snapshot["train"]["seed"] == 43
    assert evidence["data_seed"] == 17 and evidence["evaluation_seed"] == 23
    assert snapshot["data"] == original["data"] and case.r.data == original


def test_worker_calls_real_math_evaluation_with_only_dependencies_mocked(case, monkeypatch):
    bundle = prepare(case, monkeypatch)
    for key, value in et.OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)
    calls = {"model": [], "tokenizer": [], "generate": [], "dataset": []}
    class Encoded(dict):
        def to(self, device):
            assert device == "cuda"
            return self
    tokenizer = SimpleNamespace(eos_token_id=0)
    tokenizer.apply_chat_template = lambda prompt, **kwargs: Encoded(input_ids=SimpleNamespace(shape=(1, 2)))
    tokenizer.decode = lambda *args, **kwargs: "<think>隔离依赖测试</think><answer>2</answer>"
    model = SimpleNamespace()
    model.to = lambda device: model
    model.eval = lambda: None
    def generate(**kwargs):
        calls["generate"].append(kwargs)
        return [[1, 2, 3]]
    model.generate = generate
    def load_model(source, **kwargs):
        calls["model"].append((source, kwargs))
        return model
    def load_tokenizer(source, **kwargs):
        calls["tokenizer"].append((source, kwargs))
        return tokenizer
    def dataset(*args, **kwargs):
        calls["dataset"].append((args, kwargs))
        return [{"problem": f"隔离测试问题{i}", "solution": "2"} for i in range(1500)]
    dependencies = {
        "torch": SimpleNamespace(float32="test-only-float32", inference_mode=contextlib.nullcontext,
                                 backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace())),
                                 cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None),
                                 use_deterministic_algorithms=lambda value: None),
        "transformers": SimpleNamespace(set_seed=Mock(),
                                         AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
                                         AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)),
        "datasets": SimpleNamespace(load_dataset=dataset),
        "parse": lambda text: [text], "verify": lambda gold, prediction: gold == prediction,
    }
    monkeypatch.setattr(mt, "load_runtime", lambda: dependencies)
    monkeypatch.setattr(mt, "require_frozen_runtime", lambda *args: None)
    monkeypatch.setattr(mt, "runtime_evidence", lambda *args: copy.deepcopy(case.runtime))
    # 生产completed_evidence必须自行生成code/lock hash，不再由测试补缺字段。
    worker_path = et.run_worker(bundle)
    worker = read(worker_path)
    evaluation = read(bundle / worker["evaluation"]["path"])
    assert evaluation["stage"] == "completed" and len(calls["generate"]) == 500
    assert calls["model"][0][0] == calls["tokenizer"][0][0] == str(bundle / "model")
    assert calls["model"][0][1]["local_files_only"] is True
    assert calls["tokenizer"][0][1]["local_files_only"] is True
    assert calls["dataset"][0] == ((mt.DATASET_ID,), {"revision": "b" * 40, "split": "train"})
    assert all(call["do_sample"] is False and call["num_beams"] == 1 for call in calls["generate"])
    dependencies["transformers"].set_seed.assert_called_once_with(23)
    assert be.hashed_file(bundle, worker["evaluation"])


def test_completed_export_has_readonly_content_gate_and_rollback_confirmation(case, monkeypatch):
    from recipe_lab.baseline import check_passed
    mock_process(case, monkeypatch)
    result = et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    before = {p: p.read_bytes() for p in case.destination.rglob("*") if p.is_file()}
    assert et.validate_export_evidence(case.r, result)
    assert all(p.read_bytes() == content for p, content in before.items())
    case.r.data["baseline_contract"] = {"export_evidence": str(result), "export_verified": True}
    check = {"kind": "export_reload", "field": "baseline_contract.export_evidence",
             "confirmation_field": "baseline_contract.export_verified"}
    assert not check_passed(case.r, check), "复载成功不等于完成回滚演练"
    case.r.data["baseline_contract"].update(rollback_verified=True, rollback_evidence="隔离测试人工回滚记录")
    assert check_passed(case.r, check)
    (case.destination / "model/model.safetensors").write_bytes(b"changed")
    assert not check_passed(case.r, check), "确认字段不能放行篡改包"


@pytest.mark.parametrize("change", ["summary", "prediction", "statistics", "worker", "extra"])
def test_completed_export_gate_rejects_post_completion_changes(case, monkeypatch, change):
    mock_process(case, monkeypatch)
    result = et.export_bundle(case.r, case.checkpoint, case.evaluation, case.destination)
    if change == "summary":
        revise(result, lambda value: value["metrics"].update(accuracy=0.0))
    elif change == "prediction":
        (case.destination / "reload/eval/nested/predictions.json").write_text("[]", encoding="utf-8")
    elif change == "statistics":
        revise(result, lambda value: value["comparison"]["accuracy"].update(absolute_delta=0.5))
    elif change == "worker":
        revise(result, lambda value: value["subprocess"].update(pid=value["subprocess"]["parent_pid"]))
    else:
        (case.destination / "unbound.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(mt.TaskExecutionError):
        et.validate_export_evidence(case.r, result)
