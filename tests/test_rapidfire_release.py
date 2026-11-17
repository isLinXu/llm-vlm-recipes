"""RapidFire rapidfire-sft-v1 发布级 comparison / export 的隔离离线契约测试。

用 fake runtime 调用真实 rapidfire_task.run_training / run_evaluation 生成
4 配置 × 3 独立训练 seed 的合成证据，再验证：
  * 只用 validation 的 3 seed paired 均值确定性选配置，test 只在选中配置上验收一次；
  * 逐题从 raw 5 列重建 prompt/gold 并重算无 stemmer ROUGE-L / exact_match；
  * 4 配置顺序成本对照、Student-t95 跨 seed 聚合与 C306/C307 分派；
  * 选中配置 adapter-only 包、冻结 base 声明与 offline 新进程复载（C405）。

所有产物只写 tmp_path；JSON/hash 只证明仓内内容一致性，不构成 GPU 实跑证据，
也不自动晋级 B2/B3。fake runtime 明确是合成夹具，不声明任何真实模型行为。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipe_lab import baseline as bl
from recipe_lab import baseline_evidence as be
from recipe_lab import export_task as et
from recipe_lab import provenance as pv
from recipe_lab import rapidfire_export as re
from recipe_lab import rapidfire_suite as rs
from recipe_lab import rapidfire_task as rt
from recipe_lab import suite_task as st
from recipe_lab import task_router as tr
from recipe_lab.math_task import TaskExecutionError
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan
from test_rapidfire import (LOCKFILE, Tokenizer, _recipe, explicit_factory,
                            fake_runtime as base_fake_runtime, install_fake, make_rows)

SEEDS = [42, 43, 44]
CONFIGS = ["lr5e-4-r8", "lr5e-4-r32", "lr2e-4-r8", "lr2e-4-r32"]
# 合成生成质量按配置递增，让 validation paired 均值有稳定且可复核的排序。
QUALITY = {"lr5e-4-r8": 0.45, "lr5e-4-r32": 0.55,
           "lr2e-4-r8": 0.65, "lr2e-4-r32": 0.75}
BASE_QUALITY = 0.20
SELECTED = "lr2e-4-r32"
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 合成配置与 runtime

def _release_recipe(tmp_path, *, tolerance=True, task_id=True):
    r = _recipe(tmp_path)
    data = copy.deepcopy(r.data)
    if task_id:
        data["execution"]["task_id"] = rt.TASK_ID
    if tolerance:
        data["evaluation"]["export_tolerance"] = {name: 0.0 for name in rt.METRICS}
    return Recipe(r.id, r.path, data)


def release_runtime(r, rows, quality):
    """扩展三维 fake runtime：validation/test 都能生成，且按质量截断 gold。"""
    runtime = base_fake_runtime(r, rows)
    splits, _ = rt.prepare_splits(rows, r)
    gold_by_prompt = {}
    for name in ("validation", "test"):
        for row in splits[name]:
            gold_by_prompt[tuple(Tokenizer._ids(rt.render_prompt(row)))] = Tokenizer._ids(
                rt.render_target(row, Tokenizer.eos_token))
    original_loader = runtime["transformers"].AutoModelForCausalLM.from_pretrained

    def load_model(source, **kwargs):
        model = original_loader(source, **kwargs)

        def generate(input_ids, attention_mask, **kwargs):
            prompt = input_ids.rows[0]
            gold = gold_by_prompt[tuple(prompt)]
            keep = max(1, min(len(gold), int(len(gold) * quality)))
            return [prompt + gold[:keep]]

        model.generate = generate
        return model

    runtime["transformers"].AutoModelForCausalLM.from_pretrained = load_model
    return runtime


def quality_for(config_id, seed, *, base=False):
    if base:
        return BASE_QUALITY
    return QUALITY[config_id] + 0.005 * (seed - SEEDS[0])


def _plan(r, root, mode, split_name="test"):
    raw = build_plan(r, mode)
    raw["run"]["output_dir"] = str(root)
    raw["evaluation_split"] = split_name
    raw["eval_stage"] = rt.EVAL_STAGES[split_name]
    plan = rt.task_plan(r, raw, mode)
    assert plan["execution_supported"], plan.get("unsupported_reason")
    return plan


def _freeze(r, root):
    root.mkdir(parents=True, exist_ok=True)
    (root / st.CODE_SNAPSHOT).write_bytes(pv.snapshot_bytes(pv.code_snapshot()))
    (root / st.LOCKFILE).write_bytes((r.path.parent / LOCKFILE).read_bytes())


def _config_recipe(r, config_id, seed):
    return rs._config_recipe(r, config_id, seed)


def _evaluate(monkeypatch, r, rows, root, *, split_name, quality, checkpoint=None,
              zero_shot=False):
    install_fake(monkeypatch, release_runtime(r, rows, quality))
    plan = _plan(r, root, "zero-shot" if zero_shot else "eval", split_name)
    if zero_shot:
        return rt.run_evaluation(r, plan, zero_shot=True, split_name=split_name)
    return rt.run_evaluation(r, plan, checkpoint=checkpoint, split_name=split_name)


@pytest.fixture
def fcase(tmp_path, monkeypatch):
    """4 配置 × 3 seed 合成证据 + 冻结 root；只写 tmp_path，不加载真实框架。"""
    r = _release_recipe(tmp_path)
    root = tmp_path / "suite-root"
    _freeze(r, root)
    rows = make_rows()
    base = {}
    for split in (rs.SELECTION_SPLIT, rs.ACCEPTANCE_SPLIT):
        base[split] = _evaluate(
            monkeypatch, r, rows, root / "base" / split / "zero-shot",
            split_name=split, quality=BASE_QUALITY, zero_shot=True)
    checkpoints, validation, acceptance_runs = {}, {}, []
    for config_id in CONFIGS:
        for seed in SEEDS:
            effective = _config_recipe(r, config_id, seed)
            seed_root = rs._seed_dir(root, config_id, seed)
            install_fake(monkeypatch, release_runtime(
                effective, rows, quality_for(config_id, seed)))
            training = rt.run_training(effective, _plan(effective, seed_root / "full", "full"))
            checkpoints[(config_id, seed)] = training.parent
            validation[(config_id, seed)] = _evaluate(
                monkeypatch, effective, rows, seed_root / rs.SELECTION_SPLIT / "eval",
                split_name=rs.SELECTION_SPLIT, quality=quality_for(config_id, seed),
                checkpoint=training.parent)
    # 用套件自己的确定性选择规则决定 test 只评估哪个配置，不硬编码结果。
    hashes = rs._hashes(r)
    selection = rs._selection_from_root(r, SEEDS, root, hashes)
    selected = selection["selected_config_id"]
    test = {}
    for seed in SEEDS:
        effective = _config_recipe(r, selected, seed)
        seed_root = rs._seed_dir(root, selected, seed)
        test[seed] = _evaluate(
            monkeypatch, effective, rows, seed_root / rs.ACCEPTANCE_SPLIT / "eval",
            split_name=rs.ACCEPTANCE_SPLIT, quality=quality_for(selected, seed),
            checkpoint=checkpoints[(selected, seed)])
        acceptance_runs.append(test[seed])
    return SimpleNamespace(r=r, root=root, rows=rows, base=base,
                           checkpoints=checkpoints, validation=validation, test=test,
                           selected=selected, lock=tmp_path / LOCKFILE)


# ---------------------------------------------------------------- 路由 / 计划 / 失败关闭

def test_router_dispatches_all_rapidfire_release_modules(fcase):
    r = fcase.r
    assert tr.resolve(r) == rt.TASK_ID
    assert tr.load(r, "task") is rt
    assert tr.load(r, "suite") is rs
    assert tr.load(r, "export") is re
    assert be.frozen_environment(r) is True


def test_unknown_or_conflicting_task_id_fails_closed(fcase):
    data = copy.deepcopy(fcase.r.data)
    data["execution"]["task_id"] = "math-grpo-v1"
    wrong = Recipe(fcase.r.id, fcase.r.path, data)
    with pytest.raises(TaskExecutionError, match="冲突"):
        tr.resolve(wrong)
    assert "尚未实现" in be.evaluation_problems(wrong, {}, Path("x"), role="base")[0]


def test_suite_plan_is_readonly_and_declares_deferred_acceptance(fcase):
    before = {p: p.stat().st_mtime_ns for p in fcase.root.rglob("*") if p.is_file()}
    plan = rs.suite_plan(fcase.r, SEEDS, fcase.root)
    assert plan["status"] == "declared" and plan["task_id"] == rt.TASK_ID
    assert plan["seeds"] == SEEDS and plan["config_ids"] == CONFIGS
    assert len(plan["commands"]) == 2 + len(CONFIGS) * len(SEEDS) * 2
    assert len(plan["runs"]) == len(CONFIGS) * len(SEEDS)
    assert plan["selection"]["split"] == "validation"
    assert plan["selection"]["test_used_in_selection"] is False
    assert plan["selection"]["policy"] == rt.RAPIDFIRE_SPEC["selection_policy"]
    assert plan["acceptance"]["split"] == "test"
    assert plan["acceptance"]["evaluations"] == len(SEEDS)
    assert "deferred" in plan["acceptance"]["commands"]
    assert plan["costs"]["upstream_throughput_comparable"] is False
    assert plan["full_steps"] == 64 and plan["shell"] is False
    assert all("--config-id" in step["command"] for step in plan["commands"])
    assert all("--eval-split validation" in " ".join(step["command"])
               for step in plan["commands"] if step["kind"] == "evaluation")
    assert {p: p.stat().st_mtime_ns for p in fcase.root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("seeds", [[], [1, 1, 2], [1, 2, True], [1, 2, 2**32]])
@pytest.mark.parametrize("operation", [rs.suite_plan, rs.aggregate_suite, rs.run_suite])
def test_invalid_seed_lists_fail_without_writes(fcase, seeds, operation):
    root = fcase.root.parent / "unused-root"
    with pytest.raises(TaskExecutionError):
        operation(fcase.r, seeds, root)
    assert not root.exists()


def test_selection_is_deterministic_and_breaks_ties_by_config_id(fcase):
    runs = [{"config_id": cid, "train_seed": s,
             "base_metrics": {"rouge_l": 0.1, "exact_match": 0.0},
             "trained_metrics": {"rouge_l": 0.2, "exact_match": 0.0}}
            for cid in CONFIGS for s in SEEDS]
    selection = rs._selection(fcase.r, runs, SEEDS)
    assert selection["selected_config_id"] == min(CONFIGS)
    assert selection["split"] == "validation" and selection["test_used_in_selection"] is False
    assert len(selection["candidates"]) == 4


# ---------------------------------------------------------------- 严格聚合与内容重算

def test_aggregate_suite_recomputes_full_grid_and_single_acceptance(fcase):
    result = rs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    assert result == fcase.root / rs.FINAL
    data = read(result)
    assert data["kind"] == "baseline-suite" and data["status"] == "completed"
    assert data["task_id"] == rt.TASK_ID and data["recipe_id"] == fcase.r.id
    assert data["config_ids"] == CONFIGS and data["seeds"] == SEEDS
    assert len(data["runs"]) == 12 and len(data["acceptance_runs"]) == 3
    assert data["selection"]["selected_config_id"] == fcase.selected == SELECTED
    assert data["selection"]["split"] == "validation"
    assert data["selection"]["test_used_in_selection"] is False
    assert data["selection_split"] == "validation" and data["acceptance_split"] == "test"
    assert data["test_evaluated_once"] is True and data["test_evaluations"] == 3
    assert data["statistical_unit"] == "independent_training_seed"
    assert data["training_seed_variance_measured"] is True
    assert data["baseline_promotion"] == "none"
    assert data["acceptance"]["status"] == "not_evaluated"
    assert all(run["evaluation_split"] == "validation" for run in data["runs"])
    assert all(run["evaluation_split"] == "test"
               and run["config_id"] == fcase.selected for run in data["acceptance_runs"])
    for metric in rt.METRICS:
        stats = data["acceptance_stats"][metric]
        assert stats["trained"]["n"] == 3 and stats["paired_delta"]["n"] == 3
        assert stats["trained"]["student_t95_ci"]["df"] == 2
    assert set(data["validation_stats"]) == set(CONFIGS)


def test_costs_are_sequential_and_not_comparable_to_upstream(fcase):
    result = rs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    costs = read(result)["costs"]
    assert costs["sequential_execution"] is True
    assert costs["concurrent_scheduling"] is False
    assert costs["upstream_throughput_comparable"] is False
    assert set(costs["per_config"]) == set(CONFIGS)
    assert all(entry["seeds"] == 3 and entry["total_steps"] == 64 * 3
               and entry["total_seconds"] > 0 for entry in costs["per_config"].values())
    assert costs["sequential_total_seconds"] == pytest.approx(
        sum(entry["total_seconds"] for entry in costs["per_config"].values()))


def test_aggregate_rejects_tampered_prediction_scores(fcase, monkeypatch):
    path = fcase.validation[(SELECTED, SEEDS[0])].parent / rt.PREDICTIONS_FILE
    document = read(path)
    document["predictions"][0]["rouge_l"] = 0.999
    document["metrics"] = dict(document["metrics"])
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(TaskExecutionError):
        rs.aggregate_suite(fcase.r, SEEDS, fcase.root)


def test_aggregate_rejects_tampered_checkpoint_artifact(fcase):
    target = fcase.checkpoints[(SELECTED, SEEDS[0])] / "adapter_model.safetensors"
    target.write_bytes(b"tampered-adapter-weights")
    with pytest.raises(TaskExecutionError):
        rs.aggregate_suite(fcase.r, SEEDS, fcase.root)


def test_aggregate_rejects_unselected_config_test_evaluation(fcase, monkeypatch):
    other = "lr5e-4-r8"
    seed = SEEDS[0]
    effective = _config_recipe(fcase.r, other, seed)
    _evaluate(monkeypatch, effective, fcase.rows,
              rs._seed_dir(fcase.root, other, seed) / rs.ACCEPTANCE_SPLIT / "eval",
              split_name=rs.ACCEPTANCE_SPLIT, quality=quality_for(other, seed),
              checkpoint=fcase.checkpoints[(other, seed)])
    with pytest.raises(TaskExecutionError, match="test 评估出现在未选中配置"):
        rs.aggregate_suite(fcase.r, SEEDS, fcase.root)


def test_aggregate_rejects_missing_test_acceptance(fcase):
    shutil.rmtree(rs._seed_dir(fcase.root, fcase.selected, SEEDS[0]) / rs.ACCEPTANCE_SPLIT)
    with pytest.raises(TaskExecutionError):
        rs.aggregate_suite(fcase.r, SEEDS, fcase.root)


def test_validation_evidence_cannot_be_relabelled_as_test(fcase):
    path = fcase.validation[(fcase.selected, SEEDS[0])]
    document = read(path)
    document["evaluation_split"] = "test"
    document["eval_stage"] = rt.EVAL_STAGES["test"]
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(TaskExecutionError):
        rs.aggregate_suite(fcase.r, SEEDS, fcase.root)


def test_suite_payload_rejects_changed_selection(fcase):
    result = rs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    data = read(result)
    data["selection"]["selected_config_id"] = "lr5e-4-r8"
    with pytest.raises(TaskExecutionError, match="配置选择"):
        rs.validate_suite_payload(fcase.r, data, result)


def test_suite_payload_rejects_unbound_file(fcase):
    result = rs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    (fcase.root / "extra-unbound.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TaskExecutionError, match="未绑定文件"):
        rs.validate_suite_payload(fcase.r, read(result), result)


def test_aggregate_refuses_to_overwrite_completed_suite(fcase):
    result = rs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    before = result.read_bytes()
    with pytest.raises(TaskExecutionError, match="拒绝覆盖"):
        rs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    assert result.read_bytes() == before


# ---------------------------------------------------------------- C306 / C307 门禁

def test_c306_dispatch_requires_frozen_base_test_zero_shot(fcase):
    r = fcase.r
    assert be.zero_train_measured(r) is False
    data = copy.deepcopy(r.data)
    data["baseline"]["measurement"] = str(
        fcase.base[rs.ACCEPTANCE_SPLIT].relative_to(r.path.parent))
    registered = Recipe(r.id, r.path, data)
    assert be.zero_train_measured(registered) is True
    # validation 的基座评估不能冒充 test 上的零训练基线。
    data["baseline"]["measurement"] = str(
        fcase.base[rs.SELECTION_SPLIT].relative_to(r.path.parent))
    assert be.zero_train_measured(Recipe(r.id, r.path, data)) is False


def test_c307_dispatch_without_completed_suite_fails_closed(fcase):
    assert any("baseline-suite" in p for p in be.suite_problems(fcase.r))
    data = copy.deepcopy(fcase.r.data)
    data["execution"]["evidence"] = "../../not-there.json"
    assert any("baseline-suite" in p for p in be.suite_problems(
        Recipe(fcase.r.id, fcase.r.path, data)))


def test_c307_rejects_undeclared_expected_and_tolerance(fcase):
    result = rs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    data = copy.deepcopy(fcase.r.data)
    data["execution"]["evidence"] = str(result.relative_to(fcase.r.path.parent))
    problems = be.suite_problems(Recipe(fcase.r.id, fcase.r.path, data))
    assert any("期望指标必须预先声明" in p for p in problems)
    assert any("容差必须为覆盖全部" in p for p in problems)
    assert not any("重跑均值超出验收容差" in p for p in problems)


def test_c307_accepts_declared_expectations_within_tolerance(fcase):
    result = rs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    suite = read(result)
    data = copy.deepcopy(fcase.r.data)
    data["evaluation"]["expected"] = {
        name: suite["acceptance_stats"][name]["trained"]["mean"] for name in rt.METRICS}
    data["evaluation"]["tolerance"] = {name: 0.0 for name in rt.METRICS}
    data["execution"]["evidence"] = str(result.relative_to(fcase.r.path.parent))
    problems = be.suite_problems(Recipe(fcase.r.id, fcase.r.path, data))
    assert problems == [], problems


# ---------------------------------------------------------------- 导出与 C405

def fake_reload(r, plan, checkpoint=None, split_name=None, **kwargs):
    """模拟「新进程真实复载」：只复制参考评估与其绑定文件到 reload/eval。"""
    assert split_name == rs.ACCEPTANCE_SPLIT
    assert plan["evaluation_split"] == rs.ACCEPTANCE_SPLIT
    bundle = Path(checkpoint).parent
    output = Path(plan["run"]["output_dir"])
    assert output == bundle / "reload" / "eval"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation.json", "source_manifest.json", rt.TOKENIZER_IDENTITY_FILE,
                 rt.EVAL_MANIFEST_FILE, rt.PREDICTIONS_FILE):
        shutil.copyfile(bundle / "reference" / name, output / name)
    return output / "evaluation.json"


def mock_worker(monkeypatch, destination):
    child_pid, parent_pid = os.getpid() + 10000, os.getpid()
    calls = []
    monkeypatch.setattr(rt, "run_evaluation", fake_reload)

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
                re.run_worker(destination)
            return "RapidFire 测试 worker 输出", ""

        process.communicate.side_effect = communicate
        return process

    monkeypatch.setattr(et.subprocess, "Popen", launch)
    return calls


def _export_case(fcase, tmp_path):
    checkpoint = fcase.checkpoints[(fcase.selected, SEEDS[0])]
    evaluation = fcase.test[SEEDS[0]]
    return checkpoint, evaluation, tmp_path / "bundle"


def test_export_plan_is_readonly_and_tolerance_fails_closed(fcase, tmp_path):
    checkpoint, evaluation, destination = _export_case(fcase, tmp_path)
    plan = re.export_bundle(fcase.r, checkpoint, evaluation, destination, execute=False)
    assert plan["kind"] == "export-plan" and plan["stage"] == "declared"
    assert plan["task_id"] == rt.TASK_ID and plan["evaluation_split"] == "test"
    assert plan["config_id"] == "lr5e-4-r8"
    assert plan["config_id_source"].startswith("execute 时读取")
    assert plan["base_weights_bundled"] is False and plan["adapter_only"] is True
    assert plan["quantization"] == "none" and plan["compute_dtype"] == "float16"
    assert plan["metric_comparable_to_upstream_rougeL"] is False
    assert plan["tolerance_status"] == "declared"
    assert not destination.exists()
    no_tolerance = _release_recipe(fcase.r.path.parent, tolerance=False)
    with pytest.raises(TaskExecutionError, match="tolerance"):
        re.export_bundle(no_tolerance, checkpoint, evaluation, destination, execute=True)


def test_export_bundle_copies_selected_adapter_and_reloads(fcase, tmp_path, monkeypatch):
    r = fcase.r
    checkpoint, evaluation, destination = _export_case(fcase, tmp_path)
    calls = mock_worker(monkeypatch, destination)
    result = re.export_bundle(r, checkpoint, evaluation, destination)
    evidence = read(result)
    assert result == destination / re.FINAL
    assert evidence["task_id"] == rt.TASK_ID and evidence["stage"] == "completed"
    assert evidence["config_id"] == fcase.selected == SELECTED
    assert evidence["seed"] == SEEDS[0]
    assert evidence["evaluation_split"] == "test"
    assert evidence["license"]["status"] == "pending"
    assert evidence["metric_comparable_to_upstream_rougeL"] is False
    assert evidence["concurrent_scheduling"] is False
    assert evidence["base_weights_bundled"] is False
    assert evidence["training_seed_variance_measured"] is False
    assert evidence["comparison"] == {
        name: {"reference": evidence["metrics"][name], "reload": evidence["metrics"][name],
               "absolute_delta": 0.0, "tolerance": 0.0, "within_tolerance": True}
        for name in rt.METRICS}
    assert len(calls) == 1 and "--worker" in calls[0][0]
    manifest = read(destination / re.MANIFEST)
    assert manifest["config_id"] == SELECTED
    assert manifest["base_weights_bundled"] is False
    assert "model/adapter_model.safetensors" in manifest["files"]
    assert "model/effective_recipe.json" in manifest["files"]
    assert "reference/eval_manifest.json" in manifest["files"]
    # 快照必须与训练时 effective_recipe 逐字节一致，锁位置不改写内容身份。
    snapshot = read(destination / re.SNAPSHOT)
    assert snapshot == read(checkpoint / "effective_recipe.json")["data"]
    assert snapshot["execution"]["config_id"] == SELECTED
    assert snapshot["execution"]["lockfile"] == LOCKFILE
    assert (destination / LOCKFILE).read_bytes() == fcase.lock.read_bytes()
    base = read(destination / re.BASE_REFERENCE)
    assert base["model_id"] == "gpt2" and base["quantization"] == "none"
    assert base["evaluation_split"] == "test" and base["adapter_only"] is True
    assert base["metric_comparable_to_upstream_rougeL"] is False
    assert re.validate_export_evidence(r, result) is True


def test_export_derives_config_from_checkpoint_not_default(fcase, tmp_path, monkeypatch):
    r = fcase.r
    checkpoint, evaluation, destination = _export_case(fcase, tmp_path)
    assert r.get("execution.config_id") == "lr5e-4-r8"
    effective = re.effective_recipe(r, checkpoint)
    assert effective.get("execution.config_id") == SELECTED
    assert effective.get("train.seed") == SEEDS[0]
    assert effective.get("train.learning_rate") == 2e-4
    assert effective.get("peft.r") == 32
    rt.require_supported(effective)
    assert be.fingerprint(effective) == read(checkpoint / "training.json")["fingerprint"]


def test_export_rejects_validation_evaluation_as_reference(fcase, tmp_path, monkeypatch):
    checkpoint, _, destination = _export_case(fcase, tmp_path)
    wrong = fcase.validation[(fcase.selected, SEEDS[0])]
    mock_worker(monkeypatch, destination)
    with pytest.raises(TaskExecutionError, match="test 评估"):
        re.export_bundle(fcase.r, checkpoint, wrong, destination)


def test_export_evidence_rejects_tampered_reload(fcase, tmp_path, monkeypatch):
    r = fcase.r
    checkpoint, evaluation, destination = _export_case(fcase, tmp_path)
    mock_worker(monkeypatch, destination)
    result = re.export_bundle(r, checkpoint, evaluation, destination)
    (destination / "reload" / "eval" / rt.PREDICTIONS_FILE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(TaskExecutionError):
        re.validate_export_evidence(r, result)


def test_c405_dispatch_uses_rapidfire_exporter(fcase, tmp_path, monkeypatch):
    r = fcase.r
    checkpoint, evaluation, destination = _export_case(fcase, tmp_path)
    mock_worker(monkeypatch, destination)
    result = re.export_bundle(r, checkpoint, evaluation, destination)
    data = copy.deepcopy(r.data)
    data["baseline_contract"] = {
        "export_evidence": str(result.parent.relative_to(r.path.parent) / result.name),
        "export_verified": True, "rollback_verified": True,
        "rollback_evidence": "manual/rollback-review.md"}
    registered = Recipe(r.id, r.path, data)
    check = {"kind": "export_reload", "field": "baseline_contract.export_evidence",
             "confirmation_field": "baseline_contract.export_verified"}
    assert bl.check_passed(registered, check) is True
    (destination / "reload" / "eval" / rt.PREDICTIONS_FILE).write_text("{}\n", encoding="utf-8")
    assert bl.check_passed(registered, check) is False


def test_worker_requires_offline_environment(fcase, tmp_path, monkeypatch):
    checkpoint, evaluation, destination = _export_case(fcase, tmp_path)
    mock_worker(monkeypatch, destination)
    result = re.export_bundle(fcase.r, checkpoint, evaluation, destination)
    assert read(result)["subprocess"]["offline_environment"] == et.OFFLINE_ENV
    with monkeypatch.context() as child:
        child.delenv("HF_HUB_OFFLINE", raising=False)
        with pytest.raises(TaskExecutionError, match="offline"):
            re.run_worker(destination)


def test_bundle_rejects_unbound_or_tampered_file(fcase, tmp_path, monkeypatch):
    checkpoint, evaluation, destination = _export_case(fcase, tmp_path)

    def stop(bundle):
        raise TaskExecutionError("测试停止于 prepared 包")

    with monkeypatch.context() as context:
        context.setattr(et, "_launch_worker", stop)
        with pytest.raises(TaskExecutionError, match="prepared"):
            re.export_bundle(fcase.r, checkpoint, evaluation, destination)
    assert (destination / re.MANIFEST).is_file() and not (destination / re.FINAL).exists()
    (destination / "model" / "unbound.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(TaskExecutionError, match="未绑定文件"):
        re._verify_bundle(destination)


# ---------------------------------------------------------------- CLI 只读

def test_release_cli_dry_runs_are_framework_free():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "RECIPE_LAB_ROOT": str(ROOT)}
    commands = [
        [sys.executable, str(ROOT / "runner" / "suite.py"), "--recipe", rt.RECIPE_ID,
         "--seeds", "42", "43", "44", "--output-root",
         "/tmp/recipe-lab-rapidfire-release-suite", "--dry-run"],
        [sys.executable, str(ROOT / "runner" / "export.py"), "--recipe", rt.RECIPE_ID,
         "--checkpoint", "/tmp/no-checkpoint", "--evaluation", "/tmp/no-eval.json",
         "--output", "/tmp/recipe-lab-rapidfire-release-bundle", "--dry-run"],
    ]
    for command in commands:
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["task_id"] == rt.TASK_ID
        assert plan.get("stage") == "declared" or plan.get("status") == "declared"
    assert not Path("/tmp/recipe-lab-rapidfire-release-suite").exists()
    assert not Path("/tmp/recipe-lab-rapidfire-release-bundle").exists()
