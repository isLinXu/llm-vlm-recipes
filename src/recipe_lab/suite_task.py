"""独立训练 seed 编排与内容聚合；标准库、失败关闭，不证明 GPU 来源。"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import baseline_evidence as be
from . import export_task as et
from . import math_task as mt
from . import provenance as pv

CODE_ROOT = Path(__file__).resolve().parents[2]
COMMAND_TIMEOUT_SECONDS = 86400
CODE_SNAPSHOT = "code-snapshot.json"
LOCKFILE = "environment.lock.toml"
STATE = "suite-state.json"
FINAL = "suite.json"
# 双侧 95% Student-t 分位数；df>30 用 df30 的较大系数，不退化为 normal。
T95 = (0, 12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912,
       2.364624, 2.306004, 2.262157, 2.228139, 2.200985, 2.178813, 2.160369,
       2.144787, 2.131450, 2.119905, 2.109816, 2.100922, 2.093024, 2.085963,
       2.079614, 2.073873, 2.068658, 2.063899, 2.059539, 2.055529, 2.051831,
       2.048407, 2.045230, 2.042272)
LIMITATION = ("统计单位为独立训练 seed；固定同一个 held-out test 和 greedy 协议。"
              "Student-t 区间假设训练 seed 结果独立同分布且均值近似正态；"
              "不包含数据集抽样不确定性，不能作为业务显著性证据。"
              "JSON 仅证明内部内容一致性，不证明日志实际来自 GPU。")


def _need(condition, message):
    if not condition:
        raise mt.TaskExecutionError(message)


def _seeds(seeds):
    try:
        result = list(seeds)
    except TypeError as exc:
        raise mt.TaskExecutionError("seeds 必须为至少三个互异整数") from exc
    _need(len(result) >= 3 and all(type(s) is int and 0 <= s < 2**32 for s in result),
          "至少三个 [0, 2**32) 整数训练 seed，不能是 bool")
    _need(len(set(result)) == len(result), "训练 seed 不得重复")
    return result


def _root(output_root):
    path = Path(output_root).expanduser()
    _need(path.is_absolute(), "--output-root 必须为 ABS 路径")
    return path


def suite_plan(r, seeds, output_root):
    """仅读配置声明；不读取快照、lock、checkpoint、输入证据，不写磁盘。"""
    seeds, root = _seeds(seeds), _root(output_root)
    try:
        mt.require_supported(r)
        supported, reason = True, None
    except mt.TaskExecutionError as exc:
        supported, reason = False, str(exc)
    def command(script, output, seed=None, checkpoint=None):
        args = [sys.executable, str(CODE_ROOT / "runner" / script), "--recipe", r.id,
                "--full", "--output-root", str(output)]
        if seed is not None:
            args += ["--train-seed", str(seed)]
        if checkpoint is not None:
            args += ["--checkpoint", str(checkpoint)]
        return args
    commands = [{"kind": "zero-shot", "command": command("zero_shot.py", root / "base"),
                 "output_dir": str(root / "base" / "zero-shot")}]
    runs = []
    for seed in seeds:
        effective = pv.effective_recipe(r, seed)
        seed_root = root / f"seed-{seed}"
        checkpoint = seed_root / "full"
        commands += [
            {"kind": "training", "train_seed": seed,
             "command": command("train.py", seed_root, seed), "output_dir": str(checkpoint)},
            {"kind": "evaluation", "train_seed": seed,
             "command": command("eval.py", seed_root, seed, checkpoint),
             "output_dir": str(seed_root / "eval"), "checkpoint": str(checkpoint)},
        ]
        runs.append({"train_seed": seed, "fingerprint": be.fingerprint(effective),
                     "checkpoint": str(checkpoint), "evaluation": str(seed_root / "eval" / "evaluation.json")})
    return {"schema_version": 1, "kind": "suite-plan", "status": "declared", "recipe_id": r.id,
            "task_id": mt.resolve_task_id(r), "fingerprint": be.fingerprint(r),
            "execution_supported": supported, "unsupported_reason": reason,
            "seeds": seeds, "output_root": str(root), "shell": False,
            "data_seed": r.get("train.data_seed"),
            "evaluation_seed": r.get("evaluation.seed", r.get("train.seed")),
            "training_source": {"id": r.get("model.base"), "revision": r.get("model.revision"),
                                "independent_initialization": True},
            "full_steps": r.get("train.max_steps", r.get("budget.full_steps")),
            "commands": commands, "runs": runs, "timeout_seconds": COMMAND_TIMEOUT_SECONDS,
            "resume_supported": False, "baseline_promotion": "none", "limitations": LIMITATION}


def _preflight(r, seeds, root, *, executing=False):
    mt.require_supported(r)
    _need(be.frozen_environment(r), "套件需要与 Recipe 一致的 frozen_environment")
    for key in ("model.revision", "dataset.revision", "source_sha"):
        mt.require_revision(r, key)
    mt.fixed_seeds(r)
    names = r.get("evaluation.metrics")
    _need(isinstance(names, list) and bool(names) and all(isinstance(n, str) for n in names)
          and len(names) == len(set(names))
          and set(names) <= {"accuracy", "pass@1", "format_compliance"}, "数学 metrics 声明非法")
    for seed in seeds:
        effective = pv.effective_recipe(r, seed)
        mt.config_kwargs(effective, {"mode": "full"}, root / f"seed-{seed}" / "full")
    if executing:
        _need(r.get("decision.gate") != "blocked" and r.get("baseline.result") != "passes",
              "训练决策门已阻断；不能绕过零训练基线及预算决策")


def _lock(r):
    return et._no_symlinks(r.path.parent / str(r.get("execution.lockfile")))


def _current_code():
    return pv.snapshot_bytes(pv.code_snapshot())


def _frozen_inputs(r, root):
    snapshot = et._local(root, CODE_SNAPSHOT)
    lock = et._local(root, LOCKFILE)
    code_bytes, lock_bytes = _current_code(), _lock(r).read_bytes()
    _need(snapshot.is_file() and snapshot.read_bytes() == code_bytes,
          "缺少预先冻结的 canonical code snapshot 或与当前代码不一致；不得编造历史快照")
    _need(lock.is_file() and lock.read_bytes() == lock_bytes,
          "冻结 environment lock 缺失或与当前 lock 字节不一致")
    return hashlib.sha256(code_bytes).hexdigest(), hashlib.sha256(lock_bytes).hexdigest()


def _guard(r, root, hashes):
    _need(_frozen_inputs(r, root) == hashes, "代码或 environment lock hash 已漂移")


def _manifest(r, manifest):
    _need(be.valid_manifest(manifest), "manifest hash / 去重 / 内容身份无效")
    _need(manifest.get("task_id") == mt.TASK_ID and manifest.get("dataset_id") == mt.DATASET_ID
          and manifest.get("dataset_revision") == r.get("dataset.revision"), "manifest 任务或 revision 不一致")
    for key, value in mt.split_spec().items():
        _need(manifest.get(key) == value, f"manifest 非严格 1500 协议：{key}")
    for name, (lo, hi) in mt.SPLIT_RANGES.items():
        split = manifest["splits"][name]
        _need(type(split.get("count")) is int and split["count"] == hi - lo,
              "manifest 必须有三个连续且各 500 条的 split")
        for index, sample in enumerate(split["samples"], lo):
            _need(type(sample.get("source_index")) is int and sample["source_index"] == index
                  and sample.get("sample_id") == f"train:{index}", "manifest 原始索引或样本 ID 协议不一致")


def _identity(r, data, hashes):
    _need(type(data.get("schema_version")) is int and data["schema_version"] == 1
          and data.get("task_id") == mt.TASK_ID and data.get("fingerprint") == be.fingerprint(r)
          and data.get("model_id") == mt.MODEL_ID and data.get("dataset_id") == mt.DATASET_ID
          and data.get("model_revision") == r.get("model.revision")
          and data.get("dataset_revision") == r.get("dataset.revision")
          and type(data.get("seed")) is int and data["seed"] == r.get("train.seed")
          and type(data.get("data_seed")) is int and data["data_seed"] == r.get("train.data_seed"),
          "证据任务、revision、有效 fingerprint 或 seed 不一致")
    _need(data.get("code_sha256") == hashes[0] and data.get("environment_lock_sha256") == hashes[1],
          "证据混合 code / lock hash 或与冻结快照不一致")
    _need(be.runtime_consistent(r, data) and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "非真实运行协议或 runtime 不一致")
    _need(bool(be.HEX64.fullmatch(str(data.get("recipe_sha256", "")))), "缺少 Recipe 内容 hash")


def _prediction_path(root, path, data):
    record = data.get("predictions_file", {})
    _need(isinstance(record, dict), "缺少预测文件记录")
    result = et._local(path.parent, record.get("path"))
    _need(result.is_relative_to(root) and be.hashed_file(path.parent, record), "预测文件缺失或 hash 不一致")
    return result


def _evaluation_content(r, root, path, data, manifest):
    _need(data.get("split_manifest") == manifest and et._json(et._local(path.parent, "manifest.json")) == manifest,
          "评估 manifest.json 或证据 split 与套件不一致")
    _need(data.get("evaluation_protocol") == "single_checkpoint_evaluation", "评估协议不一致")
    config, model = data.get("config", {}), data.get("model_artifact", {})
    _need(config.get("task_id") == mt.TASK_ID and config.get("system_prompt") == mt.SYSTEM_PROMPT
          and config.get("model") == model and config.get("dataset_revision") == r.get("dataset.revision")
          and config.get("test_sha256") == manifest["splits"]["test"]["sha256"], "评估固定 prompt 或输入 config 不一致")
    generation = config.get("generation", {})
    _need(set(generation) == {"seed", "do_sample", "num_beams", "max_new_tokens", "pad_token_id", "eos_token_id", "use_cache"}
          and type(generation.get("seed")) is int
          and generation["seed"] == r.get("evaluation.seed", r.get("train.seed"))
          and generation.get("do_sample") is False and type(generation.get("num_beams")) is int
          and generation["num_beams"] == 1 and type(generation.get("max_new_tokens")) is int
          and generation["max_new_tokens"] == r.get("train.max_completion_length")
          and type(generation.get("eos_token_id")) is int and generation["eos_token_id"] >= 0
          and generation.get("pad_token_id") == generation["eos_token_id"]
          and type(generation.get("pad_token_id")) is int and generation.get("use_cache") is True,
          "评估不是固定 generation 协议")
    prediction_path = _prediction_path(root, path, data)
    predictions = et._json(prediction_path)
    description = data.get("predictions", {})
    _need(isinstance(predictions, list) and len(predictions) == 500
          and description.get("path") == data["predictions_file"]["path"]
          and type(description.get("count")) is int and description["count"] == len(predictions)
          and description.get("sha256") == mt.sha256_json(predictions), "逐题预测内容 hash 或数量不一致")
    for prediction, sample in zip(predictions, manifest["splits"]["test"]["samples"]):
        prompt, solution = prediction.get("prompt"), prediction.get("solution")
        _need(isinstance(prompt, list) and len(prompt) == 2 and isinstance(prompt[1], dict)
              and isinstance(prompt[1].get("content"), str) and bool(prompt[1]["content"].strip())
              and prompt == mt.make_prompt(prompt[1]["content"])
              and mt.sha256_json(prompt[1]["content"]) == sample["problem_sha256"]
              and isinstance(solution, str) and bool(solution.strip())
              and mt.sha256_json(solution) == sample.get("solution_sha256")
              and isinstance(prediction.get("completion"), str), "逐题 prompt / gold / completion 非法")
        compliant = int(mt.answer_text(prediction["completion"]) is not None)
        _need(prediction.get("format_compliance") == compliant
              and (prediction.get("accuracy") != 1 or compliant == 1), "逐题格式分数与输出不一致")
    for metric, field in (("accuracy", "accuracy"), ("pass@1", "accuracy"), ("format_compliance", "format_compliance")):
        _need(all(type(p.get(field)) is int and p[field] in (0, 1) for p in predictions), "数学逐题分数非法")
        if metric in data.get("metrics", {}):
            actual = data["metrics"][metric]
            _need(be.finite_number(actual) and math.isclose(actual, sum(p[field] for p in predictions) / 500,
                                                           rel_tol=0, abs_tol=1e-12), "数学指标与逐题重算不一致")
    counts = data.get("metric_counts")
    if counts is not None:
        _need(counts == {"correct": sum(p["accuracy"] for p in predictions),
                         "format_compliant": sum(p["format_compliance"] for p in predictions), "total": 500},
              "metric_counts 与预测不一致")
    return predictions, prediction_path


def _stats(values):
    n = len(values)
    _need(n >= 3 and all(be.finite_number(v) for v in values), "统计至少需要三个有限 seed 观测")
    mean = math.fsum(values) / n
    variance = math.fsum((v - mean)**2 for v in values) / (n - 1)
    std, sem = math.sqrt(variance), math.sqrt(variance / n)
    df = n - 1
    coefficient = T95[min(df, 30)]
    return {"n": n, "mean": mean, "sample_variance": variance, "std": std, "sem": sem,
            "student_t95_ci": {"lower": mean - coefficient * sem, "upper": mean + coefficient * sem,
                               "df": df, "critical_df": min(df, 30), "critical_value": coefficient,
                               "conservative_df30": df > 30}}


def _acceptance(r, stats):
    expected, tolerance = r.get("evaluation.expected"), r.get("evaluation.tolerance")
    names = r.get("evaluation.metrics")
    declared = (isinstance(expected, dict) and isinstance(tolerance, dict)
                and set(names) <= set(expected) and set(names) <= set(tolerance)
                and all(be.finite_number(expected[n]) and be.finite_number(tolerance[n])
                        and tolerance[n] >= 0 for n in names))
    if not declared:
        return {"status": "not_evaluated", "reason": "expected / tolerance 未完整声明；仅完成测量，不自动回填或晋级 B2"}
    comparisons = {n: {"expected": expected[n], "tolerance": tolerance[n],
                       "absolute_delta": abs(stats[n]["trained"]["mean"] - expected[n]),
                       "within_tolerance": abs(stats[n]["trained"]["mean"] - expected[n]) <= tolerance[n]}
                   for n in names}
    return {"status": "passed" if all(c["within_tolerance"] for c in comparisons.values()) else "failed",
            "comparisons": comparisons, "business_significance": "not_established"}


def aggregate_suite(r, seeds, output_root):
    """只读既有真实证据，最后仅写 root/suite.json；不合成或修复历史输入。"""
    seeds, root = _seeds(seeds), et._no_symlinks(_root(output_root))
    _need(root.is_dir(), "aggregate-only root 不存在")
    _need(not (root / FINAL).exists(), "suite.json 已存在；拒绝覆盖 completed 套件")
    _preflight(r, seeds, root)
    hashes = _frozen_inputs(r, root)
    records = {}
    def bind(path):
        _need(path.is_file(), f"缺少证据文件：{path}")
        record = et._record(path, root)
        previous = records.get(record["path"])
        _need(previous is None or previous == record, f"聚合期间证据内容漂移：{record['path']}")
        records[record["path"]] = record
        return record
    bind(et._local(root, CODE_SNAPSHOT))
    bind(et._local(root, LOCKFILE))
    base_path = et._local(root, "base/zero-shot/evaluation.json")
    bind(base_path)
    base = et._json(base_path)
    _need(isinstance(base, dict), "base evaluation 必须为对象")
    problems = be.evaluation_problems(r, base, base_path, role="base")
    _need(not problems, "base 评估拒绝：" + "；".join(problems))
    _identity(r, base, hashes)
    _need(base.get("mode") == "zero-shot" and base.get("model_artifact") == {
        "id": mt.MODEL_ID, "revision": r.get("model.revision"), "role": "base"}, "base artifact 或模式不一致")
    manifest = base["split_manifest"]
    _manifest(r, manifest)
    manifest_path = et._local(base_path.parent, "manifest.json")
    bind(manifest_path)
    base_predictions, base_prediction_path = _evaluation_content(r, root, base_path, base, manifest)
    bind(base_prediction_path)
    runs, checkpoints, artifacts = [], set(), []
    for seed in seeds:
        effective = pv.effective_recipe(r, seed)
        checkpoint = et._local(root, f"seed-{seed}/full")
        # 不允许 symlink 权重/目录，也不接受多个 seed 指向同一目录。
        et._tree_files(checkpoint)
        _need(checkpoint not in checkpoints, "重复 checkpoint 目录")
        checkpoints.add(checkpoint)
        training_path = et._local(checkpoint, "training.json")
        bind(training_path)
        mt.checkpoint_path({}, checkpoint, recipe=effective)
        training = et._training(effective, checkpoint)
        _identity(effective, training, hashes)
        expected_config = mt.config_kwargs(effective, {"mode": "full"}, checkpoint)
        integer_keys = {"seed", "data_seed", "max_steps", "per_device_train_batch_size", "num_generations",
                        "gradient_accumulation_steps", "max_completion_length"}
        _need(all(training["config"].get(k) == v and (k not in integer_keys or type(training["config"][k]) is int)
                  for k, v in expected_config.items() if k not in ("output_dir", "report_to"))
              and training["config"].get("remove_unused_columns") is False,
              "训练未应用 full 有效配置或默认参数")
        _need(training.get("split_manifest") == manifest, "训练 manifest 与套件不一致")
        training_manifest = et._local(checkpoint, "manifest.json")
        bind(training_manifest)
        _need(et._json(training_manifest) == manifest, "训练 manifest.json 不一致")
        artifact = mt.artifact_manifest(checkpoint)
        _need(training["model_artifact"].get("path") == str(checkpoint), "训练 checkpoint artifact 路径不一致")
        for name in artifact["files"]:
            bind(et._local(checkpoint, name))
        artifacts.append((checkpoint, artifact))
        evaluation_path = et._local(root, f"seed-{seed}/eval/evaluation.json")
        bind(evaluation_path)
        evaluation, _ = et._evaluation(effective, evaluation_path, artifact, recipe_sha=training["recipe_sha256"])
        _identity(effective, evaluation, hashes)
        _need(evaluation.get("mode") == "eval", "trained 评估必须使用 eval 模式")
        bind(et._local(evaluation_path.parent, "manifest.json"))
        predictions, prediction_path = _evaluation_content(effective, root, evaluation_path, evaluation, manifest)
        bind(prediction_path)
        _need(base["config"]["generation"] == evaluation["config"]["generation"], "base / trained generation 不一致")
        for before, after in zip(base_predictions, predictions):
            _need(all(before.get(k) == after.get(k) for k in ("sample_id", "content_sha256", "prompt", "solution")),
                  "base / trained 逐题输入、prompt 或 gold 不一致")
        runs.append({"kind": "paired-evaluation", "training_completed": True, "train_seed": seed,
                     "fingerprint": be.fingerprint(effective), "sample_count": 500,
                     "split_sha256": manifest["sha256"], "training_evidence": bind(training_path),
                     "base_evaluation": bind(base_path), "trained_evaluation": bind(evaluation_path),
                     "predictions": bind(prediction_path), "base_predictions": bind(base_prediction_path),
                     "model_artifact": artifact, "base_metrics": base["metrics"], "trained_metrics": evaluation["metrics"]})
    stats = {name: {"trained": _stats([run["trained_metrics"][name] for run in runs]),
                    "paired_delta": _stats([run["trained_metrics"][name] - run["base_metrics"][name] for run in runs])}
             for name in r.get("evaluation.metrics")}
    result = {"schema_version": 1, "kind": "baseline-suite", "status": "completed", "task_id": mt.TASK_ID,
              "recipe_id": r.id, "fingerprint": be.fingerprint(r), "completed_at": datetime.now(timezone.utc).isoformat(),
              "model_revision": r.get("model.revision"), "dataset_revision": r.get("dataset.revision"),
              "code_sha256": hashes[0], "environment_lock_sha256": hashes[1],
              "code_snapshot": bind(root / CODE_SNAPSHOT), "environment_lock": bind(root / LOCKFILE),
              "split_manifest": bind(manifest_path), "split_sha256": manifest["sha256"], "runs": runs,
              "stats": stats, "statistical_unit": "independent_training_seed", "fixed_test": True,
              "acceptance": _acceptance(r, stats), "records": records,
              "training_seed_variance_measured": True, "baseline_promotion": "none", "limitations": LIMITATION}
    _guard(r, root, hashes)
    for record in records.values():
        _need(be.hashed_file(root, record), f"聚合期间证据内容漂移：{record['path']}")
    for checkpoint, artifact in artifacts:
        _need(mt.artifact_manifest(checkpoint) == artifact, "聚合期间 checkpoint 权重漂移")
    _need(not (root / FINAL).exists(), "聚合期间 suite.json 已被创建，拒绝覆盖")
    return mt.save_json(root / FINAL, result)


def validate_suite_payload(r, data, path):
    """发布审计共用内容核验；只读，可核验任意目录布局但不接受旧宽松套件。"""
    root = et._no_symlinks(path.parent)
    _need(isinstance(data, dict) and data.get("task_id") == mt.TASK_ID, "套件必须绑定数学任务")
    runs = data.get("runs")
    _need(isinstance(runs, list) and all(isinstance(run, dict) for run in runs), "套件runs非法")
    seeds = _seeds([run.get("train_seed") for run in runs])
    _preflight(r, seeds, root)
    hashes = _frozen_inputs(r, root)
    _need(data.get("code_sha256") == hashes[0] and data.get("environment_lock_sha256") == hashes[1], "套件code/lock身份不一致")
    records = data.get("records")
    _need(isinstance(records, dict) and bool(records), "套件缺少完整文件records")
    for relative, record in records.items():
        _need(isinstance(record, dict) and record.get("path") == relative and be.hashed_file(root, record), "套件绑定文件校验失败")
        et._local(root, relative)
    def linked(record):
        _need(isinstance(record, dict) and records.get(record.get("path")) == record, "关联证据没有绑定到records")
        _need(be.hashed_file(root, record), "关联证据hash不一致")
        file = et._local(root, record["path"])
        return et._json(file), file
    _need(data.get("code_snapshot") == records.get(CODE_SNAPSHOT)
          and data.get("environment_lock") == records.get(LOCKFILE), "套件快照/锁记录不一致")
    manifest, manifest_path = linked(data.get("split_manifest"))
    _manifest(r, manifest)
    _need(data.get("split_sha256") == manifest["sha256"], "套件split身份不一致")
    checkpoints = set()
    shared_base = None
    for run, seed in zip(runs, seeds):
        effective = pv.effective_recipe(r, seed)
        training, training_path = linked(run.get("training_evidence"))
        _need(isinstance(training, dict), "训练不是JSON对象")
        checkpoint = training_path.parent
        _need(checkpoint not in checkpoints, "独立seed共用checkpoint目录")
        checkpoints.add(checkpoint)
        et._tree_files(checkpoint)
        mt.checkpoint_path({}, checkpoint, recipe=effective)
        et._training(effective, checkpoint)
        _identity(effective, training, hashes)
        _need(training.get("split_manifest") == manifest, "训练manifest与套件不一致")
        manifest_file = et._local(checkpoint, "manifest.json")
        _need(et._json(manifest_file) == manifest, "训练manifest文件不一致")
        _need(records.get(manifest_file.relative_to(root).as_posix()) == et._record(manifest_file, root), "训练manifest未绑定")
        config = mt.config_kwargs(effective, {"mode": "full"}, checkpoint)
        _need(all(training["config"].get(k) == v and type(training["config"].get(k)) is type(v)
                  for k, v in config.items() if k not in ("output_dir", "report_to")), "训练有效参数/类型不一致")
        artifact = mt.artifact_manifest(checkpoint)
        for name in artifact["files"]:
            file = et._local(checkpoint, name)
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root), "checkpoint文件未绑定或变化")
        base, base_path = linked(run.get("base_evaluation"))
        trained, trained_path = linked(run.get("trained_evaluation"))
        _identity(r, base, hashes)
        _identity(effective, trained, hashes)
        _need(not be.evaluation_problems(r, base, base_path, role="base", manifest=manifest), "基座评估内容不一致")
        _need(base.get("mode") == "zero-shot" and base.get("model_artifact") == {
            "id": mt.MODEL_ID, "revision": r.get("model.revision"), "role": "base"}, "基座来源不一致")
        _need(shared_base is None or shared_base == run["base_evaluation"], "套件应共享同一次固定基座评估")
        shared_base = run["base_evaluation"]
        et._evaluation(effective, trained_path, artifact, recipe_sha=training["recipe_sha256"])
        _need(trained.get("mode") == "eval", "训练模型评估模式不一致")
        before, base_predictions = _evaluation_content(r, root, base_path, base, manifest)
        after, trained_predictions = _evaluation_content(effective, root, trained_path, trained, manifest)
        _need(base["config"]["generation"] == trained["config"]["generation"], "配对解码协议不一致")
        _need(all(all(a.get(k) == b.get(k) for k in ("sample_id", "content_sha256", "prompt", "solution"))
                  for a, b in zip(before, after)), "配对输入/gold不同")
        _need(run.get("predictions") == et._record(trained_predictions, root)
              and run.get("base_predictions") == et._record(base_predictions, root), "预测记录与关联评估不同")
        for file in (base_predictions, trained_predictions, base_path.parent / "manifest.json", trained_path.parent / "manifest.json"):
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root), "评估输入未绑定到records")
        _need(run.get("fingerprint") == be.fingerprint(effective) and run.get("model_artifact") == artifact, "运行摘要身份不一致")
    stats = {name: {"trained": _stats([run["trained_metrics"][name] for run in runs]),
                    "paired_delta": _stats([run["trained_metrics"][name] - run["base_metrics"][name] for run in runs])}
             for name in r.get("evaluation.metrics")}
    _need(data.get("stats") == stats and data.get("statistical_unit") == "independent_training_seed"
          and data.get("fixed_test") is True, "套件seed统计未重算或统计单位错误")
    # acceptance是生成时的目标快照；当前发布资格仍由baseline门禁重算，不能沿用旧passed。
    _guard(r, root, hashes)


def _launch(command, root, index):
    log_dir = et._local(root, "logs")
    log_dir.mkdir(exist_ok=True)
    stdout_path, stderr_path = log_dir / f"{index:02d}.stdout.log", log_dir / f"{index:02d}.stderr.log"
    process = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(command, cwd=CODE_ROOT, env={**os.environ,
                                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false"},
                                       shell=False, stdout=stdout, stderr=stderr)
            process.wait(timeout=COMMAND_TIMEOUT_SECONDS)
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            raise
    _need(process.returncode == 0, f"子命令失败 returncode={process.returncode}；保留日志 {stderr_path}")
    return {"command": command, "pid": process.pid, "returncode": process.returncode,
            "stdout": et._record(stdout_path, root), "stderr": et._record(stderr_path, root)}


def run_suite(r, seeds, output_root):
    """在新 root 顺序实跑一次 base 与独立 full/eval；失败不允许 resume。"""
    seeds, root = _seeds(seeds), et._no_symlinks(_root(output_root))
    _preflight(r, seeds, root, executing=True)
    _need(not root.exists() or (root.is_dir() and not any(root.iterdir())), "拒绝非空 root；失败后仅允许 aggregate-only 或新 root")
    plan = suite_plan(r, seeds, root)
    code_bytes, lock_bytes = _current_code(), _lock(r).read_bytes()
    hashes = hashlib.sha256(code_bytes).hexdigest(), hashlib.sha256(lock_bytes).hexdigest()
    recipe_hash = pv.file_hash(r.path)
    root.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "kind": "suite-execution", "status": "running", "plan": plan,
             "code_sha256": hashes[0], "environment_lock_sha256": hashes[1], "subprocesses": [],
             "resume_supported": False, "baseline_promotion": "none"}
    try:
        (root / CODE_SNAPSHOT).write_bytes(code_bytes)
        (root / LOCKFILE).write_bytes(lock_bytes)
        mt.save_json(root / STATE, state)
        for index, step in enumerate(plan["commands"]):
            state["active_command"] = step["command"]
            mt.save_json(root / STATE, state)
            _guard(r, root, hashes)
            _need(pv.file_hash(r.path) == recipe_hash, "执行期间 Recipe 文件漂移")
            try:
                info = _launch(step["command"], root, index)
            finally:
                _guard(r, root, hashes)
                _need(pv.file_hash(r.path) == recipe_hash, "执行期间 Recipe 文件漂移")
            state["subprocesses"].append(info)
            mt.save_json(root / STATE, state)
        state["status"] = "commands_completed"
        mt.save_json(root / STATE, state)
        return aggregate_suite(r, seeds, root)
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                     recovery="仅允许 --aggregate-only 验证确实已完成的输入，或新 root 重跑；不支持 resume")
        mt.save_json(root / STATE, state)
        raise
