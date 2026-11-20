"""gsm8k-reward-v1 drop-one reward 消融 × 独立训练 seed 编排；标准库、失败关闭。

五个消融配置（full / no_format_exact / no_format_approx / no_correctness /
no_extraction）都在**同一 Qwen2.5-3B backbone**内进行：超参逐字相同，唯一变量是
被去掉的 reward 函数。逐题从 held-out prompt + #### 剥离后的 gold 用独立严格
Decimal verifier 重算 pass@1 / format_compliance / reward_hacking_rate，绝不把
证据摘要加总一遍。JSON 只证明内部一致性，不证明 GPU 来源，不自动晋级 B2。
"""
from __future__ import annotations

import copy
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import baseline_evidence as be
from . import export_task as et
from . import gsm8k_task as gt
from . import math_task as mt
from . import provenance as pv
from . import suite_task as st
from .model import Recipe

CODE_ROOT = st.CODE_ROOT
CODE_SNAPSHOT, LOCKFILE, STATE, FINAL = st.CODE_SNAPSHOT, st.LOCKFILE, st.STATE, st.FINAL
LIMITATION = ("统计单位为每 reward 消融配置内的独立训练 seed；固定同一官方 test held-out "
              "与 greedy 协议。drop-one 消融（full / no_format_exact / no_format_approx / "
              "no_correctness / no_extraction）都在同一 Qwen2.5-3B backbone 内进行，"
              "四函数等权相加未经量纲归一化。Student-t 区间假设每配置内 seed 结果独立同分布 "
              "且均值近似正态，不包含数据抽样不确定性，不能作为业务显著性证据。"
              "reward_hacking_rate 只是诊断信号。JSON 只证明内部一致性，不证明 GPU 来源。")
METRICS = gt.METRICS
CONFIG_IDS = tuple(gt.ABLATION_CONFIGS)
PREDICTION_IDENTITY_KEYS = ("sample_id", "content_sha256", "prompt", "gold_answer")


def _need(condition, message):
    if not condition:
        raise mt.TaskExecutionError(message)


def identity_recipe(r):
    data = copy.deepcopy(r.data)
    evaluation = data.get("evaluation")
    if isinstance(evaluation, dict):
        for key in ("expected", "tolerance", "export_tolerance"):
            evaluation.pop(key, None)
    return Recipe(r.id, r.path, data)


def _hashes(r):
    identity = mt.execution_identity(r)
    return identity["code_sha256"], identity["environment_lock_sha256"]


def _config_recipe(r, seed, config_id):
    _need(config_id in gt.ABLATION_CONFIGS, "未知 GSM8K reward 消融 config_id")
    return gt.config_recipe(pv.effective_recipe(r, seed), config_id)


def _run_dir(root, config_id, seed):
    return root / f"config-{config_id}" / f"seed-{seed}"


def _metrics(r):
    names = r.get("evaluation.metrics")
    _need(isinstance(names, list) and len(names) == len(METRICS) and set(names) == set(METRICS),
          "GSM8K evaluation.metrics 必须精确为 pass@1 / format_compliance / reward_hacking_rate")
    return names


def _preflight(r, seeds, root, *, executing=False):
    gt.require_supported(r)
    _need(be.frozen_environment(r), "套件需要与 Recipe 一致的 frozen_environment")
    for key in ("model.revision", "dataset.revision", "source_sha"):
        mt.require_revision(r, key)
    mt.fixed_seeds(r)
    _metrics(r)
    for config_id in CONFIG_IDS:
        for seed in seeds:
            effective = _config_recipe(r, seed, config_id)
            gt.config_kwargs(identity_recipe(effective),
                             {"mode": "full", "recipe_id": effective.id, "task_id": gt.TASK_ID},
                             _run_dir(root, config_id, seed) / "full")
    if executing:
        _need(r.get("decision.gate") != "blocked" and r.get("baseline.result") != "passes",
              "训练决策门已阻断；不能绕过零训练基线及预算决策")


def suite_plan(r, seeds, output_root):
    """仅读配置声明；不读取快照、lock、checkpoint 或输入证据，不写磁盘。"""
    seeds, root = st._seeds(seeds), st._root(output_root)
    try:
        gt.require_supported(r)
        supported, reason = True, None
    except mt.TaskExecutionError as exc:
        supported, reason = False, str(exc)

    def command(script, output, seed=None, checkpoint=None, config_id=None):
        args = [sys.executable, str(CODE_ROOT / "runner" / script), "--recipe", r.id,
                "--full", "--output-root", str(output)]
        if seed is not None:
            args += ["--train-seed", str(seed)]
        if config_id is not None:
            args += ["--config-id", str(config_id)]
        if checkpoint is not None:
            args += ["--checkpoint", str(checkpoint)]
        return args

    commands = [{"kind": "zero-shot", "command": command("zero_shot.py", root / "base"),
                 "output_dir": str(root / "base" / "zero-shot")}]
    runs = []
    for config_id in CONFIG_IDS:
        for seed in seeds:
            effective = _config_recipe(r, seed, config_id)
            seed_root = _run_dir(root, config_id, seed)
            checkpoint = seed_root / "full"
            commands += [
                {"kind": "training", "train_seed": seed, "config_id": config_id,
                 "command": command("train.py", seed_root, seed, config_id=config_id),
                 "output_dir": str(checkpoint)},
                {"kind": "evaluation", "train_seed": seed, "config_id": config_id,
                 "command": command("eval.py", seed_root, seed, checkpoint, config_id),
                 "output_dir": str(seed_root / "eval"), "checkpoint": str(checkpoint)},
            ]
            runs.append({"config_id": config_id, "train_seed": seed,
                         "fingerprint": be.fingerprint(effective),
                         "active_rewards": list(gt.ABLATION_CONFIGS[config_id]),
                         "disabled_reward": gt.verify_ablation(config_id)["disabled_reward"],
                         "checkpoint": str(checkpoint),
                         "evaluation": str(seed_root / "eval" / "evaluation.json")})
    return {"schema_version": 1, "kind": "suite-plan", "status": "declared", "recipe_id": r.id,
            "task_id": gt.TASK_ID, "fingerprint": be.fingerprint(r),
            "execution_supported": supported, "unsupported_reason": reason,
            "seeds": seeds, "ablation_configs": list(CONFIG_IDS), "output_root": str(root),
            "shell": False, "data_seed": r.get("train.data_seed"),
            "evaluation_seed": r.get("evaluation.seed", r.get("train.seed")),
            "training_source": {"id": gt.MODEL_ID, "revision": r.get("model.revision"),
                                "independent_initialization": True, "full_weights": True,
                                "adapter_only": False},
            "protocol": gt.EVAL_PROTOCOL, "metrics": list(METRICS),
            "ablation_is_within_backbone": True, "single_variable": "被去掉的 reward 函数",
            "full_steps": r.get("train.max_steps", r.get("budget.full_steps")),
            "commands": commands, "runs": runs, "timeout_seconds": st.COMMAND_TIMEOUT_SECONDS,
            "resume_supported": False, "baseline_promotion": "none", "limitations": LIMITATION}


def _manifest(r, manifest):
    gt.validate_source_manifest(manifest, r)
    _need(manifest.get("dataset_revision") == r.get("dataset.revision"),
          "source_manifest dataset_revision 不一致")
    return manifest


def _identity(r, data, hashes):
    config_id = gt.config_id_of(r)
    _need(type(data.get("schema_version")) is int and data["schema_version"] == 1
          and data.get("stage") == "completed" and data.get("task_id") == gt.TASK_ID
          and data.get("recipe_id") == r.id and data.get("fingerprint") == be.fingerprint(r)
          and data.get("model_id") == gt.MODEL_ID and data.get("dataset_id") == gt.DATASET_ID
          and data.get("model_revision") == r.get("model.revision")
          and data.get("dataset_revision") == r.get("dataset.revision")
          and data.get("config_id") == config_id
          and type(data.get("seed")) is int and data["seed"] == r.get("train.seed")
          and type(data.get("data_seed")) is int and data["data_seed"] == r.get("train.data_seed"),
          "GSM8K 证据任务、config_id、revision、fingerprint 或 seed 不一致")
    _need(data.get("code_sha256") == hashes[0] and data.get("environment_lock_sha256") == hashes[1],
          "证据混合 code / lock hash 或与当前执行身份不一致")
    _need(be.runtime_consistent(r, data) and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "非真实运行协议或 runtime 不一致")
    _need(bool(be.HEX64.fullmatch(str(data.get("recipe_sha256", "")))), "缺少 Recipe 内容 hash")


def _generation(r, tokenizer_eos):
    return {"seed": r.get("evaluation.seed", r.get("train.seed")), "do_sample": False,
            "num_beams": 1, "max_new_tokens": r.get("train.max_completion_length"),
            "pad_token_id": tokenizer_eos, "eos_token_id": tokenizer_eos, "use_cache": True}


def _evaluation_content(r, path, data, manifest):
    """逐题从 held-out prompt + gold 用独立严格 verifier 重算，不信任摘要分数。"""
    _need(data.get("split_manifest") == manifest
          and et._json(et._local(path.parent, "manifest.json")) == manifest,
          "评估 manifest.json 或证据 split 与套件不一致")
    _need(data.get("evaluation_protocol") == gt.EVAL_PROTOCOL, "评估协议不一致")
    config = data.get("config")
    _need(isinstance(config, dict) and data.get("config_sha256") == gt.sha256_json(config),
          "评估配置内容 hash 不一致")
    model = data.get("model_artifact", {})
    generation = config.get("generation", {})
    _need(config.get("task_id") == gt.TASK_ID and config.get("system_prompt") == gt.SYSTEM_PROMPT
          and config.get("model") == model and config.get("dataset_revision") == r.get("dataset.revision")
          and config.get("test_sha256") == manifest["splits"]["test"]["sha256"]
          and config.get("verifier") == "strict-decimal",
          "评估固定 prompt / gold / verifier 协议不一致")
    _need(generation.get("do_sample") is False and generation.get("num_beams") == 1
          and type(generation.get("seed")) is int
          and generation["seed"] == r.get("evaluation.seed", r.get("train.seed"))
          and type(generation.get("max_new_tokens")) is int
          and generation["max_new_tokens"] == r.get("train.max_completion_length")
          and type(generation.get("eos_token_id")) is int and generation["eos_token_id"] >= 0
          and generation.get("pad_token_id") == generation["eos_token_id"]
          and generation.get("use_cache") is True,
          "评估不是固定 greedy 协议")
    record = data.get("predictions_file", {})
    _need(isinstance(record, dict), "缺少预测文件记录")
    prediction_path = et._local(path.parent, record.get("path"))
    _need(be.hashed_file(path.parent, record), "预测文件缺失或 hash 不一致")
    predictions = et._json(prediction_path)
    description = data.get("predictions", {})
    test_samples = manifest["splits"]["test"]["samples"]
    _need(isinstance(predictions, list) and len(predictions) == len(test_samples) == data.get("sample_count")
          and description.get("path") == record.get("path")
          and type(description.get("count")) is int and description["count"] == len(predictions)
          and description.get("sha256") == gt.sha256_json(predictions)
          and data.get("predictions_sha256", description["sha256"]) == description["sha256"],
          "逐题预测内容 hash 或数量不一致")
    for prediction, sample in zip(predictions, test_samples):
        prompt, gold = prediction.get("prompt"), prediction.get("gold_answer")
        _need(isinstance(prompt, list) and len(prompt) == 2 and isinstance(prompt[1], dict)
              and isinstance(prompt[1].get("content"), str) and bool(prompt[1]["content"].strip())
              and prompt == gt.make_prompt(prompt[1]["content"])
              and gt.sha256_json(gt.normalize_question(prompt[1]["content"])) == sample["problem_sha256"]
              and prediction.get("sample_id") == sample["sample_id"]
              and prediction.get("content_sha256") == sample["content_sha256"]
              and isinstance(gold, str) and bool(gold.strip())
              and gt.sha256_json(gold) == sample.get("gold_sha256")
              and isinstance(prediction.get("completion"), str)
              and gt.GOLD_DELIMITER not in prompt[1]["content"],
              "逐题 prompt / gold / completion 非法或 gold 泄漏进 prompt")
        scored = gt.score_completion(prediction["completion"], gold)
        for key in ("accuracy", "format_compliance", "reward_hacking", "parse_status",
                    "near_miss", "soft_format_only"):
            _need(prediction.get(key) == scored[key], f"逐题 {key} 与严格 verifier 重算不一致")
        _need(prediction.get("pass@1", prediction.get("accuracy")) in (0, 1, scored["accuracy"]),
              "逐题 pass@1 / accuracy 不一致")
    recomputed = gt.recompute_metrics(predictions)
    _need(data.get("metrics") == recomputed, "GSM8K 指标与逐题严格 verifier 重算不一致")
    counts = data.get("metric_counts")
    _need(isinstance(counts, dict) and counts.get("total") == len(predictions)
          and counts.get("correct") == sum(p["accuracy"] for p in predictions)
          and counts.get("format_compliant") == sum(p["format_compliance"] for p in predictions)
          and counts.get("reward_hacking") == sum(p["reward_hacking"] for p in predictions),
          "metric_counts 与逐题预测不一致")
    _need(data.get("diagnostics") == gt.diagnostics(predictions),
          "diagnostics 与逐题预测不一致")
    return predictions, prediction_path


def _training(r, checkpoint, hashes, manifest):
    clean = identity_recipe(r)
    gt.checkpoint_path({}, checkpoint, recipe=clean)
    training = et._json(et._local(checkpoint, "training.json"))
    _identity(clean, training, hashes)
    _need(training.get("kind") == "training" and training.get("mode") == "full"
          and training.get("trainer") == "GRPOTrainer", "不是 full GRPOTrainer 训练证据")
    expected = gt.config_kwargs(clean, {"recipe_id": clean.id, "task_id": gt.TASK_ID, "mode": "full"},
                                checkpoint)
    config = training.get("config")
    _need(isinstance(config, dict) and training.get("config_sha256") == gt.sha256_json(config),
          "训练配置内容 hash 不一致")
    for key, value in expected.items():
        if key in ("output_dir", "report_to"):
            continue
        _need(config.get(key) == value and type(config.get(key)) is type(value),
              f"训练有效参数不一致：{key}")
    _need(config.get("remove_unused_columns") is False, "训练必须保留 answer 列")
    ablation = gt.verify_ablation(gt.config_id_of(clean), training.get("active_rewards")
                                  or training.get("rewards"))
    _need(training.get("config_id") == ablation["config_id"]
          and training.get("active_rewards", training.get("rewards")) == ablation["active_rewards"]
          and training.get("disabled_reward", ablation["disabled_reward"]) == ablation["disabled_reward"],
          "训练 reward 消融 config_id / active 名单不一致")
    max_steps = clean.get("train.max_steps", clean.get("budget.full_steps"))
    _need(type(training.get("steps")) is int and training["steps"] == training.get("max_steps")
          == expected["max_steps"] == max_steps, "训练未完成 full 预算")
    _need(training.get("split_manifest") == manifest
          and et._json(et._local(checkpoint, "manifest.json")) == manifest,
          "训练 source manifest 与套件不一致")
    artifact = mt.artifact_manifest(checkpoint)
    _need(training.get("model_artifact", {}).get("files") == artifact["files"]
          and training["model_artifact"].get("sha256") == artifact["sha256"],
          "训练 artifact 文件/hash 不一致")
    return training, artifact


def _base_artifact(r):
    return {"id": gt.MODEL_ID, "revision": r.get("model.revision"), "role": "base"}


def _paired(r, base, trained, base_predictions, trained_predictions):
    _need(base.get("mode") == "zero-shot" and base.get("model_role") == "base"
          and base.get("model_artifact") == _base_artifact(r), "基座评估不是同 revision 冻结 base")
    _need(trained.get("mode") == "eval" and trained.get("model_role") == "trained_checkpoint",
          "训练模型评估模式/角色不一致")
    _need(base["config"]["generation"] == trained["config"]["generation"], "base / trained 解码协议不一致")
    _need(base.get("split_manifest") == trained.get("split_manifest"), "base / trained held-out 不一致")
    _need(all(all(a.get(k) == b.get(k) for k in PREDICTION_IDENTITY_KEYS)
              for a, b in zip(base_predictions, trained_predictions)),
          "base / trained 逐题 prompt / gold 身份不一致")


def _evaluation_role(r, path, data, manifest, role, artifact=None):
    _need(data.get("model_role") == role, f"评估角色必须为 {role}")
    if role == "base":
        _need(data.get("model_artifact") == _base_artifact(r), "基座 artifact 不一致")
    else:
        model = data.get("model_artifact")
        _need(isinstance(artifact, dict) and isinstance(model, dict)
              and model.get("sha256") == artifact["sha256"]
              and model.get("files") == artifact["files"]
              and model.get("role") == "trained_checkpoint",
              "训练模型 artifact 与 checkpoint 不一致")
    return _evaluation_content(r, path, data, manifest)


def _stats_for(r, runs):
    return {name: {"trained": st._stats([run["trained_metrics"][name] for run in runs]),
                   "paired_delta": st._stats([run["trained_metrics"][name] - run["base_metrics"][name]
                                              for run in runs])}
            for name in _metrics(r)}


def _group_stats(r, runs):
    out = {}
    for config_id in CONFIG_IDS:
        group = [run for run in runs if run.get("config_id") == config_id]
        _need(len(group) >= 3, f"{config_id} 消融至少需要三个独立训练 seed")
        spec = gt.verify_ablation(config_id, group[0].get("active_rewards"))
        out[config_id] = {"ablation": spec, "stats": _stats_for(r, group),
                          "seeds": sorted(run["train_seed"] for run in group)}
    return out


def _records_check(r, root, records):
    _need(isinstance(records, dict) and bool(records), "套件缺少完整文件 records")
    for relative, record in records.items():
        _need(isinstance(record, dict) and record.get("path") == relative
              and be.hashed_file(root, record), f"套件绑定文件校验失败：{relative}")
        et._local(root, relative)


def validate_suite_payload(r, data, path):
    root = et._no_symlinks(path.parent)
    _need(isinstance(data, dict) and data.get("task_id") == gt.TASK_ID
          and data.get("kind") == "baseline-suite" and data.get("status") == "completed"
          and data.get("recipe_id") == r.id and data.get("fingerprint") == be.fingerprint(r),
          "不是绑定当前 GSM8K 实验的 completed baseline-suite")
    runs = data.get("runs")
    _need(isinstance(runs, list) and all(isinstance(run, dict) for run in runs), "套件 runs 非法")
    seeds = st._seeds([run.get("train_seed") for run in runs])
    _preflight(r, seeds, root)
    hashes = st._frozen_inputs(r, root)
    _need(data.get("code_sha256") == hashes[0] and data.get("environment_lock_sha256") == hashes[1],
          "套件 code/lock 身份不一致")
    _need(data.get("ablation_configs") == list(CONFIG_IDS)
          and data.get("ablation_is_within_backbone") is True
          and data.get("single_variable") == "被去掉的 reward 函数",
          "GSM8K drop-one 消融边界声明不一致")
    records = data.get("records")
    _records_check(r, root, records)
    _need(data.get("code_snapshot") == records.get(CODE_SNAPSHOT)
          and data.get("environment_lock") == records.get(LOCKFILE), "套件快照/锁记录不一致")

    def linked(record, label):
        _need(isinstance(record, dict) and records.get(record.get("path")) == record,
              f"{label} 未绑定到 records")
        _need(be.hashed_file(root, record), f"{label} hash 不一致")
        file = et._local(root, record["path"])
        return et._json(file), file

    manifest, manifest_path = linked(data.get("split_manifest"), "split_manifest")
    _manifest(r, manifest)
    _need(data.get("split_sha256") == manifest["sha256"]
          and manifest_path.relative_to(root).as_posix() == "base/zero-shot/manifest.json",
          "套件 split 身份或共享 manifest 来源不一致")
    base, base_path = linked(data.get("base_evaluation"), "base_evaluation")
    # base zero-shot 始终绑定 full（无 reward 消融差异）；套件顶层 fingerprint 仍绑定原始 recipe。
    base_recipe = gt.config_recipe(r, gt.DEFAULT_CONFIG_ID)
    _identity(base_recipe, base, hashes)
    _need(base.get("task_id") == gt.TASK_ID and base.get("recipe_id") == r.id, "base 评估 recipe 不一致")
    _need(base.get("config_id") == gt.DEFAULT_CONFIG_ID, "base 评估必须绑定 full（无 reward 消融）")
    base_predictions, base_prediction_path = _evaluation_role(r, base_path, base, manifest, "base")
    _need(data.get("base_predictions") == et._record(base_prediction_path, root), "基座预测记录不一致")
    checkpoints, shared_base = set(), data["base_evaluation"]
    for run, seed in zip(runs, seeds):
        config_id = run.get("config_id")
        effective = _config_recipe(r, seed, config_id)
        spec = gt.verify_ablation(config_id, run.get("active_rewards"))
        training, training_path = linked(run.get("training_evidence"), "training_evidence")
        checkpoint = training_path.parent
        _need(checkpoint not in checkpoints, "独立 seed / 配置共用 checkpoint 目录")
        checkpoints.add(checkpoint)
        _need(checkpoint == _run_dir(root, config_id, seed) / "full", "checkpoint 路径与 config/seed 不一致")
        et._tree_files(checkpoint)
        training, artifact = _training(effective, checkpoint, hashes, manifest)
        _identity(effective, training, hashes)
        _need(training.get("seed") == seed and training.get("config_id") == config_id
              and training.get("active_rewards", training.get("rewards")) == spec["active_rewards"],
              "训练 seed / config_id / active rewards 绑定不一致")
        for file in et._tree_files(checkpoint):
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root),
                  f"checkpoint 文件未绑定：{file.name}")
        trained, trained_path = linked(run.get("trained_evaluation"), "trained_evaluation")
        _identity(effective, trained, hashes)
        trained_predictions, trained_prediction_path = _evaluation_role(
            r, trained_path, trained, manifest, "trained_checkpoint", artifact)
        _paired(r, base, trained, base_predictions, trained_predictions)
        _need(run.get("base_evaluation") == shared_base, "套件应共享同一次固定基座评估")
        _need(run.get("predictions") == et._record(trained_prediction_path, root)
              and data["base_predictions"] == et._record(base_prediction_path, root),
              "预测记录与关联评估不同")
        _need(run.get("fingerprint") == be.fingerprint(effective)
              and run.get("config_id") == config_id
              and run.get("active_rewards") == spec["active_rewards"]
              and run.get("disabled_reward") == spec["disabled_reward"]
              and run.get("model_artifact") == artifact
              and run.get("kind") == "paired-evaluation" and run.get("training_completed") is True
              and run.get("sample_count") == len(base_predictions)
              and run.get("split_sha256") == manifest["sha256"]
              and run.get("base_metrics") == base["metrics"]
              and run.get("trained_metrics") == trained["metrics"],
              "运行摘要身份/指标/消融 config 不一致")
    _need(data.get("stats") == _group_stats(r, runs)
          and data.get("statistical_unit") == "independent_training_seed"
          and data.get("fixed_test") is True
          and data.get("training_seed_variance_measured") is True,
          "套件消融分组 seed 统计未重算或统计单位错误")
    st._guard(r, root, hashes)
    for record in records.values():
        _need(be.hashed_file(root, record), "套件校验期间证据内容漂移")


def aggregate_suite(r, seeds, output_root):
    seeds, root = st._seeds(seeds), et._no_symlinks(st._root(output_root))
    _need(root.is_dir(), "aggregate-only root 不存在")
    _need(not (root / FINAL).exists(), "suite.json 已存在；拒绝覆盖 completed 套件")
    _preflight(r, seeds, root)
    hashes = st._frozen_inputs(r, root)
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
    base_record = bind(base_path)
    base = et._json(base_path)
    _need(isinstance(base, dict), "base evaluation 必须为对象")
    base_recipe = gt.config_recipe(r, gt.DEFAULT_CONFIG_ID)
    _identity(base_recipe, base, hashes)
    _need(base.get("config_id") == gt.DEFAULT_CONFIG_ID, "base 评估必须绑定 full")
    manifest = base.get("split_manifest")
    _manifest(r, manifest)
    base_predictions, base_prediction_path = _evaluation_role(r, base_path, base, manifest, "base")
    bind(base_prediction_path)
    bind(base_path.parent / "manifest.json")
    runs, checkpoints, artifacts = [], set(), []
    for config_id in CONFIG_IDS:
        for seed in seeds:
            effective = _config_recipe(r, seed, config_id)
            checkpoint = et._local(root, _run_dir(Path("."), config_id, seed) / "full")
            _need(checkpoint not in checkpoints, "重复 checkpoint 目录")
            checkpoints.add(checkpoint)
            for file in et._tree_files(checkpoint):
                bind(file)
            training, artifact = _training(effective, checkpoint, hashes, manifest)
            _identity(effective, training, hashes)
            _need(training.get("seed") == seed and training.get("config_id") == config_id,
                  "训练 seed / config_id 与套件不一致")
            artifacts.append((checkpoint, artifact))
            evaluation_path = et._local(root, _run_dir(Path("."), config_id, seed) / "eval" / "evaluation.json")
            evaluation_record = bind(evaluation_path)
            trained = et._json(evaluation_path)
            _identity(effective, trained, hashes)
            trained_predictions, trained_prediction_path = _evaluation_role(
                r, evaluation_path, trained, manifest, "trained_checkpoint", artifact)
            bind(trained_prediction_path)
            bind(evaluation_path.parent / "manifest.json")
            _paired(r, base, trained, base_predictions, trained_predictions)
            spec = gt.verify_ablation(config_id, effective and gt.active_rewards(effective))
            runs.append({"kind": "paired-evaluation", "training_completed": True, "train_seed": seed,
                         "config_id": config_id, "active_rewards": spec["active_rewards"],
                         "disabled_reward": spec["disabled_reward"],
                         "fingerprint": be.fingerprint(effective),
                         "sample_count": len(base_predictions), "split_sha256": manifest["sha256"],
                         "training_evidence": bind(checkpoint / "training.json"),
                         "base_evaluation": base_record, "trained_evaluation": evaluation_record,
                         "predictions": bind(trained_prediction_path),
                         "base_predictions": bind(base_prediction_path),
                         "model_artifact": artifact, "base_metrics": base["metrics"],
                         "trained_metrics": trained["metrics"]})
    stats = _group_stats(r, runs)
    result = {"schema_version": 1, "kind": "baseline-suite", "status": "completed", "task_id": gt.TASK_ID,
              "recipe_id": r.id, "fingerprint": be.fingerprint(r),
              "completed_at": datetime.now(timezone.utc).isoformat(),
              "model_revision": r.get("model.revision"), "dataset_revision": r.get("dataset.revision"),
              "code_sha256": hashes[0], "environment_lock_sha256": hashes[1],
              "code_snapshot": records[CODE_SNAPSHOT], "environment_lock": records[LOCKFILE],
              "split_manifest": records[base_path.parent.joinpath("manifest.json").relative_to(root).as_posix()],
              "split_sha256": manifest["sha256"],
              "base_evaluation": base_record, "base_predictions": records[
                  base_prediction_path.relative_to(root).as_posix()],
              "runs": runs, "stats": stats, "statistical_unit": "independent_training_seed",
              "fixed_test": True, "acceptance": st._acceptance(r, stats["full"]["stats"]),
              "ablation_configs": list(CONFIG_IDS), "ablation_is_within_backbone": True,
              "single_variable": "被去掉的 reward 函数",
              "training_seed_variance_measured": True, "baseline_promotion": "none",
              "records": records, "limitations": LIMITATION}
    validate_suite_payload(r, result, root / FINAL)
    st._guard(r, root, hashes)
    for checkpoint, artifact in artifacts:
        _need(mt.artifact_manifest(checkpoint) == artifact, "聚合期间 checkpoint 权重漂移")
    _need(not (root / FINAL).exists(), "聚合期间 suite.json 已被创建，拒绝覆盖")
    return mt.save_json(root / FINAL, result)


def run_suite(r, seeds, output_root):
    seeds, root = st._seeds(seeds), et._no_symlinks(st._root(output_root))
    _preflight(r, seeds, root, executing=True)
    _need(not root.exists() or (root.is_dir() and not any(root.iterdir())),
          "拒绝非空 root；失败后仅允许 aggregate-only 或新 root 重跑")
    plan = suite_plan(r, seeds, root)
    code_bytes, lock_bytes = st._current_code(), st._lock(r).read_bytes()
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
            st._guard(r, root, hashes)
            _need(pv.file_hash(r.path) == recipe_hash, "执行期间 Recipe 文件漂移")
            try:
                info = st._launch(step["command"], root, index)
            finally:
                st._guard(r, root, hashes)
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


def evaluation_problems(r, data, path, *, role, manifest=None):
    try:
        _need(isinstance(data, dict), "评估证据不是 JSON 对象")
        expected_role = "base" if role == "base" else "trained_checkpoint"
        _need(data.get("kind") == "evaluation" and data.get("stage") == "completed"
              and data.get("model_role") == expected_role and data.get("recipe_id") == r.id
              and data.get("model_revision") == r.get("model.revision")
              and data.get("dataset_revision") == r.get("dataset.revision")
              and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
              "评估来源、角色或输入不一致")
        _need(be.runtime_consistent(r, data), "评估运行环境与 frozen lock 不一致或缺失")
        effective = (gt.config_recipe(r, gt.DEFAULT_CONFIG_ID) if expected_role == "base"
                     else r)
        if expected_role != "base" and data.get("config_id") != gt.config_id_of(r):
            effective = gt.config_recipe(r, data.get("config_id", gt.DEFAULT_CONFIG_ID))
        _identity(effective, data, _hashes(r))
        source = data.get("split_manifest")
        _manifest(r, source)
        _need(manifest is None or source == manifest, "评估 manifest 与套件不一致")
        artifact = data.get("model_artifact") if expected_role != "base" else None
        if expected_role == "base":
            _evaluation_role(r, path, data, source, "base")
        else:
            _need(isinstance(artifact, dict), "训练评估缺少 model_artifact")
            _evaluation_role(r, path, data, source, "trained_checkpoint", {
                "files": artifact.get("files"), "sha256": artifact.get("sha256")})
        names = _metrics(r)
        metrics = data.get("metrics")
        _need(isinstance(metrics, dict) and all(be.finite_number(metrics.get(name)) for name in names),
              "GSM8K 指标缺失或不是有限数值")
        return []
    except (mt.TaskExecutionError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"GSM8K 评估内容拒绝：{exc}"]


def zero_train_validation(r, data, path):
    _need(data.get("mode") == "zero-shot" and data.get("model_role") == "base"
          and data.get("model_artifact") == _base_artifact(r)
          and data.get("config_id") == gt.DEFAULT_CONFIG_ID, "不是冻结 base 的 zero-shot 评估")
    _identity(gt.config_recipe(r, gt.DEFAULT_CONFIG_ID), data, _hashes(r))
    manifest = data.get("split_manifest")
    _manifest(r, manifest)
    _evaluation_role(r, path, data, manifest, "base")


def suite_problems(r):
    try:
        data, path = be.read_json(r, "execution.evidence")
        if data is None:
            return ["缺少可读的 baseline-suite JSON"]
        problems = []

        def need(ok, text):
            if not ok:
                problems.append(text)

        need(data.get("schema_version") == 1 and data.get("kind") == "baseline-suite"
             and data.get("status") == "completed" and data.get("task_id") == gt.TASK_ID,
             "不是已完成的 GSM8K baseline-suite v1")
        need(data.get("recipe_id") == r.id and data.get("fingerprint") == be.fingerprint(r),
             "证据与当前实验配置不一致")
        need(bool(be.HEX64.fullmatch(str(data.get("code_sha256", ""))))
             and bool(be.HEX64.fullmatch(str(data.get("split_sha256", "")))), "缺少代码/划分内容指纹")
        need(data.get("model_revision") == r.get("model.revision")
             and data.get("dataset_revision") == r.get("dataset.revision"), "模型或数据 revision 不一致")
        need(data.get("ablation_configs") == list(CONFIG_IDS), "缺少五个 drop-one reward 消融配置")
        need(data.get("ablation_is_within_backbone") is True, "消融配置必须标注同一 backbone")
        need(data.get("single_variable") == "被去掉的 reward 函数", "消融单变量声明不一致")
        need(be.frozen_environment(r), "没有冻结的完整运行环境")
        need(be.hashed_file(path.parent, data.get("environment_lock")), "环境锁证据文件校验失败")
        if be.hashed_file(path.parent, data.get("environment_lock")):
            actual_lock = (r.path.parent / str(r.get("execution.lockfile"))).read_bytes()
            need((path.parent / data["environment_lock"]["path"]).read_bytes() == actual_lock,
                 "套件环境锁与当前 frozen lock 不一致")
        snapshot = data.get("code_snapshot", {})
        need(be.hashed_file(path.parent, snapshot) and snapshot.get("sha256") == data.get("code_sha256"),
             "代码快照文件缺失或 hash 不一致")
        names = _metrics(r)
        expected, tolerance = r.get("evaluation.expected"), r.get("evaluation.tolerance")
        need(isinstance(expected, dict) and set(names) <= set(expected)
             and all(be.finite_number(expected.get(n)) for n in names),
             "期望指标必须预先声明并覆盖全部 GSM8K metrics")
        need(isinstance(tolerance, dict) and set(names) <= set(tolerance)
             and all(be.finite_number(tolerance.get(n)) and tolerance[n] >= 0 for n in names),
             "容差必须为覆盖全部 metrics 的非负数值表")
        runs = data.get("runs")
        if not isinstance(runs, list) or len(runs) < 15:
            return problems + ["五个 drop-one 消融 × 三个独立训练 seed 至少需要 15 轮配对评估"]
        config_ids = [run.get("config_id") for run in runs if isinstance(run, dict)]
        need(set(config_ids) == set(CONFIG_IDS), "每轮必须绑定五个消融 config_id 之一")
        for config_id in CONFIG_IDS:
            group = [run for run in runs if isinstance(run, dict) and run.get("config_id") == config_id]
            seeds = [run.get("train_seed") for run in group]
            need(len(group) >= 3 and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds),
                 f"{config_id} 消融至少需要三个互异整数训练 seed")
            for run in group:
                spec = gt.verify_ablation(config_id, run.get("active_rewards"))
                need(run.get("kind") == "paired-evaluation" and run.get("training_completed") is True
                     and run.get("active_rewards") == spec["active_rewards"]
                     and run.get("disabled_reward") == spec["disabled_reward"],
                     f"{config_id} 轮次 reward 消融绑定或配对评估不一致")
                need(type(run.get("sample_count")) is int and run["sample_count"] > 0, "评测样本数必须为正整数")
                need(run.get("split_sha256") == data.get("split_sha256"), "基座/训练模型必须共用同一 held-out 划分")
                training = be.linked_json(path.parent, run.get("training_evidence"))
                need(isinstance(training, dict) and training.get("kind") == "training"
                     and training.get("stage") == "completed" and training.get("mode") == "full"
                     and training.get("task_id") == gt.TASK_ID and training.get("recipe_id") == r.id
                     and training.get("config_id") == config_id
                     and training.get("seed") == run.get("train_seed")
                     and training.get("model_revision") == r.get("model.revision")
                     and training.get("dataset_revision") == r.get("dataset.revision"),
                     "训练证据内容与 config_id / 运行身份不一致")
                if isinstance(training, dict):
                    need(be.runtime_consistent(r, training), "训练运行环境与 frozen lock 不一致或缺失")
                for side, role_name in (("base_metrics", "base"), ("trained_metrics", "trained_checkpoint")):
                    evaluation = be.linked_json(path.parent, run.get(
                        "base_evaluation" if role_name == "base" else "trained_evaluation"))
                    record = run.get("base_evaluation" if role_name == "base" else "trained_evaluation")
                    if not isinstance(evaluation, dict) or not isinstance(record, dict):
                        problems.append("每轮必须关联真实基座与训练模型评估 JSON")
                        continue
                    problems.extend(evaluation_problems(
                        _config_recipe(r, run["train_seed"], config_id) if role_name != "base" else r,
                        evaluation, path.parent / record["path"], role=role_name))
                    need(evaluation.get("metrics") == run.get(side)
                         and evaluation.get("sample_count") == run.get("sample_count"),
                         "套件摘要与关联评估指标或样本数不一致")
                need(be.hashed_file(path.parent, run.get("predictions")), "每轮预测证据文件必须可校验")
                for side in ("base_metrics", "trained_metrics"):
                    scores = run.get(side)
                    need(isinstance(scores, dict) and set(names) <= set(scores)
                         and all(be.finite_number(scores.get(m)) for m in names), "每轮指标都须为有限数值")
        try:
            validate_suite_payload(r, data, path)
        except (mt.TaskExecutionError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            problems.append(f"GSM8K 套件内容拒绝：{exc}")
        return problems
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"证据结构非法；失败关闭：{exc}"]
