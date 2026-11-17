"""xlam-fc-v1 独立训练 seed 编排与发布级内容重算；标准库、失败关闭。

与数学 / FIM 刻意分开：xLAM 的 held-out 是规范化内容 hash 80/10/10 + 全局去重，
生成输入是完整 <user>/<tools> prompt，gold 是严格 JSON 的 tool calls，四个指标
是 json_parse_rate / schema_valid_rate / tool_call_exact_match / argument_f1。
复用 suite_task 的通用执行器、统计与文件绑定工具，但不改写数学或 FIM 路径，
也不把「合法 JSON 率」冒充真实 API 执行正确率。
"""
from __future__ import annotations

import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import baseline_evidence as be
from . import export_task as et
from . import math_task as mt
from . import provenance as pv
from . import suite_task as st
from . import xlam_task as xt

CODE_ROOT = st.CODE_ROOT
CODE_SNAPSHOT, LOCKFILE, STATE, FINAL = st.CODE_SNAPSHOT, st.LOCKFILE, st.STATE, st.FINAL
LIMITATION = ("统计单位为独立训练 seed；固定同一个内容 hash 隔离的 held-out test 与严格 JSON/greedy 协议。"
              "Student-t 区间假设训练 seed 结果独立同分布且均值近似正态；"
              "不包含数据抽样不确定性，不能作为业务显著性证据。"
              "四项指标只衡量输出格式与 tool schema 协议，不执行生成的调用，"
              "不等于真实 API 执行正确率、pass@1 或业务安全。"
              "JSON 仅证明内部内容一致性，不证明日志实际来自 GPU。")
METRICS = xt.METRICS
REFERENCE_FILES = xt.REFERENCE_FILES
FULL_STEPS = 150


def _need(condition, message):
    if not condition:
        raise mt.TaskExecutionError(message)


def _hashes(r):
    identity = xt.execution_identity(r)
    return identity["code_sha256"], identity["environment_lock_sha256"]


def _metrics(r):
    names = r.get("evaluation.metrics")
    _need(isinstance(names, list) and len(names) == len(METRICS) and set(names) == set(METRICS),
          "xLAM evaluation.metrics 必须精确为四项严格 JSON / tool schema 指标")
    return names


def _preflight(r, seeds, root, *, executing=False):
    xt.require_supported(r)
    _need(be.frozen_environment(r), "套件需要与 Recipe 一致的 frozen_environment")
    for key in ("model.revision", "dataset.revision", "source_sha"):
        xt.require_revision(r, key)
    xt.fixed_seeds(r)
    _metrics(r)
    for seed in seeds:
        effective = pv.effective_recipe(r, seed)
        xt.config_kwargs(effective, {"mode": "full"}, root / f"seed-{seed}" / "full")
    if executing:
        _need(r.get("decision.gate") != "blocked" and r.get("baseline.result") != "passes",
              "训练决策门已阻断；不能绕过零训练基线及预算决策")


def suite_plan(r, seeds, output_root):
    """仅读配置声明；不读取快照、lock、checkpoint 或输入证据，不写磁盘。"""
    seeds, root = st._seeds(seeds), st._root(output_root)
    try:
        xt.require_supported(r)
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
            "task_id": xt.TASK_ID, "fingerprint": be.fingerprint(r),
            "execution_supported": supported, "unsupported_reason": reason,
            "seeds": seeds, "output_root": str(root), "shell": False,
            "data_seed": r.get("train.data_seed"),
            "evaluation_seed": r.get("evaluation.seed", r.get("train.seed")),
            "training_source": {"id": xt.MODEL, "revision": r.get("model.revision"),
                                "independent_initialization": True,
                                "adapter_only": True, "base_weights_bundled": False},
            "protocol": xt.EVAL_PROTOCOL, "metrics": list(METRICS),
            "full_steps": r.get("train.max_steps", r.get("budget.full_steps")),
            "commands": commands, "runs": runs, "timeout_seconds": st.COMMAND_TIMEOUT_SECONDS,
            "resume_supported": False, "baseline_promotion": "none", "limitations": LIMITATION}


def _manifest(r, manifest):
    """xLAM 内容 hash 三划分；不套用数学 1500 条 / FIM repo-hash 协议。"""
    xt.validate_source_manifest(manifest, r)
    _need(manifest.get("dataset_revision") == r.get("dataset.revision"),
          "source_manifest dataset_revision 不一致")
    counts = manifest.get("counts")
    _need(isinstance(counts, dict)
          and counts.get("source") == sum(counts.get(name, 0) for name in ("train", "validation", "test"))
          and type(counts.get("test_eval_used")) is int
          and 0 < counts["test_eval_used"] <= counts["test"],
          "xLAM source_manifest counts 不一致")
    indices = manifest.get("eval_sample_indices")
    _need(isinstance(indices, list) and len(indices) == counts["test_eval_used"]
          and all(type(i) is int and i >= 0 for i in indices)
          and len(set(indices)) == len(indices),
          "xLAM eval_sample_indices 与 held-out 样本不一致")
    return manifest


def _identity(r, data, hashes):
    _need(type(data.get("schema_version")) is int and data["schema_version"] == 1
          and data.get("stage") == "completed" and data.get("task_id") == xt.TASK_ID
          and data.get("recipe_id") == r.id and data.get("fingerprint") == be.fingerprint(r)
          and data.get("model_id") == xt.MODEL and data.get("dataset_id") == xt.DATASET
          and data.get("model_revision") == r.get("model.revision")
          and data.get("dataset_revision") == r.get("dataset.revision")
          and data.get("source_sha") == r.get("source_sha")
          and type(data.get("seed")) is int and data["seed"] == r.get("train.seed")
          and type(data.get("data_seed")) is int and data["data_seed"] == r.get("train.data_seed"),
          "xLAM 证据任务、revision、fingerprint 或 seed 不一致")
    _need(data.get("code_sha256") == hashes[0] and data.get("environment_lock_sha256") == hashes[1],
          "证据混合 code / lock hash 或与当前执行身份不一致")
    _need(be.runtime_consistent(r, data) and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "非真实运行协议或 runtime 不一致")
    _need(bool(be.HEX64.fullmatch(str(data.get("recipe_sha256", "")))), "缺少 Recipe 内容 hash")


def _generation(r, identity):
    return {"do_sample": False, "num_beams": 1,
            "max_new_tokens": r.get("evaluation.max_new_tokens"),
            "eos_token_id": identity["eos_token_id"], "pad_token_id": identity["pad_token_id"],
            "use_cache": True, "seed": r.get("evaluation.seed")}


def _evaluation_content(r, path, data, manifest):
    """逐题重建 prompt / tools / gold / 生成文本并重算四指标；不信任摘要字段。"""
    _need(data.get("split_manifest") == manifest and data.get("source_manifest") == manifest,
          "评估证据 split_manifest 与套件不一致")
    for name in REFERENCE_FILES:
        _need(et._json(et._local(path.parent, name)) is not None, f"缺少 {name}")
    _need(et._json(et._local(path.parent, "source_manifest.json")) == manifest,
          "评估 source_manifest.json 不一致")
    identity = data.get("tokenizer_identity")
    _need(isinstance(identity, dict) and identity.get("sha256") == xt._hash_manifest(identity)["sha256"],
          "tokenizer identity hash 无效")
    _need(et._json(et._local(path.parent, xt.TOKENIZER_IDENTITY_FILE)) == identity,
          "评估 tokenizer_identity.json 不一致")
    eval_manifest = data.get("evaluation_manifest")
    _need(isinstance(eval_manifest, dict)
          and eval_manifest.get("sha256") == xt._hash_manifest(eval_manifest)["sha256"],
          "xLAM evaluation manifest hash 无效")
    _need(et._json(et._local(path.parent, xt.EVAL_MANIFEST_FILE)) == eval_manifest,
          "评估 eval_manifest.json 不一致")
    _need(data.get("evaluation_protocol") == xt.EVAL_PROTOCOL
          and eval_manifest.get("kind") == "xlam-eval-manifest"
          and eval_manifest.get("protocol") == xt.EVAL_PROTOCOL
          and eval_manifest.get("variant") == xt.EVAL_VARIANT
          and eval_manifest.get("task_id") == xt.TASK_ID
          and eval_manifest.get("data_seed") == r.get("train.data_seed")
          and eval_manifest.get("max_new_tokens") == r.get("evaluation.max_new_tokens")
          and eval_manifest.get("tokenizer_identity") == identity
          and eval_manifest.get("source_manifest_sha256") == manifest["sha256"]
          and eval_manifest.get("gold_policy") == r.get("xlam.duplicate_policy")
          and eval_manifest.get("generated_code_executed") is False
          and eval_manifest.get("upstream_quantitative_evaluation") is False,
          "xLAM 评估协议 / 输出格式 / tokenizer 不一致")
    examples = eval_manifest.get("samples")
    _need(isinstance(examples, list) and bool(examples)
          and eval_manifest.get("sample_count") == len(examples)
          and eval_manifest.get("samples_sha256") == xt.sha256_json(examples),
          "xLAM 评估样本列表或内容 hash 不一致")
    used = manifest["counts"]["test_eval_used"]
    test_hashes = manifest["split_hashes"]["test"][:used]
    indices = manifest["eval_sample_indices"]
    _need(len(examples) == used == len(test_hashes), "评估样本数必须与 shared held-out 一致")
    rows = []
    for example, expected_hash, source_index in zip(examples, test_hashes, indices):
        raw = example.get("raw")
        _need(isinstance(raw, dict) and set(raw) == {"query", "tools", "answers"}
              and all(isinstance(raw[key], str) for key in raw),
              "评估样本缺少可重建的规范化原始行")
        row = xt.parse_row(raw, source_index)
        prompt = xt.render_prompt(row)
        prompt_ids = example.get("prompt_ids")
        _need(example.get("sample_id") == row["sample_id"]
              and example.get("source_index") == source_index
              and example.get("content_sha256") == row["content_sha256"] == expected_hash
              and raw["tools"] == row["tools"] and raw["answers"] == row["answers"]
              and example.get("tools_sha256") == xt.sha256_json(row["tools_obj"])
              and example.get("gold_sha256") == xt.sha256_json(row["answers_obj"])
              and example.get("prompt") == prompt
              and example.get("prompt_sha256") == xt.sha256_json(prompt)
              and isinstance(prompt_ids, list) and bool(prompt_ids)
              and all(type(token) is int and token >= 0 for token in prompt_ids)
              and example.get("input_sha256") == xt.sha256_json(prompt_ids),
              "xLAM prompt / tools / gold / token 输入重建失败")
        rows.append(row)
    config = data.get("config")
    _need(isinstance(config, dict) and data.get("config_sha256") == xt.sha256_json(config),
          "评估配置内容 hash 不一致")
    _need(set(config) == {"task_id", "protocol", "zero_shot", "checkpoint", "generation",
                          "quantization", "compute_dtype", "attention_implementation", "model",
                          "tokenizer_identity", "dataset_revision", "source_manifest_sha256",
                          "eval_manifest_sha256", "max_new_tokens", "do_sample", "num_beams",
                          "held_out_rows", "generated_code_executed", "upstream_quantitative_evaluation"}
          and config["task_id"] == xt.TASK_ID and config["protocol"] == xt.EVAL_PROTOCOL
          and config["generation"] == _generation(r, identity)
          and config["quantization"] == r.get("quantization")
          and config["compute_dtype"] == "bfloat16" and config["attention_implementation"] == "eager"
          and config["model"] == data.get("model_artifact")
          and config["tokenizer_identity"] == identity
          and config["dataset_revision"] == r.get("dataset.revision")
          and config["source_manifest_sha256"] == manifest["sha256"]
          and config["eval_manifest_sha256"] == eval_manifest["sha256"]
          and config["max_new_tokens"] == r.get("evaluation.max_new_tokens")
          and config["do_sample"] is False and config["num_beams"] == 1
          and config["held_out_rows"] == len(examples)
          and config["generated_code_executed"] is False
          and config["upstream_quantitative_evaluation"] is False,
          "评估生成 / 量化 / 输入 config 不一致")
    record = data.get("predictions_file", {})
    _need(isinstance(record, dict), "缺少预测文件记录")
    prediction_path = et._local(path.parent, record.get("path"))
    _need(be.hashed_file(path.parent, record), "预测文件缺失或 hash 不一致")
    document = et._json(prediction_path)
    _need(isinstance(document, dict) and document.get("task_id") == xt.TASK_ID
          and document.get("protocol") == xt.EVAL_PROTOCOL
          and document.get("recipe_id") == r.id
          and document.get("metrics") == data.get("metrics")
          and document.get("sha256") == xt.sha256_json(document.get("predictions")),
          "预测文档身份或指标不一致")
    predictions = document.get("predictions")
    description = data.get("predictions", {})
    _need(isinstance(predictions, list) and len(predictions) == len(examples)
          and description.get("path") == record.get("path")
          and type(description.get("count")) is int and description["count"] == len(predictions)
          and description.get("sha256") == xt.sha256_json(predictions)
          and data.get("predictions_sha256") == xt.sha256_json(predictions),
          "逐题预测内容 hash 或数量不一致")
    for prediction, example, row in zip(predictions, examples, rows):
        text = prediction.get("generated_text")
        _need(isinstance(prediction, dict) and isinstance(text, str)
              and all(prediction.get(key) == example[key] for key in (
                  "sample_id", "source_index", "content_sha256", "gold_sha256", "prompt_sha256"))
              and prediction.get("prediction_sha256") == xt.sha256_json(text)
              and prediction.get("eos_token_id") == identity["eos_token_id"]
              and prediction.get("max_new_tokens") == r.get("evaluation.max_new_tokens"),
              "逐题生成记录与评估 manifest 不一致")
        # 关键：从原始生成文本重新打分，而不是把证据里的数字加总一遍。
        _need(xt.score_prediction(row, text) == {key: prediction.get(key) for key in (
            "sample_id", "content_sha256", "gold_sha256", "json_parse_rate",
            "schema_valid_rate", "tool_call_exact_match", "argument_f1", "prediction_sha256")},
            "xLAM 四指标与逐题生成文本重算不一致")
    _need(data.get("metrics") == xt.recompute_metrics(predictions),
          "xLAM 指标与逐题重算不一致")
    counts = data.get("metric_counts")
    _need(isinstance(counts, dict) and counts.get("total") == len(predictions)
          and all(counts.get(name) == sum(row[name] for row in predictions) for name in METRICS),
          "metric_counts 与逐题预测不一致")
    _need(type(data.get("sample_count")) is int and data["sample_count"] == len(predictions),
          "sample_count 与预测数量不一致")
    return predictions, prediction_path, eval_manifest, identity


def _training(r, checkpoint, hashes, manifest):
    """复用 xlam_task.checkpoint_path 的 adapter/base/量化/transform 强校验，再核发布身份。"""
    xt.checkpoint_path({}, checkpoint, recipe=r)
    training = et._json(et._local(checkpoint, "training.json"))
    _identity(r, training, hashes)
    _need(training.get("kind") == "training" and training.get("mode") == "full"
          and training.get("trainer") == "Trainer", "不是 full Trainer 训练证据")
    expected = xt.config_kwargs(r, {"recipe_id": r.id, "task_id": xt.TASK_ID, "mode": "full"}, checkpoint)
    expected_identity = xt.training_arguments_identity(expected)
    config = training.get("config")
    _need(isinstance(config, dict) and training.get("config_sha256") == xt.sha256_json(config)
          and xt.training_arguments_identity(config.get("training_arguments")) == expected_identity
          and xt.training_arguments_identity(config.get("api_training_arguments")) in (
              expected_identity, xt.training_arguments_identity({
                  ("eval_strategy" if k == "evaluation_strategy" else k): v for k, v in expected.items()}))
          and config.get("xlam") == r.get("xlam") and config.get("peft") == r.get("peft")
          and config.get("quantization") == r.get("quantization")
          and config.get("attention_implementation") == "eager"
          and config.get("upstream_metadata") == dict(r.get("upstream"))
          and config.get("lora_execution", {}).get("tensor_reload_check", {}).get(
              "strict_values_verified") is True,
          "训练有效配置 / completion-only 协议 / 预算不一致")
    _need(type(training.get("steps")) is int and training["steps"] == training.get("max_steps")
          == expected["max_steps"] == FULL_STEPS, "训练未完成 full=150 步预算")
    _need(training.get("split_manifest") == manifest and training.get("source_manifest") == manifest
          and et._json(et._local(checkpoint, "source_manifest.json")) == manifest,
          "训练 source manifest 与套件不一致")
    identity = training.get("tokenizer_identity")
    transform = training.get("transformation_manifest")
    _need(et._json(et._local(checkpoint, xt.TOKENIZER_IDENTITY_FILE)) == identity
          and et._json(et._local(checkpoint, "xlam_transform_manifest.json")) == transform,
          "训练 tokenizer / transform 文件不一致")
    xt.validate_transform_manifest(transform, manifest, r, identity)
    artifact = xt.artifact_manifest(checkpoint)
    _need(xt.same_artifact(training.get("model_artifact"), artifact), "训练 artifact 文件/hash 不一致")
    return training, artifact, identity


def _base_artifact(r):
    return xt.base_artifact(r)


def _paired(r, base, base_predictions, trained, trained_predictions):
    _need(base.get("mode") == "zero-shot" and base.get("model_role") == "base"
          and base.get("model_artifact") == _base_artifact(r), "基座评估不是同 revision 冻结 base")
    _need(trained.get("mode") == "eval" and trained.get("model_role") == "checkpoint",
          "训练模型评估模式/角色不一致")
    _need(base["config"]["generation"] == trained["config"]["generation"], "base / trained 解码协议不一致")
    _need(base.get("evaluation_manifest") == trained.get("evaluation_manifest")
          and base.get("tokenizer_identity") == trained.get("tokenizer_identity"),
          "base / trained 严格 JSON 输入或 tokenizer 不一致")
    keys = ("sample_id", "source_index", "content_sha256", "gold_sha256", "prompt_sha256")
    _need(all(all(a.get(k) == b.get(k) for k in keys)
              for a, b in zip(base_predictions, trained_predictions)),
          "base / trained 逐题 prompt / gold 不一致")


def _evaluation_role(r, path, data, manifest, role, artifact=None, prediction_path=None):
    _need(data.get("model_role") == role, f"评估角色必须为 {role}")
    if role == "base":
        _need(data.get("model_artifact") == _base_artifact(r), "基座 artifact 不一致")
    else:
        model = data.get("model_artifact")
        _need(isinstance(artifact, dict) and isinstance(model, dict)
              and xt.same_artifact(model, artifact),
              "训练模型 artifact 与 checkpoint 不一致")
    predictions, path_of_predictions, _, _ = _evaluation_content(r, path, data, manifest)
    if prediction_path is not None:
        _need(path_of_predictions == prediction_path, "预测文件路径不一致")
    return predictions, path_of_predictions


def _stats_for(r, runs):
    return {name: {"trained": st._stats([run["trained_metrics"][name] for run in runs]),
                   "paired_delta": st._stats([run["trained_metrics"][name] - run["base_metrics"][name]
                                              for run in runs])}
            for name in _metrics(r)}


def _records_check(r, root, records):
    _need(isinstance(records, dict) and bool(records), "套件缺少完整文件 records")
    for relative, record in records.items():
        _need(isinstance(record, dict) and record.get("path") == relative
              and be.hashed_file(root, record), f"套件绑定文件校验失败：{relative}")
        et._local(root, relative)


def validate_suite_payload(r, data, path):
    """发布门禁与聚合器共用：只读重算任意目录布局的完整内容，拒绝宽松 JSON。"""
    root = et._no_symlinks(path.parent)
    _need(isinstance(data, dict) and data.get("task_id") == xt.TASK_ID
          and data.get("kind") == "baseline-suite" and data.get("status") == "completed"
          and data.get("recipe_id") == r.id and data.get("fingerprint") == be.fingerprint(r),
          "不是绑定当前 xLAM 实验的 completed baseline-suite")
    runs = data.get("runs")
    _need(isinstance(runs, list) and all(isinstance(run, dict) for run in runs), "套件 runs 非法")
    seeds = st._seeds([run.get("train_seed") for run in runs])
    _preflight(r, seeds, root)
    hashes = st._frozen_inputs(r, root)
    _need(data.get("code_sha256") == hashes[0] and data.get("environment_lock_sha256") == hashes[1],
          "套件 code/lock 身份不一致")
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
          and manifest_path.relative_to(root).as_posix() == "base/zero-shot/source_manifest.json",
          "套件 split 身份或共享 manifest 来源不一致")
    base, base_path = linked(data.get("base_evaluation"), "base_evaluation")
    _identity(r, base, hashes)
    base_predictions, base_prediction_path = _evaluation_role(r, base_path, base, manifest, "base")
    _need(data.get("base_predictions") == et._record(base_prediction_path, root), "基座预测记录不一致")
    for name in REFERENCE_FILES:
        file = base_path.parent / name
        _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root),
              f"基座参考文件未绑定：{name}")
    checkpoints, shared_base = set(), data["base_evaluation"]
    for run, seed in zip(runs, seeds):
        effective = pv.effective_recipe(r, seed)
        training, training_path = linked(run.get("training_evidence"), "training_evidence")
        checkpoint = training_path.parent
        _need(checkpoint not in checkpoints, "独立 seed 共用 checkpoint 目录")
        checkpoints.add(checkpoint)
        et._tree_files(checkpoint)
        training, artifact, _ = _training(effective, checkpoint, hashes, manifest)
        _identity(effective, training, hashes)
        _need(training.get("seed") == seed, "训练 seed 与套件记录不一致")
        for file in et._tree_files(checkpoint):
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root),
                  f"checkpoint 文件未绑定：{file.name}")
        trained, trained_path = linked(run.get("trained_evaluation"), "trained_evaluation")
        _identity(effective, trained, hashes)
        trained_predictions, trained_prediction_path = _evaluation_role(
            r, trained_path, trained, manifest, "checkpoint", artifact)
        _paired(r, base, base_predictions, trained, trained_predictions)
        _need(run.get("base_evaluation") == shared_base, "套件应共享同一次固定基座评估")
        _need(run.get("predictions") == et._record(trained_prediction_path, root)
              and data["base_predictions"] == et._record(base_prediction_path, root),
              "预测记录与关联评估不同")
        _need(run.get("fingerprint") == be.fingerprint(effective)
              and xt.artifact_identity(run.get("model_artifact")) == xt.artifact_identity(artifact)
              and run.get("kind") == "paired-evaluation" and run.get("training_completed") is True
              and run.get("sample_count") == len(base_predictions)
              and run.get("split_sha256") == manifest["sha256"]
              and run.get("base_metrics") == base["metrics"]
              and run.get("trained_metrics") == trained["metrics"],
              "运行摘要身份/指标不一致")
        for name in REFERENCE_FILES:
            file = trained_path.parent / name
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root),
                  f"训练评估参考文件未绑定：{name}")
    _need(data.get("stats") == _stats_for(r, runs)
          and data.get("statistical_unit") == "independent_training_seed"
          and data.get("fixed_test") is True
          and data.get("training_seed_variance_measured") is True,
          "套件 seed 统计未重算或统计单位错误")
    st._guard(r, root, hashes)
    for record in records.values():
        _need(be.hashed_file(root, record), "套件校验期间证据内容漂移")


def aggregate_suite(r, seeds, output_root):
    """只读已完成真实证据，构造并自校验 suite.json；不合成或修复历史输入。"""
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
    _identity(r, base, hashes)
    manifest = base.get("split_manifest")
    _manifest(r, manifest)
    base_predictions, base_prediction_path = _evaluation_role(r, base_path, base, manifest, "base")
    bind(base_prediction_path)
    for name in REFERENCE_FILES:
        bind(base_path.parent / name)
    manifest_record = records[base_path.parent.joinpath("source_manifest.json").relative_to(root).as_posix()]
    runs, checkpoints, artifacts = [], set(), []
    for seed in seeds:
        effective = pv.effective_recipe(r, seed)
        checkpoint = et._local(root, f"seed-{seed}/full")
        _need(checkpoint not in checkpoints, "重复 checkpoint 目录")
        checkpoints.add(checkpoint)
        for file in et._tree_files(checkpoint):
            bind(file)
        training, artifact, _ = _training(effective, checkpoint, hashes, manifest)
        _identity(effective, training, hashes)
        _need(training.get("seed") == seed, "训练 seed 与套件不一致")
        artifacts.append((checkpoint, artifact))
        evaluation_path = et._local(root, f"seed-{seed}/eval/evaluation.json")
        evaluation_record = bind(evaluation_path)
        trained = et._json(evaluation_path)
        _identity(effective, trained, hashes)
        trained_predictions, trained_prediction_path = _evaluation_role(
            r, evaluation_path, trained, manifest, "checkpoint", artifact)
        bind(trained_prediction_path)
        for name in REFERENCE_FILES:
            bind(evaluation_path.parent / name)
        _paired(r, base, base_predictions, trained, trained_predictions)
        runs.append({"kind": "paired-evaluation", "training_completed": True, "train_seed": seed,
                     "fingerprint": be.fingerprint(effective), "sample_count": len(base_predictions),
                     "split_sha256": manifest["sha256"], "training_evidence": bind(checkpoint / "training.json"),
                     "base_evaluation": base_record, "trained_evaluation": evaluation_record,
                     "predictions": bind(trained_prediction_path),
                     "base_predictions": bind(base_prediction_path),
                     "model_artifact": artifact, "base_metrics": base["metrics"],
                     "trained_metrics": trained["metrics"]})
    stats = _stats_for(r, runs)
    result = {"schema_version": 1, "kind": "baseline-suite", "status": "completed", "task_id": xt.TASK_ID,
              "recipe_id": r.id, "fingerprint": be.fingerprint(r),
              "completed_at": datetime.now(timezone.utc).isoformat(),
              "model_revision": r.get("model.revision"), "dataset_revision": r.get("dataset.revision"),
              "code_sha256": hashes[0], "environment_lock_sha256": hashes[1],
              "code_snapshot": records[CODE_SNAPSHOT], "environment_lock": records[LOCKFILE],
              "split_manifest": manifest_record, "split_sha256": manifest["sha256"],
              "base_evaluation": base_record, "base_predictions": records[
                  base_prediction_path.relative_to(root).as_posix()],
              "runs": runs, "stats": stats, "statistical_unit": "independent_training_seed",
              "fixed_test": True, "acceptance": st._acceptance(r, stats), "records": records,
              "training_seed_variance_measured": True, "baseline_promotion": "none",
              "limitations": LIMITATION}
    validate_suite_payload(r, result, root / FINAL)
    st._guard(r, root, hashes)
    for checkpoint, artifact in artifacts:
        _need(xt.artifact_identity(xt.artifact_manifest(checkpoint)) == xt.artifact_identity(artifact),
              "聚合期间 checkpoint 权重漂移")
    _need(not (root / FINAL).exists(), "聚合期间 suite.json 已被创建，拒绝覆盖")
    return mt.save_json(root / FINAL, result)


def run_suite(r, seeds, output_root):
    """在新 root 顺序实跑一次 base 与独立 full/eval；失败不允许 resume。"""
    seeds, root = st._seeds(seeds), et._no_symlinks(st._root(output_root))
    _preflight(r, seeds, root, executing=True)
    _need(not root.exists() or (root.is_dir() and not any(root.iterdir())),
          "拒绝非空 root；失败后仅允许 aggregate-only 或新 root 重跑")
    plan = suite_plan(r, seeds, root)
    code_bytes, lock_bytes = st._current_code(), st._lock(r).read_bytes()
    import hashlib
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
    """供 baseline_evidence 分派；返回问题列表，不抛给门禁。"""
    try:
        _need(isinstance(data, dict), "评估证据不是 JSON 对象")
        _need(data.get("kind") == "evaluation" and data.get("stage") == "completed"
              and data.get("model_role") == role and data.get("recipe_id") == r.id
              and data.get("model_revision") == r.get("model.revision")
              and data.get("dataset_revision") == r.get("dataset.revision")
              and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
              "评估来源、角色或输入不一致")
        _need(be.runtime_consistent(r, data), "评估运行环境与 frozen lock 不一致或缺失")
        _identity(r, data, _hashes(r))
        source = data.get("split_manifest")
        _manifest(r, source)
        _need(manifest is None or source == manifest, "评估 manifest 与套件不一致")
        artifact = data.get("model_artifact") if role != "base" else None
        _evaluation_role(r, path, data, source, role, artifact)
        names = _metrics(r)
        metrics = data.get("metrics")
        _need(isinstance(metrics, dict) and all(be.finite_number(metrics.get(name)) for name in names),
              "xLAM 指标缺失或不是有限数值")
        return []
    except (mt.TaskExecutionError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"xLAM 评估内容拒绝：{exc}"]


def zero_train_validation(r, data, path):
    """C306 的 xLAM 强校验；抛异常即不通过，拒绝只靠摘要 JSON。"""
    _need(data.get("mode") == "zero-shot" and data.get("model_role") == "base"
          and data.get("model_artifact") == _base_artifact(r), "不是冻结 base 的 zero-shot 评估")
    _identity(r, data, _hashes(r))
    manifest = data.get("split_manifest")
    _manifest(r, manifest)
    _evaluation_role(r, path, data, manifest, "base")


def suite_problems(r):
    """C307 的 xLAM 发布门禁：通用摘要 + 与生产聚合器共用的内容强校验。"""
    try:
        data, path = be.read_json(r, "execution.evidence")
        if data is None:
            return ["缺少可读的 baseline-suite JSON"]
        problems = []

        def need(ok, text):
            if not ok:
                problems.append(text)

        need(data.get("schema_version") == 1 and data.get("kind") == "baseline-suite"
             and data.get("status") == "completed" and data.get("task_id") == xt.TASK_ID,
             "不是已完成的 xLAM baseline-suite v1")
        need(data.get("recipe_id") == r.id and data.get("fingerprint") == be.fingerprint(r),
             "证据与当前实验配置不一致")
        need(bool(be.HEX64.fullmatch(str(data.get("code_sha256", ""))))
             and bool(be.HEX64.fullmatch(str(data.get("split_sha256", "")))), "缺少代码/划分内容指纹")
        need(data.get("model_revision") == r.get("model.revision")
             and data.get("dataset_revision") == r.get("dataset.revision"), "模型或数据 revision 不一致")
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
             "期望指标必须预先声明并覆盖全部 xLAM metrics")
        need(isinstance(tolerance, dict) and set(names) <= set(tolerance)
             and all(be.finite_number(tolerance.get(n)) and tolerance[n] >= 0 for n in names),
             "容差必须为覆盖全部 metrics 的非负数值表")
        runs = data.get("runs")
        if not isinstance(runs, list) or len(runs) < 3:
            return problems + ["至少需要三个独立训练 seed 的运行，而不是重复 greedy 解码"]
        seeds = [run.get("train_seed") for run in runs if isinstance(run, dict)]
        need(len(seeds) == len(runs) and all(type(s) is int for s in seeds) and len(set(seeds)) == len(seeds),
             "训练 seed 必须是三个以上互异整数")
        for run in runs:
            if not isinstance(run, dict):
                problems.append("run 结构非法")
                continue
            need(run.get("kind") == "paired-evaluation" and run.get("training_completed") is True,
                 "每轮需要真实训练后的配对评估")
            need(type(run.get("sample_count")) is int and run["sample_count"] > 0, "评测样本数必须为正整数")
            need(run.get("split_sha256") == data.get("split_sha256"), "基座/训练模型必须共用同一 held-out 划分")
            training = be.linked_json(path.parent, run.get("training_evidence"))
            need(isinstance(training, dict) and training.get("kind") == "training"
                 and training.get("stage") == "completed" and training.get("mode") == "full"
                 and training.get("task_id") == xt.TASK_ID and training.get("recipe_id") == r.id
                 and training.get("seed") == run.get("train_seed")
                 and training.get("data_seed") == r.get("train.data_seed")
                 and training.get("steps") == training.get("max_steps") == FULL_STEPS
                 and training.get("model_revision") == r.get("model.revision")
                 and training.get("dataset_revision") == r.get("dataset.revision"),
                 "训练证据内容与运行身份不一致")
            if isinstance(training, dict):
                need(be.runtime_consistent(r, training), "训练运行环境与 frozen lock 不一致或缺失")
            for side, role in (("base_metrics", "base"), ("trained_metrics", "checkpoint")):
                evaluation = be.linked_json(path.parent, run.get(
                    "base_evaluation" if role == "base" else "trained_evaluation"))
                record = run.get("base_evaluation" if role == "base" else "trained_evaluation")
                if not isinstance(evaluation, dict) or not isinstance(record, dict):
                    problems.append("每轮必须关联真实基座与训练模型评估 JSON")
                    continue
                problems.extend(evaluation_problems(r, evaluation, path.parent / record["path"], role=role))
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
            problems.append(f"xLAM 套件内容拒绝：{exc}")
        if problems:
            return problems
        for name in names:
            values = [run["trained_metrics"][name] for run in runs]
            need(abs(statistics.mean(values) - expected[name]) <= tolerance[name],
                 f"{name} 的重跑均值超出验收容差")
        return problems
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"证据结构非法；失败关闭：{exc}"]
