"""rapidfire-sft-v1 的 4 配置 × 独立训练 seed 发布级套件；标准库、失败关闭。

与数学 / FIM / xLAM 刻意分开：RapidFire 的 comparison 不是「一个实验三 seed」，
而是「4 组 LoRA/lr 配置 × 至少 3 个独立训练 seed」的网格，并且必须把
配置选择与最终验收分开：

  * 所有配置的所有 seed 只在 **validation** 上评估，用 3 seed 的 paired rouge_l
    增量均值确定性地选出一个配置（平局按 config_id 字典序）；
  * **test 只在选中配置上做一次最终验收**：选中配置的 3 个独立 seed 各评估一次，
    其余 12 个 checkpoint 不得产生任何 test 评估文件，test 结果绝不回流选择；
  * 逐题从 eval_manifest 的 raw 5 列重建 prompt / gold，用纯 Python 无 stemmer
    ROUGE-L 与 exact_match 重算，绝不加总证据摘要里的数字；
  * 顺序执行 4 配置，只统计时长/显存/步长成本，不推断上游并发调度或 16-24× 加速。

复用 suite_task 的冻结输入、统计与文件绑定工具，但不改写数学 / FIM / xLAM 路径，
也不把 completion-only 格式指标冒充客服业务正确率或真实 GPU 执行证明。
"""
from __future__ import annotations

import copy
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import baseline_evidence as be
from . import export_task as et
from . import math_task as mt
from . import provenance as pv
from . import rapidfire_task as rt
from . import suite_task as st

CODE_ROOT = st.CODE_ROOT
CODE_SNAPSHOT, LOCKFILE, STATE, FINAL = st.CODE_SNAPSHOT, st.LOCKFILE, st.STATE, st.FINAL
LIMITATION = (
    "统计单位为独立训练 seed × 配置；配置选择只用 validation 的 3 seed paired 均值，"
    "test 仅在选中配置上验收一次。Student-t 区间假设训练 seed 结果独立同分布且均值近似正态，"
    "不包含数据抽样不确定性与模型选择偏差，不能作为业务显著性证据。"
    "rouge_l 是纯 Python 无 stemmer LCS-F1，exact_match 是归一化全串相等；"
    "两者与上游 evaluate.load('rouge', use_stemmer=True) 数值不可比，上游从未报告任何数值，"
    "不执行生成文本、不证明客服业务正确或安全。4 配置顺序执行，"
    "并发调度收益与上游 16-24× 吞吐声明未做同预算消融，不可比较。"
    "JSON/hash 只证明仓内内容一致性，不认证证据实际来自 GPU。")
METRICS = rt.METRICS
REFERENCE_FILES = rt.REFERENCE_FILES
FULL_STEPS = rt.FULL_STEPS
SELECTION_SPLIT = "validation"
ACCEPTANCE_SPLIT = "test"
SELECTION_METRIC = "rouge_l"
SELECTION_STATISTIC = "paired_delta_mean"
CONFIG_PREFIX = "config-"


def _need(condition, message):
    if not condition:
        raise mt.TaskExecutionError(message)


def _configs(r):
    return [item["id"] for item in rt.config_grid(r)]


def _config_dir(root, config_id):
    return root / (CONFIG_PREFIX + config_id)


def _seed_dir(root, config_id, seed):
    return _config_dir(root, config_id) / f"seed-{seed}"


def _hashes(r):
    identity = rt.execution_identity(r)
    return identity["code_sha256"], identity["environment_lock_sha256"]


def _metrics(r):
    names = r.get("evaluation.metrics")
    _need(isinstance(names, list) and len(names) == len(METRICS) and set(names) == set(METRICS),
          "RapidFire evaluation.metrics 必须精确为 rouge_l / exact_match")
    return names


def _config_recipe(r, config_id, seed=None):
    """单个网格单元的生效配方；内容身份不含回填的验收阈值。"""
    content = _identity_recipe(r)
    base = pv.effective_recipe(content, seed) if seed is not None else content
    derived = rt.config_recipe(base, config_id)
    rt.require_supported(derived)
    return derived


def _identity_recipe(r):
    """内容身份不含「实跑后回填」的审核字段，只包含训练/评估协议本身。

    需剥离的回填字段：
      - evaluation.expected / evaluation.tolerance：首次实测后才写入的验收阈值；
      - execution.evidence：注册 C307/C405 时指向 baseline-suite / 导出证据的指针。
    这些字段只参与门禁判定或最终均值比较；若进入 recipe_sha256 / saved.data
    全量比较，注册 B2 的 recipe 会与训练时 checkpoint 的 effective_recipe.json
    失配，导致合法套件永远无法通过。be.fingerprint() 本来就排除了它们，
    这里让内容身份与之一致。注意 evaluation.export_tolerance 是训练前预声明的
    协议字段，必须保留，不得一并剥离。
    """
    data = copy.deepcopy(r.data)
    evaluation = data.get("evaluation")
    if isinstance(evaluation, dict):
        for key in ("expected", "tolerance"):
            evaluation.pop(key, None)
    execution = data.get("execution")
    if isinstance(execution, dict):
        execution.pop("evidence", None)
    from .model import Recipe
    return Recipe(r.id, r.path, data)


def _preflight(r, seeds, root, *, executing=False):
    rt.require_supported(r)
    _need(be.frozen_environment(r), "套件需要与 Recipe 一致的 frozen_environment")
    for key in ("model.revision", "dataset.revision", "source_sha"):
        rt.require_revision(r, key)
    rt.fixed_seeds(r)
    _metrics(r)
    _need(r.get("rapidfire.selection_policy") == rt.RAPIDFIRE_SPEC["selection_policy"],
          "selection_policy 必须与实现一致；拒绝不可重算的配置选择规则")
    ids = _configs(r)
    _need(len(ids) == 4, "必须完整覆盖 4 组配置；拒绝只挑部分配置做对比")
    for config_id in ids:
        for seed in seeds:
            effective = _config_recipe(r, config_id, seed)
            rt.config_kwargs(effective, {"mode": "full"},
                             _seed_dir(root, config_id, seed) / "full")
    if executing:
        _need(r.get("decision.gate") != "blocked" and r.get("baseline.result") != "passes",
              "训练决策门已阻断；不能绕过零训练基线及预算决策")


def _command(script, output_root, *, config_id, seed=None, checkpoint=None, eval_split=None):
    args = [sys.executable, str(CODE_ROOT / "runner" / script), "--recipe", rt.RECIPE_ID,
            "--full", "--output-root", str(output_root), "--config-id", config_id]
    if seed is not None:
        args += ["--train-seed", str(seed)]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    if eval_split is not None:
        args += ["--eval-split", eval_split]
    return args


def suite_plan(r, seeds, output_root):
    """只读声明；test 验收命令必须等 validation 选出配置后才能确定，不预先编造。"""
    seeds, root = st._seeds(seeds), st._root(output_root)
    try:
        rt.require_supported(r)
        supported, reason = True, None
    except mt.TaskExecutionError as exc:
        supported, reason = False, str(exc)
    ids = _configs(r)
    commands = []
    for split in (SELECTION_SPLIT, ACCEPTANCE_SPLIT):
        base_root = root / "base" / split
        commands.append({"kind": "zero-shot", "split": split,
                         "command": _command("zero_shot.py", base_root, config_id=r.get(
                             "execution.config_id"), eval_split=split),
                         "output_dir": str(base_root / "zero-shot")})
    runs = []
    for config_id in ids:
        for seed in seeds:
            seed_root = _seed_dir(root, config_id, seed)
            checkpoint = seed_root / "full"
            commands += [
                {"kind": "training", "config_id": config_id, "train_seed": seed,
                 "command": _command("train.py", seed_root, config_id=config_id, seed=seed),
                 "output_dir": str(checkpoint)},
                {"kind": "evaluation", "split": SELECTION_SPLIT, "config_id": config_id,
                 "train_seed": seed,
                 "command": _command("eval.py", seed_root / SELECTION_SPLIT,
                                     config_id=config_id, seed=seed,
                                     checkpoint=checkpoint, eval_split=SELECTION_SPLIT),
                 "output_dir": str(seed_root / SELECTION_SPLIT / "eval"),
                 "checkpoint": str(checkpoint)},
            ]
            runs.append({"config_id": config_id, "train_seed": seed,
                         "fingerprint": be.fingerprint(_config_recipe(r, config_id, seed)),
                         "checkpoint": str(checkpoint),
                         "validation": str(seed_root / SELECTION_SPLIT / "eval" / "evaluation.json")})
    return {"schema_version": 1, "kind": "suite-plan", "status": "declared", "recipe_id": r.id,
            "task_id": rt.TASK_ID, "fingerprint": be.fingerprint(r),
            "execution_supported": supported, "unsupported_reason": reason,
            "seeds": seeds, "config_ids": ids, "output_root": str(root), "shell": False,
            "data_seed": r.get("train.data_seed"),
            "evaluation_seed": r.get("evaluation.seed"),
            "training_source": {"id": rt.MODEL, "revision": r.get("model.revision"),
                                "independent_initialization": True,
                                "adapter_only": True, "base_weights_bundled": False},
            "protocol": rt.EVAL_PROTOCOL, "metrics": list(METRICS),
            "full_steps": FULL_STEPS,
            "selection": {"split": SELECTION_SPLIT, "metric": SELECTION_METRIC,
                          "statistic": SELECTION_STATISTIC,
                          "policy": r.get("rapidfire.selection_policy"),
                          "test_used_in_selection": False},
            "acceptance": {"split": ACCEPTANCE_SPLIT,
                           "commands": "deferred：validation 聚合后仅对选中配置生成，"
                                       "每个独立 seed 一次，test 结果不回流选择",
                           "evaluations": len(seeds)},
            "costs": {"sequential_execution": True, "concurrent_scheduling": False,
                      "upstream_throughput_comparable": False},
            "commands": commands, "runs": runs, "timeout_seconds": st.COMMAND_TIMEOUT_SECONDS,
            "resume_supported": False, "baseline_promotion": "none", "limitations": LIMITATION}


# ---------------------------------------------------------------- 内容身份与逐题重算

def _manifest(r, manifest):
    rt.validate_source_manifest(manifest, r)
    _need(manifest.get("dataset_revision") == r.get("dataset.revision"),
          "RapidFire source_manifest dataset_revision 不一致")
    counts = manifest.get("counts")
    _need(isinstance(counts, dict)
          and counts.get("source") == sum(counts.get(name, 0) for name in ("train", "validation", "test"))
          and type(counts.get("test_eval_used")) is int
          and type(counts.get("validation_used")) is int
          and 0 < counts["test_eval_used"] <= counts["test"]
          and 0 < counts["validation_used"] <= counts["validation"],
          "RapidFire source_manifest counts 不一致")
    for name, key in (("test", "test_eval_used"), ("validation", "validation_used")):
        hashes = manifest["split_hashes"][name]
        _need(counts[key] <= len(hashes), f"{name} 评估样本超出 split 内容")
    indices = manifest.get("eval_sample_indices")
    _need(isinstance(indices, list) and len(indices) == counts["test_eval_used"]
          and all(type(i) is int and i >= 0 for i in indices)
          and len(set(indices)) == len(indices),
          "RapidFire eval_sample_indices 与 held-out 样本不一致")
    return manifest


def _identity(r, data, hashes, *, split_name=None):
    _need(type(data.get("schema_version")) is int and data["schema_version"] == 1
          and data.get("stage") == "completed" and data.get("task_id") == rt.TASK_ID
          and data.get("recipe_id") == r.id and data.get("fingerprint") == be.fingerprint(r)
          and data.get("model_id") == rt.MODEL and data.get("dataset_id") == rt.DATASET
          and data.get("model_revision") == r.get("model.revision")
          and data.get("dataset_revision") == r.get("dataset.revision")
          and data.get("source_sha") == r.get("source_sha")
          and type(data.get("seed")) is int and data["seed"] == r.get("train.seed")
          and type(data.get("data_seed")) is int and data["data_seed"] == r.get("train.data_seed")
          and data.get("config_id") == r.get("execution.config_id"),
          "RapidFire 证据任务 / revision / fingerprint / config_id / seed 不一致")
    _need(data.get("code_sha256") == hashes[0] and data.get("environment_lock_sha256") == hashes[1],
          "证据混合 code / lock hash 或与当前执行身份不一致")
    _need(be.runtime_consistent(r, data) and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "非真实运行协议或 runtime 不一致")
    _need(bool(be.HEX64.fullmatch(str(data.get("recipe_sha256", "")))), "缺少 Recipe 内容 hash")
    if split_name is not None:
        _need(data.get("evaluation_split") == split_name
              and data.get("eval_stage") == rt.EVAL_STAGES[split_name],
              f"评估 split / 阶段不是 {split_name}；拒绝 validation 与 test 互串")


def _generation(r, identity):
    return {"do_sample": False, "num_beams": 1,
            "max_new_tokens": r.get("evaluation.max_new_tokens"),
            "eos_token_id": identity["eos_token_id"], "pad_token_id": identity["pad_token_id"],
            "use_cache": True, "seed": r.get("evaluation.seed"), "eos_truncation": True}


def _evaluation_content(r, path, data, manifest, split_name):
    """逐题从 raw 5 列重建 prompt/gold 并重算 rouge_l / exact_match；不信任摘要数字。"""
    _identity(r, data, _hashes(r), split_name=split_name)
    _need(data.get("split_manifest") == manifest and data.get("source_manifest") == manifest,
          "评估证据 split_manifest 与套件不一致")
    for name in REFERENCE_FILES:
        _need(et._json(et._local(path.parent, name)) is not None, f"缺少 {name}")
    _need(et._json(et._local(path.parent, "source_manifest.json")) == manifest,
          "评估 source_manifest.json 不一致")
    identity = data.get("tokenizer_identity")
    _need(isinstance(identity, dict) and identity.get("sha256") == rt._hash_manifest(identity)["sha256"],
          "tokenizer identity hash 无效")
    _need(et._json(et._local(path.parent, rt.TOKENIZER_IDENTITY_FILE)) == identity,
          "评估 tokenizer_identity.json 不一致")
    eval_manifest = data.get("evaluation_manifest")
    _need(isinstance(eval_manifest, dict)
          and eval_manifest.get("sha256") == rt._hash_manifest(eval_manifest)["sha256"],
          "RapidFire evaluation manifest hash 无效")
    _need(et._json(et._local(path.parent, rt.EVAL_MANIFEST_FILE)) == eval_manifest,
          "评估 eval_manifest.json 不一致")
    _need(data.get("evaluation_protocol") == rt.EVAL_PROTOCOL
          and eval_manifest.get("kind") == "rapidfire-eval-manifest"
          and eval_manifest.get("protocol") == rt.EVAL_PROTOCOL
          and eval_manifest.get("variant") == rt.EVAL_VARIANT
          and eval_manifest.get("evaluation_split") == split_name
          and eval_manifest.get("eval_stage") == rt.EVAL_STAGES[split_name]
          and eval_manifest.get("selection_policy") == r.get("rapidfire.selection_policy")
          and eval_manifest.get("task_id") == rt.TASK_ID
          and eval_manifest.get("config_grid_sha256") == rt.sha256_json(rt.config_grid(r))
          and eval_manifest.get("data_seed") == r.get("train.data_seed")
          and eval_manifest.get("max_new_tokens") == r.get("evaluation.max_new_tokens")
          and eval_manifest.get("tokenizer_identity") == identity
          and eval_manifest.get("source_manifest_sha256") == manifest["sha256"]
          and eval_manifest.get("gold_policy") == r.get("rapidfire.duplicate_policy")
          and eval_manifest.get("metric_definition") == r.get("rapidfire.metric_definition")
          and eval_manifest.get("generation_truncation") == r.get("rapidfire.generation_truncation")
          and eval_manifest.get("metric_comparable_to_upstream_rougeL") is False
          and eval_manifest.get("upstream_quantitative_evaluation") is False,
          "RapidFire 评估协议 / split / 输出格式 / tokenizer 不一致")
    examples = eval_manifest.get("samples")
    _need(isinstance(examples, list) and bool(examples)
          and eval_manifest.get("sample_count") == len(examples)
          and eval_manifest.get("samples_sha256") == rt.sha256_json(examples),
          "RapidFire 评估样本列表或内容 hash 不一致")
    used_key = "validation_used" if split_name == SELECTION_SPLIT else "test_eval_used"
    used = manifest["counts"][used_key]
    source_hashes = manifest["split_hashes"][split_name][:used]
    indices = (manifest["eval_sample_indices"] if split_name == ACCEPTANCE_SPLIT
               else [row["source_index"] for row in examples])
    _need(len(examples) == used == len(source_hashes), "评估样本数必须与 shared held-out 一致")
    rows = []
    for example, expected_hash, source_index in zip(examples, source_hashes, indices):
        raw = example.get("raw")
        _need(isinstance(raw, dict) and set(raw) == set(rt.REQUIRED_COLUMNS)
              and all(isinstance(raw[key], str) for key in raw),
              "评估样本缺少可重建的规范化原始 5 列")
        row = rt.parse_row(raw, source_index)
        prompt = rt.render_prompt(row)
        prompt_ids = example.get("prompt_ids")
        _need(example.get("sample_id") == row["sample_id"]
              and example.get("source_index") == source_index
              and example.get("content_sha256") == row["content_sha256"] == expected_hash
              and example.get("question_sha256") == row["question_sha256"]
              and raw["response"] == row["response"]
              and example.get("gold_sha256") == rt.sha256_json(row["response"])
              and example.get("prompt") == prompt
              and example.get("prompt_sha256") == rt.sha256_json(prompt)
              and isinstance(prompt_ids, list) and bool(prompt_ids)
              and all(type(token) is int and token >= 0 for token in prompt_ids)
              and example.get("input_sha256") == rt.sha256_json(prompt_ids),
              "RapidFire prompt / gold / token 输入重建失败")
        if split_name == ACCEPTANCE_SPLIT:
            _need(source_index == manifest["eval_sample_indices"][len(rows)],
                  "test 评估样本顺序与 source_manifest 不一致")
        rows.append(row)
    config = data.get("config")
    _need(isinstance(config, dict) and data.get("config_sha256") == rt.sha256_json(config),
          "评估配置内容 hash 不一致")
    _need(config.get("task_id") == rt.TASK_ID and config.get("protocol") == rt.EVAL_PROTOCOL
          and config.get("config_id") == r.get("execution.config_id")
          and config.get("evaluation_split") == split_name
          and config.get("eval_stage") == rt.EVAL_STAGES[split_name]
          and config.get("selection_policy") == r.get("rapidfire.selection_policy")
          and config.get("generation") == _generation(r, identity)
          and config.get("quantization") == "none"
          and config.get("compute_dtype") == "float16"
          and config.get("attention_implementation") == "eager"
          and config.get("model") == data.get("model_artifact")
          and config.get("tokenizer_identity") == identity
          and config.get("dataset_revision") == r.get("dataset.revision")
          and config.get("source_manifest_sha256") == manifest["sha256"]
          and config.get("eval_manifest_sha256") == eval_manifest["sha256"]
          and config.get("max_new_tokens") == r.get("evaluation.max_new_tokens")
          and config.get("do_sample") is False and config.get("num_beams") == 1
          and config.get("held_out_rows") == len(examples)
          and config.get("metric_definition") == r.get("rapidfire.metric_definition")
          and config.get("metric_comparable_to_upstream_rougeL") is False
          and config.get("concurrent_scheduling") is False
          and config.get("upstream_quantitative_evaluation") is False
          and config.get("zero_shot") is (data.get("model_role") == "base"),
          "评估生成 / split / 量化 / completion-only 协议不一致")
    record = data.get("predictions_file", {})
    _need(isinstance(record, dict), "缺少预测文件记录")
    prediction_path = et._local(path.parent, record.get("path"))
    _need(be.hashed_file(path.parent, record), "预测文件缺失或 hash 不一致")
    document = et._json(prediction_path)
    _need(isinstance(document, dict) and document.get("task_id") == rt.TASK_ID
          and document.get("protocol") == rt.EVAL_PROTOCOL
          and document.get("recipe_id") == r.id
          and document.get("config_id") == r.get("execution.config_id")
          and document.get("evaluation_split") == split_name
          and document.get("eval_stage") == rt.EVAL_STAGES[split_name]
          and document.get("metrics") == data.get("metrics")
          and document.get("sha256") == rt.sha256_json(document.get("predictions")),
          "预测文档身份 / split / 指标不一致")
    predictions = document.get("predictions")
    description = data.get("predictions", {})
    _need(isinstance(predictions, list) and len(predictions) == len(examples)
          and description.get("path") == record.get("path")
          and type(description.get("count")) is int and description["count"] == len(predictions)
          and description.get("sha256") == rt.sha256_json(predictions)
          and data.get("predictions_sha256") == rt.sha256_json(predictions),
          "逐题预测内容 hash 或数量不一致")
    eos = identity["eos_token"]
    for prediction, example, row in zip(predictions, examples, rows):
        text = prediction.get("generated_text")
        _need(isinstance(prediction, dict) and isinstance(text, str)
              and all(prediction.get(key) == example[key] for key in (
                  "sample_id", "source_index", "content_sha256", "question_sha256",
                  "gold_sha256", "prompt_sha256"))
              and prediction.get("eos_token_id") == identity["eos_token_id"]
              and prediction.get("max_new_tokens") == r.get("evaluation.max_new_tokens"),
              "逐题生成记录与评估 manifest 不一致")
        # 关键：从原始生成文本重新打分，而不是把证据里的数字加总一遍。
        _need(rt.score_prediction(row, text, eos) == {
            key: prediction.get(key) for key in (
                "sample_id", "source_index", "content_sha256", "question_sha256",
                "gold_sha256", "rouge_l", "exact_match", "prediction_sha256",
                "clean_prediction_sha256")},
            "RapidFire rouge_l / exact_match 与逐题生成文本重算不一致")
    _need(data.get("metrics") == rt.recompute_metrics(predictions),
          "RapidFire 指标与逐题重算不一致")
    counts = data.get("metric_counts")
    _need(isinstance(counts, dict) and counts.get("total") == len(predictions)
          and all(counts.get(name) == sum(row[name] for row in predictions) for name in METRICS),
          "metric_counts 与逐题预测不一致")
    _need(type(data.get("sample_count")) is int and data["sample_count"] == len(predictions),
          "sample_count 与预测数量不一致")
    return predictions, prediction_path, eval_manifest, identity


def _training(r, checkpoint, hashes, manifest):
    """复用 rapidfire_task.checkpoint_path 的 adapter / 张量 / 协议强校验。"""
    rt.checkpoint_path({}, checkpoint, recipe=r)
    training = et._json(et._local(checkpoint, "training.json"))
    _identity(r, training, hashes)
    _need(training.get("kind") == "training" and training.get("mode") == "full"
          and training.get("trainer") == "Trainer" and training.get("config_id") == r.get("execution.config_id"),
          "不是 RapidFire full Trainer 训练证据")
    expected = rt.config_kwargs(r, {"recipe_id": r.id, "task_id": rt.TASK_ID, "mode": "full"}, checkpoint)
    expected_identity = rt.training_arguments_identity(expected)
    config = training.get("config")
    _need(isinstance(config, dict) and training.get("config_sha256") == rt.sha256_json(config)
          and rt.training_arguments_identity(config.get("training_arguments")) == expected_identity
          and rt.training_arguments_identity(config.get("api_training_arguments")) in (
              expected_identity, rt.training_arguments_identity({
                  ("eval_strategy" if k == "evaluation_strategy" else k): v for k, v in expected.items()}))
          and config.get("config") == rt.active_config(r)
          and config.get("config_id") == r.get("execution.config_id")
          and config.get("rapidfire") == r.get("rapidfire") and config.get("peft") == r.get("peft")
          and config.get("quantization") == "none"
          and config.get("attention_implementation") == "eager"
          and config.get("concurrent_scheduling") is False
          and config.get("upstream_metadata") == dict(r.get("upstream"))
          and config.get("lora_execution", {}).get("tensor_reload_check", {}).get(
              "strict_values_verified") is True,
          "训练有效配置 / completion-only 协议 / 预算不一致")
    _need(type(training.get("steps")) is int and training["steps"] == training.get("max_steps")
          == expected["max_steps"] == FULL_STEPS, f"训练未完成 full={FULL_STEPS} 步预算")
    _need(training.get("split_manifest") == manifest and training.get("source_manifest") == manifest
          and et._json(et._local(checkpoint, "source_manifest.json")) == manifest,
          "训练 source manifest 与套件不一致")
    identity = training.get("tokenizer_identity")
    transform = training.get("transformation_manifest")
    _need(et._json(et._local(checkpoint, rt.TOKENIZER_IDENTITY_FILE)) == identity
          and et._json(et._local(checkpoint, rt.TRANSFORM_FILE)) == transform,
          "训练 tokenizer / transform 文件不一致")
    rt.validate_transform_manifest(transform, manifest, r, identity)
    artifact = rt.artifact_manifest(checkpoint)
    _need(rt.same_artifact(training.get("model_artifact"), artifact), "训练 artifact 文件/hash 不一致")
    runtime = training.get("runtime")
    _need(isinstance(runtime, dict) and be.finite_number(runtime.get("duration_seconds"))
          and runtime["duration_seconds"] > 0
          and type(runtime.get("peak_cuda_memory_bytes")) is int
          and runtime["peak_cuda_memory_bytes"] >= 0,
          "训练 runtime 缺少可复核的时长 / 显存实测")
    return training, artifact, identity


def _base_artifact(r):
    return rt.base_artifact(r)


def _paired(r, base, base_predictions, trained, trained_predictions):
    _need(base.get("mode") == "zero-shot" and base.get("model_role") == "base"
          and base.get("model_artifact") == _base_artifact(r), "基座评估不是同 revision 冻结 base")
    _need(trained.get("mode") == "eval" and trained.get("model_role") == "checkpoint",
          "训练模型评估模式/角色不一致")
    _need(base["config"]["generation"] == trained["config"]["generation"]
          and base.get("evaluation_split") == trained.get("evaluation_split")
          and base.get("eval_stage") == trained.get("eval_stage"),
          "base / trained 解码协议或评估 split 不一致")
    _need(base.get("evaluation_manifest") == trained.get("evaluation_manifest")
          and base.get("tokenizer_identity") == trained.get("tokenizer_identity"),
          "base / trained completion-only 输入或 tokenizer 不一致")
    keys = ("sample_id", "source_index", "content_sha256", "question_sha256",
            "gold_sha256", "prompt_sha256")
    _need(len(base_predictions) == len(trained_predictions)
          and all(all(a.get(k) == b.get(k) for k in keys)
                  for a, b in zip(base_predictions, trained_predictions)),
          "base / trained 逐题 prompt / gold 不一致")


def _evaluation_role(r, path, data, manifest, role, split_name, artifact=None, prediction_path=None):
    _need(data.get("model_role") == role, f"评估角色必须为 {role}")
    if role == "base":
        _need(data.get("model_artifact") == _base_artifact(r), "基座 artifact 不一致")
    else:
        model = data.get("model_artifact")
        _need(isinstance(artifact, dict) and isinstance(model, dict)
              and rt.same_artifact(model, artifact),
              "训练模型 artifact 与 checkpoint 不一致")
    predictions, path_of_predictions, _, _ = _evaluation_content(r, path, data, manifest, split_name)
    if prediction_path is not None:
        _need(path_of_predictions == prediction_path, "预测文件路径不一致")
    return predictions, path_of_predictions


def _stats_for(r, runs):
    return {name: {"trained": st._stats([run["trained_metrics"][name] for run in runs]),
                   "paired_delta": st._stats([run["trained_metrics"][name] - run["base_metrics"][name]
                                              for run in runs])}
            for name in _metrics(r)}


def _selection(r, validation_runs, seeds):
    """只用 validation 的 3 seed paired 均值选配置；平局按 config_id 字典序。"""
    ids = sorted(_configs(r))
    candidates = []
    for config_id in ids:
        subset = [run for run in validation_runs if run["config_id"] == config_id]
        _need(len(subset) == len(seeds)
              and sorted(run["train_seed"] for run in subset) == sorted(seeds),
              f"{config_id} 缺少完整的 {len(seeds)} 个独立训练 seed validation 结果")
        deltas = [run["trained_metrics"][SELECTION_METRIC] - run["base_metrics"][SELECTION_METRIC]
                  for run in subset]
        candidates.append({"config_id": config_id,
                           SELECTION_STATISTIC: math.fsum(deltas) / len(deltas),
                           "seed_paired_deltas": deltas})
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate[SELECTION_STATISTIC] > best[SELECTION_STATISTIC]:
            best = candidate
    runner_up = max((c[SELECTION_STATISTIC] for c in candidates
                     if c["config_id"] != best["config_id"]), default=None)
    return {"split": SELECTION_SPLIT, "metric": SELECTION_METRIC,
            "statistic": SELECTION_STATISTIC, "policy": r.get("rapidfire.selection_policy"),
            "tie_break": "config_id 字典序取首个最大值",
            "candidates": candidates, "selected_config_id": best["config_id"],
            "margin_to_runner_up": (best[SELECTION_STATISTIC] - runner_up
                                    if runner_up is not None else None),
            "test_used_in_selection": False}


def _costs(r, training_rows):
    """顺序执行的真实时长 / 显存 / 步长成本；不推断并发加速比。"""
    per_config, total = {}, 0.0
    for config_id in _configs(r):
        subset = [row for row in training_rows if row["config_id"] == config_id]
        _need(bool(subset), f"{config_id} 没有训练成本记录")
        durations = [row["duration_seconds"] for row in subset]
        for duration in durations:
            _need(be.finite_number(duration) and duration > 0, "训练时长必须是有限正数")
        seconds = math.fsum(durations)
        total += seconds
        steps = sum(row["steps"] for row in subset)
        _need(steps == FULL_STEPS * len(subset), "成本对照的训练步数不一致")
        per_config[config_id] = {
            "seeds": len(subset), "total_seconds": seconds,
            "mean_seconds": seconds / len(subset),
            "steps_per_second": steps / seconds,
            "peak_cuda_memory_bytes_max": max(row["peak_cuda_memory_bytes"] for row in subset),
            "total_steps": steps,
        }
    return {"statistical_unit": "independent_training_seed",
            "sequential_execution": True, "concurrent_scheduling": False,
            "upstream_throughput_comparable": False,
            "per_config": per_config, "sequential_total_seconds": total,
            "note": "4 配置顺序执行的实测时长/显存；上游 num_chunks=4 并发调度与 16-24× "
                    "吞吐声明没有同预算消融，不做数值对照"}


def _records_check(r, root, records):
    _need(isinstance(records, dict) and bool(records), "套件缺少完整文件 records")
    for relative, record in records.items():
        _need(isinstance(record, dict) and record.get("path") == relative
              and be.hashed_file(root, record), f"套件绑定文件校验失败：{relative}")
        et._local(root, relative)


def _no_unselected_test(root, selected, seeds):
    """test 只验收一次：其余配置的 checkpoint 不得存在任何 test 评估证据。"""
    found = sorted(
        p for p in et._tree_files(root)
        if p.name == "evaluation.json"
        and p.relative_to(root).as_posix().startswith(CONFIG_PREFIX)
        and f"/{ACCEPTANCE_SPLIT}/" in p.relative_to(root).as_posix())
    expected = sorted(_seed_dir(root, selected, seed) / ACCEPTANCE_SPLIT / "eval" / "evaluation.json"
                      for seed in seeds)
    _need(found == expected, "test 评估出现在未选中配置或数量不对；拒绝用 test 反复选配置")


def validate_suite_payload(r, data, path):
    """发布门禁与聚合器共用：只读重算任意目录布局的完整内容，拒绝宽松 JSON。"""
    root = et._no_symlinks(path.parent)
    _need(isinstance(data, dict) and data.get("task_id") == rt.TASK_ID
          and data.get("kind") == "baseline-suite" and data.get("status") == "completed"
          and data.get("recipe_id") == r.id and data.get("fingerprint") == be.fingerprint(r),
          "不是绑定当前 RapidFire 实验的 completed baseline-suite")
    runs = data.get("runs")
    acceptance = data.get("acceptance_runs")
    _need(isinstance(runs, list) and all(isinstance(run, dict) for run in runs)
          and isinstance(acceptance, list) and all(isinstance(run, dict) for run in acceptance),
          "套件 runs / acceptance_runs 非法")
    seeds = st._seeds(sorted({run.get("train_seed") for run in runs}))
    _need(sorted(data.get("seeds", [])) == seeds and data.get("config_ids") == _configs(r),
          "套件 seed / 配置网格声明与证据不一致")
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
          and manifest_path.relative_to(root).as_posix()
          == f"base/{ACCEPTANCE_SPLIT}/zero-shot/source_manifest.json",
          "套件 split 身份或共享 manifest 来源不一致")
    base = {}
    base_predictions = {}
    for split in (SELECTION_SPLIT, ACCEPTANCE_SPLIT):
        record = (data.get("base_evaluation") or {}).get(split)
        base[split], base_path = linked(record, f"base_evaluation.{split}")
        _identity(r, base[split], hashes, split_name=split)
        predictions, prediction_path = _evaluation_role(
            r, base_path, base[split], manifest, "base", split)
        base_predictions[split] = predictions
        _need((data.get("base_predictions") or {}).get(split) == et._record(prediction_path, root),
              f"基座 {split} 预测记录不一致")
        for name in REFERENCE_FILES:
            file = base_path.parent / name
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root),
                  f"基座参考文件未绑定：{split}/{name}")

    checkpoints, training_rows, artifacts = set(), [], {}
    for run, config_id, seed in (
            (item, item.get("config_id"), item.get("train_seed")) for item in runs):
        _need(config_id in _configs(r) and type(seed) is int, "运行记录缺少合法 config_id / seed")
        effective = _config_recipe(r, config_id, seed)
        training, training_path = linked(run.get("training_evidence"), "training_evidence")
        checkpoint = training_path.parent
        _need(checkpoint not in checkpoints, "独立 seed 共用 checkpoint 目录")
        checkpoints.add(checkpoint)
        et._tree_files(checkpoint)
        training, artifact, _ = _training(effective, checkpoint, hashes, manifest)
        _identity(effective, training, hashes)
        _need(training.get("seed") == seed and training.get("config_id") == config_id,
              "训练 seed / config_id 与套件记录不一致")
        artifacts[(config_id, seed)] = artifact
        runtime = training["runtime"]
        training_rows.append({"config_id": config_id, "train_seed": seed,
                              "duration_seconds": runtime["duration_seconds"],
                              "peak_cuda_memory_bytes": runtime["peak_cuda_memory_bytes"],
                              "steps": training["steps"]})
        for file in et._tree_files(checkpoint):
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root),
                  f"checkpoint 文件未绑定：{file.name}")
        trained, trained_path = linked(run.get("trained_evaluation"), "trained_evaluation")
        _identity(effective, trained, hashes, split_name=SELECTION_SPLIT)
        trained_predictions, trained_prediction_path = _evaluation_role(
            effective, trained_path, trained, manifest, "checkpoint", SELECTION_SPLIT, artifact)
        _paired(r, base[SELECTION_SPLIT], base_predictions[SELECTION_SPLIT],
                trained, trained_predictions)
        _need(run.get("base_evaluation") == data["base_evaluation"][SELECTION_SPLIT]
              and run.get("kind") == "paired-evaluation"
              and run.get("evaluation_split") == SELECTION_SPLIT
              and run.get("training_completed") is True
              and run.get("sample_count") == len(base_predictions[SELECTION_SPLIT])
              and run.get("split_sha256") == manifest["sha256"]
              and run.get("base_metrics") == base[SELECTION_SPLIT]["metrics"]
              and run.get("trained_metrics") == trained["metrics"]
              and run.get("predictions") == et._record(trained_prediction_path, root)
              and run.get("fingerprint") == be.fingerprint(effective)
              and rt.artifact_identity(run.get("model_artifact")) == rt.artifact_identity(artifact),
              "validation 运行摘要身份/指标不一致")
        for name in REFERENCE_FILES:
            file = trained_path.parent / name
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root),
                  f"validation 评估参考文件未绑定：{name}")
    validation_stats = {config_id: _stats_for(
        r, [run for run in runs if run["config_id"] == config_id]) for config_id in _configs(r)}
    _need(data.get("validation_stats") == validation_stats, "validation 统计未重算或已改变")
    selection = _selection(r, runs, seeds)
    _need(data.get("selection") == selection, "配置选择不是由 validation 结果确定性重算")
    selected = selection["selected_config_id"]
    _no_unselected_test(root, selected, seeds)
    _need(len(acceptance) == len(seeds)
          and sorted(run.get("train_seed") for run in acceptance) == sorted(seeds)
          and all(run.get("config_id") == selected for run in acceptance),
          "test 验收必须只覆盖选中配置的每个独立 seed")
    for run in acceptance:
        seed = run.get("train_seed")
        effective = _config_recipe(r, selected, seed)
        trained, trained_path = linked(run.get("trained_evaluation"), "acceptance_evaluation")
        _identity(effective, trained, hashes, split_name=ACCEPTANCE_SPLIT)
        artifact = artifacts[(selected, seed)]
        trained_predictions, trained_prediction_path = _evaluation_role(
            effective, trained_path, trained, manifest, "checkpoint", ACCEPTANCE_SPLIT, artifact)
        _paired(r, base[ACCEPTANCE_SPLIT], base_predictions[ACCEPTANCE_SPLIT],
                trained, trained_predictions)
        _need(run.get("base_evaluation") == data["base_evaluation"][ACCEPTANCE_SPLIT]
              and run.get("kind") == "paired-evaluation"
              and run.get("evaluation_split") == ACCEPTANCE_SPLIT
              and run.get("training_completed") is True
              and run.get("training_evidence") == next(
                  item["training_evidence"] for item in runs
                  if item["config_id"] == selected and item["train_seed"] == seed)
              and run.get("sample_count") == len(base_predictions[ACCEPTANCE_SPLIT])
              and run.get("split_sha256") == manifest["sha256"]
              and run.get("base_metrics") == base[ACCEPTANCE_SPLIT]["metrics"]
              and run.get("trained_metrics") == trained["metrics"]
              and run.get("predictions") == et._record(trained_prediction_path, root)
              and run.get("fingerprint") == be.fingerprint(effective)
              and rt.artifact_identity(run.get("model_artifact")) == rt.artifact_identity(artifact),
              "test 验收运行摘要身份/指标不一致")
        for name in REFERENCE_FILES:
            file = trained_path.parent / name
            _need(records.get(file.relative_to(root).as_posix()) == et._record(file, root),
                  f"test 评估参考文件未绑定：{name}")
    acceptance_stats = _stats_for(r, acceptance)
    # acceptance 是「用 expected / tolerance 做的验收判定」，而阈值按仓库流程允许在
    # 首次实测、聚合完成之后才预声明，因此它不是从证据重算的客观内容，不能参与
    # 强相等（否则任何先测量后声明阈值的合法套件都永远失配）。数学 / FIM / xLAM
    # 三个套件同样只写不校验。这里只校验结构；真正的验收由 suite_problems 末尾
    # 用当前 recipe 的阈值独立判定，绝不采信 JSON 里自称的 passed。
    declared_acceptance = data.get("acceptance")
    _need(isinstance(declared_acceptance, dict)
          and declared_acceptance.get("status") in ("not_evaluated", "passed", "failed"),
          "套件 acceptance 声明结构非法")
    _need(data.get("acceptance_stats") == acceptance_stats
          and data.get("stats") == acceptance_stats
          and data.get("costs") == _costs(r, training_rows)
          and data.get("statistical_unit") == "independent_training_seed"
          and data.get("selection_split") == SELECTION_SPLIT
          and data.get("acceptance_split") == ACCEPTANCE_SPLIT
          and data.get("fixed_test") is True
          and data.get("test_evaluated_once") is True
          and data.get("test_evaluations") == len(seeds)
          and data.get("training_seed_variance_measured") is True
          and data.get("concurrent_scheduling") is False
          and data.get("upstream_throughput_comparable") is False,
          "套件统计 / 成本 / 验收声明未重算")
    st._guard(r, root, hashes)
    for file in et._tree_files(root):
        relative = file.relative_to(root).as_posix()
        _need(relative in records or relative in (FINAL, STATE),
              f"套件包含未绑定文件：{relative}")
    for record in records.values():
        _need(be.hashed_file(root, record), "套件校验期间证据内容漂移")
    return selection


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
    base, base_predictions, base_records, base_prediction_records = {}, {}, {}, {}
    for split in (SELECTION_SPLIT, ACCEPTANCE_SPLIT):
        base_path = et._local(root, f"base/{split}/zero-shot/evaluation.json")
        base_records[split] = bind(base_path)
        base[split] = et._json(base_path)
        _need(isinstance(base[split], dict), f"base {split} evaluation 必须为对象")
        _identity(r, base[split], hashes, split_name=split)
        manifest = base[split].get("split_manifest")
        _manifest(r, manifest)
        predictions, prediction_path = _evaluation_role(
            r, base_path, base[split], manifest, "base", split)
        base_predictions[split] = predictions
        base_prediction_records[split] = bind(prediction_path)
        for name in REFERENCE_FILES:
            bind(base_path.parent / name)
    manifest = base[ACCEPTANCE_SPLIT]["split_manifest"]
    manifest_record = records[
        f"base/{ACCEPTANCE_SPLIT}/zero-shot/source_manifest.json"]
    runs, acceptance, checkpoints, artifacts, training_rows = [], [], set(), {}, []
    for config_id in _configs(r):
        for seed in seeds:
            effective = _config_recipe(r, config_id, seed)
            seed_root = _seed_dir(root, config_id, seed)
            checkpoint = et._local(seed_root, "full")
            _need(checkpoint not in checkpoints, "重复 checkpoint 目录")
            checkpoints.add(checkpoint)
            for file in et._tree_files(checkpoint):
                bind(file)
            training, artifact, _ = _training(effective, checkpoint, hashes, manifest)
            _identity(effective, training, hashes)
            _need(training.get("seed") == seed and training.get("config_id") == config_id,
                  "训练 seed / config_id 与套件不一致")
            artifacts[(config_id, seed)] = artifact
            runtime = training["runtime"]
            training_rows.append({"config_id": config_id, "train_seed": seed,
                                  "duration_seconds": runtime["duration_seconds"],
                                  "peak_cuda_memory_bytes": runtime["peak_cuda_memory_bytes"],
                                  "steps": training["steps"]})
            training_record = bind(checkpoint / "training.json")
            validation_path = et._local(seed_root, f"{SELECTION_SPLIT}/eval/evaluation.json")
            validation_record = bind(validation_path)
            trained = et._json(validation_path)
            _identity(effective, trained, hashes, split_name=SELECTION_SPLIT)
            trained_predictions, trained_prediction_path = _evaluation_role(
                effective, validation_path, trained, manifest, "checkpoint", SELECTION_SPLIT, artifact)
            _paired(r, base[SELECTION_SPLIT], base_predictions[SELECTION_SPLIT],
                    trained, trained_predictions)
            prediction_record = bind(trained_prediction_path)
            for name in REFERENCE_FILES:
                bind(validation_path.parent / name)
            runs.append({"kind": "paired-evaluation", "training_completed": True,
                         "config_id": config_id, "train_seed": seed,
                         "evaluation_split": SELECTION_SPLIT,
                         "fingerprint": be.fingerprint(effective),
                         "sample_count": len(base_predictions[SELECTION_SPLIT]),
                         "split_sha256": manifest["sha256"],
                         "training_evidence": training_record,
                         "base_evaluation": base_records[SELECTION_SPLIT],
                         "trained_evaluation": validation_record,
                         "predictions": prediction_record,
                         "model_artifact": artifact,
                         "base_metrics": base[SELECTION_SPLIT]["metrics"],
                         "trained_metrics": trained["metrics"]})
    validation_stats = {config_id: _stats_for(
        r, [run for run in runs if run["config_id"] == config_id]) for config_id in _configs(r)}
    selection = _selection(r, runs, seeds)
    selected = selection["selected_config_id"]
    _no_unselected_test(root, selected, seeds)
    for seed in seeds:
        effective = _config_recipe(r, selected, seed)
        test_path = et._local(_seed_dir(root, selected, seed),
                              f"{ACCEPTANCE_SPLIT}/eval/evaluation.json")
        test_record = bind(test_path)
        trained = et._json(test_path)
        _identity(effective, trained, hashes, split_name=ACCEPTANCE_SPLIT)
        trained_predictions, trained_prediction_path = _evaluation_role(
            effective, test_path, trained, manifest, "checkpoint", ACCEPTANCE_SPLIT,
            artifacts[(selected, seed)])
        _paired(r, base[ACCEPTANCE_SPLIT], base_predictions[ACCEPTANCE_SPLIT],
                trained, trained_predictions)
        prediction_record = bind(trained_prediction_path)
        for name in REFERENCE_FILES:
            bind(test_path.parent / name)
        acceptance.append({"kind": "paired-evaluation", "training_completed": True,
                           "config_id": selected, "train_seed": seed,
                           "evaluation_split": ACCEPTANCE_SPLIT,
                           "fingerprint": be.fingerprint(effective),
                           "sample_count": len(base_predictions[ACCEPTANCE_SPLIT]),
                           "split_sha256": manifest["sha256"],
                           "training_evidence": next(
                               run["training_evidence"] for run in runs
                               if run["config_id"] == selected and run["train_seed"] == seed),
                           "base_evaluation": base_records[ACCEPTANCE_SPLIT],
                           "trained_evaluation": test_record,
                           "predictions": prediction_record,
                           "model_artifact": artifacts[(selected, seed)],
                           "base_metrics": base[ACCEPTANCE_SPLIT]["metrics"],
                           "trained_metrics": trained["metrics"]})
    acceptance_stats = _stats_for(r, acceptance)
    result = {"schema_version": 1, "kind": "baseline-suite", "status": "completed",
              "task_id": rt.TASK_ID, "recipe_id": r.id, "fingerprint": be.fingerprint(r),
              "completed_at": datetime.now(timezone.utc).isoformat(),
              "model_revision": r.get("model.revision"), "dataset_revision": r.get("dataset.revision"),
              "code_sha256": hashes[0], "environment_lock_sha256": hashes[1],
              "code_snapshot": records[CODE_SNAPSHOT], "environment_lock": records[LOCKFILE],
              "split_manifest": manifest_record, "split_sha256": manifest["sha256"],
              "base_evaluation": base_records, "base_predictions": base_prediction_records,
              "seeds": seeds, "config_ids": _configs(r), "runs": runs,
              "acceptance_runs": acceptance, "validation_stats": validation_stats,
              "acceptance_stats": acceptance_stats, "stats": acceptance_stats,
              "selection": selection, "costs": _costs(r, training_rows),
              "acceptance": st._acceptance(r, acceptance_stats),
              "statistical_unit": "independent_training_seed",
              "selection_split": SELECTION_SPLIT, "acceptance_split": ACCEPTANCE_SPLIT,
              "fixed_test": True, "test_evaluated_once": True,
              "test_evaluations": len(seeds), "training_seed_variance_measured": True,
              "concurrent_scheduling": False, "upstream_throughput_comparable": False,
              "baseline_promotion": "none", "records": records, "limitations": LIMITATION}
    validate_suite_payload(r, result, root / FINAL)
    st._guard(r, root, hashes)
    for (config_id, seed), artifact in artifacts.items():
        _need(rt.artifact_identity(rt.artifact_manifest(_seed_dir(root, config_id, seed) / "full"))
              == rt.artifact_identity(artifact), "聚合期间 checkpoint 权重漂移")
    _need(not (root / FINAL).exists(), "聚合期间 suite.json 已被创建，拒绝覆盖")
    return mt.save_json(root / FINAL, result)


def _selection_from_root(r, seeds, root, hashes):
    """run_suite 阶段二：只读 validation 证据，确定性生成 test 命令，绝不看 test 结果。"""
    manifest = None
    base = {}
    for split in (SELECTION_SPLIT,):
        path = et._local(root, f"base/{split}/zero-shot/evaluation.json")
        base[split] = et._json(path)
        _identity(r, base[split], hashes, split_name=split)
        manifest = base[split]["split_manifest"]
    _manifest(r, manifest)
    runs = []
    for config_id in _configs(r):
        for seed in seeds:
            effective = _config_recipe(r, config_id, seed)
            path = et._local(_seed_dir(root, config_id, seed),
                             f"{SELECTION_SPLIT}/eval/evaluation.json")
            trained = et._json(path)
            _identity(effective, trained, hashes, split_name=SELECTION_SPLIT)
            checkpoint = _seed_dir(root, config_id, seed) / "full"
            _training(effective, checkpoint, hashes, manifest)
            runs.append({"config_id": config_id, "train_seed": seed,
                         "base_metrics": base[SELECTION_SPLIT]["metrics"],
                         "trained_metrics": trained["metrics"]})
    return _selection(r, runs, seeds)


def run_suite(r, seeds, output_root):
    """在新 root 顺序实跑；test 命令只在 validation 选出配置之后生成，失败不 resume。"""
    seeds, root = st._seeds(seeds), et._no_symlinks(st._root(output_root))
    _preflight(r, seeds, root, executing=True)
    _need(not root.exists() or (root.is_dir() and not any(root.iterdir())),
          "拒绝非空 root；失败后仅允许 aggregate-only 或新 root 重跑")
    plan = suite_plan(r, seeds, root)
    import hashlib
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
        state["status"] = "validation_completed"
        mt.save_json(root / STATE, state)
        selection = _selection_from_root(r, seeds, root, hashes)
        state["selection"] = selection
        state["status"] = "acceptance_running"
        mt.save_json(root / STATE, state)
        selected = selection["selected_config_id"]
        for offset, seed in enumerate(seeds):
            seed_root = _seed_dir(root, selected, seed)
            step = {"kind": "evaluation", "split": ACCEPTANCE_SPLIT, "config_id": selected,
                    "train_seed": seed,
                    "command": _command("eval.py", seed_root / ACCEPTANCE_SPLIT,
                                        config_id=selected, seed=seed,
                                        checkpoint=seed_root / "full", eval_split=ACCEPTANCE_SPLIT),
                    "output_dir": str(seed_root / ACCEPTANCE_SPLIT / "eval"),
                    "checkpoint": str(seed_root / "full")}
            state["active_command"] = step["command"]
            mt.save_json(root / STATE, state)
            st._guard(r, root, hashes)
            _need(pv.file_hash(r.path) == recipe_hash, "执行期间 Recipe 文件漂移")
            info = st._launch(step["command"], root, len(plan["commands"]) + offset)
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
        split = data.get("evaluation_split", ACCEPTANCE_SPLIT)
        _need(split in rt.EVAL_SPLITS, "未知评估 split")
        _need(data.get("kind") == "evaluation" and data.get("stage") == "completed"
              and data.get("model_role") == role and data.get("recipe_id") == r.id
              and data.get("model_revision") == r.get("model.revision")
              and data.get("dataset_revision") == r.get("dataset.revision")
              and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
              "评估来源、角色或输入不一致")
        _need(be.runtime_consistent(r, data), "评估运行环境与 frozen lock 不一致或缺失")
        _identity(r, data, _hashes(r), split_name=split)
        source = data.get("split_manifest")
        _manifest(r, source)
        _need(manifest is None or source == manifest, "评估 manifest 与套件不一致")
        artifact = data.get("model_artifact") if role != "base" else None
        _evaluation_role(r, path, data, source, role, split, artifact)
        names = _metrics(r)
        metrics = data.get("metrics")
        _need(isinstance(metrics, dict) and all(be.finite_number(metrics.get(name)) for name in names),
              "RapidFire 指标缺失或不是有限数值")
        return []
    except (mt.TaskExecutionError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"RapidFire 评估内容拒绝：{exc}"]


def zero_train_validation(r, data, path):
    """C306：只认 test split 上冻结 base 的 zero-shot，不接受 validation 冒充。"""
    _need(data.get("mode") == "zero-shot" and data.get("model_role") == "base"
          and data.get("model_artifact") == _base_artifact(r), "不是冻结 base 的 zero-shot 评估")
    _identity(r, data, _hashes(r), split_name=ACCEPTANCE_SPLIT)
    manifest = data.get("split_manifest")
    _manifest(r, manifest)
    _evaluation_role(r, path, data, manifest, "base", ACCEPTANCE_SPLIT)


def suite_problems(r):
    """C307：通用摘要 + 与生产聚合器共用的 4 配置内容强校验。"""
    try:
        data, path = be.read_json(r, "execution.evidence")
        if data is None:
            return ["缺少可读的 baseline-suite JSON"]
        problems = []

        def need(ok, text):
            if not ok:
                problems.append(text)

        need(data.get("schema_version") == 1 and data.get("kind") == "baseline-suite"
             and data.get("status") == "completed" and data.get("task_id") == rt.TASK_ID,
             "不是已完成的 RapidFire baseline-suite v1")
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
             "期望指标必须预先声明并覆盖全部 RapidFire metrics")
        need(isinstance(tolerance, dict) and set(names) <= set(tolerance)
             and all(be.finite_number(tolerance.get(n)) and tolerance[n] >= 0 for n in names),
             "容差必须为覆盖全部 metrics 的非负数值表")
        runs, acceptance = data.get("runs"), data.get("acceptance_runs")
        need(isinstance(runs, list) and len(runs) >= len(_configs(r)) * 3,
             "至少需要 4 配置 × 3 独立训练 seed 的 validation 运行")
        need(isinstance(acceptance, list) and len(acceptance) >= 3,
             "至少需要三个独立训练 seed 的 test 验收运行，而不是重复 greedy 解码")
        if not isinstance(runs, list) or not isinstance(acceptance, list) or len(acceptance) < 3:
            return problems + ["至少需要三个独立训练 seed 的运行，而不是重复 greedy 解码"]
        seeds = [run.get("train_seed") for run in runs if isinstance(run, dict)]
        need(len(seeds) == len(runs) and all(type(s) is int for s in seeds)
             and len(set(seeds)) >= 3, "训练 seed 必须是至少三个互异整数")
        selection = data.get("selection") or {}
        need(selection.get("split") == SELECTION_SPLIT
             and selection.get("test_used_in_selection") is False
             and data.get("selection_split") == SELECTION_SPLIT
             and data.get("acceptance_split") == ACCEPTANCE_SPLIT
             and data.get("test_evaluated_once") is True,
             "配置选择必须使用 validation，test 只能最终验收一次")
        selected = selection.get("selected_config_id")
        need(bool(selected) and all(
            isinstance(run, dict) and run.get("config_id") == selected for run in acceptance),
            "test 验收只能覆盖由 validation 选中的配置")
        for split, series in (("validation", runs), ("test", acceptance)):
            for run in series:
                if not isinstance(run, dict):
                    problems.append(f"{split} run 结构非法")
                    continue
                need(run.get("kind") == "paired-evaluation" and run.get("training_completed") is True,
                     "每轮需要真实训练后的配对评估")
                need(type(run.get("sample_count")) is int and run["sample_count"] > 0, "评测样本数必须为正整数")
                need(run.get("split_sha256") == data.get("split_sha256"), "基座/训练模型必须共用同一 held-out 划分")
                training = be.linked_json(path.parent, run.get("training_evidence"))
                need(isinstance(training, dict) and training.get("kind") == "training"
                     and training.get("stage") == "completed" and training.get("mode") == "full"
                     and training.get("task_id") == rt.TASK_ID and training.get("recipe_id") == r.id
                     and training.get("seed") == run.get("train_seed")
                     and training.get("config_id") == run.get("config_id")
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
                    if role == "checkpoint":
                        # 训练后评估绑定的是该 config_id / seed 的 effective recipe，
                        # 不能用默认配置的 fingerprint 与 seed 去校验它。
                        try:
                            effective = _config_recipe(
                                r, run.get("config_id"), run.get("train_seed"))
                        except (mt.TaskExecutionError, ValueError, TypeError,
                                KeyError, AttributeError):
                            effective = r
                            problems.append("无法按 config_id / train_seed 重建有效 recipe")
                    else:
                        effective = r
                    problems.extend(evaluation_problems(
                        effective, evaluation, path.parent / record["path"], role=role))
                    need(evaluation.get("metrics") == run.get(side)
                         and evaluation.get("sample_count") == run.get("sample_count"),
                         "套件摘要与关联评估指标或样本数不一致")
                need(be.hashed_file(path.parent, run.get("predictions")), "每轮预测证据文件必须可校验")
                for side in ("base_metrics", "trained_metrics"):
                    scores = run.get(side)
                    need(isinstance(scores, dict) and set(names) <= set(scores)
                         and all(be.finite_number(scores.get(m)) for m in names),
                         "每轮指标都须为有限数值")
        need(isinstance(data.get("costs"), dict)
             and data["costs"].get("upstream_throughput_comparable") is False,
             "缺少顺序执行成本对照或错误宣称与上游吞吐可比")
        try:
            validate_suite_payload(r, data, path)
        except (mt.TaskExecutionError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            problems.append(f"RapidFire 套件内容拒绝：{exc}")
        if problems:
            return problems
        for name in names:
            values = [run["trained_metrics"][name] for run in acceptance]
            need(abs(statistics.mean(values) - expected[name]) <= tolerance[name],
                 f"{name} 的 test 重跑均值超出验收容差")
        return problems
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [f"证据结构非法；失败关闭：{exc}"]
