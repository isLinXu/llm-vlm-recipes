"""RapidFire 多配置 SFT 的受控自建协议 rapidfire-sft-v1（data / train / eval）。

一手事实（HF Cookbook rapidfire_sft_multiconfig_training + Bitext 数据卡）与本实现的差异
必须显式保留，绝不把自建参数冒充上游复现：
  * 上游 train=dataset['train'].select(range(64))，eval=select(range(50,60))；
    eval 完全落在 train 内，ROUGE-L 不构成泛化证据。这里按规范化问题 hash 做
    80/10/10 三段不重叠划分，同题所有生成变体落同一 split，全局拒绝重复内容。
  * 上游 sample_formatting_function 生成 "Question: {instruction}\\nAnswer: {response}"
    整段文本并交给 RFSFTConfig；gold 进入训练输入与损失。这里 prompt 只到 "Answer:"，
    用 transformers.Trainer + 自建 completion-only collator 只对 " {response}"+EOS 计 loss。
  * 上游 sample_compute_metrics 把 eval_preds 的 predictions/labels 直接传给
    evaluate.load('rouge')（未见 decode，异常静默返回 {}），页面从未报告任何数值。
    这里用纯 Python、无 stemmer 的 LCS ROUGE-L 与 exact_match，明确声明数值不可比。
  * 上游用未固定版本的 rapidfireai 服务做 4 配置并发调度（num_chunks=4）。这里顺序
    执行 4 个配置，每个配置由 config_recipe() 只覆盖 [train]/[peft]/execution.config_id，
    独立输出目录与证据；不声称复现并发调度或 16-24× 吞吐声明。

发布级（comparison / export）由 rapidfire_suite.py / rapidfire_export.py 承接，经
task_router 按精确 recipe_id / task_id 分派；为此本模块的 run_evaluation 显式支持
evaluation_split ∈ {validation, test}，并把 split 与阶段写进评估证据内容身份，
使「只用 validation 选配置、test 只验收一次」可被发布门禁重算，而不是一句声明。
本模块只在本机无 GPU 环境下做过离线合成测试；任何 completed 证据都必须来自真实 CUDA 实跑。
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import re
import shlex
import sys
import time
import tomllib
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import math_task as mt
from .model import Recipe

TaskExecutionError = mt.TaskExecutionError
DatasetValidationError = TaskExecutionError
sha256_json = mt.sha256_json
fixed_seeds = mt.fixed_seeds
require_revision = mt.require_revision
check_signature = mt.check_signature
apply_determinism = mt.apply_determinism
execution_identity = mt.execution_identity
mode_output_dir = mt.mode_output_dir

TASK_ID = "rapidfire-sft-v1"
RECIPE_ID = "llm/sft/rapidfire-multiconfig"
MODEL = MODEL_ID = "gpt2"
DATASET = DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
PROTOCOL = "rapidfire-sft-v1"
EVAL_PROTOCOL = "rapidfire-sft-v1/held-out-completion-only-v1"
EVAL_VARIANT = "completion-only-rouge-l-no-stemmer"
RUNTIME_PACKAGES = ("torch", "transformers", "datasets", "peft", "safetensors")

REQUIRED_COLUMNS = frozenset({"flags", "instruction", "category", "intent", "response"})
PROMPT_TEMPLATE = "Question: {instruction}\nAnswer:"
METRICS = ("rouge_l", "exact_match")
PREDICTIONS_FILE = "predictions.json"
EVAL_MANIFEST_FILE = "eval_manifest.json"
TOKENIZER_IDENTITY_FILE = "tokenizer_identity.json"
TRANSFORM_FILE = "rapidfire_transform_manifest.json"
REFERENCE_FILES = ("source_manifest.json", TOKENIZER_IDENTITY_FILE, EVAL_MANIFEST_FILE)
FULL_STEPS = 64
SMOKE_STEPS = 5

RAPIDFIRE_SPEC = {
    "protocol": PROTOCOL,
    "base_experiment_id": "llm-sft-rapidfire-multiconfig-v1",
    "source_split": "train",
    "instruction_column": "instruction",
    "response_column": "response",
    "metadata_columns": ["flags", "category", "intent"],
    "split_percent": [80, 10, 10],
    "split_algorithm": "sha256_json({normalized_instruction,data_seed,protocol}) modulo 100；同题所有变体同 split",
    "duplicate_policy": "全局拒绝规范化(instruction,response)内容重复；问题 hash 决定 split，禁止跨 split 泄漏",
    "normalization": "内容原文保留；问题隔离 hash 用 NFKC + casefold + 空白折叠；不使用 strip 修改训练文本",
    "loss": "completion-only：' {response}'+EOS 参与 causal loss，prompt 部分 label=-100",
    "prompt_format": "Question: {instruction}\nAnswer:",
    "target_format": " {response}{eos}",
    "metric_definition": "rouge_l = 纯Python LCS-F1（NFKC+casefold+正则取词、无 stemmer）；exact_match = 同规则归一化后全串相等；与上游 evaluate.load('rouge', use_stemmer=True) 数值不可比",
    "generation_truncation": "首个 EOS 之后的内容丢弃，再 strip；空生成不得给分",
    "max_eval_examples": 300,
    "max_validation_examples": 300,
    "concurrent_scheduling": False,
    "upstream_quantitative_evaluation": False,
    "selection_policy": "只用 validation 的 3 个独立训练 seed paired rouge_l 均值选配置；test 仅在选中配置上做一次最终验收；平局按 config_id 字典序；test 结果不得回流选择",
}
EVAL_SPLITS = frozenset({"validation", "test"})
EVAL_STAGES = {"validation": "config_selection", "test": "final_acceptance"}

CONFIG_KEYS = frozenset({
    "id", "learning_rate", "lr_scheduler_type", "warmup_steps",
    "lora_r", "lora_alpha", "lora_dropout", "target_modules",
})


def expected_configs():
    """上游原文的 2 lr/scheduler × 2 LoRA 配置交叉；逐字段显式，不推断默认值。"""
    grid = []
    for label, lr, scheduler, warmup in (("lr5e-4", 5e-4, "linear", 0),
                                         ("lr2e-4", 2e-4, "cosine", 10)):
        for rank, alpha, targets in ((8, 16, ["c_attn"]),
                                    (32, 64, ["c_attn", "c_proj"])):
            grid.append({
                "id": f"{label}-r{rank}",
                "learning_rate": lr, "lr_scheduler_type": scheduler,
                "warmup_steps": warmup, "lora_r": rank, "lora_alpha": alpha,
                "lora_dropout": 0.1, "target_modules": list(targets),
            })
    return grid


TRAIN_KEYS = frozenset({
    "seed", "data_seed", "seed_mechanism", "learning_rate", "lr_scheduler_type",
    "warmup_steps", "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "max_steps", "seq_length", "weight_decay",
    "eval_steps", "save_steps", "logging_steps", "fp16", "bf16", "optim",
    "lora_r", "lora_alpha", "lora_dropout", "gradient_checkpointing",
    "dataloader_num_workers", "remove_unused_columns",
})
METADATA_TRAIN_KEYS = frozenset({
    "seed", "data_seed", "seed_mechanism", "seq_length",
    "lora_r", "lora_alpha", "lora_dropout",
})
EVALUATION_KEYS = frozenset({
    "metrics", "seed", "max_new_tokens", "multi_run", "held_out", "split",
    "expected", "tolerance", "export_tolerance",
})


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _hash_manifest(value):
    payload = {k: v for k, v in value.items() if k != "sha256"}
    return {**value, "sha256": sha256_json(payload)}


def _unknown(section, allowed, label):
    if not isinstance(section, dict):
        raise TaskExecutionError(f"[{label}] 必须是配置表")
    extra = sorted(set(section) - set(allowed))
    if extra:
        raise TaskExecutionError(f"[{label}] 含未支持参数 {extra}；拒绝静默忽略")


def resolve_task_id(r, plan=None, task_id=None):
    declared = task_id or (plan or {}).get("task_id") or r.get("execution.task_id")
    return declared or (TASK_ID if r.id == RECIPE_ID else None)


# ---------------------------------------------------------------- 配置网格

def config_grid(r):
    """校验 4 组配置网格；缺项、多字段、改值一律拒绝，不静默套默认。"""
    grid = r.get("configs")
    expected = expected_configs()
    if not isinstance(grid, list) or len(grid) != len(expected):
        raise TaskExecutionError("configs 必须是上游原文的 4 组配置数组")
    for index, (actual, want) in enumerate(zip(grid, expected)):
        if not isinstance(actual, dict) or set(actual) != set(CONFIG_KEYS):
            raise TaskExecutionError(f"configs[{index}] 字段集合不等于 {sorted(CONFIG_KEYS)}")
        for key, value in want.items():
            if actual[key] != value or type(actual[key]) is not type(value):
                raise TaskExecutionError(f"configs[{index}].{key} 必须为 {value!r}")
        if not isinstance(actual["id"], str) or not re.fullmatch(r"[a-z0-9-]+", actual["id"]):
            raise TaskExecutionError(f"configs[{index}].id 非法")
    if len({item["id"] for item in grid}) != len(grid):
        raise TaskExecutionError("configs.id 必须互异，拒绝输出目录冲突")
    return copy.deepcopy(grid)


def active_config(r):
    config_id = r.get("execution.config_id")
    if not isinstance(config_id, str) or not config_id:
        raise TaskExecutionError("execution.config_id 必须显式声明当前配置")
    for config in config_grid(r):
        if config["id"] == config_id:
            return config
    raise TaskExecutionError(f"未知配置 {config_id!r}；拒绝借用其他配置协议")


def config_recipe(r, config_id):
    """从 base recipe 派生单个配置；只覆盖 [train]/[peft]/execution.config_id/experiment_id。

    显式配置 id 而非并发进程竞争同一份配置；输出目录由 experiment_id 自然隔离，
    因此 config_recipe 可重复调用且结果幂等。
    """
    if r.id != RECIPE_ID or not isinstance(config_id, str) or not config_id:
        raise TaskExecutionError("config_recipe 只接受 RapidFire recipe 与显式配置 id")
    grid = config_grid(r)
    config = next((item for item in grid if item["id"] == config_id), None)
    if config is None:
        raise TaskExecutionError(f"未知配置 {config_id!r}；不得静默回退默认配置")
    data = copy.deepcopy(r.data)
    train = data.setdefault("train", {})
    peft = data.setdefault("peft", {})
    for key in ("learning_rate", "lr_scheduler_type", "warmup_steps",
                "lora_r", "lora_alpha", "lora_dropout"):
        train[key] = config[key]
    peft["r"] = config["lora_r"]
    peft["alpha"] = config["lora_alpha"]
    peft["dropout"] = config["lora_dropout"]
    peft["target_modules"] = list(config["target_modules"])
    data.setdefault("execution", {})["config_id"] = config_id
    base = r.get("rapidfire.base_experiment_id")
    if not isinstance(base, str) or not base:
        raise TaskExecutionError("rapidfire.base_experiment_id 必须声明")
    data.setdefault("baseline_contract", {})["experiment_id"] = f"{base}--{config_id}"
    return Recipe(r.id, r.path, data)


def require_supported(r, plan=None, task_id=None) -> str:
    resolved = resolve_task_id(r, plan, task_id)
    if r.id != RECIPE_ID or resolved != TASK_ID or (plan or {}).get("recipe_id", r.id) != r.id:
        raise TaskExecutionError(f"尚未实现该执行路由：{r.id} / {resolved!r}")
    required = {
        "method.family": "hpo", "method.trainer": "Trainer",
        "method.upstream_trainer": "Experiment.run_fit / RFSFTConfig",
        "method.peft": "lora", "model.base": MODEL, "dataset.name": DATASET,
    }
    for key, expected in required.items():
        if r.get(key) != expected:
            raise TaskExecutionError(f"{TASK_ID} 要求 {key}={expected!r}")
        section, field = key.split(".")
        planned = (plan or {}).get(section, {})
        if field in planned and planned[field] != expected:
            raise TaskExecutionError(f"plan {key} 与 RapidFire 路由不一致")
    if r.is_vlm or r.get("rewards") or r.get("model.modality", "llm") != "llm":
        raise TaskExecutionError("RapidFire 只支持纯文本客服 SFT，不使用图像或 reward")
    _unknown(r.get("method"), {"family", "trainer", "upstream_trainer", "peft"}, "method")
    _unknown(r.get("model"), {"base", "params_b", "modality", "revision"}, "model")
    _unknown(r.get("dataset"),
             {"name", "structure", "type", "kind", "has_images", "revision"}, "dataset")
    _unknown(r.get("budget"), {"full_steps", "smoke_steps"}, "budget")
    if r.get("dataset.kind", "static") != "static" or r.get("dataset.has_images") is not False:
        raise TaskExecutionError("RapidFire 只接受冻结的静态纯文本数据集")

    section = r.get("rapidfire")
    _unknown(section, RAPIDFIRE_SPEC, "rapidfire")
    for key, expected in RAPIDFIRE_SPEC.items():
        if key not in section or section[key] != expected or type(section[key]) is not type(expected):
            raise TaskExecutionError(f"rapidfire.{key} 必须为 {expected!r}")
    config = active_config(r)
    if r.get("baseline_contract.experiment_id") != (
            f"{RAPIDFIRE_SPEC['base_experiment_id']}--{config['id']}"):
        raise TaskExecutionError("baseline_contract.experiment_id 必须与当前 config_id 对应")

    peft = r.get("peft")
    _unknown(peft, {"r", "alpha", "dropout", "target_modules", "bias", "task_type"}, "peft")
    for key, expected in {
        "r": config["lora_r"], "alpha": config["lora_alpha"],
        "dropout": config["lora_dropout"], "target_modules": list(config["target_modules"]),
        "bias": "none", "task_type": "CAUSAL_LM",
    }.items():
        if peft.get(key) != expected or type(peft.get(key)) is not type(expected):
            raise TaskExecutionError(f"peft.{key} 必须等于当前配置的 {expected!r}")

    train = r.get("train")
    _unknown(train, TRAIN_KEYS, "train")
    missing = sorted(TRAIN_KEYS - set(train))
    if missing:
        raise TaskExecutionError(f"[train] 缺少参数 {missing}")
    for key in ("learning_rate", "lr_scheduler_type", "warmup_steps",
                "lora_r", "lora_alpha", "lora_dropout"):
        if train[key] != config[key] or type(train[key]) is not type(config[key]):
            raise TaskExecutionError(f"train.{key} 与 config {config['id']!r} 不一致")
    for key, expected in {
        "train.bf16": False, "train.fp16": True, "train.gradient_checkpointing": True,
        "train.remove_unused_columns": False, "train.optim": "adamw_torch",
        "evaluation.metrics": list(METRICS), "evaluation.held_out": True,
        "evaluation.multi_run": True,
    }.items():
        if r.get(key) != expected or (isinstance(expected, bool) and type(r.get(key)) is not bool):
            raise TaskExecutionError(f"{key} 必须为 {expected!r}")
    _unknown(r.get("evaluation"), EVALUATION_KEYS, "evaluation")
    fixed_seeds(r)
    if not r.has("evaluation.seed"):
        raise TaskExecutionError("evaluation.seed 必须显式固定")
    if not isinstance(train["seed_mechanism"], str) or not train["seed_mechanism"].strip():
        raise TaskExecutionError("seed_mechanism 必须保留上游无 set_seed 的元信息")
    for key in ("per_device_train_batch_size", "per_device_eval_batch_size",
                "gradient_accumulation_steps", "eval_steps", "save_steps", "logging_steps",
                "warmup_steps", "max_steps", "seq_length"):
        if type(train[key]) is not int or train[key] < (0 if key == "warmup_steps" else 1):
            raise TaskExecutionError(f"train.{key} 必须为非负/正整数")
    if type(train["dataloader_num_workers"]) is not int or train["dataloader_num_workers"] < 0:
        raise TaskExecutionError("train.dataloader_num_workers 必须为非负整数")
    for key in ("learning_rate", "weight_decay"):
        value = train[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise TaskExecutionError(f"train.{key} 必须为有限非负数")
    if train["learning_rate"] == 0:
        raise TaskExecutionError("learning_rate 必须大于零")
    sequence, new = train["seq_length"], r.get("evaluation.max_new_tokens")
    if type(new) is not int or new < 2 or sequence < new + 6:
        raise TaskExecutionError("seq_length 必须为整数且至少 max_new_tokens+6")
    if (train["max_steps"] != FULL_STEPS or r.get("budget.full_steps") != FULL_STEPS
            or r.get("budget.smoke_steps") != SMOKE_STEPS):
        raise TaskExecutionError("预算必须显式 full=max_steps=64 / smoke=5；不可静默改预算")
    upstream = r.get("upstream")
    for key in ("run_fit", "train_range", "eval_range", "max_steps", "evaluation", "seed_note"):
        if not upstream.get(key):
            raise TaskExecutionError(f"必须保留上游 {key} 元信息，不得把自建参数冒充上游原值")
    dependencies = r.get("dependencies", {})
    _unknown(dependencies, RUNTIME_PACKAGES, "dependencies")
    for name, version in dependencies.items():
        if not isinstance(version, str) or not version.strip() or version in ("unknown", "latest", "*"):
            raise TaskExecutionError(f"dependencies.{name} 不能猜测或使用未冻结版本")
    return resolved


def split_spec(r):
    return {**copy.deepcopy(r.get("rapidfire")), "data_seed": fixed_seeds(r)[1],
            # 划分与当前 LoRA 配置无关；绑定完整网格，4 配置才能共享同一冻结 held-out。
            "config_grid_sha256": sha256_json(config_grid(r)),
            "selection": "有限完整扫描，保留 source_index；无 shuffle、无补救划分",
            "gold_policy": "response 只进入训练 target 与评估 gold，绝不进入生成 prompt"}


def task_plan(r, plan, mode, task_id=None) -> dict:
    result = copy.deepcopy(plan)
    result["mode"] = mode
    try:
        if mode not in ("dry-run", "smoke", "full", "eval", "zero-shot"):
            raise TaskExecutionError(f"未知模式 {mode!r}")
        result["task_id"] = require_supported(r, plan, task_id)
        if plan.get("mode", mode) != mode:
            raise TaskExecutionError("plan.mode 与请求模式不一致")
        result["execution_supported"] = True
        result.pop("unsupported_reason", None)
        result["run"] = {**result.get("run", {}), "output_dir": str(mode_output_dir(plan, mode))}
        if mode in ("dry-run", "smoke", "full"):
            budget_mode = "smoke" if mode == "dry-run" else mode
            config = config_kwargs(r, {**result, "mode": budget_mode, "run": {
                **result["run"], "steps": r.get(f"budget.{budget_mode}_steps")}},
                Path(result["run"]["output_dir"]))
            result["run"].update(steps=config["max_steps"], params=config)
        result["dataset"] = {**result.get("dataset", {}),
                             "split": "question-hash 80/10/10 + 全局内容去重",
                             "split_plan": split_spec(r)}
        split_name = result.get("evaluation_split", "test")
        if split_name not in EVAL_SPLITS:
            raise TaskExecutionError(f"plan.evaluation_split 只允许 {sorted(EVAL_SPLITS)}")
        result["evaluation"] = {
            "metrics": list(METRICS), "seed": r.get("evaluation.seed"),
            "max_new_tokens": r.get("evaluation.max_new_tokens"), "held_out": True,
            "multi_run": True, "runs": 1, "protocol": EVAL_PROTOCOL, "do_sample": False,
            "evaluation_split": split_name, "eval_stage": EVAL_STAGES[split_name],
            "selection_policy": r.get("rapidfire.selection_policy"),
            "training_seed_variance_measured": False,
            "upstream_quantitative_evaluation": False,
            "metric_comparable_to_upstream_rougeL": False,
            "concurrent_scheduling": False,
        }
        # 计划显示的命令必须与生效配置一致；base recipe 的 execution.command 只写默认配置。
        command = shlex.split(result.get("command", ""))
        if command:
            if "--config-id" in command:
                index = command.index("--config-id")
                if index + 1 >= len(command):
                    raise TaskExecutionError("execution.command 的 --config-id 缺少取值")
                command[index + 1] = str(r.get("execution.config_id"))
            else:
                command += ["--config-id", str(r.get("execution.config_id"))]
            result["command"] = shlex.join(command)
        result["config_grid"] = {
            "active": r.get("execution.config_id"),
            "ids": [item["id"] for item in config_grid(r)],
            "execution": "顺序执行；每配置独立 experiment_id / 输出目录 / 证据",
        }
        result["migration"] = {
            "trainer": "Trainer + completion-only collator（上游 RFSFTConfig 整段文本损失）",
            "scheduling": "顺序 4 配置（上游 rapidfireai num_chunks=4 并发调度，不可重放）",
            "split": "question hash 80/10/10（上游 eval[50:60] ⊂ train[:64]）",
            "metrics": "纯 Python 无 stemmer ROUGE-L + exact_match（上游未报告任何数值）",
            "padding": "训练右 padding（上游设 left padding 便于生成）",
            "license": "数据卡未给 license，仅 (c) Bitext Innovations, 2024；商业使用未确认",
        }
    except (TaskExecutionError, KeyError, TypeError, ValueError) as exc:
        result["execution_supported"] = False
        result["unsupported_reason"] = str(exc)
    return result


def config_kwargs(r, plan, output_dir) -> dict:
    require_supported(r, plan)
    mode = plan.get("mode")
    if mode not in ("smoke", "full"):
        raise TaskExecutionError("TrainingArguments 只用于显式 smoke/full")
    steps = r.get(f"budget.{mode}_steps")
    if plan.get("run", {}).get("steps", steps) != steps:
        raise TaskExecutionError("plan 有效步数与预算不一致")
    params = {k: v for k, v in r.get("train").items() if k not in METADATA_TRAIN_KEYS}
    params.update(max_steps=steps, output_dir=str(output_dir), report_to="none", push_to_hub=False,
                  evaluation_strategy="steps", save_strategy="steps", logging_strategy="steps")
    return params


def training_arguments_kwargs(target, canonical):
    try:
        keys = inspect.signature(target).parameters
    except (TypeError, ValueError) as exc:
        raise TaskExecutionError("无法核验 TrainingArguments API") from exc
    result = dict(canonical)
    if "eval_strategy" in keys and "evaluation_strategy" not in keys:
        result["eval_strategy"] = result.pop("evaluation_strategy")
    elif "evaluation_strategy" not in keys:
        raise TaskExecutionError("TrainingArguments 没有显式的 eval_strategy / evaluation_strategy")
    check_signature(target, result, "TrainingArguments")
    return result


# ---------------------------------------------------------------- 数据协议

def normalized_question(text):
    value = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(value.split())


def parse_row(raw, index):
    """严格校验一行 Bitext 数据；未知列、缺列、空 gold 一律拒绝。"""
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"source_index={index} 不是数据行对象")
    if set(raw) != set(REQUIRED_COLUMNS):
        raise DatasetValidationError(
            f"source_index={index} 列集合必须为 {sorted(REQUIRED_COLUMNS)}，实际 {sorted(raw)}")
    for column in sorted(REQUIRED_COLUMNS):
        value = raw[column]
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(f"source_index={index} 列 {column} 不是非空字符串")
    question = normalized_question(raw["instruction"])
    if not question:
        raise DatasetValidationError(f"source_index={index} 规范化问题为空")
    content = {column: raw[column] for column in sorted(REQUIRED_COLUMNS)}
    content_sha256 = sha256_json(content)
    return {
        "source_index": index,
        "sample_id": f"{PROTOCOL}:{index}:{content_sha256[:16]}",
        "flags": raw["flags"], "instruction": raw["instruction"],
        "category": raw["category"], "intent": raw["intent"],
        "response": raw["response"],
        "normalized_question": question,
        "question_sha256": sha256_json({"question": question}),
        "content_sha256": content_sha256,
    }


def split_bucket(question_sha256, data_seed):
    digest = hashlib.sha256(f"{PROTOCOL}|{data_seed}|{question_sha256}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def prepare_splits(rows, r):
    """有限完整扫描 + 内容级去重 + 问题 hash 80/10/10；空划分拒绝，不随机补救。"""
    require_supported(r)
    _, data_seed = fixed_seeds(r)
    splits = {"train": [], "validation": [], "test": []}
    seen, questions, duplicates = {}, {}, []
    for index, raw in enumerate(rows):
        row = parse_row(raw, index)
        if row["content_sha256"] in seen:
            duplicates.append({"first": seen[row["content_sha256"]], "again": index,
                               "sha256": row["content_sha256"]})
            continue
        seen[row["content_sha256"]] = index
        name = split_bucket(row["question_sha256"], data_seed)
        owner = questions.setdefault(row["question_sha256"], name)
        if owner != name:
            raise DatasetValidationError("同一规范化问题跨 split；拒绝泄漏")
        splits[name].append(row)
    if duplicates:
        raise DatasetValidationError(
            f"检测到 {len(duplicates)} 条规范化内容重复（含跨 split）；拒绝静默去重，先复核数据源与 revision")
    for name, items in splits.items():
        if not items:
            raise DatasetValidationError(f"{name} split 为空；拒绝用其他 split 偷偷补救")
    max_eval = r.get("rapidfire.max_eval_examples")
    max_validation = r.get("rapidfire.max_validation_examples")
    for value in (max_eval, max_validation):
        if type(value) is not int or value < 1:
            raise DatasetValidationError("rapidfire 评估/验证上限必须为正整数")
    eval_rows = splits["test"][:max_eval]
    validation_rows = splits["validation"][:max_validation]
    manifest = {
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "dataset_id": DATASET, "data_seed": data_seed,         "source_split": "train",
        # 内容划分只由规范化问题 hash + data_seed + protocol 决定，与当前训练配置无关；
        # 绑定整个 4 配置网格，使所有配置能共享同一冻结 held-out，而改网格仍会失败关闭。
        "config_grid_sha256": sha256_json(config_grid(r)),
        "split_algorithm": r.get("rapidfire.split_algorithm"),
        "split_percent": [80, 10, 10],
        "duplicate_policy": r.get("rapidfire.duplicate_policy"),
        "normalization": r.get("rapidfire.normalization"),
        "duplicates_rejected": duplicates,
        "counts": {"source": len(seen), "train": len(splits["train"]),
                   "validation": len(splits["validation"]), "test": len(splits["test"]),
                   "validation_used": len(validation_rows), "test_eval_used": len(eval_rows)},
        "split_hashes": {
            name: [row["content_sha256"] for row in items]
            for name, items in splits.items()},
        "split_question_hashes": {
            name: sorted({row["question_sha256"] for row in items})
            for name, items in splits.items()},
        "eval_sample_indices": [row["source_index"] for row in eval_rows],
        "gold_policy": "response 只进入训练 target 与评估 gold，绝不进入生成 prompt",
    }
    manifest["split_hashes_sha256"] = sha256_json(manifest["split_hashes"])
    manifest["sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    return {"train": splits["train"], "validation": validation_rows, "test": eval_rows}, manifest


def validate_source_manifest(manifest, r):
    if not isinstance(manifest, dict):
        raise DatasetValidationError("source_manifest 缺失")
    computed = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    if (manifest.get("sha256") != computed or manifest.get("schema_version") != 1
            or manifest.get("protocol") != PROTOCOL or manifest.get("task_id") != TASK_ID
            or manifest.get("dataset_id") != DATASET
            or manifest.get("data_seed") != r.get("train.data_seed")
            or manifest.get("config_grid_sha256") != sha256_json(config_grid(r))
            or manifest.get("split_hashes_sha256") != sha256_json(manifest.get("split_hashes"))):
        raise DatasetValidationError("source_manifest 内容身份 / 划分 hash 不一致")
    hashes = manifest.get("split_hashes")
    question_hashes = manifest.get("split_question_hashes")
    counts = manifest.get("counts")
    if (not isinstance(hashes, dict) or set(hashes) != {"train", "validation", "test"}
            or not isinstance(question_hashes, dict) or set(question_hashes) != set(hashes)
            or not isinstance(counts, dict)):
        raise DatasetValidationError("source_manifest 缺少 split_hashes / question_hashes / counts")
    for name in ("train", "validation", "test"):
        values, qvalues = hashes[name], question_hashes[name]
        if (not isinstance(values, list) or not values or counts.get(name) != len(values)
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in values)
                or not isinstance(qvalues, list) or not qvalues
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in qvalues)):
            raise DatasetValidationError(f"source_manifest.{name} 哈希列表非法")
        for q in qvalues:
            if split_bucket(q, manifest["data_seed"]) != name:
                raise DatasetValidationError(f"source_manifest.{name} 问题 hash 与确定性划分规则不一致")
    all_hashes = hashes["train"] + hashes["validation"] + hashes["test"]
    if len(set(all_hashes)) != len(all_hashes):
        raise DatasetValidationError("source_manifest 存在跨 split 重复内容 hash")
    all_questions = [q for values in question_hashes.values() for q in values]
    if len(set(all_questions)) != len(all_questions):
        raise DatasetValidationError("source_manifest 存在跨 split 重复问题 hash")
    return manifest


def load_splits(r, dependencies):
    revision = require_revision(r, "dataset.revision")
    dataset = dependencies["datasets"].load_dataset(DATASET, revision=revision, split="train")
    splits, manifest = prepare_splits(dataset, r)
    manifest["dataset_revision"] = revision
    manifest.pop("sha256")
    manifest["sha256"] = sha256_json(manifest)
    return splits, manifest


# ---------------------------------------------------------------- 编码与评估

def token_identity(tokenizer):
    special = ("Question:", "Answer:", "\n")
    payload = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "control_tokens": {tag: tokenizer(tag, add_special_tokens=False)["input_ids"]
                           for tag in special},
    }
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


def render_prompt(row):
    prompt = PROMPT_TEMPLATE.format(instruction=row["instruction"])
    if row["response"] in prompt:
        raise DatasetValidationError("gold 泄漏进生成 prompt；拒绝执行")
    return prompt


def render_target(row, eos_token):
    return " " + row["response"] + eos_token


def encode_example(row, tokenizer, identity, r):
    prompt = render_prompt(row)
    target = render_target(row, identity["eos_token"])
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(target, add_special_tokens=False)["input_ids"])
    if not prompt_ids or not target_ids:
        raise DatasetValidationError("prompt / target token 化后为空")
    limit = r.get("train.seq_length")
    if len(prompt_ids) + len(target_ids) > limit:
        raise DatasetValidationError(
            f"source_index={row['source_index']} 长度 {len(prompt_ids)+len(target_ids)} 超过 seq_length={limit}；"
            "拒绝截断（截断会切掉 gold）")
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "prompt_tokens": len(prompt_ids), "target_tokens": len(target_ids),
        "sample_id": row["sample_id"], "gold_sha256": sha256_json(row["response"]),
    }


def collate(items, torch, pad_token_id):
    keys = ("input_ids", "labels", "attention_mask")
    width = max(len(item["input_ids"]) for item in items)
    batch = {
        "input_ids": [item["input_ids"] + [pad_token_id] * (width - len(item["input_ids"])) for item in items],
        "labels": [item["labels"] + [-100] * (width - len(item["labels"])) for item in items],
        "attention_mask": [item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in items],
    }
    if any(len(v) != width for row in batch.values() for v in row):
        raise DatasetValidationError("collate 长度不一致")
    return {key: torch.tensor(batch[key], dtype=torch.long) for key in keys}


def train_examples(splits, tokenizer, r):
    require_supported(r)
    identity = token_identity(tokenizer)
    examples, entries = {}, {}
    for name in ("train", "validation"):
        examples[name], entries[name] = [], []
        for row in splits[name]:
            item = encode_example(row, tokenizer, identity, r)
            examples[name].append({key: item[key] for key in ("input_ids", "labels", "attention_mask")})
            entries[name].append({
                "sample_id": row["sample_id"], "source_index": row["source_index"],
                "content_sha256": row["content_sha256"], "gold_sha256": item["gold_sha256"],
                "prompt_tokens": item["prompt_tokens"], "target_tokens": item["target_tokens"],
                "tokens_sha256": sha256_json(item["input_ids"]),
                "labels_sha256": sha256_json(item["labels"]),
                "example_sha256": sha256_json(examples[name][-1]),
            })
        if not examples[name]:
            raise DatasetValidationError(f"{name} 编码后为空")
    transform = _hash_manifest({
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "config_id": r.get("execution.config_id"), "tokenizer_identity": identity,
        "seq_length": r.get("train.seq_length"), "data_seed": r.get("train.data_seed"),
        "loss": r.get("rapidfire.loss"), "target_format": r.get("rapidfire.target_format"),
        "completion_only_loss": True, "gold_not_in_prompt": True,
        "boundary_policy": "分别编码 prompt/target 后拼接，labels 边界按 prompt token 数；右 padding",
        "splits": {name: {"samples": items, "sha256": sha256_json(items),
                          "examples_sha256": sha256_json(examples[name])}
                   for name, items in entries.items()},
    })
    return examples, transform


def validate_transform_manifest(transform, source, r, identity):
    if (not isinstance(transform, dict) or transform.get("sha256") != _hash_manifest(transform)["sha256"]
            or transform.get("protocol") != PROTOCOL or transform.get("task_id") != TASK_ID
            or transform.get("config_id") != r.get("execution.config_id")
            or transform.get("data_seed") != r.get("train.data_seed")
            or transform.get("tokenizer_identity") != identity
            or transform.get("completion_only_loss") is not True
            or transform.get("gold_not_in_prompt") is not True
            or transform.get("loss") != r.get("rapidfire.loss")):
        raise DatasetValidationError("transform manifest 身份 / completion-only 协议不一致")
    for name in ("train", "validation"):
        spec = transform.get("splits", {}).get(name)
        if (not isinstance(spec, dict) or spec.get("sha256") != sha256_json(spec.get("samples"))
                or spec.get("examples_sha256") is None or not spec.get("samples")):
            raise DatasetValidationError(f"transform manifest.{name} 内容不一致")
        for row in spec["samples"]:
            labels = row.get("labels_sha256")
            if (not isinstance(labels, str) or not re.fullmatch(r"[0-9a-f]{64}", labels)
                    or type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] < 1
                    or type(row.get("target_tokens")) is not int or row["target_tokens"] < 1
                    or row["prompt_tokens"] + row["target_tokens"] > r.get("train.seq_length")):
                raise DatasetValidationError(f"transform manifest.{name} 样本 token 边界非法")
    return transform


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _tokens(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold(), flags=re.UNICODE)


def _lcs_length(a, b):
    if len(a) * len(b) > 4_000_000:
        raise DatasetValidationError("ROUGE-L 输入过长；拒绝不可控的重算")
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for index, right in enumerate(b, start=1):
            current.append(previous[index - 1] + 1 if left == right
                           else max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction, reference):
    """纯 Python LCS-F1；无 stemmer、无第三方依赖，数值与上游 rougeL 不可比。"""
    a, b = _tokens(prediction), _tokens(reference)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = _lcs_length(a, b)
    return 2.0 * overlap / (len(a) + len(b))


def clean_generation(text, eos_token):
    if not isinstance(text, str):
        raise DatasetValidationError("生成文本不是字符串")
    head = text.split(eos_token, 1)[0] if eos_token else text
    return head.strip()


def score_prediction(row, text, eos_token):
    prediction = clean_generation(text, eos_token)
    gold = row["response"]
    result = {
        "sample_id": row["sample_id"], "source_index": row["source_index"],
        "content_sha256": row["content_sha256"],
        "question_sha256": row["question_sha256"],
        "gold_sha256": sha256_json(gold),
        "rouge_l": 0.0, "exact_match": 0.0,
        "prediction_sha256": sha256_json(text),
        "clean_prediction_sha256": sha256_json(prediction),
    }
    if not prediction:
        return result
    result["rouge_l"] = rouge_l(prediction, gold)
    result["exact_match"] = float(normalize_text(prediction) == normalize_text(gold))
    return result


def recompute_metrics(predictions):
    if not isinstance(predictions, list) or not predictions:
        raise DatasetValidationError("逐题预测为空，不能计算指标")
    totals = {name: 0.0 for name in METRICS}
    for row in predictions:
        if not isinstance(row, dict) or any(
                type(row.get(name)) not in (int, float) or not 0 <= row[name] <= 1 for name in METRICS):
            raise DatasetValidationError("逐题预测含非法分数字段")
        for name in METRICS:
            totals[name] += float(row[name])
    return {name: totals[name] / len(predictions) for name in METRICS}


def evaluation_examples(rows, tokenizer, identity, r):
    require_supported(r)
    samples = []
    for row in rows:
        prompt = render_prompt(row)
        prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if not prompt_ids or not all(type(token) is int and token >= 0 for token in prompt_ids):
            raise DatasetValidationError("评估 prompt token 化结果为空或非法")
        samples.append({
            "sample_id": row["sample_id"], "source_index": row["source_index"],
            "content_sha256": row["content_sha256"],
            "question_sha256": row["question_sha256"],
            "raw": {column: row[column] for column in sorted(REQUIRED_COLUMNS)},
            "gold_sha256": sha256_json(row["response"]),
            "prompt": prompt, "prompt_sha256": sha256_json(prompt),
            "prompt_ids": prompt_ids, "input_sha256": sha256_json(prompt_ids),
        })
    return samples


def evaluation_manifest(samples, identity, r, source, split_name="test"):
    """split_name 进入内容身份：validation 与 test 的评估清单不可互相冒充。"""
    if split_name not in EVAL_SPLITS:
        raise DatasetValidationError(f"未知评估 split {split_name!r}")
    return _hash_manifest({
        "schema_version": 1, "kind": "rapidfire-eval-manifest", "task_id": TASK_ID,
        "protocol": EVAL_PROTOCOL, "variant": EVAL_VARIANT,
        "evaluation_split": split_name, "eval_stage": EVAL_STAGES[split_name],
        "selection_policy": r.get("rapidfire.selection_policy"),
        # 评估输入与当前 LoRA 配置无关；绑定完整网格，base 与任意配置可共享同一输入清单。
        "config_grid_sha256": sha256_json(config_grid(r)),
        "data_seed": r.get("train.data_seed"),
        "max_new_tokens": r.get("evaluation.max_new_tokens"),
        "tokenizer_identity": identity,
        "source_manifest_sha256": source["sha256"],
        "gold_policy": r.get("rapidfire.duplicate_policy"),
        "metric_definition": r.get("rapidfire.metric_definition"),
        "generation_truncation": r.get("rapidfire.generation_truncation"),
        "upstream_quantitative_evaluation": False,
        "metric_comparable_to_upstream_rougeL": False,
        "sample_count": len(samples), "samples": samples,
        "samples_sha256": sha256_json(samples),
    })


# ---------------------------------------------------------------- 执行层（真实 CUDA 实跑才调用）

PORTABLE_ARGUMENT_KEYS = frozenset({"output_dir"})


def training_arguments_identity(arguments):
    if not isinstance(arguments, dict):
        return None
    return {k: v for k, v in arguments.items() if k not in PORTABLE_ARGUMENT_KEYS}


def artifact_identity(artifact):
    if not isinstance(artifact, dict):
        return None
    return {"files": artifact.get("files"), "sha256": artifact.get("sha256")}


def same_artifact(recorded, actual):
    return (isinstance(recorded, dict)
            and artifact_identity(recorded) == artifact_identity(actual)
            and isinstance(recorded.get("path"), str) and bool(recorded["path"]))


def _no_symlinks(path):
    path = Path(path).expanduser()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise TaskExecutionError(f"artifact/lock 路径不能为 symlink：{path}")
    if path.is_dir() and any(p.is_symlink() for p in path.rglob("*")):
        raise TaskExecutionError(f"artifact 包含 symlink：{path}")
    return path


def save_json(path, payload):
    return mt.save_json(_no_symlinks(path), payload)


def artifact_manifest(directory):
    return mt.artifact_manifest(_no_symlinks(directory))


def file_record(path, base):
    resolved = _no_symlinks(Path(path)).resolve()
    root = _no_symlinks(Path(base)).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise TaskExecutionError("证据文件必须位于输出目录内且为普通文件")
    data = resolved.read_bytes()
    return {"path": resolved.relative_to(root).as_posix(),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def base_artifact(r):
    return {"id": MODEL, "revision": r.get("model.revision"), "role": "base"}


def _read_json(path):
    try:
        value = json.loads(_no_symlinks(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("checkpoint JSON 必须为对象")
        return value
    except (OSError, ValueError) as exc:
        raise TaskExecutionError(f"无法读取 {path}：{exc}") from exc


def _lock(r):
    try:
        path = _no_symlinks(r.path.parent / r.get("execution.lockfile"))
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise TaskExecutionError("无法读取 frozen lock") from exc
    for key, value in {
        "recipe_id": r.id, "model_id": MODEL, "dataset_id": DATASET,
        "model_revision": require_revision(r, "model.revision"),
        "dataset_revision": require_revision(r, "dataset.revision"),
        "source_sha": require_revision(r, "source_sha"),
    }.items():
        if lock.get(key) != value:
            raise TaskExecutionError(f"lock {key} 与 recipe 不一致")
    if type(lock.get("schema_version")) is not int or lock["schema_version"] != 1 or lock.get("status") != "frozen":
        raise TaskExecutionError("执行只接受真实 frozen lock；模板不能用于实跑")
    names = {re.split(r"(?:==| @ )", line, maxsplit=1)[0].lower().replace("_", "-")
             for line in lock.get("requirements", []) if isinstance(line, str)}
    if not set(RUNTIME_PACKAGES) <= names:
        raise TaskExecutionError("frozen lock 必须包含 torch/transformers/datasets/peft/safetensors")
    return lock


def load_runtime(*, training=False):
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise TaskExecutionError("缺少 torch；按 frozen lock 安装，--dry-run 不需要依赖") from exc
    if not torch.cuda.is_available():
        raise TaskExecutionError("没有可用 CUDA GPU；拒绝加载或下载模型")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise TaskExecutionError("仅支持 CUDA 单进程单 GPU")
    if not torch.version.cuda:
        raise TaskExecutionError("CUDA runtime 不可用")
    runtime = {"torch": torch}
    for name in RUNTIME_PACKAGES[1:]:
        try:
            runtime[name] = importlib.import_module(name)
        except ImportError as exc:
            raise TaskExecutionError(f"缺少 {name}；必须按真实 frozen lock 安装") from exc
    return runtime


def require_frozen_runtime(r, dependencies):
    _lock(r)
    mt.require_frozen_runtime(r, dependencies)


def runtime_evidence(dependencies, started):
    torch = dependencies["torch"]
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: getattr(dependencies[name], "__version__", None) for name in RUNTIME_PACKAGES},
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "world_size": 1,
        "duration_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "attention_implementation": "eager", "compute_dtype": "float16",
        "quantization": "none",
    }


def _output(plan, mode):
    output = _no_symlinks(mode_output_dir(plan, mode)).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise TaskExecutionError(f"输出目录非空，禁止覆盖：{output}")
    return output


def _preflight(r, plan, training=False):
    require_supported(r, plan)
    require_revision(r, "model.revision")
    require_revision(r, "dataset.revision")
    require_revision(r, "source_sha")
    fixed_seeds(r)
    if training and (r.get("decision.gate") == "blocked" or r.get("baseline.result") == "passes"
                     or plan.get("baseline", {}).get("gate") == "blocked"):
        raise TaskExecutionError("训练 decision gate 已阻断，禁止绕过")
    if plan.get("requested_runs", 1) != 1:
        raise TaskExecutionError("独立训练多 seed 由主线程编排；重复 greedy 不冒充 multi-run")
    return _lock(r)


def _load_base(r, runtime, lock, *, training=False):
    transformers = runtime["transformers"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL, revision=lock["model_revision"], trust_remote_code=False)
    if tokenizer.eos_token is None:
        raise TaskExecutionError("gpt2 tokenizer 缺少 eos_token；拒绝静默替换配置")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, revision=lock["model_revision"], trust_remote_code=False,
        torch_dtype=runtime["torch"].float16, attn_implementation="eager",
        device_map={"": 0})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not training
    return tokenizer, token_identity(tokenizer), model


def _peft_kwargs(r, revision):
    return {"r": r.get("peft.r"), "lora_alpha": r.get("peft.alpha"),
            "lora_dropout": r.get("peft.dropout"),
            "target_modules": list(r.get("peft.target_modules")),
            "bias": r.get("peft.bias"), "task_type": r.get("peft.task_type"),
            "base_model_name_or_path": MODEL, "revision": revision}


def _attach_lora(model, runtime, kwargs):
    peft = runtime["peft"]
    check_signature(peft.LoraConfig, kwargs, "LoraConfig")
    check_signature(peft.get_peft_model, {"model": None, "peft_config": None}, "get_peft_model")
    matches = {target: [name for name, _ in model.named_modules()
                        if name == target or name.endswith("." + target)]
               for target in kwargs["target_modules"]}
    if any(not value for value in matches.values()):
        raise TaskExecutionError(f"LoRA target 没有全部匹配：{matches}")
    config = peft.LoraConfig(**kwargs)
    model = peft.get_peft_model(model=model, peft_config=config)
    trainable, frozen = 0, True
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise TaskExecutionError(f"base 参数未冻结：{name}")
            trainable += parameter.numel()
        elif "lora_" in name:
            frozen = False
    if trainable <= 0 or not frozen:
        raise TaskExecutionError("LoRA trainable 参数为空或状态异常")
    return model, {"matched_target_modules": matches, "trainable_parameters": trainable,
                   "base_parameters_frozen": True}


def verify_adapter_tensors(model, path, runtime):
    peft, torch = runtime["peft"], runtime["torch"]
    target = getattr(peft, "get_peft_model_state_dict", None)
    if target is None:
        raise TaskExecutionError("PEFT 缺少明确的 adapter state 提取接口")
    try:
        load_file = importlib.import_module("safetensors.torch").load_file
        saved = load_file(str(_no_symlinks(path / "adapter_model.safetensors")), device="cpu")
    except (ImportError, OSError, ValueError, RuntimeError, AttributeError) as exc:
        raise TaskExecutionError(f"无法安全读取 adapter tensors：{exc}") from exc
    check_signature(target, {"model": model, "adapter_name": "default",
                             "save_embedding_layers": False}, "get_peft_model_state_dict")
    actual = target(model=model, adapter_name="default", save_embedding_layers=False)
    if not isinstance(actual, dict) or not actual or set(actual) != set(saved):
        raise TaskExecutionError("adapter tensor keys 不完整或含多余键；拒绝非严格加载")
    for key in actual:
        if not re.fullmatch(r".+\.lora_[AB]\.weight", key):
            raise TaskExecutionError(f"adapter 含非 LoRA tensor：{key}")
        a, b = actual[key].detach().cpu(), saved[key].detach().cpu()
        if (tuple(a.shape) != tuple(b.shape) or not bool(torch.isfinite(a).all())
                or not bool(torch.isfinite(b).all()) or not torch.equal(a, b.to(dtype=a.dtype))):
            raise TaskExecutionError(f"adapter tensor shape/值/有限性不一致：{key}")
    return {"verified_keys": sorted(actual), "strict_values_verified": True}


def _adapter_structure(path, r):
    _no_symlinks(path)
    adapter = _read_json(path / "adapter_config.json")
    if not (path / "tokenizer_config.json").is_file() or not (
            path / "adapter_model.safetensors").is_file():
        raise TaskExecutionError("checkpoint 缺少 adapter weights/tokenizer config；无 base fallback")
    for key, value in {
        "base_model_name_or_path": MODEL, "revision": r.get("model.revision"),
        "r": r.get("peft.r"), "lora_alpha": r.get("peft.alpha"),
        "lora_dropout": r.get("peft.dropout"), "bias": r.get("peft.bias"),
        "task_type": r.get("peft.task_type"),
    }.items():
        if adapter.get(key) != value:
            raise TaskExecutionError(f"adapter_config {key} 与当前配置不一致")
    if set(adapter.get("target_modules", [])) != set(r.get("peft.target_modules")):
        raise TaskExecutionError("adapter_config target_modules 与当前配置不一致")


def _completed(r, plan, identity, kind, config, manifest, runtime):
    from .baseline_evidence import fingerprint
    if execution_identity(r) != identity:
        raise TaskExecutionError("执行期间 code/lock identity 漂移；禁止 completed")
    seed, data_seed = fixed_seeds(r)
    return {
        **identity, "schema_version": 1, "kind": kind, "stage": "completed", "task_id": TASK_ID,
        "recipe_id": r.id, "fingerprint": fingerprint(r), "mode": plan["mode"],
        "experiment_id": plan.get("experiment_id", ""),
        "config_id": r.get("execution.config_id"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "data_seed": data_seed, "model_id": MODEL,
        "model_revision": r.get("model.revision"), "dataset_id": DATASET,
        "dataset_revision": r.get("dataset.revision"), "source_sha": r.get("source_sha"),
        "recipe_sha256": sha256_json(r.data), "config": config, "config_sha256": sha256_json(config),
        "split_manifest": manifest, "source_manifest": manifest, "runtime": runtime,
        "baseline_promotion": "none", "training_seed_variance_measured": False,
        "concurrent_scheduling": False, "upstream_quantitative_evaluation": False,
    }


def run_training(r, plan) -> Path:
    lock = _preflight(r, plan, training=True)
    if plan.get("mode") not in ("full", "smoke"):
        raise TaskExecutionError("真实训练只能显式请求 full/smoke")
    output = _output(plan, plan["mode"])
    canonical = config_kwargs(r, plan, output)
    identity = execution_identity(r)
    if plan.get("execution_identity", identity) != identity:
        raise TaskExecutionError("plan code/lock identity 不一致")
    runtime = load_runtime(training=True)
    require_frozen_runtime(r, runtime)
    started = time.perf_counter()
    apply_determinism(r, runtime)
    transformers, torch = runtime["transformers"], runtime["torch"]
    kwargs = training_arguments_kwargs(transformers.TrainingArguments, canonical)
    check_signature(transformers.Trainer,
                    {"model": None, "args": None, "train_dataset": None,
                     "eval_dataset": None, "data_collator": None}, "Trainer")
    args = transformers.TrainingArguments(**kwargs)
    for key, value in kwargs.items():
        actual = getattr(args, key, None)
        if key == "report_to":
            if actual not in ("none", [], ["none"]):
                raise TaskExecutionError("TrainingArguments 必须禁用外部 report")
        elif actual != value:
            raise TaskExecutionError(f"TrainingArguments 未应用 {key}")
    splits, source_manifest = load_splits(r, runtime)
    tokenizer, tokenizer_identity, model = _load_base(r, runtime, lock, training=True)
    examples, transform = train_examples(splits, tokenizer, r)
    validate_transform_manifest(transform, source_manifest, r, tokenizer_identity)
    model, lora_evidence = _attach_lora(model, runtime, _peft_kwargs(r, lock["model_revision"]))
    trainer = transformers.Trainer(
        model=model, args=args,
        train_dataset=runtime["datasets"].Dataset.from_list(examples["train"]),
        eval_dataset=runtime["datasets"].Dataset.from_list(examples["validation"]),
        data_collator=lambda items: collate(items, torch, tokenizer_identity["pad_token_id"]))
    result = trainer.train()
    actual_steps = trainer.state.global_step
    if type(actual_steps) is not int or actual_steps != canonical["max_steps"]:
        raise TaskExecutionError(f"训练未完成预算：实际 {actual_steps}，要求 {canonical['max_steps']}")
    if execution_identity(r) != identity:
        raise TaskExecutionError("保存前 code/lock identity 漂移")
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    _adapter_structure(output, r)
    lora_evidence["tensor_reload_check"] = verify_adapter_tensors(model, output, runtime)
    effective_config = {
        "task_id": TASK_ID, "config_id": r.get("execution.config_id"),
        "config": active_config(r), "training_arguments": canonical,
        "api_training_arguments": kwargs, "peft": r.get("peft"),
        "quantization": "none", "attention_implementation": "eager",
        "rapidfire": r.get("rapidfire"), "upstream_metadata": dict(r.get("upstream")),
        "lora_execution": lora_evidence, "concurrent_scheduling": False,
    }
    save_json(output / "effective_recipe.json",
              {"id": r.id, "original_path": str(r.path.absolute()), "data": r.data})
    save_json(output / "effective_config.json", effective_config)
    save_json(output / "tokenizer_identity.json", tokenizer_identity)
    save_json(output / "source_manifest.json", source_manifest)
    save_json(output / TRANSFORM_FILE, transform)
    torch.cuda.synchronize()
    evidence = _completed(r, plan, identity, "training", effective_config, source_manifest,
                          runtime_evidence(runtime, started))
    evidence.update(steps=actual_steps, max_steps=canonical["max_steps"], trainer="Trainer",
                    tokenizer_identity=tokenizer_identity, transformation_manifest=transform,
                    train_metrics=dict(result.metrics), model_artifact=artifact_manifest(output))
    if execution_identity(r) != identity:
        raise TaskExecutionError("完成时 code/lock identity 漂移")
    return save_json(output / "training.json", evidence)


def checkpoint_path(plan, checkpoint=None, recipe=None) -> Path:
    path = _no_symlinks(Path(checkpoint).expanduser() if checkpoint is not None
                        else mode_output_dir(plan, "full")).resolve()
    if not path.is_dir():
        raise TaskExecutionError("没有训练 checkpoint；无 base fallback")
    evidence = _read_json(path / "training.json")
    saved = _read_json(path / "effective_recipe.json")
    if recipe is None:
        if not isinstance(saved, dict) or not isinstance(saved.get("data"), dict) or not saved.get("original_path"):
            raise TaskExecutionError("checkpoint effective recipe 缺失")
        recipe = Recipe(saved.get("id"), Path(saved["original_path"]), saved["data"])
    r = recipe
    require_supported(r, plan)
    _lock(r)
    _adapter_structure(path, r)
    from .baseline_evidence import fingerprint
    identity = execution_identity(r)
    if (evidence.get("schema_version") != 1 or evidence.get("kind") != "training"
            or evidence.get("stage") != "completed" or evidence.get("task_id") != TASK_ID
            or evidence.get("recipe_id") != RECIPE_ID or evidence.get("mode") not in ("full", "smoke")
            or evidence.get("config_id") != r.get("execution.config_id")
            or (checkpoint is None and evidence.get("mode") != "full")
            or any(evidence.get(k) is True for k in ("mock", "simulated", "dry_run"))):
        raise TaskExecutionError("checkpoint 不是已完成 RapidFire full/smoke 训练")
    if (evidence.get("fingerprint") != fingerprint(r) or evidence.get("recipe_sha256") != sha256_json(r.data)
            or saved.get("id") != r.id or saved.get("data") != r.data
            or evidence.get("model_id") != MODEL or evidence.get("dataset_id") != DATASET
            or evidence.get("model_revision") != r.get("model.revision")
            or evidence.get("dataset_revision") != r.get("dataset.revision")
            or evidence.get("source_sha") != r.get("source_sha")
            or evidence.get("seed") != r.get("train.seed")
            or evidence.get("data_seed") != r.get("train.data_seed")
            or any(evidence.get(k) != value for k, value in identity.items())
            or plan.get("execution_identity", identity) != identity):
        raise TaskExecutionError("checkpoint recipe/fingerprint/code/lock/revision/seed 不一致")
    config = _read_json(path / "effective_config.json")
    expected_args = config_kwargs(r, {"recipe_id": r.id, "task_id": TASK_ID,
                                      "mode": evidence["mode"]}, path)
    if (evidence.get("config") != config or evidence.get("config_sha256") != sha256_json(config)
            or training_arguments_identity(config.get("training_arguments"))
            != training_arguments_identity(expected_args)
            or config.get("config") != active_config(r)
            or config.get("config_id") != r.get("execution.config_id")
            or config.get("rapidfire") != r.get("rapidfire") or config.get("peft") != r.get("peft")
            or config.get("quantization") != "none"
            or config.get("attention_implementation") != "eager"
            or config.get("upstream_metadata") != dict(r.get("upstream"))
            or type(evidence.get("steps")) is not int
            or evidence["steps"] != expected_args["max_steps"]):
        raise TaskExecutionError("checkpoint steps/config hash/预算/配置不一致")
    lora = config.get("lora_execution", {})
    targets = lora.get("matched_target_modules", {})
    if (set(targets) != set(r.get("peft.target_modules"))
            or any(not isinstance(v, list) or not v for v in targets.values())
            or type(lora.get("trainable_parameters")) is not int or lora["trainable_parameters"] <= 0
            or lora.get("base_parameters_frozen") is not True
            or lora.get("tensor_reload_check", {}).get("strict_values_verified") is not True):
        raise TaskExecutionError("checkpoint LoRA / adapter 张量复载证据缺失")
    source = _read_json(path / "source_manifest.json")
    validate_source_manifest(source, r)
    if evidence.get("source_manifest") != source or evidence.get("split_manifest") != source:
        raise TaskExecutionError("checkpoint source manifest 不一致")
    tokenizer = _read_json(path / TOKENIZER_IDENTITY_FILE)
    transform = _read_json(path / TRANSFORM_FILE)
    if (not isinstance(tokenizer, dict) or tokenizer.get("sha256") != _hash_manifest(tokenizer)["sha256"]
            or evidence.get("tokenizer_identity") != tokenizer
            or transform.get("sha256") != _hash_manifest(transform)["sha256"]
            or transform.get("tokenizer_identity") != tokenizer
            or transform.get("protocol") != PROTOCOL
            or transform.get("data_seed") != r.get("train.data_seed")
            or evidence.get("transformation_manifest") != transform):
        raise TaskExecutionError("checkpoint tokenizer/transformation manifest 不一致")
    validate_transform_manifest(transform, source, r, tokenizer)
    if not same_artifact(evidence.get("model_artifact"), artifact_manifest(path)):
        raise TaskExecutionError("checkpoint 真实 artifact hash/bytes 不一致")
    return path


def run_evaluation(r, plan, checkpoint=None, zero_shot=False, split_name=None) -> Path:
    lock = _preflight(r, plan)
    if plan.get("requested_runs", 1) != 1 or plan.get("runs", 1) != 1:
        raise TaskExecutionError("仅支持单 checkpoint 单次 greedy；拒绝重复解码冒充训练 seed 方差")
    if split_name is None:
        split_name = plan.get("evaluation_split", "test")
    if split_name not in EVAL_SPLITS:
        raise TaskExecutionError(f"evaluation_split 只允许 {sorted(EVAL_SPLITS)}；拒绝未知 split")
    if plan.get("evaluation_split", split_name) != split_name:
        raise TaskExecutionError("plan.evaluation_split 与请求的评估 split 不一致")
    stage = plan.get("eval_stage", EVAL_STAGES[split_name])
    if stage != EVAL_STAGES[split_name]:
        raise TaskExecutionError(f"{split_name} 评估阶段必须为 {EVAL_STAGES[split_name]!r}")
    output = _output(plan, "zero-shot" if zero_shot else "eval")
    identity = execution_identity(r)
    runtime = load_runtime()
    require_frozen_runtime(r, runtime)
    started = time.perf_counter()
    apply_determinism(r, runtime, evaluation=True)
    torch = runtime["torch"]
    source = None
    if not zero_shot:
        ckpt = checkpoint_path(plan, checkpoint=checkpoint, recipe=r)
        source = _read_json(ckpt / "training.json")
    splits, manifest = load_splits(r, runtime)
    tokenizer, tokenizer_identity, model = _load_base(r, runtime, lock, training=False)
    if not zero_shot:
        peft = runtime["peft"]
        check_signature(peft.PeftModel.from_pretrained,
                        {"model": None, "model_id": None, "is_trainable": False},
                        "PeftModel.from_pretrained")
        model = peft.PeftModel.from_pretrained(
            model=model, model_id=str(ckpt), is_trainable=False)
    rows = splits[split_name]
    if not rows:
        raise TaskExecutionError(f"{split_name} 评估集为空；拒绝用其他 split 冒充")
    maximum = r.get("evaluation.max_new_tokens")
    examples = evaluation_examples(rows, tokenizer, tokenizer_identity, r)
    eval_manifest = evaluation_manifest(examples, tokenizer_identity, r, manifest, split_name)
    eos = tokenizer_identity["eos_token"]
    predictions = []
    for row, example in zip(rows, examples):
        encoded = tokenizer(example["prompt"], add_special_tokens=False, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        length = encoded["input_ids"].shape[1]
        if length != len(example["prompt_ids"]):
            raise TaskExecutionError("评估 prompt 的两次 token 化结果不一致；拒绝记录不可重建输入")
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=maximum, do_sample=False, num_beams=1,
                pad_token_id=tokenizer_identity["pad_token_id"],
                eos_token_id=tokenizer_identity["eos_token_id"])
        text = tokenizer.decode(generated[0][length:], skip_special_tokens=False)
        record = score_prediction(row, text, eos)
        record.update(prompt_sha256=example["prompt_sha256"], generated_text=text,
                      eos_token_id=tokenizer_identity["eos_token_id"],
                      max_new_tokens=maximum)
        predictions.append(record)
    metrics = recompute_metrics(predictions)
    generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": maximum,
                  "eos_token_id": tokenizer_identity["eos_token_id"],
                  "pad_token_id": tokenizer_identity["pad_token_id"], "use_cache": True,
                  "seed": r.get("evaluation.seed"), "eos_truncation": True}
    model_artifact = base_artifact(r) if zero_shot else artifact_manifest(ckpt)
    config = {"task_id": TASK_ID, "protocol": EVAL_PROTOCOL, "zero_shot": bool(zero_shot),
              "config_id": r.get("execution.config_id"),
              "evaluation_split": split_name, "eval_stage": stage,
              "selection_policy": r.get("rapidfire.selection_policy"),
              # checkpoint 是位置，不是内容身份；导出复载会改变它，不参与严格相等比较。
              "checkpoint": None if zero_shot else str(ckpt),
              "generation": generation, "quantization": "none",
              "compute_dtype": "float16", "attention_implementation": "eager",
              "model": model_artifact, "tokenizer_identity": tokenizer_identity,
              "dataset_revision": r.get("dataset.revision"),
              "source_manifest_sha256": manifest["sha256"],
              "eval_manifest_sha256": eval_manifest["sha256"],
              "max_new_tokens": maximum, "do_sample": False, "num_beams": 1,
              "held_out_rows": len(rows), "metric_definition": r.get("rapidfire.metric_definition"),
              "metric_comparable_to_upstream_rougeL": False,
              "concurrent_scheduling": False, "upstream_quantitative_evaluation": False}
    save_json(output / PREDICTIONS_FILE,
              {"schema_version": 1, "task_id": TASK_ID, "recipe_id": r.id,
               "config_id": r.get("execution.config_id"), "protocol": EVAL_PROTOCOL,
               "evaluation_split": split_name, "eval_stage": stage,
               "metrics": metrics, "predictions": predictions,
               "sha256": sha256_json(predictions)})
    predictions_file = file_record(output / PREDICTIONS_FILE, output)
    evidence = _completed(r, plan, identity, "evaluation", config, manifest,
                          runtime_evidence(runtime, started))
    counts = {name: sum(row[name] for row in predictions) for name in METRICS}
    counts["total"] = len(predictions)
    evidence.update(
        model_role="base" if zero_shot else "checkpoint",
        model_artifact=model_artifact,
        checkpoint_training_evidence=source,
        evaluation_split=split_name, eval_stage=stage,
        tokenizer_identity=tokenizer_identity, prompt_protocol=r.get("rapidfire.prompt_format"),
        evaluation_protocol=EVAL_PROTOCOL, evaluation_manifest=eval_manifest,
        metrics=metrics, metric_counts=counts, sample_count=len(predictions),
        predictions={"path": predictions_file["path"], "count": len(predictions),
                     "sha256": sha256_json(predictions)},
        predictions_file=predictions_file,
        predictions_sha256=sha256_json(predictions),
        predictions_path=str(output / PREDICTIONS_FILE),
        split_manifest_path="source_manifest.json",
    )
    save_json(output / "source_manifest.json", manifest)
    save_json(output / TOKENIZER_IDENTITY_FILE, tokenizer_identity)
    save_json(output / EVAL_MANIFEST_FILE, eval_manifest)
    return save_json(output / "evaluation.json", evidence)
