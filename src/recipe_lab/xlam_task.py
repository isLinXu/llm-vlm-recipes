"""xLAM function-calling SFT 的受控自建协议 xlam-fc-v1。

一手事实（HF Cookbook xLAM 页 + 数据卡）与本实现的差异必须显式保留：
  * 上游 SFTTrainer(dataset_text_field="text") 把 query/tools/answers(gold) 拼成整段文本并
    对全文做损失；这里用 transformers.Trainer + 自建 completion-only collator，
    只对 <calls> 严格 JSON 数组 </calls> + EOS 计算 loss。
  * 上游推理 prompt 截断为 "<user>...</user>\n\n<tools>"，不含完整工具定义；这里给全 tools，
    否则 schema 合法率不可解释。
  * 上游展示输出是 Python str(dict) 单引号字面量，不是合法 JSON；这里要求严格 JSON 数组，
    解析只用 json.loads，绝不 eval / literal_eval 生成文本，也绝不执行任何工具调用。
  * 上游无 seed / revision / held-out / compute_metrics；这里全部显式固定并做内容级绑定。

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
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from . import math_task as mt

TaskExecutionError = mt.TaskExecutionError
DatasetValidationError = TaskExecutionError
sha256_json = mt.sha256_json
fixed_seeds = mt.fixed_seeds
require_revision = mt.require_revision
check_signature = mt.check_signature
apply_determinism = mt.apply_determinism
execution_identity = mt.execution_identity
mode_output_dir = mt.mode_output_dir

TASK_ID = "xlam-fc-v1"
RECIPE_ID = "llm/sft/llama3-8b-xlam-function-calling"
MODEL = MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DATASET = DATASET_ID = "Salesforce/xlam-function-calling-60k"
PROTOCOL = "xlam-fc-v1"
EVAL_PROTOCOL = "xlam-fc-v1/held-out-strict-json-v1"
RUNTIME_PACKAGES = ("torch", "transformers", "datasets", "peft", "bitsandbytes", "safetensors")

REQUIRED_COLUMNS = frozenset({"query", "tools", "answers"})
OPTIONAL_COLUMNS = frozenset({"id"})
PROMPT_TEMPLATE = "<user>{query}</user>\n\n<tools>{tools}</tools>\n\n"
CALLS_OPEN, CALLS_CLOSE = "<calls>", "</calls>"
METRICS = ("json_parse_rate", "schema_valid_rate", "tool_call_exact_match", "argument_f1")
TARGET_MODULES = ("k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj")
EVAL_VARIANT = "strict-json-array"
PREDICTIONS_FILE = "predictions.json"
EVAL_MANIFEST_FILE = "eval_manifest.json"
TOKENIZER_IDENTITY_FILE = "tokenizer_identity.json"
# 发布级内容重算必须能从 evaluation.json 所在目录独立重建输入 / gold / 逐题预测。
REFERENCE_FILES = ("source_manifest.json", TOKENIZER_IDENTITY_FILE, EVAL_MANIFEST_FILE)
SCALAR_TYPES = {
    "int": "int", "integer": "int", "float": "float", "number": "float",
    "str": "str", "string": "str", "bool": "bool", "boolean": "bool",
    "list": "list", "array": "list", "dict": "dict", "object": "dict",
}

XLAM_SPEC = {
    "protocol": PROTOCOL, "source_split": "train", "query_column": "query",
    "tools_column": "tools", "answers_column": "answers", "split_percent": [80, 10, 10],
    "split_algorithm": "sha256_json({query,tools,answers,data_seed,protocol}) modulo 100",
    "duplicate_policy": "全局拒绝规范化内容重复，包含跨 split；gold 不进 prompt",
    "normalization": "JSON 解析后按键排序重新序列化；字符串不做 strip/大小写归一，保留原始内容",
    "loss": "completion-only：仅 <calls> 严格JSON数组 </calls> + EOS 参与 causal loss，prompt 部分 label=-100",
    "output_format": "strict-json-array",
    "prompt_format": "<user>{query}</user>\\n\\n<tools>{tools_json}</tools>\\n\\n",
    "target_format": "<calls>{answers_json}</calls>{eos}",
    "tool_validation": "answers.name 必须在 tools.name 内；arguments 键必须在对应 tool.parameters 内；required 参数必须存在；类型与 schema 一致；拒绝同名工具与多余参数",
    "max_eval_examples": 300, "upstream_quantitative_evaluation": False,
}
PEFT_SPEC = {"r": 16, "alpha": 16, "dropout": 0.05, "target_modules": list(TARGET_MODULES),
             "bias": "none", "task_type": "CAUSAL_LM"}
QUANT_SPEC = {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": True}

TRAIN_KEYS = frozenset({
    "seed", "data_seed", "seed_mechanism", "learning_rate", "per_device_train_batch_size",
    "per_device_eval_batch_size", "gradient_accumulation_steps", "max_steps", "seq_length",
    "lr_scheduler_type", "weight_decay", "warmup_ratio", "eval_steps", "save_steps",
    "logging_steps", "bf16", "fp16", "optim", "lora_r", "lora_alpha", "lora_dropout",
    "gradient_checkpointing", "dataloader_num_workers", "remove_unused_columns",
})
METADATA_TRAIN_KEYS = frozenset({"seed", "data_seed", "seed_mechanism", "seq_length",
                                 "lora_r", "lora_alpha", "lora_dropout"})
EVALUATION_KEYS = frozenset({"metrics", "seed", "max_new_tokens", "multi_run", "held_out",
                             "split", "expected", "tolerance", "export_tolerance"})


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


def require_supported(r, plan=None, task_id=None) -> str:
    resolved = resolve_task_id(r, plan, task_id)
    if r.id != RECIPE_ID or resolved != TASK_ID or (plan or {}).get("recipe_id", r.id) != r.id:
        raise TaskExecutionError(f"尚未实现该执行路由：{r.id} / {resolved!r}")
    required = {
        "method.family": "sft", "method.trainer": "Trainer",
        "method.upstream_trainer": "SFTTrainer", "method.peft": "qlora",
        "model.base": MODEL, "dataset.name": DATASET,
    }
    for key, expected in required.items():
        if r.get(key) != expected:
            raise TaskExecutionError(f"{TASK_ID} 要求 {key}={expected!r}")
        section, field = key.split(".")
        planned = (plan or {}).get(section, {})
        if field in planned and planned[field] != expected:
            raise TaskExecutionError(f"plan {key} 与 xLAM 路由不一致")
    if r.is_vlm or r.get("rewards") or r.get("model.modality", "llm") != "llm":
        raise TaskExecutionError("xLAM 仅支持纯文本 function calling，不使用图像或 reward")
    for name, keys in (
        ("method", {"family", "trainer", "upstream_trainer", "peft"}),
        ("model", {"base", "params_b", "modality", "revision"}),
        ("dataset", {"name", "structure", "type", "kind", "has_images", "revision"}),
        ("budget", {"full_steps", "smoke_steps"}),
    ):
        _unknown(r.get(name), keys, name)
    if r.get("dataset.kind", "static") != "static" or r.get("dataset.has_images") is not False:
        raise TaskExecutionError("xLAM 只接受冻结的静态纯文本数据集")
    for name, spec in (("xlam", XLAM_SPEC), ("peft", PEFT_SPEC), ("quantization", QUANT_SPEC)):
        section = r.get(name)
        _unknown(section, spec, name)
        for key, expected in spec.items():
            if key not in section or section[key] != expected or type(section[key]) is not type(expected):
                raise TaskExecutionError(f"{name}.{key} 必须为 {expected!r}")
    train = r.get("train")
    _unknown(train, TRAIN_KEYS, "train")
    missing = sorted(TRAIN_KEYS - set(train))
    if missing:
        raise TaskExecutionError(f"[train] 缺少参数 {missing}")
    _unknown(r.get("evaluation"), EVALUATION_KEYS, "evaluation")
    for key, expected in {
        "train.bf16": True, "train.fp16": False, "train.gradient_checkpointing": True,
        "train.remove_unused_columns": False, "train.optim": "adamw_8bit",
        "train.lora_r": 16, "train.lora_alpha": 16, "train.lora_dropout": 0.05,
        "train.lr_scheduler_type": "linear", "evaluation.metrics": list(METRICS),
        "evaluation.held_out": True, "evaluation.multi_run": True,
    }.items():
        if r.get(key) != expected or (isinstance(expected, bool) and type(r.get(key)) is not bool):
            raise TaskExecutionError(f"{key} 必须为 {expected!r}")
    fixed_seeds(r)
    if not r.has("evaluation.seed"):
        raise TaskExecutionError("evaluation.seed 必须显式固定")
    if not isinstance(train["seed_mechanism"], str) or not train["seed_mechanism"].strip():
        raise TaskExecutionError("seed_mechanism 必须保留上游无 seed 的元信息")
    for key in ("per_device_train_batch_size", "per_device_eval_batch_size",
                "gradient_accumulation_steps", "eval_steps", "save_steps", "logging_steps"):
        if type(train[key]) is not int or train[key] <= 0:
            raise TaskExecutionError(f"train.{key} 必须为正整数")
    if type(train["dataloader_num_workers"]) is not int or train["dataloader_num_workers"] < 0:
        raise TaskExecutionError("train.dataloader_num_workers 必须为非负整数")
    for key in ("learning_rate", "weight_decay", "warmup_ratio"):
        value = train[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise TaskExecutionError(f"train.{key} 必须为有限非负数")
    if train["learning_rate"] == 0 or train["warmup_ratio"] >= 1:
        raise TaskExecutionError("learning_rate 必须大于零且 warmup_ratio < 1")
    sequence, new = train["seq_length"], r.get("evaluation.max_new_tokens")
    if type(new) is not int or new < 2 or type(sequence) is not int or sequence < new + 6:
        raise TaskExecutionError("seq_length 必须为整数且至少 max_new_tokens+6")
    if (type(train["max_steps"]) is not int or train["max_steps"] != 150
            or r.get("budget.full_steps") != 150 or r.get("budget.smoke_steps") != 5):
        raise TaskExecutionError("预算必须显式 full=max_steps=150 / smoke=5；不可静默改预算")
    if not r.has("upstream.dataset_text_field") or not r.has("upstream.defaults_from"):
        raise TaskExecutionError("必须保留上游 dataclass / SFTConfig 元信息，不得把自建参数冒充上游原值")
    dependencies = r.get("dependencies", {})
    _unknown(dependencies, RUNTIME_PACKAGES, "dependencies")
    for name, version in dependencies.items():
        if not isinstance(version, str) or not version.strip() or version in ("unknown", "latest", "*"):
            raise TaskExecutionError(f"dependencies.{name} 不能猜测或使用未冻结版本")
    return resolved


def split_spec(r):
    return {**copy.deepcopy(r.get("xlam")), "data_seed": fixed_seeds(r)[1],
            "selection": "有限完整扫描，保留 source_index；无 shuffle、无补救划分",
            "gold_policy": "answers 只进入训练 target 与评估 gold，绝不进入生成 prompt"}


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
        result["dataset"] = {**result.get("dataset", {}), "split": "content-hash 80/10/10",
                             "split_plan": split_spec(r)}
        result["evaluation"] = {
            "metrics": list(METRICS), "seed": r.get("evaluation.seed"),
            "max_new_tokens": r.get("evaluation.max_new_tokens"), "held_out": True,
            "multi_run": True, "runs": 1, "protocol": EVAL_PROTOCOL, "do_sample": False,
            "training_seed_variance_measured": False, "upstream_quantitative_evaluation": False,
            "generated_code_executed": False,
        }
        result["migration"] = {
            "trainer": "Trainer + completion-only collator（上游 SFTTrainer 全文损失）",
            "prompt": "给全 tools 定义（上游推理 prompt 截断在 <tools>）",
            "output": "严格 JSON 数组（上游展示为 Python str(dict) 单引号）",
            "attention": "eager（自建，不宣称上游 flash_attention_2 复现）",
            "license": "数据卡 gated / research-only；商业使用未确认",
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

def _type_ok(value, declared):
    kind = SCALAR_TYPES.get(str(declared).lower())
    if kind is None:
        return False
    if kind == "int":
        return type(value) is int
    if kind == "float":
        return type(value) in (int, float)
    if kind == "str":
        return isinstance(value, str)
    if kind == "bool":
        return type(value) is bool
    if kind == "list":
        return isinstance(value, list)
    return isinstance(value, dict)


def parse_row(raw, index):
    """严格校验一行 xLAM 数据；任何未知列 / 非法 schema 都拒绝，不静默降级。"""
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"source_index={index} 不是数据行对象")
    columns = set(raw)
    if not REQUIRED_COLUMNS <= columns or not columns <= REQUIRED_COLUMNS | OPTIONAL_COLUMNS:
        raise DatasetValidationError(
            f"source_index={index} 列集合不匹配：{sorted(columns)}；"
            f"必需 {sorted(REQUIRED_COLUMNS)}，可选 {sorted(OPTIONAL_COLUMNS)}")
    query = raw["query"]
    if not isinstance(query, str) or not query.strip():
        raise DatasetValidationError(f"source_index={index} query 必须是非空字符串")
    try:
        tools = json.loads(raw["tools"]) if isinstance(raw["tools"], str) else None
        answers = json.loads(raw["answers"]) if isinstance(raw["answers"], str) else None
    except (TypeError, ValueError) as exc:
        raise DatasetValidationError(f"source_index={index} tools/answers 不是合法 JSON：{exc}") from exc
    if not isinstance(tools, list) or not tools or not isinstance(answers, list) or not answers:
        raise DatasetValidationError(f"source_index={index} tools/answers 必须是非空 JSON 数组")
    schemas, names = {}, set()
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"name", "description", "parameters"}:
            raise DatasetValidationError(f"source_index={index} tool 结构不是 name/description/parameters")
        name, params = tool["name"], tool["parameters"]
        if not isinstance(name, str) or not name.strip() or not isinstance(tool["description"], str):
            raise DatasetValidationError(f"source_index={index} tool.name/description 非法")
        if name in names or not isinstance(params, dict):
            raise DatasetValidationError(f"source_index={index} 工具重名或 parameters 不是对象")
        names.add(name)
        declared = {}
        for key, spec in params.items():
            if (not isinstance(key, str) or not key or not isinstance(spec, dict)
                    or not {"type", "description", "required"} <= set(spec)
                    or type(spec["required"]) is not bool
                    or not isinstance(spec["description"], str)
                    or str(spec["type"]).lower() not in SCALAR_TYPES):
                raise DatasetValidationError(f"source_index={index} tool={name} 参数 schema 非法")
            declared[key] = {"type": spec["type"], "required": spec["required"]}
        schemas[name] = declared
    for answer in answers:
        if not isinstance(answer, dict) or set(answer) != {"name", "arguments"}:
            raise DatasetValidationError(f"source_index={index} answer 结构不是 name/arguments")
        name, arguments = answer["name"], answer["arguments"]
        if name not in schemas or not isinstance(arguments, dict):
            raise DatasetValidationError(f"source_index={index} answer.name 不在 tools 或 arguments 非对象")
        declared = schemas[name]
        if not set(arguments) <= set(declared):
            raise DatasetValidationError(f"source_index={index} answer 含未声明参数 {sorted(set(arguments) - set(declared))}")
        for key, spec in declared.items():
            if spec["required"] and key not in arguments:
                raise DatasetValidationError(f"source_index={index} tool={name} 缺少 required 参数 {key}")
            if key in arguments and not _type_ok(arguments[key], spec["type"]):
                raise DatasetValidationError(f"source_index={index} tool={name} 参数 {key} 类型不符")
    tools_json, answers_json = _canonical(tools), _canonical(answers)
    content = {"query": query, "tools": tools_json, "answers": answers_json}
    return {
        "source_index": index, "sample_id": sha256_json(content)[:32],
        "query": query, "tools": tools_json, "answers": answers_json,
        "tools_obj": tools, "answers_obj": answers,
        "content_sha256": sha256_json(content),
    }


def split_bucket(content_sha256, data_seed):
    digest = hashlib.sha256(f"{PROTOCOL}|{data_seed}|{content_sha256}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def prepare_splits(rows, r):
    """有限完整扫描 + 内容级去重 + 确定性 80/10/10；空划分拒绝，不随机补救。"""
    require_supported(r)
    _, data_seed = fixed_seeds(r)
    max_eval = r.get("xlam.max_eval_examples")
    splits = {"train": [], "validation": [], "test": []}
    seen, duplicates = {}, []
    for index, raw in enumerate(rows):
        row = parse_row(raw, index)
        if row["content_sha256"] in seen:
            duplicates.append({"first": seen[row["content_sha256"]], "again": index,
                               "sha256": row["content_sha256"]})
            continue
        seen[row["content_sha256"]] = index
        splits[split_bucket(row["content_sha256"], data_seed)].append(row)
    if duplicates:
        raise DatasetValidationError(
            f"检测到 {len(duplicates)} 条规范化内容重复（含跨 split）；拒绝静默去重，先复核数据源与 revision")
    for name, items in splits.items():
        if not items:
            raise DatasetValidationError(f"{name} split 为空；拒绝用其他 split 偷偷补救")
    test_total = len(splits["test"])
    if type(max_eval) is not int or max_eval < 1:
        raise DatasetValidationError("xlam.max_eval_examples 必须为正整数")
    eval_rows = splits["test"][:max_eval]
    manifest = {
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "dataset_id": DATASET, "data_seed": data_seed, "source_split": "train",
        "split_algorithm": r.get("xlam.split_algorithm"), "split_percent": [80, 10, 10],
        "duplicate_policy": r.get("xlam.duplicate_policy"),
        "normalization": r.get("xlam.normalization"), "duplicates_rejected": duplicates,
        "counts": {"source": len(seen), "train": len(splits["train"]),
                   "validation": len(splits["validation"]), "test": test_total,
                   "test_eval_used": len(eval_rows)},
        "split_hashes": {
            name: [row["content_sha256"] for row in items] for name, items in splits.items()
        },
        "eval_sample_indices": [row["source_index"] for row in eval_rows],
        "gold_policy": "answers 只进入训练 target 与评估 gold，绝不进入生成 prompt",
    }
    manifest["split_hashes_sha256"] = sha256_json(manifest["split_hashes"])
    manifest["sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    return {"train": splits["train"], "validation": splits["validation"], "test": eval_rows}, manifest


def validate_source_manifest(manifest, r):
    if not isinstance(manifest, dict):
        raise DatasetValidationError("source_manifest 缺失")
    computed = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    if (manifest.get("sha256") != computed or manifest.get("schema_version") != 1
            or manifest.get("protocol") != PROTOCOL or manifest.get("task_id") != TASK_ID
            or manifest.get("dataset_id") != DATASET
            or manifest.get("data_seed") != r.get("train.data_seed")
            or manifest.get("split_hashes_sha256") != sha256_json(manifest.get("split_hashes"))):
        raise DatasetValidationError("source_manifest 内容身份 / 划分 hash 不一致")
    hashes, counts = manifest.get("split_hashes"), manifest.get("counts")
    if not isinstance(hashes, dict) or set(hashes) != {"train", "validation", "test"} or not isinstance(counts, dict):
        raise DatasetValidationError("source_manifest 缺少 split_hashes / counts")
    for name in ("train", "validation", "test"):
        values = hashes[name]
        if (not isinstance(values, list) or not values or counts.get(name) != len(values)
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in values)):
            raise DatasetValidationError(f"source_manifest.{name} 哈希列表非法")
    all_hashes = hashes["train"] + hashes["validation"] + hashes["test"]
    if len(set(all_hashes)) != len(all_hashes):
        raise DatasetValidationError("source_manifest 存在跨 split 重复 hash")
    for name, values in hashes.items():
        for h in values:
            if split_bucket(h, manifest["data_seed"]) != name:
                raise DatasetValidationError(f"source_manifest.{name} hash 与确定性划分规则不一致")
    return manifest


def load_splits(r, dependencies):
    revision = require_revision(r, "dataset.revision")
    dataset = dependencies["datasets"].load_dataset(DATASET, revision=revision, split="train")
    splits, manifest = prepare_splits(dataset, r)
    manifest["dataset_revision"] = revision
    manifest["sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    return splits, manifest


# ---------------------------------------------------------------- 编码与评估

def token_identity(tokenizer):
    special = {"<user>", CALLS_OPEN, CALLS_CLOSE}
    payload = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "control_tokens": {tag: tokenizer(tag, add_special_tokens=False)["input_ids"] for tag in sorted(special)},
    }
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


def render_prompt(row):
    prompt = PROMPT_TEMPLATE.format(query=row["query"], tools=row["tools"])
    if row["answers"] in prompt or row["answers_obj"] == []:
        raise DatasetValidationError("gold 泄漏进生成 prompt 或 gold 为空；拒绝执行")
    return prompt


def render_target(row, eos_token):
    return CALLS_OPEN + row["answers"] + CALLS_CLOSE + eos_token


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
        "sample_id": row["sample_id"], "gold_sha256": sha256_json(row["answers_obj"]),
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
        "tokenizer_identity": identity, "seq_length": r.get("train.seq_length"),
        "data_seed": r.get("train.data_seed"), "loss": r.get("xlam.loss"),
        "output_format": r.get("xlam.output_format"), "completion_only_loss": True,
        "gold_not_in_prompt": True, "boundary_policy": "分别编码 prompt/target 后拼接，labels 边界按 prompt token 数",
        "splits": {name: {"samples": items, "sha256": sha256_json(items),
                          "examples_sha256": sha256_json(examples[name])}
                   for name, items in entries.items()},
    })
    return examples, transform


def validate_transform_manifest(transform, source, r, identity):
    if (not isinstance(transform, dict) or transform.get("sha256") != _hash_manifest(transform)["sha256"]
            or transform.get("protocol") != PROTOCOL or transform.get("task_id") != TASK_ID
            or transform.get("data_seed") != r.get("train.data_seed")
            or transform.get("tokenizer_identity") != identity
            or transform.get("completion_only_loss") is not True
            or transform.get("gold_not_in_prompt") is not True
            or transform.get("loss") != r.get("xlam.loss")):
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


def parse_prediction(text):
    """只用 json.loads 解析 <calls>...</calls>；绝不 eval / literal_eval，绝不执行生成内容。"""
    if not isinstance(text, str):
        return None
    start, stop = text.find(CALLS_OPEN), text.find(CALLS_CLOSE)
    if start < 0 or stop < 0 or stop <= start + len(CALLS_OPEN):
        return None
    body = text[start + len(CALLS_OPEN):stop]
    try:
        value = json.loads(body)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, list) or not value:
        return None
    for call in value:
        if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict)):
            return None
    return value


def validate_calls(calls, tools):
    schemas = {}
    for tool in tools:
        params = tool.get("parameters", {})
        if not isinstance(params, dict):
            return False
        schemas[tool.get("name")] = params
    for call in calls:
        declared = schemas.get(call["name"])
        if declared is None or not set(call["arguments"]) <= set(declared):
            return False
        for key, spec in declared.items():
            if spec.get("required") is True and key not in call["arguments"]:
                return False
            if key in call["arguments"] and not _type_ok(call["arguments"][key], spec.get("type")):
                return False
    return True


def _argument_f1(gold, predicted):
    pairs, used = [], set()
    for call in gold:
        match = next((i for i, candidate in enumerate(predicted)
                      if i not in used and candidate["name"] == call["name"]), None)
        if match is not None:
            used.add(match)
        pairs.append((call["arguments"], predicted[match]["arguments"] if match is not None else {}))
    tp = fp = fn = 0
    for expected, actual in pairs:
        e = {f"{k}={_canonical(v)}" for k, v in expected.items()}
        a = {f"{k}={_canonical(v)}" for k, v in actual.items()}
        tp += len(e & a)
        fp += len(a - e)
        fn += len(e - a)
    fp += sum(len({f"{k}={_canonical(v)}" for k, v in c["arguments"].items()})
              for i, c in enumerate(predicted) if i not in used)
    if tp + fp + fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def score_prediction(row, text):
    gold = row["answers_obj"]
    parsed = parse_prediction(text)
    result = {"sample_id": row["sample_id"], "content_sha256": row["content_sha256"],
              "gold_sha256": sha256_json(gold), "json_parse_rate": 0.0,
              "schema_valid_rate": 0.0, "tool_call_exact_match": 0.0, "argument_f1": 0.0,
              "prediction_sha256": sha256_json(text)}
    if parsed is None:
        return result
    result["json_parse_rate"] = 1.0
    result["schema_valid_rate"] = float(validate_calls(parsed, row["tools_obj"]))
    result["tool_call_exact_match"] = float(_canonical(parsed) == _canonical(gold))
    result["argument_f1"] = _argument_f1(gold, parsed)
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
    """构造可离线重建的 held-out 生成输入 / gold。

    发布级校验器不接触数据集与 HF，必须能从这里落盘的 raw + prompt_ids 重新
    parse_row / render_prompt / score_prediction；所以每行保留规范化原始字段、
    gold 的 JSON 对象 hash、prompt 文本与 token ids，但 gold 绝不进入 prompt。
    """
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
            "raw": {"query": row["query"], "tools": row["tools"], "answers": row["answers"]},
            "tools_sha256": sha256_json(row["tools_obj"]),
            "gold_sha256": sha256_json(row["answers_obj"]),
            "prompt": prompt, "prompt_sha256": sha256_json(prompt),
            "prompt_ids": prompt_ids, "input_sha256": sha256_json(prompt_ids),
        })
    return samples


def evaluation_manifest(samples, identity, r, source):
    """严格 JSON 输出协议的评估清单；任何输入 / gold 改动都会改变 sha256。"""
    return _hash_manifest({
        "schema_version": 1, "kind": "xlam-eval-manifest", "task_id": TASK_ID,
        "protocol": EVAL_PROTOCOL, "variant": EVAL_VARIANT,
        "data_seed": r.get("train.data_seed"),
        "max_new_tokens": r.get("evaluation.max_new_tokens"),
        "tokenizer_identity": identity,
        "source_manifest_sha256": source["sha256"],
        "gold_policy": r.get("xlam.duplicate_policy"),
        "generated_code_executed": False,
        "upstream_quantitative_evaluation": False,
        "sample_count": len(samples),
        "samples": samples,
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
    path = _no_symlinks(path)
    return mt.save_json(path, payload)


def artifact_manifest(directory):
    return mt.artifact_manifest(_no_symlinks(directory))


def file_record(path, base):
    """与 baseline_evidence.hashed_file / export_task._record 兼容的只读文件绑定。"""
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
        raise TaskExecutionError("frozen lock 必须包含 torch/transformers/datasets/peft/bitsandbytes/safetensors")
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
    if not torch.cuda.is_bf16_supported() or not torch.version.cuda:
        raise TaskExecutionError("CUDA GPU 必须支持 bfloat16")
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
        "attention_implementation": "eager", "compute_dtype": "bfloat16",
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


def _quantization(r, runtime):
    kwargs = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
              "bnb_4bit_compute_dtype": runtime["torch"].bfloat16, "bnb_4bit_use_double_quant": True}
    target = runtime["transformers"].BitsAndBytesConfig
    check_signature(target, kwargs, "BitsAndBytesConfig")
    config = target(**kwargs)
    for key, value in kwargs.items():
        if getattr(config, key, None) != value:
            raise TaskExecutionError(f"量化 API 未应用 {key}")
    return config


def _load_base(r, runtime, lock, *, training=False):
    transformers = runtime["transformers"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL, revision=lock["model_revision"], trust_remote_code=False)
    if tokenizer.eos_token is None or tokenizer.pad_token is None:
        raise TaskExecutionError("tokenizer 缺少 eos/pad token；拒绝静默替换配置")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, revision=lock["model_revision"], trust_remote_code=False,
        quantization_config=_quantization(r, runtime), torch_dtype=runtime["torch"].bfloat16,
        attn_implementation="eager", device_map={"": 0})
    model.config.use_cache = not training
    return tokenizer, token_identity(tokenizer), model


def _peft_kwargs(r, revision):
    return {"r": r.get("peft.r"), "lora_alpha": r.get("peft.alpha"),
            "lora_dropout": r.get("peft.dropout"), "target_modules": list(r.get("peft.target_modules")),
            "bias": r.get("peft.bias"), "task_type": r.get("peft.task_type"),
            "base_model_name_or_path": MODEL, "revision": revision}


def _attach_lora(model, runtime, kwargs):
    peft = runtime["peft"]
    check_signature(peft.LoraConfig, kwargs, "LoraConfig")
    check_signature(peft.get_peft_model, {"model": None, "peft_config": None}, "get_peft_model")
    check_signature(peft.prepare_model_for_kbit_training,
                    {"model": None, "use_gradient_checkpointing": True}, "prepare_model_for_kbit_training")
    matches = {target: [name for name, _ in model.named_modules()
                        if name == target or name.endswith("." + target)]
               for target in TARGET_MODULES}
    if any(not value for value in matches.values()):
        raise TaskExecutionError(f"LoRA target 没有全部匹配：{matches}")
    model = peft.prepare_model_for_kbit_training(model=model, use_gradient_checkpointing=True)
    if any(parameter.requires_grad for _, parameter in model.named_parameters()):
        raise TaskExecutionError("prepare_kbit 后 base 参数未全部冻结")
    config = peft.LoraConfig(**kwargs)
    model = peft.get_peft_model(model=model, peft_config=config)
    trainable = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise TaskExecutionError(f"base 参数未冻结：{name}")
            trainable += parameter.numel()
    if trainable <= 0:
        raise TaskExecutionError("LoRA trainable 参数为空")
    return model, {"matched_target_modules": matches, "trainable_parameters": trainable,
                   "base_parameters_frozen": True}


def verify_adapter_tensors(model, path, runtime):
    peft, torch = runtime["peft"], runtime["torch"]
    target = getattr(peft, "get_peft_model_state_dict", None)
    if target is None:
        raise TaskExecutionError("PEFT 缺少明确的 adapter state 提取接口")
    try:
        # 只走 safetensors 的明确加载接口，禁用不安全 pickle 权重。
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
    expected = {"base_model_name_or_path": MODEL, "revision": r.get("model.revision"),
                "r": r.get("peft.r"), "lora_alpha": r.get("peft.alpha"),
                "lora_dropout": r.get("peft.dropout"), "bias": r.get("peft.bias"),
                "task_type": r.get("peft.task_type")}
    for key, value in expected.items():
        if adapter.get(key) != value:
            raise TaskExecutionError(f"adapter_config {key} 与 recipe 不一致")
    if set(adapter.get("target_modules", [])) != set(TARGET_MODULES):
        raise TaskExecutionError("adapter_config target_modules 不一致")


def _completed(r, plan, identity, kind, config, manifest, runtime):
    from .baseline_evidence import fingerprint
    if execution_identity(r) != identity:
        raise TaskExecutionError("执行期间 code/lock identity 漂移；禁止 completed")
    seed, data_seed = fixed_seeds(r)
    return {
        **identity, "schema_version": 1, "kind": kind, "stage": "completed", "task_id": TASK_ID,
        "recipe_id": r.id, "fingerprint": fingerprint(r), "mode": plan["mode"],
        "experiment_id": plan.get("experiment_id", ""),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "data_seed": data_seed, "model_id": MODEL,
        "model_revision": r.get("model.revision"), "dataset_id": DATASET,
        "dataset_revision": r.get("dataset.revision"), "source_sha": r.get("source_sha"),
        "recipe_sha256": sha256_json(r.data), "config": config, "config_sha256": sha256_json(config),
        "split_manifest": manifest, "source_manifest": manifest, "runtime": runtime,
        "baseline_promotion": "none", "training_seed_variance_measured": False,
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
    pad_id = tokenizer_identity["pad_token_id"]
    trainer = transformers.Trainer(
        model=model, args=args,
        train_dataset=runtime["datasets"].Dataset.from_list(examples["train"]),
        eval_dataset=runtime["datasets"].Dataset.from_list(examples["validation"]),
        data_collator=lambda items: collate(items, torch, pad_id))
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
        "training_arguments": canonical, "api_training_arguments": kwargs,
        "peft": r.get("peft"), "quantization": r.get("quantization"),
        "attention_implementation": "eager", "xlam": r.get("xlam"),
        "upstream_metadata": dict(r.get("upstream")),
        "lora_execution": lora_evidence,
    }
    save_json(output / "effective_recipe.json",
              {"id": r.id, "original_path": str(r.path.absolute()), "data": r.data})
    save_json(output / "effective_config.json", effective_config)
    save_json(output / "tokenizer_identity.json", tokenizer_identity)
    save_json(output / "source_manifest.json", source_manifest)
    save_json(output / "xlam_transform_manifest.json", transform)
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
        from .model import Recipe
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
            or (checkpoint is None and evidence.get("mode") != "full")
            or any(evidence.get(k) is True for k in ("mock", "simulated", "dry_run"))):
        raise TaskExecutionError("checkpoint 不是已完成 xLAM full/smoke 训练")
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
            or config.get("xlam") != r.get("xlam") or config.get("peft") != r.get("peft")
            or config.get("quantization") != r.get("quantization")
            or config.get("attention_implementation") != "eager"
            or config.get("upstream_metadata") != dict(r.get("upstream"))
            or type(evidence.get("steps")) is not int
            or evidence["steps"] != expected_args["max_steps"]):
        raise TaskExecutionError("checkpoint steps/config hash/预算不一致")
    lora = config.get("lora_execution", {})
    targets = lora.get("matched_target_modules", {})
    if (set(targets) != set(TARGET_MODULES) or any(not isinstance(v, list) or not v for v in targets.values())
            or type(lora.get("trainable_parameters")) is not int or lora["trainable_parameters"] <= 0
            or lora.get("base_parameters_frozen") is not True
            or lora.get("tensor_reload_check", {}).get("strict_values_verified") is not True):
        raise TaskExecutionError("checkpoint LoRA / adapter 张量复载证据缺失")
    source = _read_json(path / "source_manifest.json")
    validate_source_manifest(source, r)
    if evidence.get("source_manifest") != source or evidence.get("split_manifest") != source:
        raise TaskExecutionError("checkpoint source manifest 不一致")
    tokenizer = _read_json(path / "tokenizer_identity.json")
    transform = _read_json(path / "xlam_transform_manifest.json")
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


def run_evaluation(r, plan, checkpoint=None, zero_shot=False) -> Path:
    lock = _preflight(r, plan)
    if plan.get("requested_runs", 1) != 1 or plan.get("runs", 1) != 1:
        raise TaskExecutionError("仅支持单 checkpoint 单次 greedy；拒绝重复解码冒充训练 seed 方差")
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
    # 评估用 render_prompt 生成，不走训练 collator；checkpoint_path 已单独验证训练侧
    # completion-only transform manifest，这里不伪造一份「评估用 transform」。
    rows = splits["test"]
    maximum = r.get("evaluation.max_new_tokens")
    examples = evaluation_examples(rows, tokenizer, tokenizer_identity, r)
    eval_manifest = evaluation_manifest(examples, tokenizer_identity, r, manifest)
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
        record = score_prediction(row, text)
        record.update(prompt_sha256=example["prompt_sha256"], source_index=row["source_index"],
                      generated_text=text, eos_token_id=tokenizer_identity["eos_token_id"],
                      max_new_tokens=maximum)
        predictions.append(record)
    metrics = recompute_metrics(predictions)
    generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": maximum,
                  "eos_token_id": tokenizer_identity["eos_token_id"],
                  "pad_token_id": tokenizer_identity["pad_token_id"], "use_cache": True,
                  "seed": r.get("evaluation.seed")}
    model_artifact = base_artifact(r) if zero_shot else artifact_manifest(ckpt)
    config = {"task_id": TASK_ID, "protocol": EVAL_PROTOCOL, "zero_shot": bool(zero_shot),
              # checkpoint 是位置，不是内容身份；导出复载会改变它，不参与严格相等比较。
              "checkpoint": None if zero_shot else str(ckpt),
              "generation": generation, "quantization": r.get("quantization"),
              "compute_dtype": "bfloat16", "attention_implementation": "eager",
              "model": model_artifact, "tokenizer_identity": tokenizer_identity,
              "dataset_revision": r.get("dataset.revision"),
              "source_manifest_sha256": manifest["sha256"],
              "eval_manifest_sha256": eval_manifest["sha256"],
              "max_new_tokens": maximum, "do_sample": False, "num_beams": 1,
              "held_out_rows": len(rows), "generated_code_executed": False,
              "upstream_quantitative_evaluation": False}
    save_json(output / PREDICTIONS_FILE,
              {"schema_version": 1, "task_id": TASK_ID, "recipe_id": r.id,
               "protocol": EVAL_PROTOCOL, "metrics": metrics, "predictions": predictions,
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
        tokenizer_identity=tokenizer_identity, prompt_protocol=r.get("xlam.prompt_format"),
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
