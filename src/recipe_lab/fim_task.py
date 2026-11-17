"""文件隔离 FIM 任务；导入和计划只用标准库，不执行生成的源码。

上游使用 Trainer/QLoRA，FIM rate/SPM rate 均为 0.5；没有定量评估。本仓库的
repo-hash 划分、独立文件分块、eager attention 和严格 PSM 指标是自建迁移，
不是上游结果。5905 仅为 main 数据卡元信息，不是冻结 revision 的行数约束。
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
import platform
import random
import re
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import math_task as mt

TASK_ID = "code-fim-v1"
RECIPE_ID = "llm/sft/starcoder1b-fim"
MODEL = MODEL_ID = "bigcode/starcoderbase-1b"
DATASET = DATASET_ID = "smangrul/hf-stack-v1"
PROTOCOL = "repo-isolated-fim-v1"
EVAL_PROTOCOL = "repo-isolated-fim-v1/held-out-psm-v1"
SPECIAL_TOKENS = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
TARGET_MODULES = ["c_proj", "c_attn", "q_attn", "c_fc"]
REQUIRED_MODULES = {"c_proj", "c_attn", "c_fc"}  # q_attn 是候选，不强制开启 cross-attention
RUNTIME_PACKAGES = ("torch", "transformers", "datasets", "peft", "bitsandbytes")
TaskExecutionError = mt.TaskExecutionError
DatasetValidationError = mt.DatasetValidationError
sha256_json = mt.sha256_json
fixed_seeds = mt.fixed_seeds
require_revision = mt.require_revision
check_signature = mt.check_signature
execution_identity = mt.execution_identity
apply_determinism = mt.apply_determinism

FIM_SPEC = {
    "protocol": PROTOCOL, "source_split": "train", "data_dir": "data",
    "repo_column": "repo_id", "path_column": "file_path", "content_column": "content",
    "max_source_examples": 6000, "split_percent": [80, 10, 10],
    "packing": "file-isolated", "loss": "causal-all-nonpadding",
    "eval_variant": "psm", "special_tokens": SPECIAL_TOKENS,
}
PEFT_SPEC = {"r": 8, "alpha": 32, "dropout": 0.0, "target_modules": TARGET_MODULES,
             "bias": "none", "task_type": "CAUSAL_LM"}
QUANT_SPEC = {"load_in_4bit": True, "quant_type": "nf4",
              "compute_dtype": "bfloat16", "double_quant": True}
TRAIN_KEYS = {
    "seed", "seed_mechanism", "learning_rate", "per_device_train_batch_size",
    "gradient_accumulation_steps", "max_steps", "seq_length", "lr_scheduler_type",
    "weight_decay", "warmup_steps", "eval_steps", "bf16", "fim_rate", "fim_spm_rate",
    "lora_r", "lora_alpha", "lora_dropout", "held_out_examples", "data_seed",
    "per_device_eval_batch_size", "save_steps", "logging_steps", "gradient_checkpointing",
    "fp16", "optim", "dataloader_num_workers", "remove_unused_columns",
}
METADATA_TRAIN_KEYS = {"seed_mechanism", "seq_length", "fim_rate", "fim_spm_rate",
                       "lora_r", "lora_alpha", "lora_dropout", "held_out_examples"}
# export_tolerance 由 export_task._tolerance 在导出复载阶段消费；声明它不应被
# require_supported 判为「未消费参数」，否则 FIM 导出永远无法取得完整容差表。
EVALUATION_KEYS = {"metrics", "seed", "max_new_tokens", "multi_run", "held_out", "split",
                   "expected", "tolerance", "export_tolerance"}


def _hash_manifest(value):
    value = {k: v for k, v in value.items() if k != "sha256"}
    return {**value, "sha256": sha256_json(value)}


def _unknown(section, allowed, label):
    if not isinstance(section, dict):
        raise TaskExecutionError(f"[{label}] 必须为参数表")
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise TaskExecutionError(f"[{label}] 含未消费参数 {unknown}；拒绝静默忽略")


def resolve_task_id(r, plan=None, task_id=None):
    values = [v for v in (task_id, (plan or {}).get("task_id"), r.get("execution.task_id"))
              if v is not None]
    if any(v != TASK_ID for v in values):
        raise TaskExecutionError(f"任务路由冲突：{values!r}")
    return TASK_ID if values or r.id == RECIPE_ID else None


def require_supported(r, plan=None, task_id=None) -> str:
    resolved = resolve_task_id(r, plan, task_id)
    if r.id != RECIPE_ID or resolved != TASK_ID or (plan or {}).get("recipe_id", r.id) != r.id:
        raise TaskExecutionError(f"尚未实现该执行路由：{r.id} / {resolved!r}")
    required = {
        "method.family": "sft", "method.trainer": "Trainer", "method.upstream_trainer": "Trainer",
        "method.peft": "qlora", "model.base": MODEL, "dataset.name": DATASET,
    }
    for key, expected in required.items():
        if r.get(key) != expected:
            raise TaskExecutionError(f"{TASK_ID} 要求 {key}={expected!r}")
        section, field = key.split(".")
        planned = (plan or {}).get(section, {})
        if field in planned and planned[field] != expected:
            raise TaskExecutionError(f"plan {key} 与 FIM 路由不一致")
    if r.is_vlm or r.get("rewards") or r.get("model.modality", "llm") != "llm":
        raise TaskExecutionError("FIM 仅支持纯文本 causal loss，不能使用图像或 reward")
    for name, keys in (
        ("method", {"family", "trainer", "upstream_trainer", "peft"}),
        ("model", {"base", "revision", "params_b", "modality"}),
        ("dataset", {"name", "revision", "structure", "type", "kind", "has_images"}),
        ("budget", {"full_steps", "smoke_steps"}),
    ):
        _unknown(r.get(name), keys, name)
    if r.get("dataset.kind", "static") != "static":
        raise TaskExecutionError("FIM 只支持冻结的 static 数据集")
    if r.has("evaluation.held_out") and r.get("evaluation.held_out") is not True:
        raise TaskExecutionError("FIM 评估必须为 held-out")
    for name, spec in (("fim", FIM_SPEC), ("peft", PEFT_SPEC), ("quantization", QUANT_SPEC)):
        section = r.get(name)
        _unknown(section, spec, name)
        for key, expected in spec.items():
            actual = section.get(key)
            same_type = (type(actual) in (int, float) if type(expected) is float
                         else type(actual) is type(expected))
            if key not in section or actual != expected or not same_type:
                raise TaskExecutionError(f"{name}.{key} 必须为 {expected!r}")
    train = r.get("train")
    _unknown(train, TRAIN_KEYS, "train")
    missing = sorted(TRAIN_KEYS - set(train))
    if missing:
        raise TaskExecutionError(f"[train] 缺少参数 {missing}")
    _unknown(r.get("evaluation"), EVALUATION_KEYS, "evaluation")
    for key, expected in {
        "train.bf16": True, "train.fp16": False, "train.gradient_checkpointing": True,
        "train.remove_unused_columns": False, "train.optim": "adamw_torch",
        "train.dataloader_num_workers": 0, "train.lora_r": 8, "train.lora_alpha": 32,
        "train.lora_dropout": 0.0, "train.held_out_examples": 4000,
        "evaluation.metrics": ["fim_exact_match", "fim_token_exact_match"],
        "evaluation.multi_run": True,
    }.items():
        if r.get(key) != expected or (isinstance(expected, bool) and type(r.get(key)) is not bool):
            raise TaskExecutionError(f"{key} 必须为 {expected!r}")
    fixed_seeds(r)
    if not r.has("evaluation.seed"):
        raise TaskExecutionError("evaluation.seed 必须显式固定")
    if not isinstance(train["seed_mechanism"], str) or not train["seed_mechanism"].strip():
        raise TaskExecutionError("seed_mechanism 必须保留明确的上游元信息")
    for key in ("per_device_train_batch_size", "per_device_eval_batch_size",
                "gradient_accumulation_steps", "eval_steps", "save_steps", "logging_steps"):
        if type(train[key]) is not int or train[key] <= 0:
            raise TaskExecutionError(f"train.{key} 必须为正整数")
    for key in ("warmup_steps", "dataloader_num_workers"):
        if type(train[key]) is not int or train[key] < 0:
            raise TaskExecutionError(f"train.{key} 必须为非负整数")
    for key in ("learning_rate", "weight_decay", "fim_rate", "fim_spm_rate"):
        value = train[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise TaskExecutionError(f"train.{key} 必须为有限非负数")
    if train["learning_rate"] == 0 or any(train[k] > 1 for k in ("fim_rate", "fim_spm_rate")):
        raise TaskExecutionError("learning_rate 必须大于零，FIM rates 必须在 [0,1]")
    if train["lr_scheduler_type"] != "cosine":
        raise TaskExecutionError("只核验 cosine scheduler")
    sequence, new = train["seq_length"], r.get("evaluation.max_new_tokens")
    if type(new) is not int or new < 2 or type(sequence) is not int or sequence < new + 6:
        raise TaskExecutionError("seq_length 必须为整数且至少 max_new_tokens+6，保证非空 gap 和生成窗口")
    if (type(train["max_steps"]) is not int or train["max_steps"] != 2000
            or type(r.get("budget.full_steps")) is not int or r.get("budget.full_steps") != 2000
            or type(r.get("budget.smoke_steps")) is not int or r.get("budget.smoke_steps") != 5):
        raise TaskExecutionError("预算必须显式 full=max_steps=2000 / smoke=5；不可静默限为150")
    dependencies = r.get("dependencies", {})
    _unknown(dependencies, RUNTIME_PACKAGES, "dependencies")
    for name, version in dependencies.items():
        if not isinstance(version, str) or not version.strip() or version in ("unknown", "latest", "*"):
            raise TaskExecutionError(f"dependencies.{name} 不能猜测或使用未冻结版本")
    return resolved


def split_spec(r):
    return {**copy.deepcopy(r.get("fim")), "data_seed": fixed_seeds(r)[1],
            "selection": "有限完整扫描，保留 source_index；无 shuffle、无补救划分",
            "split_algorithm": "sha256_json({repo_id,data_seed,protocol}) modulo 100",
            "duplicate_policy": "全局拒绝相同 raw/normalized 源码，包含同 split",
            "normalization": "CRLF/CR 转 LF；逐行及文件末尾去尾空白；保留缩进",
            "main_card_examples_metadata": 5905}


def mode_output_dir(plan, mode):
    return mt.mode_output_dir(plan, mode)


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
        result["dataset"] = {**result.get("dataset", {}), "split": "repo-hash 80/10/10",
                             "split_plan": split_spec(r)}
        result["evaluation"] = {
            "metrics": r.get("evaluation.metrics"), "seed": r.get("evaluation.seed"),
            "max_new_tokens": r.get("evaluation.max_new_tokens"), "held_out": True,
            "multi_run": True, "runs": 1, "protocol": EVAL_PROTOCOL,
            "do_sample": False, "training_seed_variance_measured": False,
            "upstream_quantitative_evaluation": False,
        }
        result["migration"] = {"attention": "eager（自建，非上游 flash-attn）",
                               "held_out_examples": "4000 仅为上游元信息，不参与划分"}
    except (TaskExecutionError, KeyError, TypeError) as exc:
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
    """只选择实际 API 显式声明的策略名；**kwargs 不代表支持。"""
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise TaskExecutionError("无法核验 TrainingArguments API") from exc
    keys = signature.parameters
    name = "eval_strategy" if "eval_strategy" in keys else "evaluation_strategy"
    result = dict(canonical)
    if name == "eval_strategy":
        result[name] = result.pop("evaluation_strategy")
    check_signature(target, result, "TrainingArguments")
    return result


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_source(content):
    return "\n".join(line.rstrip() for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n")).rstrip()


def repo_bucket(repo_id, data_seed):
    return int(sha256_json({"repo_id": repo_id, "data_seed": data_seed, "protocol": PROTOCOL}), 16) % 100


def repo_split(repo_id, data_seed):
    bucket = repo_bucket(repo_id, data_seed)
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _schema_row(raw, index, r):
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"train:{index} 不是结构化对象")
    values = []
    for field in (r.get("fim.repo_column"), r.get("fim.path_column"), r.get("fim.content_column")):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(f"train:{index} 缺少非空字符串 {field}")
        if (any(ord(c) < 32 and c not in "\n\r\t" or ord(c) == 127 for c in value)
                or any(t in value for t in SPECIAL_TOKENS)):
            raise DatasetValidationError(f"train:{index} 的 {field} 含特殊控制 token")
        if field != r.get("fim.content_column") and any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise DatasetValidationError(f"train:{index} 元信息包含控制字符")
        values.append(value)
    repo, path, content = values
    if (PurePosixPath(path).is_absolute() or "\\" in path or ":" in path
            or any(part in ("..", ".", "") for part in path.split("/"))):
        raise DatasetValidationError(f"train:{index} file_path 越界或不规范")
    try:
        raw_hash = sha256_json(raw)
        content_hash = _text_hash(content)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DatasetValidationError(f"train:{index} schema/meta 无法稳定序列化") from exc
    entry = {
        "sample_id": f"train:{index}", "source_index": index, "repo_id": repo, "file_path": path,
        "file_id": sha256_json({"repo_id": repo, "file_path": path}), "row_sha256": raw_hash,
        "content_sha256": content_hash, "normalized_sha256": _text_hash(normalize_source(content)),
        "repo_sha256": sha256_json(repo), "repo_bucket": repo_bucket(repo, fixed_seeds(r)[1]),
    }
    return entry, content


def prepare_splits(rows, r) -> tuple[dict, dict]:
    require_supported(r)
    maximum = r.get("fim.max_source_examples")
    selected = list(itertools.islice(iter(rows), maximum + 1))
    if len(selected) > maximum:
        raise DatasetValidationError(f"source 超过 max_source_examples={maximum}；拒绝截断")
    # 在划分或 tokenization 之前检查全部 schema，包含最后一条元信息。
    checked = [_schema_row(raw, index, r) for index, raw in enumerate(selected)]
    splits = {name: [] for name in ("train", "validation", "test")}
    entries = {name: [] for name in splits}
    owners = {}
    for entry, content in checked:
        name = repo_split(entry["repo_id"], fixed_seeds(r)[1])
        for field in ("file_id", "content_sha256", "normalized_sha256"):
            key = (field, entry[field])
            if key in owners:
                raise DatasetValidationError(f"{entry['sample_id']} 重复/源码泄漏 {field}：{owners[key]} -> {name}")
            owners[key] = name
        splits[name].append({**entry, "content": content})
        entries[name].append(entry)
    if any(not values for values in splits.values()):
        raise DatasetValidationError("repo-hash 导致 empty split；拒绝随机补救")
    manifest = _hash_manifest({
        "schema_version": 1, "task_id": TASK_ID, "recipe_id": RECIPE_ID, "dataset_id": DATASET,
        "schema": {"required": ["repo_id", "file_path", "content"], "type": "nonempty-string",
                   "metadata": "保留全部原始行内容的 row_sha256", "paths": "repo-relative-posix"},
        "split_parameters": split_spec(r), "source_count": len(selected),
        "splits": {name: {"count": len(items), "repo_count": len({x['repo_id'] for x in items}),
                          "samples": items, "sha256": sha256_json(items)} for name, items in entries.items()},
    })
    return splits, manifest


def validate_source_manifest(manifest, r):
    require_supported(r)
    if not isinstance(manifest, dict) or manifest.get("sha256") != _hash_manifest(manifest)["sha256"]:
        raise TaskExecutionError("source manifest hash 不一致")
    if (manifest.get("schema_version") != 1 or manifest.get("task_id") != TASK_ID
            or manifest.get("recipe_id") != RECIPE_ID or manifest.get("dataset_id") != DATASET
            or manifest.get("split_parameters") != split_spec(r)
            or manifest.get("schema") != {"required": ["repo_id", "file_path", "content"],
                "type": "nonempty-string", "metadata": "保留全部原始行内容的 row_sha256",
                "paths": "repo-relative-posix"}):
        raise TaskExecutionError("source manifest schema/协议不一致")
    owners, indices, count = set(), set(), 0
    for name in ("train", "validation", "test"):
        split = manifest.get("splits", {}).get(name, {})
        samples = split.get("samples")
        if (not isinstance(samples, list) or not samples or split.get("count") != len(samples)
                or split.get("sha256") != sha256_json(samples)):
            raise TaskExecutionError("source manifest count/hash 不一致")
        previous = -1
        for item in samples:
            index = item.get("source_index")
            repo, path = item.get("repo_id"), item.get("file_path")
            if (type(index) is not int or index <= previous or index in indices
                    or item.get("sample_id") != f"train:{index}"
                    or not isinstance(repo, str) or not repo.strip()
                    or not isinstance(path, str) or not path.strip()
                    or item.get("file_id") != sha256_json({"repo_id": repo, "file_path": path})
                    or item.get("repo_sha256") != sha256_json(repo)
                    or item.get("repo_bucket") != repo_bucket(repo, fixed_seeds(r)[1])
                    or repo_split(repo, fixed_seeds(r)[1]) != name):
                raise TaskExecutionError("source manifest 索引/repo 划分不一致")
            previous = index
            indices.add(index)
            for field in ("row_sha256", "content_sha256", "normalized_sha256", "file_id"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(item.get(field, ""))):
                    raise TaskExecutionError("source manifest hash 字段缺失")
                if field != "row_sha256":
                    key = (field, item[field])
                    if key in owners:
                        raise TaskExecutionError("source manifest 重复/泄漏")
                    owners.add(key)
        if split.get("repo_count") != len({x["repo_id"] for x in samples}):
            raise TaskExecutionError("source manifest repo_count 不一致")
        count += len(samples)
    if manifest.get("source_count") != count or indices != set(range(count)) or count > r.get("fim.max_source_examples"):
        raise TaskExecutionError("source manifest 原始数量/索引不一致")
    if manifest.get("dataset_revision") != require_revision(r, "dataset.revision"):
        raise TaskExecutionError("source manifest revision 不一致")


def load_splits(r, dependencies):
    revision = require_revision(r, "dataset.revision")
    rows = dependencies["datasets"].load_dataset(
        DATASET, revision=revision, data_dir="data", split="train", streaming=True,
        trust_remote_code=False,
    )
    splits, manifest = prepare_splits(rows, r)
    manifest = _hash_manifest({**manifest, "dataset_revision": revision})
    validate_source_manifest(manifest, r)
    return splits, manifest


def token_identity(tokenizer):
    vocab = tokenizer.get_vocab()
    eos, unk = tokenizer.eos_token_id, tokenizer.unk_token_id
    ids = []
    if type(eos) is not int or eos < 0 or eos not in vocab.values():
        raise TaskExecutionError("tokenizer EOS 无效")
    for token in SPECIAL_TOKENS:
        value = tokenizer.convert_tokens_to_ids(token)
        if (token not in vocab or type(value) is not int or value < 0 or value == unk
                or vocab[token] != value or tokenizer.encode(token, add_special_tokens=False) != [value]):
            raise TaskExecutionError(f"缺少真实单 token FIM 标记 {token}；禁止 fallback/增新token")
        ids.append(value)
    if len(set([*ids, eos])) != 4:
        raise TaskExecutionError("FIM 三个 token 与 EOS 不能别名")
    # pad=eos，但屏蔽来自实际长度，不从 token ID 推断。
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id != eos:
        raise TaskExecutionError("tokenizer 未应用 pad=eos")
    identity = {
        "vocab_sha256": sha256_json(vocab), "vocab_size": len(vocab),
        "special_token_ids": dict(zip(SPECIAL_TOKENS, ids)), "eos_token_id": eos,
        "unk_token_id": unk, "pad_token_id": eos, "padding_side": "right",
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_ids": sorted(set(tokenizer.all_special_ids)),
        "backend_sha256": (_text_hash(tokenizer.backend_tokenizer.to_str())
                           if getattr(tokenizer, "backend_tokenizer", None) is not None else None),
        "tokenizer_class": type(tokenizer).__name__,
    }
    return _hash_manifest(identity)


def encode_source(tokenizer, content, identity):
    tokens = tokenizer.encode(content, add_special_tokens=False, truncation=False)
    if (not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens)
            or any(t in set(identity["all_special_ids"]) | set(identity["special_token_ids"].values())
                   | {identity["eos_token_id"]} for t in tokens)):
        raise DatasetValidationError("源码 tokenization 含特殊 ID 或无效 token；禁止 FIM fallback")
    return tokens


def local_rng(data_seed, sample_id, chunk_index, purpose="train"):
    return random.Random(int(sha256_json({"data_seed": data_seed, "sample_id": sample_id,
                                         "chunk_index": chunk_index, "protocol": PROTOCOL,
                                         "purpose": purpose}), 16))


def fim_sequence(tokens, identity, rng, variant):
    if len(tokens) < 3 or variant not in ("psm", "spm"):
        raise DatasetValidationError("FIM chunk 至少三个 token，variant 只能为 psm/spm")
    lo, hi = sorted(rng.sample(range(len(tokens) + 1), 2))
    prefix, middle, suffix = tokens[:lo], tokens[lo:hi], tokens[hi:]
    p, s, m = [identity["special_token_ids"][t] for t in SPECIAL_TOKENS]
    sequence = ([p] + prefix + [s] + suffix + [m] + middle if variant == "psm"
                else [p, s] + suffix + [m] + prefix + middle)
    return sequence, {"variant": variant, "gap_start": lo, "gap_stop": hi,
                      "prefix_ids": prefix, "suffix_ids": suffix, "middle_ids": middle}


def train_examples(splits, tokenizer, r):
    require_supported(r)
    identity = token_identity(tokenizer)
    examples, entries = {}, {}
    seq_length, data_seed = r.get("train.seq_length"), fixed_seeds(r)[1]
    capacity, eos = seq_length - 4, identity["eos_token_id"]
    for name in ("train", "validation"):
        examples[name], entries[name] = [], []
        for row in splits[name]:
            tokens = encode_source(tokenizer, row["content"], identity)
            chunks, dropped = [], []
            for chunk_index, offset in enumerate(range(0, len(tokens), capacity)):
                chunk = tokens[offset:offset + capacity]
                if len(chunk) < 3:
                    dropped = chunk
                    break
                rng = local_rng(data_seed, row["sample_id"], chunk_index, name)
                variant = "psm" if name == "validation" else (
                    "spm" if rng.random() < r.get("train.fim_spm_rate") else "psm")
                if name == "validation" or rng.random() < r.get("train.fim_rate"):
                    input_ids, transformation = fim_sequence(chunk, identity, rng, variant)
                else:
                    input_ids, transformation = list(chunk), {"variant": "causal"}
                input_ids = input_ids + [eos]
                real_length = len(input_ids)
                labels = list(input_ids) + [-100] * (seq_length - real_length)
                attention = [1] * real_length + [0] * (seq_length - real_length)
                input_ids += [eos] * (seq_length - real_length)
                item = {"input_ids": input_ids, "labels": labels, "attention_mask": attention}
                examples[name].append(item)
                chunks.append({"chunk_index": chunk_index, "token_start": offset,
                               "token_stop": offset + len(chunk), "source_ids": chunk,
                               "source_tokens_sha256": sha256_json(chunk),
                               "real_length": real_length, **transformation, "example_sha256": sha256_json(item)})
            entries[name].append({"sample_id": row["sample_id"], "file_id": row["file_id"],
                                  "content_sha256": row["content_sha256"], "token_count": len(tokens),
                                  "tokens_sha256": sha256_json(tokens), "chunks": chunks,
                                  "tail_dropped": {"count": len(dropped), "ids": dropped,
                                                   "sha256": sha256_json(dropped)}})
        if not examples[name]:
            raise DatasetValidationError(f"{name} 分块后为空；不把 tail_dropped 计为完整样本")
    transform = _hash_manifest({
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "tokenizer_identity": identity, "seq_length": seq_length, "chunk_capacity": capacity,
        "data_seed": data_seed, "fim_rate": r.get("train.fim_rate"), "fim_spm_rate": r.get("train.fim_spm_rate"),
        "packing": "file-isolated", "loss": "causal-all-nonpadding", "validation_variant": "psm",
        "seed_mechanism_upstream_metadata": r.get("train.seed_mechanism"),
        "held_out_examples_upstream_metadata": r.get("train.held_out_examples"),
        "target_modules_upstream_difference": "重复 c_proj 去重；q_attn 为候选可未命中，c_proj/c_attn/c_fc 必须匹配", 
        "splits": {name: {"file_count": len(items), "chunk_count": len(examples[name]),
                          "samples": items, "sha256": sha256_json(items),
                          "examples_sha256": sha256_json(examples[name])} for name, items in entries.items()},
    })
    return examples, transform


def validate_transform_manifest(transform, source, r, identity):
    """重建每个独立 chunk 的 labels/mask/hash，明确核算丢弃的短尾巴。"""
    if (not isinstance(transform, dict) or transform.get("sha256") != _hash_manifest(transform)["sha256"]
            or transform.get("protocol") != PROTOCOL or transform.get("task_id") != TASK_ID
            or transform.get("schema_version") != 1 or transform.get("data_seed") != r.get("train.data_seed")
            or transform.get("tokenizer_identity") != identity
            or transform.get("seq_length") != r.get("train.seq_length")
            or transform.get("chunk_capacity") != r.get("train.seq_length") - 4
            or transform.get("packing") != "file-isolated" or transform.get("loss") != "causal-all-nonpadding"
            or transform.get("fim_rate") != r.get("train.fim_rate")
            or transform.get("fim_spm_rate") != r.get("train.fim_spm_rate")):
        raise TaskExecutionError("transformation manifest 协议/hash 不一致")
    capacity, length, eos = transform["chunk_capacity"], transform["seq_length"], identity["eos_token_id"]
    p, s, m = [identity["special_token_ids"][t] for t in SPECIAL_TOKENS]
    for name in ("train", "validation"):
        split = transform.get("splits", {}).get(name, {})
        files = split.get("samples")
        source_files = source["splits"][name]["samples"]
        if (not isinstance(files, list) or len(files) != len(source_files)
                or split.get("file_count") != len(files) or split.get("sha256") != sha256_json(files)):
            raise TaskExecutionError("transformation manifest 文件 count/hash 不一致")
        examples = []
        for file, original in zip(files, source_files):
            if any(file.get(k) != original[k] for k in ("sample_id", "file_id", "content_sha256")):
                raise TaskExecutionError("transformation manifest 与 source 文件身份不一致")
            tokens = []
            for index, chunk in enumerate(file.get("chunks", [])):
                raw = chunk.get("source_ids")
                if (not isinstance(raw, list) or not 3 <= len(raw) <= capacity
                        or chunk.get("chunk_index") != index or chunk.get("token_start") != index * capacity
                        or chunk.get("token_stop") != index * capacity + len(raw)
                        or chunk.get("source_tokens_sha256") != sha256_json(raw)):
                    raise TaskExecutionError("transformation manifest token offset/hash 不一致")
                variant = chunk.get("variant")
                if variant == "causal" and name == "train":
                    encoded = list(raw)
                elif variant in ("psm", "spm") and not (name == "validation" and variant != "psm"):
                    prefix, middle, suffix = chunk["prefix_ids"], chunk["middle_ids"], chunk["suffix_ids"]
                    if (not middle or prefix + middle + suffix != raw or chunk["gap_start"] != len(prefix)
                            or chunk["gap_stop"] != len(prefix) + len(middle)):
                        raise TaskExecutionError("transformation manifest PSM/SPM gap 不一致")
                    encoded = ([p] + prefix + [s] + suffix + [m] + middle if variant == "psm"
                               else [p, s] + suffix + [m] + prefix + middle)
                else:
                    raise TaskExecutionError("transformation manifest variant 不一致")
                encoded += [eos]
                real = len(encoded)
                if chunk.get("real_length") != real:
                    raise TaskExecutionError("transformation manifest 真实长度不一致")
                item = {"input_ids": encoded + [eos] * (length - real),
                        "labels": encoded + [-100] * (length - real),
                        "attention_mask": [1] * real + [0] * (length - real)}
                if chunk.get("example_sha256") != sha256_json(item):
                    raise TaskExecutionError("transformation manifest labels/padding hash 不一致")
                examples.append(item)
                tokens += raw
            tail = file.get("tail_dropped", {})
            dropped = tail.get("ids")
            if (not isinstance(dropped, list) or len(dropped) >= 3 or tail.get("count") != len(dropped)
                    or tail.get("sha256") != sha256_json(dropped)):
                raise TaskExecutionError("transformation manifest tail_dropped 不一致")
            tokens += dropped
            if (file.get("token_count") != len(tokens) or file.get("tokens_sha256") != sha256_json(tokens)
                    or any(type(t) is not int or t < 0 or t in identity["all_special_ids"] for t in tokens)
                    or len(file["chunks"]) != len(tokens) // capacity + int(len(tokens) % capacity >= 3)):
                raise TaskExecutionError("transformation manifest 原始 token/分块计数不一致")
        if (not examples or split.get("chunk_count") != len(examples)
                or split.get("examples_sha256") != sha256_json(examples)):
            raise TaskExecutionError("transformation manifest chunk count/hash 不一致")


def collate(items, torch):
    if not items or any(set(item) != {"input_ids", "labels", "attention_mask"} for item in items):
        raise DatasetValidationError("collator 仅接受显式 input_ids/labels/attention_mask")
    length = len(items[0]["input_ids"])
    for item in items:
        if any(len(item[key]) != length for key in item):
            raise DatasetValidationError("collator 不重新跨文件 packing/padding")
        if any((mask == 0 and label != -100) or (mask == 1 and label != token) or mask not in (0, 1)
               for token, label, mask in zip(item["input_ids"], item["labels"], item["attention_mask"])):
            raise DatasetValidationError("labels/attention 与真实长度不一致")
    return {key: torch.tensor([item[key] for item in items], dtype=torch.long)
            for key in ("input_ids", "labels", "attention_mask")}


def load_runtime(*, training=False):
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise TaskExecutionError("缺少 torch；按真实 frozen lock 安装，dry-run 不需要依赖") from exc
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


def _no_symlinks(path):
    path = Path(path).expanduser()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise TaskExecutionError(f"artifact/lock 路径不能为 symlink：{path}")
    if path.is_dir() and any(p.is_symlink() for p in path.rglob("*")):
        raise TaskExecutionError(f"artifact 包含 symlink：{path}")
    return path


def _lock(r):
    try:
        path = _no_symlinks(r.path.parent / r.get("execution.lockfile"))
        lock = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise TaskExecutionError("无法读取 frozen lock") from exc
    for key, value in {"recipe_id": r.id, "model_id": MODEL, "dataset_id": DATASET,
                       "model_revision": require_revision(r, "model.revision"),
                       "dataset_revision": require_revision(r, "dataset.revision"),
                       "source_sha": require_revision(r, "source_sha")}.items():
        if lock.get(key) != value:
            raise TaskExecutionError(f"lock {key} 与 recipe 不一致")
    if type(lock.get("schema_version")) is not int or lock["schema_version"] != 1 or lock.get("status") != "frozen":
        raise TaskExecutionError("执行只接受真实 frozen lock")
    names = {re.split(r"(?:==| @ )", line, maxsplit=1)[0].lower().replace("_", "-")
             for line in lock.get("requirements", []) if isinstance(line, str)}
    if not set(RUNTIME_PACKAGES) <= names:
        raise TaskExecutionError("frozen lock 必须包含 torch/transformers/datasets/peft/bitsandbytes")
    return lock


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


def artifact_manifest(directory):
    return mt.artifact_manifest(_no_symlinks(directory))


# 导出/复载会把 checkpoint 复制到新目录：绝对 path 与 TrainingArguments.output_dir
# 都是位置而不是内容身份。忽略这两项，其余文件清单/hash/超参仍必须逐项严格一致。
PORTABLE_ARGUMENT_KEYS = frozenset({"output_dir"})


def artifact_identity(artifact):
    """仅保留内容身份；跨目录复载时 artifact_manifest.path 必然变化。"""
    if not isinstance(artifact, dict):
        return None
    return {"files": artifact.get("files"), "sha256": artifact.get("sha256")}


def training_arguments_identity(arguments):
    """仅去掉 output_dir；任何真实超参变化仍会导致不一致。"""
    if not isinstance(arguments, dict):
        return None
    return {key: value for key, value in arguments.items() if key not in PORTABLE_ARGUMENT_KEYS}


def same_artifact(recorded, actual):
    return (isinstance(recorded, dict)
            and artifact_identity(recorded) == artifact_identity(actual)
            and isinstance(recorded.get("path"), str) and bool(recorded["path"]))


def save_json(path, payload):
    path = _no_symlinks(path)
    _no_symlinks(path.with_suffix(path.suffix + ".tmp"))
    return mt.save_json(path, payload)


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
        raise TaskExecutionError("独立训练多 seed 由主线程编排，不重复 greedy 伪装 multi-run")
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
        MODEL, revision=lock["model_revision"], trust_remote_code=False,
    )
    identity = token_identity(tokenizer)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, revision=lock["model_revision"], trust_remote_code=False,
        quantization_config=_quantization(r, runtime), torch_dtype=runtime["torch"].bfloat16,
        attn_implementation="eager", device_map={"": 0},
    )
    model.config.use_cache = not training
    return tokenizer, identity, model


def _peft_api(runtime, r, revision):
    peft = runtime["peft"]
    kwargs = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": TARGET_MODULES,
              "bias": "none", "task_type": "CAUSAL_LM", "base_model_name_or_path": MODEL,
              "revision": revision}
    check_signature(peft.LoraConfig, kwargs, "LoraConfig")
    check_signature(peft.prepare_model_for_kbit_training,
                    {"model": None, "use_gradient_checkpointing": True}, "prepare_model_for_kbit_training")
    check_signature(peft.get_peft_model, {"model": None, "peft_config": None}, "get_peft_model")
    return kwargs


def _attach_lora(model, runtime, kwargs):
    matches = {target: [name for name, _ in model.named_modules()
                        if name == target or name.endswith("." + target)] for target in TARGET_MODULES}
    if any(not matches[target] for target in REQUIRED_MODULES):
        raise TaskExecutionError(f"LoRA target 没有全部匹配：{matches}")
    peft = runtime["peft"]
    model = peft.prepare_model_for_kbit_training(model=model, use_gradient_checkpointing=True)
    if any(parameter.requires_grad for _, parameter in model.named_parameters()):
        raise TaskExecutionError("prepare_kbit 后 base 参数未全部冻结")
    config = peft.LoraConfig(**kwargs)
    for key, value in kwargs.items():
        actual = getattr(config, key, None)
        if key == "target_modules":
            actual, value = set(actual or []), set(value)
        if actual != value:
            raise TaskExecutionError(f"LoRA API 未应用 {key}")
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


def _completed(r, plan, identity, kind, config, source_manifest, runtime):
    from .baseline_evidence import fingerprint
    if execution_identity(r) != identity:
        raise TaskExecutionError("执行期间 code/lock identity 漂移；禁止 completed")
    return {
        **identity, "schema_version": 1, "kind": kind, "stage": "completed", "task_id": TASK_ID,
        "recipe_id": r.id, "fingerprint": fingerprint(r), "mode": plan["mode"],
        "experiment_id": plan.get("experiment_id", ""), "completed_at": datetime.now(timezone.utc).isoformat(),
        "seed": r.get("train.seed"), "data_seed": r.get("train.data_seed"),
        "model_id": MODEL, "model_revision": r.get("model.revision"),
        "dataset_id": DATASET, "dataset_revision": r.get("dataset.revision"), "source_sha": r.get("source_sha"),
        "recipe_sha256": sha256_json(r.data), "config": config, "config_sha256": sha256_json(config),
        "split_manifest": source_manifest, "source_manifest": source_manifest, "runtime": runtime,
        "baseline_promotion": "none", "training_seed_variance_measured": False,
    }


def _saved_adapter_state(path):
    """只读取 safetensors，禁用不安全 pickle 权重；键和值须另与模型比对。"""
    try:
        load_file = importlib.import_module("safetensors.torch").load_file
        return load_file(str(path / "adapter_model.safetensors"), device="cpu")
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise TaskExecutionError(f"无法安全读取 adapter tensors：{exc}") from exc


def verify_adapter_tensors(model, path, runtime):
    peft, torch = runtime["peft"], runtime["torch"]
    target = getattr(peft, "get_peft_model_state_dict", None)
    if target is None:
        raise TaskExecutionError("PEFT 缺少明确的 adapter state 提取接口")
    kwargs = {"model": model, "adapter_name": "default", "save_embedding_layers": False}
    check_signature(target, kwargs, "get_peft_model_state_dict")
    actual, saved = target(**kwargs), _saved_adapter_state(path)
    if not isinstance(actual, dict) or not actual or set(actual) != set(saved):
        raise TaskExecutionError("adapter tensor keys 不完整或包含多余键；拒绝非严格加载")
    for key in actual:
        if not re.fullmatch(r".+\.lora_[AB]\.weight", key):
            raise TaskExecutionError(f"adapter 包含非 LoRA tensor：{key}")
        a, b = actual[key].detach().cpu(), saved[key].detach().cpu()
        if (tuple(a.shape) != tuple(b.shape) or not bool(torch.isfinite(a).all())
                or not bool(torch.isfinite(b).all()) or not torch.equal(a, b.to(dtype=a.dtype))):
            raise TaskExecutionError(f"adapter tensor shape/值/有限性与已载模型不一致：{key}")
    return {"verified_keys": sorted(actual), "strict_values_verified": True}


def _adapter_structure(path, revision):
    _no_symlinks(path)
    try:
        adapter = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        if not isinstance(adapter, dict):
            raise TaskExecutionError("adapter_config 必须是 JSON 对象")
        if not (path / "tokenizer_config.json").is_file() or not any(
            (path / filename).is_file() and (path / filename).stat().st_size > 0
            for filename in ("adapter_model.safetensors",)
        ):
            raise TaskExecutionError("checkpoint 缺少 adapter weights/tokenizer config；无 base fallback")
        if (adapter.get("base_model_name_or_path") != MODEL or adapter.get("revision") != revision
                or adapter.get("r") != 8 or adapter.get("lora_alpha") != 32
                or adapter.get("lora_dropout") != 0.0 or adapter.get("bias") != "none"
                or adapter.get("task_type") != "CAUSAL_LM" or adapter.get("peft_type") != "LORA"
                or set(adapter.get("target_modules", [])) != set(TARGET_MODULES)):
            raise TaskExecutionError("adapter_config base/revision/QLoRA 绑定不一致")
    except (OSError, ValueError, TypeError) as exc:
        raise TaskExecutionError("checkpoint 缺少有效 adapter_config.json；无 base fallback") from exc


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
    trainer_kwargs = {"model": None, "args": None, "train_dataset": None,
                      "eval_dataset": None, "data_collator": None}
    check_signature(transformers.Trainer, trainer_kwargs, "Trainer")
    _quantization(r, runtime)
    lora_kwargs = _peft_api(runtime, r, lock["model_revision"])
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
    model, lora_evidence = _attach_lora(model, runtime, lora_kwargs)
    trainer_kwargs.update(model=model, args=args,
                          train_dataset=runtime["datasets"].Dataset.from_list(examples["train"]),
                          eval_dataset=runtime["datasets"].Dataset.from_list(examples["validation"]),
                          data_collator=lambda items: collate(items, torch))
    trainer = transformers.Trainer(**trainer_kwargs)
    result = trainer.train()
    actual_steps = trainer.state.global_step
    if type(actual_steps) is not int or actual_steps != canonical["max_steps"]:
        raise TaskExecutionError(f"训练未完成预算：实际 {actual_steps}，要求 {canonical['max_steps']}")
    if execution_identity(r) != identity:
        raise TaskExecutionError("保存前 code/lock identity 漂移")
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    _adapter_structure(output, lock["model_revision"])
    lora_evidence["tensor_reload_check"] = verify_adapter_tensors(model, output, runtime)
    effective_config = {"training_arguments": canonical, "api_training_arguments": kwargs,
                        "peft": r.get("peft"), "quantization": r.get("quantization"),
                        "attention_implementation": "eager", "fim": r.get("fim"),
                        "upstream_metadata": {k: r.get("train." + k) for k in sorted(METADATA_TRAIN_KEYS)},
                        "lora_execution": lora_evidence}
    save_json(output / "effective_recipe.json", {"id": r.id, "original_path": str(r.path.absolute()), "data": r.data})
    save_json(output / "effective_config.json", effective_config)
    save_json(output / "tokenizer_identity.json", tokenizer_identity)
    save_json(output / "source_manifest.json", source_manifest)
    save_json(output / "fim_transform_manifest.json", transform)
    torch.cuda.synchronize()
    evidence = _completed(r, plan, identity, "training", effective_config, source_manifest,
                          runtime_evidence(runtime, started))
    evidence.update(steps=actual_steps, max_steps=canonical["max_steps"], trainer="Trainer",
                    tokenizer_identity=tokenizer_identity, transformation_manifest=transform,
                    train_metrics=dict(result.metrics), model_artifact=artifact_manifest(output))
    # 原子写 completed 前再次检查身份；不能把中途失败目录当 checkpoint。
    if execution_identity(r) != identity:
        raise TaskExecutionError("完成时 code/lock identity 漂移")
    return save_json(output / "training.json", evidence)


def _read_json(path):
    try:
        value = json.loads(_no_symlinks(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("checkpoint JSON 必须为对象")
        return value
    except (OSError, ValueError, TypeError) as exc:
        raise TaskExecutionError(f"缺少有效 checkpoint JSON：{path.name}") from exc


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
    _adapter_structure(path, require_revision(r, "model.revision"))
    from .baseline_evidence import fingerprint
    identity = execution_identity(r)
    if (not isinstance(evidence, dict) or evidence.get("schema_version") != 1
            or evidence.get("kind") != "training" or evidence.get("stage") != "completed"
            or evidence.get("task_id") != TASK_ID or evidence.get("recipe_id") != RECIPE_ID
            or evidence.get("mode") not in ("full", "smoke")
            or (checkpoint is None and evidence.get("mode") != "full")
            or any(evidence.get(k) is True for k in ("mock", "simulated", "dry_run"))):
        raise TaskExecutionError("checkpoint 不是已完成 full/smoke FIM 训练")
    if (evidence.get("fingerprint") != fingerprint(r) or evidence.get("recipe_sha256") != sha256_json(r.data)
            or saved.get("id") != r.id or saved.get("data") != r.data
            or evidence.get("model_id") != MODEL or evidence.get("dataset_id") != DATASET
            or evidence.get("model_revision") != r.get("model.revision")
            or evidence.get("dataset_revision") != r.get("dataset.revision")
            or evidence.get("source_sha") != r.get("source_sha")
            or evidence.get("seed") != r.get("train.seed") or evidence.get("data_seed") != r.get("train.data_seed")
            or any(evidence.get(k) != value for k, value in identity.items())
            or plan.get("execution_identity", identity) != identity):
        raise TaskExecutionError("checkpoint recipe/fingerprint/code/lock/revision/seed 不一致")
    config = _read_json(path / "effective_config.json")
    expected_args = config_kwargs(r, {"recipe_id": r.id, "task_id": TASK_ID, "mode": evidence["mode"]}, path)
    new_api_args = {("eval_strategy" if k == "evaluation_strategy" else k): v for k, v in expected_args.items()}
    lora = config.get("lora_execution", {})
    targets = lora.get("matched_target_modules", {})
    if (set(targets) != set(TARGET_MODULES) or any(
            not isinstance(targets[t], list) or (t in REQUIRED_MODULES and not targets[t])
            or any(not isinstance(n, str) or not (n == t or n.endswith("." + t)) for n in targets[t])
            for t in TARGET_MODULES)
            or type(lora.get("trainable_parameters")) is not int or lora["trainable_parameters"] <= 0
            or lora.get("base_parameters_frozen") is not True):
        raise TaskExecutionError("checkpoint LoRA target/trainable/base freeze 证据缺失")
    expected_identity = training_arguments_identity(expected_args)
    if (evidence.get("config") != config or evidence.get("config_sha256") != sha256_json(config)
            or training_arguments_identity(config.get("training_arguments")) != expected_identity
            or training_arguments_identity(config.get("api_training_arguments")) not in (
                expected_identity, training_arguments_identity(new_api_args))
            or config.get("upstream_metadata") != {k: r.get("train." + k) for k in sorted(METADATA_TRAIN_KEYS)}
            or config.get("fim") != r.get("fim") or config.get("peft") != r.get("peft")
            or config.get("quantization") != r.get("quantization")
            or config.get("attention_implementation") != "eager"
            or type(evidence.get("steps")) is not int or evidence["steps"] != expected_args["max_steps"]
            or evidence.get("max_steps") != expected_args["max_steps"]):
        raise TaskExecutionError("checkpoint steps/config hash/预算不一致")
    source = _read_json(path / "source_manifest.json")
    validate_source_manifest(source, r)
    if evidence.get("source_manifest") != source or evidence.get("split_manifest") != source:
        raise TaskExecutionError("checkpoint source manifest 不一致")
    tokenizer = _read_json(path / "tokenizer_identity.json")
    transform = _read_json(path / "fim_transform_manifest.json")
    if (not isinstance(tokenizer, dict) or tokenizer.get("sha256") != _hash_manifest(tokenizer)["sha256"]
            or evidence.get("tokenizer_identity") != tokenizer
            or transform.get("sha256") != _hash_manifest(transform)["sha256"]
            or transform.get("tokenizer_identity") != tokenizer
            or transform.get("protocol") != PROTOCOL or transform.get("data_seed") != r.get("train.data_seed")
            or evidence.get("transformation_manifest") != transform):
        raise TaskExecutionError("checkpoint tokenizer/transformation manifest 不一致")
    validate_transform_manifest(transform, source, r, tokenizer)
    actual_artifact = artifact_manifest(path)
    if not same_artifact(evidence.get("model_artifact"), actual_artifact):
        raise TaskExecutionError("checkpoint 真实 artifact hash/bytes 不一致")
    return path


def evaluation_examples(splits, tokenizer, r):
    require_supported(r)
    identity = token_identity(tokenizer)
    examples = []
    maximum, sequence = r.get("evaluation.max_new_tokens"), r.get("train.seq_length")
    p, s, m = [identity["special_token_ids"][t] for t in SPECIAL_TOKENS]
    for row in splits["test"]:
        tokens = encode_source(tokenizer, row["content"], identity)
        if len(tokens) < 3:
            raise DatasetValidationError("held-out 文件 token 少于3；不静默跳过题目")
        rng = local_rng(fixed_seeds(r)[1], row["sample_id"], 0, EVAL_PROTOCOL)
        window_size = min(len(tokens), sequence - 3 - maximum)
        start = rng.randrange(len(tokens) - window_size + 1)
        window = tokens[start:start + window_size]
        # 生成预算为 gold 加终止 EOS 预留一位，避免最长 gold 必然无法通过。
        gap_size = rng.randint(1, min(maximum - 1, len(window)))
        lo = rng.randrange(len(window) - gap_size + 1)
        hi = lo + gap_size
        prefix, gold, suffix = window[:lo], window[lo:hi], window[hi:]
        prompt = [p] + prefix + [s] + suffix + [m]
        if not gold or len(prompt) + maximum > sequence or prefix + gold + suffix != window:
            raise DatasetValidationError("PSM token offset/window 无效；不能截断 gold")
        examples.append({
            "sample_id": row["sample_id"], "file_id": row["file_id"],
            "content_sha256": row["content_sha256"], "tokens_sha256": sha256_json(tokens),
            "token_count": len(tokens), "window_start": start, "window_stop": start + window_size,
            "window_sha256": sha256_json(window), "gap_start": lo, "gap_stop": hi,
            "prefix_ids": prefix, "suffix_ids": suffix, "gold_ids": gold,
            "prefix_sha256": sha256_json(prefix), "suffix_sha256": sha256_json(suffix),
            "gold_sha256": sha256_json(gold), "input_ids": prompt, "input_sha256": sha256_json(prompt),
            "gold_text": tokenizer.decode(gold, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        })
    manifest = _hash_manifest({"schema_version": 1, "task_id": TASK_ID, "protocol": EVAL_PROTOCOL,
                               "variant": "psm", "data_seed": r.get("train.data_seed"),
                               "seq_length": sequence, "max_new_tokens": maximum,
                               "tokenizer_identity": identity, "sample_count": len(examples),
                               "samples": examples, "transformed_test_sha256": sha256_json(examples)})
    return examples, manifest


def score_generation(example, generated_ids, tokenizer, maximum):
    eos = tokenizer.eos_token_id
    if not isinstance(generated_ids, list) or any(type(t) is not int or t < 0 for t in generated_ids):
        raise TaskExecutionError("generate 返回非法 token IDs")
    ended = eos in generated_ids
    completion_ids = generated_ids[:generated_ids.index(eos)] if ended else list(generated_ids)
    within_budget = len(generated_ids) <= maximum
    text = tokenizer.decode(completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    gold_text = tokenizer.decode(example["gold_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {
        "generated_ids": generated_ids, "generated_sha256": sha256_json(generated_ids),
        "completion_ids": completion_ids, "completion_sha256": sha256_json(completion_ids),
        "gold_text": gold_text, "completion": text, "eos_observed": ended,
        "eos_token_id": eos, "max_new_tokens": maximum,
        "within_generation_budget": within_budget,
        "fim_token_exact_match": int(ended and within_budget and example["gold_ids"] == completion_ids),
        "fim_exact_match": int(ended and within_budget and gold_text == text),
    }


def recompute_metrics(predictions):
    """从逐题保存的 token/原样文本重算，不信任记录中的 binary 分数。"""
    if not predictions:
        raise TaskExecutionError("评估不能为空")
    counts = {"fim_exact_match": 0, "fim_token_exact_match": 0}
    for row in predictions:
        generated, gold = row["generated_ids"], row["gold_ids"]
        eos, maximum = row["eos_token_id"], row["max_new_tokens"]
        if (not gold or type(eos) is not int or type(maximum) is not int or maximum <= 0
                or any(type(t) is not int or t < 0 for t in generated + gold)
                or row["gold_sha256"] != sha256_json(gold)
                or row["generated_sha256"] != sha256_json(generated)):
            raise TaskExecutionError("预测 token/hash 无效")
        ended = eos in generated
        completion = generated[:generated.index(eos)] if ended else generated
        token_score = int(ended and len(generated) <= maximum and gold == completion)
        text_score = int(ended and len(generated) <= maximum and row["gold_text"] == row["completion"])
        if (row["completion_ids"] != completion or row["completion_sha256"] != sha256_json(completion)
                or row["eos_observed"] != ended or row["within_generation_budget"] != (len(generated) <= maximum)
                or row["fim_token_exact_match"] != token_score or row["fim_exact_match"] != text_score):
            raise TaskExecutionError("预测分数/首 EOS 截断与重算不一致")
        counts["fim_token_exact_match"] += token_score
        counts["fim_exact_match"] += text_score
    return {name: count / len(predictions) for name, count in counts.items()}


def run_evaluation(r, plan, checkpoint=None, zero_shot=False) -> Path:
    lock = _preflight(r, plan)
    mode = "zero-shot" if zero_shot else "eval"
    if plan.get("mode") != mode:
        raise TaskExecutionError("评估 mode 与模型 role 不一致")
    if zero_shot and checkpoint is not None:
        raise TaskExecutionError("zero-shot 只能使用冻结 base，不接受 checkpoint")
    source = None if zero_shot else checkpoint_path(plan, checkpoint, recipe=r)
    output = _output(plan, mode)
    if source is not None and (output == source or output.is_relative_to(source) or source.is_relative_to(output)):
        raise TaskExecutionError("评估目录不能覆盖/包含 checkpoint")
    identity = execution_identity(r)
    if plan.get("execution_identity", identity) != identity:
        raise TaskExecutionError("plan code/lock identity 不一致")
    runtime = load_runtime()
    require_frozen_runtime(r, runtime)
    started = time.perf_counter()
    apply_determinism(r, runtime, evaluation=True)
    splits, source_manifest = load_splits(r, runtime)
    tokenizer, tokenizer_identity, model = _load_base(r, runtime, lock)
    if source is not None:
        saved_tokenizer = runtime["transformers"].AutoTokenizer.from_pretrained(
            str(source), local_files_only=True, trust_remote_code=False,
        )
        if (token_identity(saved_tokenizer) != tokenizer_identity
                or _read_json(source / "tokenizer_identity.json") != tokenizer_identity
                or _read_json(source / "source_manifest.json") != source_manifest):
            raise TaskExecutionError("base/adapter tokenizer 身份或 source split 不一致")
        # base 来源只使用锁 revision；adapter_config 不能重新指定远程 base。
        model = runtime["peft"].PeftModel.from_pretrained(
            model, str(source), is_trainable=False, local_files_only=True,
        )
        verify_adapter_tensors(model, source, runtime)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise TaskExecutionError("评估模型必须全部冻结")
    model.eval()
    examples, eval_manifest = evaluation_examples(splits, tokenizer, r)
    maximum = r.get("evaluation.max_new_tokens")
    generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": maximum,
                  "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.eos_token_id,
                  "use_cache": True}
    torch = runtime["torch"]
    predictions = []
    with torch.inference_mode():
        for example in examples:
            prompt = example["input_ids"]
            input_ids = torch.tensor([prompt], dtype=torch.long, device="cuda:0")
            attention = torch.tensor([[1] * len(prompt)], dtype=torch.long, device="cuda:0")
            generated = model.generate(input_ids=input_ids, attention_mask=attention, **generation)
            output_ids = generated[0].tolist()
            if output_ids[:len(prompt)] != prompt:
                raise TaskExecutionError("generate 未保留 prompt token offset")
            generated_ids = output_ids[len(prompt):]
            predictions.append({**example, **score_generation(example, generated_ids, tokenizer, maximum)})
    torch.cuda.synchronize()
    if source is not None:
        # 同时校验读取前后 adapter/source/训练证据，防止评估期间更换产物。
        checkpoint_path(plan, source, recipe=r)
        model_artifact = {**artifact_manifest(source), "role": "trained_checkpoint"}
    else:
        model_artifact = {"id": MODEL, "revision": lock["model_revision"], "role": "base"}
    config = {"task_id": TASK_ID, "protocol": EVAL_PROTOCOL, "generation": {
        **generation, "seed": r.get("evaluation.seed")}, "quantization": r.get("quantization"),
        "compute_dtype": "bfloat16", "attention_implementation": "eager", "model": model_artifact,
        "tokenizer_identity": tokenizer_identity, "dataset_revision": lock["dataset_revision"],
        "source_manifest_sha256": source_manifest["sha256"],
        "transformed_test_sha256": eval_manifest["transformed_test_sha256"]}
    evidence = _completed(r, plan, identity, "evaluation", config, source_manifest,
                          runtime_evidence(runtime, started))
    save_json(output / "source_manifest.json", source_manifest)
    save_json(output / "tokenizer_identity.json", tokenizer_identity)
    save_json(output / "eval_manifest.json", eval_manifest)
    prediction_path = save_json(output / "predictions.json", predictions)
    evidence.update(evaluation_protocol=EVAL_PROTOCOL, model_role=model_artifact["role"],
                    model_artifact=model_artifact, tokenizer_identity=tokenizer_identity,
                    evaluation_manifest=eval_manifest, sample_count=len(predictions),
                    metrics=recompute_metrics(predictions),
                    metric_counts={"total": len(predictions), **{
                        name: sum(p[name] for p in predictions) for name in r.get("evaluation.metrics")}},
                    predictions={"path": "predictions.json", "count": len(predictions), "sha256": sha256_json(predictions)},
                    predictions_file={"path": "predictions.json", "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
    if execution_identity(r) != identity:
        raise TaskExecutionError("完成评估时 code/lock identity 漂移")
    return save_json(output / "evaluation.json", evidence)
