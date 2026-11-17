"""Granite Vision / Geoperception LineComparison 的受控自建协议 granite-linecomparison-v1。

一手事实（2026-11-14 回 HF Cookbook fine_tuning_granite_vision_sft_trl 核查）与本仓差异：
  * 上游只取 dataset["train"] 中 predicate=="LineComparison"，train_test_split(test_size=0.5,
    seed=42) 得 50/50 train/test，**无 validation**；本仓用原始图像 bytes 的 SHA256 做
    80/10/10 三段隔离，同图所有问答变体落同一 split，并全局拒绝规范化内容重复。
  * 上游 collate_fn 用 labels=input_ids.clone()，仅靠 `<|assistant|>` 位置隐式启停 loss；
    **没有显式 image_token_id 屏蔽**。本仓用 prompt/full 两次 processor 编码做 token 级前缀
    校验，再叠加「所有 image placeholder token 强制 -100」的第二层 mask。
  * 上游用 SFTTrainer + skip_prepare_dataset=True + 动态 target_modules + use_dora=True，
    无 model/dataset/processor revision、无训练 seed、无 compute_metrics；本仓用普通
    transformers.Trainer + 自建 collator，冻结 revision / seed / processor identity，
    并要求 LoRA 命中集合非空且逐项记录。
  * 上游所谓评估只是 test_idx=20 的单样本定性生成；第 5 节文字写 ChartQA，代码从未加载
    ChartQA；文字说 merge adapters，代码只有 PeftModel.from_pretrained。本仓做 held-out
    逐题 greedy + 首个 EOS 截断 + exact_match / normalized_exact_match 重算。
  * 上游从未报告任何数值，本仓指标只衡量文本一致性，**不解释几何语义、不等于人工判分**。

发布级 comparison / export 由 granite_suite.py / granite_export.py 承担，task_router
按精确 recipe_id 注册 granite-linecomparison-v1；本模块只负责 data/train/eval 三维，
不借用其他任务的 gold 重算、套件或导出协议，未知或冲突任务继续失败关闭。
本模块与发布级模块都只在本机无 GPU 环境下做过离线合成测试；任何 completed 证据
都必须来自真实 CUDA 实跑，JSON/hash 一致性不能认证 GPU 执行真实性。
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
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import math_task as mt
from . import xlam_task as xt

TaskExecutionError = mt.TaskExecutionError
DatasetValidationError = TaskExecutionError
sha256_json = mt.sha256_json
fixed_seeds = mt.fixed_seeds
require_revision = mt.require_revision
check_signature = mt.check_signature
apply_determinism = mt.apply_determinism
execution_identity = mt.execution_identity
mode_output_dir = mt.mode_output_dir
save_json = xt.save_json
artifact_manifest = xt.artifact_manifest
file_record = xt.file_record
_read_json = xt._read_json
_no_symlinks = xt._no_symlinks
training_arguments_identity = xt.training_arguments_identity
artifact_identity = xt.artifact_identity
same_artifact = xt.same_artifact

TASK_ID = "granite-linecomparison-v1"
RECIPE_ID = "vlm/sft/granite-vision-line-comparison"
MODEL = MODEL_ID = "ibm-granite/granite-vision-3.1-2b-preview"
DATASET = DATASET_ID = "euclid-multimodal/Geoperception"
PROTOCOL = "granite-linecomparison-v1"
EVAL_PROTOCOL = "granite-linecomparison-v1/held-out-image-masked-v1"
EVAL_VARIANT = "assistant-only-exact-match"
RUNTIME_PACKAGES = ("torch", "transformers", "datasets", "peft", "bitsandbytes", "safetensors")
REQUIRED_COLUMNS = frozenset({"image", "question", "answer", "predicate"})
PREDICATE = "LineComparison"
SYSTEM_MESSAGE = ("A chat between a curious user and an artificial intelligence assistant. "
                  "The assistant gives helpful, detailed, and polite answers to the user's questions.")
METRICS = ("exact_match", "normalized_exact_match")
PREDICTIONS_FILE = "predictions.json"
EVAL_MANIFEST_FILE = "eval_manifest.json"
PROCESSOR_IDENTITY_FILE = "processor_identity.json"
TRANSFORM_FILE = "granite_transform_manifest.json"
REFERENCE_FILES = ("source_manifest.json", PROCESSOR_IDENTITY_FILE, EVAL_MANIFEST_FILE)
FULL_STEPS = 150
SMOKE_STEPS = 5

GRANITE_SPEC = {
    "protocol": PROTOCOL,
    "base_experiment_id": "vlm-sft-granite-vision-line-comparison-v1",
    "source_split": "train",
    "predicate_filter": PREDICATE,
    "columns": sorted(REQUIRED_COLUMNS),
    "split_percent": [80, 10, 10],
    "split_algorithm": "sha256_json({image_sha256,data_seed,protocol}) modulo 100；同图所有问答变体同 split",
    "image_hash": "原始图像 mode + tobytes() 的 SHA256；不做 RGB 转换后再 hash，避免模式/元数据漂移",
    "duplicate_policy": "全局拒绝规范化(question,answer)内容重复；图像 hash 决定 split，禁止同图跨 split 泄漏",
    "normalization": "内容原文保留；去重用 NFKC + casefold + 空白折叠；不 strip 训练文本",
    "loss": "双层 mask：显式 image_token_id 与 assistant span 外 token label=-100；assistant 回答 + EOS 计 loss",
    "prompt_format": "system + user(image+question)；gold 不进生成 prompt",
    "target_format": "assistant 回答 + EOS",
    "chat_template": "processor.apply_chat_template 分别生成 prompt(add_generation_prompt=True) 与完整序列；token 级前缀必须完全一致",
    "image_inputs": "[[image]] 嵌套；非 RGB 显式 convert('RGB')；pixel_values 形状必须全批一致",
    "metric_definition": "exact_match 保留原文全串相等；normalized_exact_match = NFKC+casefold+空白折叠后相等；不解释几何语义、不等于人工判分或业务验收",
    "generation_truncation": "首个 EOS 之后的内容丢弃，再 strip；空生成不得给分",
    "max_eval_examples": 300,
    "max_validation_examples": 300,
    "processor_revision_policy": "AutoProcessor.from_pretrained(base, revision=model.revision)；processor identity 入证据",
    "upstream_quantitative_evaluation": False,
    "upstream_split": "train_test_split(test_size=0.5, seed=42)，无 validation",
    "upstream_mask": "按 assistant span 位置隐式屏蔽；无显式 image_token_id mask",
    "concurrent_scheduling": False,
}

METHOD_KEYS = frozenset({"family", "trainer", "upstream_trainer", "peft", "loss"})
MODEL_KEYS = frozenset({"base", "params_b", "modality", "revision"})
DATASET_KEYS = frozenset({"name", "structure", "type", "kind", "has_images", "revision"})
TRAIN_KEYS = frozenset({
    "seed", "data_seed", "seed_mechanism", "learning_rate", "lr_scheduler_type",
    "weight_decay", "warmup_steps", "per_device_train_batch_size",
    "per_device_eval_batch_size", "gradient_accumulation_steps", "max_steps",
    "seq_length", "fp16", "bf16", "optim", "gradient_checkpointing",
    "dataloader_num_workers", "remove_unused_columns", "eval_steps", "save_steps",
    "logging_steps",
})
METADATA_TRAIN_KEYS = frozenset({"seed", "data_seed", "seed_mechanism", "seq_length"})
# expected / tolerance 是「实跑后回填」，export_tolerance 是「基于实跑的训练前预声明」；
# 与 xLAM / RapidFire 一致允许这三个键存在，但真实 recipe 刻意不预声明。
# 它们不进入 fingerprint，且由 suite_problems / et._tolerance 用当前阈值独立判定，
# 未声明时发布级 --full / B2 门禁一律失败关闭。
EVALUATION_KEYS = frozenset({"metrics", "split", "seed", "max_new_tokens",
                             "held_out", "multi_run", "compute_metrics",
                             "expected", "tolerance", "export_tolerance"})
PEFT_KEYS = frozenset({"r", "alpha", "dropout", "target_module_patterns",
                       "use_dora", "init_lora_weights", "bias", "task_type"})
QUANT_KEYS = frozenset({"load_in_4bit", "bnb_4bit_quant_type",
                        "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype",
                        "llm_int8_skip_modules", "llm_int8_enable_fp32_cpu_offload"})
UPSTREAM_KEYS = frozenset({
    "split", "mask", "evaluation", "revision", "seed", "merge",
    "chatqa_text_code_mismatch", "attn_implementation", "target_modules",
    "trainer", "processing_class",
})
MODALITY_KEYS = frozenset({
    "images_column", "content_list_dict", "processing_class", "max_length",
    "image_mode", "vision_tower",
})


def _canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _hash_manifest(value):
    """与 xlam/rapidfire 一致：返回「原内容 + 其内容 hash」，不是只返回 hash。"""
    payload = {k: v for k, v in value.items() if k != "sha256"}
    return {**value, "sha256": sha256_json(payload)}


def _unknown(section, allowed, label):
    if not isinstance(section, dict):
        raise TaskExecutionError(f"[{label}] 缺失或不是表")
    extra = sorted(set(section) - set(allowed))
    if extra:
        raise TaskExecutionError(f"[{label}] 出现未声明字段 {extra}；拒绝静默套默认")


def resolve_task_id(r, plan=None, task_id=None):
    declared = task_id or (plan or {}).get("task_id") or r.get("execution.task_id")
    return declared or (TASK_ID if r.id == RECIPE_ID else None)


def require_supported(r, plan=None, task_id=None) -> str:
    """严格校验 recipe / plan / 协议 / VLM 输入 / 依赖 / 预算；任何漂移都失败关闭。"""
    resolved = resolve_task_id(r, plan, task_id)
    if r.id != RECIPE_ID or resolved != TASK_ID or (plan or {}).get("recipe_id", r.id) != r.id:
        raise TaskExecutionError(f"尚未实现该执行路由：{r.id} / {resolved!r}")
    required = {
        "method.family": "sft", "method.trainer": "Trainer",
        "method.upstream_trainer": "SFTTrainer", "method.peft": "qlora-dora",
        "model.base": MODEL, "dataset.name": DATASET,
    }
    for key, expected in required.items():
        if r.get(key) != expected:
            raise TaskExecutionError(f"{TASK_ID} 要求 {key}={expected!r}")
        section, field = key.split(".")
        planned = (plan or {}).get(section, {})
        if field in planned and planned[field] != expected:
            raise TaskExecutionError(f"plan {key} 与 Granite 路由不一致")
    if not r.is_vlm or r.get("dataset.has_images") is not True or r.get("rewards"):
        raise TaskExecutionError("Granite 只接受带图像的静态 VLM 数据，不使用 reward")
    if r.get("decision.gate") == "blocked" or r.get("baseline.result") == "passes":
        raise TaskExecutionError("训练决策门已阻断：先复核零训练基线与业务目标")
    for name, keys in (("method", METHOD_KEYS), ("model", MODEL_KEYS),
                       ("dataset", DATASET_KEYS), ("budget", {"full_steps", "smoke_steps"})):
        _unknown(r.get(name), keys, name)
    if r.get("dataset.kind", "static") != "static":
        raise TaskExecutionError("Granite 只接受冻结的静态图像数据集")
    for name, spec in (("granite", GRANITE_SPEC), ("peft", PEFT_KEYS),
                       ("quantization", QUANT_KEYS), ("upstream", UPSTREAM_KEYS)):
        section = r.get(name)
        if name == "granite":
            _unknown(section, spec, name)
            for key, expected in spec.items():
                if key not in section or section[key] != expected or type(section[key]) is not type(expected):
                    raise TaskExecutionError(f"granite.{key} 必须为 {expected!r}")
        else:
            _unknown(section, spec, name)
    granite = r.get("granite")
    if (granite["predicate_filter"] != PREDICATE or granite["columns"] != sorted(REQUIRED_COLUMNS)
            or granite["split_percent"] != [80, 10, 10]):
        raise TaskExecutionError("granite 过滤/列/划分协议漂移")
    peft = r.get("peft")
    if (peft["use_dora"] is not True or peft["init_lora_weights"] != "gaussian"
            or peft["target_module_patterns"] != ["language_model", "_proj"]):
        raise TaskExecutionError("peft 必须保留上游 DoRA/gaussian 事实与动态筛选 pattern")
    quant = r.get("quantization")
    if (quant["load_in_4bit"] is not True or quant["bnb_4bit_quant_type"] != "nf4"
            or quant["bnb_4bit_use_double_quant"] is not True
            or quant["bnb_4bit_compute_dtype"] != "bfloat16"
            or quant["llm_int8_skip_modules"] != ["vision_tower", "lm_head"]
            or quant["llm_int8_enable_fp32_cpu_offload"] is not True):
        raise TaskExecutionError("quantization 必须与一手 BitsAndBytesConfig 一致（含视觉塔跳过量化）")
    upstream = r.get("upstream")
    for key in UPSTREAM_KEYS:
        if not isinstance(upstream.get(key), str) or not upstream[key].strip():
            raise TaskExecutionError(f"upstream.{key} 必须保留一手事实原文")
    modality = r.get("modality_contract")
    _unknown(modality, MODALITY_KEYS, "modality_contract")
    if modality is None:
        raise TaskExecutionError("modality_contract 缺失：VLM 图文合同不可静默降级")
    if (modality.get("images_column") != "image"
            or modality.get("processing_class") != "processor"
            or modality.get("image_mode") != "RGB"
            or modality.get("content_list_dict") is not True
            or modality.get("vision_tower") != "skipped_from_quantization"):
        raise TaskExecutionError("modality_contract 必须保留 processor/图像列/RGB/视觉塔跳过量化合同")
    train = r.get("train")
    _unknown(train, TRAIN_KEYS, "train")
    missing = sorted(TRAIN_KEYS - set(train))
    if missing:
        raise TaskExecutionError(f"[train] 缺少参数 {missing}")
    _unknown(r.get("evaluation"), EVALUATION_KEYS, "evaluation")
    for key, expected in {
        "train.bf16": True, "train.fp16": False, "train.gradient_checkpointing": True,
        "train.remove_unused_columns": False, "train.lr_scheduler_type": "linear",
        "evaluation.metrics": list(METRICS), "evaluation.held_out": True,
        "evaluation.multi_run": True, "evaluation.compute_metrics": True,
    }.items():
        if r.get(key) != expected or (isinstance(expected, bool) and type(r.get(key)) is not bool):
            raise TaskExecutionError(f"{key} 必须为 {expected!r}")
    fixed_seeds(r)
    if not r.has("evaluation.seed"):
        raise TaskExecutionError("evaluation.seed 必须显式固定")
    if not isinstance(train["seed_mechanism"], str) or not train["seed_mechanism"].strip():
        raise TaskExecutionError("seed_mechanism 必须保留上游无训练 seed 的元信息")
    for key in ("per_device_train_batch_size", "per_device_eval_batch_size",
                "gradient_accumulation_steps", "eval_steps", "save_steps", "logging_steps"):
        if type(train[key]) is not int or train[key] <= 0:
            raise TaskExecutionError(f"train.{key} 必须为正整数")
    if type(train["dataloader_num_workers"]) is not int or train["dataloader_num_workers"] < 0:
        raise TaskExecutionError("train.dataloader_num_workers 必须为非负整数")
    for key in ("learning_rate", "weight_decay"):
        value = train[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise TaskExecutionError(f"train.{key} 必须为有限非负数")
    if train["learning_rate"] == 0:
        raise TaskExecutionError("learning_rate 必须大于零")
    sequence, new = train["seq_length"], r.get("evaluation.max_new_tokens")
    if type(new) is not int or new < 2 or type(sequence) is not int or sequence < new + 6:
        raise TaskExecutionError("seq_length 必须为整数且至少 max_new_tokens+6")
    if (type(train["max_steps"]) is not int or train["max_steps"] != FULL_STEPS
            or r.get("budget.full_steps") != FULL_STEPS or r.get("budget.smoke_steps") != SMOKE_STEPS):
        raise TaskExecutionError("预算必须显式 full=max_steps=150 / smoke=5；不可静默改预算")
    dependencies = r.get("dependencies", {}) or {}
    _unknown(dependencies, RUNTIME_PACKAGES, "dependencies")
    for name, version in dependencies.items():
        if not isinstance(version, str) or not version.strip() or version in ("unknown", "latest", "*"):
            raise TaskExecutionError(f"dependencies.{name} 不能猜测或使用未冻结版本")
    return resolved


def split_spec(r):
    return {**copy.deepcopy(r.get("granite")), "data_seed": fixed_seeds(r)[1],
            "selection": "有限完整扫描，保留 source_index；无 shuffle、无补救划分",
            "gold_policy": "answer 只进入训练 full 序列与评估 gold，绝不进入生成 prompt"}


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
        result["dataset"] = {**result.get("dataset", {}), "split": "image-hash 80/10/10",
                             "split_plan": split_spec(r)}
        result["evaluation"] = {
            "metrics": list(METRICS), "seed": r.get("evaluation.seed"),
            "max_new_tokens": r.get("evaluation.max_new_tokens"), "held_out": True,
            "multi_run": True, "runs": 1, "protocol": EVAL_PROTOCOL, "do_sample": False,
            "training_seed_variance_measured": False,
            "upstream_quantitative_evaluation": False,
            "generated_code_executed": False,
        }
        result["migration"] = {
            "trainer": "Trainer + 自建 collator（上游 SFTTrainer + skip_prepare_dataset）",
            "split": "image-hash 80/10/10（上游只有 50/50，且 eval[50:60] 无 validation）",
            "mask": "显式 image_token_id + assistant-only 双层 -100（上游只按 assistant 位置隐式 mask）",
            "evaluation": "held-out 逐题 exact_match（上游只有 test_idx=20 单样本定性，正文写 ChartQA 但代码未加载）",
            "revision": "model/dataset/processor revision 与训练 seed 全部显式冻结（上游全部缺失）",
            "license": "基座/数据许可与商业使用未确认；上游页面漂移未定位（source_sha 为空）",
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
    params.update(max_steps=steps, output_dir=str(output_dir), report_to="none",
                  push_to_hub=False, evaluation_strategy="steps", save_strategy="steps",
                  logging_strategy="steps")
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

def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def image_payload(image):
    """返回 (mode, raw_bytes)，hash 只基于原始输入；RGB 转换发生在送入 processor 之前。"""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return "bytes", bytes(image)
    mode = getattr(image, "mode", None)
    tobytes = getattr(image, "tobytes", None)
    if not isinstance(mode, str) or not callable(tobytes):
        raise DatasetValidationError("image 必须是 bytes 或提供 mode/tobytes() 的图像对象")
    return mode, bytes(tobytes())


def ensure_rgb(image):
    """非 RGB 显式转换；不改变用于划分 hash 的原始 bytes。"""
    mode = getattr(image, "mode", "bytes")
    if mode == "RGB" or mode == "bytes":
        return image
    convert = getattr(image, "convert", None)
    if not callable(convert):
        raise DatasetValidationError("非 RGB 图像必须提供 convert('RGB')")
    return convert("RGB")


def parse_row(raw, index):
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"source_index={index} 不是映射行")
    extra = sorted(set(raw) - REQUIRED_COLUMNS)
    if extra:
        raise DatasetValidationError(f"source_index={index} 出现未知列 {extra}")
    missing = sorted(REQUIRED_COLUMNS - set(raw))
    if missing:
        raise DatasetValidationError(f"source_index={index} 缺列 {missing}")
    question, answer, predicate = raw["question"], raw["answer"], raw["predicate"]
    if not isinstance(question, str) or not question.strip():
        raise DatasetValidationError(f"source_index={index} question 为空")
    if not isinstance(answer, str) or not answer.strip():
        raise DatasetValidationError(f"source_index={index} answer(gold) 为空")
    if predicate != PREDICATE:
        raise DatasetValidationError(f"source_index={index} predicate={predicate!r} 不属于本任务子集")
    mode, payload = image_payload(raw["image"])
    image_sha256 = hashlib.sha256(mode.encode("utf-8") + b"\x00" + payload).hexdigest()
    content = normalized_text(question) + "\x00" + normalized_text(answer)
    return {
        "source_index": index, "question": question, "answer": answer,
        "predicate": predicate, "image": raw["image"], "image_mode": mode,
        "image_sha256": image_sha256,
        "image_bytes": len(payload),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "sample_id": f"{image_sha256[:16]}-{index}",
    }


def split_bucket(image_sha256, data_seed):
    digest = hashlib.sha256(f"{PROTOCOL}|{data_seed}|{image_sha256}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def prepare_splits(rows, r):
    """有限完整扫描 + 图像级隔离 + 内容级去重；空划分拒绝，不随机补救。"""
    require_supported(r)
    _, data_seed = fixed_seeds(r)
    splits = {"train": [], "validation": [], "test": []}
    seen, image_owner, duplicates = {}, {}, []
    for index, raw in enumerate(rows):
        row = parse_row(raw, index)
        if row["content_sha256"] in seen:
            duplicates.append({"first": seen[row["content_sha256"]], "again": index,
                               "sha256": row["content_sha256"]})
            continue
        seen[row["content_sha256"]] = index
        bucket = split_bucket(row["image_sha256"], data_seed)
        previous = image_owner.get(row["image_sha256"])
        if previous is not None and previous != bucket:
            raise DatasetValidationError("同一图像被分到不同 split；划分规则或 hash 漂移")
        image_owner[row["image_sha256"]] = bucket
        splits[bucket].append(row)
    if duplicates:
        raise DatasetValidationError(
            f"检测到 {len(duplicates)} 条规范化内容重复（含跨 split）；拒绝静默去重，先复核数据源与 revision")
    hashes = {name: [row["image_sha256"] for row in items] for name, items in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(hashes[left]) & set(hashes[right]):
            raise DatasetValidationError("同图跨 split 泄漏；图像内容 hash 隔离失败")
    for name, items in splits.items():
        if not items:
            raise DatasetValidationError(f"{name} split 为空；拒绝用其他 split 偷偷补救")
    max_eval = r.get("granite.max_eval_examples")
    max_val = r.get("granite.max_validation_examples")
    if type(max_eval) is not int or max_eval < 1 or type(max_val) is not int or max_val < 1:
        raise DatasetValidationError("max_eval_examples / max_validation_examples 必须为正整数")
    manifest = {
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "dataset_id": DATASET, "data_seed": data_seed, "source_split": "train",
        "predicate_filter": PREDICATE,
        "split_algorithm": r.get("granite.split_algorithm"),
        "split_percent": [80, 10, 10],
        "duplicate_policy": r.get("granite.duplicate_policy"),
        "image_hash": r.get("granite.image_hash"),
        "normalization": r.get("granite.normalization"),
        "duplicates_rejected": duplicates,
        "image_split_owner": {h: split_bucket(h, data_seed) for h in image_owner},
        "counts": {"source": len(seen), "train": len(splits["train"]),
                   "validation": len(splits["validation"]), "test": len(splits["test"]),
                   "validation_eval_used": min(len(splits["validation"]), max_val),
                   "test_eval_used": min(len(splits["test"]), max_eval)},
        "split_hashes": hashes,
        "gold_policy": "answer 只进入训练 full 序列与评估 gold，绝不进入生成 prompt",
    }
    manifest["split_hashes_sha256"] = sha256_json(manifest["split_hashes"])
    manifest["sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    return {"train": splits["train"], "validation": splits["validation"][:max_val],
            "test": splits["test"][:max_eval]}, manifest


def validate_source_manifest(manifest, r):
    if not isinstance(manifest, dict):
        raise DatasetValidationError("source_manifest 缺失")
    computed = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    if (manifest.get("sha256") != computed or manifest.get("schema_version") != 1
            or manifest.get("protocol") != PROTOCOL or manifest.get("task_id") != TASK_ID
            or manifest.get("dataset_id") != DATASET
            or manifest.get("data_seed") != r.get("train.data_seed")
            or manifest.get("predicate_filter") != PREDICATE
            or manifest.get("split_hashes_sha256") != sha256_json(manifest.get("split_hashes"))):
        raise DatasetValidationError("source_manifest 内容身份 / 划分 hash 不一致")
    hashes, counts = manifest.get("split_hashes"), manifest.get("counts")
    if not isinstance(hashes, dict) or set(hashes) != {"train", "validation", "test"} or not isinstance(counts, dict):
        raise DatasetValidationError("source_manifest 缺少 split_hashes / counts")
    for name in ("train", "validation", "test"):
        values = hashes[name]
        if (not isinstance(values, list) or not values or counts.get(name) != len(values)
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in values)):
            raise DatasetValidationError(f"source_manifest.{name} 图像 hash 列表非法")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(hashes[left]) & set(hashes[right]):
            raise DatasetValidationError("source_manifest 存在同图跨 split 泄漏")
    for name, values in hashes.items():
        for h in values:
            if split_bucket(h, manifest["data_seed"]) != name:
                raise DatasetValidationError(f"source_manifest.{name} hash 与确定性划分规则不一致")
    owner = manifest.get("image_split_owner")
    if not isinstance(owner, dict) or any(owner.get(h) != name for name, values in hashes.items() for h in values):
        raise DatasetValidationError("source_manifest 图像归属与划分不一致")
    return manifest


def load_splits(r, dependencies):
    revision = require_revision(r, "dataset.revision")
    dataset = dependencies["datasets"].load_dataset(DATASET, revision=revision, split="train")
    splits, manifest = prepare_splits(dataset, r)
    manifest["dataset_revision"] = revision
    manifest["sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    return splits, manifest


# ---------------------------------------------------------------- 图文编码

def processor_identity(processor):
    tokenizer = processor.tokenizer
    image_token_id = getattr(processor, "image_token_id", None)
    if type(image_token_id) is not int:
        raise DatasetValidationError("processor 缺少显式 image_token_id；拒绝退回上游的隐式位置 mask")
    template = str(getattr(tokenizer, "chat_template", "") or "")
    payload = {
        "name_or_path": str(getattr(processor, "name_or_path", "") or getattr(tokenizer, "name_or_path", "")),
        "tokenizer": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "image_token": str(getattr(processor, "image_token", "")),
        "image_token_id": image_token_id,
        "chat_template_sha256": hashlib.sha256(str(template).encode("utf-8")).hexdigest(),
        "protocol": PROTOCOL,
    }
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


def render_messages(row, include_assistant=True):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "image", "image": row["image"]},
                                     {"type": "text", "text": row["question"]}]},
    ]
    if include_assistant:
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": row["answer"]}]})
    return messages


def render_prompt(row, processor):
    text = processor.apply_chat_template(render_messages(row, include_assistant=False),
                                        tokenize=False, add_generation_prompt=True)
    if not isinstance(text, str) or not text:
        raise DatasetValidationError("processor.apply_chat_template 未返回有效 prompt 文本")
    if row["answer"] in text:
        raise DatasetValidationError("gold 泄漏进生成 prompt；拒绝执行")
    return text


def _processor_call(processor, text, image):
    result = processor(text=text, images=[[ensure_rgb(image)]],
                       return_tensors=None, padding=False)
    if not isinstance(result, dict) or "input_ids" not in result:
        raise DatasetValidationError("processor 未返回 input_ids")
    ids = result["input_ids"]
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    ids = [int(token) for token in ids]
    return ids, result


def encode_example(row, processor, identity, r):
    """prompt/full 两次编码 + token 级前缀校验 + 显式 image token 屏蔽。"""
    prompt = render_prompt(row, processor)
    full = processor.apply_chat_template(render_messages(row, include_assistant=True),
                                        tokenize=False, add_generation_prompt=False)
    if not isinstance(full, str) or row["answer"] not in full:
        raise DatasetValidationError("完整 chat 序列不含 gold；拒绝组装 completion-only 标签")
    prompt_ids, _ = _processor_call(processor, prompt, row["image"])
    full_ids, encoded = _processor_call(processor, full, row["image"])
    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        raise DatasetValidationError("prompt/full token 边界非法；拒绝截断")
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise DatasetValidationError("full 序列不是 prompt 的 token 前缀；无法确定 assistant-only 边界")
    limit = r.get("train.seq_length")
    if len(full_ids) > limit:
        raise DatasetValidationError(
            f"source_index={row['source_index']} 长度 {len(full_ids)} 超过 seq_length={limit}；拒绝截断")
    image_token_id = identity["image_token_id"]
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    # 第二层：无论 image placeholder 落在哪个位置，一律不参与语言损失。
    masked_images = sum(1 for token in full_ids if token == image_token_id)
    if masked_images == 0:
        raise DatasetValidationError("序列中没有 image placeholder token；图像输入可能未被处理器展开")
    for index, token in enumerate(full_ids):
        if token == image_token_id:
            labels[index] = -100
    pixels = encoded.get("pixel_values")
    shape = _shape(pixels)
    if not shape:
        raise DatasetValidationError("processor 未返回有效 pixel_values")
    return {
        "input_ids": full_ids, "labels": labels,
        "attention_mask": [1] * len(full_ids),
        "pixel_values": _canonical(pixels), "pixel_shape": list(shape),
        "prompt_ids": prompt_ids, "prompt_text": prompt,
        "prompt_tokens": len(prompt_ids), "target_tokens": len(full_ids) - len(prompt_ids),
        "masked_image_tokens": masked_images,
        "sample_id": row["sample_id"], "image_sha256": row["image_sha256"],
        "gold_sha256": sha256_json(row["answer"]), "answer": row["answer"],
    }


def _shape(value):
    if isinstance(value, (list, tuple)) and value:
        head = value[0]
        inner = _shape(head) if isinstance(head, (list, tuple)) else ()
        return (len(value),) + inner
    return ()


def collate(items, torch, pad_token_id, image_token_id):
    """右 padding；labels/padding/image token 全部 -100；pixel_values 形状必须一致。"""
    if not items:
        raise DatasetValidationError("空 batch")
    shapes = {tuple(item["pixel_shape"]) for item in items}
    if len(shapes) != 1:
        raise DatasetValidationError("batch 内 pixel_values 形状不一致；需冻结统一图像处理配置")
    width = max(len(item["input_ids"]) for item in items)
    batch = {
        "input_ids": [item["input_ids"] + [pad_token_id] * (width - len(item["input_ids"])) for item in items],
        "labels": [item["labels"] + [-100] * (width - len(item["labels"])) for item in items],
        "attention_mask": [item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in items],
    }
    for row_index, row in enumerate(batch["input_ids"]):
        for index, token in enumerate(row):
            if token == image_token_id:
                batch["labels"][row_index][index] = -100
    if any(len(v) != width for row in batch.values() for v in row):
        raise DatasetValidationError("collate 长度不一致")
    out = {key: torch.tensor(batch[key], dtype=torch.long) for key in ("input_ids", "labels", "attention_mask")}
    out["pixel_values"] = torch.tensor([item["pixel_values"] for item in items], dtype=torch.float32)
    return out


def train_examples(splits, processor, r):
    require_supported(r)
    identity = processor_identity(processor)
    examples, entries = {}, {}
    for name in ("train", "validation"):
        examples[name], entries[name] = [], []
        for row in splits[name]:
            item = encode_example(row, processor, identity, r)
            examples[name].append({key: item[key] for key in
                                   ("input_ids", "labels", "attention_mask",
                                    "pixel_values", "pixel_shape")})
            entries[name].append({
                **{key: item[key] for key in
                   ("sample_id", "image_sha256", "gold_sha256", "prompt_tokens",
                    "target_tokens", "masked_image_tokens")},
                "source_index": row["source_index"],
                # 双层 mask 后的真实训练样本也要有内容身份，避免改了 input_ids/labels
                # 却仍用旧 transform manifest 冒充同一批训练输入。
                "example_sha256": sha256_json(examples[name][-1]),
            })
        if not examples[name]:
            raise DatasetValidationError(f"{name} 编码后为空；拒绝静默跳过")
    transform = _hash_manifest({
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "processor_identity": identity, "data_seed": r.get("train.data_seed"),
        "loss": r.get("granite.loss"), "image_inputs": r.get("granite.image_inputs"),
        "entries": entries,
    })
    return examples, transform


def validate_transform_manifest(transform, source, r, identity):
    if (not isinstance(transform, dict) or transform.get("task_id") != TASK_ID
            or transform.get("protocol") != PROTOCOL
            or transform.get("processor_identity") != identity
            or transform.get("loss") != r.get("granite.loss")
            or transform.get("image_inputs") != r.get("granite.image_inputs")
            or transform.get("data_seed") != r.get("train.data_seed")):
        raise TaskExecutionError("transform manifest 与当前图文协议不一致")
    entries = transform.get("entries")
    if not isinstance(entries, dict) or set(entries) != {"train", "validation"}:
        raise TaskExecutionError("transform manifest 缺少 train/validation 条目")
    for name in ("train", "validation"):
        hashes = source["split_hashes"][name]
        rows = entries[name]
        if not isinstance(rows, list) or len(rows) > len(hashes):
            raise TaskExecutionError(f"transform manifest.{name} 条目超出 split 内容")
        for row, expected in zip(rows, hashes):
            if (row.get("image_sha256") != expected or type(row.get("masked_image_tokens")) is not int
                    or row["masked_image_tokens"] < 1 or type(row.get("target_tokens")) is not int
                    or row["target_tokens"] < 1
                    or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("example_sha256", "")))):
                raise TaskExecutionError("transform manifest 条目缺少图像 hash / 双层 mask / 训练样本内容身份")
    if transform.get("sha256") != _hash_manifest(transform)["sha256"]:
        raise TaskExecutionError("transform manifest 内容 hash 不一致")
    return transform


# ---------------------------------------------------------------- 评估协议

def clean_generation(text, eos_token):
    if not isinstance(text, str):
        return ""
    if eos_token:
        text = text.split(eos_token, 1)[0]
    return text.strip()


def score_prediction(row, text, eos_token):
    prediction = clean_generation(text, eos_token)
    exact = 1.0 if prediction == row["answer"] else 0.0
    normalized = 1.0 if normalized_text(prediction) == normalized_text(row["answer"]) else 0.0
    return {"exact_match": exact, "normalized_exact_match": normalized,
            "empty": 0.0 if prediction else 1.0,
            "prediction": prediction, "gold_sha256": sha256_json(row["answer"])}


def recompute_metrics(predictions):
    if not isinstance(predictions, list) or not predictions:
        raise DatasetValidationError("逐题预测为空，不能报告指标")
    for index, record in enumerate(predictions):
        # 逐题记录必须是从 eval_manifest 重建的评分 dict；字符串/None 混入时
        # 不能静默按 0 分处理，也不能让 AttributeError 绕过失败关闭。
        if not isinstance(record, dict):
            raise DatasetValidationError(f"第 {index} 条逐题预测不是 dict；拒绝重算指标")
        for name in METRICS:
            value = record.get(name)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise DatasetValidationError(f"逐题 {name} 非法")
    return {name: sum(row[name] for row in predictions) / len(predictions) for name in METRICS}


def evaluation_examples(rows, processor, identity, r):
    examples = []
    for row in rows:
        prompt = render_prompt(row, processor)
        prompt_ids, encoded = _processor_call(processor, prompt, row["image"])
        examples.append({
            "sample_id": row["sample_id"], "source_index": row["source_index"],
            "image_sha256": row["image_sha256"], "question": row["question"],
            "answer": row["answer"], "prompt": prompt, "prompt_ids": prompt_ids,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "gold_sha256": sha256_json(row["answer"]),
            "pixel_values": _canonical(encoded.get("pixel_values")),
            "pixel_shape": list(_shape(encoded.get("pixel_values"))),
        })
    return examples


def evaluation_manifest(samples, identity, r, source):
    payload = {
        "schema_version": 1, "kind": "granite-eval-manifest", "task_id": TASK_ID,
        "protocol": EVAL_PROTOCOL, "variant": EVAL_VARIANT,
        "processor_identity": identity,
        "dataset_id": DATASET, "data_seed": r.get("train.data_seed"),
        "source_manifest_sha256": source["sha256"],
        "max_new_tokens": r.get("evaluation.max_new_tokens"),
        "do_sample": False, "num_beams": 1,
        "metric_definition": r.get("granite.metric_definition"),
        "generation_truncation": r.get("granite.generation_truncation"),
        "samples": samples,
    }
    payload["samples_sha256"] = sha256_json(samples)
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


# ---------------------------------------------------------------- 执行框架

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
        "images_per_sample": 1,
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
    spec = r.get("quantization")
    kwargs = {"load_in_4bit": spec["load_in_4bit"],
              "bnb_4bit_quant_type": spec["bnb_4bit_quant_type"],
              "bnb_4bit_compute_dtype": runtime["torch"].bfloat16,
              "bnb_4bit_use_double_quant": spec["bnb_4bit_use_double_quant"],
              "llm_int8_skip_modules": list(spec["llm_int8_skip_modules"]),
              "llm_int8_enable_fp32_cpu_offload": spec["llm_int8_enable_fp32_cpu_offload"]}
    target = runtime["transformers"].BitsAndBytesConfig
    check_signature(target, kwargs, "BitsAndBytesConfig")
    config = target(**kwargs)
    for key, value in kwargs.items():
        if getattr(config, key, None) != value:
            raise TaskExecutionError(f"BitsAndBytesConfig 未应用 {key}")
    return config


def _load_base(r, runtime, lock, *, training=False):
    transformers = runtime["transformers"]
    processor = transformers.AutoProcessor.from_pretrained(
        MODEL, revision=lock["model_revision"], trust_remote_code=False)
    tokenizer = processor.tokenizer
    if tokenizer.eos_token is None or tokenizer.pad_token is None:
        raise TaskExecutionError("processor.tokenizer 缺少 eos/pad token；拒绝静默替换配置")
    identity = processor_identity(processor)
    if identity["image_token_id"] is None:
        raise TaskExecutionError("processor 没有显式 image_token_id")
    model = transformers.AutoModelForVision2Seq.from_pretrained(
        MODEL, revision=lock["model_revision"], trust_remote_code=False,
        quantization_config=_quantization(r, runtime),
        torch_dtype=runtime["torch"].bfloat16,
        attn_implementation="eager", device_map={"": 0})
    model.config.use_cache = not training
    return processor, identity, model


def _peft_kwargs(r, revision, target_modules):
    peft = r.get("peft")
    return {"r": peft["r"], "lora_alpha": peft["alpha"], "lora_dropout": peft["dropout"],
            "target_modules": list(target_modules), "use_dora": peft["use_dora"],
            "init_lora_weights": peft["init_lora_weights"], "bias": peft["bias"],
            "task_type": peft["task_type"], "base_model_name_or_path": MODEL,
            "revision": revision}


def _attach_lora(model, runtime, kwargs):
    peft = runtime["peft"]
    check_signature(peft.LoraConfig, kwargs, "LoraConfig")
    check_signature(peft.get_peft_model, {"model": None, "peft_config": None}, "get_peft_model")
    check_signature(peft.prepare_model_for_kbit_training,
                    {"model": None, "use_gradient_checkpointing": True},
                    "prepare_model_for_kbit_training")
    targets = kwargs["target_modules"]
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise TaskExecutionError("动态 LoRA target 命中过少或重复；拒绝空命中")
    named = {name for name, _ in model.named_modules()}
    if not set(targets) <= named:
        raise TaskExecutionError("LoRA target 未全部出现在 named_modules()")
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
    return model, {"matched_target_modules": sorted(targets),
                   "trainable_parameters": trainable, "base_parameters_frozen": True,
                   "use_dora": True}


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
        if not re.fullmatch(r".+\.lora_(?:[AB]|magnitude_vector)\.weight", key):
            raise TaskExecutionError(f"adapter 含非 LoRA/DoRA tensor：{key}")
        a, b = actual[key].detach().cpu(), saved[key].detach().cpu()
        if (tuple(a.shape) != tuple(b.shape) or not bool(torch.isfinite(a).all())
                or not bool(torch.isfinite(b).all()) or not torch.equal(a, b.to(dtype=a.dtype))):
            raise TaskExecutionError(f"adapter tensor shape/值/有限性不一致：{key}")
    return {"verified_keys": sorted(actual), "strict_values_verified": True}


def _adapter_structure(path, r, targets=None):
    _no_symlinks(path)
    adapter = _read_json(path / "adapter_config.json")
    peft = r.get("peft")
    if not (path / "processor_config.json").is_file() or not (
            path / "adapter_model.safetensors").is_file():
        raise TaskExecutionError("checkpoint 缺少 adapter weights / processor config；无 base fallback")
    expected = {"base_model_name_or_path": MODEL, "revision": r.get("model.revision"),
                "r": peft["r"], "lora_alpha": peft["alpha"], "lora_dropout": peft["dropout"],
                "bias": peft["bias"], "task_type": peft["task_type"],
                "use_dora": True, "init_lora_weights": peft["init_lora_weights"]}
    for key, value in expected.items():
        if adapter.get(key) != value:
            raise TaskExecutionError(f"adapter_config {key} 与 recipe 不一致")
    if targets is not None and set(adapter.get("target_modules", [])) != set(targets):
        raise TaskExecutionError("adapter_config target_modules 与训练时动态命中集合不一致")


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


def base_artifact(r):
    return {"id": MODEL, "revision": r.get("model.revision"), "role": "base"}


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
    processor, processor_identity_value, model = _load_base(r, runtime, lock, training=True)
    examples, transform = train_examples(splits, processor, r)
    validate_transform_manifest(transform, source_manifest, r, processor_identity_value)
    patterns = list(r.get("peft.target_module_patterns"))
    targets = sorted({name for name, _ in model.named_modules()
                      if all(pattern in name for pattern in patterns)})
    if not targets:
        raise TaskExecutionError("动态 LoRA target 命中为空；拒绝用默认模块名冒充")
    model, lora_evidence = _attach_lora(
        model, runtime, _peft_kwargs(r, lock["model_revision"], targets))
    collator = lambda items: collate(  # noqa: E731
        items, torch, processor_identity_value["pad_token_id"],
        processor_identity_value["image_token_id"])
    trainer = transformers.Trainer(
        model=model, args=args,
        train_dataset=runtime["datasets"].Dataset.from_list(examples["train"]),
        eval_dataset=runtime["datasets"].Dataset.from_list(examples["validation"]),
        data_collator=collator)
    result = trainer.train()
    actual_steps = trainer.state.global_step
    if type(actual_steps) is not int or actual_steps != canonical["max_steps"]:
        raise TaskExecutionError(f"训练未完成预算：实际 {actual_steps}，要求 {canonical['max_steps']}")
    if execution_identity(r) != identity:
        raise TaskExecutionError("保存前 code/lock identity 漂移")
    model.save_pretrained(str(output), safe_serialization=True)
    processor.save_pretrained(str(output))
    _adapter_structure(output, r, targets)
    lora_evidence["tensor_reload_check"] = verify_adapter_tensors(model, output, runtime)
    effective_config = {
        "training_arguments": canonical, "api_training_arguments": kwargs,
        "peft": r.get("peft"), "quantization": r.get("quantization"),
        "attention_implementation": "eager", "granite": r.get("granite"),
        "upstream_metadata": dict(r.get("upstream")),
        "lora_execution": lora_evidence,
        "processor_identity": processor_identity_value,
        "image_token_id": processor_identity_value["image_token_id"],
    }
    save_json(output / "effective_recipe.json",
              {"id": r.id, "original_path": str(r.path.absolute()), "data": r.data})
    save_json(output / "effective_config.json", effective_config)
    save_json(output / PROCESSOR_IDENTITY_FILE, processor_identity_value)
    save_json(output / "source_manifest.json", source_manifest)
    save_json(output / TRANSFORM_FILE, transform)
    torch.cuda.synchronize()
    evidence = _completed(r, plan, identity, "training", effective_config, source_manifest,
                          runtime_evidence(runtime, started))
    evidence.update(steps=actual_steps, max_steps=canonical["max_steps"], trainer="Trainer",
                    processor_identity=processor_identity_value, transformation_manifest=transform,
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
    config = _read_json(path / "effective_config.json")
    _adapter_structure(path, r, config.get("lora_execution", {}).get("matched_target_modules"))
    from .baseline_evidence import fingerprint
    identity = execution_identity(r)
    if (evidence.get("schema_version") != 1 or evidence.get("kind") != "training"
            or evidence.get("stage") != "completed" or evidence.get("task_id") != TASK_ID
            or evidence.get("recipe_id") != RECIPE_ID or evidence.get("mode") not in ("full", "smoke")
            or (checkpoint is None and evidence.get("mode") != "full")
            or any(evidence.get(k) is True for k in ("mock", "simulated", "dry_run"))):
        raise TaskExecutionError("checkpoint 不是已完成 Granite full/smoke 训练")
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
    expected_args = config_kwargs(r, {"recipe_id": r.id, "task_id": TASK_ID,
                                      "mode": evidence["mode"]}, path)
    if (evidence.get("config") != config or evidence.get("config_sha256") != sha256_json(config)
            or training_arguments_identity(config.get("training_arguments"))
            != training_arguments_identity(expected_args)
            or config.get("granite") != r.get("granite") or config.get("peft") != r.get("peft")
            or config.get("quantization") != r.get("quantization")
            or config.get("attention_implementation") != "eager"
            or config.get("upstream_metadata") != dict(r.get("upstream"))
            or type(evidence.get("steps")) is not int
            or evidence["steps"] != expected_args["max_steps"]):
        raise TaskExecutionError("checkpoint steps/config hash/预算不一致")
    lora = config.get("lora_execution", {})
    if (not lora.get("matched_target_modules") or lora.get("use_dora") is not True
            or type(lora.get("trainable_parameters")) is not int
            or lora["trainable_parameters"] <= 0
            or lora.get("base_parameters_frozen") is not True
            or lora.get("tensor_reload_check", {}).get("strict_values_verified") is not True):
        raise TaskExecutionError("checkpoint LoRA/DoRA 或 adapter 张量复载证据缺失")
    source = _read_json(path / "source_manifest.json")
    validate_source_manifest(source, r)
    if evidence.get("source_manifest") != source or evidence.get("split_manifest") != source:
        raise TaskExecutionError("checkpoint source manifest 不一致")
    processor = _read_json(path / PROCESSOR_IDENTITY_FILE)
    transform = _read_json(path / TRANSFORM_FILE)
    if (not isinstance(processor, dict) or processor.get("sha256") != _hash_manifest(processor)["sha256"]
            or evidence.get("processor_identity") != processor
            or processor.get("image_token_id") != config.get("image_token_id")
            or type(processor.get("image_token_id")) is not int
            or transform.get("sha256") != _hash_manifest(transform)["sha256"]
            or transform.get("processor_identity") != processor
            or transform.get("protocol") != PROTOCOL
            or transform.get("data_seed") != r.get("train.data_seed")
            or evidence.get("transformation_manifest") != transform):
        raise TaskExecutionError("checkpoint processor/transformation manifest 不一致")
    validate_transform_manifest(transform, source, r, processor)
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
    ckpt = None
    if not zero_shot:
        ckpt = checkpoint_path(plan, checkpoint=checkpoint, recipe=r)
        source = _read_json(ckpt / "training.json")
    splits, manifest = load_splits(r, runtime)
    processor, processor_identity_value, model = _load_base(r, runtime, lock, training=False)
    if not zero_shot:
        peft = runtime["peft"]
        check_signature(peft.PeftModel.from_pretrained,
                        {"model": None, "model_id": None, "is_trainable": False},
                        "PeftModel.from_pretrained")
        model = peft.PeftModel.from_pretrained(model=model, model_id=str(ckpt), is_trainable=False)
    rows = splits["test"]
    maximum = r.get("evaluation.max_new_tokens")
    examples = evaluation_examples(rows, processor, processor_identity_value, r)
    eval_manifest = evaluation_manifest(examples, processor_identity_value, r, manifest)
    predictions = []
    for row, example in zip(rows, examples):
        prompt_ids = example["prompt_ids"]
        length = len(prompt_ids)
        encoded = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * length], dtype=torch.long),
            "pixel_values": torch.tensor([example["pixel_values"]], dtype=torch.float32),
        }
        with torch.no_grad():
            generated = model.generate(
                **encoded, max_new_tokens=maximum, do_sample=False, num_beams=1,
                pad_token_id=processor_identity_value["pad_token_id"],
                eos_token_id=processor_identity_value["eos_token_id"])
        trimmed = [int(token) for token in generated[0][length:]]
        # batch_decode 接受「序列的序列」；直接传一维 token id 列表会把每个 id
        # 当成一条样本，解码出与生成内容无关的字符，必须显式包成单元素 batch。
        decoded = processor.batch_decode([trimmed], skip_special_tokens=False)
        if (not isinstance(decoded, (list, tuple)) or len(decoded) != 1
                or not isinstance(decoded[0], str)):
            raise TaskExecutionError("processor.batch_decode 未返回单条解码文本；拒绝记录不可重建预测")
        text = decoded[0]
        record = score_prediction(row, text, processor_identity_value["eos_token"])
        record.update(prompt_sha256=example["prompt_sha256"], source_index=row["source_index"],
                      image_sha256=row["image_sha256"], generated_text=text,
                      eos_token_id=processor_identity_value["eos_token_id"],
                      max_new_tokens=maximum)
        predictions.append(record)
    metrics = recompute_metrics(predictions)
    generation = {"do_sample": False, "num_beams": 1, "max_new_tokens": maximum,
                  "eos_token_id": processor_identity_value["eos_token_id"],
                  "pad_token_id": processor_identity_value["pad_token_id"], "use_cache": True,
                  "seed": r.get("evaluation.seed"), "eos_truncation": True}
    model_artifact = base_artifact(r) if zero_shot else artifact_manifest(ckpt)
    config = {"task_id": TASK_ID, "protocol": EVAL_PROTOCOL, "zero_shot": bool(zero_shot),
              "checkpoint": None if zero_shot else str(ckpt),
              "generation": generation, "quantization": r.get("quantization"),
              "compute_dtype": "bfloat16", "attention_implementation": "eager",
              "model": model_artifact, "processor_identity": processor_identity_value,
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
        model_artifact=model_artifact, checkpoint_training_evidence=source,
        processor_identity=processor_identity_value,
        prompt_protocol=r.get("granite.prompt_format"),
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
    save_json(output / PROCESSOR_IDENTITY_FILE, processor_identity_value)
    save_json(output / EVAL_MANIFEST_FILE, eval_manifest)
    return save_json(output / "evaluation.json", evidence)
