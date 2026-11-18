"""vlm/mpo/qwen2-5-vl-3b-mpo 的受控自建迁移协议 qwen25vl-mpo-v1（data/train/eval）。

一手事实（HuggingFace Cookbook fine_tuning_vlm_mpo）与本仓差异：
  * base Qwen/Qwen2.5-VL-3B-Instruct；dataset HuggingFaceH4/rlaif-v_formatted，
    上游 train[:5%] / test[:1%]；本仓与 DPO 篇共享同一偏好 split（图像 hash 80/10/10）。
  * 安装 git+https://github.com/huggingface/trl.git bitsandbytes qwen-vl-utils==0.0.8；
    TRL 源码无 commit/revision，其他依赖无 pin。
  * 4bit nf4 + double quant + bf16；LoRA r=8 alpha=8 dropout=0.1，7 个 target，use_dora=True。
  * 明确说明 MPO 是 DPOTrainer 的组合损失，不是新 trainer：loss_type=['sigmoid','bco_pair','sft']，
    loss_weights=[0.8,0.2,1.0]。
  * DPOConfig 显式 bf16=False、batch=4/4、accum=8、epochs=1、dataset_num_proc=1、
    dataloader_num_workers=8、logging/save/eval steps=10；没有 seed、revision、compute_metrics；
    ref_model=None；评估仅 test_dataset[0] 定性展示。
本仓用普通 transformers.Trainer + 自建 sigmoid/bco_pair/sft 三损失加权和，不依赖未固定
TRL 源码；bf16=False 与量化 compute dtype=bfloat16 的矛盾如实登记、不静默修正。
不冒充上游复现；与 DPO 篇基座不同，cross_backbone_attribution_valid=false。

发布级 comparison/export 由 mpo_suite.py / mpo_export.py 承担，task_router 按精确
recipe_id 注册 qwen25vl-mpo-v1；本模块只负责 data/train/eval 三维。
"""
from __future__ import annotations

import copy
import importlib
import inspect
import math
import os
import platform
import re
import sys
import time
import tomllib
from pathlib import Path

from . import preference_common as pc
from .math_task import TaskExecutionError
from .model import Recipe

DatasetValidationError = pc.DatasetValidationError
sha256_json = pc.sha256_json
fixed_seeds = pc.fixed_seeds
require_revision = pc.require_revision
check_signature = pc.check_signature
apply_determinism = pc.apply_determinism
execution_identity = pc.execution_identity
mode_output_dir = pc.mode_output_dir
save_json = pc.save_json
_need = pc._need

TASK_ID = "qwen25vl-mpo-v1"
RECIPE_ID = "vlm/mpo/qwen2-5-vl-3b-mpo"
MODEL = MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET = DATASET_ID = "HuggingFaceH4/rlaif-v_formatted"
PROTOCOL = "qwen25vl-mpo-v1"
EVAL_PROTOCOL = "qwen25vl-mpo-v1/held-out-implicit-reward-v1"
EVAL_VARIANT = "pairwise-implicit-reward"
RUNTIME_PACKAGES = ("torch", "transformers", "datasets", "peft", "bitsandbytes", "safetensors")
METRICS = pc.METRICS
PREDICTIONS_FILE = "predictions.json"
EVAL_MANIFEST_FILE = "eval_manifest.json"
PROCESSOR_IDENTITY_FILE = "processor_identity.json"
REFERENCE_FILES = ("source_manifest.json", PROCESSOR_IDENTITY_FILE, EVAL_MANIFEST_FILE)
REFERENCE_CONFIG_FILE = "reference-policy.json"
FULL_STEPS = 200
SMOKE_STEPS = 5
ALGORITHM_VERSION = "qwen25vl-mpo-v1"
SYSTEM_MESSAGE = "You are a helpful assistant. Choose the better response to the user's question about the provided image."

ABLATION_CONFIGS = {
    "full": {"loss_types": ["sigmoid", "bco_pair", "sft"], "weights": [0.8, 0.2, 1.0]},
    "sigmoid_only": {"loss_types": ["sigmoid"], "weights": [1.0]},
    "no_sft": {"loss_types": ["sigmoid", "bco_pair"], "weights": [0.8, 0.2]},
}

MPO_SPEC = {
    "protocol": PROTOCOL, "source_split": "train[:5%]", "held_out_split": "test[:1%]",
    "split_percent": [80, 10, 10], "split_algorithm": pc.SPLIT_ALGORITHM,
    "loss_types": ["sigmoid", "bco_pair", "sft"], "loss_weights": [0.8, 0.2, 1.0],
    "ablation_configs": ["full", "sigmoid_only", "no_sft"], "default_config_id": "full",
    "bf16_training": False, "quant_compute_dtype": "bfloat16",
    "precision_conflict": "DPOConfig 显式 bf16=False，但 4bit 量化 compute dtype=bfloat16；如实登记矛盾，不静默修正",
    "max_eval_examples": 300,
    "loss": "三项标量 sigmoid/bco_pair/sft 按权重求和；policy/ref 仅在 completion token 上计 logprob，image placeholder 与 padding label=-100",
    "prompt_format": "system + user(images+prompt)；chosen/rejected 绝不进生成 prompt",
    "metric_definition": "win_rate/reward_accuracy 均为 policy implicit reward pairwise，不是外部 reward model/human eval",
    "upstream_install": "git+https://github.com/huggingface/trl.git bitsandbytes qwen-vl-utils==0.0.8；TRL 源码无 commit/revision，其他依赖无 pin",
    "upstream_seed": "absent（DPOConfig 无 seed/data_seed/deterministic）",
    "upstream_revision": "absent（from_pretrained/load_dataset 未传 revision）",
    "upstream_compute_metrics": "absent（有 eval_dataset 但无 compute_metrics）",
    "upstream_evaluation": "仅 test_dataset[0] 定性展示，无数值指标",
    "upstream_ref_model": "None",
    "cross_backbone_attribution_valid": False, "self_migration": True,
}

METHOD_KEYS = frozenset({"family", "trainer", "peft", "upstream_trainer", "execution_trainer"})
MODEL_KEYS = frozenset({"base", "modality", "revision", "params_b"})
DATASET_KEYS = frozenset({"name", "structure", "type", "has_images", "kind", "revision"})
MODALITY_KEYS = frozenset({"images_column", "content_list_dict", "processing_class", "max_length"})
PEFT_KEYS = frozenset({"r", "alpha", "dropout", "target_module_patterns", "use_dora",
                       "init_lora_weights", "bias", "task_type"})
QUANT_KEYS = frozenset({"load_in_4bit", "bnb_4bit_quant_type",
                        "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype"})
UPSTREAM_KEYS = frozenset({"split", "mask", "evaluation", "revision", "seed", "merge",
                           "install", "attn", "target_modules", "trainer", "processing_class"})
TRAIN_KEYS = frozenset({
    "seed", "data_seed", "seed_mechanism", "loss_types", "loss_weights", "num_train_epochs",
    "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps",
    "gradient_checkpointing", "eval_strategy", "eval_steps", "save_steps", "logging_steps",
    "dataloader_num_workers", "learning_rate", "lr_scheduler_type", "weight_decay", "bf16",
    "fp16", "seq_length", "remove_unused_columns",
})
EVALUATION_KEYS = frozenset({"metrics", "split", "seed", "multi_run", "held_out",
                             "compute_metrics", "max_new_tokens", "expected", "tolerance",
                             "export_tolerance"})
METADATA_TRAIN_KEYS = frozenset({"seed", "data_seed", "seed_mechanism", "num_train_epochs",
                                 "loss_types", "loss_weights", "seq_length"})


def resolve_task_id(r, plan=None, task_id=None):
    declared = task_id or (plan or {}).get("task_id") or r.get("execution.task_id")
    return declared or (TASK_ID if r.id == RECIPE_ID else None)


def require_supported(r, plan=None, task_id=None) -> str:
    resolved = resolve_task_id(r, plan, task_id)
    if r.id != RECIPE_ID or resolved != TASK_ID or (plan or {}).get("recipe_id", r.id) != r.id:
        raise TaskExecutionError(f"尚未实现该执行路由：{r.id} / {resolved!r}")
    required = {
        "method.family": "mpo", "method.trainer": "DPOTrainer",
        "method.peft": "qlora-dora", "method.upstream_trainer": "DPOTrainer",
        "method.execution_trainer": "Trainer",
        "model.base": MODEL, "dataset.name": DATASET,
    }
    for key, expected in required.items():
        if r.get(key) != expected:
            raise TaskExecutionError(f"{TASK_ID} 要求 {key}={expected!r}")
        section, field = key.split(".")
        planned = (plan or {}).get(section, {})
        if field in planned and planned[field] != expected:
            raise TaskExecutionError(f"plan {key} 与 MPO 路由不一致")
    if not r.is_vlm or r.get("dataset.has_images") is not True or r.get("dataset.type") != "preference":
        raise TaskExecutionError("MPO 只接受带图像的静态偏好数据")
    if r.get("dataset.kind", "static") != "static":
        raise TaskExecutionError("MPO 只接受冻结的静态偏好数据集")
    for name, keys in (("method", METHOD_KEYS), ("model", MODEL_KEYS), ("dataset", DATASET_KEYS),
                       ("preference", MPO_SPEC), ("peft", PEFT_KEYS), ("quantization", QUANT_KEYS),
                       ("upstream", UPSTREAM_KEYS), ("budget", {"full_steps", "smoke_steps"})):
        section = r.get(name)
        if name == "preference":
            pc.unknown(section, keys, name)
            for key, expected in keys.items():
                if key not in section or section[key] != expected or type(section[key]) is not type(expected):
                    raise TaskExecutionError(f"preference.{key} 必须为 {expected!r}")
        else:
            pc.unknown(section, keys, name)
    for key in ("loss.types", "loss.weights"):
        if r.get(key) is None:
            raise TaskExecutionError(f"MPO 要求显式 {key}")
    if r.get("loss.types") != ["sigmoid", "bco_pair", "sft"] or r.get("loss.weights") != [0.8, 0.2, 1.0]:
        raise TaskExecutionError("loss.types/weights 必须精确为 sigmoid/bco_pair/sft 与 [0.8,0.2,1.0]")
    train = pc.require_keys(r.get("train"), TRAIN_KEYS, "train")
    if train["loss_types"] != ["sigmoid", "bco_pair", "sft"] or train["loss_weights"] != [0.8, 0.2, 1.0]:
        raise TaskExecutionError("train.loss_types/weights 必须在每个 loss 配置中显式绑定")
    config_id = r.get("execution.config_id", "full")
    verify_loss_config(config_id, train["loss_types"], train["loss_weights"])
    peft = r.get("peft")
    if (peft["use_dora"] is not True or peft["init_lora_weights"] != "gaussian"
            or peft["target_module_patterns"] != ["language_model", "_proj"]
            or peft["r"] != 8 or peft["alpha"] != 8 or peft["dropout"] != 0.1):
        raise TaskExecutionError("peft 必须保留上游 DoRA/gaussian/r8/alpha8/dropout0.1 与 7 target pattern")
    quant = r.get("quantization")
    if (quant["load_in_4bit"] is not True or quant["bnb_4bit_quant_type"] != "nf4"
            or quant["bnb_4bit_use_double_quant"] is not True
            or quant["bnb_4bit_compute_dtype"] != "bfloat16"):
        raise TaskExecutionError("quantization 必须与一手 nf4/bf16/double quant 事实一致")
    # bf16=False 与量化 compute dtype=bfloat16 的矛盾必须显式登记，不静默修正。
    if train["bf16"] is not False or r.get("preference.bf16_training") is not False \
            or r.get("preference.quant_compute_dtype") != "bfloat16":
        raise TaskExecutionError("bf16=False 与量化 compute dtype 矛盾必须按一手事实登记")
    for key in UPSTREAM_KEYS:
        if not isinstance(r.get(f"upstream.{key}"), str) or not r.get(f"upstream.{key}").strip():
            raise TaskExecutionError(f"upstream.{key} 必须保留一手事实原文")
    modality = r.get("modality_contract")
    pc.unknown(modality, MODALITY_KEYS, "modality_contract")
    if (modality is None or modality.get("images_column") != "images"
            or modality.get("processing_class") != "processor"
            or modality.get("content_list_dict") is not True
            or modality.get("max_length") != "none"):
        raise TaskExecutionError("modality_contract 必须保留 images/processor/JSON 内容列表合同")
    pc.unknown(r.get("evaluation"), EVALUATION_KEYS, "evaluation")
    for key, expected in {
        "train.fp16": False, "train.gradient_checkpointing": True,
        "train.remove_unused_columns": False, "train.lr_scheduler_type": "linear",
        "train.eval_strategy": "steps", "evaluation.metrics": list(METRICS),
        "evaluation.held_out": True, "evaluation.multi_run": True,
        "evaluation.compute_metrics": True,
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
    for key in ("learning_rate", "weight_decay"):
        value = train[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise TaskExecutionError(f"train.{key} 必须为有限非负数")
    if train["learning_rate"] == 0:
        raise TaskExecutionError("learning_rate 必须大于零")
    new = r.get("evaluation.max_new_tokens")
    if type(new) is not int or new < 2 or type(train["seq_length"]) is not int or train["seq_length"] < new + 6:
        raise TaskExecutionError("seq_length 必须为整数且至少 max_new_tokens+6")
    if r.get("budget.full_steps") != FULL_STEPS or r.get("budget.smoke_steps") != SMOKE_STEPS:
        raise TaskExecutionError("预算必须显式 full=200 / smoke=5；不可静默改预算")
    dependencies = r.get("dependencies", {}) or {}
    pc.unknown(dependencies, ("trl", "transformers", "datasets", "bitsandbytes", "peft",
                              "accelerate", "qwen-vl-utils"), "dependencies")
    for name, version in dependencies.items():
        if not isinstance(version, str) or not version.strip() or version in ("unknown", "latest", "*"):
            raise TaskExecutionError(f"dependencies.{name} 不能猜测或使用未冻结版本")
    return resolved


def config_recipe(r, config_id):
    """从 base recipe 派生单个消融配置；只覆盖 [train] 损失字段与 execution.config_id。

    显式 config_id 而非并发进程竞争同一份配置；输出目录由 config_id 自然隔离，
    可重复调用且幂等。三损失消融只在同一 Qwen backbone 内进行，不与 DPO 篇混用。
    """
    if r.id != RECIPE_ID or not isinstance(config_id, str) or config_id not in ABLATION_CONFIGS:
        raise TaskExecutionError("config_recipe 只接受 MPO recipe 与显式消融 config_id")
    spec = verify_loss_config(config_id, ABLATION_CONFIGS[config_id]["loss_types"],
                              ABLATION_CONFIGS[config_id]["weights"])
    data = copy.deepcopy(r.data)
    data.setdefault("train", {})["loss_types"] = list(spec["loss_types"])
    data.setdefault("train", {})["loss_weights"] = list(spec["weights"])
    data.setdefault("execution", {})["config_id"] = config_id
    base = (data.get("baseline_contract") or {}).get("experiment_id")
    if isinstance(base, str) and base:
        data.setdefault("baseline_contract", {})["experiment_id"] = f"{base}--{config_id}"
    return Recipe(r.id, r.path, data)


def verify_loss_config(config_id, loss_types, weights):
    """三损失消融配置必须显式绑定 config_id；不同配置不能借用同一身份。"""
    if config_id not in ABLATION_CONFIGS or not isinstance(loss_types, list) or not isinstance(weights, list):
        raise TaskExecutionError(f"未知 MPO 消融 config_id={config_id!r} 或损失配置非法")
    spec = ABLATION_CONFIGS[config_id]
    if loss_types != spec["loss_types"] or weights != spec["weights"]:
        raise TaskExecutionError(
            f"config_id={config_id} 要求 loss_types={spec['loss_types']} weights={spec['weights']}")
    if len(loss_types) != len(weights) or any(type(w) not in (int, float) or w < 0 or not math.isfinite(w)
                                             for w in weights):
        raise TaskExecutionError("loss_weights 必须为与 loss_types 等长的非负有限数值")
    if sum(weights) <= 0:
        raise TaskExecutionError("loss_weights 不得全为零")
    return dict(spec)


def split_spec(r):
    return {**copy.deepcopy(MPO_SPEC), "data_seed": fixed_seeds(r)[1],
            "selection": "有限完整扫描，保留 source_index；无 shuffle、无补救划分",
            "gold_policy": "chosen/rejected 只进入训练 completion 与评估 gold，绝不进生成 prompt"}


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
        result["dataset"] = {**result.get("dataset", {}), "split": "共享偏好 image-hash 80/10/10",
                             "split_plan": split_spec(r)}
        result["evaluation"] = {
            "metrics": list(METRICS), "seed": r.get("evaluation.seed"),
            "max_new_tokens": r.get("evaluation.max_new_tokens"), "held_out": True,
            "multi_run": True, "runs": 1, "protocol": EVAL_PROTOCOL, "do_sample": False,
            "training_seed_variance_measured": False, "upstream_quantitative_evaluation": False,
            "generated_code_executed": False,
        }
        result["migration"] = {
            "trainer": "普通 Trainer + 自建 sigmoid/bco_pair/sft 三损失（MPO 不是新 trainer）",
            "loss": "loss_types=[sigmoid,bco_pair,sft] weights=[0.8,0.2,1.0]，消融 config_id 显式绑定",
            "precision": "bf16=False 与 4bit compute dtype=bfloat16 的矛盾如实登记，不静默修正",
            "ref": "冻结 ref + policy 双模型身份绑定（上游 ref_model=None）",
            "split": "与 DPO 篇共享偏好 image-hash 80/10/10（上游 train[:5%] / test[:1%]）",
            "cross_backbone": "与 DPO 篇基座不同，cross_backbone_attribution_valid=false",
            "license": "base / RLAIF-V 许可与商业使用未确认",
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
    train = r.get("train")
    keys = ("learning_rate", "lr_scheduler_type", "weight_decay", "per_device_train_batch_size",
            "per_device_eval_batch_size", "gradient_accumulation_steps", "bf16", "fp16",
            "gradient_checkpointing", "dataloader_num_workers", "remove_unused_columns",
            "eval_steps", "save_steps", "logging_steps")
    params = {key: train[key] for key in keys}
    params.update(max_steps=steps, output_dir=str(output_dir), report_to="none", push_to_hub=False,
                  eval_strategy="steps", save_strategy="steps", logging_strategy="steps")
    return params


def training_arguments_kwargs(target, canonical):
    try:
        keys = inspect.signature(target).parameters
    except (TypeError, ValueError) as exc:
        raise TaskExecutionError("无法核验 TrainingArguments API") from exc
    result = dict(canonical)
    if "eval_strategy" in keys and "evaluation_strategy" not in keys:
        result["eval_strategy"] = result.pop("evaluation_strategy", "steps")
    elif "evaluation_strategy" not in keys and "eval_strategy" not in keys:
        raise TaskExecutionError("TrainingArguments 缺少 eval_strategy")
    check_signature(target, result, "TrainingArguments")
    return result


def training_arguments_identity(arguments):
    if not isinstance(arguments, dict):
        return None
    return {k: v for k, v in arguments.items() if k != "output_dir"}


# ---------------------------------------------------------------- 数据

def prepare_splits(rows, r):
    return pc.prepare_splits(rows, r, max_eval_examples=r.get("preference.max_eval_examples"))


validate_source_manifest = pc.validate_source_manifest


def load_splits(r, dependencies):
    revision = require_revision(r, "dataset.revision")
    dataset = dependencies["datasets"].load_dataset(DATASET, revision=revision, split="train")
    splits, manifest = prepare_splits(dataset, r)
    manifest["dataset_id"] = DATASET
    manifest["dataset_revision"] = revision
    manifest["split_identity"] = pc.split_identity(
        manifest, dataset_id=DATASET, dataset_revision=revision, algorithm_version=ALGORITHM_VERSION)
    manifest["sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    return splits, manifest


def processor_identity(processor):
    return pc.processor_identity(processor, PROTOCOL)


def train_examples(splits, processor, r):
    require_supported(r)
    identity = processor_identity(processor)
    examples, entries = {}, {}
    for name in ("train", "validation"):
        examples[name], entries[name] = [], []
        for row in splits[name]:
            chosen = pc.encode_pair(row, processor, identity, SYSTEM_MESSAGE, "chosen",
                                    r.get("train.seq_length"))
            rejected = pc.encode_pair(row, processor, identity, SYSTEM_MESSAGE, "rejected",
                                      r.get("train.seq_length"))
            if chosen["prompt_ids"] != rejected["prompt_ids"]:
                raise DatasetValidationError("chosen/rejected 的 prompt token 边界不同；违反同 prompt 同图像协议")
            item = {"chosen": {k: chosen[k] for k in ("input_ids", "labels", "attention_mask", "pixel_values", "pixel_shape")},
                    "rejected": {k: rejected[k] for k in ("input_ids", "labels", "attention_mask", "pixel_values", "pixel_shape")}}
            examples[name].append(item)
            entries[name].append({
                "sample_id": row["sample_id"], "source_index": row["source_index"],
                "image_sha256": row["image_sha256"], "content_sha256": row["content_sha256"],
                "chosen_sha256": sha256_json(row["chosen"]), "rejected_sha256": sha256_json(row["rejected"]),
                "prompt_tokens": chosen["prompt_tokens"], "chosen_tokens": chosen["completion_tokens"],
                "rejected_tokens": rejected["completion_tokens"],
                "masked_image_tokens": chosen["masked_image_tokens"],
                "example_sha256": sha256_json(item),
            })
        if not examples[name]:
            raise DatasetValidationError(f"{name} 编码后为空；拒绝静默跳过")
    config_id = r.get("execution.config_id", "full")
    transform = pc.hash_manifest({
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "processor_identity": identity, "data_seed": r.get("train.data_seed"),
        "config_id": config_id, "loss_types": r.get("train.loss_types"),
        "loss_weights": r.get("train.loss_weights"), "loss": r.get("preference.loss"),
        "prompt_format": r.get("preference.prompt_format"), "split_algorithm": pc.SPLIT_ALGORITHM,
        "entries": entries,
    })
    return examples, transform


def validate_transform_manifest(transform, source, r, identity):
    if (not isinstance(transform, dict) or transform.get("task_id") != TASK_ID
            or transform.get("protocol") != PROTOCOL
            or transform.get("processor_identity") != identity
            or transform.get("config_id") != r.get("execution.config_id", "full")
            or transform.get("loss_types") != r.get("train.loss_types")
            or transform.get("loss_weights") != r.get("train.loss_weights")
            or transform.get("loss") != r.get("preference.loss")
            or transform.get("data_seed") != r.get("train.data_seed")
            or transform.get("sha256") != pc.hash_manifest(transform)["sha256"]):
        raise TaskExecutionError("transform manifest 与当前 MPO 协议不一致")
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
                    or row["masked_image_tokens"] < 1 or type(row.get("chosen_tokens")) is not int
                    or row["chosen_tokens"] < 1 or type(row.get("rejected_tokens")) is not int
                    or row["rejected_tokens"] < 1
                    or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("example_sha256", "")))):
                raise TaskExecutionError("transform manifest 条目缺少图像 hash / 双层 mask / 内容身份")
    return transform


# ---------------------------------------------------------------- 三损失

def _softplus_neg(x):
    return math.log1p(math.exp(-x)) if x >= 0 else -x + math.log1p(math.exp(x))


def sigmoid_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0):
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return _softplus_neg(float(beta) * float(logits))


def bco_pair_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0):
    """BCO pair：chosen 的隐式 logit 越大越好、rejected 越小越好。"""
    chosen_logit = float(beta) * (float(policy_chosen) - float(ref_chosen))
    rejected_logit = float(beta) * (float(policy_rejected) - float(ref_rejected))
    return _softplus_neg(chosen_logit) + _softplus_neg(-rejected_logit)


def sft_loss(policy_chosen):
    """chosen completion 的负 log 概率。"""
    return -float(policy_chosen)


def mpo_combined_loss(loss_types, weights, *, policy_chosen, policy_rejected,
                      ref_chosen, ref_rejected, beta=1.0):
    """三项标量函数按权重求和；config_id 决定 loss_types/weights，拒绝隐式默认。"""
    verify_loss_config("full", ["sigmoid", "bco_pair", "sft"], [0.8, 0.2, 1.0])
    if not isinstance(loss_types, list) or not isinstance(weights, list) or len(loss_types) != len(weights):
        raise TaskExecutionError("MPO loss_types / loss_weights 必须等长列表")
    table = {
        "sigmoid": sigmoid_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta),
        "bco_pair": bco_pair_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta),
        "sft": sft_loss(policy_chosen),
    }
    unknown_types = [name for name in loss_types if name not in table]
    if unknown_types:
        raise TaskExecutionError(f"未知 MPO loss 类型 {unknown_types}")
    return {"total": sum(weight * table[name] for name, weight in zip(loss_types, weights)),
            "components": {name: table[name] for name in loss_types}}


def implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=1.0):
    return (float(beta) * (float(policy_chosen) - float(ref_chosen)),
            float(beta) * (float(policy_rejected) - float(ref_rejected)))


def sequence_logprob(model, input_ids, attention_mask, pixel_values):
    target = getattr(model, "sequence_logprob", None)
    if callable(target):
        value = target(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    else:
        output = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        value = getattr(output, "logprob", None)
    if value is None:
        raise TaskExecutionError("模型未提供可核验的 sequence logprob 接口")
    return value


def verify_policy_ref_binding(policy, ref, *, expected_lora_targets=None):
    if policy is ref:
        raise TaskExecutionError("ref 与 policy 是同一模型；ref 未独立冻结")
    ref_trainable = [name for name, p in ref.named_parameters() if p.requires_grad]
    if ref_trainable:
        raise TaskExecutionError(f"ref 模型存在可训练参数 {ref_trainable[:3]}；ref 必须冻结且无梯度")
    policy_trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    if not policy_trainable:
        raise TaskExecutionError("policy 没有可训练 LoRA 参数")
    for name, _ in policy_trainable:
        if "lora_" not in name:
            raise TaskExecutionError(f"policy 存在非 LoRA 可训练参数：{name}")
    evidence = {"policy_ref_disjoint": True, "ref_trainable_parameters": 0,
                "policy_trainable_parameters": sum(p.numel() for _, p in policy_trainable),
                "policy_trainable_keys": sorted(name for name, _ in policy_trainable)}
    if expected_lora_targets is not None:
        evidence["expected_lora_targets"] = sorted(expected_lora_targets)
    return evidence


def reference_policy_identity(r, ref_binding):
    return pc.hash_manifest({
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "model_id": MODEL, "revision": r.get("model.revision"),
        "ref_model_source": "frozen base（与 policy 同 base，但不加载 adapter）",
        "ref_trainable_parameters": ref_binding["ref_trainable_parameters"],
        "policy_ref_disjoint": ref_binding["policy_ref_disjoint"],
        "upstream_ref_model": r.get("preference.upstream_ref_model"),
        "config_id": r.get("execution.config_id", "full"),
    })


# ---------------------------------------------------------------- 执行框架

def _lock(r):
    try:
        path = pc.no_symlinks(r.path.parent / r.get("execution.lockfile"))
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
    from .math_task import require_frozen_runtime as _base
    _lock(r)
    _base(r, dependencies)


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
        "bf16_training": False,
    }


def _output(plan, mode):
    output = pc.no_symlinks(mode_output_dir(plan, mode)).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise TaskExecutionError(f"输出目录非空，禁止覆盖：{output}")
    return output


def _preflight(r, plan, training=False):
    require_supported(r, plan)
    for key in ("model.revision", "dataset.revision", "source_sha"):
        require_revision(r, key)
    fixed_seeds(r)
    if training and (r.get("decision.gate") == "blocked" or r.get("baseline.result") == "passes"):
        raise TaskExecutionError("训练 decision gate 已阻断，禁止绕过")
    if plan.get("requested_runs", 1) != 1:
        raise TaskExecutionError("独立训练多 seed 由主线程编排；重复 greedy 不冒充 multi-run")
    return _lock(r)


def _quantization(r, runtime):
    spec = r.get("quantization")
    kwargs = {"load_in_4bit": spec["load_in_4bit"],
              "bnb_4bit_quant_type": spec["bnb_4bit_quant_type"],
              "bnb_4bit_compute_dtype": runtime["torch"].bfloat16,
              "bnb_4bit_use_double_quant": spec["bnb_4bit_use_double_quant"]}
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
    model = transformers.AutoModelForVision2Seq.from_pretrained(
        MODEL, revision=lock["model_revision"], trust_remote_code=False,
        quantization_config=_quantization(r, runtime), torch_dtype=runtime["torch"].bfloat16,
        attn_implementation="eager", device_map={"": 0})
    model.config.use_cache = not training
    return processor, identity, model


def _peft_kwargs(r, revision, targets):
    peft = r.get("peft")
    return {"r": peft["r"], "lora_alpha": peft["alpha"], "lora_dropout": peft["dropout"],
            "target_modules": list(targets), "use_dora": peft["use_dora"],
            "init_lora_weights": peft["init_lora_weights"], "bias": peft["bias"],
            "task_type": peft["task_type"], "base_model_name_or_path": MODEL, "revision": revision}


def _attach_lora(model, runtime, kwargs):
    peft = runtime["peft"]
    check_signature(peft.LoraConfig, kwargs, "LoraConfig")
    check_signature(peft.get_peft_model, {"model": None, "peft_config": None}, "get_peft_model")
    check_signature(peft.prepare_model_for_kbit_training,
                    {"model": None, "use_gradient_checkpointing": True},
                    "prepare_model_for_kbit_training")
    targets = kwargs["target_modules"]
    if len(targets) < 2 or len(set(targets)) != len(targets):
        raise TaskExecutionError("LoRA target 命中过少或重复；拒绝空命中")
    named = {name for name, _ in model.named_modules()}
    if not set(targets) <= named:
        raise TaskExecutionError("LoRA target 未全部出现在 named_modules()")
    model = peft.prepare_model_for_kbit_training(model=model, use_gradient_checkpointing=True)
    if any(p.requires_grad for _, p in model.named_parameters()):
        raise TaskExecutionError("prepare_kbit 后 base 参数未全部冻结")
    model = peft.get_peft_model(model=model, peft_config=peft.LoraConfig(**kwargs))
    trainable = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name:
                raise TaskExecutionError(f"base 参数未冻结：{name}")
            trainable += parameter.numel()
    if trainable <= 0:
        raise TaskExecutionError("LoRA trainable 参数为空")
    return model, {"matched_target_modules": sorted(targets), "trainable_parameters": trainable,
                   "base_parameters_frozen": True, "use_dora": True}


def make_trainer_class(base_trainer):
    class PreferenceTrainer(base_trainer):
        mpo_config = None
        mpo_calls = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            pc._need(self.mpo_config is not None, "MPO 损失配置未绑定；拒绝无配置训练")
            pc._need({"policy_chosen_logps", "policy_rejected_logps",
                      "ref_chosen_logps", "ref_rejected_logps"} <= set(inputs),
                     "MPO 训练 batch 缺少 policy/ref 双侧 logps")
            config = self.mpo_config
            result = mpo_combined_loss(
                config["loss_types"], config["weights"],
                policy_chosen=inputs["policy_chosen_logps"],
                policy_rejected=inputs["policy_rejected_logps"],
                ref_chosen=inputs["ref_chosen_logps"],
                ref_rejected=inputs["ref_rejected_logps"],
                beta=config.get("beta", 1.0))
            PreferenceTrainer.mpo_calls += 1
            return result["total"]

    return PreferenceTrainer


def _adapter_structure(path, r, targets=None):
    pc.no_symlinks(path)
    adapter = pc.read_json(path / "adapter_config.json")
    peft = r.get("peft")
    if not (path / "processor_config.json").is_file() or not (
            path / "adapter_model.safetensors").is_file():
        raise TaskExecutionError("checkpoint 缺少 adapter weights / processor config；无 base fallback")
    expected = {"base_model_name_or_path": MODEL, "revision": r.get("model.revision"),
                "r": peft["r"], "lora_alpha": peft["alpha"], "lora_dropout": peft["dropout"],
                "bias": peft["bias"], "task_type": peft["task_type"], "use_dora": True,
                "init_lora_weights": peft["init_lora_weights"]}
    for key, value in expected.items():
        if adapter.get(key) != value:
            raise TaskExecutionError(f"adapter_config {key} 与 recipe 不一致")
    if targets is not None and set(adapter.get("target_modules", [])) != set(targets):
        raise TaskExecutionError("adapter_config target_modules 与训练时命中集合不一致")


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
    processor, processor_identity_value, policy = _load_base(r, runtime, lock, training=True)
    examples, transform = train_examples(splits, processor, r)
    validate_transform_manifest(transform, source_manifest, r, processor_identity_value)
    patterns = list(r.get("peft.target_module_patterns"))
    targets = sorted({name for name, _ in policy.named_modules()
                      if all(pattern in name for pattern in patterns)})
    if not targets:
        raise TaskExecutionError("动态 LoRA target 命中为空；拒绝用默认模块名冒充")
    policy, lora_evidence = _attach_lora(
        policy, runtime, _peft_kwargs(r, lock["model_revision"], targets))
    _, _, ref = _load_base(r, runtime, lock, training=False)
    ref_binding = verify_policy_ref_binding(policy, ref, expected_lora_targets=targets)
    ref_identity = reference_policy_identity(r, ref_binding)
    config_id = r.get("execution.config_id", "full")
    loss_config = verify_loss_config(config_id, r.get("train.loss_types"), r.get("train.loss_weights"))
    TrainerClass = make_trainer_class(transformers.Trainer)
    trainer = TrainerClass(
        model=policy, args=args,
        train_dataset=runtime["datasets"].Dataset.from_list(examples["train"]),
        eval_dataset=runtime["datasets"].Dataset.from_list(examples["validation"]),
        data_collator=lambda items: pc.collate_pairs(
            items, torch, processor_identity_value["pad_token_id"], processor_identity_value["image_token_id"]))
    trainer.mpo_config = {**loss_config, "beta": 1.0}
    result = trainer.train()
    actual_steps = trainer.state.global_step
    if type(actual_steps) is not int or actual_steps != canonical["max_steps"]:
        raise TaskExecutionError(f"训练未完成预算：实际 {actual_steps}，要求 {canonical['max_steps']}")
    if TrainerClass.mpo_calls < 1:
        raise TaskExecutionError("自建 MPO 三损失未被调用；拒绝保存未训练 adapter")
    if execution_identity(r) != identity:
        raise TaskExecutionError("保存前 code/lock identity 漂移")
    policy.save_pretrained(str(output), safe_serialization=True)
    processor.save_pretrained(str(output))
    _adapter_structure(output, r, targets)
    lora_evidence["tensor_reload_check"] = pc.verify_adapter_tensors(policy, output, runtime)
    effective_config = {
        "training_arguments": canonical, "api_training_arguments": kwargs,
        "peft": r.get("peft"), "quantization": r.get("quantization"),
        "attention_implementation": "eager", "preference": r.get("preference"),
        "config_id": config_id, "loss_config": loss_config,
        "loss_types": r.get("train.loss_types"), "loss_weights": r.get("train.loss_weights"),
        "bf16_training": r.get("train.bf16"),
        "precision_conflict": r.get("preference.precision_conflict"),
        "upstream_metadata": dict(r.get("upstream")), "lora_execution": lora_evidence,
        "ref_binding": ref_binding, "ref_identity": ref_identity,
        "processor_identity": processor_identity_value,
        "image_token_id": processor_identity_value["image_token_id"],
    }
    save_json(output / "effective_recipe.json",
              {"id": r.id, "original_path": str(r.path.absolute()), "data": r.data})
    save_json(output / "effective_config.json", effective_config)
    save_json(output / PROCESSOR_IDENTITY_FILE, processor_identity_value)
    save_json(output / REFERENCE_CONFIG_FILE, ref_identity)
    save_json(output / "source_manifest.json", source_manifest)
    save_json(output / "mpo_transform_manifest.json", transform)
    torch.cuda.synchronize()
    evidence = pc.completed(r, plan, identity, "training", effective_config, source_manifest,
                            runtime_evidence(runtime, started), TASK_ID)
    evidence.update(steps=actual_steps, max_steps=canonical["max_steps"], trainer="Trainer",
                    processor_identity=processor_identity_value, transformation_manifest=transform,
                    train_metrics=dict(result.metrics), model_artifact=pc.artifact_manifest(output),
                    config_id=config_id, loss_config=loss_config, ref_binding=ref_binding,
                    ref_identity=ref_identity, bf16_training=r.get("train.bf16"))
    if execution_identity(r) != identity:
        raise TaskExecutionError("完成时 code/lock identity 漂移")
    return save_json(output / "training.json", evidence)


def checkpoint_path(plan, checkpoint=None, recipe=None) -> Path:
    path = pc.no_symlinks(Path(checkpoint).expanduser() if checkpoint is not None
                          else mode_output_dir(plan, "full")).resolve()
    if not path.is_dir():
        raise TaskExecutionError("没有训练 checkpoint；无 base fallback")
    evidence = pc.read_json(path / "training.json")
    saved = pc.read_json(path / "effective_recipe.json")
    if recipe is None:
        if not isinstance(saved, dict) or not isinstance(saved.get("data"), dict) or not saved.get("original_path"):
            raise TaskExecutionError("checkpoint effective recipe 缺失")
        recipe = Recipe(saved.get("id"), Path(saved["original_path"]), saved["data"])
    r = recipe
    require_supported(r, plan)
    _lock(r)
    _adapter_structure(path, r, evidence.get("config", {}).get("lora_execution", {}).get("matched_target_modules"))
    from .baseline_evidence import fingerprint
    identity = execution_identity(r)
    if (evidence.get("schema_version") != 1 or evidence.get("kind") != "training"
            or evidence.get("stage") != "completed" or evidence.get("task_id") != TASK_ID
            or evidence.get("recipe_id") != RECIPE_ID or evidence.get("mode") not in ("full", "smoke")
            or (checkpoint is None and evidence.get("mode") != "full")
            or any(evidence.get(k) is True for k in ("mock", "simulated", "dry_run"))):
        raise TaskExecutionError("checkpoint 不是已完成 MPO full/smoke 训练")
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
    config = pc.read_json(path / "effective_config.json")
    expected_args = config_kwargs(r, {"recipe_id": r.id, "task_id": TASK_ID,
                                      "mode": evidence["mode"]}, path)
    config_id = r.get("execution.config_id", "full")
    expected_loss = verify_loss_config(config_id, r.get("train.loss_types"), r.get("train.loss_weights"))
    if (evidence.get("config") != config or evidence.get("config_sha256") != sha256_json(config)
            or training_arguments_identity(config.get("training_arguments"))
            != training_arguments_identity(expected_args)
            or config.get("preference") != r.get("preference")
            or config.get("config_id") != config_id
            or config.get("loss_types") != r.get("train.loss_types")
            or config.get("loss_weights") != r.get("train.loss_weights")
            or config.get("loss_config") != expected_loss
            or config.get("bf16_training") is not False
            or config.get("attention_implementation") != "eager"
            or config.get("upstream_metadata") != dict(r.get("upstream"))
            or type(evidence.get("steps")) is not int
            or evidence["steps"] != expected_args["max_steps"]):
        raise TaskExecutionError("checkpoint steps/config hash/三损失配置或消融 config_id 不一致")
    binding = config.get("ref_binding", {})
    ref_identity = config.get("ref_identity", {})
    if (binding.get("policy_ref_disjoint") is not True
            or binding.get("ref_trainable_parameters") != 0
            or type(binding.get("policy_trainable_parameters")) is not int
            or binding["policy_trainable_parameters"] <= 0
            or ref_identity.get("sha256") != pc.hash_manifest(ref_identity)["sha256"]
            or ref_identity.get("ref_trainable_parameters") != 0
            or evidence.get("ref_binding") != binding or evidence.get("ref_identity") != ref_identity):
        raise TaskExecutionError("checkpoint ref/policy 分离或三损失绑定证据缺失")
    lora = config.get("lora_execution", {})
    if (not lora.get("matched_target_modules") or lora.get("use_dora") is not True
            or type(lora.get("trainable_parameters")) is not int or lora["trainable_parameters"] <= 0
            or lora.get("base_parameters_frozen") is not True
            or lora.get("tensor_reload_check", {}).get("strict_values_verified") is not True):
        raise TaskExecutionError("checkpoint LoRA/DoRA 或 adapter 张量复载证据缺失")
    source = pc.read_json(path / "source_manifest.json")
    validate_source_manifest(source, r)
    if evidence.get("source_manifest") != source or evidence.get("split_manifest") != source:
        raise TaskExecutionError("checkpoint source manifest 不一致")
    processor = pc.read_json(path / PROCESSOR_IDENTITY_FILE)
    ref_file = pc.read_json(path / REFERENCE_CONFIG_FILE)
    transform = pc.read_json(path / "mpo_transform_manifest.json")
    if (not isinstance(processor, dict) or processor.get("sha256") != pc.hash_manifest(processor)["sha256"]
            or evidence.get("processor_identity") != processor
            or processor.get("image_token_id") != config.get("image_token_id")
            or type(processor.get("image_token_id")) is not int
            or ref_file != ref_identity or evidence.get("ref_identity") != ref_file
            or transform.get("sha256") != pc.hash_manifest(transform)["sha256"]
            or transform.get("processor_identity") != processor
            or transform.get("protocol") != PROTOCOL
            or transform.get("config_id") != config_id
            or evidence.get("transformation_manifest") != transform):
        raise TaskExecutionError("checkpoint processor/ref/transformation manifest 不一致")
    validate_transform_manifest(transform, source, r, processor)
    if not pc.same_artifact(evidence.get("model_artifact"), pc.artifact_manifest(path)):
        raise TaskExecutionError("checkpoint 真实 artifact hash/bytes 不一致")
    return path


def _sequence_inputs(torch, prompt_ids, completion_ids, pixel_values):
    ids = list(prompt_ids) + list(completion_ids)
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(ids)], dtype=torch.long),
        "pixel_values": torch.tensor([pixel_values], dtype=torch.float32),
    }


def _encode_completion(row, processor, completion):
    full = processor.apply_chat_template(
        pc.render_messages(row, SYSTEM_MESSAGE, completion=completion),
        tokenize=False, add_generation_prompt=False)
    ids, _ = pc._processor_call(processor, full, row["images"])
    prompt = pc.render_prompt(row, processor, SYSTEM_MESSAGE)
    prompt_ids, _ = pc._processor_call(processor, prompt, row["images"])
    if ids[:len(prompt_ids)] != prompt_ids:
        raise DatasetValidationError("评估 completion 与 prompt token 边界不一致")
    pixel = processor(text=prompt, images=[list(row["images"])], return_tensors=None, padding=False)
    return prompt_ids, ids[len(prompt_ids):], pc.canonical(pixel.get("pixel_values"))


def evaluation_examples(rows, processor, identity, r, source):
    return pc.evaluation_examples(rows, processor, identity, SYSTEM_MESSAGE, source, EVAL_PROTOCOL,
                                  r.get("evaluation.max_new_tokens"))


def score_example(row, processor, policy, ref, torch, identity, config_id="full"):
    prompt_ids, chosen_completion, pixel = _encode_completion(row, processor, row["chosen"])
    _, rejected_completion, _ = _encode_completion(row, processor, row["rejected"])
    base = {"sample_id": row["sample_id"], "source_index": row["source_index"],
            "image_sha256": row["image_sha256"],
            "prompt_sha256": sha256_json(pc.render_prompt(row, processor, SYSTEM_MESSAGE)),
            "chosen_sha256": sha256_json(row["chosen"]), "rejected_sha256": sha256_json(row["rejected"]),
            "generated_text": "", "generated_code_executed": False, "config_id": config_id}
    if not chosen_completion or not rejected_completion:
        return pc.score_pair(None, None, 0, 0, **base)
    policy_c = sequence_logprob(policy, **_sequence_inputs(torch, prompt_ids, chosen_completion, pixel))
    policy_r = sequence_logprob(policy, **_sequence_inputs(torch, prompt_ids, rejected_completion, pixel))
    ref_c = sequence_logprob(ref, **_sequence_inputs(torch, prompt_ids, chosen_completion, pixel))
    ref_r = sequence_logprob(ref, **_sequence_inputs(torch, prompt_ids, rejected_completion, pixel))
    chosen_reward, rejected_reward = implicit_rewards(policy_c, policy_r, ref_c, ref_r)
    return pc.score_pair(
        chosen_reward, rejected_reward, len(chosen_completion), len(rejected_completion),
        policy_chosen_logps=float(policy_c), policy_rejected_logps=float(policy_r),
        ref_chosen_logps=float(ref_c), ref_rejected_logps=float(ref_r),
        chosen_reward=chosen_reward, rejected_reward=rejected_reward, **base)


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
    source_evidence = None
    if not zero_shot:
        ckpt = checkpoint_path(plan, checkpoint=checkpoint, recipe=r)
        source_evidence = pc.read_json(ckpt / "training.json")
    splits, manifest = load_splits(r, runtime)
    processor, processor_identity_value, policy = _load_base(r, runtime, lock, training=False)
    _, _, ref = _load_base(r, runtime, lock, training=False)
    if not zero_shot:
        peft = runtime["peft"]
        check_signature(peft.PeftModel.from_pretrained,
                        {"model": None, "model_id": None, "is_trainable": False},
                        "PeftModel.from_pretrained")
        policy = peft.PeftModel.from_pretrained(model=policy, model_id=str(ckpt), is_trainable=False)
    binding = verify_policy_ref_binding(policy, ref)
    config_id = r.get("execution.config_id", "full")
    rows = splits["test"]
    examples = evaluation_examples(rows, processor, processor_identity_value, r, manifest)
    samples = examples["samples"]
    predictions = []
    for row, sample in zip(rows, samples):
        record = score_example(row, processor, policy, ref, torch, processor_identity_value, config_id)
        for key in ("source_index", "image_sha256", "prompt_sha256", "chosen_sha256", "rejected_sha256"):
            if record.get(key) != sample.get(key):
                raise TaskExecutionError("逐题预测与 eval_manifest 输入身份不一致")
        predictions.append(record)
    metrics = pc.recompute_metrics(predictions)
    bias = pc.length_bias_diagnostic(predictions)
    model_artifact = pc.base_artifact(r, MODEL) if zero_shot else pc.artifact_manifest(ckpt)
    config = {
        "task_id": TASK_ID, "protocol": EVAL_PROTOCOL, "variant": EVAL_VARIANT,
        "zero_shot": bool(zero_shot), "checkpoint": None if zero_shot else str(ckpt),
        "generation": {"do_sample": False, "num_beams": 1,
                       "max_new_tokens": r.get("evaluation.max_new_tokens"),
                       "eos_token_id": processor_identity_value["eos_token_id"],
                       "pad_token_id": processor_identity_value["pad_token_id"],
                       "seed": r.get("evaluation.seed"), "eos_truncation": True},
        "config_id": config_id, "loss_types": r.get("train.loss_types"),
        "loss_weights": r.get("train.loss_weights"), "ref_binding": binding,
        "quantization": r.get("quantization"), "compute_dtype": "bfloat16", "bf16_training": False,
        "attention_implementation": "eager", "model": model_artifact,
        "processor_identity": processor_identity_value,
        "dataset_revision": r.get("dataset.revision"), "dataset_id": DATASET,
        "source_manifest_sha256": manifest["sha256"],
        "split_hashes_sha256": manifest["split_hashes_sha256"],
        "eval_manifest_sha256": examples["sha256"],
        "max_new_tokens": r.get("evaluation.max_new_tokens"), "do_sample": False, "num_beams": 1,
        "held_out_rows": len(rows), "generated_code_executed": False,
        "upstream_quantitative_evaluation": False,
        "reward_source": "policy implicit reward（不是外部 reward model / human eval）",
    }
    save_json(output / PREDICTIONS_FILE,
              {"schema_version": 1, "task_id": TASK_ID, "recipe_id": r.id,
               "protocol": EVAL_PROTOCOL, "metrics": metrics, "predictions": predictions,
               "sha256": sha256_json(predictions)})
    predictions_file = pc.file_record(output / PREDICTIONS_FILE, output)
    evidence = pc.completed(r, plan, identity, "evaluation", config, manifest,
                            runtime_evidence(runtime, started), TASK_ID)
    counts = {name: sum(row[name] for row in predictions) for name in METRICS}
    counts["total"] = len(predictions)
    evidence.update(
        model_role="base" if zero_shot else "checkpoint", model_artifact=model_artifact,
        checkpoint_training_evidence=source_evidence,
        processor_identity=processor_identity_value, ref_binding=binding,
        prompt_protocol=r.get("preference.prompt_format"), evaluation_protocol=EVAL_PROTOCOL,
        evaluation_manifest=examples, metrics=metrics, metric_counts=counts,
        length_bias_diagnostic=bias, sample_count=len(predictions), config_id=config_id,
        predictions={"path": predictions_file["path"], "count": len(predictions),
                     "sha256": sha256_json(predictions)},
        predictions_file=predictions_file, predictions_sha256=sha256_json(predictions),
        predictions_path=str(output / PREDICTIONS_FILE),
        split_manifest_path="source_manifest.json",
    )
    save_json(output / "source_manifest.json", manifest)
    save_json(output / PROCESSOR_IDENTITY_FILE, processor_identity_value)
    save_json(output / EVAL_MANIFEST_FILE, examples)
    return save_json(output / "evaluation.json", evidence)
