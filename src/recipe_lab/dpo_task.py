"""vlm/dpo/smolvlm-rlaif-v 的受控自建迁移协议 smolvlm-dpo-v1（data/train/eval）。

一手事实（HuggingFace Cookbook fine_tuning_vlm_dpo_smolvlm_instruct）与本仓差异：
  * base HuggingFaceTB/SmolVLM-Instruct；dataset HuggingFaceH4/rlaif-v_formatted，
    上游 train[:6%] / test[:1%]；本仓按图像原始字节 hash 重划 80/10/10。
  * 依赖注释 transformers 4.46.3 / trl 0.12.2 / datasets 3.2.0 / bitsandbytes 0.45.0 /
    peft 0.14.0 / accelerate 1.2.0；flash-attn 无版本。
  * 上游 4bit nf4 + double quant + bf16；LoRA r=8 alpha=8 dropout=0.1，7 个 target，
    use_dora=True，init_lora_weights='gaussian'。
  * DPOConfig 只显式给 output_dir/bf16/gradient_checkpointing/batch=1/1/accum=32/epochs=5/
    dataset_num_proc=8/dataloader_num_workers=8/logging/save/eval steps=10；没有 beta、
    loss_type、learning_rate、max_num_tokens、seed、revision、compute_metrics。
  * ref_model=None，peft_config=...，processing_class=processor；只 save_model 与
    model.load_adapter，没有 merge_and_unload；评估仅定性生成，无数值指标。
本仓用普通 transformers.Trainer + 自建显式 beta 的 sigmoid DPO loss，不依赖 TRL 未固定
源码；beta/loss/seed/revision/processor identity/冻结 ref 全部显式绑定。不冒充上游复现。

发布级 comparison/export 由 dpo_suite.py / dpo_export.py 承担，task_router 按精确
recipe_id 注册 smolvlm-dpo-v1；本模块只负责 data/train/eval 三维，不借用其他任务的
协议或身份。所有模块只在本机无 GPU 环境做过离线合成测试。
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

TaskExecutionError = TaskExecutionError
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

TASK_ID = "smolvlm-dpo-v1"
RECIPE_ID = "vlm/dpo/smolvlm-rlaif-v"
MODEL = MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
DATASET = DATASET_ID = "HuggingFaceH4/rlaif-v_formatted"
PROTOCOL = "smolvlm-dpo-v1"
EVAL_PROTOCOL = "smolvlm-dpo-v1/held-out-implicit-reward-v1"
EVAL_VARIANT = "pairwise-implicit-reward"
RUNTIME_PACKAGES = ("torch", "transformers", "datasets", "peft", "bitsandbytes", "safetensors")
METRICS = pc.METRICS
PREDICTIONS_FILE = "predictions.json"
EVAL_MANIFEST_FILE = "eval_manifest.json"
PROCESSOR_IDENTITY_FILE = "processor_identity.json"
REFERENCE_FILES = ("source_manifest.json", PROCESSOR_IDENTITY_FILE, EVAL_MANIFEST_FILE)
REFERENCE_CONFIG_FILE = "reference-policy.json"
FULL_STEPS = 150
SMOKE_STEPS = 5
ALGORITHM_VERSION = "smolvlm-dpo-v1"

DPO_SPEC = {
    "protocol": PROTOCOL, "source_split": "train[:6%]", "held_out_split": "test[:1%]",
    "split_percent": [80, 10, 10], "split_algorithm": pc.SPLIT_ALGORITHM,
    "beta": 0.1, "loss_type": "sigmoid", "max_eval_examples": 300,
    "loss": "显式 beta 的 sigmoid DPO loss；policy/ref 仅在 completion token 上计 logprob，image placeholder 与 padding label=-100",
    "prompt_format": "system + user(images+prompt)；chosen/rejected 绝不进生成 prompt",
    "metric_definition": "win_rate/reward_accuracy 均为 policy implicit reward pairwise，不是外部 reward model/human eval",
    "upstream_beta": "absent（DPOConfig 未给 beta，依赖 TRL 0.12.2 默认值；本仓自建 0.1）",
    "upstream_loss_type": "absent（本仓自建 sigmoid，不沿用未冻结的默认实现）",
    "upstream_learning_rate": "absent（本仓训练 learning_rate 为自建值）",
    "upstream_max_length": "absent（本仓用自建 seq_length）",
    "upstream_ref_model": "None（本仓显式冻结 ref，不与 policy 共享 adapter）",
    "upstream_evaluation": "仅定性生成，无 compute_metrics；本仓补 held-out 隐式奖励 pairwise 指标",
    "upstream_merge": "PeftModel.from_pretrained only；无 merge_and_unload；本仓 adapter-only 打包",
    "upstream_dataset_split": "train[:6%] / test[:1%]；本仓按图像 hash 重划 80/10/10",
    "upstream_deps": "transformers==4.46.3 / trl==0.12.2 / datasets==3.2.0 / bitsandbytes==0.45.0 / peft==0.14.0 / accelerate==1.2.0 来自注释；flash-attn 无版本",
    "cross_backbone_attribution_valid": False, "self_migration": True,
}

METHOD_KEYS = frozenset({"family", "trainer", "upstream_trainer", "peft", "execution_trainer"})
MODEL_KEYS = frozenset({"base", "modality", "revision", "params_b"})
DATASET_KEYS = frozenset({"name", "structure", "type", "has_images", "kind", "revision"})
MODALITY_KEYS = frozenset({"images_column", "content_list_dict", "processing_class", "max_length"})
PEFT_KEYS = frozenset({"r", "alpha", "dropout", "target_module_patterns", "use_dora",
                       "init_lora_weights", "bias", "task_type"})
QUANT_KEYS = frozenset({"load_in_4bit", "bnb_4bit_quant_type",
                        "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype"})
UPSTREAM_KEYS = frozenset({"split", "mask", "evaluation", "revision", "seed", "merge",
                           "attn", "target_modules", "trainer", "processing_class"})
TRAIN_KEYS = frozenset({
    "seed", "data_seed", "seed_mechanism", "num_train_epochs", "per_device_train_batch_size",
    "per_device_eval_batch_size", "gradient_accumulation_steps", "gradient_checkpointing",
    "dataset_num_proc", "dataloader_num_workers", "eval_strategy", "eval_steps",
    "save_steps", "logging_steps", "learning_rate", "lr_scheduler_type", "weight_decay",
    "bf16", "fp16", "seq_length", "remove_unused_columns",
})
EVALUATION_KEYS = frozenset({"metrics", "split", "seed", "multi_run", "held_out",
                             "compute_metrics", "max_new_tokens", "expected", "tolerance",
                             "export_tolerance"})
# 这些键是协议元数据或上游遗留项，不进入普通 Trainer 的 TrainingArguments。
METADATA_TRAIN_KEYS = frozenset({"seed", "data_seed", "seed_mechanism", "num_train_epochs",
                                 "dataset_num_proc", "seq_length"})


def resolve_task_id(r, plan=None, task_id=None):
    declared = task_id or (plan or {}).get("task_id") or r.get("execution.task_id")
    return declared or (TASK_ID if r.id == RECIPE_ID else None)


def require_supported(r, plan=None, task_id=None) -> str:
    """严格校验 recipe / plan / 协议 / 依赖 / 预算；任何漂移都失败关闭。"""
    resolved = resolve_task_id(r, plan, task_id)
    if r.id != RECIPE_ID or resolved != TASK_ID or (plan or {}).get("recipe_id", r.id) != r.id:
        raise TaskExecutionError(f"尚未实现该执行路由：{r.id} / {resolved!r}")
    required = {
        "method.family": "dpo", "method.trainer": "DPOTrainer",
        "method.upstream_trainer": "DPOTrainer", "method.peft": "qlora-dora",
        "method.execution_trainer": "Trainer",
        "model.base": MODEL, "dataset.name": DATASET,
    }
    for key, expected in required.items():
        if r.get(key) != expected:
            raise TaskExecutionError(f"{TASK_ID} 要求 {key}={expected!r}")
        section, field = key.split(".")
        planned = (plan or {}).get(section, {})
        if field in planned and planned[field] != expected:
            raise TaskExecutionError(f"plan {key} 与 DPO 路由不一致")
    if not r.is_vlm or r.get("dataset.has_images") is not True or r.get("dataset.type") != "preference":
        raise TaskExecutionError("DPO 只接受带图像的静态偏好数据")
    if r.get("dataset.kind", "static") != "static":
        raise TaskExecutionError("DPO 只接受冻结的静态偏好数据集")
    for name, keys in (("method", METHOD_KEYS), ("model", MODEL_KEYS), ("dataset", DATASET_KEYS),
                       ("preference", DPO_SPEC), ("peft", PEFT_KEYS), ("quantization", QUANT_KEYS),
                       ("upstream", UPSTREAM_KEYS), ("budget", {"full_steps", "smoke_steps"})):
        section = r.get(name)
        if name == "preference":
            pc.unknown(section, keys, name)
            for key, expected in keys.items():
                if key not in section or section[key] != expected or type(section[key]) is not type(expected):
                    raise TaskExecutionError(f"preference.{key} 必须为 {expected!r}")
        else:
            pc.unknown(section, keys, name)
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
    train = pc.require_keys(r.get("train"), TRAIN_KEYS, "train")
    pc.unknown(r.get("evaluation"), EVALUATION_KEYS, "evaluation")
    for key, expected in {
        "train.bf16": True, "train.fp16": False, "train.gradient_checkpointing": True,
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
                "gradient_accumulation_steps", "eval_steps", "save_steps", "logging_steps",
                "dataset_num_proc"):
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
    if (type(train["max_steps"]) if "max_steps" in train else type(r.get("budget.full_steps"))) is not int \
            or r.get("budget.full_steps") != FULL_STEPS or r.get("budget.smoke_steps") != SMOKE_STEPS:
        raise TaskExecutionError("预算必须显式 full=150 / smoke=5；不可静默改预算")
    dependencies = r.get("dependencies", {}) or {}
    pc.unknown(dependencies, ("trl", "transformers", "datasets", "bitsandbytes", "peft",
                              "accelerate", "qwen-vl-utils"), "dependencies")
    for name, version in dependencies.items():
        if not isinstance(version, str) or not version.strip() or version in ("unknown", "latest", "*"):
            raise TaskExecutionError(f"dependencies.{name} 不能猜测或使用未冻结版本")
    return resolved


def split_spec(r):
    return {**copy.deepcopy(DPO_SPEC), "data_seed": fixed_seeds(r)[1],
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
            "trainer": "普通 Trainer + 自建 beta sigmoid DPO loss（不依赖 TRL 0.12.2 未固定源码）",
            "beta": "显式 0.1（上游 DPOConfig 未给，依赖默认值）",
            "ref": "冻结 ref + policy 双模型身份绑定（上游 ref_model=None）",
            "split": "共享偏好 image-hash 80/10/10（上游 train[:6%] / test[:1%]）",
            "evaluation": "held-out 隐式奖励 pairwise（上游仅定性生成、无 compute_metrics）",
            "cross_backbone": "与 MPO 篇基座不同，cross_backbone_attribution_valid=false",
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


# ---------------------------------------------------------------- 数据与编码

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


def encode_pair(row, processor, identity, r, key):
    return pc.encode_pair(row, processor, identity, r.get("method.system_message", SYSTEM_MESSAGE),
                          key, r.get("train.seq_length"))


SYSTEM_MESSAGE = ("You are a helpful assistant. Choose the better response to the user's question "
                  "about the provided image.")


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
    transform = pc.hash_manifest({
        "schema_version": 1, "task_id": TASK_ID, "protocol": PROTOCOL,
        "processor_identity": identity, "data_seed": r.get("train.data_seed"),
        "beta": r.get("preference.beta"), "loss_type": r.get("preference.loss_type"),
        "loss": r.get("preference.loss"), "prompt_format": r.get("preference.prompt_format"),
        "split_algorithm": pc.SPLIT_ALGORITHM, "entries": entries,
    })
    return examples, transform


def validate_transform_manifest(transform, source, r, identity):
    if (not isinstance(transform, dict) or transform.get("task_id") != TASK_ID
            or transform.get("protocol") != PROTOCOL
            or transform.get("processor_identity") != identity
            or transform.get("beta") != r.get("preference.beta")
            or transform.get("loss_type") != r.get("preference.loss_type")
            or transform.get("loss") != r.get("preference.loss")
            or transform.get("data_seed") != r.get("train.data_seed")
            or transform.get("sha256") != pc.hash_manifest(transform)["sha256"]):
        raise TaskExecutionError("transform manifest 与当前 DPO 协议不一致")
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


# ---------------------------------------------------------------- 自建 DPO loss

def _softplus_neg(x):
    """log(1+exp(x)) 的稳定实现：x 是 -beta*logits。"""
    if x >= 0:
        return math.log1p(math.exp(-x))
    return -x + math.log1p(math.exp(x))


def sigmoid_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta, torch=None):
    """-log sigmoid(beta * ((logpi_c - logpi_r) - (logref_c - logref_r)))。"""
    if type(beta) not in (int, float) or beta <= 0 or not math.isfinite(beta):
        raise TaskExecutionError("beta 必须为正有限数值；拒绝依赖未冻结默认值")
    logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    scaled = beta * logits
    if torch is not None:
        import torch as _torch  # 仅在真实运行时调用
        return _torch.nn.functional.softplus(-scaled).mean()
    return _softplus_neg(float(scaled))


def implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """policy implicit reward = beta * (logpi - logref)，用于 pairwise 评估。"""
    if type(beta) not in (int, float) or beta <= 0:
        raise TaskExecutionError("beta 必须为正有限数值")
    return (beta * (float(policy_chosen) - float(ref_chosen)),
            beta * (float(policy_rejected) - float(ref_rejected)))


def sequence_logprob(model, input_ids, attention_mask, pixel_values):
    """优先用显式接口；否则用模型 logits 与 labels 计算 completion-only logprob。"""
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
    """policy/ref 必须是两个不同模型；ref 全冻结；policy 只训练 LoRA 参数。"""
    if policy is ref:
        raise TaskExecutionError("ref 与 policy 是同一模型；ref 未独立冻结")
    ref_trainable = [name for name, parameter in ref.named_parameters() if parameter.requires_grad]
    if ref_trainable:
        raise TaskExecutionError(f"ref 模型存在可训练参数 {ref_trainable[:3]}；ref 必须冻结且无梯度")
    policy_trainable = [(name, parameter) for name, parameter in policy.named_parameters()
                        if parameter.requires_grad]
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
        "images_per_sample": "variable",
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
    return model, {"matched_target_modules": sorted(targets), "trainable_parameters": trainable,
                   "base_parameters_frozen": True, "use_dora": True}


def make_trainer_class(base_trainer):
    """普通 Trainer 子类：compute_loss 用自建 beta sigmoid DPO loss。"""
    class PreferenceTrainer(base_trainer):
        dpo_beta = None
        dpo_torch = None
        dpo_calls = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            pc._need(self.dpo_beta is not None, "DPO beta 未绑定；拒绝无 beta 训练")
            pc._need({"policy_chosen_logps", "policy_rejected_logps",
                      "ref_chosen_logps", "ref_rejected_logps"} <= set(inputs),
                     "DPO 训练 batch 缺少 policy/ref 双侧 logps")
            loss = sigmoid_dpo_loss(
                inputs["policy_chosen_logps"], inputs["policy_rejected_logps"],
                inputs["ref_chosen_logps"], inputs["ref_rejected_logps"],
                self.dpo_beta, torch=self.dpo_torch)
            PreferenceTrainer.dpo_calls += 1
            return loss

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
    TrainerClass = make_trainer_class(transformers.Trainer)
    trainer = TrainerClass(
        model=policy, args=args,
        train_dataset=runtime["datasets"].Dataset.from_list(examples["train"]),
        eval_dataset=runtime["datasets"].Dataset.from_list(examples["validation"]),
        data_collator=lambda items: pc.collate_pairs(
            items, torch, processor_identity_value["pad_token_id"], processor_identity_value["image_token_id"]))
    trainer.dpo_beta = r.get("preference.beta")
    trainer.dpo_torch = torch
    result = trainer.train()
    actual_steps = trainer.state.global_step
    if type(actual_steps) is not int or actual_steps != canonical["max_steps"]:
        raise TaskExecutionError(f"训练未完成预算：实际 {actual_steps}，要求 {canonical['max_steps']}")
    if TrainerClass.dpo_calls < 1:
        raise TaskExecutionError("自建 DPO loss 未被调用；拒绝保存未训练 adapter")
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
        "beta": r.get("preference.beta"), "loss_type": r.get("preference.loss_type"),
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
    save_json(output / "dpo_transform_manifest.json", transform)
    torch.cuda.synchronize()
    evidence = pc.completed(r, plan, identity, "training", effective_config, source_manifest,
                            runtime_evidence(runtime, started), TASK_ID)
    evidence.update(steps=actual_steps, max_steps=canonical["max_steps"], trainer="Trainer",
                    processor_identity=processor_identity_value, transformation_manifest=transform,
                    train_metrics=dict(result.metrics), model_artifact=pc.artifact_manifest(output),
                    beta=r.get("preference.beta"), ref_binding=ref_binding, ref_identity=ref_identity)
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
        from .model import Recipe
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
        raise TaskExecutionError("checkpoint 不是已完成 DPO full/smoke 训练")
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
    if (evidence.get("config") != config or evidence.get("config_sha256") != sha256_json(config)
            or training_arguments_identity(config.get("training_arguments"))
            != training_arguments_identity(expected_args)
            or config.get("preference") != r.get("preference")
            or config.get("beta") != r.get("preference.beta")
            or config.get("loss_type") != r.get("preference.loss_type")
            or config.get("peft") != r.get("peft")
            or config.get("quantization") != r.get("quantization")
            or config.get("attention_implementation") != "eager"
            or config.get("upstream_metadata") != dict(r.get("upstream"))
            or type(evidence.get("steps")) is not int
            or evidence["steps"] != expected_args["max_steps"]):
        raise TaskExecutionError("checkpoint steps/config hash/预算或 beta 绑定不一致")
    # ref/policy 分离与 beta 绑定：发布级与 checkpoint 强校验必须显式记录。
    binding = config.get("ref_binding", {})
    ref_identity = config.get("ref_identity", {})
    if (binding.get("policy_ref_disjoint") is not True
            or binding.get("ref_trainable_parameters") != 0
            or type(binding.get("policy_trainable_parameters")) is not int
            or binding["policy_trainable_parameters"] <= 0
            or ref_identity.get("sha256") != pc.hash_manifest(ref_identity)["sha256"]
            or ref_identity.get("ref_trainable_parameters") != 0
            or evidence.get("ref_binding") != binding
            or evidence.get("ref_identity") != ref_identity):
        raise TaskExecutionError("checkpoint ref/policy 分离或 beta/ref 绑定证据缺失")
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
    transform = pc.read_json(path / "dpo_transform_manifest.json")
    if (not isinstance(processor, dict) or processor.get("sha256") != pc.hash_manifest(processor)["sha256"]
            or evidence.get("processor_identity") != processor
            or processor.get("image_token_id") != config.get("image_token_id")
            or type(processor.get("image_token_id")) is not int
            or ref_file != ref_identity or evidence.get("ref_identity") != ref_file
            or transform.get("sha256") != pc.hash_manifest(transform)["sha256"]
            or transform.get("processor_identity") != processor
            or transform.get("protocol") != PROTOCOL
            or transform.get("beta") != r.get("preference.beta")
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


def evaluation_manifest(examples, identity, r, source):
    """examples 已经是 pc.evaluation_examples 产出；此处仅绑定任务/协议身份。"""
    payload = dict(examples)
    payload["task_id"] = TASK_ID
    payload["evaluation_protocol"] = EVAL_PROTOCOL
    payload["variant"] = EVAL_VARIANT
    payload["beta"] = r.get("preference.beta")
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


def score_example(row, processor, policy, ref, torch, identity, beta):
    """同一 prompt 同图像，teacher-forced 计算 chosen/rejected 的 policy/ref logprob。"""
    prompt_ids, chosen_completion, pixel = _encode_completion(row, processor, row["chosen"])
    _, rejected_completion, _ = _encode_completion(row, processor, row["rejected"])
    if not chosen_completion or not rejected_completion:
        return pc.score_pair(None, None, 0, 0, sample_id=row["sample_id"],
                             source_index=row["source_index"], image_sha256=row["image_sha256"],
                             prompt_sha256=sha256_json(pc.render_prompt(row, processor, SYSTEM_MESSAGE)),
                             chosen_sha256=sha256_json(row["chosen"]),
                             rejected_sha256=sha256_json(row["rejected"]),
                             generated_text="", generated_code_executed=False)
    policy_c = sequence_logprob(policy, **_sequence_inputs(torch, prompt_ids, chosen_completion, pixel))
    policy_r = sequence_logprob(policy, **_sequence_inputs(torch, prompt_ids, rejected_completion, pixel))
    ref_c = sequence_logprob(ref, **_sequence_inputs(torch, prompt_ids, chosen_completion, pixel))
    ref_r = sequence_logprob(ref, **_sequence_inputs(torch, prompt_ids, rejected_completion, pixel))
    chosen_reward, rejected_reward = implicit_rewards(policy_c, policy_r, ref_c, ref_r, beta)
    prompt_text = pc.render_prompt(row, processor, SYSTEM_MESSAGE)
    return pc.score_pair(
        chosen_reward, rejected_reward, len(chosen_completion), len(rejected_completion),
        sample_id=row["sample_id"], source_index=row["source_index"],
        image_sha256=row["image_sha256"], prompt_sha256=sha256_json(prompt_text),
        chosen_sha256=sha256_json(row["chosen"]), rejected_sha256=sha256_json(row["rejected"]),
        generated_text="", generated_code_executed=False,
        policy_chosen_logps=float(policy_c), policy_rejected_logps=float(policy_r),
        ref_chosen_logps=float(ref_c), ref_rejected_logps=float(ref_r),
        chosen_reward=chosen_reward, rejected_reward=rejected_reward,
    )


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
    beta = r.get("preference.beta")
    rows = splits["test"]
    examples = evaluation_examples(rows, processor, processor_identity_value, r, manifest)
    samples = examples["samples"]
    predictions = []
    for row, sample in zip(rows, samples):
        record = score_example(row, processor, policy, ref, torch, processor_identity_value, beta)
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
        "beta": beta, "loss_type": r.get("preference.loss_type"),
        "ref_binding": binding,
        "quantization": r.get("quantization"), "compute_dtype": "bfloat16",
        "attention_implementation": "eager", "model": model_artifact,
        "processor_identity": processor_identity_value,
        "dataset_revision": r.get("dataset.revision"),
        "dataset_id": DATASET, "source_manifest_sha256": manifest["sha256"],
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
        length_bias_diagnostic=bias, sample_count=len(predictions),
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
