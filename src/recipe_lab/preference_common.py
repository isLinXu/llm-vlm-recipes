"""VLM 偏好训练（DPO / MPO）共享校验层；标准库、失败关闭。

本层只放与任务身份无关的通用能力：严格 schema、图像原始字节 hash、图像级
80/10/10 划分、prompt/full 两次 apply_chat_template 的 token 前缀校验、
显式 image_token_id 双层 label=-100、隐式奖励 pairwise 指标、eval manifest
构造、文件/artifact 绑定。任务模块必须显式绑定自己的 RECIPE_ID / TASK_ID /
协议段（beta/ref 或 loss_types/weights/ablation），不得互相借用身份。

一手事实边界：DPO 与 MPO 两篇基座不同（SmolVLM vs Qwen2.5-VL），本层只允许
各自任务内的 3 seed 同协议聚合；跨篇对照必须显式标 cross_backbone_attribution_valid=false。
两篇都是自建迁移 / 自建协议，不冒充上游 notebook 复现；fake runtime 离线测试
只证明内部一致性，不认证 GPU 执行。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from . import math_task as mt
from . import xlam_task as xt

TaskExecutionError = mt.TaskExecutionError
DatasetValidationError = mt.DatasetValidationError
sha256_json = mt.sha256_json
fixed_seeds = mt.fixed_seeds
require_revision = mt.require_revision
check_signature = mt.check_signature
apply_determinism = mt.apply_determinism
execution_identity = mt.execution_identity
mode_output_dir = mt.mode_output_dir
save_json = mt.save_json

# 两篇共用的偏好划分协议：dataset id + revision + data_seed + split 算法版本 +
# split_hashes 构成 split_identity。算法版本是「共享偏好划分」版本，不是训练算法；
# 两篇 seed/revision 相同即可校验 shared preference split，但基座不同不可归因算法。
SPLIT_ALGORITHM = "rlaif-v-preference-split-v1"
SPLIT_PERCENT = [80, 10, 10]
PREFERENCE_COLUMNS = frozenset({"prompt", "chosen", "rejected", "images"})


def _need(condition, message):
    if not condition:
        raise TaskExecutionError(message)


def canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def hash_manifest(value):
    """返回「原内容 + 其内容 sha256」，不是只返回 hash（与 xlam/granite 一致）。"""
    payload = {k: v for k, v in value.items() if k != "sha256"}
    return {**value, "sha256": sha256_json(payload)}


def unknown(section, allowed, label):
    """严格未知字段拒绝：任何未声明字段都不静默套默认值。"""
    if not isinstance(section, dict):
        raise TaskExecutionError(f"[{label}] 缺失或不是配置表")
    extra = sorted(set(section) - set(allowed))
    if extra:
        raise TaskExecutionError(f"[{label}] 出现未声明字段 {extra}；拒绝静默套默认")


def require_keys(section, allowed, label):
    unknown(section, allowed, label)
    missing = sorted(set(allowed) - set(section))
    if missing:
        raise TaskExecutionError(f"[{label}] 缺少参数 {missing}")
    return section


# ---------------------------------------------------------------- 图像与 schema

def normalized_text(value):
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _message_text(value):
    """rlaif-v_formatted 的 prompt/chosen/rejected 既可能是 str 也可能是 chat list。"""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif isinstance(item, dict) and isinstance(item.get("content"), list):
                for piece in item["content"]:
                    if isinstance(piece, dict) and isinstance(piece.get("text"), str):
                        parts.append(piece["text"])
                    elif isinstance(piece, str):
                        parts.append(piece)
            else:
                raise DatasetValidationError("prompt/chosen/rejected 消息项结构非法")
        text = "".join(parts)
        if text.strip():
            return text
    raise DatasetValidationError("prompt/chosen/rejected 必须是非空文本或消息列表")


def image_payload(image):
    """返回 (mode, raw_bytes)：hash 只基于原始输入，RGB 转换发生在送入 processor 之前。"""
    if isinstance(image, (bytes, bytearray, memoryview)):
        return "bytes", bytes(image)
    mode = getattr(image, "mode", None)
    tobytes = getattr(image, "tobytes", None)
    if not isinstance(mode, str) or not callable(tobytes):
        raise DatasetValidationError("images 元素必须是 bytes 或提供 mode/tobytes() 的图像对象")
    return mode, bytes(tobytes())


def ensure_rgb(image):
    mode = getattr(image, "mode", "bytes")
    if mode in ("RGB", "bytes"):
        return image
    convert = getattr(image, "convert", None)
    if not callable(convert):
        raise DatasetValidationError("非 RGB 图像必须提供 convert('RGB')")
    return convert("RGB")


def image_hash(image):
    mode, payload = image_payload(image)
    return hashlib.sha256(mode.encode("utf-8") + b"\x00" + payload).hexdigest()


def parse_preference_row(raw, index):
    """严格单行偏好校验：只接受 prompt/chosen/rejected/images，拒绝未知/空/非法图像。"""
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"source_index={index} 不是映射行")
    extra = sorted(set(raw) - PREFERENCE_COLUMNS)
    if extra:
        raise DatasetValidationError(f"source_index={index} 出现未知字段 {extra}；拒绝静默忽略")
    missing = sorted(PREFERENCE_COLUMNS - set(raw))
    if missing:
        raise DatasetValidationError(f"source_index={index} 缺少字段 {missing}")
    prompt_text = _message_text(raw["prompt"])
    chosen = _message_text(raw["chosen"])
    rejected = _message_text(raw["rejected"])
    if not prompt_text.strip() or not chosen.strip() or not rejected.strip():
        raise DatasetValidationError(f"source_index={index} prompt/chosen/rejected 不得为空")
    if normalized_text(chosen) == normalized_text(rejected):
        raise DatasetValidationError(f"source_index={index} chosen 与 rejected 规范化后相同；拒绝退化偏好对")
    images = raw["images"]
    if not isinstance(images, (list, tuple)) or not images:
        raise DatasetValidationError(f"source_index={index} images 必须是非空列表")
    image_hashes, image_modes, image_bytes = [], [], []
    for image in images:
        mode, payload = image_payload(image)
        image_modes.append(mode)
        image_bytes.append(len(payload))
        image_hashes.append(hashlib.sha256(mode.encode("utf-8") + b"\x00" + payload).hexdigest())
    combined = hashlib.sha256(json.dumps(
        [[m, h] for m, h in zip(image_modes, image_hashes)], ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    content = normalized_text(prompt_text) + "\x00" + normalized_text(chosen) + "\x00" + normalized_text(rejected)
    return {
        "source_index": index, "prompt_text": prompt_text, "chosen": chosen, "rejected": rejected,
        "images": list(images), "image_hashes": image_hashes, "image_modes": image_modes,
        "image_bytes": image_bytes, "image_sha256": combined,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "sample_id": f"{combined[:16]}-{index}",
    }


# ---------------------------------------------------------------- 图像级划分

def split_bucket(image_sha256, data_seed):
    digest = hashlib.sha256(
        f"{SPLIT_ALGORITHM}|{data_seed}|{image_sha256}".encode("utf-8")).hexdigest()
    bucket = int(digest[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def prepare_splits(rows, r, *, max_eval_examples):
    """图像级隔离 + 规范化偏好对全局去重 + 确定性 80/10/10；空 split 拒绝，不随机补救。"""
    from .math_task import fixed_seeds as _seeds
    _, data_seed = _seeds(r)
    if type(max_eval_examples) is not int or max_eval_examples < 1:
        raise DatasetValidationError("max_eval_examples 必须为正整数")
    splits = {"train": [], "validation": [], "test": []}
    seen, image_owner, duplicates = {}, {}, []
    for index, raw in enumerate(rows):
        row = parse_preference_row(raw, index)
        # 同一规范化偏好对全局只出现一次；跨 split 重复也拒绝。
        if row["content_sha256"] in seen:
            duplicates.append({"first": seen[row["content_sha256"]], "again": index,
                               "sha256": row["content_sha256"]})
            continue
        seen[row["content_sha256"]] = index
        bucket = split_bucket(row["image_sha256"], data_seed)
        previous = image_owner.get(row["image_sha256"])
        if previous is not None and previous != bucket:
            raise DatasetValidationError("同一图像被分到不同 split；共享偏好划分规则或 hash 漂移")
        image_owner[row["image_sha256"]] = bucket
        splits[bucket].append(row)
    if duplicates:
        raise DatasetValidationError(
            f"检测到 {len(duplicates)} 条规范化偏好对重复（含跨 split）；拒绝静默去重，先复核数据源与 revision")
    hashes = {name: [row["image_sha256"] for row in items] for name, items in splits.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(hashes[left]) & set(hashes[right]):
            raise DatasetValidationError("同图跨 split 泄漏；图像内容 hash 隔离失败")
    for name, items in splits.items():
        if not items:
            raise DatasetValidationError(f"{name} split 为空；拒绝用其他 split 偷偷补救")
    manifest = {
        "schema_version": 1, "algorithm": SPLIT_ALGORITHM, "split_percent": list(SPLIT_PERCENT),
        "data_seed": data_seed, "columns": sorted(PREFERENCE_COLUMNS),
        "duplicate_policy": "规范化 (prompt,chosen,rejected) 全局唯一；跨 split 重复与同图跨 split 一律拒绝",
        "image_hash": "每图 mode + tobytes() 的 SHA256，再组合为偏好对 image_sha256",
        "duplicates_rejected": duplicates,
        "image_split_owner": {h: split_bucket(h, data_seed) for h in image_owner},
        "counts": {"source": len(seen), "train": len(splits["train"]),
                   "validation": len(splits["validation"]), "test": len(splits["test"]),
                   "test_eval_used": min(len(splits["test"]), max_eval_examples)},
        "split_hashes": hashes,
    }
    manifest["split_hashes_sha256"] = sha256_json(manifest["split_hashes"])
    manifest["sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    return {"train": splits["train"], "validation": splits["validation"],
            "test": splits["test"][:max_eval_examples]}, manifest


def validate_source_manifest(manifest, r):
    if not isinstance(manifest, dict):
        raise DatasetValidationError("source_manifest 缺失")
    computed = sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    if (manifest.get("sha256") != computed or manifest.get("schema_version") != 1
            or manifest.get("algorithm") != SPLIT_ALGORITHM
            or manifest.get("split_percent") != list(SPLIT_PERCENT)
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
    used = counts.get("test_eval_used")
    if type(used) is not int or not 0 < used <= counts["test"]:
        raise DatasetValidationError("source_manifest test_eval_used 非法")
    return manifest


def split_identity(manifest, *, dataset_id, dataset_revision, algorithm_version):
    """共享偏好 split 身份：dataset/revision/data_seed/算法版本/split_hashes。"""
    return {"dataset_id": dataset_id, "dataset_revision": dataset_revision,
            "data_seed": manifest["data_seed"], "algorithm": SPLIT_ALGORITHM,
            "algorithm_version": algorithm_version,
            "split_hashes_sha256": manifest["split_hashes_sha256"]}


def validate_shared_preference_split(dpo_manifest, mpo_manifest, *, dpo_identity, mpo_identity):
    """校验两篇共享冻结偏好 split；backbone 不同，跨篇归因必须 false。"""
    _need(dpo_identity == mpo_identity, "两篇共享偏好 split 身份不一致；不得跨篇对照")
    _need(dpo_manifest["split_hashes_sha256"] == mpo_manifest["split_hashes_sha256"],
          "两篇共享偏好 split 的 split_hashes 不一致")
    return {"shared_preference_split_valid": True,
            "cross_backbone_attribution_valid": False,
            "reason": "DPO 使用 SmolVLM、MPO 使用 Qwen2.5-VL；基座不同，不能把指标差异归因 DPO/MPO 算法"}


# ---------------------------------------------------------------- 图文编码

def processor_identity(processor, protocol):
    tokenizer = getattr(processor, "tokenizer", None)
    image_token_id = getattr(processor, "image_token_id", None)
    if type(image_token_id) is not int:
        raise DatasetValidationError("processor 缺少显式 image_token_id；拒绝退回隐式位置 mask")
    if tokenizer is None:
        raise DatasetValidationError("processor 缺少 tokenizer")
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
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "protocol": protocol,
    }
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


def render_messages(row, system_message, *, completion=None):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": (
            [{"type": "image", "image": image} for image in row["images"]]
            + [{"type": "text", "text": row["prompt_text"]}])},
    ]
    if completion is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
    return messages


def render_prompt(row, processor, system_message):
    text = processor.apply_chat_template(
        render_messages(row, system_message), tokenize=False, add_generation_prompt=True)
    if not isinstance(text, str) or not text:
        raise DatasetValidationError("processor.apply_chat_template 未返回有效 prompt 文本")
    if row["chosen"] in text or row["rejected"] in text:
        raise DatasetValidationError("gold chosen/rejected 泄漏进生成 prompt；拒绝执行")
    return text


def _processor_call(processor, text, images):
    result = processor(text=text, images=[list(images)], return_tensors=None, padding=False)
    if not isinstance(result, dict) or "input_ids" not in result:
        raise DatasetValidationError("processor 未返回 input_ids")
    ids = result["input_ids"]
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    ids = [int(token) for token in ids]
    return ids, result


def _shape(value):
    if isinstance(value, (list, tuple)) and value:
        head = value[0]
        inner = _shape(head) if isinstance(head, (list, tuple)) else ()
        return (len(value),) + inner
    return ()


def encode_pair(row, processor, identity, system_message, completion_key, seq_length, *, require_gold=True):
    """prompt/full 两次编码 + token 级前缀校验 + image_token_id 双层 -100。"""
    completion = row[completion_key]
    prompt = render_prompt(row, processor, system_message)
    full = processor.apply_chat_template(
        render_messages(row, system_message, completion=completion),
        tokenize=False, add_generation_prompt=False)
    if require_gold and (not isinstance(full, str) or completion not in full):
        raise DatasetValidationError("完整 chat 序列不含 completion；拒绝组装 completion-only 标签")
    prompt_ids, _ = _processor_call(processor, prompt, row["images"])
    full_ids, encoded = _processor_call(processor, full, row["images"])
    if not prompt_ids or len(full_ids) <= len(prompt_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
        raise DatasetValidationError("full 序列不是 prompt 的 token 前缀；无法确定 completion-only 边界")
    if len(full_ids) > seq_length:
        raise DatasetValidationError(
            f"source_index={row['source_index']} 长度 {len(full_ids)} 超过 seq_length={seq_length}；拒绝截断")
    image_token_id = identity["image_token_id"]
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
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
        "input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids),
        "pixel_values": canonical(pixels), "pixel_shape": list(shape),
        "prompt_ids": prompt_ids, "prompt_text": prompt,
        "completion_ids": list(full_ids[len(prompt_ids):]), "completion": completion,
        "prompt_tokens": len(prompt_ids), "completion_tokens": len(full_ids) - len(prompt_ids),
        "masked_image_tokens": masked_images, "sample_id": row["sample_id"],
        "image_sha256": row["image_sha256"], "gold_sha256": sha256_json(completion),
    }


def collate_pairs(items, torch, pad_token_id, image_token_id):
    """右 padding；padding / image token 全部 -100；pixel_values 形状必须一致。"""
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
    if any(len(value) != width for row in batch.values() for value in row):
        raise DatasetValidationError("collate 长度不一致")
    out = {key: torch.tensor(batch[key], dtype=torch.long) for key in ("input_ids", "labels", "attention_mask")}
    out["pixel_values"] = torch.tensor([item["pixel_values"] for item in items], dtype=torch.float32)
    return out


# ---------------------------------------------------------------- 隐式奖励指标

METRICS = ("win_rate", "reward_accuracy")


def length_bucket(chosen_tokens, rejected_tokens):
    if chosen_tokens > rejected_tokens:
        return "chosen_longer"
    if chosen_tokens < rejected_tokens:
        return "rejected_longer"
    return "equal_length"


def score_pair(chosen_reward, rejected_reward, chosen_tokens, rejected_tokens, **extra):
    """policy implicit reward 的 pairwise 判定；空生成/解析失败显式标 invalid 且不给分。"""
    valid = (isinstance(chosen_reward, (int, float)) and isinstance(rejected_reward, (int, float))
             and math.isfinite(float(chosen_reward)) and math.isfinite(float(rejected_reward)))
    win = 0.0
    strict = 0.0
    if valid:
        win = 1.0 if chosen_reward > rejected_reward else 0.5 if chosen_reward == rejected_reward else 0.0
        strict = 1.0 if chosen_reward > rejected_reward else 0.0
    record = {
        "win": win, "strict_win": strict, "valid": bool(valid),
        "chosen_reward": chosen_reward if valid else None,
        "rejected_reward": rejected_reward if valid else None,
        "length_bucket": length_bucket(chosen_tokens, rejected_tokens),
        "chosen_tokens": chosen_tokens, "rejected_tokens": rejected_tokens,
    }
    record.update(extra)
    return record


def recompute_metrics(predictions):
    """win_rate = policy implicit reward 选 chosen 比例；reward_accuracy = 与数据卡 chosen 标签一致的严格 pairwise 准确率。

    两者都是隐式奖励 pairwise，不是外部 reward model / human eval；空生成或解析失败记 0 分。
    """
    if not isinstance(predictions, list) or not predictions:
        raise DatasetValidationError("逐题预测为空，不能计算指标")
    totals = {name: 0.0 for name in METRICS}
    for index, record in enumerate(predictions):
        if not isinstance(record, dict):
            raise DatasetValidationError(f"第 {index} 条逐题预测不是 dict；拒绝重算指标")
        for key in ("win", "strict_win"):
            value = record.get(key)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise DatasetValidationError(f"逐题 {key} 非法")
        totals["win_rate"] += float(record["win"])
        totals["reward_accuracy"] += float(record["strict_win"])
    return {name: totals[name] / len(predictions) for name in METRICS}


def length_bias_diagnostic(predictions):
    buckets = {}
    for record in predictions:
        name = record.get("length_bucket")
        if name not in ("chosen_longer", "rejected_longer", "equal_length"):
            raise DatasetValidationError("逐题 length_bucket 非法")
        entry = buckets.setdefault(name, {"count": 0, "win_sum": 0.0})
        entry["count"] += 1
        entry["win_sum"] += float(record["win"])
    return {name: {"count": entry["count"],
                   "win_rate": entry["win_sum"] / entry["count"] if entry["count"] else 0.0}
            for name, entry in buckets.items()}


def evaluation_examples(rows, processor, identity, system_message, source, protocol, max_new_tokens):
    """构造可离线重建的 held-out chosen/rejected 同 prompt 同图像输入。"""
    samples = []
    for row in rows:
        prompt = render_prompt(row, processor, system_message)
        prompt_ids, encoded = _processor_call(processor, prompt, row["images"])
        if not prompt_ids or not all(type(token) is int and token >= 0 for token in prompt_ids):
            raise DatasetValidationError("评估 prompt token 化结果为空或非法")
        samples.append({
            "sample_id": row["sample_id"], "source_index": row["source_index"],
            "images_column": "images", "image_sha256": row["image_sha256"],
            "raw": {"prompt": row["prompt_text"], "chosen": row["chosen"], "rejected": row["rejected"],
                    "image_hashes": row["image_hashes"]},
            "prompt": prompt, "prompt_sha256": sha256_json(prompt), "prompt_ids": prompt_ids,
            "chosen_sha256": sha256_json(row["chosen"]),
            "rejected_sha256": sha256_json(row["rejected"]),
            "pixel_shape": list(_shape(encoded.get("pixel_values"))),
        })
    payload = {
        "schema_version": 1, "kind": "preference-eval-manifest", "protocol": protocol,
        "source_manifest_sha256": source["sha256"], "split_hashes_sha256": source["split_hashes_sha256"],
        "data_seed": source["data_seed"], "processor_identity": identity,
        "max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1,
        "generated_code_executed": False, "upstream_quantitative_evaluation": False,
        "sample_count": len(samples), "samples": samples, "samples_sha256": sha256_json(samples),
    }
    payload["sha256"] = sha256_json({k: v for k, v in payload.items() if k != "sha256"})
    return payload


# ---------------------------------------------------------------- 文件/artifact 绑定

def no_symlinks(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise TaskExecutionError(f"artifact / 证据路径不能为 symlink：{path}")
    if path.is_dir() and any(p.is_symlink() for p in path.rglob("*")):
        raise TaskExecutionError(f"artifact 包含 symlink：{path}")
    return Path(os.path.abspath(path))


def read_json(path):
    try:
        value = json.loads(no_symlinks(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("checkpoint JSON 必须为对象")
        return value
    except (OSError, ValueError) as exc:
        raise TaskExecutionError(f"无法读取 {path}：{exc}") from exc


def file_record(path, base):
    resolved = no_symlinks(Path(path)).resolve()
    root = no_symlinks(Path(base)).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise TaskExecutionError("证据文件必须位于输出目录内且为普通文件")
    data = resolved.read_bytes()
    return {"path": resolved.relative_to(root).as_posix(),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def artifact_manifest(directory):
    return mt.artifact_manifest(no_symlinks(directory))


def artifact_identity(artifact):
    if not isinstance(artifact, dict):
        return None
    return {"files": artifact.get("files"), "sha256": artifact.get("sha256")}


def same_artifact(recorded, actual):
    return (isinstance(recorded, dict)
            and artifact_identity(recorded) == artifact_identity(actual)
            and isinstance(recorded.get("path"), str) and bool(recorded["path"]))


def base_artifact(r, model):
    return {"id": model, "revision": r.get("model.revision"), "role": "base"}


def lock_layout(raw, bundle):
    """导出包虚拟锁锚点：锁是位置不是内容身份，不改写 recipe 快照（ADR-015/022）。"""
    _need(isinstance(raw, str) and raw and not raw.startswith(("/", "\\")) and "\\" not in raw,
          "recipe snapshot 缺少本地 lockfile 相对路径")
    parts = raw.split("/")
    depth = 0
    while depth < len(parts) and parts[depth] == "..":
        depth += 1
    tail = parts[depth:]
    _need(tail and all(part not in ("", ".", "..") for part in tail), "lockfile 含非法相对路径片段")
    lock_relative = "/".join(tail)
    reserved = {"recipe-snapshot.json", "export.json", "reload-worker.json", "bundle-manifest.json",
                "frozen-base-reference.json", "recipe.toml", "preference-config.json"}
    _need(lock_relative not in reserved
          and not lock_relative.startswith(("model/", "reference/", "reload/", "_lock-anchor/"))
          and not any(name.startswith(lock_relative + "/") for name in ("model", "reference", "reload")),
          "lockfile 与包内证据路径冲突")
    anchor = bundle.joinpath(*(["_lock-anchor"] * depth)) if depth else bundle
    recipe_path = anchor / "recipe.toml"
    lock_path = bundle / lock_relative
    _need(Path(os.path.abspath(anchor / raw)) == lock_path
          and Path(os.path.abspath(recipe_path.parent / raw)) == lock_path, "锁重定位解析越界")
    return lock_relative, lock_path, recipe_path


def verify_adapter_tensors(model, path, runtime, *, allow_magnitude_vector=True):
    """safetensors 严格复载：keys/shape/值/有限性一致；DoRA 允许 magnitude_vector。"""
    peft, torch = runtime["peft"], runtime["torch"]
    target = getattr(peft, "get_peft_model_state_dict", None)
    if target is None:
        raise TaskExecutionError("PEFT 缺少明确的 adapter state 提取接口")
    try:
        import importlib
        load_file = importlib.import_module("safetensors.torch").load_file
        saved = load_file(str(no_symlinks(path / "adapter_model.safetensors")), device="cpu")
    except (ImportError, OSError, ValueError, RuntimeError, AttributeError) as exc:
        raise TaskExecutionError(f"无法安全读取 adapter tensors：{exc}") from exc
    check_signature(target, {"model": model, "adapter_name": "default",
                             "save_embedding_layers": False}, "get_peft_model_state_dict")
    actual = target(model=model, adapter_name="default", save_embedding_layers=False)
    if not isinstance(actual, dict) or not actual or set(actual) != set(saved):
        raise TaskExecutionError("adapter tensor keys 不完整或含多余键；拒绝非严格加载")
    pattern = r".+\.lora_(?:[AB]|magnitude_vector)\.weight" if allow_magnitude_vector else r".+\.lora_[AB]\.weight"
    for key in actual:
        if not re.fullmatch(pattern, key):
            raise TaskExecutionError(f"adapter 含非 LoRA/DoRA tensor：{key}")
        a, b = actual[key].detach().cpu(), saved[key].detach().cpu()
        if (tuple(a.shape) != tuple(b.shape) or not bool(torch.isfinite(a).all())
                or not bool(torch.isfinite(b).all()) or not torch.equal(a, b.to(dtype=a.dtype))):
            raise TaskExecutionError(f"adapter tensor shape/值/有限性不一致：{key}")
    return {"verified_keys": sorted(actual), "strict_values_verified": True}


def completed(r, plan, identity, kind, config, manifest, runtime, task_id):
    from .baseline_evidence import fingerprint
    if execution_identity(r) != identity:
        raise TaskExecutionError("执行期间 code/lock identity 漂移；禁止 completed")
    seed, data_seed = fixed_seeds(r)
    return {
        **identity, "schema_version": 1, "kind": kind, "stage": "completed", "task_id": task_id,
        "recipe_id": r.id, "fingerprint": fingerprint(r), "mode": plan["mode"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "data_seed": data_seed,
        "model_id": r.get("model.base"), "model_revision": r.get("model.revision"),
        "dataset_id": r.get("dataset.name"), "dataset_revision": r.get("dataset.revision"),
        "source_sha": r.get("source_sha"), "recipe_sha256": sha256_json(r.data),
        "config": config, "config_sha256": sha256_json(config),
        "split_manifest": manifest, "source_manifest": manifest, "runtime": runtime,
        "baseline_promotion": "none", "training_seed_variance_measured": False,
    }
