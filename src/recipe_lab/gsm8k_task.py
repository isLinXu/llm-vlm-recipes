"""gsm8k-reward-v1：GSM8K 四函数渐进 reward 的 data / train / eval 三维；导入仅依赖标准库。

本模块是**本仓自建协议**，不是上游 notebook 的复现：上游一手页面只核实了四个 reward 的
名字与分值上限、官方 GSM8K 数据与单题 expected_answer 格式检查，没有 held-out 评估、
训练 seed 与 GRPOConfig 完整超参。凡 recipe 与本模块共同冻结的值（分档条件、拆分、标签）
都在 `[gsm8k]` / `[upstream_deviations]` 里标明为本仓选择。

刻意与数学路由分开：gold 来自 `####` 之后的规范化数值，推理文本绝不进 prompt；评估用
独立的严格数值 verifier（不依赖 math_verify，也不复用 reward 的 0.5 近似档），并给出
reward hacking 诊断。只实装 data/train/eval 三维；comparison / export 未实装，因此本任务
不注册 task_router、不声明 execution.task_id（resolve() 对「声明了 task_id 却无管线」
会直接拒绝），发布级套件 / 导出显式 unavailable，--full 的 baseline 门禁失败关闭。
"""
from __future__ import annotations

import hashlib
import importlib
import math
import os
import platform
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import math_task as mt
from .math_task import DatasetValidationError, TaskExecutionError, sha256_json

TASK_ID = "gsm8k-reward-v1"
RECIPE_ID = "llm/grpo/qwen2-5-3b-advanced-reward"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
TRL_VERSION = "0.14.0"
GOLD_DELIMITER = "####"
OFFICIAL_TRAIN_ROWS = 7473
OFFICIAL_TEST_ROWS = 1319
VALIDATION_PERCENT = 10
NEAR_MISS_RELATIVE_TOLERANCE = Decimal("0.1")
EVAL_PROTOCOL = "single_checkpoint_evaluation"
SPLIT_NAMES = ("train", "validation", "test")
TAGS = ("<think>", "</think>", "<answer>", "</answer>")
SYSTEM_PROMPT = (
    "Solve the math word problem. Reason step by step inside <think></think>, then give only "
    "the final number inside <answer></answer> with no units or extra text.\n"
    "Format: <think>your reasoning</think><answer>final number</answer>"
)
# 名字、顺序与分值上限来自 recipe 已核实的一手事实（rewards.spec）；分档条件是本仓选择。
REWARD_NAMES = (
    "match_format_exactly", "match_format_approximately",
    "check_answer_correctness", "check_numbers_extraction",
)
REWARD_MAX = {"match_format_exactly": 3.0, "match_format_approximately": 2.0,
              "check_answer_correctness": 3.0, "check_numbers_extraction": 1.5}
CORRECTNESS_TIERS = (3.0, 1.5, 0.5, -0.5)
METRICS = ("pass@1", "format_compliance", "reward_hacking_rate")
TRAIN_REQUIRED = (
    "learning_rate", "lr_scheduler_type", "warmup_ratio", "weight_decay", "max_grad_norm",
    "beta", "temperature", "num_generations", "max_prompt_length", "max_completion_length",
    "per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs", "bf16",
    "gradient_checkpointing", "use_vllm", "remove_unused_columns", "logging_steps",
    "save_strategy", "seed", "data_seed",
)
TRAIN_KEYS = set(TRAIN_REQUIRED) | {"max_steps"}
GSM8K_KEYS = {
    "protocol", "split_rule", "validation_percent", "official_train_rows", "official_test_rows",
    "gold_delimiter", "near_miss_relative_tolerance", "accuracy_definition",
    "reward_hacking_definition",
}
UPSTREAM_FACTS = {
    "reward_names": list(REWARD_NAMES),
    "reward_max_scores": dict(REWARD_MAX),
    "model": MODEL_ID,
    "dataset": f"{DATASET_ID} (config={DATASET_CONFIG})",
    "upstream_has_heldout_metrics": False,
    "upstream_has_training_seed": False,
}

FORMAT_RE = re.compile(r"\s*<think>(.+?)</think>\s*<answer>(.+?)</answer>\s*", re.DOTALL)
GROUPED_RE = re.compile(r"-?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?", re.ASCII)
NUMBER_RE = re.compile(r"-?[0-9]+(?:\.[0-9]+)?", re.ASCII)


# --------------------------------------------------------------------------- #
# 数值与 gold
# --------------------------------------------------------------------------- #
def parse_number(text) -> Decimal | None:
    """严格单个数字：只接受可选 $ 前缀、千分位、末尾句号；带单位/多数字/科学计数法一律 None。"""
    if not isinstance(text, str):
        return None
    value = text.strip()
    if value.startswith("$"):
        value = value[1:].lstrip()
    if value.endswith(".") and not value.endswith(".."):
        value = value[:-1]
    if not value or len(value) > 40:
        return None
    if GROUPED_RE.fullmatch(value):
        value = value.replace(",", "")
    if not NUMBER_RE.fullmatch(value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def canonical_number(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def extract_gold(answer) -> str:
    """GSM8K `answer` 必须恰有一个 `####`，其后是单个数字；失败即数据错误，绝不给满分。"""
    if not isinstance(answer, str) or not answer.strip():
        raise DatasetValidationError("GSM8K answer 必须为非空字符串")
    if answer.count(GOLD_DELIMITER) != 1:
        raise DatasetValidationError(f"GSM8K answer 必须恰含一个 {GOLD_DELIMITER!r}")
    parsed = parse_number(answer.split(GOLD_DELIMITER)[1])
    if parsed is None:
        raise DatasetValidationError("GSM8K #### 之后不是可解析的单个数字")
    return canonical_number(parsed)


# --------------------------------------------------------------------------- #
# completion / 格式
# --------------------------------------------------------------------------- #
def completion_text(completion) -> str | None:
    return mt.completion_text(completion)


def strict_answer(completion) -> str | None:
    """严格全文格式 `<think>..</think><answer>..</answer>`；评估口径只认它。"""
    text = completion_text(completion)
    if not text:
        return None
    match = FORMAT_RE.fullmatch(text)
    if not match:
        return None
    think, answer = match.groups()
    if not think.strip() or not answer.strip():
        return None
    if any(tag in part for part in (think, answer) for tag in TAGS):
        return None
    return answer.strip()


def answer_block(completion) -> str | None:
    """恰有一对 <answer></answer> 即取块内文本（reward 用，比严格格式宽松）。"""
    text = completion_text(completion)
    if not text or text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    start, end = text.index("<answer>"), text.index("</answer>")
    if end < start:
        return None
    block = text[start + len("<answer>"):end].strip()
    return block or None


# --------------------------------------------------------------------------- #
# 四个 reward（TRL 逐列传入 dataset 列：这里需要 `answer`）
# --------------------------------------------------------------------------- #
def match_format_exactly(completions, **kwargs) -> list[float]:
    return [REWARD_MAX["match_format_exactly"] if strict_answer(c) is not None else 0.0
            for c in completions]


def soft_format_score(completion) -> float:
    text = completion_text(completion) or ""
    return sum(0.5 if text.count(tag) == 1 else -0.5 for tag in TAGS)


def match_format_approximately(completions, **kwargs) -> list[float]:
    return [soft_format_score(c) for c in completions]


def correctness_score(completion, gold_text: str) -> float:
    gold = parse_number(gold_text)
    if gold is None:
        raise DatasetValidationError("gold answer 无法解析；拒绝给任何奖励")
    block = answer_block(completion)
    predicted = parse_number(block) if block is not None else None
    if predicted is None:
        return CORRECTNESS_TIERS[3]
    if block == canonical_number(gold) and predicted == gold:
        return CORRECTNESS_TIERS[0]
    if predicted == gold:
        return CORRECTNESS_TIERS[1]
    if gold != 0 and abs(predicted - gold) <= NEAR_MISS_RELATIVE_TOLERANCE * abs(gold):
        return CORRECTNESS_TIERS[2]
    return CORRECTNESS_TIERS[3]


def check_answer_correctness(completions, answer, **kwargs) -> list[float]:
    if not isinstance(answer, (list, tuple)) or len(completions) != len(answer):
        raise DatasetValidationError("completions 与 answer 数量不一致或缺少 answer 列")
    return [correctness_score(c, a) for c, a in zip(completions, answer)]


def check_numbers_extraction(completions, **kwargs) -> list[float]:
    result = []
    for completion in completions:
        block = answer_block(completion)
        result.append(REWARD_MAX["check_numbers_extraction"]
                      if block is not None and parse_number(block) is not None else 0.0)
    return result


REWARD_FUNCTIONS = {f.__name__: f for f in (
    match_format_exactly, match_format_approximately,
    check_answer_correctness, check_numbers_extraction)}


def reward_functions() -> list:
    return [REWARD_FUNCTIONS[name] for name in REWARD_NAMES]


# --------------------------------------------------------------------------- #
# 评估：独立严格 verifier + 诊断
# --------------------------------------------------------------------------- #
def score_completion(completion, gold_text: str) -> dict:
    gold = parse_number(gold_text)
    if gold is None:
        raise DatasetValidationError("gold answer 无法解析")
    strict = strict_answer(completion)
    result = {"format_compliance": int(strict is not None), "accuracy": 0, "answer": strict,
              "parse_status": "invalid_format"}
    if strict is not None:
        predicted = parse_number(strict)
        if predicted is None:
            result["parse_status"] = "unparseable_answer"
        elif predicted == gold:
            result.update(accuracy=1, parse_status="correct")
        else:
            result["parse_status"] = "wrong"
    rewards = {
        "match_format_exactly": match_format_exactly([completion])[0],
        "match_format_approximately": match_format_approximately([completion])[0],
        "check_answer_correctness": correctness_score(completion, gold_text),
        "check_numbers_extraction": check_numbers_extraction([completion])[0],
    }
    format_full = (rewards["match_format_exactly"] == REWARD_MAX["match_format_exactly"]
                   and rewards["match_format_approximately"] == REWARD_MAX["match_format_approximately"]
                   and rewards["check_numbers_extraction"] == REWARD_MAX["check_numbers_extraction"])
    result.update(
        rewards=rewards, total_reward=math.fsum(rewards.values()),
        near_miss=int(rewards["check_answer_correctness"] == CORRECTNESS_TIERS[2]),
        reward_hacking=int(format_full and result["accuracy"] == 0),
        soft_format_only=int(rewards["match_format_approximately"]
                             == REWARD_MAX["match_format_approximately"]
                             and rewards["match_format_exactly"] == 0.0),
    )
    return result


def recompute_metrics(predictions) -> dict:
    """从逐题 completion + gold 重算，不信任预测行里的摘要字段。"""
    if not isinstance(predictions, list) or not predictions:
        raise TaskExecutionError("predictions 必须为非空列表")
    scored = [score_completion(p["completion"], p["gold_answer"]) for p in predictions]
    count = len(scored)
    return {
        "pass@1": sum(s["accuracy"] for s in scored) / count,
        "format_compliance": sum(s["format_compliance"] for s in scored) / count,
        "reward_hacking_rate": sum(s["reward_hacking"] for s in scored) / count,
    }


def diagnostics(predictions) -> dict:
    count = len(predictions)
    scored = [score_completion(p["completion"], p["gold_answer"]) for p in predictions]
    return {
        "mean_total_reward": math.fsum(s["total_reward"] for s in scored) / count,
        "mean_reward": {name: math.fsum(s["rewards"][name] for s in scored) / count
                        for name in REWARD_NAMES},
        "near_miss_rate": sum(s["near_miss"] for s in scored) / count,
        "soft_format_only_rate": sum(s["soft_format_only"] for s in scored) / count,
        "reward_hacking_rate": sum(s["reward_hacking"] for s in scored) / count,
        "parse_status_counts": {status: sum(s["parse_status"] == status for s in scored)
                                for status in ("correct", "wrong", "unparseable_answer", "invalid_format")},
        "definition": "reward_hacking = 三项格式分满额(3.0/2.0/1.5)但独立严格 verifier 判错；"
                      "只是诊断信号，不能单独证明或排除 reward hacking。",
    }


# --------------------------------------------------------------------------- #
# 数据：严格 schema、question hash 90/10、官方 test 冻结
# --------------------------------------------------------------------------- #
def normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).split())


def make_prompt(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


def parse_row(raw, split: str, index: int) -> dict:
    where = f"{split}:{index}"
    if not isinstance(raw, dict) or set(raw) != {"question", "answer"}:
        raise DatasetValidationError(f"{where} 必须恰含 question / answer 两列")
    question, answer = raw["question"], raw["answer"]
    if not isinstance(question, str) or not question.strip():
        raise DatasetValidationError(f"{where} 的 question 为空或类型错误")
    if GOLD_DELIMITER in question:
        raise DatasetValidationError(f"{where} 的 question 含 gold 分隔符，拒绝泄漏")
    try:
        gold = extract_gold(answer)
    except DatasetValidationError as exc:
        raise DatasetValidationError(f"{where} 数据无效：{exc}") from exc
    normalized = normalize_question(question)
    return {"question": question, "normalized": normalized, "gold": gold,
            "problem_sha256": sha256_json(normalized),
            "content_sha256": sha256_json({"question": normalized, "answer": answer}),
            "answer_sha256": sha256_json(answer), "gold_sha256": sha256_json(gold)}


def split_bucket(problem_sha256: str, data_seed: int) -> int:
    digest = hashlib.sha256(f"{data_seed}:{problem_sha256}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def split_spec(r) -> dict:
    return {
        "source": f"{DATASET_ID}/{DATASET_CONFIG}",
        "train_validation": f"官方 train 按 sha256(data_seed:规范化 question) 取模 100，"
                            f"< {VALIDATION_PERCENT} 为 validation，其余为 train；保持原始顺序",
        "test": "官方 test 整体冻结为 held-out；任何题与 train/validation 重复即拒绝",
        "data_seed": r.get("train.data_seed"),
        "official_train_rows": OFFICIAL_TRAIN_ROWS, "official_test_rows": OFFICIAL_TEST_ROWS,
        "duplicate_policy": "train/validation 全局去重（保留首条并登记丢弃索引）；test 内重复或跨 split 重复拒绝",
    }


def prepare_splits(train_rows, test_rows, r) -> tuple[dict, dict]:
    train_rows, test_rows = list(train_rows), list(test_rows)
    expected = (r.get("gsm8k.official_train_rows"), r.get("gsm8k.official_test_rows"))
    if (len(train_rows), len(test_rows)) != expected:
        raise DatasetValidationError(
            f"官方行数不符：train {len(train_rows)}/{expected[0]}，test {len(test_rows)}/{expected[1]}")
    data_seed = r.get("train.data_seed")
    percent = r.get("gsm8k.validation_percent")
    splits = {name: [] for name in SPLIT_NAMES}
    seen_content, seen_problem, dropped = {}, {}, []
    for index, raw in enumerate(train_rows):
        row = parse_row(raw, "train", index)
        if row["content_sha256"] in seen_content or row["problem_sha256"] in seen_problem:
            dropped.append(index)
            continue
        seen_content[row["content_sha256"]] = seen_problem[row["problem_sha256"]] = index
        name = "validation" if split_bucket(row["problem_sha256"], data_seed) < percent else "train"
        splits[name].append({**row, "sample_id": f"train:{index}", "source_split": "train",
                             "source_index": index})
    test_problems = set()
    for index, raw in enumerate(test_rows):
        row = parse_row(raw, "test", index)
        if row["problem_sha256"] in test_problems:
            raise DatasetValidationError(f"test:{index} 在官方 test 内重复")
        if row["problem_sha256"] in seen_problem:
            raise DatasetValidationError(f"test:{index} 与官方 train 题目重叠，拒绝污染 held-out")
        test_problems.add(row["problem_sha256"])
        splits["test"].append({**row, "sample_id": f"test:{index}", "source_split": "test",
                               "source_index": index})
    if any(not splits[name] for name in SPLIT_NAMES):
        raise DatasetValidationError("train / validation / test 均不得为空")
    entries = {name: [{k: row[k] for k in ("sample_id", "source_split", "source_index",
                                              "content_sha256", "problem_sha256",
                                              "answer_sha256", "gold_sha256")}
                      for row in rows] for name, rows in splits.items()}
    manifest = {
        "task_id": TASK_ID, "dataset_id": DATASET_ID, "dataset_config": DATASET_CONFIG,
        "split_spec": split_spec(r), "dropped_duplicate_train_indices": dropped,
        "splits": {name: {"count": len(items), "samples": items, "sha256": sha256_json(items)}
                   for name, items in entries.items()},
    }
    manifest["sha256"] = sha256_json(manifest)
    runtime_splits = {name: [{**row, "prompt": make_prompt(row["question"])} for row in rows]
                      for name, rows in splits.items()}
    return runtime_splits, manifest


def validate_source_manifest(manifest, r) -> dict:
    from . import baseline_evidence as be
    if not be.valid_manifest(manifest):
        raise TaskExecutionError("manifest hash / 去重 / 内容身份无效")
    if (manifest.get("task_id") != TASK_ID or manifest.get("dataset_id") != DATASET_ID
            or manifest.get("dataset_config") != DATASET_CONFIG
            or manifest.get("dataset_revision") != r.get("dataset.revision")):
        raise TaskExecutionError("manifest 任务 / 数据集 / revision 不一致")
    if manifest.get("split_spec") != split_spec(r):
        raise TaskExecutionError("manifest 拆分协议与当前 recipe 不一致")
    for name, split in manifest["splits"].items():
        for sample in split["samples"]:
            if not sample["sample_id"].startswith(f"{sample.get('source_split')}:"):
                raise TaskExecutionError("样本 ID 与来源 split 不一致")
    return manifest


# --------------------------------------------------------------------------- #
# 路由与配置校验
# --------------------------------------------------------------------------- #
def resolve_task_id(r, plan=None, task_id=None):
    declared = task_id or (plan or {}).get("task_id") or r.get("execution.task_id")
    return declared or (TASK_ID if r.id == RECIPE_ID else None)


def _finite(value, *, positive=False, allow_zero=True) -> bool:
    ok = type(value) in (int, float) and math.isfinite(value)
    if ok and positive:
        return value > 0
    return ok and (allow_zero or value != 0)


def require_supported(r, plan=None, task_id=None) -> str:
    resolved = resolve_task_id(r, plan, task_id)
    if r.id != RECIPE_ID or resolved != TASK_ID:
        raise TaskExecutionError(
            f"尚未实现该执行路由：recipe={r.id}, task_id={resolved!r}；"
            f"仅支持 {RECIPE_ID} / {TASK_ID}，其他 recipe 请使用 --dry-run。")
    required = {
        "method.family": "grpo", "method.trainer": "GRPOTrainer",
        "model.base": MODEL_ID, "dataset.name": DATASET_ID, "dataset.config": DATASET_CONFIG,
        "dependencies.trl": TRL_VERSION, "rewards.functions": list(REWARD_NAMES),
        "evaluation.metrics": list(METRICS),
    }
    for key, value in required.items():
        if r.get(key) != value:
            raise TaskExecutionError(f"{TASK_ID} 要求 {key}={value!r}，不能替换为 {r.get(key)!r}")
    if r.is_vlm or r.get("peft") or r.get("quantization"):
        raise TaskExecutionError("该路由仅实现纯文本全参数 GRPO，不支持 VLM / PEFT / 量化")
    specs = {item.get("name"): item.get("max_score") for item in (r.get("rewards.spec") or [])}
    if specs != REWARD_MAX:
        raise TaskExecutionError(f"rewards.spec 分值上限与一手核实值不一致：{specs}")
    section = r.get("gsm8k")
    if not isinstance(section, dict) or set(section) != GSM8K_KEYS:
        raise TaskExecutionError(f"[gsm8k] 段必须恰含 {sorted(GSM8K_KEYS)}")
    expected_section = {
        "protocol": TASK_ID, "validation_percent": VALIDATION_PERCENT,
        "official_train_rows": OFFICIAL_TRAIN_ROWS, "official_test_rows": OFFICIAL_TEST_ROWS,
        "gold_delimiter": GOLD_DELIMITER,
        "near_miss_relative_tolerance": float(NEAR_MISS_RELATIVE_TOLERANCE),
    }
    for key, value in expected_section.items():
        if section.get(key) != value or type(section.get(key)) is not type(value):
            raise TaskExecutionError(f"[gsm8k].{key} 必须为 {value!r}，实际 {section.get(key)!r}")
    for key in ("split_rule", "accuracy_definition", "reward_hacking_definition"):
        if not isinstance(section.get(key), str) or not section[key].strip():
            raise TaskExecutionError(f"[gsm8k].{key} 必须为非空说明")
    train = r.get("train", {}) or {}
    unknown = sorted(set(train) - TRAIN_KEYS)
    if unknown:
        raise TaskExecutionError(f"[train] 含未支持参数 {unknown}；拒绝静默忽略")
    missing = [key for key in TRAIN_REQUIRED if key not in train]
    if missing:
        raise TaskExecutionError(f"[train] 必须显式冻结 {missing}；不能依赖框架默认值")
    if train["remove_unused_columns"] is not False:
        raise TaskExecutionError("train.remove_unused_columns 必须为 false，保留 reward 所需 answer 列")
    if train["use_vllm"] is not False:
        raise TaskExecutionError("本任务不使用 vLLM；use_vllm 必须显式为 false（vLLM 由另一篇配方单独对照）")
    if train["save_strategy"] != "no":
        raise TaskExecutionError("train.save_strategy 必须为 'no'，只在预算完成后显式保存")
    if not all(type(train[k]) is bool for k in ("bf16", "gradient_checkpointing")):
        raise TaskExecutionError("train.bf16 / gradient_checkpointing 必须为显式布尔值")
    if not isinstance(train["lr_scheduler_type"], str) or not train["lr_scheduler_type"]:
        raise TaskExecutionError("train.lr_scheduler_type 必须为非空字符串")
    for key in ("learning_rate", "temperature", "max_grad_norm"):
        if not _finite(train[key], positive=True):
            raise TaskExecutionError(f"train.{key} 必须为有限正数")
    for key in ("beta", "weight_decay", "warmup_ratio"):
        if not _finite(train[key]) or train[key] < 0:
            raise TaskExecutionError(f"train.{key} 必须为有限非负数")
    return resolved


def mode_output_dir(plan, mode: str) -> Path:
    return mt.mode_output_dir(plan, mode)


def task_plan(r, plan: dict, mode: str, task_id=None) -> dict:
    result = dict(plan)
    result["task_id"] = resolve_task_id(r, plan, task_id)
    try:
        require_supported(r, plan, task_id)
    except TaskExecutionError as exc:
        result["execution_supported"] = False
        result["unsupported_reason"] = str(exc)
        return result
    result["execution_supported"] = True
    result["dataset"] = {**plan["dataset"], "split": "官方 train 的 hash 90/10 + 官方 test 整体 held-out",
                         "split_plan": split_spec(r)}
    result["run"] = {**plan["run"], "output_dir": str(mode_output_dir(plan, mode))}
    if mode in ("dry-run", "smoke", "full"):
        budget_mode = "smoke" if mode == "dry-run" else mode
        steps = r.get("train.max_steps", r.get(f"budget.{budget_mode}_steps"))
        result["run"]["steps"] = steps
        result["run"]["params"] = {**plan["run"]["params"], "max_steps": steps}
    result["warnings"] = [w for w in plan.get("warnings", []) if "evaluation.expected" not in w]
    result["prompt"] = {"system": SYSTEM_PROMPT, "gold_in_prompt": False}
    result["rewards"] = {**(plan.get("rewards") or {}), "functions": list(REWARD_NAMES),
                         "max_scores": dict(REWARD_MAX),
                         "correctness_tiers": list(CORRECTNESS_TIERS)}
    result["evaluation"] = {
        "metrics": list(METRICS), "held_out": True, "seed": r.get("evaluation.seed", r.get("train.seed")),
        "runs": 1, "do_sample": False, "protocol": EVAL_PROTOCOL,
        "verifier": "strict-decimal（独立于 reward 的 0.5 近似档，不依赖 math_verify）",
        "training_seed_variance_measured": False, "baseline_promotion": "none",
        "release_pipeline": "unavailable（comparison/export 未实装，不注册 task_router）",
    }
    return result


def config_kwargs(r, plan: dict, output_dir: Path) -> dict:
    require_supported(r, plan)
    seed, data_seed = mt.fixed_seeds(r)
    params = dict(r.get("train", {}))
    mode = plan["mode"]
    budget = r.get(f"budget.{mode}_steps")
    max_steps = params.get("max_steps", budget)
    if type(budget) is not int or budget <= 0 or type(max_steps) is not int or not 0 < max_steps <= budget:
        raise TaskExecutionError(f"{mode} 预算必须覆盖 max_steps：budget={budget}, max_steps={max_steps}")
    batch, generations = params["per_device_train_batch_size"], params["num_generations"]
    if (type(generations) is not int or generations < 2 or type(batch) is not int
            or batch < generations or batch % generations):
        raise TaskExecutionError("单 GPU GRPO batch 必须是 num_generations（至少 2）的正整数倍")
    for key in ("gradient_accumulation_steps", "max_completion_length", "max_prompt_length",
                "num_train_epochs", "logging_steps"):
        if type(params[key]) is not int or params[key] <= 0:
            raise TaskExecutionError(f"train.{key} 必须为正整数")
    params.update(seed=seed, data_seed=data_seed, max_steps=max_steps,
                  output_dir=str(output_dir), report_to="none")
    return params


# --------------------------------------------------------------------------- #
# 运行时（延迟导入框架）
# --------------------------------------------------------------------------- #
def load_runtime(*, training=False) -> dict:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise TaskExecutionError("缺少 torch CUDA build；请按 lock 安装依赖，或使用 --dry-run") from exc
    if not torch.cuda.is_available():
        raise TaskExecutionError("没有可用 CUDA GPU；拒绝执行或下载模型，请使用 --dry-run")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise TaskExecutionError("受控样板仅支持单进程单 GPU，不支持分布式运行")
    dependencies = {"torch": torch}
    for name in ["transformers", "datasets"] + (["trl"] if training else []):
        try:
            dependencies[name] = importlib.import_module(name)
        except ImportError as exc:
            raise TaskExecutionError(f"缺少 {name} 或其依赖：{exc}；请按 recipe lock 安装环境") from exc
    if training and getattr(dependencies["trl"], "__version__", None) != TRL_VERSION:
        raise TaskExecutionError(
            f"该路由要求 trl=={TRL_VERSION}，实际为 {getattr(dependencies['trl'], '__version__', None)}")
    return dependencies


def runtime_evidence(dependencies, started: float) -> dict:
    torch = dependencies["torch"]
    versions = {name: getattr(module, "__version__", None) for name, module in dependencies.items()}
    return {
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "world_size": 1,
        "duration_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def load_splits(r, dependencies):
    revision = mt.require_revision(r, "dataset.revision")
    load = dependencies["datasets"].load_dataset
    train = load(DATASET_ID, DATASET_CONFIG, revision=revision, split="train")
    test = load(DATASET_ID, DATASET_CONFIG, revision=revision, split="test")
    splits, manifest = prepare_splits(train, test, r)
    manifest["dataset_revision"] = revision
    manifest.pop("sha256")
    manifest["sha256"] = sha256_json(manifest)
    return splits, manifest


def prompt_token_audit(tokenizer, splits, limit: int) -> dict:
    """提示被 TRL 左截断会静默破坏题目；任一 prompt 超限即拒绝训练/评估。"""
    longest = 0
    for rows in splits.values():
        for row in rows:
            tokens = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True)
            longest = max(longest, len(tokens))
    if longest > limit:
        raise TaskExecutionError(
            f"最长 prompt {longest} token 超过 max_prompt_length={limit}；拒绝静默截断")
    return {"max_prompt_tokens": longest, "limit": limit}


def completed_evidence(r, plan, kind, config, manifest, runtime) -> dict:
    seed, data_seed = mt.fixed_seeds(r)
    from .baseline_evidence import fingerprint
    identity = mt.execution_identity(r)
    if plan.get("execution_identity", identity) != identity:
        raise TaskExecutionError("执行期间代码或环境锁发生漂移；不写completed证据")
    return {
        **identity, "fingerprint": fingerprint(r), "schema_version": 1, "kind": kind,
        "stage": "completed", "task_id": TASK_ID, "recipe_id": r.id,
        "experiment_id": plan.get("experiment_id", ""), "mode": plan["mode"],
        "completed_at": datetime.now(timezone.utc).isoformat(), "seed": seed, "data_seed": data_seed,
        "model_id": MODEL_ID, "model_revision": mt.require_revision(r, "model.revision"),
        "dataset_id": DATASET_ID, "dataset_revision": mt.require_revision(r, "dataset.revision"),
        "recipe_sha256": sha256_json(r.data), "config": config, "config_sha256": sha256_json(config),
        "split_manifest": manifest, "runtime": runtime,
        "baseline_promotion": "none", "training_seed_variance_measured": False,
    }


def checkpoint_path(plan, checkpoint=None, *, recipe=None, require_evidence=True) -> Path:
    path = Path(checkpoint).expanduser().resolve() if checkpoint else mode_output_dir(plan, "full").resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise TaskExecutionError(f"没有训练 checkpoint：{path}；请显式 --checkpoint 本地模型目录，或先完成 --full 训练")
    if not (path / "tokenizer_config.json").is_file() or not any(
            p.is_file() for pattern in ("*.safetensors", "pytorch_model*.bin") for p in path.glob(pattern)):
        raise TaskExecutionError(f"checkpoint 缺少完整模型权重或 tokenizer：{path}；不会回退到 base")
    if require_evidence:
        import json
        try:
            evidence = json.loads((path / "training.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TaskExecutionError("checkpoint 缺少已完成 training.json；显式路径也不能绕过训练来源校验") from exc
        if not (isinstance(evidence, dict) and evidence.get("kind") == "training"
                and evidence.get("stage") == "completed" and evidence.get("task_id") == TASK_ID
                and evidence.get("recipe_id") == RECIPE_ID and evidence.get("mode") in ("smoke", "full")
                and (checkpoint is not None or evidence.get("mode") == "full")):
            raise TaskExecutionError("checkpoint 没有对应的已完成训练证据")
        if evidence.get("model_artifact", {}).get("sha256") != mt.artifact_manifest(path)["sha256"]:
            raise TaskExecutionError("checkpoint 内容与训练证据不一致")
        if recipe is not None:
            from .baseline_evidence import fingerprint
            if (evidence.get("fingerprint") != fingerprint(recipe)
                    or evidence.get("model_revision") != recipe.get("model.revision")
                    or evidence.get("dataset_revision") != recipe.get("dataset.revision")):
                raise TaskExecutionError("checkpoint 训练配置或 revision 与当前 recipe 不一致")
    return path


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def run_training(r, plan: dict) -> Path:
    """全参数 GRPO；所有路由 / 预算 / 配置校验先于框架导入与下载。返回 training.json 路径。"""
    require_supported(r, plan)
    if r.get("decision.gate") == "blocked" or r.get("baseline.result") == "passes":
        raise TaskExecutionError("训练决策门已阻断：先复核零训练基线及业务目标，不允许绕过预算决策")
    revision = mt.require_revision(r, "model.revision")
    mt.require_revision(r, "dataset.revision")
    output = Path(plan["run"]["output_dir"])
    kwargs = config_kwargs(r, plan, output)
    if output.exists() and any(output.iterdir()):
        raise TaskExecutionError(f"训练目录非空：{output}；拒绝覆盖旧 checkpoint 或证据，请使用独立实验目录")
    dependencies = load_runtime(training=True)
    mt.require_frozen_runtime(r, dependencies)
    plan = {**plan, "execution_identity": mt.execution_identity(r)}
    started = time.perf_counter()
    mt.apply_determinism(r, dependencies)
    trl = dependencies["trl"]
    try:
        config_cls, trainer_cls = trl.GRPOConfig, trl.GRPOTrainer
    except (AttributeError, ImportError) as exc:
        raise TaskExecutionError(f"trl=={TRL_VERSION} 的 GRPOConfig / GRPOTrainer 不可用，请按 lock 重装依赖") from exc
    mt.check_signature(config_cls, kwargs, "GRPOConfig")
    trainer_kwargs = {"model": None, "processing_class": None, "args": None, "train_dataset": None,
                      "eval_dataset": None, "reward_funcs": reward_functions()}
    mt.check_signature(trainer_cls, trainer_kwargs, "GRPOTrainer")
    config = config_cls(**kwargs)
    for key, value in kwargs.items():
        actual = getattr(config, key, None)
        if key != "report_to" and actual != value:
            raise TaskExecutionError(f"GRPOConfig 未应用 {key}：请求 {value!r}，实际 {actual!r}")
    splits, manifest = load_splits(r, dependencies)
    transformers = dependencies["transformers"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.eos_token is None:
        raise TaskExecutionError("tokenizer 缺少 eos_token，不能组装 GRPO")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    audit = prompt_token_audit(tokenizer, splits, kwargs["max_prompt_length"])
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, torch_dtype=dependencies["torch"].float32)
    dataset_cls = dependencies["datasets"].Dataset
    datasets = {name: dataset_cls.from_list([{"prompt": row["prompt"], "answer": row["gold"]}
                                             for row in splits[name]])
                for name in ("train", "validation")}
    trainer_kwargs.update(model=model, processing_class=tokenizer, args=config,
                          train_dataset=datasets["train"], eval_dataset=datasets["validation"])
    trainer = trainer_cls(**trainer_kwargs)
    train_result = trainer.train()
    actual_steps = trainer.state.global_step
    if actual_steps != kwargs["max_steps"]:
        raise TaskExecutionError(
            f"训练未完成预算：实际 {actual_steps} 步，要求 {kwargs['max_steps']}；不写 completed evidence")
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    checkpoint_path(plan, checkpoint=output, require_evidence=False)
    dependencies["torch"].cuda.synchronize()
    manifest_path = mt.save_json(output / "manifest.json", manifest)
    evidence = completed_evidence(r, plan, "training", config.to_dict(), manifest,
                                  runtime_evidence(dependencies, started))
    evidence.update(
        steps=actual_steps, max_steps=kwargs["max_steps"], trainer="GRPOTrainer",
        rewards=list(REWARD_NAMES), reward_max_scores=dict(REWARD_MAX),
        reward_weights="四函数等权相加（TRL 默认求和）；无量纲归一化，权重未经消融",
        prompt_audit=audit, gsm8k=dict(r.get("gsm8k")), upstream_facts=UPSTREAM_FACTS,
        model_artifact=mt.artifact_manifest(output), split_manifest_path=str(manifest_path),
        train_metrics=dict(train_result.metrics),
    )
    return mt.save_json(output / "training.json", evidence)


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def run_evaluation(r, plan, checkpoint=None, zero_shot=False) -> Path:
    require_supported(r, plan)
    revision = mt.require_revision(r, "model.revision")
    mt.require_revision(r, "dataset.revision")
    mt.fixed_seeds(r)
    if plan.get("requested_runs", 1) != 1:
        raise TaskExecutionError("仅支持 --runs 1；重复 greedy decode 不是训练多 seed 方差，拒绝伪造统计")
    if zero_shot and checkpoint is not None:
        raise TaskExecutionError("zero-shot 只能加载 base，不接受 checkpoint")
    source = MODEL_ID if zero_shot else checkpoint_path(plan, checkpoint, recipe=r)
    output = mode_output_dir(plan, "zero-shot" if zero_shot else "eval").resolve()
    if not zero_shot and output == source:
        raise TaskExecutionError("评估输出目录不能覆盖 checkpoint")
    if (output / "evaluation.json").exists():
        raise TaskExecutionError(f"已存在评估证据 {output}，请使用独立实验输出目录，避免覆盖")
    dependencies = load_runtime()
    mt.require_frozen_runtime(r, dependencies)
    plan = {**plan, "execution_identity": mt.execution_identity(r)}
    started = time.perf_counter()
    mt.apply_determinism(r, dependencies, evaluation=True)
    splits, manifest = load_splits(r, dependencies)
    transformers, torch = dependencies["transformers"], dependencies["torch"]
    loader_kwargs = {"revision": revision} if zero_shot else {"local_files_only": True}
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(source), **loader_kwargs)
    if tokenizer.eos_token_id is None:
        raise TaskExecutionError("tokenizer 缺少 eos_token_id")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.float32, **loader_kwargs).to("cuda")
    model.eval()
    max_tokens = r.get("train.max_completion_length")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise TaskExecutionError("train.max_completion_length 必须为正整数")
    audit = prompt_token_audit(tokenizer, {"test": splits["test"]}, r.get("train.max_prompt_length"))
    generation_config = {
        "seed": r.get("evaluation.seed", mt.fixed_seeds(r)[0]), "do_sample": False, "num_beams": 1,
        "max_new_tokens": max_tokens, "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id, "use_cache": True,
    }
    predictions = []
    with torch.inference_mode():
        for row in splits["test"]:
            encoded = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True).to("cuda")
            input_length = encoded["input_ids"].shape[-1]
            generated = model.generate(
                **encoded, **{k: v for k, v in generation_config.items() if k != "seed"})
            completion = tokenizer.decode(generated[0][input_length:], skip_special_tokens=True)
            scored = score_completion(completion, row["gold"])
            predictions.append({
                "sample_id": row["sample_id"], "content_sha256": row["content_sha256"],
                "prompt": row["prompt"], "gold_answer": row["gold"], "completion": completion,
                "input_tokens": input_length, "generated_tokens": len(generated[0]) - input_length,
                **scored,
            })
    torch.cuda.synchronize()
    count = len(predictions)
    metrics = recompute_metrics(predictions)
    model_artifact = ({"id": MODEL_ID, "revision": revision, "role": "base"} if zero_shot
                      else {**mt.artifact_manifest(source), "role": "trained_checkpoint"})
    config = {"task_id": TASK_ID, "system_prompt": SYSTEM_PROMPT, "generation": generation_config,
              "model": model_artifact, "dataset_revision": manifest["dataset_revision"],
              "test_sha256": manifest["splits"]["test"]["sha256"], "prompt_audit": audit,
              "verifier": "strict-decimal"}
    evidence = completed_evidence(r, plan, "evaluation", config, manifest,
                                  runtime_evidence(dependencies, started))
    evidence.update(
        evaluation_protocol=EVAL_PROTOCOL, model_role=model_artifact["role"],
        model_artifact=model_artifact, sample_count=count, metrics=metrics,
        metric_counts={"correct": sum(p["accuracy"] for p in predictions),
                       "format_compliant": sum(p["format_compliance"] for p in predictions),
                       "reward_hacking": sum(p["reward_hacking"] for p in predictions), "total": count},
        diagnostics=diagnostics(predictions),
        predictions={"path": "predictions.json", "sha256": sha256_json(predictions), "count": count},
    )
    prediction_path = mt.save_json(output / "predictions.json", predictions)
    evidence["predictions_file"] = {"path": "predictions.json",
                                    "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()}
    mt.save_json(output / "manifest.json", manifest)
    return mt.save_json(output / "evaluation.json", evidence)
