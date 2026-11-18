"""gsm8k-reward-v1 的离线测试：只用 tmp_path / 内存里的确定性合成 GSM8K 行与 fake runtime。

不联网、不导入 torch/trl/transformers/datasets；通过不证明训练收敛，也不产生 GPU 证据。
"""
from __future__ import annotations

import contextlib
import copy
import importlib.abc
import importlib.util
import inspect
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import gsm8k_task as gt
from recipe_lab import math_task as mt
from recipe_lab import task_router
from recipe_lab.loader import load_recipes
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan

ROOT = Path(__file__).resolve().parents[1]
GOOD = "<think>推理</think><answer>5</answer>"


def real_recipe():
    return next(r for r in load_recipes(ROOT) if r.id == gt.RECIPE_ID)


def mutated(recipe, path, value, *, delete=False):
    data = copy.deepcopy(recipe.data)
    node = data
    *parents, last = path.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    if delete:
        node.pop(last, None)
    else:
        node[last] = value
    return Recipe(recipe.id, recipe.path, data)


@pytest.fixture
def recipe(tmp_path):
    """真实 recipe + 仅用于隔离测试的假 revision / 假锁；不是 frozen 环境。"""
    data = copy.deepcopy(real_recipe().data)
    data["model"]["revision"] = "a" * 40
    data["dataset"]["revision"] = "b" * 40
    data["execution"]["lockfile"] = "fixture.lock.toml"
    (tmp_path / "fixture.lock.toml").write_text('status = "test-only"\n', encoding="utf-8")
    return Recipe(gt.RECIPE_ID, tmp_path / "recipe.toml", data)


def gsm_rows(count, prefix):
    return [{"question": f"{prefix} 问题 {i}：小明有 {i} 个苹果，又买了 3 个，共几个？",
             "answer": f"小明开始有 {i} 个。<<{i}+3={i + 3}>>\n共 {i + 3} 个。\n#### {i + 3}"}
            for i in range(count)]


@pytest.fixture(scope="module")
def official():
    return gsm_rows(gt.OFFICIAL_TRAIN_ROWS, "训练"), gsm_rows(gt.OFFICIAL_TEST_ROWS, "测试")


# --------------------------------------------------------------------------- #
# 数值与 gold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("72", "72"), (" 72 ", "72"), ("-5", "-5"), ("72.0", "72"), ("0.50", "0.5"),
    ("1,000", "1000"), ("$18", "18"), ("$ 18", "18"), ("18.", "18"), ("0", "0"), ("-0", "0"),
])
def test_parse_number_accepts_only_a_single_plain_number(text, expected):
    assert gt.canonical_number(gt.parse_number(text)) == expected


@pytest.mark.parametrize("text", [
    "", "  ", "72 clips", "about 72", "7 2", "1e3", "1,00", "12,34", "1.2.3", "٣", "７",
    "7$2", "--5", "5-", "72%", "0x10", "9" * 41, None, 72, ["72"], "nan", "inf",
])
def test_parse_number_rejects_everything_else(text):
    assert gt.parse_number(text) is None


@pytest.mark.parametrize("answer,expected", [
    ("推理 <<2*3=6>> 所以\n#### 6", "6"), ("x\n#### 1,000", "1000"),
    ("x\n#### -5", "-5"), ("x\n#### 72.0", "72"), ("x #### 18 ", "18"),
])
def test_extract_gold(answer, expected):
    assert gt.extract_gold(answer) == expected


@pytest.mark.parametrize("answer", [
    "没有分隔符 6", "#### 1\n#### 2", "x\n#### ", "x\n#### six", "x\n#### 6 apples",
    "", "   ", None, 6, {"answer": "#### 6"},
])
def test_gold_parse_failure_is_a_dataset_error_never_a_score(answer):
    with pytest.raises(mt.DatasetValidationError):
        gt.extract_gold(answer)


# --------------------------------------------------------------------------- #
# 四个 reward
# --------------------------------------------------------------------------- #
def chat(text):
    return [{"role": "assistant", "content": text}]


def test_reward_names_order_and_spec_match_registered_facts():
    recipe = real_recipe()
    assert list(gt.REWARD_NAMES) == recipe.get("rewards.functions")
    specs = {s["name"]: s["max_score"] for s in recipe.get("rewards.spec")}
    assert specs == gt.REWARD_MAX
    assert [f.__name__ for f in gt.reward_functions()] == list(gt.REWARD_NAMES)


@pytest.mark.parametrize("text,expected", [
    (GOOD, 3.0), ("  " + GOOD + "\n", 3.0), ("<think>a</think>\n<answer>5</answer>", 3.0),
    ("", 0.0), ("<think></think><answer>5</answer>", 0.0), ("<think>a</think><answer> </answer>", 0.0),
    ("前缀 " + GOOD, 0.0), (GOOD + " 后缀", 0.0), ("<answer>5</answer><think>a</think>", 0.0),
    ("<think>a<think>b</think><answer>5</answer>", 0.0),
    ("<think>a</think><answer>5<answer>6</answer>", 0.0),
])
def test_match_format_exactly(text, expected):
    assert gt.match_format_exactly([text, chat(text)]) == [expected, expected]


def test_match_format_exactly_rejects_non_assistant_message_lists():
    bad = [{"role": "user", "content": GOOD}]
    assert gt.match_format_exactly([bad, None, 5, []]) == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("text,expected", [
    (GOOD, 2.0), ("", -2.0), ("<think>a</think>", 0.0), ("<think>a</think><answer>5", 1.0),
    ("<think>a</think><think>b</think><answer>5</answer>", 0.0),
    ("<think><answer></think></answer>", 2.0),
])
def test_match_format_approximately_counts_each_token(text, expected):
    assert gt.match_format_approximately([text]) == [expected]
    assert -2.0 <= gt.match_format_approximately([text])[0] <= gt.REWARD_MAX["match_format_approximately"]


@pytest.mark.parametrize("text,gold,expected", [
    ("<think>a</think><answer>5</answer>", "5", 3.0),
    ("<think>a</think><answer>5.0</answer>", "5", 1.5),
    ("<think>a</think><answer>$5</answer>", "5", 1.5),
    ("<think>a</think><answer>1,000</answer>", "1000", 1.5),
    ("<think>a</think><answer>105</answer>", "100", 0.5),
    ("<think>a</think><answer>110</answer>", "100", 0.5),
    ("<think>a</think><answer>111</answer>", "100", -0.5),
    ("<think>a</think><answer>6</answer>", "5", -0.5),
    ("<think>a</think><answer>0</answer>", "0", 3.0),
    ("<think>a</think><answer>1</answer>", "0", -0.5),
    ("<think>a</think><answer>-5</answer>", "-5", 3.0),
    ("<think>a</think><answer>-5.2</answer>", "-5", 0.5),
    ("<think>a</think><answer>5 apples</answer>", "5", -0.5),
    ("<answer>5</answer>", "5", 3.0),
    ("答案 5", "5", -0.5), ("", "5", -0.5),
    ("<answer>5</answer><answer>5</answer>", "5", -0.5),
    ("</answer>5<answer>", "5", -0.5),
])
def test_check_answer_correctness_tiers(text, gold, expected):
    assert gt.check_answer_correctness([text, chat(text)], [gold, gold]) == [expected, expected]


def test_correctness_only_ever_returns_the_four_registered_tiers():
    seen = {gt.correctness_score(f"<answer>{value}</answer>", "100")
            for value in ("100", "100.0", "101", "95", "1", "x", "")}
    assert seen == set(gt.CORRECTNESS_TIERS) and max(seen) == gt.REWARD_MAX["check_answer_correctness"]


@pytest.mark.parametrize("bad_gold", ["", "x", "5 apples", "#### 5", None])
def test_unparseable_reward_gold_is_fatal_not_full_score(bad_gold):
    with pytest.raises(mt.DatasetValidationError):
        gt.check_answer_correctness([GOOD], [bad_gold])


def test_reward_length_or_missing_answer_column_is_an_error():
    with pytest.raises(mt.DatasetValidationError):
        gt.check_answer_correctness([GOOD, GOOD], ["5"])
    with pytest.raises(mt.DatasetValidationError):
        gt.check_answer_correctness([GOOD], "5")
    with pytest.raises(TypeError):
        gt.check_answer_correctness([GOOD])


@pytest.mark.parametrize("text,expected", [
    ("<think>a</think><answer>72</answer>", 1.5), ("<answer>1,000</answer>", 1.5),
    ("<answer>72 clips</answer>", 0.0), ("<answer>1 2</answer>", 0.0), ("72", 0.0), ("", 0.0),
    ("<answer></answer>", 0.0), ("<answer>5</answer><answer>6</answer>", 0.0),
    ("the answer is 72 <think>72</think>", 0.0),
])
def test_check_numbers_extraction_only_reads_the_answer_block(text, expected):
    assert gt.check_numbers_extraction([text]) == [expected]


# --------------------------------------------------------------------------- #
# 评估：独立严格 verifier 与诊断
# --------------------------------------------------------------------------- #
def test_eval_accuracy_needs_strict_format_even_when_reward_is_lenient():
    lenient = "前缀 <answer>5</answer>"
    assert gt.correctness_score(lenient, "5") == 3.0
    scored = gt.score_completion(lenient, "5")
    assert scored["accuracy"] == 0 and scored["format_compliance"] == 0
    assert scored["parse_status"] == "invalid_format" and scored["reward_hacking"] == 0


def test_score_completion_correct_wrong_and_unparseable():
    correct = gt.score_completion(GOOD, "5")
    assert correct["accuracy"] == 1 and correct["parse_status"] == "correct"
    assert correct["total_reward"] == 3.0 + 2.0 + 3.0 + 1.5 and correct["reward_hacking"] == 0
    wrong = gt.score_completion("<think>x</think><answer>999</answer>", "5")
    assert wrong["accuracy"] == 0 and wrong["parse_status"] == "wrong"
    assert wrong["reward_hacking"] == 1 and wrong["rewards"]["check_answer_correctness"] == -0.5
    text = gt.score_completion("<think>x</think><answer>五</answer>", "5")
    assert text["accuracy"] == 0 and text["parse_status"] == "unparseable_answer"
    assert text["reward_hacking"] == 0 and text["rewards"]["check_numbers_extraction"] == 0.0


def test_near_miss_and_soft_format_only_diagnostics():
    near = gt.score_completion("<think>x</think><answer>105</answer>", "100")
    assert near["near_miss"] == 1 and near["accuracy"] == 0 and near["reward_hacking"] == 1
    soft = gt.score_completion("<think>a<answer>b</think></answer>", "5")
    assert soft["soft_format_only"] == 1 and soft["format_compliance"] == 0
    assert soft["rewards"]["match_format_exactly"] == 0.0
    assert gt.score_completion(GOOD, "5")["soft_format_only"] == 0


def test_recompute_metrics_ignores_prediction_summary_fields():
    predictions = [
        {"completion": GOOD, "gold_answer": "5", "accuracy": 0, "format_compliance": 0},
        {"completion": "<think>x</think><answer>9</answer>", "gold_answer": "5", "accuracy": 1},
        {"completion": "乱写", "gold_answer": "5", "reward_hacking": 1},
        {"completion": GOOD, "gold_answer": "5"},
    ]
    assert gt.recompute_metrics(predictions) == {
        "pass@1": 0.5, "format_compliance": 0.75, "reward_hacking_rate": 0.25}
    report = gt.diagnostics(predictions)
    assert report["parse_status_counts"] == {"correct": 2, "wrong": 1, "unparseable_answer": 0,
                                             "invalid_format": 1}
    assert set(report["mean_reward"]) == set(gt.REWARD_NAMES)
    with pytest.raises(mt.TaskExecutionError):
        gt.recompute_metrics([])


# --------------------------------------------------------------------------- #
# 数据拆分
# --------------------------------------------------------------------------- #
def test_splits_are_stable_disjoint_and_gold_never_enters_prompts(recipe, official):
    train, test = official
    splits, manifest = gt.prepare_splits(train, test, recipe)
    again, second = gt.prepare_splits(iter(train), iter(test), recipe)
    assert splits == again and manifest == second
    assert {name: len(rows) for name, rows in splits.items()}["test"] == gt.OFFICIAL_TEST_ROWS
    assert len(splits["train"]) + len(splits["validation"]) == gt.OFFICIAL_TRAIN_ROWS
    share = len(splits["validation"]) / gt.OFFICIAL_TRAIN_ROWS
    assert 0.07 < share < 0.13, share
    owners = {}
    for name, rows in splits.items():
        for row in rows:
            for key in ("sample_id", "problem_sha256", "content_sha256"):
                assert owners.setdefault((key, row[key]), name) == name
            assert row["prompt"] == gt.make_prompt(row["question"])
            assert "####" not in json.dumps(row["prompt"], ensure_ascii=False)
            assert row["gold"] == str(int(row["gold"]))
    reasoning = train[0]["answer"].split("\n")[0]
    assert all(reasoning not in json.dumps(r["prompt"], ensure_ascii=False)
               for r in splits["train"][:50])
    assert [r["source_index"] for r in splits["train"]] == sorted(r["source_index"] for r in splits["train"])
    assert splits["test"][0]["sample_id"] == "test:0" and splits["test"][-1]["sample_id"] == f"test:{len(test) - 1}"
    assert be.valid_manifest(manifest) and manifest["task_id"] == gt.TASK_ID


def test_manifest_only_stores_hashes_not_text(recipe, official):
    train, test = official
    _, manifest = gt.prepare_splits(train, test, recipe)
    dumped = json.dumps(manifest, ensure_ascii=False)
    assert "小明" not in dumped and "####" not in dumped
    assert manifest["splits"]["test"]["samples"][0]["gold_sha256"] == mt.sha256_json(
        gt.extract_gold(test[0]["answer"]))


def test_validation_bucket_depends_on_data_seed_but_test_never_changes(recipe, official):
    train, test = official
    base, _ = gt.prepare_splits(train, test, recipe)
    other, manifest = gt.prepare_splits(train, test, mutated(recipe, "train.data_seed", 7))
    assert [r["sample_id"] for r in other["test"]] == [r["sample_id"] for r in base["test"]]
    assert {r["sample_id"] for r in other["validation"]} != {r["sample_id"] for r in base["validation"]}
    assert manifest["split_spec"]["data_seed"] == 7


@pytest.mark.parametrize("train_n,test_n", [(100, 1319), (7473, 10), (7472, 1319), (7473, 1320)])
def test_official_row_counts_are_enforced(recipe, train_n, test_n):
    with pytest.raises(mt.DatasetValidationError, match="官方行数"):
        gt.prepare_splits(gsm_rows(train_n, "训练"), gsm_rows(test_n, "测试"), recipe)


def mutate_row(rows, index, value):
    rows = list(rows)
    rows[index] = value
    return rows


@pytest.mark.parametrize("bad", [
    {"question": "q", "answer": "x\n#### 5", "extra": 1}, {"question": "q"}, {"answer": "#### 5"},
    {"question": "", "answer": "#### 5"}, {"question": "  ", "answer": "#### 5"},
    {"question": 5, "answer": "#### 5"}, {"question": "q", "answer": "无分隔符"},
    {"question": "q", "answer": "#### 1\n#### 2"}, {"question": "q", "answer": "#### five"},
    {"question": "答案 #### 5", "answer": "#### 5"}, ["q", "a"], None,
])
def test_invalid_rows_reject_the_whole_split(recipe, official, bad):
    train, test = official
    with pytest.raises(mt.DatasetValidationError):
        gt.prepare_splits(mutate_row(train, 3, bad), test, recipe)
    with pytest.raises(mt.DatasetValidationError):
        gt.prepare_splits(train, mutate_row(test, 3, bad), recipe)


def test_train_duplicates_are_dropped_and_recorded(recipe, official):
    train, test = official
    duplicated = mutate_row(train, 10, dict(train[3]))
    splits, manifest = gt.prepare_splits(duplicated, test, recipe)
    assert manifest["dropped_duplicate_train_indices"] == [10]
    assert len(splits["train"]) + len(splits["validation"]) == gt.OFFICIAL_TRAIN_ROWS - 1
    spaced = dict(train[3], question="  " + train[3]["question"].replace(" ", " \n ") + "\t")
    _, other = gt.prepare_splits(mutate_row(train, 11, spaced), test, recipe)
    assert other["dropped_duplicate_train_indices"] == [11]


def test_test_overlap_with_train_and_test_internal_duplicates_are_rejected(recipe, official):
    train, test = official
    with pytest.raises(mt.DatasetValidationError, match="重叠"):
        gt.prepare_splits(train, mutate_row(test, 5, dict(train[42])), recipe)
    with pytest.raises(mt.DatasetValidationError, match="重复"):
        gt.prepare_splits(train, mutate_row(test, 6, dict(test[1])), recipe)


def test_manifest_validation_binds_revision_and_split_protocol(recipe, official):
    train, test = official
    _, manifest = gt.prepare_splits(train, test, recipe)
    manifest["dataset_revision"] = recipe.get("dataset.revision")
    manifest.pop("sha256")
    manifest["sha256"] = mt.sha256_json(manifest)
    assert gt.validate_source_manifest(manifest, recipe) is manifest
    with pytest.raises(mt.TaskExecutionError, match="revision"):
        gt.validate_source_manifest(manifest, mutated(recipe, "dataset.revision", "c" * 40))
    with pytest.raises(mt.TaskExecutionError, match="拆分协议"):
        gt.validate_source_manifest(manifest, mutated(recipe, "train.data_seed", 9))
    broken = copy.deepcopy(manifest)
    broken["splits"]["test"]["samples"][0]["content_sha256"] = "0" * 64
    with pytest.raises(mt.TaskExecutionError):
        gt.validate_source_manifest(broken, recipe)


# --------------------------------------------------------------------------- #
# 路由与配置漂移
# --------------------------------------------------------------------------- #
def test_real_recipe_is_supported_and_not_registered_in_task_router():
    recipe = real_recipe()
    assert gt.require_supported(recipe) == gt.TASK_ID
    assert recipe.get("execution.task_id") is None
    assert task_router.resolve(recipe) is None
    # 未注册发布级管线：声明 task_id 会被 resolve() 拒绝，而不是借用别的任务协议。
    with pytest.raises(mt.TaskExecutionError, match="没有该 recipe 的发布级管线实现"):
        task_router.resolve(mutated(recipe, "execution.task_id", gt.TASK_ID))


@pytest.mark.parametrize("path,value", [
    ("method.family", "sft"), ("method.trainer", "SFTTrainer"), ("model.base", "Qwen/Qwen2.5-7B-Instruct"),
    ("dataset.name", "other/gsm8k"), ("dataset.config", "socratic"), ("dependencies.trl", "0.15.0"),
    ("rewards.functions", ["match_format_exactly"]),
    ("rewards.functions", list(reversed(gt.REWARD_NAMES))),
    ("evaluation.metrics", ["pass@1"]), ("evaluation.metrics", ["pass@1", "format_compliance"]),
    ("peft", {"r": 8}), ("quantization", {"load_in_4bit": True}),
    ("train.use_vllm", True), ("train.save_strategy", "steps"),
    ("train.remove_unused_columns", True), ("train.bf16", "true"), ("train.gradient_checkpointing", 1),
    ("train.learning_rate", 0), ("train.learning_rate", float("nan")), ("train.temperature", -1.0),
    ("train.max_grad_norm", 0), ("train.beta", -0.1), ("train.warmup_ratio", float("inf")),
    ("train.lr_scheduler_type", ""), ("train.unknown_knob", 1),
    ("gsm8k.validation_percent", 20), ("gsm8k.official_test_rows", 1000),
    ("gsm8k.gold_delimiter", "="), ("gsm8k.near_miss_relative_tolerance", 0.2),
    ("gsm8k.protocol", "other-v1"), ("gsm8k.split_rule", ""), ("gsm8k.extra", 1),
    ("rewards.spec", [{"name": "match_format_exactly", "max_score": 1.0}]),
])
def test_drift_rejected_before_any_framework_use(path, value):
    with pytest.raises(mt.TaskExecutionError):
        gt.require_supported(mutated(real_recipe(), path, value))


@pytest.mark.parametrize("key", list(gt.TRAIN_REQUIRED))
def test_every_hyperparameter_must_be_explicit(key):
    with pytest.raises(mt.TaskExecutionError, match="显式冻结|必须"):
        gt.require_supported(mutated(real_recipe(), f"train.{key}", None, delete=True))


def test_other_recipes_and_foreign_task_ids_are_rejected():
    recipe = real_recipe()
    other = Recipe("llm/grpo/other", recipe.path, copy.deepcopy(recipe.data))
    with pytest.raises(mt.TaskExecutionError, match="尚未实现该执行路由"):
        gt.require_supported(other)
    with pytest.raises(mt.TaskExecutionError, match="尚未实现该执行路由"):
        gt.require_supported(recipe, task_id=mt.TASK_ID)
    assert gt.task_plan(other, build_plan(real_recipe(), "dry-run"), "dry-run")["execution_supported"] is False


def test_config_kwargs_budget_batch_seeds(recipe, tmp_path):
    out = tmp_path / "out"
    kwargs = gt.config_kwargs(recipe, {"mode": "full"}, out)
    assert kwargs["max_steps"] == 150 and kwargs["seed"] == 42 and kwargs["data_seed"] == 42
    assert kwargs["report_to"] == "none" and kwargs["output_dir"] == str(out)
    assert kwargs["use_vllm"] is False and kwargs["remove_unused_columns"] is False
    assert set(kwargs) == gt.TRAIN_KEYS | {"output_dir", "report_to"}
    assert gt.config_kwargs(recipe, {"mode": "smoke"}, out)["max_steps"] == 5
    with pytest.raises(mt.TaskExecutionError, match="预算"):
        gt.config_kwargs(mutated(recipe, "train.max_steps", 151), {"mode": "full"}, out)
    with pytest.raises(mt.TaskExecutionError, match="预算"):
        gt.config_kwargs(recipe, {"mode": "dry-run"}, out)
    for key, value in (("per_device_train_batch_size", 6), ("per_device_train_batch_size", 2),
                       ("num_generations", 1)):
        with pytest.raises(mt.TaskExecutionError, match="num_generations"):
            gt.config_kwargs(mutated(recipe, f"train.{key}", value), {"mode": "full"}, out)
    for key in ("gradient_accumulation_steps", "max_completion_length", "max_prompt_length",
                "num_train_epochs", "logging_steps"):
        for value in (0, -1, 1.5, True):
            with pytest.raises(mt.TaskExecutionError, match="正整数"):
                gt.config_kwargs(mutated(recipe, f"train.{key}", value), {"mode": "full"}, out)
    for seed in (None, -1, True, 2**32):
        with pytest.raises(mt.TaskExecutionError):
            gt.config_kwargs(mutated(recipe, "train.seed", seed), {"mode": "full"}, out)


@pytest.mark.parametrize("mode", ["dry-run", "smoke", "full"])
def test_task_plan_exposes_protocol_and_marks_release_unavailable(mode):
    recipe = real_recipe()
    plan = gt.task_plan(recipe, build_plan(recipe, mode), mode)
    assert plan["execution_supported"] is True and plan["task_id"] == gt.TASK_ID
    assert plan["prompt"] == {"system": gt.SYSTEM_PROMPT, "gold_in_prompt": False}
    assert plan["rewards"]["functions"] == list(gt.REWARD_NAMES)
    assert plan["rewards"]["correctness_tiers"] == list(gt.CORRECTNESS_TIERS)
    assert plan["run"]["steps"] == (150 if mode == "full" else 5)
    assert plan["evaluation"]["metrics"] == list(gt.METRICS)
    assert "unavailable" in plan["evaluation"]["release_pipeline"]
    assert plan["dataset"]["split_plan"]["official_test_rows"] == 1319
    assert build_plan(recipe, mode)["task_id"] == gt.TASK_ID


def test_fingerprint_binds_the_gsm8k_section_and_train_params():
    recipe = real_recipe()
    base = be.fingerprint(recipe)
    assert be.fingerprint(mutated(recipe, "gsm8k.validation_percent", 20)) != base
    assert be.fingerprint(mutated(recipe, "gsm8k.accuracy_definition", "改")) != base
    assert be.fingerprint(mutated(recipe, "train.beta", 0.05)) != base
    assert be.fingerprint(mutated(recipe, "rewards.functions", ["a"])) != base
    assert be.fingerprint(real_recipe()) == base


def test_release_pipeline_is_unavailable_and_baseline_stays_closed():
    recipe = real_recipe()
    assert be.suite_problems(recipe)
    assert be.zero_train_measured(recipe) is False
    assert not be.frozen_environment(recipe)
    runner_suite = runner("suite")
    plan = runner_suite.suite_plan(recipe, [42, 43, 44], str(ROOT / "runs" / "x"))
    assert plan["release_pipeline"] == "unavailable"
    with pytest.raises(mt.TaskExecutionError):
        runner_suite.run_suite(recipe, [42, 43, 44], str(ROOT / "runs" / "x"))
    with pytest.raises(mt.TaskExecutionError):
        runner_suite.aggregate_suite(recipe, [42, 43, 44], str(ROOT / "runs" / "x"))


def runner(name):
    spec = importlib.util.spec_from_file_location(f"gsm8k_test_runner_{name}", ROOT / "runner" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_tables_dispatch_exact_recipe_to_gsm8k_adapter():
    for name in ("train", "eval", "zero_shot"):
        assert runner(name).TASK_ADAPTERS[gt.RECIPE_ID] is gt


def test_math_route_still_rejects_this_recipe():
    with pytest.raises(mt.TaskExecutionError, match="尚未实现该执行路由"):
        mt.require_supported(real_recipe())


class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "trl", "transformers", "datasets", "math_verify", "peft"}:
            raise AssertionError("不应导入框架: " + fullname)


def test_module_import_and_dry_run_never_import_frameworks():
    program = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trl', 'transformers', 'datasets', 'math_verify', 'peft'}:
            raise AssertionError('不应导入框架: ' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, 'src')
from recipe_lab import gsm8k_task as gt
from recipe_lab.loader import load_recipes
from pathlib import Path
r = next(x for x in load_recipes(Path('.')) if x.id == gt.RECIPE_ID)
from recipe_lab.plan import build_plan
assert gt.task_plan(r, build_plan(r, 'dry-run'), 'dry-run')['execution_supported'] is True
"""
    result = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name,dry_flags,real_flag", [
    ("train", ["--dry-run", "--json"], "--smoke"),
    ("eval", [], "--full"),
    ("zero_shot", [], "--full"),
])
def test_cli_dry_run_and_real_run_fail_closed_without_frozen_inputs(name, dry_flags, real_flag):
    def invoke(*flags):
        return subprocess.run([sys.executable, str(ROOT / "runner" / f"{name}.py"),
                               "--recipe", gt.RECIPE_ID, *flags],
                              cwd=ROOT, capture_output=True, text=True, check=False)
    dry = invoke(*dry_flags)
    assert dry.returncode == 0, dry.stderr
    assert json.loads(dry.stdout)["execution_supported"] is True
    real = invoke(real_flag)
    assert real.returncode == 1 and "执行失败" in real.stderr and "commit SHA" in real.stderr


# --------------------------------------------------------------------------- #
# fake runtime：训练与评估
# --------------------------------------------------------------------------- #
TRAIN_SIGNATURE = inspect.Signature([
    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    for name in sorted(gt.TRAIN_KEYS | {"output_dir", "report_to"})])


class FakeConfig:
    __signature__ = TRAIN_SIGNATURE

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return dict(self.__dict__)


class FakeEncoded(dict):
    def to(self, device):
        assert device == "cuda"
        return self


def make_runtime(official, *, completions=None, prompt_tokens=10):
    train, test = official
    calls = {"models": [], "tokenizers": [], "prompts": [], "generations": [], "datasets": [],
             "loads": []}
    tokenizer = SimpleNamespace(eos_token="<eos>", eos_token_id=0, pad_token=None)

    def save_tokenizer(directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    tokenizer.save_pretrained = save_tokenizer

    def template(prompt, **kwargs):
        assert kwargs["add_generation_prompt"] is True and kwargs["tokenize"] is True
        if kwargs.get("return_tensors"):
            calls["prompts"].append(copy.deepcopy(prompt))
            return FakeEncoded(input_ids=SimpleNamespace(shape=(1, 2)), attention_mask="mask")
        return [0] * prompt_tokens
    tokenizer.apply_chat_template = template
    counter = {"n": 0}

    def decode(*args, **kwargs):
        index = counter["n"]
        counter["n"] += 1
        return completions(index) if completions else GOOD
    tokenizer.decode = decode

    def load_tokenizer(source, **kwargs):
        calls["tokenizers"].append((source, kwargs))
        return tokenizer
    model = SimpleNamespace(to=lambda device: model, eval=lambda: None)

    def generate(**kwargs):
        calls["generations"].append(kwargs)
        return [[1, 2, 3, 4]]
    model.generate = generate

    def load_model(source, **kwargs):
        calls["models"].append((source, kwargs))
        return model

    def from_list(data):
        calls["datasets"].append(data)
        return data

    def load_dataset(name, config, revision=None, split=None):
        calls["loads"].append((name, config, revision, split))
        return {"train": train, "test": test}[split]
    dependencies = {
        "torch": SimpleNamespace(
            __version__="mock", float32="mock-float32", version=SimpleNamespace(cuda="mock"),
            cuda=SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                                 max_memory_allocated=lambda: 0, get_device_name=lambda _: "mock-GPU"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace())),
            use_deterministic_algorithms=lambda enabled: None, inference_mode=contextlib.nullcontext),
        "transformers": SimpleNamespace(
            __version__="mock", set_seed=Mock(),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model)),
        "datasets": SimpleNamespace(__version__="mock", Dataset=SimpleNamespace(from_list=from_list),
                                    load_dataset=load_dataset),
    }
    return dependencies, calls


def trainer_class(calls, *, incomplete=False):
    class FakeTrainer:
        def __init__(self, model, processing_class, args, train_dataset, eval_dataset, reward_funcs):
            calls["trainer"] = {"model": model, "processing_class": processing_class, "args": args,
                                "train_dataset": train_dataset, "eval_dataset": eval_dataset,
                                "reward_funcs": reward_funcs}
            self.state = SimpleNamespace(global_step=args.max_steps - int(incomplete))

        def train(self):
            calls["train_called"] = True
            return SimpleNamespace(metrics={"mock_train_loss": 0.25})

        def save_model(self, directory):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"mock weights")
    return FakeTrainer


def plan_for(recipe, tmp_path, mode):
    plan = gt.task_plan(recipe, build_plan(recipe, mode), mode)
    plan["run"]["output_dir"] = str(tmp_path / mode)
    return plan


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(mt, "require_frozen_runtime", lambda *args: None)


def test_training_assembles_contract_and_writes_evidence_only_after_budget(recipe, official, tmp_path, monkeypatch, patched):
    dependencies, calls = make_runtime(official)
    dependencies["trl"] = SimpleNamespace(__version__=gt.TRL_VERSION, GRPOConfig=FakeConfig,
                                          GRPOTrainer=trainer_class(calls))
    monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: dependencies)
    plan = plan_for(recipe, tmp_path, "full")
    path = gt.run_training(recipe, plan)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    output = tmp_path / "full"
    assert path == output / "training.json"
    assert evidence["kind"] == "training" and evidence["stage"] == "completed"
    assert evidence["task_id"] == gt.TASK_ID and evidence["steps"] == evidence["max_steps"] == 150
    assert evidence["rewards"] == list(gt.REWARD_NAMES) and evidence["baseline_promotion"] == "none"
    assert evidence["config_sha256"] == mt.sha256_json(evidence["config"])
    assert evidence["model_artifact"]["sha256"] == mt.artifact_manifest(output)["sha256"]
    assert evidence["prompt_audit"] == {"max_prompt_tokens": 10, "limit": 512}
    assert evidence["gsm8k"] == recipe.get("gsm8k") and evidence["upstream_facts"]["upstream_has_heldout_metrics"] is False
    assert evidence["fingerprint"] == be.fingerprint(recipe)
    assert evidence["seed"] == 42 and evidence["data_seed"] == 42
    train_set, validation_set = calls["datasets"]
    assert set(train_set[0]) == {"prompt", "answer"} == set(validation_set[0])
    assert all(row["answer"].lstrip("-").isdigit() for row in train_set[:100])
    assert not any("####" in json.dumps(row, ensure_ascii=False) for row in train_set + validation_set)
    assert len(train_set) + len(validation_set) == gt.OFFICIAL_TRAIN_ROWS
    assert calls["trainer"]["reward_funcs"] == gt.reward_functions()
    assert calls["models"][0][0] == gt.MODEL_ID and calls["models"][0][1]["revision"] == "a" * 40
    assert calls["tokenizers"][0][1]["revision"] == "a" * 40
    assert [load[3] for load in calls["loads"]] == ["train", "test"]
    assert all(load[:3] == (gt.DATASET_ID, "main", "b" * 40) for load in calls["loads"])
    assert calls["trainer"]["args"].use_vllm is False and calls["trainer"]["args"].bf16 is True
    assert gt.checkpoint_path(plan, recipe=recipe) == output.resolve()
    (output / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(mt.TaskExecutionError, match="不一致"):
        gt.checkpoint_path(plan)


def test_incomplete_training_never_writes_completed_evidence(recipe, official, tmp_path, monkeypatch, patched):
    dependencies, calls = make_runtime(official)
    dependencies["trl"] = SimpleNamespace(__version__=gt.TRL_VERSION, GRPOConfig=FakeConfig,
                                          GRPOTrainer=trainer_class(calls, incomplete=True))
    monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: dependencies)
    with pytest.raises(mt.TaskExecutionError, match="未完成预算"):
        gt.run_training(recipe, plan_for(recipe, tmp_path, "smoke"))
    assert not (tmp_path / "smoke" / "training.json").exists()


def test_api_mismatch_and_unapplied_config_rejected_before_loading_models(recipe, official, tmp_path, monkeypatch, patched):
    dependencies, calls = make_runtime(official)
    narrow = inspect.Signature([inspect.Parameter("learning_rate", inspect.Parameter.POSITIONAL_OR_KEYWORD)])

    class Narrow:
        __signature__ = narrow
    dependencies["trl"] = SimpleNamespace(GRPOConfig=Narrow, GRPOTrainer=trainer_class(calls))
    monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: dependencies)
    with pytest.raises(mt.TaskExecutionError, match="API 不显式支持"):
        gt.run_training(recipe, plan_for(recipe, tmp_path, "smoke"))

    class Ignoring(FakeConfig):
        def __init__(self, **kwargs):
            super().__init__(**{**kwargs, "beta": 0.0})
    dependencies["trl"] = SimpleNamespace(GRPOConfig=Ignoring, GRPOTrainer=trainer_class(calls))
    with pytest.raises(mt.TaskExecutionError, match="未应用 beta"):
        gt.run_training(recipe, plan_for(recipe, tmp_path, "smoke"))
    assert calls["models"] == calls["tokenizers"] == [] and calls["loads"] == []


def test_overlong_prompt_is_rejected_instead_of_silently_truncated(recipe, official, tmp_path, monkeypatch, patched):
    dependencies, calls = make_runtime(official, prompt_tokens=513)
    dependencies["trl"] = SimpleNamespace(__version__=gt.TRL_VERSION, GRPOConfig=FakeConfig,
                                          GRPOTrainer=trainer_class(calls))
    monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: dependencies)
    with pytest.raises(mt.TaskExecutionError, match="拒绝静默截断"):
        gt.run_training(recipe, plan_for(recipe, tmp_path, "smoke"))
    assert calls["models"] == [] and "trainer" not in calls


def test_training_guards_run_before_framework_loading(recipe, tmp_path, monkeypatch):
    loader = Mock(side_effect=AssertionError("加载依赖之前应拒绝"))
    monkeypatch.setattr(gt, "load_runtime", loader)
    plan = plan_for(recipe, tmp_path, "smoke")
    with pytest.raises(mt.TaskExecutionError, match="commit SHA"):
        gt.run_training(real_recipe(), plan_for(real_recipe(), tmp_path, "smoke"))
    blocked = mutated(recipe, "decision.gate", "blocked")
    with pytest.raises(mt.TaskExecutionError, match="决策门"):
        gt.run_training(blocked, plan)
    (tmp_path / "smoke").mkdir()
    (tmp_path / "smoke" / "old").write_text("x", encoding="utf-8")
    with pytest.raises(mt.TaskExecutionError, match="训练目录非空"):
        gt.run_training(recipe, plan)
    with pytest.raises(mt.TaskExecutionError, match="尚未实现"):
        gt.run_training(Recipe("llm/other", recipe.path, recipe.data), plan)
    loader.assert_not_called()


def create_checkpoint(path, recipe, *, mode="full"):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"mock checkpoint")
    mt.save_json(path / "training.json", {
        "kind": "training", "stage": "completed", "mode": mode, "task_id": gt.TASK_ID,
        "recipe_id": recipe.id, "model_revision": recipe.get("model.revision"),
        "dataset_revision": recipe.get("dataset.revision"), "fingerprint": be.fingerprint(recipe),
        "model_artifact": mt.artifact_manifest(path), "mock": True})
    return path


def scripted(official):
    """偶数题答对（严格格式），其余答错但格式满额，用来同时覆盖 accuracy 与 hacking。"""
    golds = [gt.extract_gold(row["answer"]) for row in official[1]]

    def completion(index):
        if index % 2 == 0:
            return f"<think>算</think><answer>{golds[index]}</answer>"
        return "<think>算</think><answer>999999</answer>"
    return completion


def test_zero_shot_and_trained_eval_share_prompt_seed_and_greedy_protocol(recipe, official, tmp_path, monkeypatch, patched):
    checkpoint = create_checkpoint(tmp_path / "explicit", recipe)
    results = []
    for mode, zero_shot in (("eval", False), ("zero-shot", True)):
        dependencies, calls = make_runtime(official, completions=scripted(official))
        monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: dependencies)
        path = gt.run_evaluation(recipe, plan_for(recipe, tmp_path, mode),
                                 checkpoint=None if zero_shot else checkpoint, zero_shot=zero_shot)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        predictions = json.loads((path.parent / "predictions.json").read_text(encoding="utf-8"))
        assert len(predictions) == 1319 and predictions[0]["sample_id"] == "test:0"
        assert evidence["kind"] == "evaluation" and evidence["stage"] == "completed"
        assert evidence["task_id"] == gt.TASK_ID and evidence["sample_count"] == 1319
        assert evidence["evaluation_protocol"] == gt.EVAL_PROTOCOL
        assert evidence["model_role"] == ("base" if zero_shot else "trained_checkpoint")
        assert set(evidence["metrics"]) == set(gt.METRICS)
        assert evidence["metrics"] == gt.recompute_metrics(predictions)
        assert evidence["metrics"]["pass@1"] == 660 / 1319
        assert evidence["metrics"]["format_compliance"] == 1.0
        assert evidence["metrics"]["reward_hacking_rate"] == 659 / 1319
        assert evidence["metric_counts"] == {"correct": 660, "format_compliant": 1319,
                                             "reward_hacking": 659, "total": 1319}
        assert evidence["diagnostics"] == gt.diagnostics(predictions)
        assert evidence["config_sha256"] == mt.sha256_json(evidence["config"])
        assert evidence["predictions"]["sha256"] == mt.sha256_json(predictions)
        assert evidence["training_seed_variance_measured"] is False
        assert "expected" not in evidence and evidence["baseline_promotion"] == "none"
        assert all(c["do_sample"] is False and c["num_beams"] == 1 for c in calls["generations"])
        assert all(p[0]["content"] == gt.SYSTEM_PROMPT for p in calls["prompts"])
        assert calls["models"][0][0] == (gt.MODEL_ID if zero_shot else str(checkpoint.resolve()))
        loader = calls["models"][0][1]
        assert loader.get("revision") == ("a" * 40 if zero_shot else None)
        assert loader.get("local_files_only") == (None if zero_shot else True)
        dependencies["transformers"].set_seed.assert_called_once_with(42)
        assert all("####" not in json.dumps(p["prompt"], ensure_ascii=False) for p in predictions)
        results.append((calls["prompts"], evidence["split_manifest"], evidence["config"]["test_sha256"]))
    assert results[0] == results[1]


def test_eval_gold_is_in_predictions_but_never_in_generation_prompts(recipe, official, tmp_path, monkeypatch, patched):
    dependencies, calls = make_runtime(official)
    monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: dependencies)
    path = gt.run_evaluation(recipe, plan_for(recipe, tmp_path, "zero-shot"), zero_shot=True)
    predictions = json.loads((path.parent / "predictions.json").read_text(encoding="utf-8"))
    assert all(set(p["prompt"][1]) == {"role", "content"} for p in predictions)
    assert all(p["gold_answer"] == gt.extract_gold(official[1][i]["answer"]) for i, p in enumerate(predictions))
    assert all(official[1][i]["answer"] not in json.dumps(p["prompt"], ensure_ascii=False)
               for i, p in enumerate(predictions[:100]))
    assert all(p["completion"] == GOOD for p in predictions)


def test_eval_never_falls_back_to_base_and_rejects_repeated_runs(recipe, tmp_path, monkeypatch):
    loader = Mock(side_effect=AssertionError("加载依赖之前应拒绝"))
    monkeypatch.setattr(gt, "load_runtime", loader)
    plan = plan_for(recipe, tmp_path, "eval")
    with pytest.raises(mt.TaskExecutionError, match="没有训练 checkpoint"):
        gt.run_evaluation(recipe, plan)
    plan["requested_runs"] = 5
    with pytest.raises(mt.TaskExecutionError, match="训练多 seed"):
        gt.run_evaluation(recipe, plan)
    with pytest.raises(mt.TaskExecutionError, match="zero-shot 只能加载 base"):
        gt.run_evaluation(recipe, plan_for(recipe, tmp_path, "zero-shot"), checkpoint=tmp_path, zero_shot=True)
    with pytest.raises(mt.TaskExecutionError, match="commit SHA"):
        gt.run_evaluation(real_recipe(), plan_for(real_recipe(), tmp_path, "zero-shot"), zero_shot=True)
    loader.assert_not_called()
    assert not list(tmp_path.rglob("*.json"))


def test_checkpoint_provenance_is_enforced(recipe, tmp_path):
    plan = plan_for(recipe, tmp_path, "eval")
    path = tmp_path / "full"
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"w")
    with pytest.raises(mt.TaskExecutionError, match="缺少已完成"):
        gt.checkpoint_path(plan)
    other = create_checkpoint(tmp_path / "other", mutated(recipe, "train.beta", 0.05))
    with pytest.raises(mt.TaskExecutionError, match="配置或 revision"):
        gt.checkpoint_path(plan, other, recipe=recipe)
    smoke = create_checkpoint(tmp_path / "smoke_ckpt", recipe, mode="smoke")
    # 显式路径允许 smoke checkpoint，但默认 full 输出目录只认 full 训练。
    assert gt.checkpoint_path(plan, smoke, recipe=recipe) == smoke.resolve()
    implicit = create_checkpoint(tmp_path / "implicit" / "full", recipe, mode="smoke")
    with pytest.raises(mt.TaskExecutionError, match="已完成训练证据"):
        gt.checkpoint_path({"run": {"output_dir": str(implicit.parent / "eval")}})
    foreign = create_checkpoint(tmp_path / "foreign", recipe)
    evidence = json.loads((foreign / "training.json").read_text(encoding="utf-8"))
    evidence["task_id"] = mt.TASK_ID
    mt.save_json(foreign / "training.json", evidence)
    with pytest.raises(mt.TaskExecutionError, match="已完成训练证据"):
        gt.checkpoint_path(plan, foreign)


def test_generation_failure_writes_no_completed_evidence(recipe, official, tmp_path, monkeypatch, patched):
    dependencies, calls = make_runtime(official)

    def broken(**kwargs):
        raise RuntimeError("cuda oom")
    dependencies["transformers"].AutoModelForCausalLM.from_pretrained = lambda *a, **k: SimpleNamespace(
        to=lambda d: SimpleNamespace(eval=lambda: None, generate=broken), eval=lambda: None, generate=broken)
    monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: dependencies)
    with pytest.raises(RuntimeError, match="cuda oom"):
        gt.run_evaluation(recipe, plan_for(recipe, tmp_path, "zero-shot"), zero_shot=True)
    assert not list(tmp_path.rglob("evaluation.json"))


def test_existing_evaluation_directory_is_never_overwritten(recipe, official, tmp_path, monkeypatch, patched):
    dependencies, _ = make_runtime(official)
    monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: dependencies)
    plan = plan_for(recipe, tmp_path, "zero-shot")
    gt.run_evaluation(recipe, plan, zero_shot=True)
    with pytest.raises(mt.TaskExecutionError, match="已存在评估证据"):
        gt.run_evaluation(recipe, plan, zero_shot=True)


def test_digits_helper_matches_decimal_semantics():
    assert gt.canonical_number(Decimal("100")) == "100" and gt.canonical_number(Decimal("1E+2")) == "100"
    assert gt.canonical_number(Decimal("-0.0")) == "0" and gt.canonical_number(Decimal("0.100")) == "0.1"
