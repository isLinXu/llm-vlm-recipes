"""xLAM code-path 的隔离离线测试：只写 tmp_path，不联网、不导入训练框架、不产 completed 证据。"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import task_router as tr
from recipe_lab import xlam_task as xt
from recipe_lab.loader import load_recipes
from recipe_lab.math_task import TaskExecutionError
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan

ROOT = Path(__file__).resolve().parents[1]
XLAM = "llm/sft/llama3-8b-xlam-function-calling"


class Tokenizer:
    """确定性字符级假 tokenizer；不加载 HF，不声明任何真实模型行为。"""
    name_or_path = "fake/xlam-tokenizer"
    vocab_size = 256
    eos_token = "</s>"
    eos_token_id = 2
    pad_token = "</s>"
    pad_token_id = 2

    def __call__(self, text, add_special_tokens=True, **kwargs):
        assert isinstance(text, str)
        return {"input_ids": [ord(char) % 256 for char in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(index % 256) for index in ids)


class FakeTorch:
    long = "int64"

    @staticmethod
    def tensor(value, dtype=None):
        return {"value": value, "dtype": dtype}


@pytest.fixture
def recipe(tmp_path):
    data = {
        "id": xt.RECIPE_ID, "source_sha": "c" * 40,
        "method": {"family": "sft", "trainer": "Trainer", "upstream_trainer": "SFTTrainer", "peft": "qlora"},
        "model": {"base": xt.MODEL, "params_b": 8.0, "modality": "llm", "revision": "a" * 40},
        "dataset": {"name": xt.DATASET, "structure": "conversational", "type": "prompt-completion",
                    "kind": "static", "has_images": False, "revision": "b" * 40},
        "xlam": copy.deepcopy(xt.XLAM_SPEC),
        "upstream": {"defaults_from": "TrainingConfig dataclass", "dataset_text_field": "text",
                     "max_steps": 1000, "max_seq_length": 2048, "save_steps": 250,
                     "logging_steps": 10, "warmup_ratio": 0.1,
                     "seed_note": "上游无 seed", "evaluation": "无 held-out"},
        "peft": copy.deepcopy(xt.PEFT_SPEC), "quantization": copy.deepcopy(xt.QUANT_SPEC),
        "train": {
            "seed": 42, "data_seed": 42, "seed_mechanism": "上游 TrainingConfig/SFTConfig 无 seed",
            "learning_rate": 1e-4, "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16, "gradient_accumulation_steps": 8,
            "max_steps": 150, "seq_length": 2048, "lr_scheduler_type": "linear",
            "weight_decay": 0.0, "warmup_ratio": 0.1, "eval_steps": 25, "save_steps": 25,
            "logging_steps": 10, "bf16": True, "fp16": False, "optim": "adamw_8bit",
            "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.05,
            "gradient_checkpointing": True, "dataloader_num_workers": 0,
            "remove_unused_columns": False,
        },
        "budget": {"full_steps": 150, "smoke_steps": 5},
        "evaluation": {"metrics": list(xt.METRICS), "seed": 42, "max_new_tokens": 8,
                       "multi_run": True, "held_out": True, "split": "content-hash 80/10/10"},
        "execution": {"task_id": xt.TASK_ID, "lockfile": "test.lock.toml"},
        "baseline": {"result": "unmeasured"},
    }
    lock = "\n".join([
        "schema_version = 1", 'status = "frozen"', f'recipe_id = "{xt.RECIPE_ID}"',
        f'model_id = "{xt.MODEL}"', f'dataset_id = "{xt.DATASET}"',
        f'model_revision = "{"a" * 40}"', f'dataset_revision = "{"b" * 40}"',
        f'source_sha = "{"c" * 40}"',
        'requirements = ["torch==test", "transformers==test", "datasets==test", "peft==test", "bitsandbytes==test", "safetensors==test"]',
        "[runtime]", 'python = "test"', 'cuda = "test"', 'driver = "test"', 'gpu = "test"',
    ])
    (tmp_path / "test.lock.toml").write_text(lock, encoding="utf-8")
    return Recipe(xt.RECIPE_ID, tmp_path / "recipe.toml", data)


def make_rows(count=200):
    rows = []
    for index in range(count):
        tools = [{"name": f"tool_{index}", "description": "synthetic tool",
                  "parameters": {"amount": {"type": "int", "description": "n", "required": True},
                                 "label": {"type": "str", "description": "s", "required": False}}}]
        answers = [{"name": f"tool_{index}", "arguments": {"amount": index}}]
        rows.append({"id": index, "query": f"Synthetic query number {index}",
                     "tools": json.dumps(tools), "answers": json.dumps(answers)})
    return rows


@pytest.fixture
def rows():
    return make_rows()


def real_recipe(root):
    return {r.id: r for r in load_recipes(root)}[XLAM]


# ---------------------------------------------------------------- 契约与计划

def test_real_recipe_matches_module_contract(root, recipe):
    real = real_recipe(root)
    assert xt.require_supported(real) == xt.TASK_ID
    assert xt.require_supported(recipe) == xt.TASK_ID
    assert real.get("xlam") == xt.XLAM_SPEC
    assert real.get("peft") == xt.PEFT_SPEC
    assert real.get("method.upstream_trainer") == "SFTTrainer"
    assert real.get("method.trainer") == "Trainer"


@pytest.mark.parametrize("mutate", [
    lambda d: d["train"].update({"surprise": 1}),
    lambda d: d["train"].update({"max_steps": 151}),
    lambda d: d["evaluation"].update({"metrics": ["accuracy"]}),
    lambda d: d["xlam"].update({"output_format": "python-repr"}),
    lambda d: d.pop("upstream"),
    lambda d: d["method"].update({"trainer": "SFTTrainer"}),
    lambda d: d["quantization"].update({"double_quant": False}),
    lambda d: d["evaluation"].update({"max_new_tokens": 4096}),
])
def test_require_supported_rejects_any_drift(recipe, mutate):
    data = copy.deepcopy(recipe.data)
    mutate(data)
    with pytest.raises(TaskExecutionError):
        xt.require_supported(Recipe(recipe.id, recipe.path, data))


def test_task_plan_is_readonly_and_task_specific(recipe, tmp_path):
    before = json.dumps(recipe.data, sort_keys=True, ensure_ascii=False)
    plan = xt.task_plan(recipe, build_plan(recipe, "dry-run"), "dry-run")
    assert plan["execution_supported"] is True
    assert plan["task_id"] == xt.TASK_ID
    assert plan["evaluation"]["protocol"] == xt.EVAL_PROTOCOL
    assert plan["evaluation"]["generated_code_executed"] is False
    assert plan["evaluation"]["upstream_quantitative_evaluation"] is False
    assert plan["migration"]["license"].startswith("数据卡 gated")
    assert json.dumps(recipe.data, sort_keys=True, ensure_ascii=False) == before
    assert not (tmp_path / "runs").exists()
    config = xt.config_kwargs(recipe, {"mode": "full", "recipe_id": recipe.id,
                                       "task_id": xt.TASK_ID, "run": {"steps": 150}},
                              tmp_path / "output")
    assert config["max_steps"] == 150 and config["output_dir"] == str(tmp_path / "output")
    assert xt.training_arguments_identity(config)["max_steps"] == 150
    assert "output_dir" not in xt.training_arguments_identity(config)


# ---------------------------------------------------------------- 数据协议

def test_parse_row_accepts_primary_schema(rows):
    row = xt.parse_row(rows[0], 0)
    assert row["source_index"] == 0 and row["content_sha256"]
    assert row["tools"] == json.dumps(json.loads(rows[0]["tools"]), ensure_ascii=False,
                                      sort_keys=True, separators=(",", ":"))
    assert row["answers_obj"][0]["name"] == "tool_0"
    assert row["answers"] not in xt.render_prompt(row)
    assert xt.render_target(row, "</s>").startswith("<calls>[")
    assert xt.render_target(row, "</s>").endswith("</calls></s>")


@pytest.mark.parametrize("mutate", [
    lambda d: d.update({"unexpected": 1}),
    lambda d: d.pop("answers"),
    lambda d: d.update({"tools": "not-json"}),
    lambda d: d.update({"tools": "[]"}),
    lambda d: d.update({"answers": "[]"}),
    lambda d: d.update({"query": ""}),
    lambda d: d.update({"answers": json.dumps([{"name": "missing_tool", "arguments": {}}])}),
    lambda d: d.update({"answers": json.dumps([{"name": "tool_0", "arguments": {"extra": 1}}])}),
    lambda d: d.update({"answers": json.dumps([{"name": "tool_0", "arguments": {}}])}),
    lambda d: d.update({"answers": json.dumps([{"name": "tool_0", "arguments": {"amount": "text"}}])}),
])
def test_parse_row_rejects_invalid_rows(rows, mutate):
    raw = copy.deepcopy(rows[0])
    mutate(raw)
    with pytest.raises(TaskExecutionError):
        xt.parse_row(raw, 0)


def test_duplicate_tool_names_rejected(rows):
    raw = copy.deepcopy(rows[0])
    tools = json.loads(raw["tools"])
    tools.append(copy.deepcopy(tools[0]))
    raw["tools"] = json.dumps(tools)
    with pytest.raises(TaskExecutionError, match="重名"):
        xt.parse_row(raw, 0)


def test_prepare_splits_is_deterministic_and_manifest_verifiable(recipe, rows):
    splits, manifest = xt.prepare_splits(rows, recipe)
    assert all(splits[name] for name in ("train", "validation", "test"))
    assert manifest["counts"]["source"] == len(rows)
    assert sum(manifest["counts"][n] for n in ("train", "validation", "test")) == len(rows)
    assert manifest["counts"]["test_eval_used"] <= manifest["counts"]["test"]
    xt.validate_source_manifest(manifest, recipe)
    again, second = xt.prepare_splits(rows, recipe)
    assert second == manifest
    assert [r["sample_id"] for r in again["test"]] == [r["sample_id"] for r in splits["test"]]
    for name, items in splits.items():
        for row in items:
            assert row["answers"] not in xt.render_prompt(row)


def test_prepare_splits_rejects_duplicates_and_empty_splits(recipe, rows):
    with pytest.raises(TaskExecutionError, match="重复"):
        xt.prepare_splits(rows + [copy.deepcopy(rows[0])], recipe)
    with pytest.raises(TaskExecutionError, match="为空"):
        xt.prepare_splits(rows[:1], recipe)


def test_validate_source_manifest_rejects_tampering(recipe, rows):
    _, manifest = xt.prepare_splits(rows, recipe)
    for mutate in (
        lambda m: m["split_hashes"]["train"].append("0" * 64),
        lambda m: m["split_hashes"]["test"].__setitem__(0, m["split_hashes"]["train"][0]),
        lambda m: m.update({"data_seed": 43}),
        lambda m: m.update({"protocol": "invented-v1"}),
        lambda m: m.update({"counts": {**m["counts"], "train": 1}}),
    ):
        broken = copy.deepcopy(manifest)
        mutate(broken)
        with pytest.raises(TaskExecutionError):
            xt.validate_source_manifest(broken, recipe)


# ---------------------------------------------------------------- 编码与 completion-only loss

def test_encode_example_masks_prompt_and_keeps_gold_out(recipe, rows):
    tokenizer = Tokenizer()
    identity = xt.token_identity(tokenizer)
    row = xt.parse_row(rows[0], 0)
    item = xt.encode_example(row, tokenizer, identity, recipe)
    prompt_tokens = item["prompt_tokens"]
    assert prompt_tokens > 0 and item["target_tokens"] > 0
    assert item["labels"][:prompt_tokens] == [-100] * prompt_tokens
    assert item["labels"][prompt_tokens:] == item["input_ids"][prompt_tokens:]
    assert all(mask == 1 for mask in item["attention_mask"])
    assert xt.render_target(row, tokenizer.eos_token) not in xt.render_prompt(row)
    assert row["answers"] not in tokenizer.decode(item["input_ids"][:prompt_tokens])


def test_encode_example_rejects_overflow(recipe, rows):
    tokenizer = Tokenizer()
    identity = xt.token_identity(tokenizer)
    small = copy.deepcopy(recipe.data)
    small["train"]["seq_length"] = 8
    small["evaluation"]["max_new_tokens"] = 1
    r = Recipe(recipe.id, recipe.path, small)
    with pytest.raises(TaskExecutionError, match="seq_length"):
        xt.encode_example(xt.parse_row(rows[0], 0), tokenizer, identity, r)


def test_collate_pads_labels_and_mask(recipe, rows):
    tokenizer = Tokenizer()
    identity = xt.token_identity(tokenizer)
    items = [xt.encode_example(xt.parse_row(raw, i), tokenizer, identity, recipe)
             for i, raw in enumerate(rows[:3])]
    batch = xt.collate(items, FakeTorch, tokenizer.pad_token_id)
    width = max(len(item["input_ids"]) for item in items)
    assert all(len(row) == width for row in batch["labels"]["value"])
    for index, item in enumerate(items):
        pad = width - len(item["input_ids"])
        assert batch["labels"]["value"][index][-pad:] == [-100] * pad if pad else True
        assert batch["attention_mask"]["value"][index][-pad:] == [0] * pad if pad else True


# ---------------------------------------------------------------- 评估协议

def test_parse_prediction_requires_strict_json_and_never_evaluates(recipe, rows):
    row = xt.parse_row(rows[0], 0)
    gold = row["answers"]
    assert xt.parse_prediction(f"<calls>{gold}</calls>") == row["answers_obj"]
    assert xt.parse_prediction("<calls>[{'name': 'tool_0', 'arguments': {'amount': 0}}]</calls>") is None
    assert xt.parse_prediction("<user>no calls</user>") is None
    assert xt.parse_prediction("<calls>__import__('os').system('echo pwned')</calls>") is None
    assert xt.parse_prediction("<calls>[{\"name\": 1, \"arguments\": {}}]</calls>") is None
    assert xt.parse_prediction("<calls>[]</calls>") is None


def test_score_prediction_and_metric_recompute(recipe, rows):
    row = xt.parse_row(rows[0], 0)
    gold = row["answers"]
    perfect = xt.score_prediction(row, f"<calls>{gold}</calls>")
    assert perfect["json_parse_rate"] == perfect["schema_valid_rate"] == 1.0
    assert perfect["tool_call_exact_match"] == 1.0 and perfect["argument_f1"] == 1.0
    broken = xt.score_prediction(row, "<calls>{'name': 'tool_0'}</calls>")
    assert all(broken[name] == 0.0 for name in xt.METRICS)
    wrong_tool = xt.score_prediction(row, '<calls>[{"name":"other","arguments":{}}]</calls>')
    assert wrong_tool["json_parse_rate"] == 1.0 and wrong_tool["schema_valid_rate"] == 0.0
    assert wrong_tool["tool_call_exact_match"] == 0.0
    metrics = xt.recompute_metrics([perfect, broken])
    assert metrics == {name: 0.5 for name in xt.METRICS}
    tampered = copy.deepcopy(perfect)
    tampered["tool_call_exact_match"] = 1.5
    with pytest.raises(TaskExecutionError, match="非法分数"):
        xt.recompute_metrics([tampered])


def test_validate_calls_checks_schema_not_execution(recipe, rows):
    row = xt.parse_row(rows[0], 0)
    assert xt.validate_calls(row["answers_obj"], row["tools_obj"]) is True
    assert xt.validate_calls([{"name": "unknown", "arguments": {}}], row["tools_obj"]) is False
    assert xt.validate_calls([{"name": "tool_0", "arguments": {"amount": "bad"}}], row["tools_obj"]) is False


def test_train_examples_and_transform_manifest(recipe, rows):
    splits, source = xt.prepare_splits(rows, recipe)
    tokenizer = Tokenizer()
    identity = xt.token_identity(tokenizer)
    examples, transform = xt.train_examples(splits, tokenizer, recipe)
    assert examples["train"] and examples["validation"]
    assert transform["completion_only_loss"] is True and transform["gold_not_in_prompt"] is True
    xt.validate_transform_manifest(transform, source, recipe, identity)
    for item in examples["train"]:
        assert item["labels"].count(-100) >= 1
        assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])
    broken = copy.deepcopy(transform)
    broken["completion_only_loss"] = False
    with pytest.raises(TaskExecutionError):
        xt.validate_transform_manifest(broken, source, recipe, identity)


# ---------------------------------------------------------------- 路由与发布门禁

def test_router_dispatches_all_xlam_pipelines(root, recipe):
    """comparison/export 已补齐：三类模块都必须按精确 task_id 落到 xLAM 实现。"""
    from recipe_lab import xlam_export as xe
    from recipe_lab import xlam_suite as xs
    real = real_recipe(root)
    assert tr.resolve(real) == xt.TASK_ID
    assert tr.load(real, "task") is xt
    assert tr.load(real, "suite") is xs and tr.load(real, "export") is xe
    # 未知任务仍必须失败关闭，绝不借用数学 / FIM 协议。
    with pytest.raises(TaskExecutionError, match="没有实现|不得用数学协议替代"):
        tr._module("invented-v1", "suite")


def test_baseline_evidence_dispatches_to_xlam_and_fails_closed(root, tmp_path):
    """发布门禁改为按 xLAM 内容重算器拒绝，而不是「未实现」占位文本。"""
    real = real_recipe(root)
    problems = be.evaluation_problems(
        real, {"kind": "evaluation", "stage": "completed"}, tmp_path / "e.json", role="base")
    assert problems and "xLAM" in problems[0]
    assert be.suite_problems(real)  # 无 completed suite 证据时仍失败关闭
    assert be.zero_train_measured(real) is False


def test_fingerprint_binds_xlam_protocol(root):
    real = real_recipe(root)
    changed = copy.deepcopy(real.data)
    changed["xlam"]["output_format"] = "python-repr"
    other = Recipe(real.id, real.path, changed)
    assert be.fingerprint(real) != be.fingerprint(other)


def cli_process(script, arguments, tmp_path):
    program = '''import importlib.abc, runpy, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trl', 'transformers', 'datasets', 'math_verify', 'peft', 'bitsandbytes', 'safetensors'}:
            raise AssertionError('隔离测试禁止框架: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    return subprocess.run(
        [sys.executable, "-B", "-c", program, str(ROOT / "runner" / script), *arguments],
        cwd=tmp_path, env={**os.environ, "RECIPE_LAB_ROOT": str(tmp_path)},
        capture_output=True, text=True, timeout=30, check=False)


def test_cli_dry_runs_are_framework_free_and_write_nothing(root, tmp_path):
    directory = tmp_path / "recipes" / "llm" / "sft" / "llama3-8b-xlam-function-calling"
    directory.mkdir(parents=True)
    (directory / "recipe.toml").write_text(
        (root / "recipes" / "llm" / "sft" / "llama3-8b-xlam-function-calling" / "recipe.toml").read_text(
            encoding="utf-8"), encoding="utf-8")
    commands = [
        ("train.py", ["--recipe", XLAM, "--dry-run", "--json", "--output-root", str(tmp_path / "out-t")]),
        ("eval.py", ["--recipe", XLAM, "--output-root", str(tmp_path / "out-e")]),
        ("zero_shot.py", ["--recipe", XLAM, "--output-root", str(tmp_path / "out-z")]),
    ]
    for script, arguments in commands:
        result = cli_process(script, arguments, tmp_path)
        assert result.returncode == 0, result.stderr
        assert "AssertionError" not in result.stderr
        payload = json.loads(result.stdout)
        assert payload["task_id"] == xt.TASK_ID
        assert payload["execution_supported"] is True
    assert not any((tmp_path / name).exists() for name in ("out-t", "out-e", "out-z", "runs"))
