"""RapidFire rapidfire-sft-v1（data / train / eval）隔离离线契约测试。

一手事实边界：上游 eval range(50,60) ⊂ train range(64)，无 revision / 版本锁定 /
任何实际指标数值；本仓自建无泄漏 question-hash 划分、completion-only loss、
纯 Python 无 stemmer ROUGE-L，并顺序执行 4 配置。comparison / export 由
tests/test_rapidfire_release.py 另测：已按 ADR-019 精确注册 task_router，
发布级 dry-run 分派到 RapidFire 专用协议，不借用数学 / FIM / xLAM 协议。

所有产物只写 tmp_path；fake runtime 明确是合成夹具，不声明任何真实模型行为；
JSON/hash 只证明仓内内容一致性，不构成 GPU 实跑证据，也不晋级 B2/B3。
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import rapidfire_task as rt
from recipe_lab import task_router as tr
from recipe_lab.math_task import TaskExecutionError
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan

ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = "frozen.lock.toml"
BASE_CONFIG = "lr5e-4-r8"


# ---------------------------------------------------------------- 合成配置

def _recipe(tmp_path, config_id=BASE_CONFIG):
    base_id = rt.RAPIDFIRE_SPEC["base_experiment_id"]
    data = {
        "id": rt.RECIPE_ID, "source_sha": "c" * 40,
        "method": {"family": "hpo", "trainer": "Trainer",
                   "upstream_trainer": "Experiment.run_fit / RFSFTConfig", "peft": "lora"},
        "model": {"base": rt.MODEL, "params_b": 0.124, "modality": "llm", "revision": "a" * 40},
        "dataset": {"name": rt.DATASET, "structure": "standard", "type": "prompt-completion",
                    "kind": "static", "has_images": False, "revision": "b" * 40},
        "rapidfire": copy.deepcopy(rt.RAPIDFIRE_SPEC),
        "configs": rt.expected_configs(),
        "upstream": {"run_fit": "experiment.run_fit(..., num_chunks=4, seed=42)",
                     "train_range": "range(64)", "eval_range": "range(50, 60)",
                     "max_steps": 64, "evaluation": "无实际数值", "seed_note": "无 set_seed"},
        "peft": {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["c_attn"],
                 "bias": "none", "task_type": "CAUSAL_LM"},
        "train": {
            "seed": 0, "data_seed": 0, "seed_mechanism": "上游只在 shuffle/run_fit 用 seed=42",
            "learning_rate": 5e-4, "lr_scheduler_type": "linear", "warmup_steps": 0,
            "per_device_train_batch_size": 2, "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 2, "max_steps": 64, "seq_length": 512,
            "weight_decay": 0.0, "eval_steps": 4, "save_steps": 4, "logging_steps": 2,
            "fp16": True, "bf16": False, "optim": "adamw_torch",
            "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
            "gradient_checkpointing": True, "dataloader_num_workers": 0,
            "remove_unused_columns": False,
        },
        "budget": {"full_steps": 64, "smoke_steps": 5},
        "evaluation": {"metrics": list(rt.METRICS), "seed": 0, "max_new_tokens": 128,
                       "multi_run": True, "held_out": True,
                       "split": "question-hash 80/10/10"},
        "execution": {"config_id": BASE_CONFIG, "lockfile": LOCKFILE,
                      "command": "python3 runner/train.py --recipe llm/sft/rapidfire-multiconfig "
                                 "--config-id lr5e-4-r8 --full"},
        "baseline": {"result": "unmeasured"},
        "baseline_contract": {"experiment_id": f"{base_id}--{BASE_CONFIG}"},
    }
    lock = "\n".join([
        "schema_version = 1", 'status = "frozen"', f'recipe_id = "{rt.RECIPE_ID}"',
        f'model_id = "{rt.MODEL}"', f'dataset_id = "{rt.DATASET}"',
        f'model_revision = "{"a" * 40}"', f'dataset_revision = "{"b" * 40}"',
        f'source_sha = "{"c" * 40}"',
        'requirements = ["torch==test", "transformers==test", "datasets==test", '
        '"peft==test", "safetensors==test"]',
        "[runtime]", f'python = "{sys.version.split()[0]}"', 'cuda = "test-cuda"',
        'driver = "test-driver"', 'gpu = "test-device"',
    ]) + "\n"
    (tmp_path / LOCKFILE).write_text(lock, encoding="utf-8")
    (tmp_path / "recipe.toml").write_text(f'id = "{rt.RECIPE_ID}"\n', encoding="utf-8")
    recipe = Recipe(rt.RECIPE_ID, tmp_path / "recipe.toml", data)
    return recipe if config_id == BASE_CONFIG else rt.config_recipe(recipe, config_id)


def make_rows(count=400):
    rows = []
    for index in range(count):
        rows.append({
            "flags": "M" if index % 2 else "B",
            "instruction": f"Synthetic customer question number {index} about invoice ZQ{index}",
            "category": "INVOICE" if index % 3 else "ORDER",
            "intent": "check_invoice" if index % 3 else "place_order",
            "response": f"Synthetic assistant answer {index} with distinctive token ZQ{index} end.",
        })
    return rows


class Tokenizer:
    name_or_path = "synthetic/gpt2-tokenizer"
    vocab_size = 256
    eos_token = "</s>"
    eos_token_id = 2
    pad_token = "</s>"
    pad_token_id = 2
    padding_side = "right"

    @staticmethod
    def _ids(text):
        return [ord(char) % 256 for char in text]

    def __call__(self, text, add_special_tokens=True, return_tensors=None, **kwargs):
        assert isinstance(text, str)
        ids = self._ids(text)
        if return_tensors == "pt":
            return {"input_ids": Encoded([ids]),
                    "attention_mask": Encoded([[1] * len(ids)])}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(index % 256) for index in ids)

    def save_pretrained(self, directory):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "SyntheticTokenizer"}), encoding="utf-8")


class Encoded:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def to(self, device):
        return self

    def __getitem__(self, index):
        return self.rows[index]


class Tensor:
    dtype = "synthetic-float"

    def __init__(self, values):
        self.values = copy.deepcopy(values)

    @property
    def shape(self):
        return (len(self.values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, dtype=None):
        return self

    def all(self):
        return all(self.values)


class Parameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad

    def numel(self):
        return 10

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


class FakeTensorList(list):
    pass


def explicit_factory(keys):
    def factory(**kwargs):
        return SimpleNamespace(**kwargs)
    factory.__signature__ = inspect.Signature([
        inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY) for key in keys
    ])
    return factory


def fake_runtime(r, rows):
    """确定性合成 runtime：只实现 rapidfire_task 显式检查过的 API，不加载任何训练框架。"""
    tokenizer = Tokenizer()
    splits, _ = rt.prepare_splits(rows, r)
    gold_by_prompt = {
        tuple(Tokenizer._ids(rt.render_prompt(row))):
            Tokenizer._ids(rt.render_target(row, Tokenizer.eos_token))
        for row in splits["test"]
    }
    parameters = {"base.weight": Parameter(True)}
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True), device="synthetic:0")
    targets = list(r.get("peft.target_modules"))
    model.named_modules = lambda: [(f"layer.{name}", object()) for name in targets]
    model.named_parameters = lambda: list(parameters.items())
    model.parameters = lambda: list(parameters.values())
    model.eval = lambda: None

    def generate(input_ids, attention_mask, **kwargs):
        prompt = input_ids.rows[0]
        assert set(kwargs) == {"max_new_tokens", "do_sample", "num_beams",
                               "pad_token_id", "eos_token_id"}
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        gold = gold_by_prompt[tuple(prompt)]
        return FakeTensorList([prompt + gold[:kwargs["max_new_tokens"]]])

    model.generate = generate

    def save_pretrained(directory, safe_serialization):
        assert safe_serialization is True
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        adapter = {"base_model_name_or_path": rt.MODEL, "revision": r.get("model.revision"),
                   "r": r.get("peft.r"), "lora_alpha": r.get("peft.alpha"),
                   "lora_dropout": r.get("peft.dropout"),
                   "target_modules": list(r.get("peft.target_modules")),
                   "bias": r.get("peft.bias"), "task_type": r.get("peft.task_type"),
                   "peft_type": "LORA"}
        (target / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"synthetic-adapter-weights")

    model.save_pretrained = save_pretrained

    def load_dataset(source, **kwargs):
        return iter(copy.deepcopy(rows))

    def get_peft_model(model, peft_config):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name in targets:
            parameters[f"adapter.{name}.lora_A.weight"] = Parameter(True)
            parameters[f"adapter.{name}.lora_B.weight"] = Parameter(True)
        return model

    def state_dict(model, adapter_name, save_embedding_layers):
        return {"base_model.model.layer.lora_A.weight": Tensor([1.0, 2.0]),
                "base_model.model.layer.lora_B.weight": Tensor([3.0, 4.0])}

    def load_adapter(model, model_id, is_trainable):
        assert is_trainable is False
        return model

    canonical = rt.config_kwargs(r, {"mode": "full"}, Path("unused"))

    class Trainer:
        def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
            self.state = SimpleNamespace(global_step=args.max_steps)
            self.collator = data_collator

        def train(self):
            batch = self.collator([{"input_ids": [1, 2], "labels": [-100, 2],
                                   "attention_mask": [1, 1]},
                                  {"input_ids": [3], "labels": [3],
                                   "attention_mask": [1]}])
            assert set(batch) == {"input_ids", "labels", "attention_mask"}
            return SimpleNamespace(metrics={"train_loss": 0.5})

    runtime = {
        "torch": SimpleNamespace(
            __version__="test", float16="test-float16", long="test-long",
            tensor=lambda values, **kwargs: Tensor(values),
            isfinite=lambda tensor: Tensor([True] * len(tensor.values)),
            equal=lambda a, b: a.values == b.values,
            no_grad=contextlib.nullcontext,
            version=SimpleNamespace(cuda="test-cuda"),
            cuda=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda _: "test-device",
                                 max_memory_allocated=lambda: 0, is_available=lambda: True,
                                 device_count=lambda: 1),
            are_deterministic_algorithms_enabled=lambda: True,
        ),
        "transformers": SimpleNamespace(
            __version__="test", TrainingArguments=explicit_factory(list(canonical)), Trainer=Trainer,
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda source, **kwargs: Tokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda source, **kwargs: model),
            set_seed=lambda seed: None,
        ),
        "datasets": SimpleNamespace(__version__="test", load_dataset=load_dataset,
                                    Dataset=SimpleNamespace(from_list=lambda items: items)),
        "peft": SimpleNamespace(
            __version__="test", get_peft_model=get_peft_model,
            LoraConfig=explicit_factory(["r", "lora_alpha", "lora_dropout", "target_modules",
                                         "bias", "task_type", "base_model_name_or_path", "revision"]),
            get_peft_model_state_dict=state_dict,
            PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        ),
        "safetensors": SimpleNamespace(__version__="test"),
    }
    return runtime


def install_fake(monkeypatch, runtime):
    monkeypatch.setattr(rt, "load_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(rt, "require_frozen_runtime", lambda *args: None)
    monkeypatch.setattr(rt, "apply_determinism", lambda *args, **kwargs: None)
    monkeypatch.setattr(rt, "importlib", SimpleNamespace(
        import_module=lambda name: SimpleNamespace(
            load_file=lambda path, device: runtime["peft"].get_peft_model_state_dict(
                None, "default", False))))


def _plan(r, root, mode):
    raw = build_plan(r, mode)
    raw["run"]["output_dir"] = str(root)
    plan = rt.task_plan(r, raw, mode)
    assert plan["execution_supported"], plan.get("unsupported_reason")
    return plan


@pytest.fixture
def recipe(tmp_path):
    return _recipe(tmp_path)


@pytest.fixture
def rows():
    return make_rows()


# ---------------------------------------------------------------- 一手事实与配置网格

def test_primary_fact_constants_are_explicit():
    assert rt.MODEL == "gpt2"
    assert rt.DATASET == "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    assert rt.FULL_STEPS == 64 and rt.SMOKE_STEPS == 5
    assert "不可比" in rt.RAPIDFIRE_SPEC["metric_definition"]
    assert rt.RAPIDFIRE_SPEC["concurrent_scheduling"] is False
    assert rt.RAPIDFIRE_SPEC["upstream_quantitative_evaluation"] is False
    assert rt.RAPIDFIRE_SPEC["prompt_format"].endswith("Answer:")


def test_config_grid_is_exact_and_ids_unique(recipe):
    grid = rt.config_grid(recipe)
    assert [item["id"] for item in grid] == ["lr5e-4-r8", "lr5e-4-r32",
                                             "lr2e-4-r8", "lr2e-4-r32"]
    assert {(item["learning_rate"], item["lr_scheduler_type"], item["warmup_steps"])
            for item in grid} == {(5e-4, "linear", 0), (2e-4, "cosine", 10)}
    assert {(item["lora_r"], item["lora_alpha"], tuple(item["target_modules"]))
            for item in grid} == {(8, 16, ("c_attn",)),
                                  (32, 64, ("c_attn", "c_proj"))}


@pytest.mark.parametrize("change", [
    lambda r: r.data["configs"][0].update(learning_rate=9e-4),
    lambda r: r.data["configs"][0].pop("warmup_steps"),
    lambda r: r.data["configs"][0].update(extra_field=1),
    lambda r: r.data["configs"][1].update(id="lr5e-4-r8"),
    lambda r: r.data["configs"].pop(),
    lambda r: r.data["configs"][0].update(id="Invalid_ID"),
])
def test_config_grid_rejects_drift(recipe, change):
    change(recipe)
    with pytest.raises(TaskExecutionError):
        rt.config_grid(recipe)


def test_config_recipe_overrides_only_declared_fields_and_is_idempotent(tmp_path):
    base = _recipe(tmp_path)
    for config in rt.config_grid(base):
        derived = rt.config_recipe(base, config["id"])
        assert derived.id == base.id and derived.path == base.path
        changed = {key for key in set(base.data) | set(derived.data)
                   if base.data.get(key) != derived.data.get(key)}
        assert changed <= {"train", "peft", "execution", "baseline_contract"}
        assert derived.get("train.learning_rate") == config["learning_rate"]
        assert derived.get("peft.r") == config["lora_r"]
        assert derived.get("peft.target_modules") == config["target_modules"]
        assert derived.get("execution.config_id") == config["id"]
        assert derived.get("baseline_contract.experiment_id").endswith("--" + config["id"])
        rt.require_supported(derived)
        assert rt.config_recipe(derived, config["id"]).data == derived.data


def test_config_recipe_rejects_unknown_id_and_wrong_recipe(tmp_path):
    base = _recipe(tmp_path)
    with pytest.raises(TaskExecutionError, match="未知配置"):
        rt.config_recipe(base, "invented-config")
    other = Recipe("llm/sft/other", base.path, copy.deepcopy(base.data))
    with pytest.raises(TaskExecutionError):
        rt.config_recipe(other, BASE_CONFIG)


def test_require_supported_rejects_foreign_route_and_conflicting_task(tmp_path):
    base = _recipe(tmp_path)
    with pytest.raises(TaskExecutionError):
        rt.require_supported(base, task_id="math-grpo-v1")
    other = Recipe("llm/sft/other", base.path, copy.deepcopy(base.data))
    with pytest.raises(TaskExecutionError):
        rt.require_supported(other)


@pytest.mark.parametrize("change", [
    lambda r: r.data["train"].update(learning_rate=1e-3),
    lambda r: r.data["train"].update(lr_scheduler_type="cosine"),
    lambda r: r.data["train"].update(warmup_steps=10),
    lambda r: r.data["train"].update(lora_r=32),
    lambda r: r.data["train"].pop("remove_unused_columns"),
    lambda r: r.data["train"].update(unknown_key=1),
    lambda r: r.data["peft"].update(r=32),
    lambda r: r.data["peft"].update(target_modules=["c_proj"]),
    lambda r: r.data["rapidfire"].update(concurrent_scheduling=True),
    lambda r: r.data["rapidfire"].update(split_percent=[70, 20, 10]),
    lambda r: r.data["rapidfire"].pop("metric_definition"),
    lambda r: r.data["evaluation"].update(metrics=["accuracy"]),
    lambda r: r.data["evaluation"].update(held_out=False),
    lambda r: r.data["budget"].update(full_steps=65),
    lambda r: r.data["train"].update(max_steps=65),
    lambda r: r.data["train"].update(fp16=False),
    lambda r: r.data["upstream"].pop("eval_range"),
    lambda r: r.data["execution"].update(config_id="invented"),
    lambda r: r.data["baseline_contract"].update(experiment_id="wrong"),
    lambda r: r.data["method"].update(upstream_trainer="SFTTrainer"),
])
def test_require_supported_fails_closed_on_drift(tmp_path, change):
    r = _recipe(tmp_path)
    change(r)
    with pytest.raises(TaskExecutionError):
        rt.require_supported(r)


def test_require_supported_on_all_four_configs(tmp_path):
    base = _recipe(tmp_path)
    for config in rt.config_grid(base):
        assert rt.require_supported(rt.config_recipe(base, config["id"])) == rt.TASK_ID


# ---------------------------------------------------------------- 计划层

def test_task_plan_dry_run_is_config_specific_and_readonly(recipe, tmp_path):
    before = copy.deepcopy(recipe.data)
    plan = _plan(recipe, tmp_path / "plan", "dry-run")
    assert recipe.data == before
    assert plan["task_id"] == rt.TASK_ID and plan["execution_supported"] is True
    assert plan["config_grid"]["active"] == BASE_CONFIG
    assert plan["config_grid"]["ids"] == [c["id"] for c in rt.config_grid(recipe)]
    assert plan["evaluation"]["metrics"] == list(rt.METRICS)
    assert plan["evaluation"]["metric_comparable_to_upstream_rougeL"] is False
    assert "eval[50:60]" in plan["migration"]["split"]
    assert not (tmp_path / "plan").exists()


def test_task_plan_command_tracks_active_config(tmp_path):
    for config in rt.expected_configs():
        r = _recipe(tmp_path, config["id"])
        plan = _plan(r, tmp_path / "plan", "dry-run")
        assert f"--config-id {config['id']}" in plan["command"]
        assert plan["run"]["params"]["lr_scheduler_type"] == config["lr_scheduler_type"]
        assert plan["run"]["params"]["learning_rate"] == config["learning_rate"]
        assert config["id"] in plan["experiment_id"]


def test_config_kwargs_uses_declared_budget(recipe, tmp_path):
    for mode, steps in (("smoke", 5), ("full", 64)):
        config = rt.config_kwargs(recipe, {"mode": mode, "recipe_id": recipe.id},
                                  tmp_path / mode)
        assert config["max_steps"] == steps and config["output_dir"] == str(tmp_path / mode)
        assert config["evaluation_strategy"] == "steps"
    with pytest.raises(TaskExecutionError):
        rt.config_kwargs(recipe, {"mode": "dry-run", "recipe_id": recipe.id}, tmp_path / "x")


# ---------------------------------------------------------------- 数据协议

def test_parse_row_rejects_unknown_columns_and_empty_gold():
    good = make_rows(1)[0]
    row = rt.parse_row(good, 7)
    assert row["source_index"] == 7 and row["question_sha256"] != row["content_sha256"]
    for bad in ({**good, "extra": "x"}, {k: v for k, v in good.items() if k != "flags"},
                {**good, "response": "   "}, {**good, "instruction": 123}):
        with pytest.raises(TaskExecutionError):
            rt.parse_row(bad, 0)


def test_splits_have_no_cross_split_question_leakage(recipe, rows):
    splits, manifest = rt.prepare_splits(rows, recipe)
    assert all(splits[name] for name in ("train", "validation", "test"))
    questions = {}
    for name, items in splits.items():
        for row in items:
            questions.setdefault(row["question_sha256"], name)
            assert rt.split_bucket(row["question_sha256"], 0) == name
    assert len(questions) == len(rows)
    contents = [row["content_sha256"] for items in splits.values() for row in items]
    assert len(set(contents)) == len(contents)
    assert manifest["counts"]["source"] == len(rows)
    assert manifest["counts"]["test_eval_used"] == len(splits["test"])
    rt.validate_source_manifest(manifest, recipe)


def test_same_question_variants_stay_in_one_split(recipe, rows):
    # 同题不同 response（模拟 Bitext 的语言生成变体）必须落同一个 split。
    variant = dict(rows[0], flags="Q", response=rows[0]["response"] + " Alternative phrasing.")
    combined = rows + [variant]
    splits, manifest = rt.prepare_splits(combined, recipe)
    owner = {row["question_sha256"]: name for name, items in splits.items() for row in items}
    assert len(owner) == len(rows)
    rt.validate_source_manifest(manifest, recipe)


def test_prepare_splits_rejects_duplicate_content(recipe, rows):
    with pytest.raises(TaskExecutionError, match="重复"):
        rt.prepare_splits(rows + [dict(rows[0])], recipe)


def test_prepare_splits_rejects_empty_split(recipe):
    single = make_rows(1)
    with pytest.raises(TaskExecutionError, match="为空"):
        rt.prepare_splits(single, recipe)


@pytest.mark.parametrize("change", [
    lambda m: m.update(sha256="0" * 64),
    lambda m: m.update(config_grid_sha256="0" * 64),
    lambda m: m["split_hashes"]["test"].append("f" * 64),
    lambda m: m["split_question_hashes"]["train"].append("e" * 64),
    lambda m: m["counts"].update(train=0),
    lambda m: m.update(data_seed=1),
])
def test_validate_source_manifest_detects_tampering(recipe, rows, change):
    _, manifest = rt.prepare_splits(rows, recipe)
    change(manifest)
    with pytest.raises(TaskExecutionError):
        rt.validate_source_manifest(manifest, recipe)


# ---------------------------------------------------------------- 编码与指标

def test_render_prompt_never_contains_gold(recipe, rows):
    row = rt.parse_row(rows[0], 0)
    prompt = rt.render_prompt(row)
    assert row["response"] not in prompt and prompt.endswith("Answer:")
    leaked = dict(rows[0], instruction=rows[0]["response"])
    with pytest.raises(TaskExecutionError, match="gold 泄漏"):
        rt.render_prompt(rt.parse_row(leaked, 0))


def test_encode_example_is_completion_only(recipe, rows):
    row = rt.parse_row(rows[0], 0)
    identity = rt.token_identity(Tokenizer())
    item = rt.encode_example(row, Tokenizer(), identity, recipe)
    assert item["labels"][:item["prompt_tokens"]] == [-100] * item["prompt_tokens"]
    assert item["labels"][item["prompt_tokens"]:] == item["input_ids"][item["prompt_tokens"]:]
    assert item["prompt_tokens"] + item["target_tokens"] == len(item["input_ids"])


def test_collate_right_pads_and_masks_labels(recipe):
    import types
    torch = types.SimpleNamespace(long="long", tensor=lambda values, dtype=None: values)
    batch = rt.collate([
        {"input_ids": [1, 2, 3], "labels": [-100, -100, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4], "labels": [4], "attention_mask": [1]},
    ], torch, 0)
    assert batch["input_ids"][1] == [4, 0, 0]
    assert batch["labels"][1] == [4, -100, -100]
    assert batch["attention_mask"][1] == [1, 0, 0]


def test_rouge_l_and_exact_match_are_deterministic_and_not_stemmed():
    assert rt.rouge_l("the cat sat", "the cat sat") == 1.0
    assert rt.rouge_l("", "") == 1.0
    assert rt.rouge_l("hello", "") == 0.0
    assert 0 < rt.rouge_l("the cat sat", "the cat stood") < 1
    # 无 stemmer：单复数/词形变化不会自动归并
    assert rt.rouge_l("running", "run") == 0.0
    assert rt.normalize_text("  Héllo   WORLD ") == rt.normalize_text("héllo world")


def test_score_prediction_eos_truncation_and_empty(recipe, rows):
    row = rt.parse_row(rows[0], 0)
    gold = " " + row["response"]
    perfect = rt.score_prediction(row, gold + "</s>trailing noise", "</s>")
    assert perfect["rouge_l"] == 1.0 and perfect["exact_match"] == 1.0
    empty = rt.score_prediction(row, "</s>", "</s>")
    assert empty["rouge_l"] == 0.0 and empty["exact_match"] == 0.0
    assert perfect["gold_sha256"] == empty["gold_sha256"]
    assert perfect["prediction_sha256"] != perfect["clean_prediction_sha256"]


def test_recompute_metrics_and_rejects_invalid_scores():
    predictions = [{"rouge_l": 0.5, "exact_match": 1.0},
                   {"rouge_l": 0.25, "exact_match": 0.0}]
    assert rt.recompute_metrics(predictions) == {"rouge_l": 0.375, "exact_match": 0.5}
    for bad in ([], [{"rouge_l": 2.0, "exact_match": 0.0}],
                [{"rouge_l": 0.5}], "not-a-list"):
        with pytest.raises(TaskExecutionError):
            rt.recompute_metrics(bad)


def test_evaluation_manifest_binds_raw_prompt_and_gold(recipe, rows):
    r = recipe
    splits, source = rt.prepare_splits(rows, r)
    identity = rt.token_identity(Tokenizer())
    samples = rt.evaluation_examples(splits["test"], Tokenizer(), identity, r)
    manifest = rt.evaluation_manifest(samples, identity, r, source)
    assert manifest["sample_count"] == len(splits["test"])
    assert manifest["sha256"] == rt._hash_manifest(manifest)["sha256"]
    for sample in manifest["samples"]:
        rebuilt = rt.parse_row(sample["raw"], sample["source_index"])
        assert rebuilt["response"] not in sample["prompt"]
        assert sample["gold_sha256"] == rt.sha256_json(rebuilt["response"])
        assert sample["prompt_ids"] == Tokenizer._ids(sample["prompt"])
    changed = copy.deepcopy(manifest)
    changed["samples"][0]["raw"]["response"] = "tampered"
    assert rt._hash_manifest(changed)["sha256"] != changed["sha256"]


# ---------------------------------------------------------------- 真实执行链路（合成夹具）

@pytest.fixture
def executed(tmp_path, monkeypatch, recipe, rows):
    root = tmp_path / "run"
    install_fake(monkeypatch, fake_runtime(recipe, rows))
    training = rt.run_training(recipe, _plan(recipe, root, "full"))
    install_fake(monkeypatch, fake_runtime(recipe, rows))
    base = rt.run_evaluation(recipe, _plan(recipe, root / "base", "zero-shot"), zero_shot=True)
    install_fake(monkeypatch, fake_runtime(recipe, rows))
    evaluation = rt.run_evaluation(
        recipe, _plan(recipe, root / "eval", "eval"),
        checkpoint=training.parent, zero_shot=False)
    return SimpleNamespace(r=recipe, root=root, training=training, base=base,
                           evaluation=evaluation, checkpoint=training.parent)


def test_run_training_writes_strict_completed_evidence(executed):
    evidence = json.loads(executed.training.read_text(encoding="utf-8"))
    assert evidence["stage"] == "completed" and evidence["kind"] == "training"
    assert evidence["task_id"] == rt.TASK_ID and evidence["config_id"] == BASE_CONFIG
    assert evidence["steps"] == evidence["max_steps"] == 64
    assert evidence["fingerprint"] == be.fingerprint(executed.r)
    assert evidence["config"]["quantization"] == "none"
    assert evidence["config"]["config"]["id"] == BASE_CONFIG
    assert evidence["config"]["lora_execution"]["tensor_reload_check"]["strict_values_verified"] is True
    assert evidence["runtime"]["compute_dtype"] == "float16"
    assert evidence["source_manifest"]["config_grid_sha256"] == rt.sha256_json(
        rt.config_grid(executed.r))
    assert "config_id" not in evidence["source_manifest"]
    assert rt.checkpoint_path({"recipe_id": executed.r.id, "mode": "eval"},
                              checkpoint=executed.checkpoint, recipe=executed.r) == executed.checkpoint


def test_training_rejects_step_shortfall(tmp_path, monkeypatch, recipe, rows):
    runtime = fake_runtime(recipe, rows)

    class ShortTrainer:
        def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
            self.state = SimpleNamespace(global_step=args.max_steps - 1)

        def train(self):
            return SimpleNamespace(metrics={"train_loss": 0.5})

    runtime["transformers"].Trainer = ShortTrainer
    install_fake(monkeypatch, runtime)
    with pytest.raises(TaskExecutionError, match="训练未完成预算"):
        rt.run_training(recipe, _plan(recipe, tmp_path / "short", "full"))


def test_run_evaluation_recomputes_metrics_and_binds_predictions(executed):
    for path, role in ((executed.base, "base"), (executed.evaluation, "checkpoint")):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        assert evidence["stage"] == "completed" and evidence["kind"] == "evaluation"
        assert evidence["model_role"] == role
        assert evidence["evaluation_protocol"] == rt.EVAL_PROTOCOL
        assert evidence["config"]["metric_comparable_to_upstream_rougeL"] is False
        predictions = json.loads((path.parent / rt.PREDICTIONS_FILE).read_text(encoding="utf-8"))
        assert evidence["metrics"] == rt.recompute_metrics(predictions["predictions"])
        assert evidence["sample_count"] == len(predictions["predictions"])
        assert evidence["predictions_sha256"] == rt.sha256_json(predictions["predictions"])
        reference = json.loads((path.parent / rt.EVAL_MANIFEST_FILE).read_text(encoding="utf-8"))
        assert evidence["evaluation_manifest"] == reference
        for record, sample in zip(predictions["predictions"], reference["samples"]):
            rebuilt = rt.parse_row(sample["raw"], sample["source_index"])
            expected = rt.score_prediction(rebuilt, record["generated_text"], "</s>")
            assert all(record[key] == expected[key] for key in
                       ("sample_id", "gold_sha256", "rouge_l", "exact_match"))


def test_checkpoint_path_rejects_tampered_artifact(executed, tmp_path):
    adapter = executed.checkpoint / "adapter_config.json"
    payload = json.loads(adapter.read_text(encoding="utf-8"))
    payload["r"] = 32
    adapter.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TaskExecutionError):
        rt.checkpoint_path({"recipe_id": executed.r.id, "mode": "eval"},
                           checkpoint=executed.checkpoint, recipe=executed.r)


def test_checkpoint_path_rejects_config_mismatch(executed, tmp_path):
    other = rt.config_recipe(executed.r, "lr2e-4-r32")
    with pytest.raises(TaskExecutionError):
        rt.checkpoint_path({"recipe_id": other.id, "mode": "eval"},
                           checkpoint=executed.checkpoint, recipe=other)


def test_evaluation_rejects_repeated_greedy_as_variance(tmp_path, monkeypatch, recipe, rows):
    install_fake(monkeypatch, fake_runtime(recipe, rows))
    plan = _plan(recipe, tmp_path / "eval", "zero-shot")
    plan["requested_runs"] = 3
    with pytest.raises(TaskExecutionError, match="重复"):
        rt.run_evaluation(recipe, plan, zero_shot=True)


# ---------------------------------------------------------------- 证据身份与发布级失败关闭

def test_fingerprint_binds_protocol_grid_and_config(tmp_path):
    base = _recipe(tmp_path)
    assert be.fingerprint(base) != be.fingerprint(rt.config_recipe(base, "lr2e-4-r32"))
    changed = copy.deepcopy(base.data)
    changed["rapidfire"]["loss"] = "full-text-loss"
    assert be.fingerprint(Recipe(base.id, base.path, changed)) != be.fingerprint(base)
    changed = copy.deepcopy(base.data)
    changed["configs"][0]["learning_rate"] = 9e-4
    assert be.fingerprint(Recipe(base.id, base.path, changed)) != be.fingerprint(base)


def test_frozen_environment_required_set_does_not_demand_trl(recipe):
    assert be.frozen_environment(recipe) is True
    data = copy.deepcopy(recipe.data)
    data["rapidfire"] = None
    # 协议段缺失时回退到数学默认集合，不会误把 RapidFire 当成已冻结
    assert be.frozen_environment(Recipe(recipe.id, recipe.path, data)) is False


def test_router_dispatches_all_rapidfire_pipelines(recipe):
    from recipe_lab import rapidfire_export as re
    from recipe_lab import rapidfire_suite as rs
    assert tr.resolve(recipe) == rt.TASK_ID
    assert tr.load(recipe, "task") is rt
    assert tr.load(recipe, "suite") is rs
    assert tr.load(recipe, "export") is re
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "RECIPE_LAB_ROOT": str(ROOT)}
    result = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "suite.py"), "--recipe", rt.RECIPE_ID,
         "--seeds", "42", "43", "44", "--output-root", "/tmp/recipe-lab-rapidfire-suite"],
        env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["kind"] == "suite-plan" and plan["task_id"] == rt.TASK_ID
    assert plan["selection"]["split"] == "validation"
    assert plan["acceptance"]["split"] == "test"
    assert len(plan["runs"]) == 12 and "deferred" in plan["acceptance"]["commands"]


def test_baseline_evidence_dispatches_to_rapidfire_and_fails_closed(recipe, executed):
    assert be.evaluation_problems(
        recipe, json.loads(executed.evaluation.read_text(encoding="utf-8")),
        executed.evaluation, role="checkpoint") == []
    assert be.zero_train_measured(recipe) is False
    assert any("baseline-suite" in problem for problem in be.suite_problems(recipe))


def test_cli_dry_runs_are_framework_free_and_config_aware():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "RECIPE_LAB_ROOT": str(ROOT)}
    commands = [
        [sys.executable, str(ROOT / "runner" / "train.py"), "--recipe", rt.RECIPE_ID, "--dry-run"],
        [sys.executable, str(ROOT / "runner" / "eval.py"), "--recipe", rt.RECIPE_ID],
        [sys.executable, str(ROOT / "runner" / "zero_shot.py"), "--recipe", rt.RECIPE_ID],
    ]
    for command in commands:
        result = subprocess.run(command + ["--json"], env=env,
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["execution_supported"] is True
        assert plan["task_id"] == rt.TASK_ID
    bad = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "train.py"), "--recipe", rt.RECIPE_ID,
         "--config-id", "invented", "--dry-run", "--json"],
        env=env, capture_output=True, text=True, check=False)
    assert bad.returncode != 0 and "未知配置" in (bad.stdout + bad.stderr)
