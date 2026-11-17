"""纯函数和隔离 fake runtime 测试；JSON 只写 tmp_path，不联网、不真实训练。"""
from __future__ import annotations

import contextlib
import copy
import importlib.abc
import inspect
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe_lab import fim_task as ft
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recipe(tmp_path):
    data = {
        "id": ft.RECIPE_ID, "source_sha": "c" * 40,
        "method": {"family": "sft", "trainer": "Trainer", "upstream_trainer": "Trainer", "peft": "qlora"},
        "model": {"base": ft.MODEL, "revision": "a" * 40, "modality": "llm"},
        "dataset": {"name": ft.DATASET, "revision": "b" * 40},
        "fim": copy.deepcopy(ft.FIM_SPEC), "peft": copy.deepcopy(ft.PEFT_SPEC),
        "quantization": copy.deepcopy(ft.QUANT_SPEC),
        "train": {
            "seed": 0, "data_seed": 0, "seed_mechanism": "上游 set_seed + shuffle + ConstantLengthDataset",
            "learning_rate": 5e-4, "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1, "max_steps": 2000, "seq_length": 32,
            "lr_scheduler_type": "cosine", "weight_decay": 0.01, "warmup_steps": 30,
            "eval_steps": 100, "bf16": True, "fim_rate": 0.5, "fim_spm_rate": 0.5,
            "lora_r": 8, "lora_alpha": 32, "lora_dropout": 0.0, "held_out_examples": 4000,
            "per_device_eval_batch_size": 16, "save_steps": 100, "logging_steps": 25,
            "gradient_checkpointing": True, "fp16": False, "optim": "adamw_torch",
            "dataloader_num_workers": 0, "remove_unused_columns": False,
        },
        "budget": {"full_steps": 2000, "smoke_steps": 5},
        "evaluation": {"metrics": ["fim_exact_match", "fim_token_exact_match"],
                       "seed": 0, "max_new_tokens": 8, "multi_run": True},
        "execution": {"task_id": ft.TASK_ID, "lockfile": "test.lock.toml"},
        "baseline": {"result": "unmeasured"},
    }
    # SHA、版本和 CUDA 仅为测试合成输入，不声明任何真实环境或上游 commit。
    lock = '\n'.join([
        'schema_version = 1', 'status = "frozen"', f'recipe_id = "{ft.RECIPE_ID}"',
        f'model_id = "{ft.MODEL}"', f'dataset_id = "{ft.DATASET}"',
        f'model_revision = "{"a" * 40}"', f'dataset_revision = "{"b" * 40}"',
        f'source_sha = "{"c" * 40}"',
        'requirements = ["torch==test", "transformers==test", "datasets==test", "peft==test", "bitsandbytes==test"]',
        '[runtime]', 'python = "test"', 'cuda = "test"', 'driver = "test"', 'gpu = "test"',
    ])
    (tmp_path / "test.lock.toml").write_text(lock, encoding="utf-8")
    return Recipe(ft.RECIPE_ID, tmp_path / "recipe.toml", data)


def plan_for(recipe, tmp_path, mode):
    plan = ft.task_plan(recipe, build_plan(recipe, mode), mode)
    assert plan["execution_supported"], plan.get("unsupported_reason")
    plan["run"]["output_dir"] = str(tmp_path / mode)
    return plan


@pytest.fixture
def rows():
    owners = {name: [] for name in ("train", "validation", "test")}
    for i in range(1000):
        repo = f"owner/repo-{i}"
        name = ft.repo_split(repo, 0)
        if len(owners[name]) < 3:
            owners[name].append(repo)
        if all(len(v) == 3 for v in owners.values()):
            break
    return [
        {"repo_id": repo, "file_path": f"src/{name}-{i}.py", "content": f"  unique_{name}_{i} = {i}\n  body_{name}_{i}\n", "meta": i}
        for name, repos in owners.items() for i, repo in enumerate(repos)
    ]


class Tokenizer:
    eos_token_id = 0
    unk_token_id = 4
    eos_token = "<eos>"
    all_special_ids = [0, 1, 2, 3, 4]
    backend_tokenizer = None

    def __init__(self):
        self.vocab = {"<eos>": 0, **dict(zip(ft.SPECIAL_TOKENS, [1, 2, 3])), "<unk>": 4,
                      **{chr(i): i + 10 for i in range(9, 127)}}
        self.pad_token = None
        self.padding_side = "left"

    @property
    def pad_token_id(self):
        return self.vocab.get(self.pad_token)

    @property
    def special_tokens_map(self):
        return {"eos_token": self.eos_token, "unk_token": "<unk>", "pad_token": self.pad_token,
                "additional_special_tokens": list(ft.SPECIAL_TOKENS)}

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False and truncation is False
        if text in self.vocab and text in [*ft.SPECIAL_TOKENS, "<eos>", "<unk>"]:
            return [self.vocab[text]]
        return [self.vocab.get(c, self.unk_token_id) for c in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        inverse = {v: k for k, v in self.vocab.items()}
        return "".join(inverse[i] for i in ids)

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer_config.json").write_text(json.dumps(self.special_tokens_map), encoding="utf-8")
        (directory / "tokenizer.json").write_text(json.dumps(self.vocab), encoding="utf-8")


def test_schema_source_identity_and_order(recipe, rows):
    splits, manifest = ft.prepare_splits(iter(rows), recipe)
    assert ft.prepare_splits(rows, recipe) == (splits, manifest)
    assert manifest["source_count"] == 9
    assert manifest["sha256"] == ft.sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    for name in splits:
        assert len(splits[name]) == 3
        assert [x["source_index"] for x in splits[name]] == sorted(x["source_index"] for x in splits[name])
        assert all(ft.repo_split(x["repo_id"], 0) == name for x in splits[name])
        assert all(x["content_sha256"] == ft._text_hash(x["content"]) for x in splits[name])
    rows[-1]["meta"] = "改变上游元信息"
    assert ft.prepare_splits(rows, recipe)[1]["sha256"] != manifest["sha256"]


@pytest.mark.parametrize("field,value", [
    ("repo_id", None), ("repo_id", ""), ("file_path", None), ("content", None),
    ("content", " \n "), ("content", "x<fim_middle>y"), ("content", "x\x00y"),
    ("repo_id", "x\ny"), ("file_path", "../escape.py"), ("file_path", "/root.py"),
    ("file_path", "C:\\bad.py"), ("file_path", "a//b.py"), ("file_path", "a/./b.py"),
])
def test_all_schema_and_meta_checked_before_transform(recipe, rows, field, value):
    rows[-1][field] = value
    with pytest.raises(ft.DatasetValidationError):
        ft.prepare_splits(rows, recipe)


def test_non_json_metadata_rejected(recipe, rows):
    rows[-1]["meta"] = object()
    with pytest.raises(ft.DatasetValidationError, match="meta"):
        ft.prepare_splits(rows, recipe)


def test_duplicate_repo_file_even_changed_content_rejected(recipe, rows):
    duplicate = {**rows[0], "content": "different source\n"}
    with pytest.raises(ft.DatasetValidationError, match="file_id"):
        ft.prepare_splits(rows + [duplicate], recipe)


@pytest.mark.parametrize("same_split,normalized", [(False, False), (False, True), (True, False), (True, True)])
def test_raw_and_normalized_leakage_global_policy(recipe, rows, same_split, normalized):
    target = 1 if same_split else -1
    rows[0]["content"] = "  a=1\r\n    b=2  \r\n"
    rows[target]["content"] = "  a=1\n    b=2\n" if normalized else rows[0]["content"]
    with pytest.raises(ft.DatasetValidationError, match="泄漏"):
        ft.prepare_splits(rows, recipe)
    assert ft.normalize_source("  a  \r\n\tb\t\r") == "  a\n\tb"
    assert ft.normalize_source("  a\n\n  \t") == ft.normalize_source("  a")
    assert ft.normalize_source(" a") != ft.normalize_source("a")


def test_empty_splits_fail_without_resampling(recipe, rows):
    with pytest.raises(ft.DatasetValidationError, match="empty split"):
        ft.prepare_splits(rows[:3], recipe)


def test_finite_max_plus_one_scan(recipe):
    consumed = []
    def source():
        for index in itertools_count():
            consumed.append(index)
            yield {"repo_id": f"r{index}", "file_path": "x.py", "content": f"x{index}"}
    with pytest.raises(ft.DatasetValidationError, match="max_source_examples=6000"):
        ft.prepare_splits(source(), recipe)
    assert len(consumed) == 6001


def itertools_count():
    index = 0
    while True:
        yield index
        index += 1


def test_data_and_transform_seed_independent_from_training_seed(recipe, rows):
    splits, manifest = ft.prepare_splits(rows, recipe)
    original = ft.train_examples(splits, Tokenizer(), recipe)
    evaluation = ft.evaluation_examples(splits, Tokenizer(), recipe)
    recipe.data["train"]["seed"] = 871
    assert ft.prepare_splits(rows, recipe) == (splits, manifest)
    assert ft.train_examples(splits, Tokenizer(), recipe) == original
    assert ft.evaluation_examples(splits, Tokenizer(), recipe) == evaluation
    recipe.data["evaluation"]["seed"] = 100
    assert ft.evaluation_examples(splits, Tokenizer(), recipe) == evaluation


@pytest.mark.parametrize("variant", ["psm", "spm"])
def test_exact_fim_concatenation_and_nonempty_middle(variant):
    identity = ft.token_identity(Tokenizer())
    both_edges = False
    for seed in range(100):
        sequence, info = ft.fim_sequence([21, 22, 23, 24, 25], identity, random.Random(seed), variant)
        prefix, middle, suffix = info["prefix_ids"], info["middle_ids"], info["suffix_ids"]
        assert middle and prefix + middle + suffix == [21, 22, 23, 24, 25]
        expected = ([1] + prefix + [2] + suffix + [3] + middle if variant == "psm"
                    else [1, 2] + suffix + [3] + prefix + middle)
        assert sequence == expected
        both_edges |= not prefix and not suffix
    assert both_edges


@pytest.mark.parametrize("change", ["missing", "alias", "unknown", "multi-token", "eos-alias"])
def test_special_tokens_cannot_fallback_or_alias(change):
    tokenizer = Tokenizer()
    if change == "missing":
        del tokenizer.vocab[ft.SPECIAL_TOKENS[0]]
    elif change == "alias":
        tokenizer.vocab[ft.SPECIAL_TOKENS[0]] = 2
    elif change == "unknown":
        tokenizer.vocab[ft.SPECIAL_TOKENS[0]] = 4
    elif change == "eos-alias":
        tokenizer.vocab[ft.SPECIAL_TOKENS[0]] = 0
    else:
        tokenizer.encode = lambda *args, **kwargs: [1, 2]
    with pytest.raises(ft.TaskExecutionError):
        ft.token_identity(tokenizer)


def test_eos_may_share_unknown_id_but_not_fim_id(recipe):
    tokenizer = Tokenizer()
    tokenizer.unk_token_id = tokenizer.eos_token_id
    identity = ft.token_identity(tokenizer)
    assert identity["eos_token_id"] == identity["unk_token_id"] == 0
    assert set(identity["special_token_ids"].values()) == {1, 2, 3}
    recipe.data["peft"]["dropout"] = 0
    assert ft.require_supported(recipe) == ft.TASK_ID
    recipe.data["peft"]["dropout"] = False
    with pytest.raises(ft.TaskExecutionError):
        ft.require_supported(recipe)


def test_special_id_in_source_rejected():
    tokenizer = Tokenizer()
    identity = ft.token_identity(tokenizer)
    tokenizer.encode = lambda *args, **kwargs: [20, 1, 21]
    with pytest.raises(ft.DatasetValidationError, match="特殊 ID"):
        ft.encode_source(tokenizer, "abc", identity)


def test_long_chunks_tail_and_eos_labels(recipe, rows):
    # seq_length-4=28，两个完整 chunk + 两 token 尾巴；不跨文件。
    rows[0]["content"] = "A" * 58
    splits, _ = ft.prepare_splits(rows, recipe)
    examples, transform = ft.train_examples(splits, Tokenizer(), recipe)
    item = transform["splits"]["train"]["samples"][0]
    assert [x["token_start"] for x in item["chunks"]] == [0, 28]
    assert [x["token_stop"] for x in item["chunks"]] == [28, 56]
    assert item["tail_dropped"]["count"] == 2
    assert len(item["chunks"]) == 2
    for split in examples.values():
        for encoded in split:
            assert len(encoded["input_ids"]) == len(encoded["labels"]) == len(encoded["attention_mask"]) == 32
            length = sum(encoded["attention_mask"])
            assert encoded["input_ids"][length - 1] == encoded["labels"][length - 1] == 0
            assert encoded["labels"][:length] == encoded["input_ids"][:length]
            assert encoded["labels"][length:] == [-100] * (32 - length)
            assert encoded["input_ids"][length:] == [0] * (32 - length)
    assert {x["variant"] for file in transform["splits"]["validation"]["samples"] for x in file["chunks"]} == {"psm"}


def test_zero_rate_is_causal_all_tokens_not_assistant_mask(recipe, rows):
    recipe.data["train"]["fim_rate"] = 0.0
    splits, _ = ft.prepare_splits(rows, recipe)
    examples, transform = ft.train_examples(splits, Tokenizer(), recipe)
    assert all(chunk["variant"] == "causal" for row in transform["splits"]["train"]["samples"] for chunk in row["chunks"])
    first = examples["train"][0]
    length = sum(first["attention_mask"])
    assert first["labels"][:length] == first["input_ids"][:length]
    assert -100 not in first["labels"][:length]


def test_held_out_exact_offsets_no_gold_input_and_no_gold_truncation(recipe, rows):
    rows[-1]["content"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 20
    splits, _ = ft.prepare_splits(rows, recipe)
    tokenizer = Tokenizer()
    examples, manifest = ft.evaluation_examples(splits, tokenizer, recipe)
    assert len(examples) == len(splits["test"])
    for row, item in zip(splits["test"], examples):
        tokens = tokenizer.encode(row["content"])
        window = tokens[item["window_start"]:item["window_stop"]]
        assert item["gold_ids"] == window[item["gap_start"]:item["gap_stop"]]
        assert item["prefix_ids"] + item["gold_ids"] + item["suffix_ids"] == window
        assert item["input_ids"] == [1] + item["prefix_ids"] + [2] + item["suffix_ids"] + [3]
        assert 0 < len(item["gold_ids"]) <= 8
        assert len(item["input_ids"]) + 8 <= 32
        assert item["gold_sha256"] == ft.sha256_json(item["gold_ids"])
        assert item["gold_text"] == tokenizer.decode(item["gold_ids"])
    assert manifest["transformed_test_sha256"] == ft.sha256_json(examples)


def test_strict_text_whitespace_eos_and_token_exact_scoring():
    tokenizer = Tokenizer()
    gold = tokenizer.encode(" a ")
    example = {"gold_ids": gold, "gold_sha256": ft.sha256_json(gold)}
    score = ft.score_generation(example, gold + [0], tokenizer, 8)
    assert score["fim_exact_match"] == score["fim_token_exact_match"] == 1
    assert score["completion"] == " a "
    assert ft.score_generation(example, tokenizer.encode("a") + [0], tokenizer, 8)["fim_exact_match"] == 0
    assert ft.score_generation(example, gold, tokenizer, 8)["fim_token_exact_match"] == 0
    assert ft.score_generation(example, gold + [0] * 6, tokenizer, 8)["fim_token_exact_match"] == 0
    scored = {**example, **score}
    assert ft.recompute_metrics([scored]) == {"fim_exact_match": 1.0, "fim_token_exact_match": 1.0}
    scored["fim_exact_match"] = 0
    with pytest.raises(ft.TaskExecutionError, match="重算"):
        ft.recompute_metrics([scored])


@pytest.mark.parametrize("section", ["train", "fim", "peft", "quantization", "evaluation", "dependencies", "method", "model", "dataset", "budget"])
def test_unknown_parameters_fail_closed(recipe, section):
    recipe.data.setdefault(section, {})["unconsumed"] = True
    with pytest.raises(ft.TaskExecutionError, match="未消费"):
        ft.require_supported(recipe)


@pytest.mark.parametrize("key,value", [
    ("method.trainer", "SFTTrainer"), ("method.family", "grpo"), ("model.base", "other"),
    ("dataset.name", "other"), ("method.peft", "lora"), ("fim.protocol", "other"),
    ("train.seed", True), ("evaluation.seed", True), ("budget.full_steps", 150),
    ("train.max_steps", 150), ("train.fp16", True), ("quantization.double_quant", False),
    ("evaluation.metrics", ["pass@1"]), ("train.held_out_examples", 5),
])
def test_route_seed_budget_protocol_validation(recipe, key, value):
    section, field = key.split(".")
    recipe.data[section][field] = value
    with pytest.raises(ft.TaskExecutionError):
        ft.require_supported(recipe)


def test_task_route_double_validation(recipe):
    assert ft.require_supported(recipe) == ft.TASK_ID
    with pytest.raises(ft.TaskExecutionError, match="冲突"):
        ft.require_supported(recipe, {"task_id": "math-grpo-v1"}, task_id=ft.TASK_ID)
    with pytest.raises(ft.TaskExecutionError, match="路由"):
        ft.require_supported(recipe, {"recipe_id": "other"})
    recipe.data["execution"]["task_id"] = "other"
    with pytest.raises(ft.TaskExecutionError, match="冲突"):
        ft.require_supported(recipe, task_id=ft.TASK_ID)


@pytest.mark.parametrize("mode,steps", [("full", 2000), ("smoke", 5), ("dry-run", 5)])
def test_plan_effective_budget_and_training_arguments(recipe, tmp_path, mode, steps):
    plan = plan_for(recipe, tmp_path, mode)
    assert plan["run"]["steps"] == plan["run"]["params"]["max_steps"] == steps
    assert plan["evaluation"]["protocol"] == ft.EVAL_PROTOCOL
    assert plan["run"]["params"]["push_to_hub"] is False
    assert plan["run"]["params"]["report_to"] == "none"
    assert not ft.METADATA_TRAIN_KEYS & plan["run"]["params"].keys()
    from recipe_lab.provenance import output_plan
    isolated = output_plan(plan, ROOT, str(tmp_path / "isolated"))
    assert isolated["run"]["params"]["output_dir"] == isolated["run"]["output_dir"]
    if mode != "dry-run":
        plan["run"]["steps"] = 150
        with pytest.raises(ft.TaskExecutionError, match="预算"):
            ft.config_kwargs(recipe, plan, tmp_path / mode)


def explicit_factory(keys):
    # Fake API 使用明确 Signature；生产代码不依赖 **kwargs 支持。
    def factory(**kwargs):
        return SimpleNamespace(**kwargs)
    factory.__signature__ = inspect.Signature([
        inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY) for key in keys
    ])
    return factory


def test_actual_training_arguments_api_selection_and_no_silent_drop(recipe, tmp_path):
    config = ft.config_kwargs(recipe, plan_for(recipe, tmp_path, "full"), tmp_path / "full")
    old_api = explicit_factory(config)
    assert ft.training_arguments_kwargs(old_api, config) == config
    new_keys = ["eval_strategy" if key == "evaluation_strategy" else key for key in config]
    new_api = explicit_factory(new_keys)
    adapted = ft.training_arguments_kwargs(new_api, config)
    assert adapted["eval_strategy"] == "steps" and "evaluation_strategy" not in adapted
    with pytest.raises(ft.TaskExecutionError, match="不显式支持"):
        ft.training_arguments_kwargs(lambda **kwargs: None, config)
    with pytest.raises(ft.TaskExecutionError, match="save_steps"):
        ft.training_arguments_kwargs(explicit_factory([k for k in config if k != "save_steps"]), config)


def test_no_cuda_before_any_other_dependency(monkeypatch):
    names = []
    def fake_import(name):
        names.append(name)
        assert name == "torch"
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(ft.importlib, "import_module", fake_import)
    with pytest.raises(ft.TaskExecutionError, match="没有可用 CUDA"):
        ft.load_runtime(training=True)
    assert names == ["torch"]


@pytest.mark.parametrize("count,bf16", [(2, True), (1, False)])
def test_runtime_single_gpu_bf16_requirements(monkeypatch, count, bf16):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: count,
                                                 is_bf16_supported=lambda: bf16), version=SimpleNamespace(cuda="test"))
    monkeypatch.setattr(ft.importlib, "import_module", lambda name: torch)
    with pytest.raises(ft.TaskExecutionError):
        ft.load_runtime()


def test_runtime_imports_only_five_packages_and_missing_dependency(monkeypatch):
    names = []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                 is_bf16_supported=lambda: True), version=SimpleNamespace(cuda="test"))
    def fake_import(name):
        names.append(name)
        return torch if name == "torch" else SimpleNamespace()
    monkeypatch.setattr(ft.importlib, "import_module", fake_import)
    assert set(ft.load_runtime(training=True)) == set(ft.RUNTIME_PACKAGES)
    assert names == list(ft.RUNTIME_PACKAGES)
    monkeypatch.setattr(ft.importlib, "import_module", lambda _: (_ for _ in ()).throw(ImportError("test")))
    with pytest.raises(ft.TaskExecutionError, match="缺少 torch"):
        ft.load_runtime()


def test_module_and_read_only_dryrun_without_third_party(recipe, tmp_path):
    # 子进程 import finder 拒绝全部训练框架；临时输入仅为测试数据。
    payload = tmp_path / "recipe-input.json"
    payload.write_text(json.dumps(recipe.data), encoding="utf-8")
    program = '''import importlib.abc, json, pathlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','datasets','peft','bitsandbytes','trl','math_verify'}:
            raise AssertionError('禁止导入第三方框架')
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan
from recipe_lab.fim_task import task_plan
r = Recipe('llm/sft/starcoder1b-fim', pathlib.Path(sys.argv[2]), json.loads(pathlib.Path(sys.argv[2]).read_text()))
p = task_plan(r, build_plan(r, 'dry-run'), 'dry-run')
assert p['execution_supported'] and p['run']['steps'] == 5
'''
    before = sorted(p.name for p in tmp_path.iterdir())
    result = subprocess.run([sys.executable, "-B", "-c", program, str(ROOT / "src"), str(payload)],
                            capture_output=True, text=True, check=False,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == before


class Tensor:
    dtype = "test-float"

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

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def tolist(self):
        return copy.deepcopy(self.values)


class Parameter:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad

    def numel(self):
        return 10

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


def fake_runtime(recipe, rows, *, incomplete=False, new_api=False):
    calls = {"models": [], "tokenizers": [], "load_dataset": [], "dataset_lists": [], "generations": []}
    tokenizer = Tokenizer()
    # 模拟生成器由测试预先计算答案，生产 generate 调用只得到 PSM prompt 和 mask。
    splits, _ = ft.prepare_splits(rows, recipe)
    held_out, _ = ft.evaluation_examples(splits, tokenizer, recipe)
    answers = {tuple(e["input_ids"]): e["gold_ids"] for e in held_out}
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    base = Parameter()
    parameters = {"base.weight": base}
    model.named_modules = lambda: [(f"layer.{name}", object()) for name in ft.TARGET_MODULES]
    model.named_parameters = lambda: list(parameters.items())
    model.parameters = lambda: list(parameters.values())
    model.eval = lambda: None
    def generate(input_ids, attention_mask, **kwargs):
        prompt = input_ids.tolist()[0]
        assert attention_mask.tolist() == [[1] * len(prompt)]
        assert set(kwargs) == {"do_sample", "num_beams", "max_new_tokens", "eos_token_id", "pad_token_id", "use_cache"}
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        calls["generations"].append({"input_ids": prompt, "kwargs": kwargs})
        gold = answers[tuple(prompt)]
        completion = (gold + [0])[:kwargs["max_new_tokens"]]
        return Tensor([prompt + completion])
    model.generate = generate
    def save_model(directory, safe_serialization):
        assert safe_serialization is True
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        adapter = {"base_model_name_or_path": ft.MODEL, "revision": recipe.get("model.revision"),
                   "r": 8, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ft.TARGET_MODULES,
                   "bias": "none", "task_type": "CAUSAL_LM", "peft_type": "LORA"}
        (directory / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
        (directory / "adapter_model.safetensors").write_bytes(b"test-only-adapter")
    model.save_pretrained = save_model
    def load_model(source, **kwargs):
        calls["models"].append((source, kwargs))
        return model
    def load_tokenizer(source, **kwargs):
        calls["tokenizers"].append((source, kwargs))
        return Tokenizer()
    def load_dataset(source, **kwargs):
        calls["load_dataset"].append((source, kwargs))
        return iter(rows)
    def dataset_from_list(items):
        calls["dataset_lists"].append(copy.deepcopy(items))
        return items
    def prepare_model_for_kbit_training(model, use_gradient_checkpointing):
        assert use_gradient_checkpointing is True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        calls["prepare_kbit"] = True
        return model
    def get_peft_model(model, peft_config):
        parameters["adapter.lora_A.weight"] = Parameter(True)
        calls["lora_config"] = peft_config
        return model
    def load_adapter(base_model, source, is_trainable, local_files_only):
        assert is_trainable is False and local_files_only is True
        calls["adapter_source"] = source
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        return base_model
    canonical = ft.config_kwargs(recipe, {"mode": "full"}, Path("unused"))
    keys = ["eval_strategy" if new_api and k == "evaluation_strategy" else k for k in canonical]
    class Trainer:
        def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
            calls["trainer"] = {"args": args, "train": train_dataset, "validation": eval_dataset,
                                "data_collator": data_collator, "model": model}
            self.state = SimpleNamespace(global_step=args.max_steps - int(incomplete))
        def train(self):
            calls["train_called"] = True
            return SimpleNamespace(metrics={"train_loss": 0.25})
    runtime = {
        "torch": SimpleNamespace(
            __version__="test", bfloat16="test-bfloat16", long="test-long", tensor=lambda values, **kwargs: Tensor(values),
            isfinite=lambda tensor: Tensor([True] * len(tensor.values)),
            equal=lambda a, b: a.values == b.values,
            inference_mode=contextlib.nullcontext, version=SimpleNamespace(cuda="test"),
            cuda=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda _: "test-device",
                                 max_memory_allocated=lambda: 0),
            are_deterministic_algorithms_enabled=lambda: True,
        ),
        "transformers": SimpleNamespace(
            __version__="test", TrainingArguments=explicit_factory(keys), Trainer=Trainer,
            BitsAndBytesConfig=explicit_factory(["load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant"]),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        ),
        "datasets": SimpleNamespace(__version__="test", load_dataset=load_dataset,
                                    Dataset=SimpleNamespace(from_list=dataset_from_list)),
        "peft": SimpleNamespace(
            __version__="test", prepare_model_for_kbit_training=prepare_model_for_kbit_training,
            get_peft_model=get_peft_model,
            LoraConfig=explicit_factory(["r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type", "base_model_name_or_path", "revision"]),
            get_peft_model_state_dict=lambda model, adapter_name, save_embedding_layers: {
                "base_model.model.layer.c_attn.lora_A.weight": Tensor([1.0, 2.0]),
                "base_model.model.layer.c_attn.lora_B.weight": Tensor([3.0, 4.0]),
            },
            PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        ),
        "bitsandbytes": SimpleNamespace(__version__="test"),
    }
    return runtime, calls


def install_fake(monkeypatch, runtime):
    monkeypatch.setattr(ft, "load_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(ft, "require_frozen_runtime", lambda *args: None)
    monkeypatch.setattr(ft, "apply_determinism", lambda *args, **kwargs: None)
    monkeypatch.setattr(ft, "_saved_adapter_state", lambda path:
                        runtime["peft"].get_peft_model_state_dict(None, "default", False))


@pytest.mark.parametrize("mode,new_api", [("full", False), ("smoke", True)])
def test_trainer_assembly_adapter_evidence_and_real_tmp_json(recipe, rows, tmp_path, monkeypatch, mode, new_api):
    runtime, calls = fake_runtime(recipe, rows, new_api=new_api)
    install_fake(monkeypatch, runtime)
    plan = plan_for(recipe, tmp_path, mode)
    path = ft.run_training(recipe, plan)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / mode / "training.json"
    assert evidence["stage"] == "completed" and evidence["kind"] == "training"
    assert evidence["steps"] == (2000 if mode == "full" else 5)
    assert evidence["config_sha256"] == ft.sha256_json(evidence["config"])
    assert evidence["model_artifact"] == ft.artifact_manifest(path.parent)
    assert not {"mock", "simulated", "math_verify"} & evidence.keys()
    assert "math_verify" not in evidence["runtime"]["packages"]
    assert set(evidence["runtime"]["packages"]) == set(ft.RUNTIME_PACKAGES)
    assert calls["prepare_kbit"] and calls["lora_config"].r == 8
    assert len(calls["dataset_lists"]) == 2
    assert calls["dataset_lists"][0] == calls["trainer"]["train"]
    assert calls["dataset_lists"][1] == calls["trainer"]["validation"]
    assert all(set(item) == {"input_ids", "labels", "attention_mask"} for items in calls["dataset_lists"] for item in items)
    encoded = calls["trainer"]["data_collator"](calls["trainer"]["train"][:2])
    assert set(encoded) == {"input_ids", "labels", "attention_mask"}
    assert encoded["labels"].tolist() == [x["labels"] for x in calls["trainer"]["train"][:2]]
    loader = calls["models"][0][1]
    assert loader["trust_remote_code"] is False and loader["attn_implementation"] == "eager"
    assert loader["device_map"] == {"": 0} and loader["torch_dtype"] == "test-bfloat16"
    assert calls["load_dataset"][0][1] == {"revision": "b" * 40, "data_dir": "data", "split": "train", "streaming": True, "trust_remote_code": False}
    assert ft.checkpoint_path(plan, path.parent, recipe) == path.parent.resolve()
    assert ft.checkpoint_path(plan, path.parent) == path.parent.resolve()
    if mode == "full":
        assert ft.checkpoint_path(plan, recipe=recipe) == path.parent.resolve()
    else:
        with pytest.raises(ft.TaskExecutionError, match="没有训练"):
            ft.checkpoint_path(plan, recipe=recipe)


def test_collate_rejects_inconsistent_mask_and_implicit_padding():
    torch = SimpleNamespace(tensor=lambda *args, **kwargs: None, long=0)
    with pytest.raises(ft.DatasetValidationError):
        ft.collate([{"input_ids": [0], "labels": [0], "attention_mask": [0]}], torch)
    with pytest.raises(ft.DatasetValidationError):
        ft.collate([{"input_ids": [1], "labels": [1, 2], "attention_mask": [1]}], torch)


@pytest.mark.parametrize("failure", ["budget", "api", "target", "base-freeze", "empty-lora", "incomplete"])
def test_training_failures_never_write_completed(recipe, rows, tmp_path, monkeypatch, failure):
    runtime, calls = fake_runtime(recipe, rows, incomplete=failure == "incomplete")
    install_fake(monkeypatch, runtime)
    plan = plan_for(recipe, tmp_path, "smoke")
    if failure == "budget":
        plan["run"]["steps"] = 150
    elif failure == "api":
        runtime["transformers"].TrainingArguments = lambda **kwargs: None
    elif failure == "target":
        runtime["transformers"].AutoModelForCausalLM.from_pretrained(ft.MODEL).named_modules = lambda: []
    elif failure == "base-freeze":
        def prepare(model, use_gradient_checkpointing):
            return model
        runtime["peft"].prepare_model_for_kbit_training = prepare
    elif failure == "empty-lora":
        def attach(model, peft_config):
            return model
        runtime["peft"].get_peft_model = attach
    with pytest.raises(ft.TaskExecutionError):
        ft.run_training(recipe, plan)
    assert not (tmp_path / "smoke" / "training.json").exists()
    if failure in ("budget", "api"):
        assert calls["models"] == [] and calls["tokenizers"] == []


@pytest.mark.parametrize("field", ["decision", "source", "nonempty", "unfrozen"])
def test_preflight_rejects_before_runtime(recipe, tmp_path, monkeypatch, field):
    plan = plan_for(recipe, tmp_path, "full")
    if field == "decision":
        recipe.data["decision"] = {"gate": "blocked"}
    elif field == "source":
        recipe.data["source_sha"] = "main"
    elif field == "nonempty":
        (tmp_path / "full").mkdir()
        (tmp_path / "full" / "keep.txt").write_text("用户已有文件", encoding="utf-8")
    else:
        (tmp_path / "test.lock.toml").write_text('status="template"', encoding="utf-8")
    monkeypatch.setattr(ft, "load_runtime", lambda **kwargs: pytest.fail("门禁之前不能导入框架"))
    with pytest.raises(ft.TaskExecutionError):
        ft.run_training(recipe, plan)


@pytest.fixture
def completed(recipe, rows, tmp_path, monkeypatch):
    runtime, calls = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    plan = plan_for(recipe, tmp_path, "full")
    path = ft.run_training(recipe, plan)
    return path.parent, runtime, calls


@pytest.mark.parametrize("tamper", ["weights", "adapter-base", "steps", "config", "source", "tokenizer", "transform", "mock", "lock", "symlink"])
def test_checkpoint_tamper_rejected_no_fallback(recipe, completed, tmp_path, tamper):
    directory, _, _ = completed
    if tamper == "weights":
        (directory / "adapter_model.safetensors").write_bytes(b"changed")
    elif tamper == "adapter-base":
        path = directory / "adapter_config.json"
        data = json.loads(path.read_text())
        data["base_model_name_or_path"] = "other/base"
        path.write_text(json.dumps(data), encoding="utf-8")
    elif tamper == "lock":
        with (tmp_path / "test.lock.toml").open("a", encoding="utf-8") as handle:
            handle.write('\n# 改变锁字节\n')
    elif tamper == "symlink":
        (directory / "linked.json").symlink_to(directory / "source_manifest.json")
    else:
        filename = {"config": "effective_config.json", "source": "source_manifest.json", "tokenizer": "tokenizer_identity.json",
                    "transform": "fim_transform_manifest.json"}.get(tamper, "training.json")
        path = directory / filename
        data = json.loads(path.read_text())
        data["steps" if tamper == "steps" else "mock" if tamper == "mock" else "tampered"] = True
        path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ft.TaskExecutionError):
        ft.checkpoint_path(plan_for(recipe, tmp_path, "eval"), directory, recipe)


def test_missing_checkpoint_before_model_or_runtime(recipe, tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "load_runtime", lambda **kwargs: pytest.fail("不应加载 base"))
    with pytest.raises(ft.TaskExecutionError, match="没有训练 checkpoint"):
        ft.run_evaluation(recipe, plan_for(recipe, tmp_path, "eval"))


def test_base_and_adapter_eval_same_protocol_and_no_gold_to_generate(recipe, rows, completed, tmp_path, monkeypatch):
    directory, _, _ = completed
    results, manifests = [], []
    for mode, zero_shot in (("zero-shot", True), ("eval", False)):
        runtime, calls = fake_runtime(recipe, rows)
        install_fake(monkeypatch, runtime)
        path = ft.run_evaluation(recipe, plan_for(recipe, tmp_path, mode),
                                 None if zero_shot else directory, zero_shot)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        predictions = json.loads((path.parent / "predictions.json").read_text(encoding="utf-8"))
        assert evidence["kind"] == "evaluation" and evidence["stage"] == "completed" and evidence["schema_version"] == 1
        assert evidence["evaluation_protocol"] == ft.EVAL_PROTOCOL
        assert set(evidence["metrics"]) == {"fim_exact_match", "fim_token_exact_match"}
        assert evidence["metrics"] == ft.recompute_metrics(predictions)
        assert evidence["predictions"]["sha256"] == ft.sha256_json(predictions)
        assert evidence["config_sha256"] == ft.sha256_json(evidence["config"])
        assert evidence["sample_count"] == len(predictions) == len(calls["generations"])
        assert calls["models"][0][0] == ft.MODEL
        assert calls["models"][0][1]["revision"] == recipe.get("model.revision")
        assert calls["models"][0][1]["attn_implementation"] == "eager"
        for prediction, call in zip(predictions, calls["generations"]):
            assert call["input_ids"] == [1] + prediction["prefix_ids"] + [2] + prediction["suffix_ids"] + [3]
            assert "labels" not in call["kwargs"] and "gold_ids" not in call["kwargs"]
            assert prediction["input_ids"] == call["input_ids"]
        if not zero_shot:
            assert calls["adapter_source"] == str(directory.resolve())
            assert calls["tokenizers"][-1][1] == {"local_files_only": True, "trust_remote_code": False}
        results.append(evidence["config"]["generation"])
        manifests.append(evidence["evaluation_manifest"])
    assert results[0] == results[1] and manifests[0] == manifests[1]


def test_eval_tokenizer_drift_rejected(recipe, rows, completed, tmp_path, monkeypatch):
    directory, _, _ = completed
    runtime, calls = fake_runtime(recipe, rows)
    original = runtime["transformers"].AutoTokenizer.from_pretrained
    def load(source, **kwargs):
        tokenizer = original(source, **kwargs)
        if kwargs.get("local_files_only"):
            tokenizer.vocab["z"] = 300
        return tokenizer
    runtime["transformers"].AutoTokenizer.from_pretrained = load
    install_fake(monkeypatch, runtime)
    with pytest.raises(ft.TaskExecutionError, match="tokenizer"):
        ft.run_evaluation(recipe, plan_for(recipe, tmp_path, "eval"), directory)
    assert calls["generations"] == []
    assert not (tmp_path / "eval" / "evaluation.json").exists()


@pytest.mark.parametrize("field", sorted(ft.TRAIN_KEYS))
def test_each_training_parameter_must_be_declared(recipe, field):
    del recipe.data["train"][field]
    with pytest.raises(ft.TaskExecutionError, match="缺少参数"):
        ft.require_supported(recipe)


@pytest.mark.parametrize("section,field,value", [("model", "base", "other"), ("dataset", "name", "other"),
                                                ("method", "trainer", "SFTTrainer")])
def test_plan_cannot_redirect_valid_recipe(recipe, section, field, value):
    with pytest.raises(ft.TaskExecutionError, match="plan"):
        ft.require_supported(recipe, {section: {field: value}})


@pytest.mark.parametrize("filename", ["adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "training.json"])
def test_checkpoint_missing_required_component(recipe, completed, tmp_path, filename):
    directory, _, _ = completed
    # 只重命名本测试创建的 tmp_path 文件，不操作仓库产物。
    path = directory / filename
    path.rename(path.with_suffix(path.suffix + ".missing"))
    with pytest.raises(ft.TaskExecutionError):
        ft.checkpoint_path(plan_for(recipe, tmp_path, "eval"), directory, recipe)


@pytest.mark.parametrize("filename", ["training.json", "adapter_config.json", "effective_config.json"])
def test_checkpoint_json_wrong_shape_is_task_error(recipe, completed, tmp_path, filename):
    directory, _, _ = completed
    (directory / filename).write_text("[]", encoding="utf-8")
    with pytest.raises(ft.TaskExecutionError):
        ft.checkpoint_path(plan_for(recipe, tmp_path, "eval"), directory, recipe)


@pytest.mark.parametrize("field", ["gap", "tail", "offset", "labelhash"])
def test_rehashed_transform_still_has_to_reconstruct_source_and_labels(recipe, rows, field):
    splits, source = ft.prepare_splits(rows, recipe)
    _, manifest = ft.train_examples(splits, Tokenizer(), recipe)
    ft.validate_transform_manifest(manifest, source, recipe, manifest["tokenizer_identity"])
    file = manifest["splits"]["validation"]["samples"][0]
    chunk = file["chunks"][0]
    if field == "gap":
        chunk["gap_stop"] += 1
    elif field == "offset":
        chunk["token_start"] += 1
    elif field == "labelhash":
        chunk["example_sha256"] = "d" * 64
    else:
        file["tail_dropped"]["count"] += 1
    manifest["splits"]["validation"]["sha256"] = ft.sha256_json(manifest["splits"]["validation"]["samples"])
    manifest = ft._hash_manifest(manifest)
    with pytest.raises(ft.TaskExecutionError):
        ft.validate_transform_manifest(manifest, source, recipe, manifest["tokenizer_identity"])


@pytest.mark.parametrize("api", ["quantization", "lora", "prepare", "attach", "trainer"])
def test_other_runtime_api_signatures_fail_before_loaders(recipe, rows, tmp_path, monkeypatch, api):
    runtime, calls = fake_runtime(recipe, rows)
    replacements = {
        "quantization": (runtime["transformers"], "BitsAndBytesConfig"),
        "lora": (runtime["peft"], "LoraConfig"),
        "prepare": (runtime["peft"], "prepare_model_for_kbit_training"),
        "attach": (runtime["peft"], "get_peft_model"),
        "trainer": (runtime["transformers"], "Trainer"),
    }
    target, name = replacements[api]
    setattr(target, name, lambda **kwargs: None)
    install_fake(monkeypatch, runtime)
    with pytest.raises(ft.TaskExecutionError, match="不显式支持"):
        ft.run_training(recipe, plan_for(recipe, tmp_path, "smoke"))
    assert calls["models"] == [] and calls["tokenizers"] == []


def test_only_text_comparison_no_execution_of_generated_code():
    import ast
    tree = ast.parse((ROOT / "src/recipe_lab/fim_task.py").read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in {"eval", "exec", "compile"} for node in ast.walk(tree))


def test_frozen_runtime_delegates_to_math_guard_without_importing_verifier(recipe, monkeypatch):
    calls = []
    monkeypatch.setattr(ft.mt, "require_frozen_runtime", lambda r, dependencies: calls.append((r, dependencies)))
    dependencies = {name: object() for name in ft.RUNTIME_PACKAGES}
    ft.require_frozen_runtime(recipe, dependencies)
    assert calls == [(recipe, dependencies)]


def test_runtime_missing_peft_before_model_download(monkeypatch):
    names = []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                                 is_bf16_supported=lambda: True), version=SimpleNamespace(cuda="test"))
    def importer(name):
        names.append(name)
        if name == "peft":
            raise ImportError("测试缺少 peft")
        return torch if name == "torch" else object()
    monkeypatch.setattr(ft.importlib, "import_module", importer)
    with pytest.raises(ft.TaskExecutionError, match="缺少 peft"):
        ft.load_runtime()
    assert names == ["torch", "transformers", "datasets", "peft"]


def test_code_lock_identity_drift_before_save(recipe, rows, tmp_path, monkeypatch):
    runtime, _ = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    identity = ft.execution_identity(recipe)
    counter = 0
    def drift(r):
        nonlocal counter
        counter += 1
        return identity if counter == 1 else {**identity, "code_sha256": "d" * 64}
    monkeypatch.setattr(ft, "execution_identity", drift)
    with pytest.raises(ft.TaskExecutionError, match="漂移"):
        ft.run_training(recipe, plan_for(recipe, tmp_path, "smoke"))
    assert not (tmp_path / "smoke" / "training.json").exists()


def test_upstream_candidate_q_attn_can_be_absent(recipe, rows, tmp_path, monkeypatch):
    runtime, calls = fake_runtime(recipe, rows)
    model = runtime["transformers"].AutoModelForCausalLM.from_pretrained(ft.MODEL)
    model.named_modules = lambda: [(f"layer.{name}", object()) for name in ft.REQUIRED_MODULES]
    install_fake(monkeypatch, runtime)
    path = ft.run_training(recipe, plan_for(recipe, tmp_path, "smoke"))
    evidence = json.loads(path.read_text())
    assert evidence["config"]["lora_execution"]["matched_target_modules"]["q_attn"] == []
    assert ft.checkpoint_path(plan_for(recipe, tmp_path, "eval"), path.parent, recipe) == path.parent


@pytest.mark.parametrize("tamper", ["missing", "extra", "shape", "value", "nonfinite", "unsafe-key"])
def test_strict_adapter_tensor_keys_shape_values(recipe, rows, tmp_path, monkeypatch, tamper):
    runtime, _ = fake_runtime(recipe, rows)
    model = runtime["transformers"].AutoModelForCausalLM.from_pretrained(ft.MODEL)
    state = runtime["peft"].get_peft_model_state_dict(model, "default", False)
    if tamper == "missing":
        state.pop(next(iter(state)))
    elif tamper == "extra":
        state["extra.lora_A.weight"] = Tensor([0])
    elif tamper == "shape":
        state[next(iter(state))] = Tensor([1.0])
    elif tamper == "value":
        state[next(iter(state))] = Tensor([9.0, 2.0])
    elif tamper == "nonfinite":
        runtime["torch"].isfinite = lambda tensor: Tensor([False])
    else:
        def unsafe(model, adapter_name, save_embedding_layers):
            return {"base.weight": Tensor([1.0])}
        runtime["peft"].get_peft_model_state_dict = unsafe
        state = unsafe(model, "default", False)
    monkeypatch.setattr(ft, "_saved_adapter_state", lambda path: state)
    with pytest.raises(ft.TaskExecutionError, match="tensor"):
        ft.verify_adapter_tensors(model, tmp_path, runtime)


def test_actual_recipe_routes_and_publication_fail_closed():
    from recipe_lab.loader import load_recipes
    from recipe_lab import baseline_evidence as be
    r = next(x for x in load_recipes(ROOT) if x.id == ft.RECIPE_ID)
    for mode in ("dry-run", "smoke", "full", "eval", "zero-shot"):
        assert build_plan(r, mode)["execution_supported"]
    assert be.suite_problems(r)
    assert be.evaluation_problems(r, {}, ROOT / "absent.json", role="base")
    assert not be.zero_train_measured(r)
    from test_math_task import runner
    assert runner("train").fim_adapter is ft
    assert runner("eval").eval_plan(r)["protocol"] == ft.EVAL_PROTOCOL
    assert runner("zero_shot").zero_shot_plan(r)["protocol"] == ft.EVAL_PROTOCOL
    other = copy.deepcopy(r.data)
    other["execution"]["task_id"] = "math-grpo-v1"
    assert not build_plan(Recipe(r.id, r.path, other))["execution_supported"]
