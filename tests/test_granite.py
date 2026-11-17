"""Granite Vision / Geoperception LineComparison 自建协议（data/train/eval）隔离离线测试。

一手事实边界：上游只做 50/50 train/test、无 validation，collator 按 `<|assistant|>`
位置隐式 mask 且没有显式 image_token_id 屏蔽，无 revision / 训练 seed / compute_metrics，
评估只有 test_idx=20 单样本定性生成，正文写 ChartQA 但代码从未加载，说 merge 却只有
PeftModel.from_pretrained，标题写 LoRA 而 use_dora=True。本仓自建协议不冒充上游复现。

所有产物只写 tmp_path；fake runtime 是确定性合成夹具，不加载 torch/transformers，
不联网、不产 GPU 证据。JSON/hash 只证明仓内内容一致性，不构成 GPU 实跑证据，也不晋级 B2/B3。
comparison/export 发布级两维与 test_granite_release.py 另测：task_router 已按精确 recipe_id
注册 granite-lincomparison-v1，发布级 dry-run 分派到 Granite 专用协议，--full 仍失败关闭。
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import granite_task as gt
from recipe_lab import task_router as tr
from recipe_lab.loader import load_recipes
from recipe_lab.math_task import TaskExecutionError
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan

ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = "frozen.lock.toml"
MODEL_REV = "a" * 40
DATA_REV = "b" * 40
SOURCE_SHA = "c" * 40
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 32000
EOS = "</s>"
EOS_ID = 2


# ---------------------------------------------------------------- 合成输入

class Image:
    """最小图像对象：mode + tobytes()，与 PIL 的 hash 相关行为一致。"""

    def __init__(self, payload: bytes, mode: str = "RGB"):
        self.mode = mode
        self._payload = payload

    def tobytes(self):
        return self._payload

    def convert(self, mode):
        return Image(self._payload, mode)


class Tokenizer:
    name_or_path = "fake/granite-tokenizer"
    vocab_size = 512
    eos_token = EOS
    eos_token_id = EOS_ID
    pad_token = EOS
    pad_token_id = EOS_ID
    chat_template = "fake-granite-chat-template"


def _render(messages):
    out = []
    for message in messages:
        out.append("<|" + message["role"] + "|>")
        for part in message["content"]:
            if part["type"] == "text":
                out.append(part["text"])
            elif part["type"] == "image":
                out.append(IMAGE_TOKEN)
    return "".join(out)


class Processor:
    """字符级确定性 processor：prompt 必为 full 的字符前缀，<image> 映射为单一 token id。"""

    name_or_path = "fake/granite-processor"
    image_token = IMAGE_TOKEN
    image_token_id = IMAGE_TOKEN_ID

    def __init__(self):
        self.tokenizer = Tokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        body = _render(messages)
        return body + ("<|assistant|>" if add_generation_prompt else EOS)

    @staticmethod
    def _ids(text):
        out, parts = [], text.split(IMAGE_TOKEN)
        for index, part in enumerate(parts):
            out.extend(ord(char) % 256 for char in part)
            if index < len(parts) - 1:
                out.append(IMAGE_TOKEN_ID)
        return out

    @staticmethod
    def _decode(ids):
        return "".join(
            IMAGE_TOKEN if int(token) == IMAGE_TOKEN_ID
            else EOS if int(token) == EOS_ID
            else chr(int(token) % 256)
            for token in ids
        )

    def __call__(self, text, images, return_tensors=None, padding=False):
        assert isinstance(text, str) and isinstance(images, list)
        assert len(images) == 1 and len(images[0]) == 1
        return {"input_ids": [self._ids(text)],
                "pixel_values": [[[0.0] * 4, [1.0] * 4]]}

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self._decode(sequence) for sequence in sequences]

    def save_pretrained(self, directory):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "processor_config.json").write_text(
            json.dumps({"processor_class": "fake-granite-processor"}), encoding="utf-8")


class Tensor:
    def __init__(self, values, dtype=None):
        self.values = list(values)
        self.dtype = dtype
        self.shape = (len(self.values),)

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, dtype=None):
        return self

    def all(self):
        return all(bool(value) for value in self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class Parameter:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad
        self.dtype = None

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return 1


def make_rows(count=240):
    """120 张不同图像 × 2 个问答变体；同图必须落同一 split，内容全局唯一。"""
    rows = []
    for index in range(count):
        image_id = index % 120
        rows.append({
            "image": Image(f"raw-image-{image_id}".encode()),
            "question": f"Which line is longer in figure {image_id}, variant {index}?",
            "answer": f"line_{index}_is_longer",
            "predicate": "LineComparison",
        })
    return rows


def _recipe(tmp_path):
    real = next(r for r in load_recipes(ROOT) if r.id == gt.RECIPE_ID)
    data = copy.deepcopy(real.data)
    data["model"]["revision"] = MODEL_REV
    data["dataset"]["revision"] = DATA_REV
    data["source_sha"] = SOURCE_SHA
    data["execution"]["lockfile"] = LOCKFILE
    lock = "\n".join([
        "schema_version = 1", 'status = "frozen"', f'recipe_id = "{gt.RECIPE_ID}"',
        f'model_id = "{gt.MODEL}"', f'dataset_id = "{gt.DATASET}"',
        f'model_revision = "{MODEL_REV}"', f'dataset_revision = "{DATA_REV}"',
        f'source_sha = "{SOURCE_SHA}"',
        "requirements = [" + ", ".join(f'"{name}==test"' for name in gt.RUNTIME_PACKAGES) + "]",
        "[runtime]", 'python = "test"', 'cuda = "test"', 'driver = "test"', 'gpu = "test"',
    ]) + "\n"
    (tmp_path / LOCKFILE).write_text(lock, encoding="utf-8")
    return Recipe(gt.RECIPE_ID, tmp_path / "recipe.toml", data)


def explicit_factory(keys):
    def factory(**kwargs):
        return SimpleNamespace(**kwargs)

    factory.__signature__ = inspect.Signature([
        inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY) for key in keys
    ])
    return factory


def fake_runtime(r, rows):
    """确定性合成 runtime：只实现 granite_task 显式检查过的 API，不加载任何训练框架。"""
    processor = Processor()
    splits, _ = gt.prepare_splits(rows, r)
    gold_by_prompt = {
        tuple(Processor._ids(gt.render_prompt(row, processor))): Processor._ids(row["answer"])
        for row in splits["test"]
    }
    parameters = {"base.weight": Parameter(True)}
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True), device="synthetic:0")
    model.named_modules = lambda: [
        ("language_model.layers.0.q_proj", object()),
        ("language_model.layers.0.v_proj", object()),
        ("vision_tower.0", object()),
    ]
    model.named_parameters = lambda: list(parameters.items())
    model.parameters = lambda: list(parameters.values())
    calls = {"adapter_load": 0, "generate": 0}

    def generate(input_ids, attention_mask, pixel_values, **kwargs):
        assert set(kwargs) == {"max_new_tokens", "do_sample", "num_beams",
                               "pad_token_id", "eos_token_id"}
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        prompt = list(input_ids.values[0])
        gold = gold_by_prompt[tuple(prompt)][:kwargs["max_new_tokens"]]
        calls["generate"] += 1
        return [prompt + gold + [EOS_ID]]

    model.generate = generate

    def save_pretrained(directory, safe_serialization=True):
        assert safe_serialization is True
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        adapter = {
            "base_model_name_or_path": gt.MODEL, "revision": r.get("model.revision"),
            "r": r.get("peft.r"), "lora_alpha": r.get("peft.alpha"),
            "lora_dropout": r.get("peft.dropout"),
            "target_modules": list(model.lora_targets),
            "bias": r.get("peft.bias"), "task_type": r.get("peft.task_type"),
            "use_dora": True, "init_lora_weights": r.get("peft.init_lora_weights"),
        }
        (target / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"synthetic-granite-dora-adapter")

    model.save_pretrained = save_pretrained

    def load_dataset(source, **kwargs):
        assert kwargs.get("revision") == DATA_REV and kwargs.get("split") == "train"
        return iter(copy.deepcopy(rows))

    def prepare_kbit(model, use_gradient_checkpointing):
        assert use_gradient_checkpointing is True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def get_peft_model(model, peft_config):
        targets = list(peft_config.target_modules)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name in targets:
            parameters[f"adapter.{name}.lora_A.weight"] = Parameter(True)
            parameters[f"adapter.{name}.lora_B.weight"] = Parameter(True)
        model.lora_targets = targets
        return model

    def state_dict(model, adapter_name, save_embedding_layers):
        return {
            "base_model.model.layer.q_proj.lora_A.weight": Tensor([1.0, 2.0]),
            "base_model.model.layer.q_proj.lora_B.weight": Tensor([3.0, 4.0]),
        }

    def load_adapter(model, model_id, is_trainable):
        assert is_trainable is False
        calls["adapter_load"] += 1
        return model

    canonical = gt.config_kwargs(
        r, {"mode": "full", "recipe_id": r.id, "task_id": gt.TASK_ID}, Path("unused"))

    class Trainer:
        def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
            self.state = SimpleNamespace(global_step=args.max_steps)
            self.collator = data_collator

        def train(self):
            item = {"input_ids": [IMAGE_TOKEN_ID, 11, 12],
                    "labels": [-100, -100, 12], "attention_mask": [1, 1, 1],
                    "pixel_values": [[[0.0] * 4, [1.0] * 4]], "pixel_shape": [1, 2, 4]}
            batch = self.collator([copy.deepcopy(item), copy.deepcopy(item)])
            assert set(batch) == {"input_ids", "labels", "attention_mask", "pixel_values"}
            return SimpleNamespace(metrics={"train_loss": 0.5})

    fake_torch = SimpleNamespace(
        __version__="test", long="test-long", float32="test-float32",
        bfloat16="test-bfloat16",
        tensor=lambda values, **kwargs: Tensor(values, kwargs.get("dtype")),
        isfinite=lambda tensor: Tensor([True] * len(tensor.values)),
        equal=lambda a, b: a.values == b.values,
        no_grad=contextlib.nullcontext,
        version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(
            synchronize=lambda: None, get_device_name=lambda _: "test-device",
            max_memory_allocated=lambda: 0, is_available=lambda: True,
            device_count=lambda: 1),
        are_deterministic_algorithms_enabled=lambda: True,
    )
    runtime = {
        "torch": fake_torch,
        "transformers": SimpleNamespace(
            __version__="test",
            TrainingArguments=explicit_factory(list(canonical)),
            BitsAndBytesConfig=explicit_factory([
                "load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype",
                "bnb_4bit_use_double_quant", "llm_int8_skip_modules",
                "llm_int8_enable_fp32_cpu_offload"]),
            Trainer=Trainer,
            AutoProcessor=SimpleNamespace(from_pretrained=lambda source, **kwargs: Processor()),
            AutoModelForVision2Seq=SimpleNamespace(from_pretrained=lambda source, **kwargs: model),
            set_seed=lambda seed: None,
        ),
        "datasets": SimpleNamespace(__version__="test", load_dataset=load_dataset,
                                    Dataset=SimpleNamespace(from_list=lambda items: items)),
        "peft": SimpleNamespace(
            __version__="test", get_peft_model=get_peft_model,
            LoraConfig=explicit_factory([
                "r", "lora_alpha", "lora_dropout", "target_modules", "use_dora",
                "init_lora_weights", "bias", "task_type", "base_model_name_or_path",
                "revision"]),
            prepare_model_for_kbit_training=prepare_kbit,
            get_peft_model_state_dict=state_dict,
            PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        ),
        "bitsandbytes": SimpleNamespace(__version__="test"),
        "safetensors": SimpleNamespace(__version__="test"),
    }
    return runtime, calls, model


def install_fake(monkeypatch, runtime):
    monkeypatch.setattr(gt, "load_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(gt, "require_frozen_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(gt, "apply_determinism", lambda *args, **kwargs: None)
    monkeypatch.setattr(gt, "importlib", SimpleNamespace(
        import_module=lambda name: SimpleNamespace(
            load_file=lambda path, device: runtime["peft"].get_peft_model_state_dict(
                None, "default", False))))


def _plan(r, root, mode):
    raw = build_plan(r, mode)
    raw["run"]["output_dir"] = str(root)
    plan = gt.task_plan(r, raw, mode)
    assert plan["execution_supported"], plan.get("unsupported_reason")
    return plan


@pytest.fixture
def recipe(tmp_path):
    return _recipe(tmp_path)


@pytest.fixture
def rows():
    return make_rows()


# ---------------------------------------------------------------- 一手事实与契约

def test_primary_fact_constants_are_explicit():
    assert gt.MODEL == "ibm-granite/granite-vision-3.1-2b-preview"
    assert gt.DATASET == "euclid-multimodal/Geoperception"
    assert gt.PREDICATE == "LineComparison"
    assert gt.FULL_STEPS == 150 and gt.SMOKE_STEPS == 5
    assert gt.GRANITE_SPEC["upstream_quantitative_evaluation"] is False
    assert gt.GRANITE_SPEC["concurrent_scheduling"] is False
    assert "image_token_id" in gt.GRANITE_SPEC["loss"]
    assert gt.GRANITE_SPEC["split_percent"] == [80, 10, 10]
    assert "不解释几何语义" in gt.GRANITE_SPEC["metric_definition"]


def test_real_recipe_matches_module_contract(root, recipe):
    real = next(r for r in load_recipes(root) if r.id == gt.RECIPE_ID)
    assert gt.require_supported(real) == gt.TASK_ID
    assert gt.require_supported(recipe) == gt.TASK_ID
    assert real.get("granite") == gt.GRANITE_SPEC
    assert real.get("method.trainer") == "Trainer"
    assert real.get("method.upstream_trainer") == "SFTTrainer"
    assert real.get("method.peft") == "qlora-dora"
    assert real.get("execution.task_id") == gt.TASK_ID, "发布级管线已注册，task_id 必须精确声明"
    assert real.get("baseline.result") == "unmeasured"


@pytest.mark.parametrize("mutate", [
    lambda d: d["train"].update({"surprise": 1}),
    lambda d: d["train"].update({"max_steps": 151}),
    lambda d: d["train"].update({"seq_length": 8}),
    lambda d: d["evaluation"].update({"metrics": ["accuracy"]}),
    lambda d: d["evaluation"].update({"max_new_tokens": 4096}),
    lambda d: d["granite"].update({"split_percent": [50, 50, 0]}),
    lambda d: d["granite"].pop("image_hash"),
    lambda d: d["peft"].update({"use_dora": False}),
    lambda d: d["quantization"].update({"load_in_4bit": False}),
    lambda d: d["upstream"].pop("chatqa_text_code_mismatch"),
    lambda d: d.pop("modality_contract"),
    lambda d: d["model"].update({"has_images": False}),
    lambda d: d["execution"].update({"task_id": "invented-v1"}),
    lambda d: d["baseline"].update({"result": "passes"}),
])
def test_require_supported_rejects_any_drift(recipe, mutate):
    data = copy.deepcopy(recipe.data)
    mutate(data)
    with pytest.raises(TaskExecutionError):
        gt.require_supported(Recipe(recipe.id, recipe.path, data))


def test_task_plan_is_readonly_and_task_specific(recipe, tmp_path):
    before = json.dumps(recipe.data, sort_keys=True, ensure_ascii=False)
    plan = gt.task_plan(recipe, build_plan(recipe, "dry-run"), "dry-run")
    assert plan["execution_supported"] is True
    assert plan["task_id"] == gt.TASK_ID
    assert plan["evaluation"]["protocol"] == gt.EVAL_PROTOCOL
    assert plan["evaluation"]["upstream_quantitative_evaluation"] is False
    assert plan["evaluation"]["generated_code_executed"] is False
    assert plan["evaluation"]["training_seed_variance_measured"] is False
    assert "image_token_id" in plan["migration"]["mask"]
    assert "ChartQA" in plan["migration"]["evaluation"]
    assert json.dumps(recipe.data, sort_keys=True, ensure_ascii=False) == before
    assert not (tmp_path / "runs").exists()


def test_build_plan_dispatches_granite_adapter(root):
    r = next(x for x in load_recipes(root) if x.id == gt.RECIPE_ID)
    plan = build_plan(r, "dry-run")
    assert plan["execution_supported"] is True and plan["task_id"] == gt.TASK_ID
    assert plan["evaluation"]["protocol"] == gt.EVAL_PROTOCOL
    assert plan["modality_contract"]["processing_class"] == "processor"


def explicit_factory(keys):
    """Fake TrainingArguments：可调用且带明确 Signature，不依赖 **kwargs 宽容。"""

    def factory(**kwargs):
        return SimpleNamespace(**kwargs)

    factory.__signature__ = inspect.Signature([
        inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY) for key in keys
    ])
    return factory


def test_config_kwargs_and_training_arguments_api(recipe, tmp_path):
    config = gt.config_kwargs(
        recipe, {"mode": "full", "recipe_id": recipe.id, "task_id": gt.TASK_ID,
                 "run": {"steps": 150}}, tmp_path / "output")
    assert config["max_steps"] == 150 and config["output_dir"] == str(tmp_path / "output")
    assert config["evaluation_strategy"] == "steps" and config["report_to"] == "none"
    assert "seq_length" not in config and "seed_mechanism" not in config
    assert gt.training_arguments_identity(config)["max_steps"] == 150
    assert "output_dir" not in gt.training_arguments_identity(config)
    legacy = explicit_factory(config)
    assert gt.training_arguments_kwargs(legacy, config) == config
    newer_keys = [k for k in config if k != "evaluation_strategy"] + ["eval_strategy"]
    newer = explicit_factory(newer_keys)
    translated = gt.training_arguments_kwargs(newer, config)
    assert "eval_strategy" in translated and "evaluation_strategy" not in translated


# ---------------------------------------------------------------- 数据协议

def test_parse_row_accepts_primary_schema(rows):
    row = gt.parse_row(rows[0], 0)
    assert row["source_index"] == 0 and row["predicate"] == "LineComparison"
    assert row["image_mode"] == "RGB" and row["image_bytes"] > 0
    assert len(row["image_sha256"]) == 64 and len(row["content_sha256"]) == 64
    assert row["sample_id"].startswith(row["image_sha256"][:16])
    assert gt.parse_row(rows[0], 7)["image_sha256"] == row["image_sha256"]


def test_image_payload_and_ensure_rgb_do_not_change_split_hash(rows):
    raw = rows[0]
    mode, payload = gt.image_payload(raw["image"])
    assert mode == "RGB" and payload == raw["image"].tobytes()
    assert gt.image_payload(b"abc") == ("bytes", b"abc")
    original = gt.parse_row(raw, 0)["image_sha256"]
    converted = gt.ensure_rgb(Image(payload, "L"))
    assert converted.mode == "RGB"
    assert gt.parse_row({**raw, "image": Image(payload, "L")}, 0)["image_sha256"] != original
    with pytest.raises(TaskExecutionError):
        gt.image_payload(object())


@pytest.mark.parametrize("mutate", [
    lambda d: d.update({"unexpected": 1}),
    lambda d: d.pop("image"),
    lambda d: d.pop("answer"),
    lambda d: d.update({"question": ""}),
    lambda d: d.update({"answer": "   "}),
    lambda d: d.update({"predicate": "OtherPredicate"}),
    lambda d: d.update({"image": object()}),
    lambda d: d.update({"question": 42}),
])
def test_parse_row_rejects_invalid_rows(rows, mutate):
    raw = copy.deepcopy(rows[0])
    mutate(raw)
    with pytest.raises(TaskExecutionError):
        gt.parse_row(raw, 0)


def test_split_bucket_is_deterministic_and_three_way():
    digest = hashlib.sha256(b"RGB\x00raw-image-0").hexdigest()
    assert gt.split_bucket(digest, 42) == gt.split_bucket(digest, 42)
    assert {gt.split_bucket(hashlib.sha256(f"x{i}".encode()).hexdigest(), 42)
            for i in range(200)} == {"train", "validation", "test"}


def test_prepare_splits_is_deterministic_and_image_isolated(recipe, rows):
    splits, manifest = gt.prepare_splits(rows, recipe)
    assert all(splits[name] for name in ("train", "validation", "test"))
    assert manifest["counts"]["source"] == len(rows)
    assert sum(manifest["counts"][n] for n in ("train", "validation", "test")) == len(rows)
    gt.validate_source_manifest(manifest, recipe)
    again, second = gt.prepare_splits(rows, recipe)
    assert second == manifest
    assert [r["sample_id"] for r in again["test"]] == [r["sample_id"] for r in splits["test"]]
    owners = {}
    for name, items in splits.items():
        for row in items:
            previous = owners.setdefault(row["image_sha256"], name)
            assert previous == name, "同图问答变体不得跨 split"
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not ({r["image_sha256"] for r in splits[left]}
                    & {r["image_sha256"] for r in splits[right]})


def test_prepare_splits_rejects_duplicates_and_empty_splits(recipe, rows):
    with pytest.raises(TaskExecutionError, match="重复"):
        gt.prepare_splits(rows + [copy.deepcopy(rows[0])], recipe)
    normalized = copy.deepcopy(rows[0])
    normalized["question"] = "  " + normalized["question"].upper() + "  "
    with pytest.raises(TaskExecutionError, match="重复"):
        gt.prepare_splits(rows + [normalized], recipe)
    with pytest.raises(TaskExecutionError, match="为空"):
        gt.prepare_splits(rows[:1], recipe)


def test_validate_source_manifest_rejects_tampering(recipe, rows):
    _, manifest = gt.prepare_splits(rows, recipe)
    for mutate in (
        lambda m: m["split_hashes"]["train"].append("0" * 64),
        lambda m: m["split_hashes"]["test"].__setitem__(0, m["split_hashes"]["train"][0]),
        lambda m: m.update({"data_seed": 43}),
        lambda m: m.update({"protocol": "invented-v1"}),
        lambda m: m.update({"counts": {**m["counts"], "train": 1}}),
        lambda m: m.update({"image_split_owner": {}}),
        lambda m: m.update({"sha256": "0" * 64}),
    ):
        broken = copy.deepcopy(manifest)
        mutate(broken)
        with pytest.raises(TaskExecutionError):
            gt.validate_source_manifest(broken, recipe)


def test_load_splits_binds_dataset_revision(recipe, rows, monkeypatch):
    runtime = {"datasets": SimpleNamespace(load_dataset=lambda *a, **k: iter(rows))}
    splits, manifest = gt.load_splits(recipe, runtime)
    assert manifest["dataset_revision"] == DATA_REV and splits["test"]
    assert manifest["sha256"] == gt.sha256_json(
        {k: v for k, v in manifest.items() if k != "sha256"})


# ---------------------------------------------------------------- 图文编码与 mask

def test_processor_identity_requires_explicit_image_token_id():
    processor = Processor()
    identity = gt.processor_identity(processor)
    assert identity["image_token_id"] == IMAGE_TOKEN_ID
    assert identity["eos_token"] == EOS and identity["chat_template_sha256"]
    assert identity["sha256"] == gt._hash_manifest(identity)["sha256"]
    # 上游 processor 不暴露显式 image_token_id：用不含该属性的最小对象模拟，
    # 而不是 del 类属性（Processor.image_token_id 是类属性，del 实例会抛 AttributeError）。
    bare = SimpleNamespace(tokenizer=processor.tokenizer,
                           name_or_path="fake/no-explicit-image-token-id",
                           image_token=IMAGE_TOKEN)
    with pytest.raises(TaskExecutionError, match="image_token_id"):
        gt.processor_identity(bare)


def test_render_prompt_never_contains_gold(recipe, rows):
    processor = Processor()
    row = gt.parse_row(rows[0], 0)
    prompt = gt.render_prompt(row, processor)
    assert row["answer"] not in prompt
    assert gt.SYSTEM_MESSAGE in prompt and row["question"] in prompt
    assert IMAGE_TOKEN in prompt
    full = processor.apply_chat_template(gt.render_messages(row), tokenize=False)
    assert row["answer"] in full and full.startswith(prompt)


def test_encode_example_double_masks_and_checks_prefix(recipe, rows):
    processor = Processor()
    identity = gt.processor_identity(processor)
    item = gt.encode_example(gt.parse_row(rows[0], 0), processor, identity, recipe)
    prompt_tokens = item["prompt_tokens"]
    assert prompt_tokens > 0 and item["target_tokens"] > 0
    assert item["labels"][:prompt_tokens] == [-100] * prompt_tokens
    assert item["labels"][prompt_tokens:] == item["input_ids"][prompt_tokens:]
    assert item["masked_image_tokens"] >= 1
    assert all(label == -100 for token, label in zip(item["input_ids"], item["labels"])
               if token == IMAGE_TOKEN_ID)
    assert item["input_ids"][:prompt_tokens] == item["prompt_ids"]
    assert item["pixel_shape"] == [1, 2, 4]


def test_encode_example_rejects_overflow_and_prompt_leak(recipe, rows):
    processor = Processor()
    identity = gt.processor_identity(processor)
    small = copy.deepcopy(recipe.data)
    small["train"]["seq_length"] = 16
    small["evaluation"]["max_new_tokens"] = 2
    with pytest.raises(TaskExecutionError, match="seq_length"):
        gt.encode_example(gt.parse_row(rows[0], 0), processor, identity,
                          Recipe(recipe.id, recipe.path, small))

    # 假 processor 必须真的把 gold 拼进生成 prompt，才能触发 render_prompt 的
    # 「row['answer'] in text」泄漏检测；写死一句不含 answer 的文本是无效负例。
    answer = gt.parse_row(rows[0], 0)["answer"]

    class LeakyProcessor(Processor):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if add_generation_prompt:
                return "leaked gold: " + answer
            return "leaked gold: " + answer + EOS

    with pytest.raises(TaskExecutionError, match="gold 泄漏"):
        gt.render_prompt(gt.parse_row(rows[0], 0), LeakyProcessor())


def test_encode_example_rejects_non_prefix_full(recipe, rows):
    class BrokenProcessor(Processor):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if add_generation_prompt:
                return "PROMPT<|assistant|>"
            return "DIFFERENT" + gt.parse_row(rows[0], 0)["answer"] + EOS

    processor = BrokenProcessor()
    with pytest.raises(TaskExecutionError, match="token 前缀"):
        gt.encode_example(gt.parse_row(rows[0], 0), processor,
                          gt.processor_identity(processor), recipe)


def test_encode_example_rejects_missing_image_placeholder(recipe, rows):
    class NoImageProcessor(Processor):
        @staticmethod
        def _ids(text):
            return [ord(char) % 256 for char in text]

    processor = NoImageProcessor()
    with pytest.raises(TaskExecutionError, match="image placeholder"):
        gt.encode_example(gt.parse_row(rows[0], 0), processor,
                          gt.processor_identity(processor), recipe)


def test_collate_pads_and_masks_every_image_token(recipe, rows):
    processor = Processor()
    identity = gt.processor_identity(processor)
    items = [gt.encode_example(gt.parse_row(raw, i), processor, identity, recipe)
             for i, raw in enumerate(rows[:3])]
    batch = gt.collate(items, SimpleNamespace(
        long="int64", float32="float32",
        tensor=lambda values, **kwargs: Tensor(values, kwargs.get("dtype"))),
        EOS_ID, IMAGE_TOKEN_ID)
    width = max(len(item["input_ids"]) for item in items)
    assert all(len(row) == width for row in batch["labels"].values)
    for row_index, row in enumerate(batch["input_ids"].values):
        for index, token in enumerate(row):
            if token in (IMAGE_TOKEN_ID, EOS_ID) and index >= len(items[row_index]["input_ids"]):
                assert batch["labels"].values[row_index][index] == -100
    with pytest.raises(TaskExecutionError, match="空 batch"):
        gt.collate([], SimpleNamespace(), EOS_ID, IMAGE_TOKEN_ID)
    mismatched = copy.deepcopy(items[0])
    mismatched["pixel_shape"] = [9, 9, 9]
    with pytest.raises(TaskExecutionError, match="pixel_values 形状不一致"):
        gt.collate([items[0], mismatched], SimpleNamespace(
            long="int64", float32="float32",
            tensor=lambda values, **kwargs: Tensor(values, kwargs.get("dtype"))),
            EOS_ID, IMAGE_TOKEN_ID)


def test_train_examples_and_transform_manifest(recipe, rows):
    splits, source = gt.prepare_splits(rows, recipe)
    processor = Processor()
    identity = gt.processor_identity(processor)
    examples, transform = gt.train_examples(splits, processor, recipe)
    assert examples["train"] and examples["validation"]
    assert transform["sha256"] == gt._hash_manifest(transform)["sha256"]
    gt.validate_transform_manifest(transform, source, recipe, identity)
    for name in ("train", "validation"):
        for item in examples[name]:
            assert item["labels"].count(-100) >= 1
            assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])
    for mutate in (
        lambda m: m["entries"]["train"][0].update({"image_sha256": "0" * 64}),
        lambda m: m["entries"]["train"][0].update({"masked_image_tokens": 0}),
        lambda m: m["entries"]["train"][0].pop("example_sha256"),
        lambda m: m.update({"sha256": "0" * 64}),
        lambda m: m.update({"loss": "invented"}),
        lambda m: m.update({"processor_identity": {}}),
    ):
        broken = copy.deepcopy(transform)
        mutate(broken)
        with pytest.raises(TaskExecutionError):
            gt.validate_transform_manifest(broken, source, recipe, identity)


# ---------------------------------------------------------------- 评估协议

def test_clean_generation_and_score_prediction(rows):
    row = gt.parse_row(rows[0], 0)
    perfect = gt.score_prediction(row, row["answer"] + EOS + "trailing", EOS)
    assert perfect["exact_match"] == perfect["normalized_exact_match"] == 1.0
    assert perfect["empty"] == 0.0 and perfect["prediction"] == row["answer"]
    case = gt.score_prediction(row, "  " + row["answer"].upper() + "  ", EOS)
    assert case["exact_match"] == 0.0 and case["normalized_exact_match"] == 1.0
    empty = gt.score_prediction(row, EOS + "ignored", EOS)
    assert empty["exact_match"] == 0.0 and empty["empty"] == 1.0
    assert gt.clean_generation(None, EOS) == ""


def test_recompute_metrics_rejects_invalid_scores(rows):
    row = gt.parse_row(rows[0], 0)
    good = gt.score_prediction(row, row["answer"], EOS)
    bad = gt.score_prediction(row, "wrong", EOS)
    assert gt.recompute_metrics([good, bad]) == {name: 0.5 for name in gt.METRICS}
    for mutate in (
        lambda p: p[0].update({"exact_match": 1.5}),
        lambda p: p[0].update({"exact_match": "1"}),
        lambda p: p.append("not-a-dict"),
    ):
        predictions = [dict(good), dict(bad)]
        mutate(predictions)
        with pytest.raises(TaskExecutionError):
            gt.recompute_metrics(predictions)
    with pytest.raises(TaskExecutionError, match="逐题预测为空"):
        gt.recompute_metrics([])


def test_evaluation_manifest_rebuilds_gold_from_raw(recipe, rows):
    splits, source = gt.prepare_splits(rows, recipe)
    processor = Processor()
    identity = gt.processor_identity(processor)
    samples = gt.evaluation_examples(splits["test"], processor, identity, recipe)
    manifest = gt.evaluation_manifest(samples, identity, recipe, source)
    assert manifest["source_manifest_sha256"] == source["sha256"]
    assert manifest["samples_sha256"] == gt.sha256_json(samples)
    assert manifest["sha256"] == gt._hash_manifest(manifest)["sha256"]
    for sample, row in zip(samples, splits["test"]):
        assert sample["answer"] == row["answer"]
        assert row["answer"] not in sample["prompt"]
        assert sample["gold_sha256"] == gt.sha256_json(row["answer"])
        assert sample["prompt_ids"] == Processor._ids(sample["prompt"])
    tampered = copy.deepcopy(manifest)
    tampered["samples"][0]["answer"] = "forged"
    with pytest.raises(AssertionError):
        assert tampered["sha256"] == gt._hash_manifest(tampered)["sha256"]


# ---------------------------------------------------------------- 执行框架（fake runtime）

@pytest.fixture
def executed(tmp_path, recipe, rows, monkeypatch):
    runtime, calls, model = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    root = tmp_path / "experiment"
    train_plan = _plan(recipe, root, "full")
    training = gt.run_training(recipe, train_plan)
    eval_plan = _plan(recipe, root, "eval")
    evaluation = gt.run_evaluation(recipe, eval_plan, checkpoint=training.parent)
    return SimpleNamespace(root=root, training=training, evaluation=evaluation,
                           calls=calls, model=model, runtime=runtime)


def test_run_training_writes_verifiable_checkpoint(executed, recipe):
    training = json.loads(executed.training.read_text(encoding="utf-8"))
    assert training["kind"] == "training" and training["stage"] == "completed"
    assert training["task_id"] == gt.TASK_ID and training["mode"] == "full"
    assert training["steps"] == training["max_steps"] == gt.FULL_STEPS
    assert training["fingerprint"] == be.fingerprint(recipe)
    assert training["runtime"]["packages"]["torch"] == "test"
    assert training["config"]["lora_execution"]["use_dora"] is True
    assert training["config"]["lora_execution"]["tensor_reload_check"]["strict_values_verified"]
    assert training["processor_identity"]["image_token_id"] == IMAGE_TOKEN_ID
    assert training["transformation_manifest"]["entries"]["train"]
    assert gt.checkpoint_path(_plan(recipe, executed.root, "eval"),
                              checkpoint=executed.training.parent, recipe=recipe) == executed.training.parent


def test_checkpoint_path_rejects_tampering(executed, recipe):
    checkpoint = executed.training.parent
    plan = _plan(recipe, executed.root, "eval")
    original = (checkpoint / "training.json").read_bytes()
    training = json.loads(original)
    for mutate in (
        lambda d: d.update({"fingerprint": "0" * 64}),
        lambda d: d.update({"recipe_sha256": "0" * 64}),
        lambda d: d.update({"steps": 149}),
        lambda d: d.update({"model_revision": "0" * 40}),
        lambda d: d.update({"stage": "running"}),
        lambda d: d.update({"mock": True}),
        lambda d: d["config"].update({"granite": {"protocol": "invented"}}),
        lambda d: d.update({"processor_identity": {}}),
        lambda d: d.update({"transformation_manifest": {}}),
    ):
        broken = copy.deepcopy(training)
        mutate(broken)
        (checkpoint / "training.json").write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(TaskExecutionError):
            gt.checkpoint_path(plan, checkpoint=checkpoint, recipe=recipe)
    (checkpoint / "training.json").write_bytes(original)
    assert gt.checkpoint_path(plan, checkpoint=checkpoint, recipe=recipe) == checkpoint


def test_run_evaluation_recomputes_metrics_per_example(executed, recipe, rows):
    evidence = json.loads(executed.evaluation.read_text(encoding="utf-8"))
    assert evidence["kind"] == "evaluation" and evidence["model_role"] == "checkpoint"
    assert evidence["metrics"] == {name: 1.0 for name in gt.METRICS}
    assert evidence["sample_count"] > 0
    predictions = json.loads((executed.evaluation.parent / gt.PREDICTIONS_FILE).read_text())
    assert len(predictions["predictions"]) == evidence["sample_count"]
    assert predictions["sha256"] == gt.sha256_json(predictions["predictions"])
    assert gt.recompute_metrics(predictions["predictions"]) == evidence["metrics"]
    assert executed.calls["generate"] == evidence["sample_count"]
    assert executed.calls["adapter_load"] == 1
    assert evidence["evaluation_manifest"]["processor_identity"]["image_token_id"] == IMAGE_TOKEN_ID
    assert evidence["config"]["upstream_quantitative_evaluation"] is False


def test_zero_shot_uses_base_only_and_never_loads_adapter(tmp_path, recipe, rows, monkeypatch):
    runtime, calls, _ = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    root = tmp_path / "zero"
    plan = _plan(recipe, root, "zero-shot")
    path = gt.run_evaluation(recipe, plan, zero_shot=True)
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["model_role"] == "base"
    assert evidence["model_artifact"] == gt.base_artifact(recipe)
    assert evidence["config"]["checkpoint"] is None
    assert evidence["checkpoint_training_evidence"] is None
    assert calls["adapter_load"] == 0
    assert evidence["metrics"] == {name: 1.0 for name in gt.METRICS}


def test_evaluation_without_checkpoint_fails_closed(tmp_path, recipe, rows, monkeypatch):
    runtime, _, _ = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    root = tmp_path / "empty"
    plan = _plan(recipe, root, "eval")
    with pytest.raises(TaskExecutionError, match="没有训练 checkpoint"):
        gt.run_evaluation(recipe, plan)


def test_evaluation_rejects_repeated_greedy_as_seed_variance(tmp_path, recipe, rows, monkeypatch):
    runtime, _, _ = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    plan = _plan(recipe, tmp_path / "repeats", "eval")
    plan["requested_runs"] = 3
    with pytest.raises(TaskExecutionError, match="重复 greedy 不冒充 multi-run"):
        gt.run_evaluation(recipe, plan)


def test_output_directory_is_never_overwritten(executed, recipe, rows, monkeypatch):
    runtime, _, _ = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    plan = _plan(recipe, executed.root, "full")
    with pytest.raises(TaskExecutionError, match="输出目录非空"):
        gt.run_training(recipe, plan)


# ---------------------------------------------------------------- 发布级与证据边界

def test_task_router_registers_granite_exactly(recipe, root):
    real = next(r for r in load_recipes(root) if r.id == gt.RECIPE_ID)
    assert tr.resolve(real) == tr.resolve(recipe) == gt.TASK_ID
    assert tr.load(recipe, "task").__name__ == "recipe_lab.granite_task"
    assert tr.load(recipe, "suite").__name__ == "recipe_lab.granite_suite"
    assert tr.load(recipe, "export").__name__ == "recipe_lab.granite_export"
    changed = copy.deepcopy(recipe.data)
    changed["execution"]["task_id"] = "math-grpo-v1"
    with pytest.raises(TaskExecutionError, match="任务路由冲突"):
        tr.resolve(Recipe(recipe.id, recipe.path, changed))


def test_baseline_evidence_fails_closed_without_real_release_evidence(recipe, executed):
    # 发布级已按 Granite 精确分派，但隔离合成夹具没有三 seed completed suite、
    # 也没有与 frozen lock 一致的运行环境；门禁仍必须失败关闭，不能因「有代码」放行。
    assert tr.resolve(recipe) == gt.TASK_ID
    assert any("baseline-suite" in p for p in be.suite_problems(recipe))
    problems = be.evaluation_problems(
        recipe, json.loads(executed.evaluation.read_text(encoding="utf-8")),
        executed.evaluation, role="checkpoint")
    assert problems and any("Granite 评估内容拒绝" in p for p in problems)
    assert be.zero_train_measured(recipe) is False


def test_fingerprint_binds_granite_protocol(recipe):
    base = be.fingerprint(recipe)
    for path, value in (
        ("granite.loss", "invented mask"),
        ("granite.image_hash", "visible pixels only"),
        ("granite.split_algorithm", "random split"),
        ("upstream.mask", "invented upstream fact"),
        ("modality_contract.processing_class", "tokenizer"),
        ("evaluation.max_new_tokens", 65),
    ):
        changed = copy.deepcopy(recipe.data)
        cursor = changed
        parts = path.split(".")
        for key in parts[:-1]:
            cursor = cursor[key]
        cursor[parts[-1]] = value
        assert be.fingerprint(Recipe(recipe.id, recipe.path, changed)) != base, path


def test_frozen_environment_does_not_require_trl_for_granite(recipe):
    assert be.frozen_environment(recipe) is True
    lock = (recipe.path.parent / LOCKFILE).read_text(encoding="utf-8")
    assert "trl" not in lock
    assert all(f"{name}==test" in lock for name in gt.RUNTIME_PACKAGES)
    broken = copy.deepcopy(recipe.data)
    broken["dependencies"] = {"trl": "0.14.0"}
    assert be.frozen_environment(Recipe(recipe.id, recipe.path, broken)) is False


def test_cli_dry_runs_are_framework_free_and_task_specific(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "RECIPE_LAB_ROOT": str(ROOT)}
    for script in ("train.py", "eval.py", "zero_shot.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / "runner" / script), "--recipe", gt.RECIPE_ID,
             "--dry-run", "--json"], env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["execution_supported"] is True and plan["task_id"] == gt.TASK_ID
    # dry-run 是只读预演：显式冲突的 task-id 以 plan 明确标记 unsupported，而非进程失败；
    # 真实执行必须在框架导入之前失败关闭，不能带着 unsupported plan 继续训练。
    bad = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "train.py"), "--recipe", gt.RECIPE_ID,
         "--task-id", "math-grpo-v1", "--dry-run", "--json"],
        env=env, capture_output=True, text=True, check=False)
    assert bad.returncode == 0, bad.stderr
    conflict = json.loads(bad.stdout)
    assert conflict["execution_supported"] is False
    assert "尚未实现该执行路由" in conflict["unsupported_reason"]
    full = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "train.py"), "--recipe", gt.RECIPE_ID,
         "--task-id", "math-grpo-v1", "--full", "--output-root", str(tmp_path / "full")],
        env=env, capture_output=True, text=True, check=False)
    assert full.returncode != 0
    assert "参数错误" in (full.stdout + full.stderr) or "执行失败" in (full.stdout + full.stderr)


def test_release_dry_runs_dispatch_to_granite_and_full_fails_closed(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "RECIPE_LAB_ROOT": str(ROOT)}
    suite = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "suite.py"), "--recipe", gt.RECIPE_ID,
         "--seeds", "42", "43", "44", "--output-root", str(tmp_path / "suite"), "--dry-run"],
        env=env, capture_output=True, text=True, check=False)
    assert suite.returncode == 0, suite.stderr
    suite_plan = json.loads(suite.stdout)
    assert "release_pipeline" not in suite_plan, "已注册发布级管线，不得再声明 unavailable"
    assert suite_plan["task_id"] == gt.TASK_ID and suite_plan["execution_supported"] is True
    assert suite_plan["protocol"] == gt.EVAL_PROTOCOL
    assert suite_plan["metrics"] == list(gt.METRICS)
    assert suite_plan["baseline_promotion"] == "none" and len(suite_plan["runs"]) == 3
    export = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "export.py"), "--recipe", gt.RECIPE_ID,
         "--checkpoint", "missing", "--evaluation", "missing",
         "--output", str(tmp_path / "bundle"), "--dry-run"],
        env=env, capture_output=True, text=True, check=False)
    assert export.returncode == 0, export.stderr
    export_plan = json.loads(export.stdout)
    assert "release_pipeline" not in export_plan
    assert export_plan["task_id"] == gt.TASK_ID and export_plan["tolerance"] is None
    assert "not_declared" in export_plan["tolerance_status"]
    assert export_plan["license"]["status"] == "pending"
    assert export_plan["base_weights_bundled"] is False
    assert export_plan["vision_contract"]["explicit_image_token_mask"] is True
    full = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "suite.py"), "--recipe", gt.RECIPE_ID,
         "--seeds", "42", "43", "44", "--output-root", str(tmp_path / "full"), "--full"],
        env=env, capture_output=True, text=True, check=False)
    assert full.returncode != 0 and "frozen_environment" in (full.stdout + full.stderr)
    assert not (tmp_path / "full").exists()
    full_export = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "export.py"), "--recipe", gt.RECIPE_ID,
         "--checkpoint", str(tmp_path / "absent-checkpoint"),
         "--evaluation", str(tmp_path / "absent-eval.json"),
         "--output", str(tmp_path / "bundle-full"), "--full"],
        env=env, capture_output=True, text=True, check=False)
    assert full_export.returncode != 0
    assert not (tmp_path / "bundle-full").exists(), "失败不得留下 completed 导出包"
