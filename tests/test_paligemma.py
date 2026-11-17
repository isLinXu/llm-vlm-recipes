"""PaliGemma2 / RefCOCO grounding 自建协议（五维代码）隔离离线测试。

一手事实边界：上游训练/推理 prompt 三处不一致、正文 bbox 写 xyxy 而训练代码按
COCO xywh、图像来自 Flickr URL 运行时下载、labels 只 mask pad/image token 且
prompt 也计 loss、有 eval_dataset 却无任何定量指标、无 revision/训练 seed。
本仓自建协议不冒充上游复现，也不冒充 mAP/AP 或业务验收。

所有产物只写 tmp_path；fake runtime 是确定性合成夹具，不加载 torch/transformers，
不联网、不产 GPU 证据。JSON/hash 只证明仓内内容一致性，不构成 GPU 实跑证据，
也不晋级 B2/B3。发布级 comparison/export 由 test_paligemma_release.py 另测。
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe_lab import baseline_evidence as be
from recipe_lab import paligemma_task as pt
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
SEP_ID = 10
EOS = "</s>"
EOS_ID = 2
PIXELS = [[[0.0] * 4, [1.0] * 4]]


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
    name_or_path = "fake/paligemma-tokenizer"
    vocab_size = 262144
    eos_token = EOS
    eos_token_id = EOS_ID
    pad_token = EOS
    pad_token_id = EOS_ID
    # PaliGemma2-pt 没有 chat template；<image> 是 additional special token。
    chat_template = ""
    additional_special_tokens = [IMAGE_TOKEN]
    additional_special_tokens_ids = [IMAGE_TOKEN_ID]


class Processor:
    """字符级确定性 processor：suffix 编码为 SEP + target，prompt 必为 full 前缀。"""

    name_or_path = "fake/paligemma-processor"
    image_token_id = IMAGE_TOKEN_ID

    def __init__(self):
        self.tokenizer = Tokenizer()

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
            else "\n" if int(token) == SEP_ID
            else chr(int(token) % 256)
            for token in ids
        )

    def __call__(self, text, images, return_tensors=None, padding=False, suffix=None):
        assert isinstance(text, str) and isinstance(images, list)
        assert len(images) == 1 and len(images[0]) == 1
        assert return_tensors is None and padding is False
        ids = self._ids(text)
        if suffix is not None:
            ids = ids + [SEP_ID] + self._ids(suffix)
        return {"input_ids": [ids], "pixel_values": copy.deepcopy(PIXELS)}

    def batch_decode(self, sequences, skip_special_tokens=False):
        assert skip_special_tokens is False
        return [self._decode(sequence) for sequence in sequences]

    def save_pretrained(self, directory):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "processor_config.json").write_text(
            json.dumps({"processor_class": "fake-paligemma-processor"}), encoding="utf-8")


class Tensor:
    def __init__(self, values, dtype=None):
        self.values = list(values)
        self.dtype = dtype
        self.shape = (len(self.values),)

    def detach(self): return self
    def cpu(self): return self
    def to(self, dtype=None): return self
    def all(self): return all(bool(value) for value in self.values)
    def __iter__(self): return iter(self.values)
    def __getitem__(self, index): return self.values[index]
    def __len__(self): return len(self.values)


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
    """120 张不同图像 × 2 个 caption 变体；同图必须落同一 split，内容全局唯一。"""
    return [
        {"image": Image(f"raw-image-{index % 120}".encode()),
         "bbox": [10 + index % 20, 12 + index % 15, 40, 30],
         "caption": f"object variant {index} in figure {index % 120}",
         "width": 320, "height": 240}
        for index in range(count)
    ]


def _recipe(tmp_path):
    real = next(r for r in load_recipes(ROOT) if r.id == pt.RECIPE_ID)
    data = copy.deepcopy(real.data)
    data["model"]["revision"] = MODEL_REV
    data["dataset"]["revision"] = DATA_REV
    data["source_sha"] = SOURCE_SHA
    data["execution"]["lockfile"] = LOCKFILE
    lock = "\n".join([
        "schema_version = 1", 'status = "frozen"', f'recipe_id = "{pt.RECIPE_ID}"',
        f'model_id = "{pt.MODEL}"', f'dataset_id = "{pt.DATASET}"',
        f'model_revision = "{MODEL_REV}"', f'dataset_revision = "{DATA_REV}"',
        f'source_sha = "{SOURCE_SHA}"',
        "requirements = [" + ", ".join(f'"{name}==test"' for name in pt.RUNTIME_PACKAGES) + "]",
        "[runtime]", f'python = "{sys.version.split()[0]}"',
        'cuda = "test-cuda"', 'driver = "test-driver"', 'gpu = "test-device"',
    ]) + "\n"
    (tmp_path / LOCKFILE).write_text(lock, encoding="utf-8")
    return Recipe(pt.RECIPE_ID, tmp_path / "recipe.toml", data)


def explicit_factory(keys):
    def factory(**kwargs):
        return SimpleNamespace(**kwargs)
    factory.__signature__ = inspect.Signature([
        inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY) for key in keys])
    return factory


def fake_runtime(r, rows):
    """确定性合成 runtime：只实现 paligemma_task 显式检查过的 API，不加载任何训练框架。"""
    processor = Processor()
    splits, _ = pt.prepare_splits(rows, r)
    gold_by_prompt = {
        tuple(Processor._ids(pt.render_prompt(row))): Processor._ids(row["answer"])
        for row in splits["test"]
    }
    parameters = {"base.weight": Parameter(False)}
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True), device="synthetic:0")
    targets = list(r.get("peft.target_modules"))
    model.named_modules = lambda: [(f"layers.0.{name}", object()) for name in targets] + [
        ("vision_tower.0", object())]
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
            "base_model_name_or_path": pt.MODEL, "revision": r.get("model.revision"),
            "r": r.get("peft.r"), "lora_alpha": r.get("peft.alpha"),
            "lora_dropout": r.get("peft.dropout"), "target_modules": list(targets),
            "bias": r.get("peft.bias"), "task_type": r.get("peft.task_type"),
            "use_dora": False, "init_lora_weights": r.get("peft.init_lora_weights"),
        }
        (target / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"synthetic-paligemma-lora-adapter")

    model.save_pretrained = save_pretrained

    def load_dataset(source, **kwargs):
        assert kwargs.get("revision") == DATA_REV and kwargs.get("split") == "train"
        return iter(copy.deepcopy(rows))

    def get_peft_model(model, peft_config):
        for name in list(peft_config.target_modules):
            parameters[f"adapter.{name}.lora_A.weight"] = Parameter(True)
            parameters[f"adapter.{name}.lora_B.weight"] = Parameter(True)
        model.lora_targets = list(peft_config.target_modules)
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

    canonical = pt.config_kwargs(
        r, {"mode": "full", "recipe_id": r.id, "task_id": pt.TASK_ID}, Path("unused"))

    class Trainer:
        def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
            self.state = SimpleNamespace(global_step=args.max_steps)
            self.collator = data_collator

        def train(self):
            item = {"input_ids": [IMAGE_TOKEN_ID, 11, 12],
                    "labels": [-100, -100, 12], "attention_mask": [1, 1, 1],
                    "pixel_values": copy.deepcopy(PIXELS), "pixel_shape": [1, 2, 4]}
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
            device_count=lambda: 1, is_bf16_supported=lambda: True),
        are_deterministic_algorithms_enabled=lambda: True,
    )
    runtime = {
        "torch": fake_torch,
        "transformers": SimpleNamespace(
            __version__="test",
            TrainingArguments=explicit_factory(list(canonical)),
            Trainer=Trainer,
            PaliGemmaProcessor=SimpleNamespace(from_pretrained=lambda source, **kwargs: Processor()),
            PaliGemmaForConditionalGeneration=SimpleNamespace(
                from_pretrained=lambda source, **kwargs: model),
            set_seed=lambda seed: None,
        ),
        "datasets": SimpleNamespace(__version__="test", load_dataset=load_dataset,
                                    Dataset=SimpleNamespace(from_list=lambda items: items)),
        "peft": SimpleNamespace(
            __version__="test", get_peft_model=get_peft_model,
            LoraConfig=explicit_factory([
                "r", "lora_alpha", "lora_dropout", "target_modules",
                "init_lora_weights", "bias", "task_type", "base_model_name_or_path",
                "revision"]),
            get_peft_model_state_dict=state_dict,
            PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        ),
        "safetensors": SimpleNamespace(__version__="test"),
    }
    return runtime, calls, model


def install_fake(monkeypatch, runtime):
    monkeypatch.setattr(pt, "load_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(pt, "require_frozen_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(pt, "apply_determinism", lambda *args, **kwargs: None)
    monkeypatch.setattr(pt, "importlib", SimpleNamespace(
        import_module=lambda name: SimpleNamespace(
            load_file=lambda path, device: runtime["peft"].get_peft_model_state_dict(
                None, "default", False))))


def _plan(r, root, mode):
    raw = build_plan(r, mode)
    raw["run"]["output_dir"] = str(root)
    plan = pt.task_plan(r, raw, mode)
    assert plan["execution_supported"], plan.get("unsupported_reason")
    return plan


@pytest.fixture
def recipe(tmp_path):
    return _recipe(tmp_path)


@pytest.fixture
def rows():
    return make_rows()


@pytest.fixture
def executed(tmp_path, recipe, rows, monkeypatch):
    runtime, calls, model = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    root = tmp_path / "experiment"
    training = pt.run_training(recipe, _plan(recipe, root, "full"))
    evaluation = pt.run_evaluation(
        recipe, _plan(recipe, root, "eval"), checkpoint=training.parent)
    return SimpleNamespace(root=root, training=training, evaluation=evaluation,
                           calls=calls, model=model, runtime=runtime)


# ---------------------------------------------------------------- 一手事实与契约

def test_primary_fact_constants_are_explicit():
    assert pt.MODEL == "google/paligemma2-3b-pt-448"
    assert pt.DATASET == "jxu124/refcoco"
    assert pt.PROMPT_TEMPLATE == "<image>Detect {caption}."
    assert pt.LOCATION_BINS == 1024 and pt.LOCATION_ORDER == ("y1", "x1", "y2", "x2")
    assert pt.METRICS == ("bbox_iou", "iou_at_0_5", "parse_rate", "exact_match")
    assert pt.FULL_STEPS == 150 and pt.SMOKE_STEPS == 5
    assert pt.PALIGEMMA_SPEC["bbox_source_format"] == "coco_xywh"
    assert pt.PALIGEMMA_SPEC["bbox_internal_format"] == "xyxy"
    assert pt.PALIGEMMA_SPEC["upstream_quantitative_evaluation"] is False
    assert pt.PALIGEMMA_SPEC["online_augmentation"] is False
    assert "mAP/AP" in pt.PALIGEMMA_SPEC["metric_definition"]
    assert "bitsandbytes" not in pt.RUNTIME_PACKAGES and "trl" not in pt.RUNTIME_PACKAGES


def test_real_recipe_matches_module_contract(root, recipe):
    real = next(r for r in load_recipes(root) if r.id == pt.RECIPE_ID)
    assert real.get("method.trainer") == "Trainer"
    assert real.get("method.upstream_trainer") == "SFTTrainer"
    assert real.get("execution.task_id") == pt.TASK_ID
    assert real.get("evaluation.metrics") == list(pt.METRICS)
    assert real.get("quantization.mode") == "none"
    assert real.get("peft.use_dora") is False
    assert pt.require_supported(recipe, _plan(recipe, Path("unused"), "dry-run")) == pt.TASK_ID
    assert pt.PALIGEMMA_SPEC == real.get("paligemma")


def test_upstream_fact_strings_are_retained(root):
    real = next(r for r in load_recipes(root) if r.id == pt.RECIPE_ID)
    upstream = real.get("upstream")
    assert "format='coco'" in upstream["bbox_format_mismatch"]
    assert "<seg###>" in upstream["loc_tokens"]
    assert "prompt 部分也参与损失" in upstream["label_mask"]
    assert "Flickr" in upstream["dataset"]
    assert "未给" in upstream["license"]


def test_protocol_drift_is_rejected(recipe):
    for key, value in (("location_bins", 512), ("bbox_source_format", "xyxy"),
                       ("prompt_format", "<image>detect {caption}"),
                       ("round_trip_min_iou", 0.5)):
        broken = copy.deepcopy(recipe.data)
        broken["paligemma"][key] = value
        r = Recipe(recipe.id, recipe.path, broken)
        with pytest.raises(TaskExecutionError, match="paligemma"):
            pt.require_supported(r)
    broken = copy.deepcopy(recipe.data)
    broken["peft"]["use_dora"] = True
    with pytest.raises(TaskExecutionError, match="peft"):
        pt.require_supported(Recipe(recipe.id, recipe.path, broken))
    broken = copy.deepcopy(recipe.data)
    broken["quantization"]["mode"] = "nf4"
    with pytest.raises(TaskExecutionError, match="量化"):
        pt.require_supported(Recipe(recipe.id, recipe.path, broken))


def test_unknown_route_fails_closed(recipe):
    with pytest.raises(TaskExecutionError):
        pt.require_supported(recipe, task_id="granite-linecomparison-v1")
    other = copy.deepcopy(recipe.data)
    other["id"] = "vlm/sft/other"
    with pytest.raises(TaskExecutionError, match="执行路由"):
        pt.require_supported(Recipe("vlm/sft/other", recipe.path, other))


# ---------------------------------------------------------------- bbox / loc

def test_coco_to_xyxy_is_exact_and_strict():
    assert pt.coco_to_xyxy([10, 20, 40, 30], 320, 240) == [10.0, 20.0, 50.0, 50.0]
    for bad in ([0, 0, 0, 30], [0, 0, 40, -1], [315, 0, 40, 30],
                [0, 235, 40, 30], [float("nan"), 0, 40, 30],
                [float("inf"), 0, 40, 30], [True, 0, 40, 30],
                [0, 0, 40], [0, 0, 40, 30, 1], "xyxy", None):
        with pytest.raises(TaskExecutionError):
            pt.coco_to_xyxy(bad, 320, 240)
    for bad in (0, -1, True, 3.5, "320"):
        with pytest.raises(TaskExecutionError):
            pt.coco_to_xyxy([1, 2, 3, 4], bad, 240)


def test_location_tokens_are_1024_bins_and_y1x1y2x2_order():
    assert pt.format_location(0, 320) == "<loc0000>"
    assert pt.format_location(160, 320) == "<loc0512>"
    assert pt.format_location(320, 320) == "<loc1023>"  # 边界 clamp，不生成 <loc1024>
    assert pt.format_location(-5, 320) == "<loc0000>"
    gold = pt.encode_detection([10.0, 20.0, 50.0, 50.0], 320, 240, "a vase")
    tokens = pt.LOC_PATTERN.findall(gold)
    assert len(tokens) == 4
    assert gold.startswith("".join(f"<loc{v}>" for v in tokens))
    assert gold.endswith(" a vase")
    assert tokens[0] == pt.format_location(20, 240)[4:-1]
    assert tokens[1] == pt.format_location(10, 320)[4:-1]


def test_round_trip_never_loses_or_degrades_a_box():
    for bbox in ([0, 0, 320, 240], [1, 1, 20, 10], [10, 12, 40, 30],
                 [200, 100, 100, 120]):
        xyxy = pt.coco_to_xyxy(bbox, 320, 240)
        gold = pt.encode_detection(xyxy, 320, 240, "the left object")
        decoded = pt.decode_detection(gold, 320, 240)
        assert decoded is not None
        assert pt.bbox_iou(xyxy, decoded) >= pt.ROUND_TRIP_MIN_IOU
    assert pt.ROUND_TRIP_MIN_IOU == 0.98


@pytest.mark.parametrize("text", [
    "", "no tokens here", "<loc0001><loc0002><loc0003>",
    "<loc0001><loc0002><loc0003><loc0004> a ; <loc0001><loc0002><loc0003><loc0004> b",
    "<loc1024><loc0002><loc0003><loc0004> a",
    "<loc0500><loc0500><loc0400><loc0400> a",  # x2<x1 / y2<y1
    "<loc0000><loc0000><loc0000><loc0000> a",  # 零面积
])
def test_decode_detection_rejects_invalid_single_box(text):
    assert pt.decode_detection(text, 320, 240) is None


def test_decode_detection_accepts_only_well_formed_single_box():
    decoded = pt.decode_detection("<loc0100><loc0200><loc0300><loc0400> a cat", 320, 240)
    assert decoded == [62.5, 23.4375, 125.0, 70.3125]
    assert pt.decode_detection(None, 320, 240) is None


def test_bbox_iou_is_bounded_and_handles_degenerate_boxes():
    box = [0.0, 0.0, 10.0, 10.0]
    assert pt.bbox_iou(box, box) == pytest.approx(1.0)
    assert pt.bbox_iou(box, [20.0, 20.0, 30.0, 30.0]) == 0.0
    assert 0 < pt.bbox_iou(box, [5.0, 5.0, 15.0, 15.0]) < 1
    assert pt.bbox_iou(box, [0.0, 0.0, 0.0, 0.0]) == 0.0
    assert pt.bbox_iou(box, None) == 0.0
    assert pt.bbox_iou(box, [float("nan"), 0, 1, 1]) == 0.0


def test_caption_injection_is_rejected():
    for bad in ("", "  ", "a <loc0001> box", "a ; b", "line\nbreak", "a>b", "x\x00y"):
        with pytest.raises(TaskExecutionError, match="caption"):
            pt.parse_row({"image": Image(b"i"), "bbox": [1, 2, 40, 30],
                          "caption": bad, "width": 320, "height": 240}, 0)


def test_parse_row_rebuilds_gold_and_never_puts_it_in_prompt():
    row = pt.parse_row({"image": Image(b"img"), "bbox": [10, 20, 40, 30],
                        "caption": "the middle vase", "width": 320, "height": 240}, 7)
    assert row["source_index"] == 7
    assert row["bbox_xyxy"] == [10.0, 20.0, 50.0, 50.0]
    assert row["answer"] == pt.encode_detection(row["bbox_xyxy"], 320, 240, "the middle vase")
    assert row["answer"] not in pt.render_prompt(row)
    assert pt.render_prompt(row) == "<image>Detect the middle vase."
    assert row["round_trip_iou"] >= pt.ROUND_TRIP_MIN_IOU
    assert len(row["image_sha256"]) == 64


def test_parse_row_rejects_missing_extra_columns_and_bad_boxes():
    good = {"image": Image(b"img"), "bbox": [10, 20, 40, 30],
            "caption": "a bear", "width": 320, "height": 240}
    with pytest.raises(TaskExecutionError, match="缺列"):
        pt.parse_row({k: v for k, v in good.items() if k != "height"}, 0)
    with pytest.raises(TaskExecutionError, match="未知列"):
        pt.parse_row({**good, "score": 1.0}, 0)
    with pytest.raises(TaskExecutionError, match="越界"):
        pt.parse_row({**good, "bbox": [300, 200, 100, 100]}, 0)


# ---------------------------------------------------------------- 划分与去重

def test_splits_are_image_isolated_and_non_empty(recipe, rows):
    splits, manifest = pt.prepare_splits(rows, recipe)
    assert all(splits[name] for name in ("train", "validation", "test"))
    owners = {}
    for name, items in splits.items():
        for row in items:
            owners.setdefault(row["image_sha256"], set()).add(name)
    assert all(len(value) == 1 for value in owners.values())
    assert manifest["counts"]["source"] == len(rows)
    assert manifest["location_bins"] == 1024
    assert manifest["prompt_format"] == pt.PROMPT_TEMPLATE
    assert pt.validate_source_manifest(manifest, recipe) == manifest


def test_duplicate_content_is_rejected_not_silently_deduped(recipe, rows):
    duplicate = copy.deepcopy(rows[0])
    with pytest.raises(TaskExecutionError, match="重复"):
        pt.prepare_splits(rows + [duplicate], recipe)


def test_empty_split_is_rejected(recipe):
    rows = [{"image": Image(b"same"), "bbox": [1 + i, 2, 20, 20],
             "caption": f"object {i}", "width": 320, "height": 240} for i in range(6)]
    with pytest.raises(TaskExecutionError, match="split 为空"):
        pt.prepare_splits(rows, recipe)


def test_split_manifest_rejects_tampering(recipe, rows):
    _, manifest = pt.prepare_splits(rows, recipe)
    for mutation in (
        lambda m: m.update(sha256="0" * 64),
        lambda m: m["split_hashes"]["test"].append(m["split_hashes"]["train"][0]),
        lambda m: m.update(location_bins=512),
        lambda m: m.update(prompt_format="<image>detect {caption}"),
        lambda m: m["counts"].update(test=999999),
    ):
        broken = copy.deepcopy(manifest)
        mutation(broken)
        with pytest.raises(TaskExecutionError):
            pt.validate_source_manifest(broken, recipe)


# ---------------------------------------------------------------- 图文编码与 mask

def test_processor_identity_uses_explicit_image_token_and_records_loc_protocol():
    identity = pt.processor_identity(Processor())
    assert identity["image_token_id"] == IMAGE_TOKEN_ID
    assert identity["location_bins"] == 1024
    assert identity["location_token_order"] == "y1,x1,y2,x2"
    assert identity["prompt_template_sha256"] == hashlib.sha256(
        pt.PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    assert identity["sha256"] == pt._hash_manifest(identity)["sha256"]


def test_processor_identity_falls_back_to_additional_special_tokens():
    class NoAttr(Processor):
        image_token_id = None
    identity = pt.processor_identity(NoAttr())
    assert identity["image_token_id"] == IMAGE_TOKEN_ID
    class Nothing(Processor):
        image_token_id = None
        def __init__(self):
            self.tokenizer = SimpleNamespace(
                name_or_path="x", vocab_size=1, eos_token=EOS, eos_token_id=EOS_ID,
                pad_token=EOS, pad_token_id=EOS_ID, chat_template="",
                additional_special_tokens=[], additional_special_tokens_ids=[])
    with pytest.raises(TaskExecutionError, match="image_token_id"):
        pt.processor_identity(Nothing())


def test_encode_example_enforces_prompt_prefix_and_double_mask(recipe, rows):
    processor = Processor()
    identity = pt.processor_identity(processor)
    row = pt.parse_row(rows[0], 0)
    item = pt.encode_example(row, processor, identity, recipe)
    assert item["input_ids"][:item["prompt_tokens"]] == item["prompt_ids"]
    assert item["target_tokens"] > 0
    assert item["masked_image_tokens"] == 1
    assert all(label == -100 for label in item["labels"][:item["prompt_tokens"]])
    assert all(label == -100 for token, label in zip(item["input_ids"], item["labels"])
               if token == IMAGE_TOKEN_ID)
    assert any(label != -100 for label in item["labels"][item["prompt_tokens"]:])
    assert item["gold_sha256"] == pt.sha256_json(row["answer"])


def test_encode_example_rejects_gold_leak_and_missing_suffix(recipe, rows):
    processor = Processor()
    identity = pt.processor_identity(processor)
    row = pt.parse_row(rows[0], 0)
    leaky = {**row, "caption": row["answer"]}
    with pytest.raises(TaskExecutionError, match="泄漏"):
        pt.encode_example(leaky, processor, identity, recipe)

    class NoSuffix(Processor):
        def __call__(self, text, images, return_tensors=None, padding=False, suffix=None):
            return super().__call__(text, images, return_tensors, padding, suffix=None)
    with pytest.raises(TaskExecutionError, match="边界"):
        pt.encode_example(row, NoSuffix(), identity, recipe)

    class BadPrefix(Processor):
        def __call__(self, text, images, return_tensors=None, padding=False, suffix=None):
            result = super().__call__(text, images, return_tensors, padding, suffix)
            if suffix is not None:
                result["input_ids"][0][0] = 999
            return result
    with pytest.raises(TaskExecutionError, match="前缀"):
        pt.encode_example(row, BadPrefix(), identity, recipe)


def test_train_examples_transform_manifest_is_content_bound(recipe, rows):
    splits, manifest = pt.prepare_splits(rows, recipe)
    processor = Processor()
    identity = pt.processor_identity(processor)
    examples, transform = pt.train_examples(splits, processor, recipe)
    assert set(examples) == {"train", "validation"} and all(examples.values())
    assert pt.validate_transform_manifest(transform, manifest, recipe, identity) == transform
    assert transform["prompt_format"] == pt.PROMPT_TEMPLATE
    assert transform["location_bins"] == 1024
    broken = copy.deepcopy(transform)
    broken["location_bins"] = 512
    with pytest.raises(TaskExecutionError, match="协议"):
        pt.validate_transform_manifest(broken, manifest, recipe, identity)


def test_collate_masks_image_tokens_and_requires_same_pixel_shape(recipe, rows):
    torch = SimpleNamespace(tensor=lambda values, **kwargs: Tensor(values, kwargs.get("dtype")),
                            long="long", float32="float32")
    items = [{"input_ids": [IMAGE_TOKEN_ID, 11, 12], "labels": [-100, 11, 12],
              "attention_mask": [1, 1, 1], "pixel_values": copy.deepcopy(PIXELS),
              "pixel_shape": [1, 2, 4]} for _ in range(2)]
    batch = pt.collate(items, torch, EOS_ID, IMAGE_TOKEN_ID)
    assert batch["labels"].values[0][0] == -100
    assert batch["labels"].values[0][1] == EOS_ID or batch["labels"].values[0][1] == 11
    with pytest.raises(TaskExecutionError, match="形状"):
        pt.collate([{**items[0], "pixel_shape": [9, 9]}, items[0]],
                   torch, EOS_ID, IMAGE_TOKEN_ID)


# ---------------------------------------------------------------- 评分

def _row():
    return pt.parse_row({"image": Image(b"img"), "bbox": [10, 20, 40, 30],
                         "caption": "the middle vase", "width": 320, "height": 240}, 0)


def test_score_prediction_exact_unparsed_and_empty_paths():
    row = _row()
    exact = pt.score_prediction(row, row["answer"] + EOS, EOS)
    assert exact["exact_match"] == 1.0 and exact["parse_rate"] == 1.0
    assert exact["bbox_iou"] == pytest.approx(1.0, abs=0.02)
    assert exact["iou_at_0_5"] == 1.0 and exact["empty"] == 0.0
    assert exact["gold_sha256"] == pt.sha256_json(row["answer"])
    unparsed = pt.score_prediction(row, "I cannot see it" + EOS, EOS)
    assert unparsed["parse_rate"] == 0.0 and unparsed["bbox_iou"] == 0.0
    assert unparsed["iou_at_0_5"] == 0.0 and unparsed["exact_match"] == 0.0
    assert unparsed["parsed_bbox_xyxy"] is None
    empty = pt.score_prediction(row, EOS, EOS)
    assert empty["empty"] == 1.0 and empty["parse_rate"] == 0.0
    trailing = pt.score_prediction(row, row["answer"] + EOS + "ignored", EOS)
    assert trailing["exact_match"] == 1.0


def test_score_prediction_iou_at_threshold_and_multi_box_rejection():
    row = _row()
    partial = "<loc0064><loc0026><loc0179><loc0160> the middle vase"
    score = pt.score_prediction(row, partial + EOS, EOS)
    assert score["parse_rate"] == 1.0
    assert 0 < score["bbox_iou"] < 1
    assert score["exact_match"] == 0.0
    multi = row["answer"] + " ; " + row["answer"]
    assert pt.score_prediction(row, multi + EOS, EOS)["parse_rate"] == 0.0


def test_recompute_metrics_is_strict():
    row = _row()
    records = [pt.score_prediction(row, row["answer"] + EOS, EOS),
               pt.score_prediction(row, "nonsense" + EOS, EOS)]
    metrics = pt.recompute_metrics(records)
    assert set(metrics) == set(pt.METRICS)
    assert metrics["exact_match"] == pytest.approx(0.5)
    assert metrics["parse_rate"] == pytest.approx(0.5)
    with pytest.raises(TaskExecutionError):
        pt.recompute_metrics([])
    with pytest.raises(TaskExecutionError):
        pt.recompute_metrics(["not a dict"])
    broken = copy.deepcopy(records)
    broken[0]["bbox_iou"] = float("nan")
    with pytest.raises(TaskExecutionError):
        pt.recompute_metrics(broken)
    broken = copy.deepcopy(records)
    broken[0]["bbox_iou"] = 1.5
    with pytest.raises(TaskExecutionError):
        pt.recompute_metrics(broken)


def test_metrics_never_claim_map_or_business_acceptance(root):
    real = next(r for r in load_recipes(root) if r.id == pt.RECIPE_ID)
    assert real.get("paligemma.metric_definition").find("mAP/AP") >= 0
    assert "不等于" in real.get("paligemma.metric_definition")


# ---------------------------------------------------------------- 计划与路由

def test_task_plan_declares_protocol_and_no_gpu_claim(recipe, tmp_path):
    plan = _plan(recipe, tmp_path / "out", "dry-run")
    assert plan["task_id"] == pt.TASK_ID
    assert plan["evaluation"]["protocol"] == pt.EVAL_PROTOCOL
    assert plan["evaluation"]["do_sample"] is False
    assert plan["evaluation"]["training_seed_variance_measured"] is False
    assert plan["evaluation"]["upstream_quantitative_evaluation"] is False
    assert "suffix-only" in plan["migration"]["mask"]
    assert plan["dataset"]["split_plan"]["gold_policy"].startswith("loc detection")


def test_task_router_resolves_exactly_and_rejects_conflicts(recipe):
    assert tr.resolve(recipe) == pt.TASK_ID
    assert tr.load(recipe, "task") is pt
    from recipe_lab import paligemma_suite as ps
    from recipe_lab import paligemma_export as pe
    assert tr.load(recipe, "suite") is ps
    assert tr.load(recipe, "export") is pe
    broken = copy.deepcopy(recipe.data)
    broken["execution"]["task_id"] = "granite-linecomparison-v1"
    with pytest.raises(TaskExecutionError, match="冲突"):
        tr.resolve(Recipe(recipe.id, recipe.path, broken))


def test_other_recipes_are_not_rerouted_to_paligemma(root):
    others = [r for r in load_recipes(root) if r.id != pt.RECIPE_ID]
    assert others
    for r in others:
        assert tr.resolve(r) != pt.TASK_ID


def test_fingerprint_binds_paligemma_protocol(recipe):
    changed = copy.deepcopy(recipe.data)
    changed["paligemma"]["location_bins"] = 512
    other = Recipe(recipe.id, recipe.path, changed)
    assert be.fingerprint(recipe) != be.fingerprint(other)
    changed = copy.deepcopy(recipe.data)
    changed["upstream"]["dataset"] = "tampered"
    assert be.fingerprint(recipe) != be.fingerprint(
        Recipe(recipe.id, recipe.path, changed))
    assert be.frozen_environment(recipe) is True


def test_baseline_evidence_dispatches_paligemma(executed, recipe):
    data = json.loads(executed.evaluation.read_text(encoding="utf-8"))
    problems = be.evaluation_problems(recipe, data, executed.evaluation, role="checkpoint")
    assert problems == [], problems
    assert be.zero_train_measured(recipe) is False
    assert be.suite_problems(recipe)


# ---------------------------------------------------------------- 端到端合成执行

def test_run_training_writes_verifiable_checkpoint(executed, recipe):
    training = json.loads(executed.training.read_text(encoding="utf-8"))
    assert training["kind"] == "training" and training["stage"] == "completed"
    assert training["task_id"] == pt.TASK_ID and training["mode"] == "full"
    assert training["steps"] == training["max_steps"] == pt.FULL_STEPS
    assert training["fingerprint"] == be.fingerprint(recipe)
    assert training["config"]["lora_execution"]["use_dora"] is False
    assert training["config"]["lora_execution"]["tensor_reload_check"]["strict_values_verified"]
    assert training["config"]["prompt_template"] == pt.PROMPT_TEMPLATE
    assert training["processor_identity"]["image_token_id"] == IMAGE_TOKEN_ID
    assert training["transformation_manifest"]["entries"]["train"]
    assert pt.checkpoint_path(_plan(recipe, executed.root, "eval"),
                              checkpoint=executed.training.parent,
                              recipe=recipe) == executed.training.parent


def test_checkpoint_path_rejects_tampering(executed, recipe):
    checkpoint = executed.training.parent
    plan = _plan(recipe, executed.root, "eval")
    original = (checkpoint / "training.json").read_bytes()
    for mutation in (
        lambda d: d.update(seed=999),
        lambda d: d.update(recipe_sha256="0" * 64),
        lambda d: d["config"]["lora_execution"].update(use_dora=True),
        lambda d: d["config"].update(prompt_template="<image>detect {caption}"),
    ):
        data = json.loads(original)
        mutation(data)
        (checkpoint / "training.json").write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(TaskExecutionError):
            pt.checkpoint_path(plan, checkpoint=checkpoint, recipe=recipe)
    (checkpoint / "training.json").write_bytes(original)


def test_run_evaluation_recomputes_metrics_from_generated_text(executed, recipe):
    data = json.loads(executed.evaluation.read_text(encoding="utf-8"))
    assert data["model_role"] == "checkpoint"
    assert data["evaluation_protocol"] == pt.EVAL_PROTOCOL
    assert data["prompt_protocol"] == pt.PROMPT_TEMPLATE
    assert set(data["metrics"]) == set(pt.METRICS)
    assert data["metrics"]["parse_rate"] == pytest.approx(1.0)
    assert data["metrics"]["exact_match"] == pytest.approx(1.0)
    assert data["metrics"]["bbox_iou"] >= 0.98
    assert data["sample_count"] == len(
        json.loads((executed.evaluation.parent / pt.PREDICTIONS_FILE).read_text(
            encoding="utf-8"))["predictions"])
    manifest = json.loads((executed.evaluation.parent / pt.EVAL_MANIFEST_FILE).read_text(
        encoding="utf-8"))
    assert manifest["kind"] == "paligemma-eval-manifest"
    assert manifest["prompt_format"] == pt.PROMPT_TEMPLATE
    assert manifest["location_bins"] == 1024
    assert all("gold_sha256" in s and "bbox_coco" in s and "round_trip_iou" in s
               for s in manifest["samples"])
    assert executed.calls["adapter_load"] == 1
    assert executed.calls["generate"] == data["sample_count"]


def test_evaluation_predictions_are_bound_to_raw_gold(executed, recipe):
    from recipe_lab.export_task import _json
    path = executed.evaluation
    data = _json(path)
    from recipe_lab import paligemma_suite as ps
    predictions, _, manifest, _ = ps._evaluation_content(
        recipe, path, data, data["split_manifest"])
    assert len(predictions) == len(manifest["samples"])
    assert all(p["gold_sha256"] == pt.sha256_json(
        pt.encode_detection(s["bbox_xyxy"], s["width"], s["height"], s["caption"]))
        for p, s in zip(predictions, manifest["samples"]))


def test_zero_shot_uses_base_only(recipe, rows, tmp_path, monkeypatch):
    runtime, calls, model = fake_runtime(recipe, rows)
    install_fake(monkeypatch, runtime)
    root = tmp_path / "zero"
    path = pt.run_evaluation(recipe, _plan(recipe, root, "zero-shot"), zero_shot=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model_role"] == "base" and data["model_artifact"] == pt.base_artifact(recipe)
    assert calls["adapter_load"] == 0
    assert data["checkpoint_training_evidence"] is None


# ---------------------------------------------------------------- CLI 与失败关闭

def test_release_dry_runs_dispatch_to_paligemma_and_full_fails_closed(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "RECIPE_LAB_ROOT": str(ROOT)}
    for script, extra in (("suite.py", ["--seeds", "42", "43", "44",
                                        "--output-root", str(tmp_path / "suite")]),
                          ("export.py", ["--checkpoint", "missing", "--evaluation", "missing",
                                         "--output", str(tmp_path / "bundle")])):
        run = subprocess.run(
            [sys.executable, str(ROOT / "runner" / script), "--recipe", pt.RECIPE_ID,
             *extra, "--dry-run"], env=env, capture_output=True, text=True, check=False)
        assert run.returncode == 0, run.stderr
        data = json.loads(run.stdout)
        assert data["task_id"] == pt.TASK_ID
        assert "unavailable" not in data.get("release_pipeline", "")
    for script, extra in (("train.py", ["--output-root", str(tmp_path / "train")]),
                          ("suite.py", ["--seeds", "42", "43", "44",
                                        "--output-root", str(tmp_path / "full")])):
        run = subprocess.run(
            [sys.executable, str(ROOT / "runner" / script), "--recipe", pt.RECIPE_ID,
             *extra, "--full"], env=env, capture_output=True, text=True, check=False)
        assert run.returncode == 1, (script, run.stderr[-300:])
    export = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "export.py"), "--recipe", pt.RECIPE_ID,
         "--checkpoint", "missing", "--evaluation", "missing",
         "--output", str(tmp_path / "export-full"), "--full"],
        env=env, capture_output=True, text=True, check=False)
    assert export.returncode == 1
    assert not (tmp_path / "full").exists() and not (tmp_path / "export-full").exists()


def test_release_pipeline_dry_run_never_claims_gpu_evidence(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "RECIPE_LAB_ROOT": str(ROOT)}
    run = subprocess.run(
        [sys.executable, str(ROOT / "runner" / "suite.py"), "--recipe", pt.RECIPE_ID,
         "--seeds", "42", "43", "44", "--output-root", str(tmp_path / "p"), "--dry-run"],
        env=env, capture_output=True, text=True, check=False)
    data = json.loads(run.stdout)
    assert data["status"] == "declared"
    assert data["baseline_promotion"] == "none"
    assert data["resume_supported"] is False
    assert data["training_source"]["quantization"] == "none"
