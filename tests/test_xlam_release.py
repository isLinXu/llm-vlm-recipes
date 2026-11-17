"""xLAM 发布级 comparison / export 的隔离离线契约测试。

用 fake runtime 调用真实 xlam_task.run_training / run_evaluation 生成三份独立训练 seed
的合成证据，再验证：
  * 三 seed 套件的严格 JSON / tool schema 逐题内容重算与 Student-t95 聚合；
  * C306 / C307 按 task_router 分派到 xlam_suite，不借用数学或 FIM 协议；
  * adapter-only 导出包、冻结 base 声明与「新进程」offline 复载（C405）。

所有产物只写 tmp_path；JSON/hash 只证明内部一致性，不构成 GPU 实跑证据，
也不自动晋级 B2/B3。fake runtime 明确是合成夹具，不声明任何真实模型行为。
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipe_lab import baseline as bl
from recipe_lab import baseline_evidence as be
from recipe_lab import export_task as et
from recipe_lab import provenance as pv
from recipe_lab import suite_task as st
from recipe_lab import task_router as tr
from recipe_lab import xlam_export as xe
from recipe_lab import xlam_suite as xs
from recipe_lab import xlam_task as xt
from recipe_lab.math_task import TaskExecutionError
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan
from test_export import read

SEEDS = [42, 43, 44]
LOCKFILE = "frozen.lock.toml"


# ---------------------------------------------------------------- 合成配置与 fake runtime

def _recipe(tmp_path):
    data = {
        "id": xt.RECIPE_ID, "source_sha": "c" * 40,
        "method": {"family": "sft", "trainer": "Trainer",
                   "upstream_trainer": "SFTTrainer", "peft": "qlora"},
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
            "seed": 0, "data_seed": 0, "seed_mechanism": "上游 TrainingConfig/SFTConfig 无 seed",
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
        # export_tolerance 是导出容差正式入口；expected/tolerance 刻意不声明，
        # 套件不得回填 expected 或自证通过。
        "evaluation": {"metrics": list(xt.METRICS), "seed": 0, "max_new_tokens": 256,
                       "multi_run": True, "held_out": True, "split": "content-hash 80/10/10",
                       "export_tolerance": {name: 0.0 for name in xt.METRICS}},
        "execution": {"task_id": xt.TASK_ID, "lockfile": LOCKFILE},
        "baseline": {"result": "unmeasured"},
    }
    lock = "\n".join([
        "schema_version = 1", 'status = "frozen"', f'recipe_id = "{xt.RECIPE_ID}"',
        f'model_id = "{xt.MODEL}"', f'dataset_id = "{xt.DATASET}"',
        f'model_revision = "{"a" * 40}"', f'dataset_revision = "{"b" * 40}"',
        f'source_sha = "{"c" * 40}"',
        'requirements = ["torch==test", "transformers==test", "datasets==test", "peft==test", '
        '"bitsandbytes==test", "safetensors==test"]',
        "[runtime]", f'python = "{sys.version.split()[0]}"', 'cuda = "test-cuda"',
        'driver = "test-driver"', 'gpu = "test-device"',
    ]) + "\n"
    (tmp_path / LOCKFILE).write_text(lock, encoding="utf-8")
    (tmp_path / "recipe.toml").write_text(f'id = "{xt.RECIPE_ID}"\n', encoding="utf-8")
    return Recipe(xt.RECIPE_ID, tmp_path / "recipe.toml", data)


def make_rows(count=180):
    rows = []
    for index in range(count):
        tools = [{"name": f"tool_{index}", "description": "synthetic tool",
                  "parameters": {"amount": {"type": "int", "description": "n", "required": True},
                                 "label": {"type": "str", "description": "s", "required": False}}}]
        answers = [{"name": f"tool_{index}", "arguments": {"amount": index}}]
        rows.append({"id": index, "query": f"Synthetic query number {index}",
                     "tools": json.dumps(tools), "answers": json.dumps(answers)})
    return rows


class Tokenizer:
    name_or_path = "synthetic/xlam-tokenizer"
    vocab_size = 256
    eos_token = "</s>"
    eos_token_id = 2
    pad_token = "</s>"
    pad_token_id = 2

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


def explicit_factory(keys):
    def factory(**kwargs):
        return SimpleNamespace(**kwargs)
    factory.__signature__ = inspect.Signature([
        inspect.Parameter(key, inspect.Parameter.KEYWORD_ONLY) for key in keys
    ])
    return factory


def fake_runtime(r, rows):
    """确定性合成 runtime：只实现 xlam_task 显式检查过的 API，不加载任何训练框架。"""
    tokenizer = Tokenizer()
    splits, _ = xt.prepare_splits(rows, r)
    gold_by_prompt = {
        tuple(Tokenizer._ids(xt.render_prompt(row))):
            Tokenizer._ids(f"<calls>{row['answers']}</calls>")
        for row in splits["test"]
    }
    parameters = {"base.weight": Parameter()}
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True), device="synthetic:0")
    model.named_modules = lambda: [(f"layer.{name}", object()) for name in xt.TARGET_MODULES]
    model.named_parameters = lambda: list(parameters.items())
    model.parameters = lambda: list(parameters.values())
    model.eval = lambda: None

    def generate(input_ids, attention_mask, **kwargs):
        prompt = input_ids.rows[0]
        assert set(kwargs) == {"max_new_tokens", "do_sample", "num_beams",
                               "pad_token_id", "eos_token_id"}
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        gold = gold_by_prompt[tuple(prompt)]
        return [prompt + gold[:kwargs["max_new_tokens"]]]
    model.generate = generate

    def save_model(directory, safe_serialization):
        assert safe_serialization is True
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        adapter = {"base_model_name_or_path": xt.MODEL, "revision": r.get("model.revision"),
                   "r": 16, "lora_alpha": 16, "lora_dropout": 0.05,
                   "target_modules": list(xt.TARGET_MODULES), "bias": "none",
                   "task_type": "CAUSAL_LM", "peft_type": "LORA"}
        (target / "adapter_config.json").write_text(json.dumps(adapter), encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"synthetic-adapter-weights")
    model.save_pretrained = save_model

    def load_tokenizer(source, **kwargs):
        return Tokenizer()

    def load_model(source, **kwargs):
        return model

    def load_dataset(source, **kwargs):
        return iter(copy.deepcopy(rows))

    def dataset_from_list(items):
        return items

    def prepare(model, use_gradient_checkpointing):
        assert use_gradient_checkpointing is True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def get_peft_model(model, peft_config):
        parameters["adapter.lora_A.weight"] = Parameter(True)
        return model

    def load_adapter(model, model_id, is_trainable):
        assert is_trainable is False
        return model

    def state_dict(model, adapter_name, save_embedding_layers):
        return {"base_model.model.layer.q_proj.lora_A.weight": Tensor([1.0, 2.0]),
                "base_model.model.layer.q_proj.lora_B.weight": Tensor([3.0, 4.0])}

    canonical = xt.config_kwargs(r, {"mode": "full"}, Path("unused"))

    class Trainer:
        def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
            self.state = SimpleNamespace(global_step=args.max_steps)

        def train(self):
            return SimpleNamespace(metrics={"train_loss": 0.25})

    runtime = {
        "torch": SimpleNamespace(
            __version__="test", bfloat16="test-bfloat16", long="test-long",
            tensor=lambda values, **kwargs: Tensor(values),
            isfinite=lambda tensor: Tensor([True] * len(tensor.values)),
            equal=lambda a, b: a.values == b.values,
            no_grad=contextlib.nullcontext,
            version=SimpleNamespace(cuda="test-cuda"),
            cuda=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda _: "test-device",
                                 max_memory_allocated=lambda: 0),
            are_deterministic_algorithms_enabled=lambda: True,
        ),
        "transformers": SimpleNamespace(
            __version__="test", TrainingArguments=explicit_factory(list(canonical)), Trainer=Trainer,
            BitsAndBytesConfig=explicit_factory([
                "load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype",
                "bnb_4bit_use_double_quant"]),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        ),
        "datasets": SimpleNamespace(__version__="test", load_dataset=load_dataset,
                                    Dataset=SimpleNamespace(from_list=dataset_from_list)),
        "peft": SimpleNamespace(
            __version__="test", prepare_model_for_kbit_training=prepare,
            get_peft_model=get_peft_model,
            LoraConfig=explicit_factory(["r", "lora_alpha", "lora_dropout", "target_modules",
                                         "bias", "task_type", "base_model_name_or_path", "revision"]),
            get_peft_model_state_dict=state_dict,
            PeftModel=SimpleNamespace(from_pretrained=load_adapter),
        ),
        "bitsandbytes": SimpleNamespace(__version__="test"),
        "safetensors": SimpleNamespace(__version__="test"),
    }
    return runtime


def install_fake(monkeypatch, runtime):
    monkeypatch.setattr(xt, "load_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(xt, "require_frozen_runtime", lambda *args: None)
    monkeypatch.setattr(xt, "apply_determinism", lambda *args, **kwargs: None)
    monkeypatch.setattr(xt, "importlib", SimpleNamespace(
        import_module=lambda name: SimpleNamespace(
            load_file=lambda path, device: runtime["peft"].get_peft_model_state_dict(
                None, "default", False))))


def _plan(r, root, mode):
    raw = build_plan(r, mode)
    raw["run"]["output_dir"] = str(root)
    plan = xt.task_plan(r, raw, mode)
    assert plan["execution_supported"], plan.get("unsupported_reason")
    return plan


def _freeze(r, root):
    root.mkdir(parents=True, exist_ok=True)
    (root / st.CODE_SNAPSHOT).write_bytes(pv.snapshot_bytes(pv.code_snapshot()))
    (root / st.LOCKFILE).write_bytes((r.path.parent / LOCKFILE).read_bytes())


@pytest.fixture
def fcase(tmp_path, monkeypatch):
    """三 seed 合成训练/评估证据 + 冻结 root；只写 tmp_path，不加载真实框架。"""
    r = _recipe(tmp_path)
    root = tmp_path / "suite-root"
    _freeze(r, root)
    rows = make_rows()
    install_fake(monkeypatch, fake_runtime(r, rows))
    base_path = xt.run_evaluation(r, _plan(r, root / "base", "zero-shot"), zero_shot=True)
    checkpoints, evaluations = {}, {}
    for seed in SEEDS:
        effective = pv.effective_recipe(r, seed)
        install_fake(monkeypatch, fake_runtime(effective, rows))
        training = xt.run_training(effective, _plan(effective, root / f"seed-{seed}", "full"))
        install_fake(monkeypatch, fake_runtime(effective, rows))
        evaluation = xt.run_evaluation(
            effective, _plan(effective, root / f"seed-{seed}", "eval"),
            checkpoint=training.parent, zero_shot=False)
        checkpoints[seed] = training.parent
        evaluations[seed] = evaluation
    return SimpleNamespace(r=r, root=root, base=base_path, checkpoints=checkpoints,
                           evaluations=evaluations, lock=tmp_path / LOCKFILE)


# ---------------------------------------------------------------- comparison / 发布门禁

def test_router_and_baseline_evidence_dispatch_to_xlam_release_modules(fcase):
    r = fcase.r
    assert tr.resolve(r) == xt.TASK_ID
    assert tr.load(r, "suite") is xs and tr.load(r, "export") is xe
    base = read(fcase.base)
    assert be.evaluation_problems(r, base, fcase.base, role="base") == []
    assert be.zero_train_measured(_with_measurement(r, fcase)) is True
    assert any("baseline-suite" in p for p in be.suite_problems(r))


def _with_measurement(r, fcase):
    data = copy.deepcopy(r.data)
    data["baseline"]["measurement"] = str(fcase.base.parent.relative_to(r.path.parent) / fcase.base.name)
    return Recipe(r.id, r.path, data)


def test_suite_plan_is_readonly_and_xlam_specific(fcase):
    before = {p: p.stat().st_mtime_ns for p in fcase.root.rglob("*") if p.is_file()}
    plan = xs.suite_plan(fcase.r, SEEDS, fcase.root)
    assert plan["status"] == "declared" and plan["task_id"] == xt.TASK_ID
    assert plan["seeds"] == SEEDS and len(plan["commands"]) == 7
    assert plan["protocol"] == xt.EVAL_PROTOCOL and plan["metrics"] == list(xt.METRICS)
    assert plan["full_steps"] == 150 and plan["training_source"]["adapter_only"] is True
    assert plan["shell"] is False and plan["baseline_promotion"] == "none"
    assert not plan["resume_supported"] and "不执行生成的调用" in plan["limitations"]
    assert {p: p.stat().st_mtime_ns for p in fcase.root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("seeds", [[], [1, 1, 2], [1, 2, True], [1, 2, 2**32]])
@pytest.mark.parametrize("operation", [xs.suite_plan, xs.aggregate_suite, xs.run_suite])
def test_invalid_seed_lists_fail_without_writes(fcase, seeds, operation):
    root = fcase.root.parent / "unused-root"
    with pytest.raises(TaskExecutionError):
        operation(fcase.r, seeds, root)
    assert not root.exists()


def test_aggregate_suite_recomputes_three_independent_seeds(fcase):
    result = xs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    assert result == fcase.root / xs.FINAL
    data = read(result)
    assert data["kind"] == "baseline-suite" and data["status"] == "completed"
    assert data["task_id"] == xt.TASK_ID and data["baseline_promotion"] == "none"
    assert [run["train_seed"] for run in data["runs"]] == SEEDS
    assert len({run["training_evidence"]["path"] for run in data["runs"]}) == 3
    assert data["statistical_unit"] == "independent_training_seed" and data["fixed_test"] is True
    assert data["training_seed_variance_measured"] is True
    for name in xs.METRICS:
        stats = data["stats"][name]
        assert stats["trained"]["n"] == 3 and stats["paired_delta"]["n"] == 3
        assert stats["trained"]["student_t95_ci"]["df"] == 2
    assert data["acceptance"]["status"] == "not_evaluated"  # 不回填 expected/tolerance
    assert xs.validate_suite_payload(fcase.r, data, result) is None


@pytest.mark.parametrize("change", ["stats", "fingerprint", "runs", "task_id",
                                    "base_metrics", "records", "evaluation_protocol"])
def test_suite_payload_rejects_tampered_summary(fcase, change):
    result = xs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    data = read(result)
    if change == "stats":
        data["stats"]["json_parse_rate"]["trained"]["mean"] = 0.123
    elif change == "fingerprint":
        data["fingerprint"] = "0" * 64
    elif change == "runs":
        data["runs"] = data["runs"][:2]
    elif change == "task_id":
        data["task_id"] = "math-grpo-v1"
    elif change == "base_metrics":
        data["runs"][0]["base_metrics"]["json_parse_rate"] = 0.5
    elif change == "evaluation_protocol":
        data["base_evaluation"] = {**data["base_evaluation"], "sha256": "0" * 64}
    else:
        key = next(iter(data["records"]))
        data["records"][key]["sha256"] = "0" * 64
    with pytest.raises(TaskExecutionError):
        xs.validate_suite_payload(fcase.r, data, result)


def test_suite_payload_rejects_checkpoint_weight_drift(fcase):
    result = xs.aggregate_suite(fcase.r, SEEDS, fcase.root)
    (fcase.checkpoints[43] / "adapter_model.safetensors").write_bytes(b"drifted adapter")
    with pytest.raises(TaskExecutionError):
        xs.validate_suite_payload(fcase.r, read(result), result)


def test_evaluation_content_recomputes_strict_json_score_and_rejects_tampering(fcase):
    base = read(fcase.base)
    assert xs.evaluation_problems(fcase.r, base, fcase.base, role="base") == []
    assert xs.zero_train_validation(fcase.r, base, fcase.base) is None
    # 摘要分数被替换，但生成文本没变：逐题重算必须拒绝。
    swapped = copy.deepcopy(base)
    swapped["metric_counts"]["tool_call_exact_match"] = 999
    assert xs.evaluation_problems(fcase.r, swapped, fcase.base, role="base")
    # eval_manifest 的 raw.tools 被改成别的工具定义：prompt/gold 重算必须拒绝。
    manifest_path = fcase.base.parent / xt.EVAL_MANIFEST_FILE
    manifest = read(manifest_path)
    manifest["samples"][0]["raw"]["tools"] = json.dumps(
        [{"name": "other", "description": "x", "parameters": {}}])
    (fcase.base.parent / xt.EVAL_MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    assert xs.evaluation_problems(fcase.r, base, fcase.base, role="base")
    shutil.copyfile(fcase.evaluations[42].parent / xt.EVAL_MANIFEST_FILE,
                    fcase.base.parent / xt.EVAL_MANIFEST_FILE)
    assert xs.evaluation_problems(fcase.r, base, fcase.base, role="base") == []


def test_suite_problems_fails_closed_without_completed_evidence(fcase):
    problems = xs.suite_problems(fcase.r)
    assert problems and "baseline-suite" in problems[0]
    assert be.suite_problems(fcase.r) == problems


# ---------------------------------------------------------------- export / offline 复载

def fake_reload(r, plan, checkpoint=None, zero_shot=False):
    """测试专用：把包内参考证据落到 worker 固定输出目录，不加载 GPU 框架。"""
    bundle = Path(checkpoint).parent
    output = Path(plan["run"]["output_dir"])
    assert output == bundle / "reload" / "eval"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation.json", "source_manifest.json", xt.TOKENIZER_IDENTITY_FILE,
                 xt.EVAL_MANIFEST_FILE, xt.PREDICTIONS_FILE):
        shutil.copyfile(bundle / "reference" / name, output / name)
    return output / "evaluation.json"


def mock_worker(monkeypatch, destination):
    child_pid, parent_pid = os.getpid() + 10000, os.getpid()
    calls = []
    monkeypatch.setattr(xt, "run_evaluation", fake_reload)

    def launch(command, **kwargs):
        calls.append((command, kwargs))
        process = Mock(pid=child_pid, returncode=0)

        def communicate(timeout):
            assert timeout == et.WORKER_TIMEOUT_SECONDS
            with monkeypatch.context() as child:
                for key, value in kwargs["env"].items():
                    child.setenv(key, value)
                child.setattr(os, "getpid", lambda: child_pid)
                child.setattr(os, "getppid", lambda: parent_pid)
                xe.run_worker(destination)
            return "xLAM 测试 worker 输出", ""
        process.communicate.side_effect = communicate
        return process
    monkeypatch.setattr(et.subprocess, "Popen", launch)
    return calls


def test_export_bundle_copies_adapter_and_reloads_in_isolated_worker(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "bundle"
    calls = mock_worker(monkeypatch, destination)
    result = xe.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    evidence = read(result)
    assert result == destination / xe.FINAL
    assert evidence["task_id"] == xt.TASK_ID and evidence["stage"] == "completed"
    assert evidence["seed"] == 42 and evidence["license"]["status"] == "pending"
    assert evidence["base_weights_bundled"] is False and evidence["baseline_promotion"] == "none"
    assert evidence["protocol"] == xt.EVAL_PROTOCOL and evidence["generated_code_executed"] is False
    assert all(value["within_tolerance"] for value in evidence["comparison"].values())
    assert evidence["subprocess"]["pid"] != os.getpid() and evidence["subprocess"]["returncode"] == 0
    assert calls[0][0] == [sys.executable, str(xe.CODE_ROOT / "runner" / "export.py"),
                           "--worker", "--bundle", str(destination)]
    assert xe.validate_export_evidence(r, result) is True
    base = read(destination / xe.BASE_REFERENCE)
    assert base["base_weights_bundled"] is False and base["adapter_only"] is True
    assert base["model_id"] == xt.MODEL and base["revision"] == r.get("model.revision")
    assert base["output_format"] == xt.XLAM_SPEC["output_format"]
    assert base["protocol"] == xt.EVAL_PROTOCOL and base["generated_code_executed"] is False
    assert "adapter_model.safetensors" in {p.name for p in (destination / "model").iterdir()}
    assert not list(destination.glob("model/*.bin"))
    assert (destination / "model" / "adapter_model.safetensors").read_bytes() == \
           (fcase.checkpoints[42] / "adapter_model.safetensors").read_bytes()
    # 快照与训练时 effective recipe 逐字节一致，且不得改写 execution.lockfile
    #（位置不是内容身份；虚拟锚点负责在包内解析锁）。
    snapshot = read(destination / xe.SNAPSHOT)
    effective = pv.effective_recipe(r, 42)
    assert snapshot == effective.data and snapshot["train"]["seed"] == 42
    assert snapshot["execution"]["lockfile"] == LOCKFILE
    import tomllib
    assert tomllib.loads((destination / "recipe.toml").read_text(
        encoding="utf-8"))["execution"]["lockfile"] == LOCKFILE


def test_c405_dispatch_uses_xlam_exporter(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "bundle"
    mock_worker(monkeypatch, destination)
    result = xe.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    data = copy.deepcopy(r.data)
    data["baseline_contract"] = {
        "export_evidence": str(result.parent.relative_to(r.path.parent) / result.name),
        "export_verified": True, "rollback_verified": True,
        "rollback_evidence": "manual/rollback-review.md"}
    registered = Recipe(r.id, r.path, data)
    check = {"kind": "export_reload", "field": "baseline_contract.export_evidence",
             "confirmation_field": "baseline_contract.export_verified"}
    assert bl.check_passed(registered, check) is True
    # 篡改复载预测后，人工布尔值不能放行。
    (destination / "reload" / "eval" / xt.PREDICTIONS_FILE).write_text("{}\n", encoding="utf-8")
    assert bl.check_passed(registered, check) is False


def test_export_evidence_rejects_tampered_reload_or_bundle(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "bundle"
    mock_worker(monkeypatch, destination)
    result = xe.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    (destination / "reload" / "eval" / xt.PREDICTIONS_FILE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(TaskExecutionError):
        xe.validate_export_evidence(r, result)
    shutil.copyfile(destination / "reference" / xt.PREDICTIONS_FILE,
                    destination / "reload" / "eval" / xt.PREDICTIONS_FILE)
    assert xe.validate_export_evidence(r, result) is True
    (destination / "model" / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(TaskExecutionError):
        xe.validate_export_evidence(r, result)


def test_lock_layout_keeps_content_identity_for_repository_relative_lock(tmp_path):
    raw = "../../../../runner/locks/llm__sft__llama3-8b-xlam-function-calling.lock.toml"
    bundle = tmp_path / "bundle"
    lock_bytes = b'schema_version = 1\nstatus = "frozen"\n'
    lock_path = bundle / "runner" / "locks" / "llm__sft__llama3-8b-xlam-function-calling.lock.toml"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(lock_bytes)
    relative, resolved, recipe_path = xe._lock_layout(raw, bundle)
    assert relative == "runner/locks/llm__sft__llama3-8b-xlam-function-calling.lock.toml"
    assert resolved == lock_path
    assert recipe_path.parent == bundle / xe.LOCK_ANCHOR / xe.LOCK_ANCHOR / xe.LOCK_ANCHOR / xe.LOCK_ANCHOR
    assert Path(os.path.abspath(recipe_path.parent / raw)) == lock_path
    data = {"id": xt.RECIPE_ID, "execution": {"task_id": xt.TASK_ID, "lockfile": raw}}
    r = Recipe(xt.RECIPE_ID, recipe_path, data)
    assert r.get("execution.lockfile") == raw
    assert xt.execution_identity(r)["environment_lock_sha256"] == hashlib.sha256(lock_bytes).hexdigest()
    assert be.frozen_environment(r) is False
    relative2, resolved2, _ = xe._lock_layout("../locks/frozen.lock.toml", bundle)
    assert relative2 == "locks/frozen.lock.toml" and resolved2 == bundle / "locks" / "frozen.lock.toml"


def test_lock_layout_rejects_escaping_or_conflicting_paths(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for raw in ("model/adapter_model.safetensors", "recipe-snapshot.json",
                "reference/evaluation.json", "../../a/../../b.lock.toml",
                "/absolute.lock.toml", "..\\windows.lock.toml", ""):
        with pytest.raises(TaskExecutionError):
            xe._lock_layout(raw, bundle)


def test_verify_bundle_rejects_unbound_extra_file(fcase, tmp_path, monkeypatch):
    r = fcase.r
    destination = tmp_path / "prepared"

    def stop(bundle):
        raise TaskExecutionError("测试停止于 prepared 包")
    with monkeypatch.context() as context:
        context.setattr(et, "_launch_worker", stop)
        with pytest.raises(TaskExecutionError, match="prepared"):
            xe.export_bundle(r, fcase.checkpoints[42], fcase.evaluations[42], destination)
    assert (destination / xe.MANIFEST).is_file() and not (destination / xe.FINAL).exists()
    (destination / "model" / "unbound.json").write_text('{}\n', encoding="utf-8")
    with pytest.raises(TaskExecutionError, match="未绑定文件"):
        xe._verify_bundle(destination)
