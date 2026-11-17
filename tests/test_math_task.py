"""数学路由的纯函数 / mock 测试；绝不导入真实训练框架或下载资源。"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recipe_lab import math_task as mt
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan

ROOT = Path(__file__).resolve().parents[1]
GOOD = "<think>计算过程</think><answer>2</answer>"


@pytest.fixture
def recipe(tmp_path):
    # revision 仅为隔离测试用值，不代表任何真实模型或数据 commit。
    data = {
        "id": mt.RECIPE_ID,
        "method": {"family": "grpo", "trainer": "GRPOTrainer"},
        "model": {"base": mt.MODEL_ID, "revision": "a" * 40, "modality": "llm"},
        "dataset": {"name": mt.DATASET_ID, "revision": "b" * 40},
        "dependencies": {"trl": "0.14.0"},
        "rewards": {"functions": ["format_reward", "accuracy_reward"]},
        "train": {"learning_rate": 1e-5, "gradient_accumulation_steps": 16,
                  "num_train_epochs": 1, "max_completion_length": 64, "num_generations": 4,
                  "remove_unused_columns": False, "seed": 42, "data_seed": 42},
        "evaluation": {"seed": 42, "multi_run": True},
        "budget": {"smoke_steps": 5, "full_steps": 150},
        "execution": {"lockfile": "fixture.lock.toml"},
    }
    # 仅用于依赖被mock的测试身份；不是可用frozen环境。
    (tmp_path / "fixture.lock.toml").write_text('status = "test-only"\n', encoding="utf-8")
    return Recipe(mt.RECIPE_ID, tmp_path / "recipe.toml", data)


@pytest.fixture
def rows():
    return [{"problem": f"唯一问题 {i}", "solution": "2", "source": f"来源 {i}"}
            for i in range(1500)]


def parse(text):
    return [text] if text else []


def verify(gold, prediction):
    return gold == prediction


def runner(name):
    spec = importlib.util.spec_from_file_location(f"math_test_runner_{name}", ROOT / "runner" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_for(recipe, tmp_path, mode):
    plan = mt.task_plan(recipe, build_plan(recipe, mode), mode)
    plan["run"]["output_dir"] = str(tmp_path / mode)
    return plan


def test_splits_ids_content_hashes_and_manifest_are_stable(rows):
    splits, manifest = mt.prepare_splits(rows, parse=parse)
    again, second = mt.prepare_splits(iter(rows), parse=parse)
    assert splits == again and manifest == second
    ids = []
    for name, (lo, hi) in mt.SPLIT_RANGES.items():
        assert len(splits[name]) == 500
        assert [s["sample_id"] for s in splits[name]] == [f"train:{i}" for i in range(lo, hi)]
        assert manifest["splits"][name]["count"] == 500
        ids.extend(s["sample_id"] for s in splits[name])
    assert len(set(ids)) == 1500
    assert splits["test"][0]["content_sha256"] == mt.sha256_json(rows[1000])
    assert splits["test"][0]["prompt"] == mt.make_prompt(rows[1000]["problem"])
    original = manifest["sha256"]
    assert original == mt.sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
    rows[1000]["source"] = "修改真实内容"
    _, changed = mt.prepare_splits(rows, parse=parse)
    assert changed["sha256"] != original


@pytest.mark.parametrize("bad", [[], [{"problem": "p", "solution": "2"}] * 1499])
def test_short_dataset_rejected(bad):
    with pytest.raises(mt.DatasetValidationError, match="1500"):
        mt.prepare_splits(bad, parse=parse)


@pytest.mark.parametrize("field,value", [("problem", ""), ("solution", ""), ("solution", None)])
def test_invalid_dataset_fields_rejected(rows, field, value):
    rows[501][field] = value
    with pytest.raises(mt.DatasetValidationError, match="train:501"):
        mt.prepare_splits(rows, parse=parse)


def test_cross_split_duplicates_rejected(rows):
    rows[1000] = copy.deepcopy(rows[3])
    with pytest.raises(mt.DatasetValidationError, match="重叠"):
        mt.prepare_splits(rows, parse=parse)
    rows[1000] = {"problem": rows[4]["problem"], "solution": "不同答案", "source": "不同元数据"}
    with pytest.raises(mt.DatasetValidationError, match="重叠"):
        mt.prepare_splits(rows, parse=parse)


@pytest.mark.parametrize("bad", ["", "2", "<think></think><answer>2</answer>",
                                "<think>x</think><answer></answer>",
                                "<think>x</think><answer>2</answer>额外内容",
                                "<think>x</think><answer><answer>2</answer></answer>",
                                [], None, [{"role": "user", "content": GOOD}]])
def test_bad_format_never_gets_full_reward(bad):
    assert mt.format_reward([bad]) == [0.0]
    verifier = Mock(return_value=True)
    assert mt.accuracy_reward([bad], ["2"], parse=parse, verify=verifier) == [0.0]
    verifier.assert_not_called()


def test_string_chat_and_wrong_answer_rewards():
    chat = [{"role": "assistant", "content": GOOD}]
    assert mt.format_reward([GOOD, chat]) == [1.0, 1.0]
    assert mt.accuracy_reward([GOOD, chat], ["2", "2"], parse=parse, verify=verify) == [1.0, 1.0]
    assert mt.accuracy_reward([GOOD], ["3"], parse=parse, verify=verify) == [0.0]


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_prediction_parse_failure_is_zero(failure):
    def failing_parse(text):
        if text == "gold":
            return ["2"]
        if failure == "exception":
            raise ValueError("预测解析失败")
        return []
    assert mt.accuracy_reward([GOOD], ["gold"], parse=failing_parse, verify=lambda *_: True) == [0.0]


def test_verify_failure_is_zero():
    def broken_verify(*args):
        raise RuntimeError("验证器异常")
    assert mt.accuracy_reward([GOOD], ["2"], parse=parse, verify=broken_verify) == [0.0]


@pytest.mark.parametrize("completion", [GOOD, ""])
@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_gold_parse_failure_is_dataset_error(completion, failure):
    def broken_parse(text):
        if failure == "exception":
            raise ValueError("gold 解析异常")
        return []
    with pytest.raises(mt.DatasetValidationError, match="gold"):
        mt.accuracy_reward([completion], ["gold"], parse=broken_parse, verify=lambda *_: True)


def test_dataset_gold_parse_failure_is_fatal(rows):
    with pytest.raises(mt.DatasetValidationError, match="gold"):
        mt.prepare_splits(rows, parse=lambda _: [])


def test_reward_length_mismatch_is_error():
    with pytest.raises(mt.DatasetValidationError, match="数量"):
        mt.accuracy_reward([GOOD], [], parse=parse, verify=verify)


def test_budget_batch_seed_unknown_keys_and_task_route(recipe, tmp_path):
    smoke = plan_for(recipe, tmp_path, "smoke")
    full = plan_for(recipe, tmp_path, "full")
    assert smoke["run"]["output_dir"] != full["run"]["output_dir"]
    config = mt.config_kwargs(recipe, smoke, tmp_path / "smoke")
    assert config["max_steps"] == 5
    assert config["seed"] == config["data_seed"] == 42
    assert config["per_device_train_batch_size"] == 4
    assert mt.config_kwargs(recipe, full, tmp_path / "full")["max_steps"] == 150
    recipe.data["train"]["max_steps"] = 6
    with pytest.raises(mt.TaskExecutionError, match="预算"):
        mt.config_kwargs(recipe, smoke, tmp_path / "smoke")
    recipe.data["train"].pop("max_steps")
    recipe.data["train"]["per_device_train_batch_size"] = 3
    with pytest.raises(mt.TaskExecutionError, match="batch"):
        mt.config_kwargs(recipe, smoke, tmp_path / "smoke")
    recipe.data["train"].pop("per_device_train_batch_size")
    recipe.data["train"]["imaginary_parameter"] = 1
    with pytest.raises(mt.TaskExecutionError, match="未支持参数"):
        mt.require_supported(recipe)
    recipe.data["train"].pop("imaginary_parameter")
    with pytest.raises(mt.TaskExecutionError, match="执行路由"):
        mt.require_supported(recipe, {"task_id": "other-v1"})
    recipe.data["evaluation"]["seed"] = 9
    assert mt.fixed_seeds(recipe) == (42, 42), "评估seed必须能独立于训练seed"
    recipe.data["evaluation"]["seed"] = True
    with pytest.raises(mt.TaskExecutionError, match="evaluation.seed"):
        mt.fixed_seeds(recipe)


def test_signature_does_not_accept_kwargs_as_support():
    def target(seed, **kwargs):
        pass
    with pytest.raises(mt.TaskExecutionError, match="max_steps"):
        mt.check_signature(target, {"seed": 42, "max_steps": 5}, "目标 API")
    mt.check_signature(target, {"seed": 42}, "目标 API")
    with pytest.raises(mt.TaskExecutionError, match="不兼容"):
        mt.check_signature(target, {}, "目标 API")


@pytest.mark.parametrize("name", ["train", "eval", "zero_shot"])
def test_unsupported_entry_rejects_before_any_framework_import(recipe, tmp_path, monkeypatch, name):
    module = runner(name)
    other = Recipe("llm/sft/not-supported", recipe.path, recipe.data)
    plan = plan_for(other, tmp_path, "full" if name == "train" else "eval")
    def forbidden_import(*args):
        pytest.fail("拒绝路由之前不得导入框架")
    monkeypatch.setattr(mt.importlib, "import_module", forbidden_import)
    with pytest.raises(mt.TaskExecutionError, match="尚未实现"):
        (module._build_and_train if name == "train" else module.run_real)(other, plan)
    assert not list(tmp_path.rglob("*.json"))


def test_no_gpu_rejects_before_other_dependencies(monkeypatch):
    calls = []
    def fake_import(name):
        calls.append(name)
        assert name == "torch"
        return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(mt.importlib, "import_module", fake_import)
    with pytest.raises(mt.TaskExecutionError, match="没有可用 CUDA"):
        mt.load_runtime(training=True)
    assert calls == ["torch"]


def test_missing_dependencies_are_actionable(monkeypatch):
    def unavailable(name):
        raise ImportError(name)
    monkeypatch.setattr(mt.importlib, "import_module", unavailable)
    with pytest.raises(mt.TaskExecutionError, match="lock"):
        mt.load_runtime()
    with pytest.raises(mt.TaskExecutionError, match="math_verify==0.3.3"):
        mt.math_backend()


def test_missing_or_floating_revision_rejected(recipe):
    recipe.data["model"]["revision"] = "main"
    with pytest.raises(mt.TaskExecutionError, match="commit SHA"):
        mt.require_revision(recipe, "model.revision")


@pytest.mark.parametrize("name", ["train", "eval", "zero_shot"])
@pytest.mark.parametrize("real", [False, True])
def test_cli_isolated_root_environment_and_no_dry_run_evidence(tmp_path, name, real):
    recipes = tmp_path / "recipes" / "other"
    recipes.mkdir(parents=True)
    recipe_file = recipes / "recipe.toml"
    recipe_file.write_text('id = "llm/sft/isolated"\n', encoding="utf-8")
    program = '''import importlib.abc, os, runpy, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trl', 'transformers', 'datasets', 'math_verify'}:
            raise AssertionError('不应导入框架: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
sys.argv = [sys.argv[1], '--recipe', 'llm/sft/isolated', sys.argv[2], '--json']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
finally:
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
'''
    result = subprocess.run(
        [sys.executable, "-c", program, str(ROOT / "runner" / f"{name}.py"),
         "--full" if real else "--dry-run"], cwd=tmp_path,
        env={**os.environ, "RECIPE_LAB_ROOT": str(tmp_path), "CUBLAS_WORKSPACE_CONFIG": "invalid"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == (1 if real else 0), result.stderr
    if real:
        assert "尚未实现" in result.stderr
    else:
        assert json.loads(result.stdout)["recipe_id"] == "llm/sft/isolated"
    assert list(tmp_path.rglob("*.json")) == []


class FakeConfig:
    def __init__(self, learning_rate, gradient_accumulation_steps, num_train_epochs,
                 max_completion_length, num_generations, remove_unused_columns,
                 seed, data_seed, per_device_train_batch_size, max_steps, output_dir, report_to):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
    def to_dict(self):
        return dict(self.__dict__)


class FakeEncoded(dict):
    def to(self, device):
        assert device == "cuda"
        return self


def fake_runtime(rows):
    calls = {"models": [], "tokenizers": [], "prompts": [], "generations": [], "datasets": []}
    tokenizer = SimpleNamespace(eos_token="<eos>", eos_token_id=0, pad_token=None)
    def save_tokenizer(directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text('{}', encoding="utf-8")
    tokenizer.save_pretrained = save_tokenizer
    def template(prompt, **kwargs):
        calls["prompts"].append(copy.deepcopy(prompt))
        assert kwargs["add_generation_prompt"] is True
        return FakeEncoded(input_ids=SimpleNamespace(shape=(1, 2)), attention_mask="mock-mask")
    tokenizer.apply_chat_template = template
    tokenizer.decode = lambda *args, **kwargs: GOOD
    def load_tokenizer(source, **kwargs):
        calls["tokenizers"].append((source, kwargs))
        return tokenizer
    model = SimpleNamespace()
    model.to = lambda device: model
    model.eval = lambda: None
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
    dependencies = {
        "torch": SimpleNamespace(
            __version__="mock", float32="mock-float32", version=SimpleNamespace(cuda="mock"),
            cuda=SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                                 max_memory_allocated=lambda: 0, get_device_name=lambda _: "mock-GPU"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace())),
            use_deterministic_algorithms=lambda enabled: None, inference_mode=contextlib.nullcontext,
        ),
        "transformers": SimpleNamespace(
            __version__="mock", set_seed=Mock(),
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
        ),
        "datasets": SimpleNamespace(__version__="mock", Dataset=SimpleNamespace(from_list=from_list),
                                    load_dataset=lambda *args, **kwargs: rows),
        "parse": parse, "verify": verify,
    }
    return dependencies, calls


def trainer_class(calls, *, incomplete=False):
    class FakeTrainer:
        def __init__(self, model, processing_class, args, train_dataset, eval_dataset, reward_funcs):
            calls["trainer"] = {
                "model": model, "processing_class": processing_class, "args": args,
                "train_dataset": train_dataset, "eval_dataset": eval_dataset, "reward_funcs": reward_funcs,
            }
            self.args = args
            self.state = SimpleNamespace(global_step=args.max_steps - int(incomplete))
        def train(self):
            calls["train_called"] = True
            return SimpleNamespace(metrics={"mock_train_loss": 0.25})
        def save_model(self, directory):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text('{}', encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"mock weights")
    return FakeTrainer


def test_training_assembles_real_contract_and_only_then_writes_evidence(recipe, rows, tmp_path, monkeypatch):
    module = runner("train")
    dependencies, calls = fake_runtime(rows)
    dependencies["trl"] = SimpleNamespace(__version__="0.14.0", GRPOConfig=FakeConfig,
                                          GRPOTrainer=trainer_class(calls))
    monkeypatch.setattr(module, "load_runtime", lambda **kwargs: dependencies)
    monkeypatch.setattr(module, "require_frozen_runtime", lambda *args: None)
    plan = plan_for(recipe, tmp_path, "full")
    result = module._build_and_train(recipe, plan)
    output = tmp_path / "full"
    assert not (output / "training.json").exists()
    evidence_path = module.write_evidence(recipe, plan, result)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["kind"] == "training" and evidence["stage"] == "completed"
    assert evidence["task_id"] == mt.TASK_ID and evidence["steps"] == 150
    assert evidence["config_sha256"] == mt.sha256_json(evidence["config"])
    assert evidence["model_artifact"]["sha256"] == mt.artifact_manifest(output)["sha256"]
    assert evidence["baseline_promotion"] == "none"
    assert len(calls["datasets"]) == 2 and all(len(d) == 500 for d in calls["datasets"])
    assert set(calls["datasets"][0][0]) == {"prompt", "solution"}
    assert calls["trainer"]["reward_funcs"] == [mt.format_reward, mt.accuracy_reward]
    assert calls["models"][0][0] == mt.MODEL_ID
    assert calls["tokenizers"][0][1]["revision"] == recipe.get("model.revision")
    assert mt.checkpoint_path(plan) == output.resolve()
    (output / "model.safetensors").write_bytes(b"changed mock weights")
    with pytest.raises(mt.TaskExecutionError, match="不一致"):
        mt.checkpoint_path(plan)


def test_incomplete_training_never_writes_completed_evidence(recipe, rows, tmp_path, monkeypatch):
    module = runner("train")
    dependencies, calls = fake_runtime(rows)
    dependencies["trl"] = SimpleNamespace(GRPOConfig=FakeConfig, GRPOTrainer=trainer_class(calls, incomplete=True))
    monkeypatch.setattr(module, "load_runtime", lambda **kwargs: dependencies)
    monkeypatch.setattr(module, "require_frozen_runtime", lambda *args: None)
    with pytest.raises(mt.TaskExecutionError, match="未完成预算"):
        module._build_and_train(recipe, plan_for(recipe, tmp_path, "smoke"))
    assert not list(tmp_path.rglob("*.json"))


def test_api_mismatch_rejected_before_loading_models(recipe, rows, tmp_path, monkeypatch):
    module = runner("train")
    dependencies, calls = fake_runtime(rows)
    dependencies["trl"] = SimpleNamespace(GRPOConfig=lambda **kwargs: None, GRPOTrainer=trainer_class(calls))
    monkeypatch.setattr(module, "load_runtime", lambda **kwargs: dependencies)
    monkeypatch.setattr(module, "require_frozen_runtime", lambda *args: None)
    with pytest.raises(mt.TaskExecutionError, match="API 不显式支持"):
        module._build_and_train(recipe, plan_for(recipe, tmp_path, "smoke"))
    assert calls["models"] == calls["tokenizers"] == []


def create_checkpoint(path, recipe=None):
    path.mkdir(parents=True)
    (path / "config.json").write_text('{}', encoding="utf-8")
    (path / "tokenizer_config.json").write_text('{}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"mock checkpoint")
    if recipe is not None:
        from recipe_lab.baseline_evidence import fingerprint
        # 只在隔离测试目录生成合成来源，不作真实训练证据。
        mt.save_json(path / "training.json", {
            "kind": "training", "stage": "completed", "mode": "full", "task_id": mt.TASK_ID,
            "recipe_id": recipe.id, "model_revision": recipe.get("model.revision"),
            "dataset_revision": recipe.get("dataset.revision"), "fingerprint": fingerprint(recipe),
            "model_artifact": mt.artifact_manifest(path), "mock": True,
        })
    return path


def test_eval_never_falls_back_to_base_and_rejects_repeated_runs(recipe, tmp_path, monkeypatch):
    module = runner("eval")
    plan = plan_for(recipe, tmp_path, "eval")
    dependencies = Mock(side_effect=AssertionError("加载依赖之前应拒绝"))
    monkeypatch.setattr(mt, "load_runtime", dependencies)
    with pytest.raises(mt.TaskExecutionError, match="没有训练 checkpoint"):
        module.run_real(recipe, plan)
    plan["requested_runs"] = 5
    with pytest.raises(mt.TaskExecutionError, match="训练多 seed"):
        module.run_real(recipe, plan)
    dependencies.assert_not_called()
    assert not list(tmp_path.rglob("*.json"))


def test_default_full_output_needs_completed_training_evidence(recipe, tmp_path):
    plan = plan_for(recipe, tmp_path, "eval")
    create_checkpoint(tmp_path / "full")
    with pytest.raises(mt.TaskExecutionError, match="缺少已完成"):
        mt.checkpoint_path(plan)


def test_zero_shot_and_trained_eval_share_test_prompt_seed_and_greedy_protocol(recipe, rows, tmp_path, monkeypatch):
    checkpoint = create_checkpoint(tmp_path / "explicit_checkpoint", recipe)
    all_calls, all_evidence = [], []
    for mode, zero_shot in (("eval", False), ("zero-shot", True)):
        dependencies, calls = fake_runtime(rows)
        monkeypatch.setattr(mt, "load_runtime", lambda **kwargs: dependencies)
        monkeypatch.setattr(mt, "require_frozen_runtime", lambda *args: None)
        path = mt.run_evaluation(recipe, plan_for(recipe, tmp_path, mode),
                                 checkpoint=None if zero_shot else checkpoint, zero_shot=zero_shot)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        predictions = json.loads((path.parent / "predictions.json").read_text(encoding="utf-8"))
        assert len(predictions) == 500 and predictions[0]["sample_id"] == "train:1000"
        assert evidence["kind"] == "evaluation" and evidence["stage"] == "completed"
        assert evidence["sample_count"] == 500
        assert evidence["metrics"] == {"accuracy": 1.0, "pass@1": 1.0, "format_compliance": 1.0}
        assert evidence["config_sha256"] == mt.sha256_json(evidence["config"])
        assert evidence["predictions"]["sha256"] == mt.sha256_json(predictions)
        assert evidence["evaluation_protocol"] == "single_checkpoint_evaluation"
        assert evidence["training_seed_variance_measured"] is False
        assert "expected" not in evidence and evidence["baseline_promotion"] == "none"
        assert calls["models"][0][0] == (mt.MODEL_ID if zero_shot else str(checkpoint.resolve()))
        loader_options = calls["models"][0][1]
        assert loader_options.get("revision") == (recipe.get("model.revision") if zero_shot else None)
        assert loader_options.get("local_files_only") == (None if zero_shot else True)
        assert all(c["do_sample"] is False and c["num_beams"] == 1 for c in calls["generations"])
        assert all(p[0]["content"] == mt.SYSTEM_PROMPT for p in calls["prompts"])
        dependencies["transformers"].set_seed.assert_called_once_with(42)
        all_calls.append(calls)
        all_evidence.append(evidence)
    assert all_calls[0]["prompts"] == all_calls[1]["prompts"]
    assert all_evidence[0]["split_manifest"] == all_evidence[1]["split_manifest"]


@pytest.mark.parametrize("mode", ["dry-run", "smoke", "full"])
def test_supported_task_preview_exposes_effective_budget_and_batch(recipe, mode):
    plan = mt.task_plan(recipe, build_plan(recipe, mode), mode, mt.TASK_ID)
    assert plan["execution_supported"] is True
    assert plan["task_id"] == mt.TASK_ID
    assert plan["run"]["params"]["per_device_train_batch_size"] == 4
    assert plan["run"]["params"]["max_steps"] == (150 if mode == "full" else 5)
    assert plan["dataset"]["split_plan"]["ranges"]["test"] == {"start": 1000, "stop": 1500, "count": 500}
    assert "expected" not in plan["evaluation"]
    assert plan["evaluation"]["runs"] == 1
    recipe.data["train"]["unsupported"] = 1
    rejected = mt.task_plan(recipe, build_plan(recipe, mode), mode)
    assert rejected["execution_supported"] is False
    assert "未支持参数" in rejected["unsupported_reason"]


def test_math_module_has_no_eager_framework_imports():
    program = '''import importlib.abc, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trl', 'transformers', 'datasets', 'math_verify'}:
            raise AssertionError('不应导入框架: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
sys.path.insert(0, sys.argv[1])
from recipe_lab.math_task import format_reward, SYSTEM_PROMPT
assert format_reward(['']) == [0.0]
assert '<think>' in SYSTEM_PROMPT and '<answer>' in SYSTEM_PROMPT
'''
    subprocess.run([sys.executable, "-c", program, str(ROOT / "src")], check=True,
                   capture_output=True, text=True)


def test_explicit_checkpoint_requires_training_provenance(recipe, tmp_path):
    path = create_checkpoint(tmp_path / "explicit")
    plan = plan_for(recipe, tmp_path, "eval")
    with pytest.raises(mt.TaskExecutionError, match="缺少已完成"):
        mt.checkpoint_path(plan, path, recipe=recipe)


def test_checkpoint_from_other_config_rejected(recipe, tmp_path):
    path = create_checkpoint(tmp_path / "other-config", recipe)
    plan = plan_for(recipe, tmp_path, "eval")
    assert mt.checkpoint_path(plan, path, recipe=recipe) == path.resolve()
    recipe.data["train"]["learning_rate"] *= 2
    with pytest.raises(mt.TaskExecutionError, match="配置或 revision"):
        mt.checkpoint_path(plan, path, recipe=recipe)


def test_multiple_training_seeds_keep_same_evaluation_protocol(recipe, rows, tmp_path, monkeypatch):
    generations = []
    for seed in (42, 43, 44):
        recipe.data["train"]["seed"] = seed
        # data/evaluation seed固定；只改变独立训练seed。
        deps, calls = fake_runtime(rows)
        monkeypatch.setattr(mt, "load_runtime", lambda **kwargs: deps)
        monkeypatch.setattr(mt, "require_frozen_runtime", lambda *args: None)
        plan = plan_for(recipe, tmp_path / str(seed), "zero-shot")
        evidence_path = mt.run_evaluation(recipe, plan, zero_shot=True)
        evidence = json.loads(evidence_path.read_text())
        assert evidence["seed"] == seed and evidence["data_seed"] == 42
        assert evidence["config"]["generation"]["seed"] == 42
        deps["transformers"].set_seed.assert_called_once_with(42)
        generations.append(evidence["config"]["generation"])
    assert generations[0] == generations[1] == generations[2]


def test_generation_failure_writes_no_completed_evidence(recipe, rows, tmp_path, monkeypatch):
    dependencies, _ = fake_runtime(rows)
    def broken_model(*args, **kwargs):
        model = SimpleNamespace()
        model.to = lambda device: model
        model.eval = lambda: None
        model.generate = Mock(side_effect=RuntimeError("mock decode failure"))
        return model
    dependencies["transformers"].AutoModelForCausalLM.from_pretrained = broken_model
    monkeypatch.setattr(mt, "load_runtime", lambda **kwargs: dependencies)
    monkeypatch.setattr(mt, "require_frozen_runtime", lambda *args: None)
    with pytest.raises(RuntimeError, match="decode failure"):
        mt.run_evaluation(recipe, plan_for(recipe, tmp_path, "zero-shot"), zero_shot=True)
    assert not list(tmp_path.rglob("*.json"))
