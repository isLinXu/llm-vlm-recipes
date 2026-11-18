"""DPO / MPO 偏好协议的隔离离线测试。

合成图像与 fake processor 只写断言，不联网、不加载 torch/CUDA，
也不把 dry-run 计划当成 completed 证据或 B2 验收。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipe_lab import dpo_export as de
from recipe_lab import dpo_suite as ds
from recipe_lab import dpo_task as dt
from recipe_lab import mpo_export as me
from recipe_lab import mpo_suite as ms
from recipe_lab import mpo_task as mpt
from recipe_lab import preference_common as pc
from recipe_lab import task_router as tr
from recipe_lab.loader import load_recipes
from recipe_lab.math_task import DatasetValidationError, TaskExecutionError
from recipe_lab.model import Recipe
from recipe_lab.plan import build_plan

ROOT = Path(__file__).resolve().parents[1]
DPO_ID = dt.RECIPE_ID
MPO_ID = mpt.RECIPE_ID


def recipes():
    return {r.id: r for r in load_recipes(ROOT)}


def image(payload: bytes):
    return SimpleNamespace(mode="RGB", tobytes=lambda: payload, convert=lambda mode: image(payload))


def row(index, payload, prompt=None, chosen="chosen-answer", rejected="rejected-answer"):
    return {
        "prompt": prompt or f"question-{index}",
        "chosen": chosen,
        "rejected": rejected,
        "images": [image(payload)],
    }


def rows_covering_splits(data_seed):
    found = {}
    n = 0
    while len(found) < 3 and n < 5000:
        payload = n.to_bytes(4, "little")
        parsed = pc.parse_preference_row(row(n, payload), n)
        bucket = pc.split_bucket(parsed["image_sha256"], data_seed)
        found.setdefault(bucket, row(n, payload))
        n += 1
    assert set(found) == {"train", "validation", "test"}
    return list(found.values())


class _Tensor:
    def __init__(self, values, dtype):
        self.values = values
        self.dtype = dtype


class _Processor:
    image_token_id = 7
    image_token = "<image>"
    name_or_path = "fake-processor"

    def __init__(self):
        self.tokenizer = SimpleNamespace(
            name_or_path="fake-tokenizer", vocab_size=32, eos_token="</s>",
            eos_token_id=2, pad_token="<pad>", pad_token_id=0, chat_template="preference")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                parts.append("".join(
                    piece.get("text", "") for piece in content if isinstance(piece, dict)))
            else:
                parts.append(str(content))
        text = "/".join(parts)
        return text + "/GEN" if add_generation_prompt else text

    def __call__(self, text, images, return_tensors=None, padding=False):
        ids = [self.image_token_id, 1, 2] if text.endswith("/GEN") else [self.image_token_id, 1, 2, 9]
        return {"input_ids": ids, "pixel_values": [[0.0, 1.0]]}


def _torch():
    return SimpleNamespace(
        tensor=lambda values, dtype=None: _Tensor(values, dtype),
        long="long", float32="float32")


def test_real_recipes_match_task_contracts():
    loaded = recipes()
    dpo, mpo = loaded[DPO_ID], loaded[MPO_ID]
    assert dt.require_supported(dpo) == dt.TASK_ID
    assert mpt.require_supported(mpo) == mpt.TASK_ID
    assert dpo.get("preference.cross_backbone_attribution_valid") is False
    assert mpo.get("preference.cross_backbone_attribution_valid") is False
    assert dpo.get("preference.beta") == 0.1
    assert mpo.get("train.bf16") is False
    assert mpo.get("quantization.bnb_4bit_compute_dtype") == "bfloat16"
    assert tr.resolve(dpo) == dt.TASK_ID and tr.resolve(mpo) == mpt.TASK_ID
    assert tr.load(dpo, "suite").__name__ == "recipe_lab.dpo_suite"
    assert tr.load(mpo, "export").__name__ == "recipe_lab.mpo_export"
    assert tr.load(dpo, "task") is not tr.load(mpo, "task")


def test_require_supported_rejects_protocol_drift():
    dpo = recipes()[DPO_ID]
    data = copy.deepcopy(dpo.data)
    data["preference"]["beta"] = 0.2
    with pytest.raises(TaskExecutionError, match="preference.beta"):
        dt.require_supported(Recipe(dpo.id, dpo.path, data))
    mpo = recipes()[MPO_ID]
    drifted = copy.deepcopy(mpo.data)
    drifted["train"]["loss_weights"] = [1.0, 1.0, 1.0]
    with pytest.raises(TaskExecutionError):
        mpt.require_supported(Recipe(mpo.id, mpo.path, drifted))


def test_task_plan_dry_run_is_supported_and_does_not_attribute_across_backbones():
    loaded = recipes()
    for module, recipe in ((dt, loaded[DPO_ID]), (mpt, loaded[MPO_ID])):
        plan = module.task_plan(recipe, build_plan(recipe, "dry-run"), "dry-run")
        assert plan["execution_supported"] is True, plan.get("unsupported_reason")
        assert plan["task_id"] == module.TASK_ID
        assert plan["evaluation"]["training_seed_variance_measured"] is False
        assert "cross_backbone_attribution_valid=false" in plan["migration"]["cross_backbone"]


def test_preference_schema_and_degenerate_pairs_fail_closed():
    payload = b"img"
    with pytest.raises(DatasetValidationError, match="未知字段"):
        pc.parse_preference_row({**row(0, payload), "extra": 1}, 0)
    with pytest.raises(DatasetValidationError, match="相同"):
        pc.parse_preference_row(row(0, payload, chosen="Same", rejected=" same "), 0)
    with pytest.raises(DatasetValidationError, match="images"):
        pc.parse_preference_row({"prompt": "q", "chosen": "a", "rejected": "b", "images": []}, 0)


def test_splits_are_image_isolated_shared_and_reject_duplicates():
    dpo, mpo = recipes()[DPO_ID], recipes()[MPO_ID]
    assert dpo.get("train.data_seed") == mpo.get("train.data_seed") == 42
    covered = rows_covering_splits(42)
    train_image = covered[0]["images"][0].tobytes()
    covered.append(row(90, train_image, prompt="same-image-other-question"))
    dpo_splits, dpo_manifest = dt.prepare_splits(covered, dpo)
    mpo_splits, mpo_manifest = mpt.prepare_splits(covered, mpo)
    assert dpo_manifest["split_hashes_sha256"] == mpo_manifest["split_hashes_sha256"]
    owners = {row["image_sha256"] for row in dpo_splits["train"]}
    assert all(row["image_sha256"] in owners for row in dpo_splits["train"])
    assert not (set(dpo_manifest["split_hashes"]["train"]) & set(dpo_manifest["split_hashes"]["test"]))
    identity = pc.split_identity(
        dpo_manifest, dataset_id=dt.DATASET, dataset_revision="", algorithm_version=pc.SPLIT_ALGORITHM)
    shared = pc.validate_shared_preference_split(
        dpo_manifest, mpo_manifest, dpo_identity=identity, mpo_identity=identity)
    assert shared["shared_preference_split_valid"] is True
    assert shared["cross_backbone_attribution_valid"] is False
    with pytest.raises(TaskExecutionError, match="身份不一致"):
        pc.validate_shared_preference_split(
            dpo_manifest, mpo_manifest, dpo_identity=identity,
            mpo_identity={**identity, "algorithm_version": "other"})
    with pytest.raises(DatasetValidationError, match="重复"):
        dt.prepare_splits(covered + [copy.deepcopy(covered[0])], dpo)
    with pytest.raises(DatasetValidationError, match="为空"):
        dt.prepare_splits([covered[0]], dpo)


def test_encode_pair_masks_prompt_and_image_tokens_and_rejects_gold_leak():
    parsed = pc.parse_preference_row(row(0, b"rgb", chosen="CHOSEN_GOLD", rejected="REJECTED_GOLD"), 0)
    processor = _Processor()
    identity = pc.processor_identity(processor, dt.PROTOCOL)
    encoded = pc.encode_pair(parsed, processor, identity, "system", "chosen", 32)
    assert encoded["labels"][:encoded["prompt_tokens"]] == [-100] * encoded["prompt_tokens"]
    assert encoded["masked_image_tokens"] >= 1
    assert all(label == -100 for token, label in zip(encoded["input_ids"], encoded["labels"])
               if token == processor.image_token_id)
    assert "CHOSEN_GOLD" not in encoded["prompt_text"] and "REJECTED_GOLD" not in encoded["prompt_text"]
    batch = pc.collate_pairs([encoded, encoded], _torch(), 0, processor.image_token_id)
    assert batch["labels"].dtype == "long"
    leaking = _Processor()

    def leak(messages, tokenize=False, add_generation_prompt=False):
        return "CHOSEN_GOLD/GEN" if add_generation_prompt else "CHOSEN_GOLD"

    leaking.apply_chat_template = leak
    with pytest.raises(DatasetValidationError, match="泄漏"):
        pc.encode_pair(parsed, leaking, identity, "system", "chosen", 32)


def test_implicit_reward_and_losses_are_explicit():
    assert dt.sigmoid_dpo_loss(0.0, -2.0, 0.0, 0.0, 0.1) < dt.sigmoid_dpo_loss(-2.0, 0.0, 0.0, 0.0, 0.1)
    with pytest.raises(TaskExecutionError, match="beta"):
        dt.sigmoid_dpo_loss(0.0, 0.0, 0.0, 0.0, 0.0)
    chosen, rejected = dt.implicit_rewards(1.0, -1.0, 0.0, 0.0, 0.1)
    scored = pc.score_pair(chosen, rejected, 4, 8)
    assert scored["win"] == 1.0 and scored["length_bucket"] == "rejected_longer"
    metrics = pc.recompute_metrics([scored, pc.score_pair(0.0, 1.0, 2, 2)])
    assert metrics["win_rate"] == 0.5 and metrics["reward_accuracy"] == 0.5
    full = mpt.mpo_combined_loss(
        ["sigmoid", "bco_pair", "sft"], [0.8, 0.2, 1.0],
        policy_chosen=-0.2, policy_rejected=-1.0, ref_chosen=-0.2, ref_rejected=-0.2, beta=1.0)
    only = mpt.mpo_combined_loss(
        ["sigmoid"], [1.0],
        policy_chosen=-0.2, policy_rejected=-1.0, ref_chosen=-0.2, ref_rejected=-0.2, beta=1.0)
    assert set(full["components"]) == {"sigmoid", "bco_pair", "sft"}
    assert only["components"] == {"sigmoid": full["components"]["sigmoid"]}
    assert full["total"] != only["total"]
    with pytest.raises(TaskExecutionError, match="未知 MPO loss"):
        mpt.mpo_combined_loss(["nope"], [1.0], policy_chosen=0, policy_rejected=0,
                              ref_chosen=0, ref_rejected=0)


def test_ref_must_be_frozen_and_disjoint_and_ablation_is_within_mpo():
    class Param:
        def __init__(self, grad):
            self.requires_grad = grad

        def numel(self):
            return 3

    class Model:
        def __init__(self, params):
            self._params = params

        def named_parameters(self):
            return list(self._params.items())

    policy = Model({"lora_a": Param(True)})
    ref = Model({"base.weight": Param(False)})
    evidence = dt.verify_policy_ref_binding(policy, ref)
    assert evidence["policy_ref_disjoint"] is True and evidence["ref_trainable_parameters"] == 0
    with pytest.raises(TaskExecutionError, match="同一模型"):
        dt.verify_policy_ref_binding(policy, policy)
    with pytest.raises(TaskExecutionError, match="可训练参数"):
        dt.verify_policy_ref_binding(policy, Model({"lora_b": Param(True)}))
    mpo = recipes()[MPO_ID]
    derived = mpt.config_recipe(mpo, "sigmoid_only")
    assert derived.get("execution.config_id") == "sigmoid_only"
    assert derived.get("train.loss_types") == ["sigmoid"]
    assert derived.get("model.base") == mpo.get("model.base")
    again = mpt.config_recipe(derived, "sigmoid_only")
    assert again.data["train"]["loss_types"] == derived.data["train"]["loss_types"]
    with pytest.raises(TaskExecutionError, match="config_recipe"):
        mpt.config_recipe(recipes()[DPO_ID], "full")


def test_router_rejects_cross_task_protocols():
    dpo = recipes()[DPO_ID]
    stolen = Recipe(dpo.id, dpo.path, copy.deepcopy(dpo.data))
    stolen.data["execution"]["task_id"] = mpt.TASK_ID
    with pytest.raises(TaskExecutionError, match="冲突"):
        tr.resolve(stolen)
    with pytest.raises(TaskExecutionError):
        tr.load(stolen, "suite")


def test_suite_and_export_plans_fail_closed_for_full_runs(tmp_path):
    loaded = recipes()
    dpo_plan = ds.suite_plan(loaded[DPO_ID], [42, 43, 44], tmp_path / "dpo-suite")
    assert dpo_plan["execution_supported"] is True and dpo_plan["cross_backbone_attribution_valid"] is False
    assert len(dpo_plan["runs"]) == 3 and dpo_plan["baseline_promotion"] == "none"
    mpo_plan = ms.suite_plan(loaded[MPO_ID], [42, 43, 44], tmp_path / "mpo-suite")
    assert mpo_plan["execution_supported"] is True
    assert mpo_plan["ablation_configs"] == ["full", "sigmoid_only", "no_sft"]
    assert mpo_plan["cross_backbone_attribution_valid"] is False and len(mpo_plan["runs"]) == 9
    for module, recipe in ((de, loaded[DPO_ID]), (me, loaded[MPO_ID])):
        plan = module.export_bundle(
            recipe, tmp_path / "absent-checkpoint", tmp_path / "absent-eval.json",
            tmp_path / "bundle", execute=False)
        assert plan["execution_supported"] is True
        assert plan["base_weights_bundled"] is False and plan["ref_weights_bundled"] is False
        assert plan["cross_backbone_attribution_valid"] is False
        assert "not_declared" in plan["tolerance_status"]
        assert plan["license"]["status"] == "pending"
        assert plan["vision_contract"]["explicit_image_token_mask"] is True


def test_cli_dry_run_dispatches_and_full_does_not_create_evidence(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "RECIPE_LAB_ROOT": str(ROOT)}
    for recipe_id, task_id in ((DPO_ID, dt.TASK_ID), (MPO_ID, mpt.TASK_ID)):
        result = subprocess.run(
            [sys.executable, str(ROOT / "runner" / "train.py"), "--recipe", recipe_id,
             "--dry-run", "--json"],
            env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["execution_supported"] is True and plan["task_id"] == task_id
        suite = subprocess.run(
            [sys.executable, str(ROOT / "runner" / "suite.py"), "--recipe", recipe_id,
             "--seeds", "42", "43", "44", "--output-root", str(tmp_path / recipe_id / "suite"),
             "--dry-run"],
            env=env, capture_output=True, text=True, check=False)
        assert suite.returncode == 0, suite.stderr
        suite_plan = json.loads(suite.stdout)
        assert "release_pipeline" not in suite_plan
        assert suite_plan["task_id"] == task_id and suite_plan["cross_backbone_attribution_valid"] is False
        export = subprocess.run(
            [sys.executable, str(ROOT / "runner" / "export.py"), "--recipe", recipe_id,
             "--checkpoint", "missing", "--evaluation", "missing",
             "--output", str(tmp_path / recipe_id / "bundle"), "--dry-run"],
            env=env, capture_output=True, text=True, check=False)
        assert export.returncode == 0, export.stderr
        assert "not_declared" in json.loads(export.stdout)["tolerance_status"]
        full = subprocess.run(
            [sys.executable, str(ROOT / "runner" / "suite.py"), "--recipe", recipe_id,
             "--seeds", "42", "43", "44", "--output-root", str(tmp_path / recipe_id / "full"),
             "--full"],
            env=env, capture_output=True, text=True, check=False)
        assert full.returncode != 0
        assert "frozen_environment" in (full.stdout + full.stderr)
        assert not (tmp_path / recipe_id / "full").exists()
