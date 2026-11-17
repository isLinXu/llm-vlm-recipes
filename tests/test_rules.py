"""规则有效性测试：既要证明「好的能通过」，也要证明「坏的会被拦住」。

一个永远返回 0 error 的验证器毫无价值，所以这里大量使用故意写错的配置。
"""

from __future__ import annotations

from dataclasses import replace

from conftest import bad_data, make_recipe, mutate, valid_data

from recipe_lab.rules import Context
from recipe_lab.runner import validate


def findings_for(ctx: Context, data: dict, level: int = 3):
    recipe = make_recipe(ctx, data)
    single = Context(
        recipes=[recipe], taxonomy=ctx.taxonomy, root=ctx.root, level=level
    )
    return validate(single).findings


def error_rules(ctx: Context, data: dict) -> set[str]:
    return {f.rule for f in findings_for(ctx, data) if f.severity == "error"}


def all_rules(ctx: Context, data: dict) -> set[str]:
    return {f.rule for f in findings_for(ctx, data)}


# ---------------------------------------------------------------- 正向


def test_valid_recipe_has_zero_errors(ctx: Context) -> None:
    assert error_rules(ctx, valid_data()) == set()


def test_full_repository_has_zero_errors(ctx: Context) -> None:
    report = validate(ctx)
    assert report.errors == [], "\n".join(f.render() for f in report.errors)


# ---------------------------------------------------------------- L1 契约


def test_bad_vlm_fixture_triggers_all_four_contract_rules(ctx: Context) -> None:
    hits = error_rules(ctx, bad_data())
    assert {"R105", "R107", "R108", "R109", "R110"} <= hits, hits


def test_trainer_dataset_type_mismatch(ctx: Context) -> None:
    assert "R104" in error_rules(ctx, mutate(valid_data(), "dataset.type", "prompt-only"))


def test_unknown_family(ctx: Context) -> None:
    assert "R101" in error_rules(ctx, mutate(valid_data(), "method.family", "made-up"))


def test_wrong_trainer_for_family(ctx: Context) -> None:
    assert "R103" in error_rules(ctx, mutate(valid_data(), "method.trainer", "DPOTrainer"))


def test_removed_ppo_family_warns(ctx: Context) -> None:
    data = mutate(valid_data(), "method.family", "ppo")
    data = mutate(data, "method.trainer", "PPOTrainer")
    data = mutate(data, "dataset.type", "prompt-only")
    data = mutate(data, "modality_contract.max_prompt_length", 512)
    data["rewards"] = {"functions": ["r"], "failure_path_tested": True}
    warns = {f.rule for f in findings_for(ctx, data) if f.severity == "warning"}
    assert "R102" in warns


def test_train_inference_system_message_mismatch(ctx: Context) -> None:
    data = mutate(valid_data(), "consistency", {
        "train_system_message": True,
        "inference_system_message": False,
    })
    assert "R111" in error_rules(ctx, data)


def test_dora_label_mismatch_warns(ctx: Context) -> None:
    data = mutate(valid_data(), "tags", ["qlora", "vlm"])
    data = mutate(data, "peft", {"use_dora": True})
    warns = {f.rule for f in findings_for(ctx, data) if f.severity == "warning"}
    assert "R112" in warns


def test_grpo_requires_reward_functions(ctx: Context) -> None:
    data = mutate(valid_data(), "method.family", "grpo")
    data = mutate(data, "method.trainer", "GRPOTrainer")
    data = mutate(data, "dataset.type", "prompt-only")
    data = mutate(data, "modality_contract.max_prompt_length", 512)
    assert "R115" in error_rules(ctx, data)


def _prompt_only_vlm(pre_rendered: bool) -> dict:
    """VLM GRPO 的 prompt-only 形态：上游用 apply_chat_template 预渲染成字符串。"""
    data = mutate(valid_data(), "method.family", "grpo")
    data = mutate(data, "method.trainer", "GRPOTrainer")
    data = mutate(data, "dataset.type", "prompt-only")
    data = mutate(data, "dataset.structure", "standard")
    data = mutate(data, "modality_contract.images_column", "image")
    data = mutate(data, "modality_contract.content_list_dict", False)
    data = mutate(data, "modality_contract.max_prompt_length", 2048)
    data = mutate(data, "rewards", {
        "functions": ["format_reward", "accuracy_reward"],
        "failure_path_tested": True,
    })
    if pre_rendered:
        data = mutate(data, "modality_contract.pre_rendered_prompt", True)
    return data


def test_prompt_only_vlm_pre_rendered_is_a_registered_exception(ctx: Context) -> None:
    """evidence.toml F07：prompt-only VLM 的 image 列 + 字符串 prompt 是合法例外。"""
    fs = findings_for(ctx, _prompt_only_vlm(pre_rendered=True))
    errors = {f.rule for f in fs if f.severity == "error"}
    assert "R105" not in errors, errors
    assert "R108" not in errors, errors
    infos = {f.rule for f in fs if f.severity == "info"}
    assert "R105" in infos and "R108" in infos


def test_undeclared_prompt_only_vlm_still_fails(ctx: Context) -> None:
    """没有 pre_rendered_prompt 声明，同样的配置必须被 R105/R108 拦下。"""
    errors = error_rules(ctx, _prompt_only_vlm(pre_rendered=False))
    assert {"R105", "R108"} <= errors, errors


# ---------------------------------------------------------------- L2 版本


def test_floating_dependency_is_rejected(ctx: Context) -> None:
    assert "R201" in error_rules(ctx, mutate(valid_data(), "dependencies.trl", ">=0.14.0"))


def test_dev_version_warns(ctx: Context) -> None:
    data = mutate(valid_data(), "dependencies.trl", "0.22.0.dev0")
    warns = {f.rule for f in findings_for(ctx, data) if f.severity == "warning"}
    assert "R203" in warns


def test_unverified_version_pair_warns(ctx: Context) -> None:
    data = mutate(valid_data(), "dependencies", {"trl": "9.9.9", "transformers": "9.9.9"})
    assert "R202" in all_rules(ctx, data)


def test_bad_source_url(ctx: Context) -> None:
    assert "R205" in error_rules(ctx, mutate(valid_data(), "source", "http://example.com/x"))


# ---------------------------------------------------------------- L3 纪律


def test_baseline_passing_blocks_training(ctx: Context) -> None:
    data = mutate(valid_data(), "baseline.result", "passes")
    assert "R302" in error_rules(ctx, data)


def test_baseline_passing_on_cataloged_is_gate_info(ctx: Context) -> None:
    data = mutate(valid_data(), "status", "cataloged")
    data = mutate(data, "baseline.result", "passes")
    assert error_rules(ctx, data) == set()
    assert "R303" in all_rules(ctx, data)


def test_held_out_is_mandatory(ctx: Context) -> None:
    assert "R305" in error_rules(ctx, mutate(valid_data(), "evaluation.held_out", False))


def test_qualitative_metric_is_rejected(ctx: Context) -> None:
    assert "R304" in error_rules(ctx, mutate(valid_data(), "evaluation.metrics", ["vibe_check"]))


def test_inverted_smoke_budget(ctx: Context) -> None:
    assert "R308" in error_rules(ctx, mutate(valid_data(), "budget.full_steps", 3))


def test_rl_requires_reward_failure_path(ctx: Context) -> None:
    data = mutate(valid_data(), "method.family", "grpo")
    data = mutate(data, "method.trainer", "GRPOTrainer")
    data = mutate(data, "dataset.type", "prompt-only")
    data = mutate(data, "modality_contract.max_prompt_length", 512)
    data = mutate(data, "rewards", {"functions": ["format_reward"]})
    assert "R307" in error_rules(ctx, data)


def test_known_issues_need_disposition(ctx: Context) -> None:
    data = mutate(valid_data(), "known_issues", {"items": ["某个上游 bug"]})
    data = mutate(data, "todo", {"items": []})
    data = mutate(data, "upstream_deviations", {"items": []})
    warns = {f.rule for f in findings_for(ctx, data) if f.severity == "warning"}
    assert "R309" in warns


# ---------------------------------------------------------------- L0 / 全局


def test_id_path_mismatch(ctx: Context) -> None:
    # path 必须保持不变，只改 id，才能真的触发不一致。
    recipe = replace(make_recipe(ctx, valid_data()), id="vlm/sft/somewhere-else")
    single = Context(
        recipes=[recipe], taxonomy=ctx.taxonomy, root=ctx.root, level=3
    )
    hits = {f.rule for f in validate(single).findings if f.severity == "error"}
    assert "R002" in hits


def test_missing_required_field(ctx: Context) -> None:
    data = valid_data()
    del data["source"]
    assert "R001" in error_rules(ctx, data)


def test_version_drift_detected_globally(ctx: Context) -> None:
    rules = {f.rule for f in validate(ctx).findings}
    assert "R204" in rules
    assert "R207" in rules
