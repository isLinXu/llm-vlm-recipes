"""算力 / 数据规模规划登记的一致性：规划表不得与 recipe.toml / methodology.toml 漂移，也不得冒充实测。

只读 registry/compute_planning.toml、registry/methodology.toml 与 recipes/**/recipe.toml；不跑 GPU，不改任何生成物。
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path

import pytest

from recipe_lab.loader import load_recipes

ROOT = Path(__file__).resolve().parents[1]
PLAN = tomllib.loads((ROOT / "registry" / "compute_planning.toml").read_text(encoding="utf-8"))
METH = tomllib.loads((ROOT / "registry" / "methodology.toml").read_text(encoding="utf-8"))
DOC = (ROOT / "docs" / "COMPUTE-PLANNING.md").read_text(encoding="utf-8")
RECIPES = {r.id: r for r in load_recipes(ROOT)}
ROWS = {row["id"]: row for row in PLAN["recipe"]}
RULES = {rule["id"]: rule for rule in PLAN["memory_rule"]}


def _first(r, *paths):
    for path in paths:
        value = r.get(path)
        if value is not None:
            return value
    return None


def _fmt_lr(v):
    return "—" if v is None else f"{v:.0e}".replace("e-0", "e-")


def test_covers_exactly_the_registered_recipes():
    assert set(ROWS) == set(RECIPES)
    assert len(PLAN["recipe"]) == len(ROWS) == 14


def test_meta_never_claims_gpu_evidence():
    meta = PLAN["meta"]
    assert meta["estimates_are_not_evidence"] is True
    assert meta["gpu_verified"] == 0
    assert meta["min_independent_seeds_for_release"] == 3
    assert all(band["estimate"] is True for band in PLAN["utilization_band"])
    assert PLAN["memory"]["estimate"] is True
    assert all(tier["heuristic"] is True for tier in PLAN["tier"])
    assert all(item["estimate"] is True for item in PLAN["parallel_strategy"])
    assert all(item["estimate"] is True for item in PLAN["gpu_tier_starter"])


@pytest.mark.parametrize("rid", sorted(ROWS))
def test_scale_facts_match_recipe_toml(rid):
    row, r = ROWS[rid], RECIPES[rid]
    assert row["dataset_name"] == r.get("dataset.name")
    assert row.get("micro_batch") == _first(
        r, "train.per_device_train_batch_size", "hardware.per_device_train_batch_size")
    assert row.get("grad_accum") == _first(
        r, "train.gradient_accumulation_steps", "budget.full_grad_accum")
    if r.get("train.max_steps") is not None:
        assert row["steps_kind"] == "max_steps"
        assert row["steps_full"] == r.get("train.max_steps")
    else:
        assert row["steps_kind"] == "budget_cap"
        assert row["steps_full"] == r.get("budget.full_steps")
    assert row.get("num_generations") == r.get("train.num_generations")
    assert row.get("grad_ckpt") == r.get("train.gradient_checkpointing")
    assert row.get("optim") == r.get("train.optim")
    if r.get("train.learning_rate") is None:
        assert row.get("learning_rate") is None
    else:
        assert row["learning_rate"] == pytest.approx(r.get("train.learning_rate"))
    assert row["hardware_declared"] == r.get("resources.gpu")
    assert row["gpus"] == 1
    assert row["release_runs_with_zeroshot"] == 1 + row["release_configs"] * 3


@pytest.mark.parametrize("rid", sorted(ROWS))
def test_precision_and_quantization_match_recipe_toml(rid):
    row, r = ROWS[rid], RECIPES[rid]
    bf16, fp16 = r.get("train.bf16"), r.get("train.fp16")
    qdtype = r.get("quantization.compute_dtype") or r.get("quantization.bnb_4bit_compute_dtype")
    if bf16 is False and qdtype == "bfloat16":
        assert row["precision_mode"] == "bf16_false_mismatch"
    elif bf16 is True:
        assert row["precision_mode"] == "bf16"
    elif fp16 is True:
        assert row["precision_mode"] == "fp16"
    else:
        assert row["precision_mode"] == "unspecified"

    if r.get("quantization.load_in_4bit") is True:
        assert row["quantization_mode"].startswith("nf4_4bit")
    elif r.get("quantization.mode") == "none":
        assert row["quantization_mode"] == "none"
    else:
        assert row["quantization_mode"] == "unspecified"


def test_release_config_multipliers_match_suites():
    assert ROWS["llm/sft/rapidfire-multiconfig"]["release_configs"] == len(
        RECIPES["llm/sft/rapidfire-multiconfig"].get("configs") or [])
    assert ROWS["vlm/mpo/qwen2-5-vl-3b-mpo"]["release_configs"] == 3
    assert ROWS["llm/grpo/qwen2-5-3b-advanced-reward"]["release_configs"] == 5
    for rm in PLAN["release_multiplier"]:
        assert rm["total_runs"] == rm["zero_shot_runs"] + rm["configs"] * rm["seeds_per_config"]


@pytest.mark.parametrize("rid", sorted(ROWS))
def test_params_b_is_recipe_or_flagged_estimate(rid):
    row, declared = ROWS[rid], RECIPES[rid].get("model.params_b")
    if row["params_b_source"] == "recipe":
        assert declared == row["params_b"]
    else:
        assert row["params_b_source"] == "estimate"
        assert declared is None, "recipe 已声明 params_b，不应再标 estimate"


@pytest.mark.parametrize("rid", [i for i, row in ROWS.items()
                                 if "micro_batch" in row and "num_generations" in row])
def test_grpo_batch_is_multiple_of_generations(rid):
    row = ROWS[rid]
    assert row["micro_batch"] % row["num_generations"] == 0


def test_wordle_anchor_and_budget_projection_table():
    r = RECIPES["agent/grpo/wordle-openenv-hf-jobs"]
    seconds = r.get("throughput.seconds_per_rollout")
    accum = r.get("budget.full_grad_accum")
    hours = (r.get("budget.full_steps") * accum * seconds) / 3600
    assert hours == pytest.approx(r.get("budget.est_full_gpu_hours"), abs=0.1)
    for wb in PLAN["wordle_budget_row"]:
        expect_steps = int((wb["budget_seconds"] * wb["headroom"]) // (seconds * accum))
        assert wb["max_steps"] == expect_steps
        assert wb["fits_full_150"] is (expect_steps >= r.get("budget.full_steps"))


def test_vllm_colocate_table_matches_recipe_benchmark():
    r = RECIPES["llm/grpo/qwen2-0-5b-vllm-colocate"]
    by_len = {row["max_completion_length"]: row for row in PLAN["vllm_colocate_row"]}
    assert sorted(by_len) == [64, 128, 256, 512, 1024]
    assert by_len[512]["vllm_on_its"] == r.get("benchmark.throughput_with_vllm_its")
    assert by_len[512]["vllm_off_its"] == r.get("benchmark.throughput_without_vllm_its")
    assert by_len[512]["speedup"] == r.get("benchmark.speedup_512")
    assert by_len[1024]["speedup"] == r.get("benchmark.speedup_1024")
    assert by_len[512]["in_recipe_toml"] is True and by_len[1024]["in_recipe_toml"] is True


def test_paper_anchors_bind_to_verified_methodology_papers():
    papers = METH["papers"]
    for pa in PLAN["paper_anchor"]:
        pid = pa["paper_id"]
        assert pid in papers, pid
        assert papers[pid]["verified"] is True, pid
        assert papers[pid]["arxiv_id"] == pa["arxiv_id"], pid


def _weights_gb(rule_id: str, params_b: float) -> float:
    rule = RULES[rule_id]
    return rule["coef_per_b"] * params_b + rule["const_gb"]


def test_qlora_rule_reproduces_paper_anchor():
    assert _weights_gb("qlora_sft", 65.0) < 48 * 0.9


def test_matrix_rules_are_monotonic_and_cover_models():
    assert {m["id"] for m in PLAN["matrix_model"]} >= {"0.5B", "3B", "7B", "70B"}
    sizes = [m["params_b"] for m in PLAN["matrix_model"]]
    assert sizes == sorted(sizes)
    for rule_id in RULES:
        values = [_weights_gb(rule_id, p) for p in sizes]
        assert values == sorted(values)
    for params in sizes:
        assert _weights_gb("full_sft", params) > _weights_gb("lora_bf16", params)
        assert _weights_gb("grpo_full", params) > _weights_gb("full_sft", params)
        assert _weights_gb("qlora_sft", params) < _weights_gb("lora_bf16", params)


def test_every_recipe_method_has_a_memory_rule():
    assert {row["method"] for row in ROWS.values()} <= set(RULES) | {"grpo"}


def test_doc_matrix_matches_rules():
    """文档矩阵（模型 × 方法）的推荐显存必须与规则重算一致，防止改系数不改表。"""
    headroom = PLAN["memory"]["headroom"]
    tiers = PLAN["memory"]["single_card_tiers_gb"]
    cap = PLAN["memory"]["multi_card_assume_gb"]
    lines = DOC.splitlines()
    for model in PLAN["matrix_model"]:
        row = next(line for line in lines if line.startswith(f"| {model['id']} |"))
        cells = [c.strip() for c in row.strip().strip("|").split("|")][1:]
        assert len(cells) == len(RULES), model["id"]
        for cell, rule_id in zip(cells, RULES):
            need = _weights_gb(rule_id, model["params_b"]) * headroom
            tier = next((t for t in tiers if need <= t), None)
            expect = f"{need:.0f}GB→{tier}GB×1" if tier else f"{need:.0f}GB→{math.ceil(need / cap)}×{cap}GB"
            assert cell == expect, (model["id"], rule_id, cell, expect)


def test_doc_mentions_every_recipe_tier_and_honesty_flags():
    for rid in ROWS:
        assert f"`{rid}`" in DOC, rid
    for tier in PLAN["tier"]:
        assert tier["name"] in DOC, tier["id"]
    for gate in PLAN["stage_gate"]:
        assert f"{gate['id']} {gate['name']}" in DOC, gate["id"]
    for anchor in PLAN["anchor"]:
        assert anchor["value"].split("（")[0].split("：")[0] in DOC, anchor["id"]
    for pa in PLAN["paper_anchor"]:
        assert f"`{pa['arxiv_id']}`" in DOC, pa["paper_id"]
    for phrase in ("估算", "非实测", "GPU verified 0", "不是 GPU 实跑证据"):
        assert phrase in DOC, phrase


@pytest.mark.parametrize("rid", sorted(ROWS))
def test_doc_recipe_rows_match_registry(rid):
    """文档表 2a 与表 2b 的字段必须由登记重算一致。"""
    row = ROWS[rid]
    lines = [x for x in DOC.splitlines() if x.startswith(f"| `{rid}` |")]
    assert len(lines) == 2, rid
    cells_2a = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    cells_2b = [c.strip() for c in lines[1].strip().strip("|").split("|")]
    mb, ga, g = row.get("micro_batch"), row.get("grad_accum"), row.get("num_generations")
    eff = mb * ga * row["gpus"] if mb and ga else None

    def show(value):
        return "—" if value is None else str(value)

    star = "*" if row["params_b_source"] == "estimate" else ""
    kind = "cap" if row["steps_kind"] == "budget_cap" else "max"
    expect_2a = [
        row["method"], f"{row['params_b']}{star}", f"`{row['dataset_name']}`",
        row["dataset_scale_note"], show(mb), show(ga), show(eff), show(g),
        show(eff // g if eff and g else None), f"{row['steps_full']} ({kind})",
        show(eff * row["steps_full"] if eff else None), row["hardware_declared"],
    ]
    assert cells_2a[1:13] == expect_2a, (rid, cells_2a[1:13], expect_2a)

    gc = row.get("grad_ckpt")
    gc_str = "—" if gc is None else ("是" if gc else "否")
    runs_str = f"{row['release_runs_with_zeroshot']}（{row['release_configs']}×3 + 1 zero-shot）"
    expect_2b = [
        row["precision_mode"], row["quantization_mode"], row["peft_spec"],
        row["seq_or_length"], gc_str, row.get("optim") or "—",
        _fmt_lr(row.get("learning_rate")), runs_str,
    ]
    assert cells_2b[1:9] == expect_2b, (rid, cells_2b[1:9], expect_2b)


def test_doc_vllm_and_wordle_tables_match_registry():
    lines = DOC.splitlines()
    for vr in PLAN["vllm_colocate_row"]:
        prefix = f"| {vr['max_completion_length']} | {vr['num_generations']} |"
        assert any(line.startswith(prefix) and vr["speedup"] in line for line in lines), vr
    for wb in PLAN["wordle_budget_row"]:
        prefix = f"| {wb['budget_seconds']} | {wb['flavor_hint']} |"
        assert any(line.startswith(prefix) and f"| {wb['max_steps']} |" in line for line in lines), wb


def test_doc_does_not_claim_measured_utilization():
    assert "没有任何" in DOC and "利用率" in DOC
    assert "已验证" not in DOC
