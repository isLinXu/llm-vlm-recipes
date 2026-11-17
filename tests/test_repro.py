"""可复现性审计测试。

核心立场：等级是**算出来的**，不是填出来的。所以这里既测等级函数本身，
也测 R503 会不会把虚报的等级打回 error，还测「0/14 可复现」这个结论
有没有被后续改动悄悄改掉。
"""

from __future__ import annotations

from pathlib import Path

from conftest import make_recipe, mutate, repro_data

from recipe_lab import cli
from recipe_lab.repro import (
    DIMS,
    STATE_ORDER,
    audit_one,
    compute_level,
    evidence_ref_problems,
    gap_counts,
    load_contract,
    load_evidence,
    render_report,
    render_text,
    state_rank,
    summarize,
)
from recipe_lab.rules import Context
from recipe_lab.runner import validate

ALL_MISSING = {d: "missing" for d in DIMS}


def _dims(**overrides: str) -> dict[str, str]:
    return {**ALL_MISSING, **overrides}


# ---------------------------------------------------------------- 等级计算


def test_state_rank_order() -> None:
    assert state_rank("missing") < state_rank("declared")
    assert state_rank("declared") < state_rank("pinned")
    assert state_rank("pinned") < state_rank("verified")
    assert state_rank("不存在的状态") == -1
    assert STATE_ORDER["pinned"] == 2


def test_compute_level_l0(root: Path) -> None:
    contract = load_contract(root)
    assert compute_level(ALL_MISSING, contract=contract) == "L0"
    # L0 至少要有 provenance，完全空白也归 L0（不会返回越界值）
    assert compute_level({}, contract=contract) == "L0"


def test_compute_level_l1(root: Path) -> None:
    contract = load_contract(root)
    dims = _dims(**{
        "model": "declared", "data": "declared", "params": "declared",
        "env": "declared", "eval": "declared", "provenance": "declared",
    })
    assert compute_level(dims, contract=contract) == "L1"


def test_compute_level_l2(root: Path) -> None:
    contract = load_contract(root)
    dims = _dims(**{
        "model": "pinned", "data": "pinned", "params": "pinned",
        "env": "pinned", "seed": "pinned", "eval": "declared",
        "provenance": "pinned",
    })
    assert compute_level(dims, contract=contract) == "L2"


def test_compute_level_l3(root: Path) -> None:
    contract = load_contract(root)
    dims = _dims(**{
        "model": "pinned", "data": "pinned", "params": "pinned",
        "env": "pinned", "seed": "pinned", "code": "pinned",
        "eval": "pinned", "provenance": "pinned",
    })
    assert compute_level(dims, contract=contract) == "L3"


def test_compute_level_l4_requires_verified_code_and_eval(root: Path) -> None:
    contract = load_contract(root)
    base = {
        "model": "pinned", "data": "pinned", "params": "pinned",
        "env": "pinned", "seed": "pinned", "provenance": "pinned",
    }
    l3 = _dims(**{**base, "code": "pinned", "eval": "pinned"})
    l4 = _dims(**{**base, "code": "verified", "eval": "verified"})
    assert compute_level(l3, contract=contract) == "L3"
    assert compute_level(l4, contract=contract) == "L4"


def test_single_missing_dim_caps_the_level(root: Path) -> None:
    """最短板决定等级：L2 条件少一个 seed 就掉回 L1（甚至 L0）。"""
    contract = load_contract(root)
    almost = _dims(**{
        "model": "pinned", "data": "pinned", "params": "pinned",
        "env": "pinned", "provenance": "pinned",
    })  # 故意缺 seed / eval
    assert compute_level(almost, contract=contract) in ("L0", "L1")


# ---------------------------------------------------------------- 全仓审计结论


def test_full_repository_l5_has_zero_errors(ctx: Context) -> None:
    full = Context(recipes=ctx.recipes, taxonomy=ctx.taxonomy, root=ctx.root, level=5)
    report = validate(full)
    assert report.errors == [], "\n".join(f.render() for f in report.errors)


def test_no_recipe_inflates_its_level(ctx: Context, root: Path) -> None:
    contract = load_contract(root)
    audits = [audit_one(r, contract) for r in ctx.recipes]
    assert len(audits) == 14
    assert all(a.level_ok for a in audits), [a.recipe_id for a in audits if not a.level_ok]


def test_audit_conclusion_zero_reproducible(ctx: Context, root: Path) -> None:
    """把「0/14 达到 L2」钉成回归测试：谁把数字改漂亮了，测试会先响。"""
    contract = load_contract(root)
    audits = [audit_one(r, contract) for r in ctx.recipes]
    dist = summarize(audits)
    assert contract["reproducible_threshold"] == "L2"
    assert sum(dist[k] for k in ("L2", "L3", "L4")) == 0
    assert dist["L0"] + dist["L1"] == len(audits)


def test_every_dimension_is_blocking_somewhere(ctx: Context, root: Path) -> None:
    contract = load_contract(root)
    audits = [audit_one(r, contract) for r in ctx.recipes]
    counts = gap_counts(audits)
    assert set(counts) == set(DIMS)
    assert all(v > 0 for v in counts.values()), counts


def test_evidence_coverage_is_14_of_14(root: Path) -> None:
    evidence = load_evidence(root)
    assert evidence["method"]["coverage"] == "14/14"
    assert len(evidence["method"]["local_notebooks"]) == 3
    assert len(evidence["method"]["webfetch_pages"]) == 11


# ---------------------------------------------------------------- 证据引用


def test_all_evidence_refs_are_traceable(ctx: Context, root: Path) -> None:
    contract, evidence = load_contract(root), load_evidence(root)
    problems = evidence_ref_problems(ctx.recipes, contract, evidence)
    assert problems == {}, problems


def test_unregistered_evidence_ref_is_detected(ctx: Context, root: Path) -> None:
    contract, evidence = load_contract(root), load_evidence(root)
    tampered = [make_recipe(ctx, mutate(r.data, "repro.evidence_ref", "webfetch:虚构页面"))
                for r in ctx.recipes[:1]]
    # 保持同一 id，只改引用
    problems = evidence_ref_problems(tampered, contract, evidence)
    assert tampered[0].id in problems
    assert problems[tampered[0].id][0] == "error"


def test_research_report_only_is_a_warning(ctx: Context, root: Path) -> None:
    contract, evidence = load_contract(root), load_evidence(root)
    one = make_recipe(ctx, mutate(ctx.recipes[0].data, "repro.evidence_ref",
                                  "research-report-only"))
    problems = evidence_ref_problems([one], contract, evidence)
    assert problems[one.id][0] == "warning"


# ---------------------------------------------------------------- 报告渲染


def test_render_text_contains_distribution_and_conclusion(ctx: Context, root: Path) -> None:
    contract = load_contract(root)
    audits = [audit_one(r, contract) for r in ctx.recipes]
    text = render_text(audits)
    assert "14 篇" in text
    assert "等级分布" in text
    assert "当前 0 篇" in text


def test_render_report_states_zero_reproducible(ctx: Context, root: Path) -> None:
    contract, evidence = load_contract(root), load_evidence(root)
    audits = [audit_one(r, contract) for r in ctx.recipes]
    text = render_report(audits, contract, evidence)
    assert "0 篇达到「可复现」" in text
    assert "14/14" in text
    assert "配置自洽" in text and "结果可复现" in text
    assert "Qwen2.5-0.5B-Instruct" in text  # 编造值的纠错记录必须保留
    assert text.endswith("\n")


# ---------------------------------------------------------------- CLI 门禁


def test_cli_repro_check_passes(monkeypatch, root: Path, capsys) -> None:
    monkeypatch.setenv("RECIPE_LAB_ROOT", str(root))
    assert cli.main(["repro", "--check"]) == 0
    out = capsys.readouterr().out
    assert "审计完整性通过" in out
    assert "0/14" in out


def test_cli_repro_min_level_l2_fails(monkeypatch, root: Path, capsys) -> None:
    """--check 默认不因为「0/14 可复现」而失败；显式设目标才会失败。"""
    monkeypatch.setenv("RECIPE_LAB_ROOT", str(root))
    assert cli.main(["repro", "--check", "--min-level", "L2"]) == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_cli_repro_min_level_l0_passes(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("RECIPE_LAB_ROOT", str(root))
    assert cli.main(["repro", "--check", "--min-level", "L0"]) == 0


def test_generated_report_is_in_sync(monkeypatch, root: Path, tmp_path, capsys) -> None:
    """docs/REPRODUCIBILITY.md 必须是生成器产物，不能手改。"""
    monkeypatch.setenv("RECIPE_LAB_ROOT", str(root))
    assert cli.main(["repro", "--report", str(tmp_path / "r.md")]) == 0
    capsys.readouterr()
    assert (tmp_path / "r.md").read_text(encoding="utf-8") == (
        root / "docs" / "REPRODUCIBILITY.md"
    ).read_text(encoding="utf-8")


def test_cli_repro_json_shape(monkeypatch, root: Path, capsys) -> None:
    import json

    monkeypatch.setenv("RECIPE_LAB_ROOT", str(root))
    assert cli.main(["repro", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage"] == "14/14"
    assert payload["reproducible"] == 0
    assert len(payload["audits"]) == 14
    assert payload["evidence_ref_problems"] == {}


# ---------------------------------------------------------------- L5 规则


def _l5_errors(ctx: Context, data: dict) -> set[str]:
    recipe = make_recipe(ctx, data)
    single = Context(recipes=[recipe], taxonomy=ctx.taxonomy, root=ctx.root, level=5)
    return {
        f.rule for f in validate(single).findings
        if f.severity == "error" and f.level == "L5"
    }


def _l5_warnings(ctx: Context, data: dict) -> set[str]:
    recipe = make_recipe(ctx, data)
    single = Context(recipes=[recipe], taxonomy=ctx.taxonomy, root=ctx.root, level=5)
    return {
        f.rule for f in validate(single).findings
        if f.severity == "warning" and f.level == "L5"
    }


def test_self_consistent_repro_has_no_l5_errors(ctx: Context) -> None:
    assert _l5_errors(ctx, repro_data()) == set()


def test_missing_repro_section_triggers_r501(ctx: Context) -> None:
    from conftest import valid_data

    assert "R501" in _l5_errors(ctx, valid_data())


def test_missing_required_repro_field_triggers_r502(ctx: Context) -> None:
    data = repro_data()
    del data["repro"]["assessed_by"]
    assert "R502" in _l5_errors(ctx, data)


def test_illegal_level_triggers_r502(ctx: Context) -> None:
    assert "R502" in _l5_errors(ctx, mutate(repro_data(), "repro.level", "L9"))


def test_inflated_level_triggers_r503(ctx: Context) -> None:
    """支点规则：把 L0 手工改成 L4，必须被重算打回。"""
    assert "R503" in _l5_errors(ctx, mutate(repro_data(), "repro.level", "L4"))


def test_missing_dimension_section_triggers_r504(ctx: Context) -> None:
    data = repro_data()
    del data["repro"]["seed"]
    assert "R504" in _l5_errors(ctx, data)


def test_illegal_dimension_state_triggers_r504(ctx: Context) -> None:
    assert "R504" in _l5_errors(
        ctx, mutate(repro_data(), "repro.seed.state", "probably")
    )


def test_pinned_without_backing_triggers_r505(ctx: Context) -> None:
    data = repro_data()
    data = mutate(data, "repro.model.state", "pinned")
    data = mutate(data, "repro.model.gap", "")
    # model 的 backing 是 model.revision，最小 fixture 里没有
    assert "R505" in _l5_errors(ctx, data)


def test_empty_gap_warns_r506(ctx: Context) -> None:
    assert "R506" in _l5_warnings(ctx, mutate(repro_data(), "repro.seed.gap", ""))


def test_unregistered_evidence_ref_triggers_r507(ctx: Context) -> None:
    assert "R507" in _l5_errors(
        ctx, mutate(repro_data(), "repro.evidence_ref", "webfetch:不存在的页面")
    )


def test_stale_assessed_at_warns(ctx: Context) -> None:
    assert "R502" in _l5_warnings(
        ctx, mutate(repro_data(), "repro.assessed_at", "2020-01-01")
    )
