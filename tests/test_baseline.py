"""baseline.py 与 L6 规则：状态由 checks 重算，不允许手工抬高。

同时做真实仓库集成测试：14 篇声明的 state 必须与重算结果一致，
L0–L6 全量校验必须 0 error。
"""

from __future__ import annotations

from pathlib import Path

from conftest import baseline_data, make_recipe, mutate

from recipe_lab.baseline import (
    STATE_ORDER,
    audit_one,
    check_passed,
    check_stats,
    compute_state,
    group_totals,
    load_contract,
    render_report,
    render_text,
    summarize,
)
from recipe_lab.loader import load_recipes, load_taxonomy
from recipe_lab.rules import Context
from recipe_lab.runner import validate
from test_rules import findings_for


def _ctx6(root: Path) -> Context:
    return Context(
        recipes=load_recipes(root), taxonomy=load_taxonomy(root), root=root, level=6
    )


# ---------------------------------------------------------------- 计算内核


def test_compute_state_is_monotonic(root: Path) -> None:
    contract = load_contract(root)
    ids = [c["id"] for c in contract["checks"]]
    assert compute_state({i: False for i in ids}, contract) == "B0"
    b1 = {i: c["min_state"] == "B1" for i, c in (
        (x["id"], x) for x in contract["checks"])}
    assert compute_state(b1, contract) == "B1"
    all_pass = {i: True for i in ids}
    assert compute_state(all_pass, contract) == "B3"


def test_check_passed_all_kinds(root: Path, ctx: Context) -> None:
    r = make_recipe(ctx, baseline_data())
    assert check_passed(r, {"kind": "field", "field": "baseline_contract.experiment_id"})
    assert not check_passed(r, {"kind": "field", "field": "does.not.exist"})
    assert check_passed(r, {"kind": "equals", "field": "evaluation.multi_run", "expect": True})
    assert not check_passed(r, {"kind": "equals", "field": "evaluation.multi_run", "expect": False})
    assert check_passed(r, {"kind": "not_in", "field": "baseline.result", "expect": ["unmeasured"]})
    assert not check_passed(r, {"kind": "not_in", "field": "baseline.result", "expect": ["fails"]})
    # 虚拟路径下 file 类必然失败，这正是 fixture 重算为 B0 的原因
    assert not check_passed(r, {"kind": "file", "field": "execution.entrypoint"})
    try:
        check_passed(r, {"kind": "wat", "field": "x"})
    except ValueError:
        pass
    else:
        raise AssertionError("未知 kind 必须报错，不能静默返回 False")


def test_file_check_resolves_relative_to_recipe_dir(root: Path) -> None:
    """真实 14 篇的 execution.* 都指向存在的 runner 文件。"""
    contract = load_contract(root)
    file_checks = [c for c in contract["checks"] if c["kind"] == "file"]
    for r in load_recipes(root):
        for c in file_checks:
            if c["min_state"] == "B1":
                assert check_passed(r, c), f"{r.id} 未通过 {c['id']}: {c['field']}"


def test_audit_reports_gaps_and_groups(root: Path, ctx: Context) -> None:
    a = audit_one(make_recipe(ctx, baseline_data()), load_contract(root))
    assert a.computed == "B0" and a.state_ok
    assert a.n_total == len(load_contract(root)["checks"])
    assert a.n_passed < a.n_total
    assert set(a.groups) == {"G1", "G2", "G3", "G4"}
    assert a.blocking, "B0 应列出所有 B1 阻塞项"


def test_summaries(root: Path) -> None:
    contract = load_contract(root)
    audits = [audit_one(r, contract) for r in load_recipes(root)]
    dist = summarize(audits)
    assert set(dist) == {"B0", "B1", "B2", "B3"}
    assert sum(dist.values()) == len(audits)
    totals = group_totals(audits)
    assert all(0 <= p <= t for p, t in totals.values())
    stats = check_stats(audits, contract)
    assert [n for _, n in stats] == sorted(n for _, n in stats), "通过率应升序"


def test_render_helpers_are_nonempty(root: Path) -> None:
    contract = load_contract(root)
    audits = [audit_one(r, contract) for r in load_recipes(root)]
    assert "Baseline Readiness" in render_text(audits)
    report = render_report(audits, contract)
    assert "# Baseline Readiness 与系统能力审计" in report
    assert "本机无法确认的部分" in report
    assert "未完成GPU运行" in report, "报告必须区分实现与真实GPU验证"
    assert "adapter_status" in report, "报告必须说明适配状态字段"


# ---------------------------------------------------------------- L6 规则


def test_self_consistent_baseline_data_has_no_l6_errors(ctx: Context) -> None:
    errors = [f for f in findings_for(ctx, baseline_data(), level=6) if f.severity == "error"]
    assert errors == [], "\n".join(f.render() for f in errors)


def test_r601_missing_baseline_contract(ctx: Context) -> None:
    data = baseline_data()
    del data["baseline_contract"]
    hits = {f.rule for f in findings_for(ctx, data, level=6) if f.severity == "error"}
    assert "R601" in hits


def test_r602_required_fields_and_enums(ctx: Context) -> None:
    data = baseline_data()
    del data["baseline_contract"]["experiment_id"]
    assert "R602" in {f.rule for f in findings_for(ctx, data, level=6) if f.severity == "error"}

    bad_state = mutate(baseline_data(), "baseline_contract.state", "B9")
    assert "R602" in {f.rule for f in findings_for(ctx, bad_state, level=6) if f.severity == "error"}

    bad_tier = mutate(baseline_data(), "baseline_contract.target_tier", "enterprise")
    assert "R602" in {f.rule for f in findings_for(ctx, bad_tier, level=6) if f.severity == "error"}


def test_r603_inflated_state_is_rejected(ctx: Context) -> None:
    inflated = mutate(baseline_data(), "baseline_contract.state", "B1")
    hits = {f.rule for f in findings_for(ctx, inflated, level=6) if f.severity == "error"}
    assert "R603" in hits


def test_r604_b0_without_todo_warns(ctx: Context) -> None:
    data = baseline_data()
    del data["todo"]
    warns = {f.rule for f in findings_for(ctx, data, level=6) if f.severity == "warning"}
    assert "R604" in warns


def test_r605_claiming_b2_without_evidence(ctx: Context) -> None:
    inflated = mutate(baseline_data(), "baseline_contract.state", "B2")
    findings = findings_for(ctx, inflated, level=6)
    assert "R605" in {f.rule for f in findings if f.severity == "error"}
    assert "R603" in {f.rule for f in findings if f.severity == "error"}


def test_r606_gap_to_target_is_info(ctx: Context) -> None:
    rules = {f.rule for f in findings_for(ctx, baseline_data(), level=6)}
    assert "R606" in rules


def test_state_order_is_complete() -> None:
    assert STATE_ORDER == {"B0": 0, "B1": 1, "B2": 2, "B3": 3}


# ---------------------------------------------------------------- 真实仓库集成


def test_all_14_declared_states_match_computed(root: Path) -> None:
    contract = load_contract(root)
    recipes = load_recipes(root)
    assert len(recipes) == 14
    for r in recipes:
        a = audit_one(r, contract)
        assert a.state_ok, f"{r.id}: 声明 {a.claimed}，重算 {a.computed}"


def test_actual_ready_claims_require_valid_evidence(root: Path) -> None:
    from recipe_lab.baseline_evidence import suite_problems
    for r in load_recipes(root):
        a = audit_one(r, load_contract(root))
        assert a.state_ok
        if STATE_ORDER[a.computed] >= STATE_ORDER["B2"]:
            assert not suite_problems(r), "不能只改声明晋级"


def test_full_repo_l6_has_zero_errors(root: Path) -> None:
    report = validate(_ctx6(root))
    assert report.errors == [], "\n".join(f.render() for f in report.errors)


def test_declared_revisions_are_commit_shas(root: Path) -> None:
    """允许日后真实锁定；格式验证不宣称hash真实性已经核实。"""
    from recipe_lab.baseline_evidence import HEX40
    for r in load_recipes(root):
        for field in ("model.revision", "dataset.revision", "source_sha"):
            value = r.get(field)
            if value:
                assert isinstance(value, str) and HEX40.fullmatch(value), (r.id, field)


def test_every_recipe_has_an_execution_command(root: Path) -> None:
    for r in load_recipes(root):
        assert r.get("execution.command"), r.id
        assert r.get("execution.adapter_status"), r.id
        assert r.get("execution.entrypoint"), r.id
        assert r.get("execution.lockfile"), r.id


def test_every_recipe_has_seed_and_split(root: Path) -> None:
    for r in load_recipes(root):
        assert isinstance(r.get("train.seed"), int), r.id
        assert isinstance(r.get("train.data_seed"), int), r.id
        assert r.get("evaluation.split"), r.id
        assert r.get("evaluation.multi_run") is True, r.id


def test_no_duplicate_execution_tables(root: Path) -> None:
    for r in load_recipes(root):
        text = r.path.read_text(encoding="utf-8")
        assert text.count("[execution]") == 1, r.id
        assert text.count("[baseline_contract]") == 1, r.id
