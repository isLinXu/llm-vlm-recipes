"""recipe-lab 命令行入口。

零第三方依赖；未安装包时可用 `PYTHONPATH=src python3 -m recipe_lab ...`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .loader import (
    LoadError,
    find_legacy_configs,
    load_recipes,
    load_taxonomy,
    read_toml,
    repo_root,
)
from .matrix import render as render_matrix
from .rules import LEVEL_NAMES, Context
from .runner import validate


def _root() -> Path:
    start = Path(os.environ["RECIPE_LAB_ROOT"]) if os.environ.get("RECIPE_LAB_ROOT") else None
    return repo_root(start)


def _ctx(root: Path, level: int, only: list[str] | None = None) -> Context:
    return Context(
        recipes=load_recipes(root, only=only),
        taxonomy=load_taxonomy(root),
        root=root,
        level=level,
    )


def _display_path(path: Path, root: Path) -> str:
    """输出路径在仓库内时显示相对路径，否则显示绝对路径。"""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def cmd_doctor(args: argparse.Namespace) -> int:
    print(f"recipe-lab {__version__} · Python {sys.version.split()[0]}")
    try:
        import tomllib  # noqa: F401

        print("  [ok] tomllib 可用（核心工具链零第三方依赖）")
    except ImportError:
        print("  [FAIL] 需要 Python >= 3.11（tomllib）")
        return 1
    try:
        root = _root()
    except LoadError as exc:
        print(f"  [FAIL] {exc}")
        return 1
    print(f"  [ok] 仓库根 {root}")
    taxonomy = load_taxonomy(root)
    print(f"  [ok] taxonomy: {len(taxonomy.get('families', {}))} 个 family")
    recipes = load_recipes(root)
    print(f"  [ok] recipes: {len(recipes)} 篇")
    legacy = find_legacy_configs(root)
    if legacy:
        print(f"  [!] 发现 {len(legacy)} 个遗留 YAML 配置，尚未迁移为 recipe.toml：")
        for p in legacy:
            print(f"        {p.relative_to(root)}")
    bad = []
    for r in recipes:
        try:
            read_toml(r.path)
        except LoadError as exc:
            bad.append(str(exc))
    if bad:
        for b in bad:
            print(f"  [FAIL] {b}")
        return 1
    print("  [ok] 全部 recipe.toml 可解析")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = _root()
    ctx = _ctx(root, 0)
    rows = [r for r in ctx.recipes if not args.family or r.family == args.family]
    rows = [r for r in rows if not args.status or r.status == args.status]
    rows = [r for r in rows if not args.modality or r.modality == args.modality]
    if not rows:
        print("没有匹配的 recipe")
        return 0
    width = max(len(r.id) for r in rows)
    for r in rows:
        print(
            f"{r.id:<{width}}  {r.family:<14} {r.status:<13} "
            f"{r.get('model.base', '-')}"
        )
    print(f"\n共 {len(rows)} 篇")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    root = _root()
    matches = load_recipes(root, only=[args.id])
    if not matches:
        print(f"未找到 recipe: {args.id}", file=sys.stderr)
        return 1
    for r in matches:
        print(f"# {r.id}")
        print(f"# {r.path.relative_to(root)}")
        print(r.path.read_text(encoding="utf-8"))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    root = _root()
    level = LEVEL_NAMES[args.level]
    ctx = _ctx(root, level, only=args.ids or None)
    report = validate(ctx)

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"recipe-lab validate · L0–L{level} · {len(report.recipes)} 篇 recipe\n")
        for r in report.recipes:
            fs = report.for_recipe(r.id)
            e = sum(f.severity == "error" for f in fs)
            w = sum(f.severity == "warning" for f in fs)
            i = sum(f.severity == "info" for f in fs)
            mark = "FAIL" if e else "ok"
            print(f"[{mark:>4}] {r.id}  ({e} error / {w} warning / {i} info)")
            if not args.quiet:
                for f in fs:
                    if f.severity != "info" or args.verbose:
                        print(f.render())
        globals_ = [f for f in report.findings if f.recipe_id == "GLOBAL"]
        if globals_:
            print("\n全局检查")
            for f in globals_:
                print(f.render())
        print(
            f"\n合计: {len(report.errors)} error / "
            f"{len(report.warnings)} warning / {len(report.infos)} info"
        )
        if report.warnings and not args.strict:
            print("warning 是已登记的待办风险，不阻塞；发布前请用 --strict 复核。")

    if report.errors:
        return 1
    if args.strict and report.warnings:
        return 2
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    root = _root()
    ctx = _ctx(root, LEVEL_NAMES[args.level])
    report = validate(ctx)
    text = render_matrix(report)
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"已生成 {_display_path(out, root)}（{len(report.recipes)} 篇）")
    return 0


def _audit_dict(a) -> dict:
    return {
        "recipe_id": a.recipe_id,
        "claimed": a.claimed,
        "computed": a.computed,
        "level_ok": a.level_ok,
        "dims": a.dims,
        "gaps": a.gaps,
        "evidence_ref": a.evidence_ref,
        "blocking_dims": a.blocking_dims,
    }


def cmd_repro(args: argparse.Namespace) -> int:
    """可复现性审计。

    注意 `--check` 校验的是**审计完整性**（等级有没有虚报、证据引用能不能追溯），
    不是「所有 recipe 都必须可复现」。当前 0/14 达到 L2 是审计结论，
    把它当成 CI 失败只会逼人去改数字。要设提升目标请显式加 --min-level。
    """
    from .repro import (
        audit_one,
        evidence_ref_problems,
        load_contract,
        load_evidence,
        render_report,
        render_text,
        summarize,
    )

    root = _root()
    contract = load_contract(root)
    evidence = load_evidence(root)
    ctx = _ctx(root, 5, only=args.ids or None)
    audits = [audit_one(r, contract) for r in ctx.recipes]
    dist = summarize(audits)

    if args.report is not None:
        out = Path(args.report)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_report(audits, contract, evidence), encoding="utf-8")
        print(f"已生成 {_display_path(out, root)}（{len(audits)} 篇）")
        if not args.check and args.min_level is None and not args.json:
            return 0

    if args.json:
        print(json.dumps({
            "assessed_at": contract.get("assessed_at"),
            "coverage": evidence["method"].get("coverage"),
            "distribution": dist,
            "reproducible_threshold": contract.get("reproducible_threshold"),
            "reproducible": sum(dist[k] for k in ("L2", "L3", "L4")),
            "audits": [_audit_dict(a) for a in audits],
            "evidence_ref_problems": {
                rid: {"severity": sev, "message": msg}
                for rid, (sev, msg) in evidence_ref_problems(
                    ctx.recipes, contract, evidence
                ).items()
            },
        }, ensure_ascii=False, indent=2))
    elif args.report is None:
        print(render_text(audits))
        problems = evidence_ref_problems(ctx.recipes, contract, evidence)
        if problems:
            print("\n证据引用问题：")
            for rid, (sev, msg) in sorted(problems.items()):
                print(f"  [{sev}] {rid}: {msg}")

    # ---- 完整性门禁 ----
    if not args.check and args.min_level is None:
        return 0

    failures: list[str] = []
    for a in audits:
        if not a.level_ok:
            failures.append(
                f"{a.recipe_id}: repro.level 声明 {a.claimed or '(空)'}，重算为 {a.computed}"
            )
    for rid, (sev, msg) in evidence_ref_problems(
        ctx.recipes, contract, evidence
    ).items():
        if sev == "error":
            failures.append(f"{rid}: {msg}")

    threshold = args.min_level
    if threshold:
        for a in audits:
            if state_rank_level(a.computed) < state_rank_level(threshold):
                failures.append(
                    f"{a.recipe_id}: 等级 {a.computed} 低于要求 {threshold}"
                )

    if not args.json:
        reproducible = sum(dist[k] for k in ("L2", "L3", "L4"))
        print(f"\nrecipe-lab repro --check · {len(audits)} 篇")
        print(
            f"  等级分布 L0 {dist['L0']} / L1 {dist['L1']} / L2 {dist['L2']} "
            f"/ L3 {dist['L3']} / L4 {dist['L4']}"
        )
        print(
            f"  达到 {contract.get('reproducible_threshold', 'L2')} 及以上（可复现）："
            f"{reproducible}/{len(audits)} —— 这是审计结论，不是校验失败"
        )
        if threshold:
            print(f"  提升目标：--min-level {threshold}")
        if failures:
            print("\n审计完整性未通过：")
            for line in failures:
                print(f"  [FAIL] {line}")
        else:
            print("  [ok] 无虚报等级，evidence_ref 全部可追溯，审计完整性通过")

    return 1 if failures else 0


def state_rank_level(level: str) -> int:
    """L0..L4 -> 0..4，用于 --min-level 比较。"""
    return {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}.get(str(level), -1)


def _baseline_dict(a) -> dict:
    return {
        "recipe_id": a.recipe_id,
        "claimed": a.claimed,
        "computed": a.computed,
        "state_ok": a.state_ok,
        "target_tier": a.target_tier,
        "groups": {g: list(v) for g, v in a.groups.items()},
        "passed": a.passed,
        "gaps": [g["id"] for g in a.gaps],
    }


def cmd_baseline(args: argparse.Namespace) -> int:
    """基线就绪审计：判断 recipe 能不能当 research / 业务训练的参照物。"""
    from .baseline import (
        STATE_ORDER,
        audit_one,
        load_contract,
        render_report,
        render_text,
        summarize,
    )
    from .plan import build_plan, render_plan

    root = _root()
    contract = load_contract(root)
    ctx = _ctx(root, 6, only=args.ids or None)
    audits = [audit_one(r, contract) for r in ctx.recipes]
    dist = summarize(audits)

    if args.dry_run:
        for r in ctx.recipes:
            print(render_plan(build_plan(r, mode="dry-run"), compact=not args.json))
        if not args.check and args.min_state is None and not args.report:
            return 0

    if args.report is not None:
        out = Path(args.report)
        if not out.is_absolute():
            out = root / out
        sample = ""
        if ctx.recipes:
            # 与报告里「先拿它做端到端样板」的建议保持一致；不要随手拿
            # ctx.recipes[0]（那会是 adapter 尚未实现的 Wordle/OpenEnv 篇）。
            preferred = next(
                (r for r in ctx.recipes if r.id == "llm/grpo/qwen2-0-5b-numinamath"),
                ctx.recipes[0],
            )
            sample = render_plan(build_plan(preferred, mode="dry-run"), compact=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        from .capabilities import render as render_capabilities
        from .feasibility import render as render_feasibility
        all_recipes = load_recipes(root)
        out.write_text(
            render_report(audits, contract, dry_run_note=sample)
            + "\n" + render_capabilities(root, all_recipes)
            + "\n" + render_feasibility(root, all_recipes), encoding="utf-8"
        )
        print(f"已生成 {_display_path(out, root)}（{len(audits)} 篇）")
        if not args.check and args.min_state is None and not args.json and not args.dry_run:
            return 0

    if args.json:
        print(json.dumps({
            "assessed_at": contract.get("assessed_at"),
            "distribution": dist,
            "audits": [_baseline_dict(a) for a in audits],
        }, ensure_ascii=False, indent=2))
    elif not args.dry_run and args.report is None:
        print(render_text(audits))

    if not args.check and args.min_state is None:
        return 0

    failures: list[str] = []
    for a in audits:
        if not a.state_ok:
            failures.append(
                f"{a.recipe_id}: state 声明 {a.claimed or '(空)'}，重算为 {a.computed}"
            )
    if args.min_state:
        for a in audits:
            if STATE_ORDER.get(a.computed, 0) < STATE_ORDER[args.min_state]:
                failures.append(f"{a.recipe_id}: 状态 {a.computed} 低于要求 {args.min_state}")
        # 代码/证据齐备也不等于实验目标成立：零训练基线已达标、基座预训练污染
        # 这两类必须在此显式失败关闭，且不接受追加 evidence 绕过。
        from .feasibility import promotion_problems
        failures.extend(f"[可行性] {p}" for p in promotion_problems(root, load_recipes(root)))

    if not args.json:
        print(f"\nrecipe-lab baseline --check · {len(audits)} 篇")
        print(
            f"  状态分布 B0 {dist['B0']} / B1 {dist['B1']} / "
            f"B2 {dist['B2']} / B3 {dist['B3']}"
        )
        print(
            f"  可作为 research baseline（B2+）：{dist['B2'] + dist['B3']}/{len(audits)} "
            "—— 这是审计结论，不是校验失败"
        )
        if failures:
            print("\n审计完整性未通过：")
            for line in failures:
                print(f"  [FAIL] {line}")
        else:
            print("  [ok] 无虚报状态，执行契约自洽")

    return 1 if failures else 0


def cmd_capabilities(args: argparse.Namespace) -> int:
    from .capabilities import audit, render
    root = _root()
    recipes = load_recipes(root)
    result = audit(root, recipes)
    if args.json:
        print(json.dumps({"ok": result.ok, "errors": result.errors, "counts": result.counts,
                          "recipes": result.rows, "missing_families": result.missing_families,
                          "gpu_verified_recipes": result.gpu_verified_recipes}, ensure_ascii=False, indent=2))
    elif args.check:
        print(f"能力矩阵：{result.registered_recipes}篇，{len(result.errors)}个登记错误；"
              f"代码部分实现{result.implemented_recipes}篇，GPU train+eval verified {result.gpu_verified_recipes}篇")
        for error in result.errors:
            print("[FAIL] " + error)
    else:
        print(render(root, recipes))
    return 0 if result.ok else 1


def cmd_feasibility(args: argparse.Namespace) -> int:
    """逐篇判定「原理上能否作 baseline」；与代码实装/GPU 证据分开计数。"""
    from .feasibility import audit, load, promotion_problems, render

    root = _root()
    # 判定必须精确覆盖 14 篇：不支持按 id 子集降级，否则会漏掉原理性阻塞。
    recipes = load_recipes(root)
    checked = audit(root, recipes)
    if args.report is not None:
        out = Path(args.report)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(root, recipes), encoding="utf-8")
        print(f"已生成 {_display_path(out, root)}（{len(recipes)} 篇）")
        if not args.check and not args.json:
            return 0
    if args.json:
        print(json.dumps({
            "assessed_at": load(root).get("assessed_at"),
            "ok": checked.ok,
            "errors": checked.errors,
            "counts": checked.counts,
            "promotion_blocking": checked.promotion_blocking,
            "needs_override": checked.needs_override,
            "rows": checked.rows,
        }, ensure_ascii=False, indent=2))
    elif not args.report:
        counts = checked.counts
        print(f"baseline 可行性：{len(recipes)} 篇，eligible {counts.get('eligible', 0)}，"
              f"pending_freeze {counts.get('pending_freeze', 0)}，"
              f"not_implemented {counts.get('not_implemented', 0)}，"
              f"blocked_zero_train {counts.get('blocked_zero_train', 0)}，"
              f"contaminated_pretraining {counts.get('contaminated_pretraining', 0)}")
        for rid in sorted(checked.rows):
            row = checked.rows[rid]
            print(f"  {rid}: {row['verdict']}")
        for rid in sorted(checked.needs_override):
            print(f"  [原理性阻塞] {rid}: 需改写实验目标 / 换数据后重新判定，不接受加证据绕过")
    if not args.check and args.min_state is None:
        return 0
    failures = list(checked.errors)
    if args.min_state:
        failures.extend(promotion_problems(root, recipes))
    if not args.json:
        print(f"\nrecipe-lab feasibility --check · {len(recipes)} 篇")
        print(f"  登记完整性：{'通过' if checked.ok else '未通过'}；"
              f"直接可作 baseline（eligible）：{checked.counts.get('eligible', 0)}/{len(recipes)}；"
              f"晋级阻塞 {len(checked.promotion_blocking)} 篇"
              f"（其中原理性阻塞 {len(checked.needs_override)} 篇）")
        if not args.min_state:
            print("  --check 只校验登记与一手事实一致性；晋级阻塞请用 --min-state B2/B3 显式判定")
        if failures:
            print("可行性 / 晋级门禁未通过：")
            for line in failures:
                print("  [FAIL] " + line)
        else:
            print("  [ok] 登记完整，一手事实交叉校验通过")
    return 1 if failures else 0


def cmd_methodology(args: argparse.Namespace) -> int:
    """方法论依据层审计：设计决策是否绑定到一手核查过的 practice 论文。

    `--check` 只校验登记完整性（引用闭合、一手字段、边界声明、与可行性层一致），
    不是「所有论文都必须已核实」——unverified 篇是审计结论而非失败。要设提升
    目标用 `--min-verified-papers N`。
    """
    from .methodology import audit, load

    root = _root()
    recipes = load_recipes(root)
    checked = audit(root, recipes)
    if args.report is not None:
        out = Path(args.report)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        from .methodology import render as render_methodology
        out.write_text(render_methodology(root, recipes), encoding="utf-8")
        print(f"已生成 {_display_path(out, root)}（{len(recipes)} 篇）")
        if not args.check and not args.json and args.min_verified_papers is None:
            return 0
    if args.json:
        print(json.dumps({
            "assessed_at": load(root).get("assessed_at"),
            "ok": checked.ok,
            "errors": checked.errors,
            "counts": checked.counts,
            "recipes": checked.rows,
            "papers": checked.papers,
            "principles": checked.principles,
        }, ensure_ascii=False, indent=2))
    elif args.report is None:
        c = checked.counts
        print(f"方法论依据：{len(recipes)} 篇，论文 {c.get('papers', 0)}（一手核实 "
              f"{c.get('verified_papers', 0)} / 未核实 {c.get('unverified_papers', 0)}），"
              f"主张 {c.get('principles', 0)}，编号更正 {c.get('corrections', 0)}，"
              f"登记错误 {len(checked.errors)}")
        for rid in sorted(checked.rows):
            row = checked.rows[rid]
            print(f"  {rid}: {len(row.get('primary_papers', []))} 篇主论文 / "
                  f"{len(row.get('principles', []))} 条主张")
    failures = list(checked.errors)
    if args.min_verified_papers is not None:
        actual = checked.counts.get("verified_papers", 0)
        if actual < args.min_verified_papers:
            failures.append(
                f"一手核实论文 {actual} 篇低于要求 {args.min_verified_papers}")
    if not args.json:
        print(f"\nrecipe-lab methodology --check · {len(recipes)} 篇")
        print(f"  登记完整性：{'通过' if checked.ok else '未通过'}；"
              f"一手核实论文 {checked.counts.get('verified_papers', 0)}/"
              f"{checked.counts.get('papers', 0)}；编号更正 {checked.counts.get('corrections', 0)} 条")
        if not args.min_verified_papers:
            print("  --check 只校验登记闭合与一手字段；提升目标请用 --min-verified-papers N")
        if failures:
            print("方法论登记未通过：")
            for line in failures:
                print("  [FAIL] " + line)
        else:
            print("  [ok] 14 篇全覆盖，引用闭合，一手字段与边界声明完整")
    return 1 if failures else 0


def cmd_post_training(args: argparse.Namespace) -> int:
    """MLLM 后训练方向层审计：五方向覆盖、来源边界与能力/可行性交叉。"""
    from .post_training import audit, load, promotion_problems, render

    root = _root()
    recipes = load_recipes(root)
    checked = audit(root, recipes)
    if args.report is not None:
        out = Path(args.report)
        if not out.is_absolute():
            out = root / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(root, recipes), encoding="utf-8")
        print(f"已生成 {_display_path(out, root)}（{len(recipes)} 篇）")
        if not args.check and not args.json and args.min_fivedim_directions is None:
            return 0
    if args.json:
        print(json.dumps({
            "assessed_at": load(root).get("assessed_at"),
            "ok": checked.ok,
            "errors": checked.errors,
            "counts": checked.counts,
            "sources": checked.sources,
            "directions": checked.directions,
            "coverage": checked.coverage,
            "recipes": checked.rows,
        }, ensure_ascii=False, indent=2))
    elif args.report is None:
        c = checked.counts
        print(f"MLLM 后训练方向：{c.get('directions', 0)} 个方向，映射 "
              f"{c.get('recipes', 0)}/{c.get('total', 0)} 篇，来源 {c.get('sources', 0)} 条，"
              f"登记错误 {len(checked.errors)}")
        for did in sorted(checked.coverage):
            row = checked.coverage[did]
            print(f"  {did}: 主方向 {row['primary_count']} 篇 / 五维实现 "
                  f"{row['implemented_count']} 篇 / {'缺口' if row['gap'] else '有支撑'}")
    failures = list(checked.errors)
    if args.min_fivedim_directions is not None:
        failures.extend(promotion_problems(checked, args.min_fivedim_directions))
    if not args.json:
        print(f"\nrecipe-lab post-training --check · {len(recipes)} 篇")
        print(f"  登记完整性：{'通过' if checked.ok else '未通过'}；"
              f"有五维实现支撑的方向 {checked.counts.get('implemented_directions', 0)}/"
              f"{checked.counts.get('directions', 0)}；缺口 {checked.counts.get('gap_directions', 0)}")
        if not args.min_fivedim_directions:
            print("  --check 只校验登记闭合与来源边界；提升目标请用 --min-fivedim-directions N")
        if failures:
            print("方向层登记未通过：")
            for line in failures:
                print("  [FAIL] " + line)
        else:
            print("  [ok] 14 篇精确映射，五方向闭合，来源边界与能力/可行性交叉通过")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recipe-lab", description="LLM/VLM recipe 登记与验证")
    p.add_argument("--version", action="version", version=f"recipe-lab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="环境自检").set_defaults(func=cmd_doctor)

    pl = sub.add_parser("list", help="列出 recipe")
    pl.add_argument("--family")
    pl.add_argument("--status")
    pl.add_argument("--modality", choices=["llm", "vlm"])
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", help="打印某篇 recipe.toml 原文")
    ps.add_argument("id")
    ps.set_defaults(func=cmd_show)

    pv = sub.add_parser("validate", help="分层验证")
    pv.add_argument("--level", choices=list(LEVEL_NAMES), default="l3")
    pv.add_argument("--strict", action="store_true", help="warning 也算失败")
    pv.add_argument("--json", action="store_true")
    pv.add_argument("--quiet", action="store_true", help="只输出每篇汇总")
    pv.add_argument("--verbose", action="store_true", help="连 info 一起输出")
    pv.add_argument("ids", nargs="*", help="只验证指定 id 或 id 前缀")
    pv.set_defaults(func=cmd_validate)

    pm = sub.add_parser("matrix", help="生成 docs/MATRIX.md")
    pm.add_argument("--out", default="docs/MATRIX.md")
    pm.add_argument("--level", choices=list(LEVEL_NAMES), default="l3")
    pm.set_defaults(func=cmd_matrix)

    pr = sub.add_parser("repro", help="可复现性审计（L0–L4 八维度）")
    pr.add_argument(
        "--report", nargs="?", const="docs/REPRODUCIBILITY.md", default=None,
        metavar="PATH", help="生成 Markdown 审计报告，默认 docs/REPRODUCIBILITY.md",
    )
    pr.add_argument(
        "--check", action="store_true",
        help="校验审计完整性（等级无虚报、证据可追溯），不要求所有 recipe 都可复现",
    )
    pr.add_argument(
        "--min-level", choices=["L0", "L1", "L2", "L3", "L4"], default=None,
        metavar="LEVEL", help="额外要求所有 recipe 至少达到该等级（用于设定提升目标）",
    )
    pr.add_argument("--json", action="store_true", help="输出结构化审计结果")
    pr.add_argument("ids", nargs="*", help="只审计指定 id 或 id 前缀")
    pr.set_defaults(func=cmd_repro)

    pb = sub.add_parser("baseline", help="基线就绪审计（能否作为 research / 业务 baseline）")
    pb.add_argument(
        "--report", nargs="?", const="docs/BASELINE-READINESS.md", default=None,
        metavar="PATH", help="生成 Markdown 就绪报告，默认 docs/BASELINE-READINESS.md",
    )
    pb.add_argument(
        "--check", action="store_true", help="校验状态无虚报；不要求全部达到 B2",
    )
    pb.add_argument(
        "--min-state", choices=["B0", "B1", "B2", "B3"], default=None,
        metavar="STATE", help="额外要求所有 recipe 至少达到该状态",
    )
    pb.add_argument(
        "--dry-run", action="store_true",
        help="打印各篇的训练计划（不装训练框架、不需要 GPU）",
    )
    pb.add_argument("--json", action="store_true", help="输出结构化就绪结果")
    pb.add_argument("ids", nargs="*", help="只审计指定 id 或 id 前缀")
    pb.set_defaults(func=cmd_baseline)
    pc = sub.add_parser("capabilities", help="任务实装、方法范围与实施依赖的全量能力审计")
    pc.add_argument("--check", action="store_true")
    pc.add_argument("--json", action="store_true")
    pc.set_defaults(func=cmd_capabilities)

    pf = sub.add_parser("feasibility", help="逐篇判定原理上能否作 research / business baseline")
    pf.add_argument("--check", action="store_true", help="校验登记完整性与一手事实一致性")
    pf.add_argument("--min-state", choices=["B2", "B3"], default=None,
                    help="额外要求所有篇目均可晋级到该状态（当前必须失败关闭）")
    pf.add_argument("--report", nargs="?", const="docs/FEASIBILITY.md", default=None,
                    metavar="PATH", help="生成独立的 Markdown 判定报告")
    pf.add_argument("--json", action="store_true")
    pf.set_defaults(func=cmd_feasibility)

    pm2 = sub.add_parser("methodology", help="方法论依据审计（practice papers 一手核查）")
    pm2.add_argument("--check", action="store_true", help="校验登记闭合与一手字段完整性")
    pm2.add_argument("--min-verified-papers", type=int, default=None, metavar="N",
                     help="额外要求一手核实论文不少于 N 篇（提升目标）")
    pm2.add_argument("--report", nargs="?", const="docs/METHODOLOGY.md", default=None,
                     metavar="PATH", help="生成 Markdown 方法论报告")
    pm2.add_argument("--json", action="store_true")
    pm2.set_defaults(func=cmd_methodology)

    pt = sub.add_parser("post-training", help="MLLM 后训练五方向覆盖与来源边界审计")
    pt.add_argument("--check", action="store_true", help="校验登记闭合与来源边界")
    pt.add_argument("--min-fivedim-directions", type=int, default=None, metavar="N",
                    help="额外要求至少 N 个方向有五维实现支撑（提升目标）")
    pt.add_argument("--report", nargs="?", const="docs/MLLM-POST-TRAINING.md", default=None,
                    metavar="PATH", help="生成 Markdown 方向覆盖报告")
    pt.add_argument("--json", action="store_true")
    pt.set_defaults(func=cmd_post_training)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LoadError as exc:
        print(f"recipe-lab: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
