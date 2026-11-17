"""生成 docs/MATRIX.md —— 仓库的可读索引，由 CI 重建。"""

from __future__ import annotations

from collections import Counter

from .runner import Report


def _cell(value: object) -> str:
    text = str(value if value not in (None, "") else "-")
    return text.replace("|", "\\|").replace("\n", " ")


def render(report: Report) -> str:
    lines: list[str] = []
    lines.append("# Recipe 矩阵")
    lines.append("")
    lines.append(
        "> 由 `recipe-lab matrix` 自动生成，请勿手工编辑。"
        "单一事实源是各 `recipes/**/recipe.toml`。"
    )
    lines.append("")
    lines.append(
        f"当前登记 **{len(report.recipes)}** 篇 · 验证层级 L0–L{report.level} · "
        f"**{len(report.errors)}** error / **{len(report.warnings)}** warning / "
        f"**{len(report.infos)}** info。"
    )
    lines.append("")
    lines.append("状态机：`cataloged` → `ported` → `smoke-passed` → `benchmarked`。")
    lines.append("")

    lines.append("## 全量矩阵")
    lines.append("")
    header = [
        "id", "family", "trainer", "modality", "model", "dataset",
        "status", "coverage", "repro", "ready", "zero_shot", "trl", "errors",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in report.recipes:
        n_err = sum(1 for f in report.for_recipe(r.id) if f.severity == "error")
        row = r.as_row()
        lines.append(
            "| "
            + " | ".join(
                _cell(x)
                for x in (
                    f"`{row['id']}`",
                    row["family"],
                    row["trainer"],
                    row["modality"],
                    row["model"],
                    row["dataset"],
                    row["status"],
                    r.coverage,
                    r.get("repro.level", "-"),
                    r.get("baseline_contract.state", "-"),
                    row["baseline"],
                    row["trl"],
                    n_err,
                )
            )
            + " |"
        )
    lines.append("")

    fam_counts = Counter(r.family for r in report.recipes)
    tier_counts = Counter(r.status for r in report.recipes)
    lines.append("## 分布")
    lines.append("")
    lines.append("| 维度 | 分布 |")
    lines.append("| --- | --- |")
    lines.append(
        "| method.family | "
        + ", ".join(f"`{k}` {v}" for k, v in sorted(fam_counts.items()))
        + " |"
    )
    lines.append(
        "| status | "
        + ", ".join(f"`{k}` {v}" for k, v in sorted(tier_counts.items()))
        + " |"
    )
    lines.append("")

    repro_counts = Counter(str(r.get("repro.level", "-")) for r in report.recipes)
    reproducible = sum(repro_counts.get(k, 0) for k in ("L2", "L3", "L4"))
    lines.append("## 可复现性分布")
    lines.append("")
    lines.append("| 等级 | 篇数 |")
    lines.append("| --- | --- |")
    for lv in ("L0", "L1", "L2", "L3", "L4", "-"):
        if repro_counts.get(lv):
            lines.append(f"| `{lv}` | {repro_counts[lv]} |")
    lines.append("")
    lines.append(
        f"**达到 L2 及以上（可复现）：{reproducible}/{len(report.recipes)} 篇。** "
        "`repro` 列由 `registry/reproducibility.toml` 按八维度重算（`R503` 会拦截虚报），"
        "不是配置自洽的同义词。完整结论与补齐路径见 "
        "[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。"
    )
    lines.append("")

    ready_counts = Counter(str(r.get("baseline_contract.state", "-")) for r in report.recipes)
    research_ready = sum(ready_counts.get(k, 0) for k in ("B2", "B3"))
    lines.append("## 基线就绪分布")
    lines.append("")
    lines.append("| 状态 | 篇数 |")
    lines.append("| --- | --- |")
    for st in ("B0", "B1", "B2", "B3", "-"):
        if ready_counts.get(st):
            lines.append(f"| `{st}` | {ready_counts[st]} |")
    lines.append("")
    lines.append(
        f"**可作为 research baseline（B2 及以上）：{research_ready}/{len(report.recipes)} 篇。** "
        "`ready` 列是 `[baseline_contract].state`，由 `registry/baseline.toml` 的 "
        "四组检查项重算（`R603` 拦截虚报）。注意它与 `repro` 是两件事："
        "复刻不了上游原实验，不妨碍我们锁定自己的基线；B1 只代表契约齐备且可 dry-run，"
        "**不代表已经实跑**。详见 [BASELINE-READINESS.md](BASELINE-READINESS.md)。"
    )
    lines.append("")

    trl_versions: dict[str, list[str]] = {}
    for r in report.recipes:
        trl = str(r.get("dependencies.trl", ""))
        if trl:
            trl_versions.setdefault(trl, []).append(r.id)
    lines.append("## 版本漂移")
    lines.append("")
    if trl_versions:
        lines.append("| trl | 篇数 | recipe |")
        lines.append("| --- | --- | --- |")
        for ver, ids in sorted(trl_versions.items()):
            lines.append(f"| `{ver}` | {len(ids)} | {', '.join('`' + i + '`' for i in ids)} |")
    else:
        lines.append("未登记任何 trl 版本。")
    lines.append("")
    lines.append(
        "> 不同 trl 版本之间不要直接复制训练代码。上游 cookbook 覆盖 "
        "0.12.2 / 0.14.0 / 0.22.0.dev0 / 0.23.1，API 差异显著。"
        "（曾误记的 1.7.0 已核实为编造并删除，见 evidence.toml。）"
    )
    lines.append("")

    if report.errors:
        lines.append("## 阻塞项（error）")
        lines.append("")
        for f in report.errors:
            lines.append(f"- `{f.rule}` · `{f.recipe_id}` — {f.message}")
        lines.append("")
    if report.warnings:
        lines.append("## 待办清单（warning）")
        lines.append("")
        lines.append("warning 不代表失败，而是已登记、尚未消解的风险。")
        lines.append("")
        for f in report.warnings:
            lines.append(f"- `{f.rule}` · `{f.recipe_id}` — {f.message}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "字段规范见 [docs/DESIGN.md](DESIGN.md)，VLM 数据契约见 "
        "[docs/CONTRACT.md](CONTRACT.md)，验证分层见 [docs/VERIFICATION.md](VERIFICATION.md)，"
        "可复现性审计见 [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md)，"
        "基线就绪见 [docs/BASELINE-READINESS.md](BASELINE-READINESS.md)。"
    )
    lines.append("")
    return "\n".join(lines)
