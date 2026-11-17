"""方法论依据层审计：逐篇设计决策是否绑定到一手核查过的 practice 论文。

与另外三层审计互不替代，也不解锁 GPU 晋级：
  * capabilities.py：代码 + offline test 是否实装（能不能跑起来）；
  * baseline.py / baseline_evidence.py：有没有真实跑过、证据够不够；
  * feasibility.py：实验目标是否成立、数据是否污染；
  * 本模块：设计决策依据哪篇论文的一手结论、边界在哪、论文没证明什么。

关键约束（audit 强制）：
  - paper 一手字段完整，verified=true 必须有 arxiv_id/title/year/core_claim/boundary；
  - verified=false 的论文只能进 supplement_papers，不能作 principle 主依据；
  - 14 篇精确覆盖，每篇有立场、主张、主论文；
  - principle 引用的论文必须是该篇 primary_papers 的子集，不允许悬空引用；
  - 每篇 recipe.toml 必须有 [methodology] 引用式回填，registry 指针正确且
    principles / primary_papers / supplement_papers / stance 与登记逐字段相等；
  - 禁止把论文分数当本仓 expected/tolerance（forbidden_claim_phrases）；
  - corrections 的 was != now，且至少覆盖核查中发现的误用编号；
  - 与 feasibility 交叉：blocked_zero_train 必须有 chartqa_zero_train_gate，
    contaminated_pretraining 必须有 chartqa_contamination，防止论文背书绕过原理性阻塞。

零第三方依赖；只读 registry，不写运行证据、不认证 GPU 执行。
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .model import Recipe

REGISTRY = "registry/methodology.toml"
FEASIBILITY = "registry/feasibility.toml"
RECIPE_FIELDS = ("principles", "primary_papers", "supplement_papers", "stance")
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
YEAR = re.compile(r"^\d{4}$")

# 原理性阻塞篇必须持有的方法论主张（否则论文层可能「洗白」阻塞判定）。
BLOCKED_REQUIRED = {
    "blocked_zero_train": "chartqa_zero_train_gate",
    "contaminated_pretraining": "chartqa_contamination",
}


class MethodologyError(ValueError):
    pass


@dataclass
class MethodologyAudit:
    ok: bool
    errors: list[str] = field(default_factory=list)
    papers: dict[str, dict] = field(default_factory=dict)
    principles: dict[str, dict] = field(default_factory=dict)
    rows: dict[str, dict] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    recipes: list[str] = field(default_factory=list)


def load(root: Path) -> dict[str, Any]:
    path = Path(root) / REGISTRY
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MethodologyError(f"无法加载方法论登记 {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise MethodologyError("methodology.toml 必须是 schema=1 的登记")
    return data


def _feasibility(root: Path) -> dict[str, dict]:
    try:
        with (Path(root) / FEASIBILITY).open("rb") as handle:
            return tomllib.load(handle).get("recipes", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _all_strings(value: Any) -> bool:
    return isinstance(value, list) and all(_nonempty_str(v) for v in value)


def audit(root: Path, recipes: Iterable[Recipe]) -> MethodologyAudit:
    """只读审计：结构完整性 + 一手字段 + 交叉引用 + 与可行性层一致性。"""
    recipes = list(recipes)
    ids = [r.id for r in recipes]
    by_id = {r.id: r for r in recipes}
    result = MethodologyAudit(ok=True, recipes=ids)
    try:
        data = load(root)
    except MethodologyError as exc:
        result.ok = False
        result.errors.append(str(exc))
        return result

    integrity = data.get("integrity", {})
    forbidden = _list(integrity.get("forbidden_claim_phrases"))
    supports = set(_list(integrity.get("supports")))
    scopes = set(_list(integrity.get("scopes")))
    if not forbidden or not all(_nonempty_str(v) for v in forbidden):
        result.errors.append("integrity.forbidden_claim_phrases 必须是非空字符串列表")
    if not supports or not scopes:
        result.errors.append("integrity.supports / scopes 不得为空")

    # ---- papers ----
    papers = data.get("papers", {})
    if not isinstance(papers, dict) or not papers:
        result.errors.append("papers 不得为空")
    for pid, paper in papers.items():
        if not isinstance(paper, dict):
            result.errors.append(f"paper {pid} 必须是表")
            continue
        result.papers[pid] = dict(paper)
        verified = paper.get("verified") is True
        required = ("title", "year", "core_claim", "boundary") if verified else ()
        for key in required:
            if not _nonempty_str(paper.get(key)):
                result.errors.append(f"paper {pid}: verified=true 要求 {key} 非空")
        if verified:
            aid = paper.get("arxiv_id", "")
            if not isinstance(aid, str) or not ARXIV_ID.match(aid):
                result.errors.append(f"paper {pid}: verified=true 需要合法 arxiv_id（NNNN.NNNNN），实际 {aid!r}")
            if not isinstance(paper.get("year", ""), str) or not YEAR.match(str(paper.get("year", ""))):
                result.errors.append(f"paper {pid}: verified=true 需要四位年份")
            for key in ("verified_at", "url"):
                if not _nonempty_str(paper.get(key)):
                    result.errors.append(f"paper {pid}: verified=true 要求 {key} 非空")
        else:
            # 未核实的论文必须说明为什么不能作一手依据，且不得携带伪造的一手字段。
            if not _nonempty_str(paper.get("boundary")):
                result.errors.append(f"paper {pid}: verified=false 仍须 boundary 说明为何未核实")
            if _nonempty_str(paper.get("core_claim")):
                result.errors.append(f"paper {pid}: verified=false 不得填写 core_claim（未一手核实）")

    # 禁止宣称模式：扫描所有会对外呈现的文本字段。
    def _scan(text: str, where: str) -> None:
        for phrase in forbidden:
            if phrase and phrase in text:
                result.errors.append(f"{where}: 出现禁止宣称 {phrase!r}（论文依据不等于本仓实测）")

    # ---- principles ----
    principles = data.get("principles", {})
    if not isinstance(principles, dict) or not principles:
        result.errors.append("principles 不得为空")
    for claim_id, claim in principles.items():
        if not isinstance(claim, dict):
            result.errors.append(f"principle {claim_id} 必须是表")
            continue
        result.principles[claim_id] = dict(claim)
        for key in ("paper", "support", "scope", "claim", "boundary"):
            if not _nonempty_str(claim.get(key)):
                result.errors.append(f"principle {claim_id}: {key} 必须是非空字符串")
        if claim.get("support") not in supports:
            result.errors.append(f"principle {claim_id}: 非法 support {claim.get('support')!r}")
        if claim.get("scope") not in scopes:
            result.errors.append(f"principle {claim_id}: 非法 scope {claim.get('scope')!r}")
        paper_ref = claim.get("paper")
        if paper_ref not in papers:
            result.errors.append(f"principle {claim_id}: 引用了不存在的 paper {paper_ref!r}")
        elif papers[paper_ref].get("verified") is not True and claim.get("support") == "paper_supports":
            result.errors.append(
                f"principle {claim_id}: 未核实论文 {paper_ref} 不得作为 paper_supports 主依据")
        for key in ("claim", "boundary"):
            if _nonempty_str(claim.get(key)):
                _scan(claim[key], f"principle {claim_id}.{key}")

    # ---- corrections ----
    corrections = data.get("corrections", [])
    if not isinstance(corrections, list) or not corrections:
        result.errors.append("corrections 不得为空（一手核查发现的编号误用必须登记）")
    else:
        for i, item in enumerate(corrections):
            if not isinstance(item, dict):
                result.errors.append(f"corrections[{i}] 必须是表")
                continue
            for key in ("was", "topic", "reason", "checked_at"):
                if not _nonempty_str(item.get(key)):
                    result.errors.append(f"corrections[{i}]: {key} 必须是非空字符串")
            if _nonempty_str(item.get("was")) and item.get("was") == item.get("now"):
                result.errors.append(f"corrections[{i}]: was 与 now 不得相同")
            now = item.get("now", "")
            if now and not ARXIV_ID.match(str(now)):
                result.errors.append(f"corrections[{i}]: now 若非空须为合法 arxiv_id，实际 {now!r}")
            if _nonempty_str(item.get("reason")):
                _scan(item["reason"], f"corrections[{i}].reason")

    # ---- recipes ----
    declared = data.get("recipes", {})
    expected = data.get("expected_recipe_count")
    if type(expected) is not int or expected != len(recipes):
        result.errors.append(f"expected_recipe_count 必须等于实际 recipe 数 {len(recipes)}")
    missing = sorted(set(ids) - set(declared))
    extra = sorted(set(declared) - set(ids))
    if missing or extra:
        result.errors.append(f"方法论登记未精确覆盖 14 篇：missing={missing} extra={extra}")

    feas = _feasibility(root)
    covered_principles: set[str] = set()
    for rid in sorted(set(ids) & set(declared)):
        row = declared[rid]
        if not isinstance(row, dict):
            result.errors.append(f"{rid}: 登记项必须是表")
            continue
        result.rows[rid] = dict(row)

        # 配方本体是引用式视图，不是第二个事实源：任何一侧漂移都失败关闭。
        recipe = by_id.get(rid)
        if recipe is None:
            result.errors.append(f"{rid}: 找不到对应 recipe.toml，无法核对方法论回填")
        else:
            local = recipe.get("methodology")
            if not isinstance(local, dict):
                result.errors.append(f"{rid}: recipe.toml 缺少 [methodology] 段")
            else:
                if local.get("registry") != REGISTRY:
                    result.errors.append(
                        f"{rid}: recipe.toml [methodology].registry 必须指向 {REGISTRY}"
                        f"，实际 {local.get('registry')!r}")
                for key in RECIPE_FIELDS:
                    if key not in local:
                        result.errors.append(f"{rid}: recipe.toml 缺少 [methodology].{key}")
                    elif local[key] != row.get(key):
                        result.errors.append(
                            f"{rid}: recipe.toml [methodology].{key} 与 {REGISTRY} 不一致")

        claims = _list(row.get("principles"))
        primary = _list(row.get("primary_papers"))
        supplement = _list(row.get("supplement_papers"))
        if not _all_strings(claims) or not _all_strings(primary):
            result.errors.append(f"{rid}: principles / primary_papers 必须是非空字符串列表")
            continue
        if not _nonempty_str(row.get("stance")):
            result.errors.append(f"{rid}: stance 必须是非空立场说明")
        else:
            _scan(row["stance"], f"{rid}.stance")

        # 主张依据的论文必须在该篇声明的依据并集内（primary = 该篇特有方法的一手
        # 论文；supplement = 通用方法论或未核实来源），防止删掉依据后主张悬空。
        declared_papers = set(primary) | set(supplement)
        for claim_id in claims:
            if claim_id not in principles:
                result.errors.append(f"{rid}: 引用了不存在的 principle {claim_id!r}")
                continue
            covered_principles.add(claim_id)
            ref = principles[claim_id].get("paper")
            if ref not in declared_papers:
                result.errors.append(
                    f"{rid}: principle {claim_id} 依据 {ref}，但未列入 primary/supplement papers")
            if ref not in supplement and ref not in papers:
                result.errors.append(f"{rid}: principle {claim_id} 的 paper {ref} 未登记")
        for pid in primary:
            if pid not in papers:
                result.errors.append(f"{rid}: primary_papers 含未登记论文 {pid!r}")
            elif papers[pid].get("verified") is not True:
                result.errors.append(f"{rid}: 未核实论文 {pid} 不得进 primary_papers")
        for pid in supplement:
            if pid not in papers:
                result.errors.append(f"{rid}: supplement_papers 含未登记论文 {pid!r}")
        overlap = set(primary) & set(supplement)
        if overlap:
            result.errors.append(f"{rid}: 论文同时出现在 primary 与 supplement：{sorted(overlap)}")

        # 与可行性层交叉：原理性阻塞篇必须显式持有对应主张，防止「论文背书洗白」。
        verdict = feas.get(rid, {}).get("verdict")
        required_claim = BLOCKED_REQUIRED.get(verdict)
        if required_claim and required_claim not in claims:
            result.errors.append(
                f"{rid}: 可行性 verdict={verdict} 要求方法论主张 {required_claim}，实际缺失")

    # 未被任何篇目引用的主张视为僵尸登记（可能掩盖覆盖缺口）。
    orphan = sorted(set(principles) - covered_principles)
    if orphan:
        result.errors.append(f"以下 principle 未被任何 recipe 引用：{orphan}")

    verified_count = sum(1 for p in papers.values() if p.get("verified") is True)
    result.counts = {
        "papers": len(papers),
        "verified_papers": verified_count,
        "unverified_papers": len(papers) - verified_count,
        "principles": len(principles),
        "corrections": len(corrections) if isinstance(corrections, list) else 0,
        "recipes": len(result.rows),
        "total": len(ids),
    }
    result.ok = not result.errors
    return result


def render(root: Path, recipes: Iterable[Recipe]) -> str:
    """生成可插入文档的 Markdown 方法论章节。"""
    recipes = list(recipes)
    checked = audit(root, recipes)
    try:
        data = load(root)
    except MethodologyError:
        data = {}
    c = checked.counts
    lines = [
        "# 方法论依据层（practice papers）", "",
        "> 由 `recipe-lab methodology` 生成；单一事实源是 `registry/methodology.toml`。"
        "本文件每次 `make report` 重建，请勿手工编辑。", "",
        "## 为什么需要这一层", "",
        "前三层审计各有分工，但都回答不了「为什么这么设计」：", "",
        "- `capabilities.toml`：代码 + offline test 是否实装（**能不能跑起来**）；",
        "- `baseline.toml` / `baseline_evidence.py`：证据与就绪度（**有没有真实跑过**）；",
        "- `feasibility.toml`：实验目标是否成立、数据是否污染（**值不值得当 baseline**）；",
        "- 本层：设计决策依据哪篇 practice 论文的一手结论、论文没证明什么（**为什么这么设计**）。", "",
        "四层互不替代，任何一层通过都不解锁 GPU 晋级或 B2/B3。论文的方法论背书**不等于**本仓实测；"
        "本仓的 expected/tolerance 必须来自真实 CUDA 实跑，绝不拿论文 benchmark 数字回填。", "",
        "每篇 `recipe.toml` 另有一个 `[methodology]` 段，只保存 registry 指针、主张 id、论文 id 与简短立场，"
        "方便读配方时直接看到依据。它不是第二份事实源：`methodology --check` 会把它与 "
        "`registry/methodology.toml` 的 `principles` / `primary_papers` / `supplement_papers` / `stance` "
        "逐字段比较，任一侧漂移即失败关闭。", "",
        "## 核查方法与诚信边界", "",
        "每条 `principles.*` 绑定一篇论文的 `paper_supports` / `we_adapted` / `paper_limited` 关系与一句 `boundary`，"
        "明确区分「论文一手结论直接支持」「我们据此修正上游（本仓工程判断）」「论文结论有明确边界或本仓不适用」。"
        "所有论文的 arXiv 编号、标题、年份与核心结论均回一手摘要页核对；未核到一手来源的（如 MPO 论文、RefCOCO "
        "原始出处、xLAM 论文）一律 `verified=false`，只能进 `supplement_papers`，不得支撑任何主张。", "",
        "## 当前结论", "",
        f"当前核查论文 **{c.get('verified_papers', 0)} 篇一手确认** / "
        f"{c.get('unverified_papers', 0)} 篇未核实（仅作补充）；"
        f"方法论主张 {c.get('principles', 0)} 条，编号更正 {c.get('corrections', 0)} 条，"
        f"覆盖 {c.get('recipes', 0)}/{c.get('total', 0)} 篇。", "",
        "所有论文的 arXiv 编号、标题与结论均回一手摘要页核对。核查中实际发现 "
        f"{c.get('corrections', 0)} 条凭记忆写错的编号（FIM、RLAIF-V、MPO、RefCOCO 等），"
        "已逐条登记 `was → now`，作为「一手核查不可省略」的物证；未核到一手来源的一律标 unverified，"
        "不得作主张依据，也不把论文 benchmark 数字写成本仓 expected/tolerance。", "",
        "| recipe | 方法论立场 | 一手主论文 | 主张 |", "| --- | --- | --- | --- |",
    ]
    papers = checked.papers
    for rid in sorted(checked.rows):
        row = checked.rows[rid]
        stance = str(row.get("stance", "")).replace("|", "\\|")
        prim = ", ".join(f"`{p}`" for p in _list(row.get("primary_papers")))
        claims = ", ".join(f"`{c}`" for c in _list(row.get("principles")))
        lines.append(f"| `{rid}` | {stance} | {prim} | {claims} |")
    lines.extend(["", "### 一手核查到的核心论文", "", "| id | arXiv | 标题 | 年份 |", "| --- | --- | --- | --- |"])
    for pid in sorted(papers):
        p = papers[pid]
        if p.get("verified") is not True:
            continue
        title = str(p.get("title", "")).replace("|", "\\|")
        lines.append(f"| `{pid}` | {p.get('arxiv_id')} | {title} | {p.get('year')} |")
    unverified = [pid for pid, p in papers.items() if p.get("verified") is not True]
    if unverified:
        lines.extend(["", "### 未核实来源（仅作补充，不支撑主张）", ""])
        for pid in sorted(unverified):
            lines.append(f"- `{pid}`：{papers[pid].get('boundary', '')}")
    lines.append("")
    if checked.errors:
        lines.extend(["### 登记完整性错误", ""] + [f"- [FAIL] {e}" for e in checked.errors] + [""])
    return "\n".join(lines)
