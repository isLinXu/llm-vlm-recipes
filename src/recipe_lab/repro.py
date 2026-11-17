"""可复现性审计：等级计算、逐篇判定与报告生成。

判定完全由 registry/reproducibility.toml 驱动。等级是**算出来的**，
不是从 recipe.toml 里读出来的——R503 会用同一函数重算，防止虚报。
"""

from __future__ import annotations

import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .model import Recipe

STATE_ORDER = {"missing": 0, "declared": 1, "pinned": 2, "verified": 3}
DIMS = ["code", "env", "model", "data", "params", "seed", "eval", "provenance"]
LEVELS = ["L4", "L3", "L2", "L1", "L0"]

STATE_LABEL = {
    "missing": "缺失",
    "declared": "已声明",
    "pinned": "已锁定",
    "verified": "已验证",
}
LEVEL_LABEL = {
    "L0": "仅登记",
    "L1": "规格完整",
    "L2": "完全锁定",
    "L3": "可执行",
    "L4": "已复现验证",
}


def state_rank(state: str) -> int:
    return STATE_ORDER.get(str(state), -1)


def load_contract(root: Path | None = None) -> dict[str, Any]:
    from .loader import repo_root

    root = root or repo_root()
    with (root / "registry" / "reproducibility.toml").open("rb") as fh:
        return tomllib.load(fh)


def load_evidence(root: Path | None = None) -> dict[str, Any]:
    from .loader import repo_root

    root = root or repo_root()
    with (root / "registry" / "evidence.toml").open("rb") as fh:
        return tomllib.load(fh)


def compute_level(dims: dict[str, str], rank_fn=state_rank, contract: dict | None = None) -> str:
    """返回八个维度允许达到的最高等级。"""
    contract = contract or load_contract()
    for level in LEVELS:
        requires = contract["levels"][level]["requires"]
        if all(rank_fn(dims.get(k, "missing")) >= rank_fn(v) for k, v in requires.items()):
            return level
    return "L0"


@dataclass
class Audit:
    recipe_id: str
    claimed: str
    computed: str
    dims: dict[str, str]
    gaps: dict[str, str]
    evidence_ref: str
    level_ok: bool

    @property
    def blocking_dims(self) -> list[str]:
        return [d for d in DIMS if state_rank(self.dims[d]) < STATE_ORDER["pinned"]]


def audit_one(r: Recipe, contract: dict | None = None) -> Audit:
    contract = contract or load_contract()
    dims: dict[str, str] = {}
    gaps: dict[str, str] = {}
    for d in DIMS:
        state = r.get(f"repro.{d}.state")
        dims[d] = str(state) if state else "missing"
        gaps[d] = str(r.get(f"repro.{d}.gap", "") or "")
    claimed = str(r.get("repro.level", ""))
    computed = compute_level(dims, contract=contract)
    return Audit(
        recipe_id=r.id,
        claimed=claimed,
        computed=computed,
        dims=dims,
        gaps=gaps,
        evidence_ref=str(r.get("repro.evidence_ref", "") or ""),
        level_ok=claimed == computed,
    )


def evidence_ref_problems(
    recipes: list[Recipe], contract: dict, evidence: dict
) -> dict[str, tuple[str, str]]:
    """跨文件校验 evidence_ref 可追溯性。R507 与 `repro --check` 共用此函数。

    返回 {recipe_id: (severity, message)}；空 dict 表示全部可追溯。
    """
    pages = set(evidence["method"].get("webfetch_pages", []))
    notebooks = {Path(p).name for p in evidence["method"].get("local_notebooks", [])}
    literals = set(contract.get("allowed_evidence_literals", []))
    prefixes = set(contract.get("allowed_evidence_prefixes", []))
    out: dict[str, tuple[str, str]] = {}
    for r in recipes:
        ref = str(r.get("repro.evidence_ref", "") or "")
        if not ref:
            continue  # R502 负责报缺字段
        ok = ref in literals
        if not ok:
            for prefix in prefixes:
                if not ref.startswith(prefix):
                    continue
                value = ref[len(prefix):]
                if prefix == "webfetch:" and value in pages:
                    ok = True
                if prefix == "local-notebook:" and value in notebooks:
                    ok = True
        if not ok:
            out[r.id] = (
                "error",
                f"repro.evidence_ref = {ref!r} 未在 evidence.toml 登记",
            )
        elif ref == "research-report-only":
            out[r.id] = ("warning", "只有二手调研报告，未回上游原文核对")
    return out


def summarize(audits: list[Audit]) -> dict[str, int]:
    counts = Counter(a.computed for a in audits)
    return {lv: counts.get(lv, 0) for lv in ["L0", "L1", "L2", "L3", "L4"]}


def gap_counts(audits: list[Audit]) -> dict[str, int]:
    """每个维度上有多少篇未达到 pinned。"""
    return {
        d: sum(1 for a in audits if state_rank(a.dims[d]) < STATE_ORDER["pinned"])
        for d in DIMS
    }


def render_text(audits: list[Audit]) -> str:
    dist = summarize(audits)
    lines = [
        f"可复现性审计 · {len(audits)} 篇 · 评估日期 2026-11-13",
        "",
        f"等级分布：L0 {dist['L0']} / L1 {dist['L1']} / L2 {dist['L2']} / L3 {dist['L3']} / L4 {dist['L4']}",
        "",
        f"{'recipe':<52} {'等级':<5} {'声明':<5} {'未锁定维度'}",
        "-" * 100,
    ]
    for a in sorted(audits, key=lambda x: x.recipe_id):
        flag = "" if a.level_ok else "  <-- 虚报"
        lines.append(
            f"{a.recipe_id:<52} {a.computed:<5} {a.claimed:<5} "
            f"{','.join(a.blocking_dims)}{flag}"
        )
    lines += ["", "维度缺口（未达到 pinned 的篇数）："]
    for d, n in sorted(gap_counts(audits).items(), key=lambda x: -x[1]):
        lines.append(f"  {d:<12} {n}/{len(audits)}")
    lines += [
        "",
        "结论：达到 L2 及以上才算「可复现」。"
        f"当前 {sum(dist[k] for k in ('L2','L3','L4'))} 篇。",
    ]
    return "\n".join(lines)


def render_report(audits: list[Audit], contract: dict, evidence: dict) -> str:
    dist = summarize(audits)
    n = len(audits)
    reproducible = sum(dist[k] for k in ("L2", "L3", "L4"))
    L: list[str] = []
    L += [
        "# 可复现性审计报告",
        "",
        f"> 评估日期：{evidence.get('assessed_at', '2026-11-13')} · "
        f"对象：{n} 篇 recipe · 生成器：`recipe-lab repro --report`",
        "",
        "## 结论（先说最重要的）",
        "",
        f"**确认结果：{n} 篇中，{reproducible} 篇达到「可复现」（L2 及以上）。**",
        "",
        f"- L0 仅登记：{dist['L0']} 篇",
        f"- L1 规格完整但未锁定：{dist['L1']} 篇",
        f"- L2 完全锁定：{dist['L2']} 篇",
        f"- L3 可执行：{dist['L3']} 篇",
        f"- L4 已复现验证：{dist['L4']} 篇",
        "",
        "这里把「可复现」定义为 **L2 完全锁定**：模型/数据 revision、依赖 lock 与 commit、"
        "训练种子、上游 commit 全部可重放。仅仅把超参写进 TOML（L1）不算可复现——"
        "上游用的是 `pip install -U -q git+https://github.com/huggingface/trl.git`，"
        "任何人重跑都会装到不同的 trl；`from_pretrained` 不锁 revision，"
        "Hub 权重更新会静默改变结果；训练 config 里没有 seed。",
        "",
        "因此上一轮交付的「0 error / 33 tests passed」只证明**配置自洽**，"
        "不证明**结果可复现**。这两件事必须分开。",
        "",
        "## 判定标准",
        "",
        "| 等级 | 含义 | 达标条件 |",
        "| --- | --- | --- |",
    ]
    for lv in ["L0", "L1", "L2", "L3", "L4"]:
        meta = contract["levels"][lv]
        req = "、".join(f"`{k}` ≥ `{v}`" for k, v in meta["requires"].items())
        L.append(f"| {lv} {LEVEL_LABEL[lv]} | {meta['description']} | {req} |")
    L += [
        "",
        "八个维度：`code` 执行入口、`env` 环境依赖、`model` 基座模型、`data` 数据集、"
        "`params` 训练超参、`seed` 确定性、`eval` 评估闭环、`provenance` 上游溯源。",
        "每个维度的状态取 `missing → declared → pinned → verified`，"
        "等级由八个维度的**最低达标面**自动计算，`R503` 会用同一函数重算，虚报即报错。",
        "",
        "## 逐篇审计",
        "",
        "| recipe | 等级 | code | env | model | data | params | seed | eval | prov | evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    short = {"missing": "—", "declared": "声明", "pinned": "锁定", "verified": "验证"}
    for a in sorted(audits, key=lambda x: x.recipe_id):
        cells = " | ".join(short.get(a.dims[d], "—") for d in DIMS)
        L.append(f"| `{a.recipe_id}` | **{a.computed}** | {cells} | `{a.evidence_ref}` |")
    L += ["", "## 阻塞项统计", "", "| 维度 | 未达 pinned 的篇数 | 主要缺口 |", "| --- | --- | --- |"]
    for d, count in sorted(gap_counts(audits).items(), key=lambda x: -x[1]):
        L.append(f"| `{d}` | {count}/{n} | {contract['dimensions'][d]['common_gap']} |")
    L += ["", "### 硬阻塞（来自 registry/reproducibility.toml）", "",
          "| ID | 维度 | 封顶等级 | 条件 | 一手证据 |", "| --- | --- | --- | --- | --- |"]
    for b in contract.get("blockers", []):
        L.append(
            f"| {b['id']} | `{b['dimension']}` | {b['caps_level']} | "
            f"{b['condition']} | {b['upstream_evidence']} |"
        )
    L += ["", "## 一手核查发现", "",
          f"方法：{evidence['method']['summary']}（覆盖 {evidence['method'].get('coverage', 'n/a')}）。",
          f"规则：{evidence['method']['rule']}。", "",
          "本轮把上一轮只有二手调研描述的 7 篇（SmolVLM SFT、VLM DPO、VLM MPO、"
          "grounding、xLAM、code LLM、RapidFire）全部回上游页面做了事实抽取，"
          "并据此修正了模型 / 数据集 id、补齐训练配置、纠正了两处「以为没有但实际上有」的判断："
          "code LLM 篇其实有 `SEED=0 + set_seed` 和 4000 条 held-out，"
          "RapidFire 篇其实有 `ROUGE-L compute_metrics` 和 `run_fit(seed=42)`。"
          "这两篇仍不可复现，但原因从「完全没有」变成了「有但不完整 / 评估集泄漏」。", ""]
    for f in evidence.get("findings", []):
        L += [
            f"### {f['id']} · {f['summary']}（{f['severity']}）",
            "",
            f"- 证据：{f['evidence']}",
            f"- 后果：{f['consequence']}",
            "",
        ]
    L += [
        "## 已纠正的事实错误",
        "",
        "上一轮部分配置值来自二手调研总结，未逐篇回上游核对。本次核查后已修正：",
        "",
        "| recipe | 字段 | 原值 | 核实后 | 来源 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in evidence.get("corrections", []):
        L.append(
            f"| `{c['recipe']}` | `{c['field']}` | {c['was']} | {c['now']} | `{c['source']}` |"
        )
    L += [
        "",
        "其中最严重的是 `agent/grpo/wordle-openenv-hf-jobs` 的 `model.base`："
        "上一轮写入的 `Qwen2.5-0.5B-Instruct` **没有任何一手依据，属于编造**；"
        "上游实际按显存自动选择 `Qwen/Qwen3-0.6B` 或 `Qwen/Qwen3-1.7B`。",
        "",
        "## 推断值处置",
        "",
        "| recipe | 字段 | 值 | 为什么不可信 | 处置 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for u in evidence.get("unverified", []):
        L.append(
            f"| `{u['recipe']}` | `{u['field']}` | `{u['value']}` | "
            f"{u['reason']} | {u['action']} |"
        )
    L += [
        "",
        "## 补齐路径",
        "",
        "按依赖顺序，每一步都能用 `recipe-lab repro` 复查等级是否真的推进：",
        "",
        "| 阶段 | 动作 | 解锁维度 | 目标等级 |",
        "| --- | --- | --- | --- |",
        "| P0 溯源 | 为每篇锁定 `source_sha`，记录抓取时的上游 commit | provenance | — |",
        "| P1 锁定输入 | 记录 `model.revision` / `dataset.revision`，写权重与数据校验和 | model, data | — |",
        "| P2 重建环境 | 把 `-U / git+` 安装改成 `requirements.lock`（含 git commit），声明 Python/CUDA | env | — |",
        "| P3 确定性 | 在 trainer config 显式设 `seed` / `data_seed` / deterministic 策略 | seed | **L2** |",
        "| P4 可执行 | 为每篇写 `train.py` + `eval.py`，给出确定命令与 evidence 产物 | code, eval | L3 |",
        "| P5 实跑 | 在声明硬件上重跑，记录指标与容差，人工确认后方可标 L4 | code, eval | L4 |",
        "",
        "建议先拿 `llm/grpo/qwen2-0-5b-numinamath` 做端到端样板：它是 0.5B 模型、"
        "参数与测试版本已按上游原文补齐，改动面最小。样板跑通后再把同一套锁文件、"
        "入口、评估脚本模板推广到其余 13 篇。",
        "",
        "## 本机无法确认的部分",
        "",
        "- **没有 GPU**，无法执行任何真实训练，因此不存在 L4 的证据产物。",
        f"- 一手材料覆盖 {evidence['method'].get('coverage', 'n/a')}，但只有 "
        f"{len(evidence['method']['local_notebooks'])} 篇拿到了本地 `.ipynb` 原文；"
        f"其余 {len(evidence['method']['webfetch_pages'])} 篇只有网页事实抽取，"
        "未逐字节校验代码单元格，也没有抓取时的上游 commit。",
        f"- {evidence['method']['not_verified_pages']}。",
        "- `baseline.result` 与 `evaluation.metrics` 是声明的协议，不是实测数值；"
        "验证器只检查它们与 `status` 自洽，不重新跑实验。",
        "- `# Tested with trl==0.14.0` 这类注释只是作者当时的环境快照，"
        "本仓库把它登记为 `dependencies` 里的「测试版本」，但 `repro.env` 仍判为 "
        "`declared` / `missing`，因为没有 `requirements.lock`。",
        "- 即使 P0–P4 全部完成，仍需真实硬件重跑才能确认数值层面的可复现性。",
        "",
        "---",
        "",
        "判定标准：[registry/reproducibility.toml](../registry/reproducibility.toml) · "
        "证据登记：[registry/evidence.toml](../registry/evidence.toml) · "
        "规则：[docs/VERIFICATION.md](VERIFICATION.md) · "
        "纠错脚本：[scripts/apply_repro_audit.py](../scripts/apply_repro_audit.py)",
        "",
    ]
    return "\n".join(L)
