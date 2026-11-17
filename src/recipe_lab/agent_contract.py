"""Agent 交接契约：把人类可读入口与 Agent 可复现命令绑定到同一事实源。

这不是第六层审计，也不新增晋级规则。它只做三件事：
  - 列出真实事实源与生成物，阻止手改 docs 下的生成报告；
  - 给出标准命令与必须失败关闭的提升门禁；
  - 校验生成物能由 CLI 从当前源码/注册表确定性重建。

零第三方依赖；只读仓库，不认证 GPU 执行。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .model import Recipe

REGISTRY = "registry/agent_contract.toml"
DIMS = ("data", "train", "eval", "comparison", "export")
ALLOWED_TOP = {
    "doctor", "validate", "ci", "report", "handoff", "make",
    "baseline", "repro", "feasibility", "methodology", "post-training",
}
GENERATORS = {"matrix", "repro", "baseline", "methodology", "post-training"}
AGENT_MARKERS = ("AGENT-HANDOFF.json", "make report", "recipe-lab handoff --check")


class AgentContractError(ValueError):
    pass


@dataclass
class ContractAudit:
    ok: bool
    errors: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    regeneration_checked: bool = False


def _read(root: Path, rel: str) -> dict[str, Any]:
    path = Path(root) / rel
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise AgentContractError(f"无法加载 {rel}: {exc}") from exc


def load(root: Path) -> dict[str, Any]:
    data = _read(root, REGISTRY)
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise AgentContractError("agent_contract.toml 必须是 schema=1 的登记")
    return data


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _argv(value: Any) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(x, str) for x in value) else []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _facts(root: Path, recipes: Iterable[Recipe]) -> dict[str, Any]:
    recipes = list(recipes)
    cap = _read(root, "registry/capabilities.toml").get("recipes", {})
    feas = _read(root, "registry/feasibility.toml").get("recipes", {})
    meth = _read(root, "registry/methodology.toml")
    post = _read(root, "registry/mllm_post_training.toml")

    implemented_recipes = sum(
        1 for rid in cap
        if all(cap[rid].get(dim, {}).get("status") == "implemented" for dim in DIMS))
    eligible = sum(1 for row in feas.values() if row.get("verdict") == "eligible")
    verified_papers = sum(
        1 for paper in meth.get("papers", {}).values() if paper.get("verified") is True)
    blocked = sorted(rid for rid, row in feas.items()
                     if row.get("verdict") in {"blocked_zero_train", "contaminated_pretraining"})

    # 方向缺口按同一套 post_training 审计器重算，不在交接契约里另写计数。
    from .post_training import audit as post_audit
    pa = post_audit(root, recipes)

    return {
        "recipes": len(recipes),
        "implemented_recipes": implemented_recipes,
        "gpu_verified": 0,
        "eligible": eligible,
        "verified_papers": verified_papers,
        "methodology_papers": len(meth.get("papers", {})),
        "post_training_directions": len(post.get("directions", {})),
        "implemented_directions": pa.counts.get("implemented_directions", 0),
        "gap_directions": pa.counts.get("gap_directions", 0),
        "principle_blocked_recipes": blocked,
    }


def audit(root: Path, recipes: Iterable[Recipe], *, data: dict[str, Any] | None = None) -> ContractAudit:
    root = Path(root)
    result = ContractAudit(ok=True)
    try:
        data = load(root) if data is None else data
        result.data = data
        result.facts = _facts(root, recipes)
    except AgentContractError as exc:
        result.ok = False
        result.errors.append(str(exc))
        return result
    except Exception as exc:  # 交叉注册表损坏时也必须失败关闭，不能生成假交接文档。
        result.ok = False
        result.errors.append(f"交叉注册表读取失败: {exc}")
        return result

    integrity = data.get("integrity", {})
    forbidden = integrity.get("forbidden_claim_phrases", [])
    allowed_expectations = set(integrity.get("allowed_expectations", []))
    if not _argv(forbidden) or not allowed_expectations:
        result.errors.append("integrity.forbidden_claim_phrases / allowed_expectations 不得为空")

    def _scan(text: Any, where: str) -> None:
        if not _nonempty_str(text):
            return
        for phrase in forbidden:
            if phrase and phrase in text:
                result.errors.append(f"{where}: 出现禁止宣称 {phrase!r}")

    # 静态入口文件必须存在且含 Agent 可发现标记。
    agents_rel = data.get("agent_instructions", "")
    if not _nonempty_str(agents_rel):
        result.errors.append("agent_instructions 不得为空")
    else:
        agents = root / str(agents_rel)
        if not agents.is_file():
            result.errors.append(f"Agent 入口文件不存在: {agents_rel}")
        else:
            text = agents.read_text(encoding="utf-8")
            for marker in AGENT_MARKERS:
                if marker not in text:
                    result.errors.append(f"{agents_rel}: 缺少 Agent 可发现标记 {marker!r}")

    for key in ("handoff_md", "handoff_json", "root_marker", "python_min", "entrypoint"):
        if not _nonempty_str(data.get(key)):
            result.errors.append(f"{key} 不得为空")
    if not (root / str(data.get("root_marker", ""))).is_file():
        result.errors.append(f"根目录标记缺失: {data.get('root_marker')}")

    # 事实源：支持 glob；无匹配视为登记失效。
    sources = data.get("sources_of_truth", [])
    if not isinstance(sources, list) or not sources:
        result.errors.append("sources_of_truth 不得为空")
    else:
        for i, item in enumerate(sources):
            if not isinstance(item, dict):
                result.errors.append(f"sources_of_truth[{i}] 必须是表")
                continue
            pattern = item.get("path")
            if not _nonempty_str(pattern) or not _nonempty_str(item.get("purpose")):
                result.errors.append(f"sources_of_truth[{i}]: path / purpose 不得为空")
                continue
            if not list(root.glob(str(pattern))):
                result.errors.append(f"sources_of_truth[{i}]: 路径无匹配 {pattern!r}")
            _scan(item["purpose"], f"sources_of_truth[{i}].purpose")

    # 生成物：路径存在、argv 可重建、哈希入清单。
    artifacts = data.get("generated_artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        result.errors.append("generated_artifacts 不得为空")
    else:
        for i, item in enumerate(artifacts):
            if not isinstance(item, dict):
                result.errors.append(f"generated_artifacts[{i}] 必须是表")
                continue
            rel = item.get("path")
            argv = _argv(item.get("argv"))
            if not _nonempty_str(rel) or not str(rel).startswith("docs/"):
                result.errors.append(f"generated_artifacts[{i}]: path 必须是 docs/ 下的文件")
                continue
            if not argv or argv[0] not in GENERATORS or argv.count("{out}") != 1:
                result.errors.append(
                    f"generated_artifacts[{i}]: argv 必须是已知生成器且含一个 {{out}}")
            if not _nonempty_str(item.get("purpose")):
                result.errors.append(f"generated_artifacts[{i}]: purpose 不得为空")
            path = root / str(rel)
            if not path.is_file():
                result.errors.append(f"生成物缺失: {rel}")
                continue
            result.artifacts.append({
                "path": str(rel),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "argv": argv,
                "purpose": item.get("purpose"),
            })
            _scan(item["purpose"], f"generated_artifacts[{i}].purpose")

    # 命令：id 唯一，期望值枚举合法，提升目标必须登记为 fail_closed。
    commands = data.get("commands", [])
    if not isinstance(commands, list) or not commands:
        result.errors.append("commands 不得为空")
    else:
        seen: set[str] = set()
        for i, item in enumerate(commands):
            if not isinstance(item, dict):
                result.errors.append(f"commands[{i}] 必须是表")
                continue
            cid = item.get("id")
            argv = _argv(item.get("argv"))
            expectation = item.get("expectation")
            if not _nonempty_str(cid) or cid in seen:
                result.errors.append(f"commands[{i}]: id 缺失或重复 {cid!r}")
            seen.add(str(cid))
            if not argv or argv[0] not in ALLOWED_TOP:
                result.errors.append(f"commands[{i}]: argv 非法 {argv!r}")
            if expectation not in allowed_expectations:
                result.errors.append(f"commands[{i}]: 非法 expectation {expectation!r}")
            if not _nonempty_str(item.get("purpose")):
                result.errors.append(f"commands[{i}]: purpose 不得为空")
            _scan(item.get("purpose"), f"commands[{i}].purpose")
        if not any(c.get("expectation") == "fail_closed" for c in commands if isinstance(c, dict)):
            result.errors.append("commands 必须至少登记一个 fail_closed 提升目标")

    guards = data.get("guardrails", [])
    if not isinstance(guards, list) or not guards:
        result.errors.append("guardrails 不得为空")
    else:
        seen_guard: set[str] = set()
        for i, item in enumerate(guards):
            if not isinstance(item, dict):
                result.errors.append(f"guardrails[{i}] 必须是表")
                continue
            gid = item.get("id")
            if not _nonempty_str(gid) or gid in seen_guard:
                result.errors.append(f"guardrails[{i}]: id 缺失或重复 {gid!r}")
            seen_guard.add(str(gid))
            if not _nonempty_str(item.get("statement")):
                result.errors.append(f"guardrails[{i}]: statement 不得为空")
            _scan(item.get("statement"), f"guardrails[{i}].statement")

    result.ok = not result.errors
    return result


def verify_regeneration(root: Path, checked: ContractAudit) -> ContractAudit:
    """用真实 CLI 把每个生成物重建到临时路径，逐字节比较。"""
    if not checked.ok:
        return checked
    root = Path(root)
    checked.regeneration_checked = True
    child_env = dict(os.environ)
    child_env["RECIPE_LAB_ROOT"] = str(root)
    child_env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, artifact in enumerate(checked.artifacts):
            rel = artifact["path"]
            out = tmp / f"{i}.md"
            argv = [x if x != "{out}" else str(out) for x in artifact["argv"]]
            p = subprocess.run(
                [sys.executable, "-S", "-m", "recipe_lab", *argv],
                cwd=root, env=child_env, capture_output=True, text=True,
            )
            if p.returncode != 0:
                checked.errors.append(f"{rel}: 重建命令失败 rc={p.returncode}: {p.stderr.strip()[:200]}")
                continue
            if not out.is_file() or out.read_bytes() != (root / rel).read_bytes():
                checked.errors.append(f"{rel}: 仓库文件与当前源码/注册表重建结果不一致")
    checked.ok = not checked.errors
    return checked


def manifest(root: Path, checked: ContractAudit) -> dict[str, Any]:
    data = checked.data
    return {
        "schema": 1,
        "contract_version": data.get("contract_version"),
        "assessed_at": data.get("assessed_at"),
        "root_marker": data.get("root_marker"),
        "python_min": data.get("python_min"),
        "entrypoint": data.get("entrypoint"),
        "ok": checked.ok,
        "errors": checked.errors,
        "facts": checked.facts,
        "sources_of_truth": data.get("sources_of_truth", []),
        "generated_artifacts": checked.artifacts,
        "commands": data.get("commands", []),
        "guardrails": data.get("guardrails", []),
    }


def render_json(root: Path, checked: ContractAudit) -> str:
    return json.dumps(manifest(root, checked), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def render_markdown(root: Path, checked: ContractAudit) -> str:
    data = checked.data
    f = checked.facts
    lines = [
        "# Agent 交接契约（人类可读视图）", "",
        "> 由 `recipe-lab handoff --report` 生成；单一事实源是 `registry/agent_contract.toml`。"
        "机器可读版本见 [AGENT-HANDOFF.json](AGENT-HANDOFF.json)。请勿手工编辑本文件。", "",
        "## 这份文件解决什么", "",
        "仓库同时服务两类读者：人需要快速理解现状与边界；Agent 需要确定性命令、事实源和"
        "可校验的重建结果。本页是人的入口，JSON 是机器的入口，两者都由同一份契约和真实审计器生成。", "",
        "这不是第六层审计，也不认证 GPU 执行。静态 JSON、hash、测试和 dry-run 只证明内部一致性；"
        "GPU verified 0、B2+/B3 0、L2+ 0、eligible 0/14 仍独立判定。", "",
        "## 当前事实", "",
        f"- 配方：**{f.get('recipes', 0)} 篇**；五维已实现 **{f.get('implemented_recipes', 0)} 篇**；"
        f"GPU verified **{f.get('gpu_verified', 0)}**；eligible **{f.get('eligible', 0)}**。",
        f"- 方法论：**{f.get('verified_papers', 0)}/{f.get('methodology_papers', 0)}** 篇论文一手核实。",
        f"- 后训练方向：**{f.get('implemented_directions', 0)}/{f.get('post_training_directions', 0)}** "
        f"有五维实现支撑，缺口 **{f.get('gap_directions', 0)}** 个。",
        f"- 原理性阻塞：{', '.join('`'+x+'`' for x in f.get('principle_blocked_recipes', [])) or '无'}。", "",
        "## 标准工作流", "",
        "1. 先读 `AGENTS.md`，再看本页与 `AGENT-HANDOFF.json`。",
        "2. 只修改事实源：`registry/`、`recipes/`、`src/`、`tests/`；不要手改 `docs/` 下生成报告。",
        "3. 运行 `make report` 重建五份核心审计生成物与两份 Agent 交接视图，再运行 `recipe-lab handoff --check` 验证可重建性。",
        "4. 运行 `make ci`；提升目标 `gate-*` 必须继续失败关闭。",
        "5. 没有用户明确要求，不要 push 或对外发布。", "",
        "## 事实源", "", "| path | purpose |", "| --- | --- |",
    ]
    for item in data.get("sources_of_truth", []):
        purpose = str(item.get("purpose", "")).replace("|", "\\|")
        lines.append(f"| `{item.get('path')}` | {purpose} |")
    lines.extend(["", "## 生成物与当前哈希", "",
                  "| path | 生成器 | sha256 | purpose |", "| --- | --- | --- | --- |"])
    for item in checked.artifacts:
        lines.append(
            f"| `{item['path']}` | `{' '.join(item['argv'])}` | `{item['sha256'][:12]}` | "
            f"{str(item['purpose']).replace('|', '\\|')} |")
    lines.extend(["", "## 命令契约", "", "| id | expectation | argv | purpose |",
                  "| --- | --- | --- | --- |"])
    for item in data.get("commands", []):
        lines.append(
            f"| `{item.get('id')}` | {item.get('expectation')} | "
            f"`{' '.join(_argv(item.get('argv')))}` | "
            f"{str(item.get('purpose', '')).replace('|', '\\|')} |")
    lines.extend(["", "## Agent Guardrails", ""])
    for item in data.get("guardrails", []):
        lines.append(f"- **{item.get('id')}**：{item.get('statement')}")
    lines.extend([
        "", "## 校验状态", "",
        f"- 交接契约：{'通过' if checked.ok else '未通过'}；"
        f"生成物可重建性：{'已校验' if checked.regeneration_checked else '未校验（仅静态检查）'}。",
    ])
    if checked.errors:
        lines.extend(["", "### 错误", ""] + [f"- [FAIL] {e}" for e in checked.errors] + [""])
    return "\n".join(lines)
