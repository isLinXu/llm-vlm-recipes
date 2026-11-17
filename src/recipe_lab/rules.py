"""验证规则集：把调研报告里的结论编码成可执行约束。

分层：
  L0 结构      —— schema / id / 枚举 / 必填字段
  L1 契约      —— trainer 映射、TRL dataset type、VLM 数据契约、训练-推理一致性
  L2 版本      —— 精确 pin、组合白名单、来源可信度、跨 recipe 漂移
  L3 纪律      —— 零训练基线、评估闭环、reward 失败路径、成本闸门
  L4 执行      —— 入口/证据文件存在性（不做真实训练，需 GPU 时由外部 runner 接管）
  L5 可复现性  —— [repro] 八维度、等级重算防虚报、声明与实际证据一致性
  L6 基线就绪  —— [baseline_contract] 四组能力、状态重算防虚报、可执行契约

L4 / L5 / L6 是三件事：
  L4 查「声称跑过就要有 evidence 文件」；
  L5 查「声称可复现就必须真的锁定模型/数据/环境/种子」；
  L6 查「声称能当 baseline 就必须有确定命令、可对照指标与实测门槛」。
静态验证通过（0 error）只证明配置自洽，不证明结果可复现，更不证明能当基线。

严重度：error 必须修；warning 是已登记的待办风险；info 是提示。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlparse

from . import STATUS_ORDER
from .loader import relative_recipe_id
from .model import Finding, Recipe

LEVEL_NAMES = {
    "l0": 0,
    "l1": 1,
    "l2": 2,
    "l3": 3,
    "l4": 4,
    "l5": 5,
    "l6": 6,
}

REQUIRED_TOP = [
    "schema",
    "id",
    "name",
    "status",
    "coverage",
    "confidence",
    "source",
    "method",
    "model",
    "dataset",
]
STATUS_VALUES = set(STATUS_ORDER)
COVERAGE_VALUES = {"params", "catalog"}
CONFIDENCE_VALUES = {"primary", "secondary", "unknown"}
MODALITY_VALUES = {"llm", "vlm"}
PINNED = re.compile(r"^\d[\w.+-]*$")


@dataclass
class Context:
    recipes: list[Recipe]
    taxonomy: dict
    root: Path
    level: int = 3

    @property
    def families(self) -> dict:
        return self.taxonomy.get("families", {})

    @property
    def compat(self) -> list[dict]:
        return self.taxonomy.get("compat", {}).get("matrix", [])


Rule = Callable[[Recipe, Context], Iterator[Finding]]
GlobalRule = Callable[[Context], Iterator[Finding]]


def _f(
    r: Recipe | None,
    level: int,
    rule: str,
    severity: str,
    message: str,
    hint: str = "",
) -> Finding:
    return Finding(
        level=f"L{level}",
        rule=rule,
        severity=severity,
        recipe_id=r.id if r else "GLOBAL",
        message=message,
        hint=hint,
    )


def _emit(
    r: Recipe, level: int, rule: str, severity: str, message: str, hint: str = ""
) -> Iterator[Finding]:
    yield _f(r, level, rule, severity, message, hint)


# --------------------------------------------------------------------------- L0


def l0_structure(r: Recipe, ctx: Context) -> Iterator[Finding]:
    for key in REQUIRED_TOP:
        if not r.has(key):
            yield from _emit(r, 0, "R001", "error", f"缺少必填字段 `{key}`")

    expected = relative_recipe_id(r.path, ctx.root)
    if r.id != expected:
        yield from _emit(
            r, 0, "R002", "error", f"id `{r.id}` 与目录路径 `{expected}` 不一致",
            "id 必须等于 recipes/ 下的相对目录，便于按路径定位",
        )

    for dotted, allowed in (
        ("status", STATUS_VALUES),
        ("coverage", COVERAGE_VALUES),
        ("confidence", CONFIDENCE_VALUES),
        ("model.modality", MODALITY_VALUES),
    ):
        if r.has(dotted) and r.get(dotted) not in allowed:
            yield from _emit(
                r, 0, "R003", "error",
                f"`{dotted}` = {r.get(dotted)!r} 不在允许值 {sorted(allowed)} 内",
            )

    if r.has("schema") and r.get("schema") != ctx.taxonomy.get("schema", 1):
        yield from _emit(
            r, 0, "R004", "error",
            f"schema 版本 {r.get('schema')} 与 registry/taxonomy.toml 的 "
            f"{ctx.taxonomy.get('schema', 1)} 不匹配",
        )

    if r.has("status") and r.get("status") in ("smoke-passed", "benchmarked"):
        if r.get("coverage") != "params":
            yield from _emit(
                r, 0, "R005", "error",
                "声称已跑通/已评测，但 coverage 不是 params（缺少逐参数配置）",
            )


# --------------------------------------------------------------------------- L1


def l1_contract(r: Recipe, ctx: Context) -> Iterator[Finding]:
    fam_key = r.family
    fam = ctx.families.get(fam_key)

    if fam is None:
        yield from _emit(
            r, 1, "R101", "error", f"method.family `{fam_key}` 未在 taxonomy 中登记",
            "先在 registry/taxonomy.toml 补一条 family，或改用已登记的 family",
        )
        return

    if fam.get("tier") == "removed":
        yield from _emit(
            r, 1, "R102", "warning",
            f"`{fam_key}` 已从 TRL v1 稳定层移除：{fam.get('note', '')}",
            "新 recipe 不应基于已移除的 trainer；仅作历史对照可保留 cataloged 状态",
        )

    trainer = r.get("method.trainer")
    trainers = fam.get("task_trainers", {})
    adapter = trainers.get(r.get("execution.task_id"), {})
    if not adapter:
        # 未声明 execution.task_id 的任务（含 Granite 三维阶段与未来尚未注册
        # 发布级管线的任务）按设计不写 task_id，否则 resolve() 会拒绝借用未实现
        # 管线。这类任务仍必须按精确 recipe_id 登记 trainer 例外，绝不按 family 猜测；
        # 已注册发布级管线的任务走上面的 task_id 精确索引。
        adapter = next(
            (item for item in trainers.values()
             if isinstance(item, dict) and item.get("recipe_id") == r.id), {})
    expected_trainer = (adapter.get("trainer") if adapter.get("recipe_id") == r.id
                        else fam.get("trainer"))
    if trainer and trainer != expected_trainer:
        yield from _emit(
            r, 1, "R103", "error",
            f"trainer `{trainer}` 与 family `{fam_key}` 的登记值 `{expected_trainer}` 不符",
            fam.get("note", ""),
        )

    ds_type = r.get("dataset.type")
    allowed_types = set(fam.get("dataset_types", []))
    all_types = set(ctx.taxonomy.get("allowed_dataset_types", []))
    if not ds_type:
        yield from _emit(r, 1, "R104", "error", "缺少 `dataset.type`（TRL dataset type）")
    elif ds_type not in all_types:
        yield from _emit(
            r, 1, "R104", "error", f"未知的 dataset.type `{ds_type}`",
            f"允许值：{sorted(all_types)}",
        )
    elif ds_type not in allowed_types:
        yield from _emit(
            r, 1, "R104", "error",
            f"`{fam.get('trainer')}` 不接受 dataset.type `{ds_type}`",
            f"该 trainer 允许：{sorted(allowed_types)}（依据 TRL dataset_formats）",
        )

    structure = r.get("dataset.structure")
    if structure and structure not in ctx.taxonomy.get("allowed_structures", []):
        yield from _emit(r, 1, "R105", "error", f"未知的 dataset.structure `{structure}`")

    # 已登记例外（见 registry/evidence.toml F07）：prompt-only VLM GRPO 用
    # processor.apply_chat_template 把多模态对话预渲染成字符串 prompt，
    # 数据集里没有 messages 列，只有 prompt(str) + image(PIL)。
    pre_rendered = r.get("modality_contract.pre_rendered_prompt") is True
    prompt_only_vlm = r.get("dataset.type") == "prompt-only" and pre_rendered

    if r.is_vlm and structure != "conversational":
        if prompt_only_vlm:
            yield from _emit(
                r, 1, "R105", "info",
                "prompt-only VLM 使用预渲染字符串 prompt，属已登记例外",
                "见 docs/CONTRACT.md「prompt-only VLM 例外」与 evidence.toml F07",
            )
        else:
            yield from _emit(
                r, 1, "R105", "error",
                "VLM 必须使用 conversational 结构",
                "TRL 官方：messages 中 content 为 list[dict]，用 {'type':'image'} / {'type':'text'} 表达",
            )

    if r.get("model.modality") == "vlm" and not r.get("dataset.has_images", False):
        yield from _emit(r, 1, "R114", "error", "model.modality=vlm 但 dataset.has_images 未置 true")
    if r.get("dataset.has_images", False) and r.get("model.modality") != "vlm":
        yield from _emit(r, 1, "R114", "error", "数据集含图像，但 model.modality 不是 vlm")

    if not r.is_vlm:
        return

    # --- 以下四条即调研报告「决定 80% VLM 训练失败」的数据契约 ---
    if not r.has("modality_contract"):
        yield from _emit(
            r, 1, "R106", "error", "VLM recipe 缺少 [modality_contract] 段",
            "必须显式声明图像列、content 结构、processor 与长度策略",
        )
        return

    image_col = r.get("modality_contract.images_column")
    if image_col not in ("images", "image"):
        yield from _emit(
            r, 1, "R107", "error",
            f"images_column = {image_col!r}，必须是顶层 `images`（多图）或 `image`（单图）列",
            "图像不能只塞在 messages 内部；TRL 要求与 messages/prompt 并列的顶层列",
        )

    if r.get("modality_contract.content_list_dict") is not True:
        if prompt_only_vlm:
            yield from _emit(
                r, 1, "R108", "info",
                "prompt-only VLM 的 prompt 列是已渲染字符串，content_list_dict=false 属已登记例外",
                "上游 make_conversation 内部仍用 list[dict] 构造对话，再交给 apply_chat_template",
            )
        else:
            yield from _emit(
                r, 1, "R108", "error",
                "modality_contract.content_list_dict 必须为 true",
                "VLM 的 message content 必须是 list[dict]，不是纯字符串",
            )

    if r.get("modality_contract.processing_class") != "processor":
        yield from _emit(
            r, 1, "R109", "error",
            "VLM 必须传 processing_class=processor，不能用 tokenizer",
            "processor 同时负责图像与文本；用 tokenizer 会静默丢掉视觉输入",
        )

    if fam_key in ("sft", "dpo", "mpo", "kto"):
        if r.get("modality_contract.max_length") != "none":
            yield from _emit(
                r, 1, "R110", "error",
                "VLM SFT/偏好训练必须设 max_length=None，否则图像 token 会被截断",
                "TRL 官方 dataset_formats 对 VLM 的明确建议",
            )
    if fam_key in ("grpo", "rloo", "online-dpo", "gspo") and not r.has(
        "modality_contract.max_prompt_length"
    ):
        yield from _emit(
            r, 1, "R110", "warning",
            "VLM 在线 RL 未声明 max_prompt_length 策略",
            "需确认图像 token 不被截断；官方建议关注 max_prompt_length=None",
        )

    if r.has("consistency"):
        train_sys = r.get("consistency.train_system_message")
        infer_sys = r.get("consistency.inference_system_message")
        if train_sys is not None and infer_sys is not None and train_sys != infer_sys:
            yield from _emit(
                r, 1, "R111", "error",
                "训练与推理的 system message 策略不一致",
                "上游多个 recipe 训练含 system、推理用 messages[1:2] 丢掉 system，属典型静默掉点",
            )

    if r.get("peft.use_dora") is True:
        tags = {str(t).lower() for t in r.get("tags", [])}
        if ("qlora" in tags or "lora" in tags) and "dora" not in tags:
            yield from _emit(
                r, 1, "R112", "warning",
                "代码启用 use_dora=True，但 tags 仍宣称 LoRA/QLoRA",
                "上游 SmolVLM / Granite / MPO 篇存在此文案与代码不符；此处以代码为准并补 tags",
            )

    if r.get("quantization.load_in_4bit") is True:
        dtype = r.get("quantization.bnb_4bit_compute_dtype")
        if dtype not in ("bfloat16", "float16"):
            yield from _emit(
                r, 1, "R113", "error",
                f"4bit 量化的 compute dtype = {dtype!r}，应为 bfloat16 / float16",
            )

    if fam.get("requires_verifiable_reward"):
        funcs = r.get("rewards.functions", [])
        if not funcs:
            yield from _emit(
                r, 1, "R115", "error",
                f"`{fam_key}` 依赖可验证奖励，但 [rewards].functions 为空",
                "GRPO/RLOO 没有 value model，reward 定义就是算法本身",
            )


# --------------------------------------------------------------------------- L2


def l2_versions(r: Recipe, ctx: Context) -> Iterator[Finding]:
    policy = ctx.taxonomy.get("version_policy", {})
    pattern = re.compile(policy.get("pinned_pattern", r"^\d[\w.+-]*$"))
    dev_markers = policy.get("dev_markers", [])

    for pkg, version in (r.get("dependencies", {}) or {}).items():
        ver = str(version)
        if not pattern.fullmatch(ver):
            yield from _emit(
                r, 2, "R201", "error",
                f"依赖 `{pkg} = {ver!r}` 不是精确版本 pin",
                "禁止 >=、^、~、latest、未带 commit 的 git URL；跨 recipe 复制代码时这是首要风险",
            )
            continue
        if any(m in ver.lower() for m in dev_markers):
            yield from _emit(
                r, 2, "R203", "warning",
                f"`{pkg} = {ver}` 是预发布/dev 版本，通常需要从源码安装",
                "上游 VLM SFT/GRPO/MPO 篇因此要求 pip install git+.../trl.git；建议锁定 commit",
            )

    trl = str(r.get("dependencies.trl", ""))
    tf = str(r.get("dependencies.transformers", ""))
    if trl and tf:
        known = [m for m in ctx.compat if m.get("trl") == trl and m.get("transformers") == tf]
        if not known:
            yield from _emit(
                r, 2, "R202", "warning",
                f"trl {trl} + transformers {tf} 不在已验证组合白名单",
                "补一次冒烟验证后把组合加入 registry/taxonomy.toml 的 [[compat.matrix]]",
            )

    source = str(r.get("source", ""))
    parsed = urlparse(source)
    hosts = policy.get("allowed_source_hosts", [])
    if parsed.scheme != "https" or parsed.netloc not in hosts:
        yield from _emit(
            r, 2, "R205", "error",
            f"source 必须是登记域名的 HTTPS 链接，当前为 {source!r}",
            f"允许域名：{hosts}",
        )
    if not r.get("source_sha"):
        yield from _emit(
            r, 2, "R206", "info",
            "未锁定上游 commit（source_sha 为空）",
            "上游 notebook 会漂移；建议记录抓取时的 commit sha，便于 diff",
        )


def l2_drift(ctx: Context) -> Iterator[Finding]:
    versions: dict[str, list[str]] = {}
    for r in ctx.recipes:
        trl = str(r.get("dependencies.trl", ""))
        if trl:
            versions.setdefault(trl, []).append(r.id)

    if len(versions) > 1:
        summary = ", ".join(f"{k} ({len(v)} 篇)" for k, v in sorted(versions.items()))
        yield _f(
            None, 2, "R204", "warning",
            f"仓库内存在 {len(versions)} 个不同 trl 版本：{summary}",
            "跨 recipe 复制代码前必须核对该版本组合；这正是上游 cookbook 最常见的隐性故障源",
        )

    by_family: dict[str, set[str]] = {}
    for r in ctx.recipes:
        if r.get("dependencies.trl"):
            by_family.setdefault(r.family, set()).add(str(r.get("dependencies.trl")))
    for family, vers in sorted(by_family.items()):
        if len(vers) > 1:
            yield _f(
                None, 2, "R207", "info",
                f"family `{family}` 下存在多个 trl 版本 {sorted(vers)}",
                "做同族方法对照实验时，版本差异会污染结论，建议对齐到同一环境",
            )


# --------------------------------------------------------------------------- L3


def l3_discipline(r: Recipe, ctx: Context) -> Iterator[Finding]:
    catalog = r.is_catalog_only

    if not r.has("baseline"):
        yield from _emit(
            r, 3, "R301", "warning" if catalog else "error",
            "缺少 [baseline] 零训练基线",
            "调研第一结论：先问 prompting / RAG 基线能否达标，再决定是否微调",
        )
    else:
        if r.get("baseline.zero_train") is not True:
            yield from _emit(
                r, 3, "R301", "error",
                "baseline.zero_train 必须为 true（先做零训练基线）",
            )
        result = r.get("baseline.result")
        if result == "passes" and not catalog:
            yield from _emit(
                r, 3, "R302", "error",
                "零训练基线已达标，但 recipe 状态不是 cataloged —— 不应投入训练预算",
                "起点 recipe 第 6 节：base model + system message 无需训练即可答对",
            )
        elif result == "passes" and catalog:
            yield from _emit(
                r, 3, "R303", "info",
                "零训练基线达标：本 recipe 仅作教学/方法对照，请勿据此启动训练",
            )
        elif result == "unmeasured":
            yield from _emit(
                r, 3, "R303", "warning" if not catalog else "info",
                "零训练基线尚未测量",
                "把基线结果补进 baseline.result（passes / fails）后再推进 status",
            )

    metrics = r.get("evaluation.metrics", []) or []
    soft_metrics = {"vibe_check", "qualitative", "single_sample", "demo"}
    if not metrics:
        yield from _emit(
            r, 3, "R304", "info" if catalog else "warning",
            "未声明定量评估指标",
            "Cookbook 最大方法论缺口：多数 recipe 只展示 1 个样本的定性生成",
        )
    elif set(map(str, metrics)) & soft_metrics:
        yield from _emit(
            r, 3, "R304", "error",
            "评估指标包含定性/单样本占位，不构成评估闭环",
        )

    if r.has("evaluation"):
        if r.get("evaluation.held_out") is not True:
            yield from _emit(
                r, 3, "R305", "error",
                "evaluation.held_out 必须为 true（未见样本）",
                "上游 RapidFire eval range(50,60) 落在训练 range(64) 内，属数据泄漏",
            )
        if not r.has("evaluation.seed"):
            yield from _emit(r, 3, "R306", "warning", "评估未固定 seed")
        if not catalog and r.get("evaluation.multi_run") is not True:
            yield from _emit(
                r, 3, "R306", "warning",
                "非 catalog 状态应声明 multi_run=true（多次评估取统计）",
                "上游 grounding 篇 do_sample=True 且无种子，结果不可比",
            )

    fam = ctx.families.get(r.family, {})
    if fam.get("requires_verifiable_reward"):
        if r.get("rewards.failure_path_tested") is not True:
            yield from _emit(
                r, 3, "R307", "error",
                "RL recipe 未声明 reward 失败路径测试",
                "Wordle 篇：invalid move 也拿 partial credit，不能用 reward>0 判定胜利；"
                "正确条件是 bool(env.done and reward >= ENV_SOLVED_REWARD)",
            )

    if r.has("budget"):
        smoke = r.get("budget.smoke_steps")
        full = r.get("budget.full_steps")
        if isinstance(smoke, int) and isinstance(full, int) and smoke >= full:
            yield from _emit(
                r, 3, "R308", "error",
                f"smoke_steps ({smoke}) 必须小于 full_steps ({full})",
                "SMOKE 是成本闸门，不应退化为全量训练",
            )
    elif not catalog:
        yield from _emit(
            r, 3, "R308", "warning",
            "缺少 [budget] 成本预算（smoke/full 双档）",
        )

    issues = r.get("known_issues.items", []) or []
    deviations = r.get("upstream_deviations.items", []) or []
    todo = r.get("todo.items", []) or []
    if issues and not (deviations or todo):
        yield from _emit(
            r, 3, "R309", "warning",
            "登记了上游缺陷，但没有 upstream_deviations 或 todo 说明如何处置",
            "本仓库维护的是「修正后的配方」，上游问题必须对应一条明确处置",
        )
    if catalog and not todo:
        yield from _emit(
            r, 3, "R312", "warning",
            "cataloged recipe 必须有 [todo] 列出补齐项",
            "cataloged 是「仅登记、待移植」状态，没有待办就无法推进",
        )


# --------------------------------------------------------------------------- L4


def l4_execution(r: Recipe, ctx: Context) -> Iterator[Finding]:
    entry = r.get("execution.entrypoint")
    if entry:
        target = r.path.parent / str(entry)
        if not target.is_file():
            yield from _emit(
                r, 4, "R401", "error",
                f"execution.entrypoint 指向的文件不存在：{entry}",
                f"期望路径：{target}",
            )

    if r.get("execution.smoke_command") and not r.get("execution.evidence"):
        yield from _emit(
            r, 4, "R403", "warning",
            "声明了 smoke_command 但没有 execution.evidence 产物路径",
        )
    if r.status in ("smoke-passed", "benchmarked"):
        evidence = r.get("execution.evidence")
        if not evidence:
            yield from _emit(
                r, 4, "R403", "error",
                f"status={r.status} 但未提供 evidence 路径",
                "声称跑通/评测过，必须留可复查的日志或指标文件",
            )
        elif not (r.path.parent / str(evidence)).exists():
            yield from _emit(
                r, 4, "R403", "error", f"evidence 文件不存在：{evidence}",
            )

    if not entry:
        yield from _emit(
            r, 4, "R402", "info",
            "未声明 execution.entrypoint：本仓库当前只做静态契约验证",
            "接入真实训练时补入口脚本，并由外部 GPU runner 执行 L4",
        )


# --------------------------------------------------------------------------- L5
# 可复现性审计：等级不是「填」出来的，而是用 registry/reproducibility.toml
# 对八个维度重算出来的。R503 是整套机制的支点——任何虚报都会在这里变成 error。


def l5_reproducibility(r: Recipe, ctx: Context) -> Iterator[Finding]:
    from .repro import DIMS, STATE_ORDER, audit_one, load_contract, state_rank

    contract = load_contract(ctx.root)
    states = set(contract.get("states", []))

    if not r.has("repro"):
        yield from _emit(
            r, 5, "R501", "error", "缺少 [repro] 可复现性审计段",
            "运行 PYTHONPATH=src python3 scripts/apply_repro_audit.py 生成，或手工填八维度",
        )
        return

    for key in ("level", "assessed_at", "assessed_by", "evidence_ref"):
        if not r.has(f"repro.{key}"):
            yield from _emit(r, 5, "R502", "error", f"[repro] 缺少必填字段 `{key}`")

    claimed = str(r.get("repro.level", "") or "")
    if claimed and claimed not in contract.get("levels", {}):
        yield from _emit(
            r, 5, "R502", "error",
            f"repro.level = {claimed!r} 不是合法等级",
            f"允许值：{sorted(contract.get('levels', {}))}",
        )
    assessed_at = str(r.get("repro.assessed_at", "") or "")
    if assessed_at and assessed_at != str(contract.get("assessed_at", "")):
        yield from _emit(
            r, 5, "R502", "warning",
            f"repro.assessed_at = {assessed_at} 与契约登记日期 "
            f"{contract.get('assessed_at')} 不一致，说明该篇未随本轮审计更新",
        )

    for dim in DIMS:
        prefix = f"repro.{dim}"
        if not r.has(prefix):
            yield from _emit(r, 5, "R504", "error", f"缺少 [{prefix}] 维度段")
            continue
        state = str(r.get(f"{prefix}.state", "") or "")
        if state not in states:
            yield from _emit(
                r, 5, "R504", "error",
                f"`{prefix}.state` = {state!r} 非法",
                f"允许值：{sorted(states)}",
            )
            continue
        gap = str(r.get(f"{prefix}.gap", "") or "")

        if state_rank(state) >= STATE_ORDER["pinned"]:
            # 声称已锁定，就必须有 backing 字段，且路径类字段指向真实文件。
            field = contract["dimensions"][dim].get("evidence_field")
            if field and not r.has(field):
                yield from _emit(
                    r, 5, "R505", "error",
                    f"`{prefix}.state = {state}` 但缺少 backing 字段 `{field}`",
                    "等级只能由证据支撑；先在 recipe.toml 补齐该字段再提升 state",
                )
            elif field and dim in ("code", "env", "eval"):
                target = r.path.parent / str(r.get(field))
                if not target.exists():
                    yield from _emit(
                        r, 5, "R505", "error",
                        f"`{field}` 指向的文件不存在：{r.get(field)}",
                        f"期望路径：{target}",
                    )
            if gap:
                yield from _emit(
                    r, 5, "R506", "warning",
                    f"`{prefix}` 已标 {state} 但仍留有 gap 文本，请清空或降级",
                )
        elif not gap:
            yield from _emit(
                r, 5, "R506", "warning",
                f"`{prefix}.state = {state}` 但没有写清 gap（缺什么、为什么不能锁定）",
                "审计的价值在缺口说明；空 gap 无法驱动补齐",
            )

    audit = audit_one(r, contract)
    if not audit.level_ok:
        yield from _emit(
            r, 5, "R503", "error",
            f"repro.level 虚报：声明 {audit.claimed or '(空)'}，按八维度重算为 {audit.computed}",
            "等级由 registry/reproducibility.toml 自动计算，不允许手工抬高",
        )


def l5_evidence_refs(ctx: Context) -> Iterator[Finding]:
    from .repro import evidence_ref_problems, load_contract, load_evidence

    contract = load_contract(ctx.root)
    evidence = load_evidence(ctx.root)
    for rid, (severity, message) in evidence_ref_problems(
        ctx.recipes, contract, evidence
    ).items():
        recipe = next((r for r in ctx.recipes if r.id == rid), None)
        yield _f(
            recipe, 5, "R507", severity, message,
            "一手材料必须可追溯；把页面/文件名登记进 evidence.toml [method] 后再引用",
        )


# --------------------------------------------------------------------------- L6
# 基线就绪：状态与 L5 同构，由 registry/baseline.toml 的 checks 重算，
# R603 是支点。区别在于 L6 允许「B1 契约齐备但未实跑」，因为 baseline
# 是我们自己定义并锁定的参照物，不需要复刻上游原实验。


def l6_baseline(r: Recipe, ctx: Context) -> Iterator[Finding]:
    from .baseline import STATE_ORDER, audit_one, check_passed, load_contract

    contract = load_contract(ctx.root)
    states = set(contract.get("states", []))
    tiers = set(contract.get("tiers", ["research", "business"]))

    if not r.has("baseline_contract"):
        yield from _emit(
            r, 6, "R601", "error", "缺少 [baseline_contract] 基线就绪契约段",
            "运行 scripts/apply_baseline_contract.py 生成，或手工填四组能力字段",
        )
        return

    for key in ("state", "target_tier", "experiment_id", "revision_policy"):
        if not r.has(f"baseline_contract.{key}"):
            yield from _emit(
                r, 6, "R602", "error", f"[baseline_contract] 缺少必填字段 `{key}`"
            )

    claimed = str(r.get("baseline_contract.state", "") or "")
    if claimed and claimed not in states:
        yield from _emit(
            r, 6, "R602", "error",
            f"baseline_contract.state = {claimed!r} 不是合法状态",
            f"允许值：{sorted(states)}",
        )
    tier = str(r.get("baseline_contract.target_tier", "") or "")
    if tier and tier not in tiers:
        yield from _emit(
            r, 6, "R602", "error",
            f"baseline_contract.target_tier = {tier!r} 不是合法档位",
            f"允许值：{sorted(tiers)}",
        )

    # 声称 research/business-ready 却没有实跑证据，check 本身会失败，
    # 这里额外给出更直接的错误信息，避免维护者以为改个字符串就能升级。
    if claimed in ("B2", "B3"):
        for c in contract["checks"]:
            if not 1 < STATE_ORDER[c["min_state"]] <= STATE_ORDER[claimed]:
                continue
            if not check_passed(r, c):
                yield from _emit(
                    r, 6, "R605", "error",
                    f"声称 {claimed} 但 {c['id']} `{c['label']}` 未满足",
                    c["how"],
                )

    audit = audit_one(r, contract)
    if not audit.state_ok:
        yield from _emit(
            r, 6, "R603", "error",
            f"baseline_contract.state 虚报：声明 {audit.claimed or '(空)'}，"
            f"按 {audit.n_total} 条检查重算为 {audit.computed}",
            "状态由 registry/baseline.toml 自动计算，不允许手工抬高",
        )

    if audit.computed == "B0" and not r.has("todo"):
        yield from _emit(
            r, 6, "R604", "warning",
            "基线未就绪（B0）却没有 [todo] 说明补齐项",
            "B0 至少列出缺失的 B1 检查项，否则无法推进",
        )

    target_state = "B3" if tier == "business" else "B2"
    if STATE_ORDER.get(audit.computed, 0) < STATE_ORDER[target_state]:
        yield from _emit(
            r, 6, "R606", "info",
            f"目标 {tier} baseline ({target_state})，当前 {audit.computed}，"
            f"尚有 {len(audit.gaps)} 项检查未通过",
            "按 docs/BASELINE-READINESS.md 的五维能力矩阵与wave0–wave6队列推进",
        )


PER_RECIPE_RULES: list[tuple[int, Rule]] = [
    (0, l0_structure),
    (1, l1_contract),
    (2, l2_versions),
    (3, l3_discipline),
    (4, l4_execution),
    (5, l5_reproducibility),
    (6, l6_baseline),
]

GLOBAL_RULES: list[tuple[int, GlobalRule]] = [
    (2, l2_drift),
    (5, l5_evidence_refs),
]
