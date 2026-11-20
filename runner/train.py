#!/usr/bin/env python3
"""统一训练入口：按精确 recipe_id 白名单分派各任务适配器；其余只读预演。"""
from __future__ import annotations

import os

# 必须先于任何 runner / 框架模块导入；进程已有 torch 时也不能补救其初始化。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RECIPE_LAB_ROOT", str(CODE_ROOT))).resolve()
sys.path.insert(0, str(CODE_ROOT / "src"))

from recipe_lab.loader import load_recipes
from recipe_lab import dpo_task as dpo_adapter
from recipe_lab import fim_task as fim_adapter
from recipe_lab import granite_task as granite_adapter
from recipe_lab import gsm8k_task as gsm8k_adapter
from recipe_lab import mpo_task as mpo_adapter
from recipe_lab import paligemma_task as paligemma_adapter
from recipe_lab import rapidfire_task as rapidfire_adapter
from recipe_lab import xlam_task as xlam_adapter
from recipe_lab.plan import build_plan, render_plan
from recipe_lab.provenance import effective_recipe, output_plan
from recipe_lab.math_task import (
    DatasetValidationError, TaskExecutionError, MODEL_ID, apply_determinism,
    artifact_manifest, check_signature, checkpoint_path, completed_evidence, config_kwargs, execution_identity,
    load_runtime, load_splits, require_revision, require_supported,
    runtime_evidence, save_json, task_plan, format_reward, accuracy_reward, require_frozen_runtime,
)

# 精确 recipe_id 白名单：数学是默认路径（GRPO 全参数），其余任务必须由自己的
# 适配器提供 task_plan / run_training；不在表内的 recipe 只能得到 --dry-run 声明。
# Granite 是 wave3 首个 VLM 任务，五维齐备且发布级由 task_router 精确分派。
TASK_ADAPTERS = {
    dpo_adapter.RECIPE_ID: dpo_adapter,
    fim_adapter.RECIPE_ID: fim_adapter,
    granite_adapter.RECIPE_ID: granite_adapter,
    gsm8k_adapter.RECIPE_ID: gsm8k_adapter,
    mpo_adapter.RECIPE_ID: mpo_adapter,
    paligemma_adapter.RECIPE_ID: paligemma_adapter,
    xlam_adapter.RECIPE_ID: xlam_adapter,
    rapidfire_adapter.RECIPE_ID: rapidfire_adapter,
}


def find_recipe(recipe_id: str):
    matches = [r for r in load_recipes(ROOT) if r.id.startswith(recipe_id)]
    if not matches:
        raise SystemExit(f"未找到 recipe: {recipe_id}")
    if len(matches) > 1:
        raise SystemExit(f"--recipe {recipe_id!r} 匹配 {len(matches)} 篇，请用完整 id")
    return matches[0]


def _build_and_train(r, plan: dict) -> dict:
    # 所有路由和预算校验在框架导入 / 下载之前执行。
    require_supported(r, plan)
    if r.get("decision.gate") == "blocked" or r.get("baseline.result") == "passes":
        raise TaskExecutionError("训练决策门已阻断：先复核零训练基线及业务目标，不允许绕过预算决策")
    revision = require_revision(r, "model.revision")
    require_revision(r, "dataset.revision")
    output = Path(plan["run"]["output_dir"])
    kwargs = config_kwargs(r, plan, output)
    if output.exists() and any(output.iterdir()):
        raise TaskExecutionError(f"训练目录非空：{output}；拒绝覆盖旧 checkpoint 或证据，请使用独立实验目录")
    dependencies = load_runtime(training=True)
    require_frozen_runtime(r, dependencies)
    plan = {**plan, "execution_identity": execution_identity(r)}
    started = time.perf_counter()
    apply_determinism(r, dependencies)
    trl = dependencies["trl"]
    try:
        config_cls, trainer_cls = trl.GRPOConfig, trl.GRPOTrainer
    except (AttributeError, ImportError) as exc:
        raise TaskExecutionError("trl==0.14.0 的 GRPOConfig / GRPOTrainer 不可用，请按 lock 重装依赖") from exc
    check_signature(config_cls, kwargs, "GRPOConfig")
    trainer_kwargs = {
        "model": None, "processing_class": None, "args": None,
        "train_dataset": None, "eval_dataset": None,
        "reward_funcs": [format_reward, accuracy_reward],
    }
    check_signature(trainer_cls, trainer_kwargs, "GRPOTrainer")
    config = config_cls(**kwargs)
    for key, value in kwargs.items():
        actual = getattr(config, key, None)
        if key != "report_to" and actual != value:
            raise TaskExecutionError(f"GRPOConfig 未应用 {key}：请求 {value!r}，实际 {actual!r}")
    splits, manifest = load_splits(r, dependencies)
    transformers = dependencies["transformers"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.eos_token is None:
        raise TaskExecutionError("tokenizer 缺少 eos_token，不能组装 GRPO")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, torch_dtype=dependencies["torch"].float32,
    )
    dataset_cls = dependencies["datasets"].Dataset
    datasets = {name: dataset_cls.from_list([
        {"prompt": row["prompt"], "solution": row["solution"]} for row in splits[name]
    ]) for name in ("train", "validation")}
    trainer_kwargs.update(model=model, processing_class=tokenizer, args=config,
                          train_dataset=datasets["train"], eval_dataset=datasets["validation"])
    trainer = trainer_cls(**trainer_kwargs)
    train_result = trainer.train()
    actual_steps = trainer.state.global_step
    if actual_steps != kwargs["max_steps"]:
        raise TaskExecutionError(f"训练未完成预算：实际 {actual_steps} 步，要求 {kwargs['max_steps']}；不写 completed evidence")
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    # 保存阶段尚未写 training.json，只验证产物结构；评估入口必须核对完整来源。
    checkpoint_path(plan, checkpoint=output, require_evidence=False)
    dependencies["torch"].cuda.synchronize()
    manifest_path = save_json(output / "manifest.json", manifest)
    effective_config = config.to_dict()
    evidence = completed_evidence(
        r, plan, "training", effective_config, manifest, runtime_evidence(dependencies, started),
    )
    evidence.update(
        steps=actual_steps, max_steps=kwargs["max_steps"],
        trainer="GRPOTrainer", rewards=["format_reward", "accuracy_reward"],
        model_artifact=artifact_manifest(output),
        split_manifest_path=str(manifest_path),
        train_metrics=dict(train_result.metrics),
    )
    return evidence


def write_evidence(r, plan: dict, result: dict) -> Path:
    if result.get("kind") != "training" or result.get("stage") != "completed":
        raise TaskExecutionError("只能保存真实完成的 training evidence")
    return save_json(Path(plan["run"]["output_dir"]) / "training.json", result)


def main() -> int:
    ap = argparse.ArgumentParser(description="recipe-lab 受控训练入口")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--task-id", help="显式任务路由：math-grpo-v1 / code-fim-v1 / xlam-fc-v1 / rapidfire-sft-v1 / granite-linecomparison-v1；与recipe必须匹配")
    ap.add_argument("--config-id", help="RapidFire 4 配置、MPO 三损失消融或 GSM8K drop-one reward 消融之一；只覆盖 [train]/[peft]/execution.config_id，默认取 recipe 声明")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--train-seed", type=int, help="临时覆盖训练seed，不修改数据/评估seed或TOML")
    ap.add_argument("--output-root", help="绝对实验输出根，各模式独立子目录")
    args = ap.parse_args()
    mode = "smoke" if args.smoke else "full" if args.full else "dry-run"
    try:
        r = effective_recipe(find_recipe(args.recipe), args.train_seed)
        if args.config_id:
            if r.id == rapidfire_adapter.RECIPE_ID:
                r = rapidfire_adapter.config_recipe(r, args.config_id)
            elif r.id == mpo_adapter.RECIPE_ID:
                r = mpo_adapter.config_recipe(r, args.config_id)
            elif r.id == gsm8k_adapter.RECIPE_ID:
                r = gsm8k_adapter.config_recipe(r, args.config_id)
            else:
                raise TaskExecutionError("--config-id 只适用于 RapidFire / MPO / GSM8K 多配置 recipe")
        adapter = TASK_ADAPTERS.get(r.id)
        planner = adapter.task_plan if adapter else task_plan
        plan = output_plan(planner(r, build_plan(r, mode=mode), mode, args.task_id), ROOT, args.output_root)
    except (ValueError, TaskExecutionError) as exc:
        ap.exit(1, f"参数错误：{exc}\n")
    if mode == "dry-run":
        print(render_plan(plan, compact=not args.json))
        return 0
    try:
        adapter = TASK_ADAPTERS.get(r.id)
        if adapter:
            evidence = adapter.run_training(r, plan)
        else:
            result = _build_and_train(r, plan)
            evidence = write_evidence(r, plan, result)
    except (TaskExecutionError, DatasetValidationError, ImportError) as exc:
        ap.exit(1, f"执行失败：{exc}\n请核对 recipe lock 与任务配置；--dry-run 不需要框架。\n")
    print(f"[ok] 真实训练完成，证据：{evidence}（不自动晋级 baseline）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
