"""rapidfire-sft-v1 选中配置的 adapter-only 导出包与全新进程 HF offline 复载。

与数学 / FIM / xLAM 刻意分开：这里导出的是「validation 选出的单一 LoRA 配置」的
fp16 adapter，冻结 base 是 gpt2（非 gated）、无量化、eager attention；包内显式声明
completion-only 训练协议、prompt 只到 "Answer:"、纯 Python 无 stemmer ROUGE-L 指标
与「数值不可比于上游 rougeL」。worker 必须真实调用 rapidfire_task.run_evaluation
在 test split 上复评，而不是复制 JSON 冒充复载。

ADR-015 的「位置不是内容身份」同样适用：TrainingArguments.output_dir、
artifact_manifest.path 与 execution.lockfile 是位置；文件清单 / sha256 / 超参 /
预算 / 锁字节仍严格绑定。包内 recipe-snapshot.json 必须与训练时 r.data 逐字节一致，
锁用虚拟锚点目录自包含；导出容差未在 recipe 预声明时 --full 失败关闭。
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from . import baseline_evidence as be
from . import export_task as et
from . import math_task as mt
from . import rapidfire_suite as rs
from . import rapidfire_task as rt
from .model import Recipe

CODE_ROOT = et.CODE_ROOT
OFFLINE_ENV, WORKER_TIMEOUT_SECONDS = et.OFFLINE_ENV, et.WORKER_TIMEOUT_SECONDS
MANIFEST, SNAPSHOT, FINAL, WORKER_EVIDENCE = (
    et.MANIFEST, et.SNAPSHOT, et.FINAL, et.WORKER_EVIDENCE)
DATA_DEPENDENCY = (
    "离线复载需要预先缓存冻结 revision 的 gpt2 base、tokenizer 与 Bitext 数据集；"
    "本包只含选中配置的 adapter，不含 base 权重，不是无数据依赖的安装镜像。"
    "Bitext 数据卡未给 license，仅 (c) Bitext Innovations, 2024，商业使用与"
    "再分发权限未确认；上游 rapidfireai 并发调度服务不在复载范围内。")
BASE_REFERENCE = "frozen-base-reference.json"
LOCK_ANCHOR = "_lock-anchor"
LICENSE = {"status": "pending", "commercial_use_confirmed": False,
           "dataset_license": "not_declared_on_data_card"}


def _need(condition, message):
    if not condition:
        raise mt.TaskExecutionError(message)


def _lock_layout(raw, bundle):
    """锁虚拟锚点：保持 recipe 内容身份逐字节不变，同时让包内锁可解析（ADR-015）。"""
    _need(isinstance(raw, str) and raw and not raw.startswith(("/", "\\")) and "\\" not in raw,
          "recipe snapshot 缺少本地 lockfile 相对路径")
    parts = raw.split("/")
    depth = 0
    while depth < len(parts) and parts[depth] == "..":
        depth += 1
    tail = parts[depth:]
    _need(tail and all(part not in ("", ".", "..") for part in tail),
          "lockfile 含非法相对路径片段")
    lock_relative = "/".join(tail)
    reserved = {SNAPSHOT, FINAL, WORKER_EVIDENCE, MANIFEST, BASE_REFERENCE, "recipe.toml"}
    _need(lock_relative not in reserved
          and not lock_relative.startswith(("model/", "reference/", "reload/", LOCK_ANCHOR + "/"))
          and not any(name.startswith(lock_relative + "/") for name in ("model", "reference", "reload")),
          "lockfile 与包内证据路径冲突")
    anchor = bundle.joinpath(*([LOCK_ANCHOR] * depth)) if depth else bundle
    recipe_path = anchor / "recipe.toml"
    lock_path = et._local(bundle, lock_relative)
    _need(Path(os.path.abspath(anchor / raw)) == lock_path
          and Path(os.path.abspath(recipe_path.parent / raw)) == lock_path,
          "锁重定位解析越界")
    return lock_relative, lock_path, recipe_path


def effective_recipe(r, checkpoint):
    """按 checkpoint 的 config_id 精确选择配置，再只覆盖训练 seed。

    RapidFire 是 4 配置网格：不能让默认 config_id 的 recipe 冒充另一个配置的 adapter。
    """
    source = et._no_symlinks(Path(checkpoint))
    training = et._json(et._local(source, "training.json"))
    config_id = training.get("config_id")
    ids = [item["id"] for item in rt.config_grid(r)]
    _need(isinstance(config_id, str) and config_id in ids,
          "checkpoint 缺少合法 config_id；拒绝用默认配置冒充选中配置")
    selected = rt.config_recipe(rs._identity_recipe(r), config_id)
    effective = et.effective_recipe(selected, source)
    _need(effective.get("execution.config_id") == config_id
          and effective.get("train.seed") == training.get("seed"),
          "导出有效 recipe 的配置 / seed 与 checkpoint 不一致")
    rt.require_supported(effective)
    return effective


def _training(r, checkpoint, hashes, manifest):
    return rs._training(r, checkpoint, hashes, manifest)


def _evaluation(r, path, artifact, manifest, *, recipe_sha=None, hashes=None):
    data = et._json(path)
    _need(isinstance(data, dict) and data.get("kind") == "evaluation"
          and data.get("stage") == "completed" and data.get("mode") == "eval"
          and data.get("evaluation_protocol") == rt.EVAL_PROTOCOL
          and data.get("evaluation_split") == rs.ACCEPTANCE_SPLIT
          and data.get("eval_stage") == rt.EVAL_STAGES[rs.ACCEPTANCE_SPLIT]
          and data.get("model_role") == "checkpoint"
          and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "参考评估不是已完成的 RapidFire 选中配置 test 评估")
    _need(bool(be.HEX64.fullmatch(str(data.get("recipe_sha256", "")))), "评估缺少 recipe_sha256")
    if recipe_sha is not None:
        _need(data.get("recipe_sha256") == recipe_sha, "评估 recipe snapshot hash 不一致")
    model = data.get("model_artifact")
    _need(isinstance(model, dict) and rt.same_artifact(model, artifact),
          "评估 model artifact 与 adapter checkpoint 不一致")
    rs._identity(r, data, hashes or rs._hashes(r), split_name=rs.ACCEPTANCE_SPLIT)
    source = data.get("split_manifest")
    rs._manifest(r, source)
    _need(source == manifest, "参考评估 split_manifest 与训练不一致")
    predictions, prediction_path, _, _ = rs._evaluation_content(
        r, path, data, manifest, rs.ACCEPTANCE_SPLIT)
    return data, predictions, prediction_path


def _same_protocol(reference, actual, reference_predictions, actual_predictions):
    """参考评估与离线复载必须共享同一输入 / gold / 解码协议；生成文本可不同。"""
    for key in ("recipe_id", "fingerprint", "task_id", "seed", "data_seed", "model_id",
                "model_revision", "dataset_id", "dataset_revision", "source_sha",
                "split_manifest", "sample_count", "evaluation_protocol",
                "evaluation_split", "eval_stage", "code_sha256",
                "environment_lock_sha256", "evaluation_manifest",
                "tokenizer_identity", "prompt_protocol", "config_id"):
        _need(actual.get(key) == reference.get(key), f"复载输入或协议不一致：{key}")
    # config.checkpoint / config.model 是位置，不参与严格相等；由 artifact_identity 单独绑定。
    for key in ("generation", "quantization", "compute_dtype", "attention_implementation",
                "evaluation_split", "eval_stage", "selection_policy", "dataset_revision",
                "source_manifest_sha256", "eval_manifest_sha256", "max_new_tokens",
                "do_sample", "num_beams", "held_out_rows", "metric_definition",
                "metric_comparable_to_upstream_rougeL", "concurrent_scheduling",
                "upstream_quantitative_evaluation", "zero_shot", "task_id", "protocol",
                "config_id"):
        _need(actual["config"].get(key) == reference["config"].get(key), f"复载 config 不一致：{key}")
    _need(rt.artifact_identity(actual.get("model_artifact"))
          == rt.artifact_identity(reference.get("model_artifact"))
          and rt.artifact_identity(actual["config"].get("model"))
          == rt.artifact_identity(reference["config"].get("model")),
          "复载 adapter 内容身份不一致")
    keys = ("sample_id", "source_index", "content_sha256", "question_sha256",
            "gold_sha256", "prompt_sha256", "eos_token_id", "max_new_tokens")
    _need(len(reference_predictions) == len(actual_predictions), "复载逐题数量不一致")
    for before, after in zip(reference_predictions, actual_predictions):
        _need(all(after.get(k) == before.get(k) for k in keys),
              "复载逐题 prompt / gold / source 输入不一致")


def _base_reference(r, identity):
    return {"schema_version": 1, "kind": "frozen-base-reference",
            "task_id": rt.TASK_ID, "model_id": rt.MODEL,
            "revision": r.get("model.revision"),
            "quantization": "none", "compute_dtype": "float16",
            "attention_implementation": "eager", "trust_remote_code": False,
            "tokenizer_identity_sha256": identity["sha256"],
            "config_id": r.get("execution.config_id"),
            "prompt_format": r.get("rapidfire.prompt_format"),
            "target_format": r.get("rapidfire.target_format"),
            "loss": r.get("rapidfire.loss"),
            "metric_definition": r.get("rapidfire.metric_definition"),
            "selection_policy": r.get("rapidfire.selection_policy"),
            "protocol": rt.EVAL_PROTOCOL, "variant": rt.EVAL_VARIANT,
            "evaluation_split": rs.ACCEPTANCE_SPLIT,
            "metric_comparable_to_upstream_rougeL": False,
            "concurrent_scheduling": False,
            "upstream_quantitative_evaluation": False,
            "generated_code_executed": False,
            "base_weights_bundled": False, "adapter_only": True,
            "data_dependency": DATA_DEPENDENCY, "license": dict(LICENSE)}


def _verify_bundle(bundle, *, expected_manifest_sha=None):
    bundle = et._no_symlinks(bundle)
    _need(bundle.is_dir(), "bundle 目录不存在")
    manifest_path = et._local(bundle, MANIFEST)
    if expected_manifest_sha is not None:
        _need(et._hash(manifest_path) == expected_manifest_sha, "bundle manifest 被篡改")
    manifest = et._json(manifest_path)
    _need(isinstance(manifest, dict) and manifest.get("schema_version") == 1
          and manifest.get("kind") == "export-bundle" and manifest.get("task_id") == rt.TASK_ID
          and manifest.get("stage") == "prepared"
          and manifest.get("sha256") == mt.sha256_json({k: v for k, v in manifest.items() if k != "sha256"}),
          "bundle manifest 结构或 hash 非法")
    records = manifest.get("files")
    _need(isinstance(records, dict) and bool(records), "bundle 未绑定文件")
    for relative, record in records.items():
        path = et._local(bundle, relative)
        _need(isinstance(record, dict) and record.get("path") == relative and path.is_file()
              and record.get("bytes") == path.stat().st_size and et._hash(path) == record.get("sha256"),
              f"bundle 文件 hash 不一致：{relative}")
    for path in et._tree_files(bundle):
        relative = path.relative_to(bundle).as_posix()
        _need(relative in records or relative in (MANIFEST, WORKER_EVIDENCE, FINAL)
              or relative.startswith("reload/eval/"), f"bundle 包含未绑定文件：{relative}")
    snapshot = et._json(et._local(bundle, SNAPSHOT))
    _need(isinstance(snapshot, dict) and snapshot.get("id") == manifest.get("recipe_id"), "Recipe snapshot 非法")
    lock_relative, lock_path, recipe_path = _lock_layout(
        (snapshot.get("execution") or {}).get("lockfile"), bundle)
    r = Recipe(manifest["recipe_id"], recipe_path, snapshot)
    stub = tomllib.loads((bundle / "recipe.toml").read_text(encoding="utf-8"))
    _need(lock_relative == manifest.get("lock_path")
          and mt.sha256_json(snapshot) == manifest.get("source_recipe_sha256")
          and be.fingerprint(r) == manifest.get("fingerprint")
          and stub.get("id") == r.id
          and (stub.get("execution") or {}).get("lockfile") == r.get("execution.lockfile"),
          "snapshot 内容身份 / 锁路径 / stub / fingerprint 不一致")
    rt.require_supported(r)
    _need(r.get("execution.config_id") == manifest.get("config_id"),
          "导出包配置与快照不一致")
    _need(et._tolerance(r) == manifest.get("tolerance"), "bundle tolerance 被替换")
    checkpoint = rt.checkpoint_path({}, et._local(bundle, "model"), recipe=r)
    hashes = rs._hashes(r)
    # _training 需要 manifest；先从包内训练证据读取再交叉校验。
    training_json = et._json(et._local(checkpoint, "training.json"))
    source = training_json.get("split_manifest")
    rs._manifest(r, source)
    training, artifact, _ = _training(r, checkpoint, hashes, source)
    reference_path = et._local(bundle, "reference/evaluation.json")
    reference, predictions, _ = _evaluation(
        r, reference_path, artifact, source,
        recipe_sha=manifest.get("reference_recipe_sha256"), hashes=hashes)
    base = et._json(et._local(bundle, BASE_REFERENCE))
    _need(base == _base_reference(r, reference["tokenizer_identity"]),
          "冻结 base 引用与 RapidFire completion-only 协议不一致")
    _need(training.get("split_manifest") == reference["split_manifest"],
          "训练与评估内容 hash 划分不一致")
    et._input_hashes(r, training, reference, lock_path)
    _need(manifest.get("model_artifact_sha256") == artifact["sha256"], "bundle adapter 身份不一致")
    required = {SNAPSHOT, "recipe.toml", lock_relative, BASE_REFERENCE,
                "model/training.json", "reference/evaluation.json",
                "reference/source_manifest.json",
                "reference/" + rt.TOKENIZER_IDENTITY_FILE,
                "reference/" + rt.EVAL_MANIFEST_FILE,
                "reference/" + reference["predictions_file"]["path"]}
    required.update("model/" + name for name in artifact["files"])
    _need(set(records) == required, "bundle 文件清单不完整或含额外输入")
    return r, manifest, reference, predictions


def run_worker(bundle):
    """内部固定 worker；真实评估委派给延迟导入框架的 rapidfire_task。"""
    _need(all(os.environ.get(k) == v for k, v in OFFLINE_ENV.items()), "worker 拒绝非 offline / 非确定性环境")
    bundle = Path(bundle)
    _need(bundle.is_absolute(), "worker --bundle 必须为 ABS 路径")
    r, manifest, reference, predictions = _verify_bundle(bundle)
    _need(not (bundle / "reload").exists() and not (bundle / WORKER_EVIDENCE).exists()
          and not (bundle / FINAL).exists(), "worker 输出已存在，拒绝重用证据")
    manifest_sha = et._hash(bundle / MANIFEST)
    from .plan import build_plan
    plan = build_plan(r, mode="eval")
    plan["evaluation_split"] = rs.ACCEPTANCE_SPLIT
    plan["eval_stage"] = rt.EVAL_STAGES[rs.ACCEPTANCE_SPLIT]
    plan["run"]["output_dir"] = str(bundle / "reload" / "eval")
    try:
        evaluation_path = rt.run_evaluation(
            r, plan, checkpoint=bundle / "model", split_name=rs.ACCEPTANCE_SPLIT)
    except Exception as exc:
        raise mt.TaskExecutionError(f"RapidFire 离线真实评估失败：{exc}。{DATA_DEPENDENCY}") from exc
    _need(Path(evaluation_path) == bundle / "reload" / "eval" / "evaluation.json",
          "worker 评估输出路径不一致")
    _verify_bundle(bundle, expected_manifest_sha=manifest_sha)
    actual_path = et._local(bundle, "reload/eval/evaluation.json")
    artifact = rt.artifact_manifest(bundle / "model")
    actual, actual_predictions, _ = _evaluation(
        r, actual_path, artifact, reference["split_manifest"],
        recipe_sha=mt.sha256_json(r.data), hashes=rs._hashes(r))
    _same_protocol(reference, actual, predictions, actual_predictions)
    for name, limit in et._tolerance(r).items():
        _need(abs(reference["metrics"][name] - actual["metrics"][name]) <= limit,
              f"复载指标 {name} 超出 export tolerance")
    evidence = {"schema_version": 1, "kind": "export-reload-worker", "stage": "completed",
                "task_id": rt.TASK_ID, "config_id": r.get("execution.config_id"),
                "pid": os.getpid(), "ppid": os.getppid(),
                "runtime": actual["runtime"], "bundle_manifest_sha256": manifest_sha,
                "fingerprint": be.fingerprint(r), "evaluation_split": rs.ACCEPTANCE_SPLIT,
                "evaluation": et._record(Path(evaluation_path), bundle),
                "offline_environment": dict(OFFLINE_ENV)}
    return mt.save_json(bundle / WORKER_EVIDENCE, evidence)


def validate_export_evidence(r, path):
    """C405：审核布尔值不能替代 adapter / 逐题重算 / 新进程复载内容检查。"""
    path = et._no_symlinks(Path(path))
    _need(path.name == FINAL, "导出证据必须为包内 export.json")
    bundle = path.parent
    data = et._json(path)
    _need(isinstance(data, dict) and data.get("schema_version") == 1
          and data.get("kind") == "export-reload" and data.get("stage") == "completed"
          and data.get("recipe_id") == r.id and data.get("task_id") == rt.TASK_ID
          and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "导出完成身份非法")
    effective = effective_recipe(r, bundle / "model")
    copied, manifest, reference, predictions = _verify_bundle(bundle)
    _need(be.fingerprint(effective) == be.fingerprint(copied) == data.get("fingerprint"),
          "导出与当前有效 recipe 不一致")
    _need(et._tolerance(effective) == manifest["tolerance"], "当前导出容差与包内协议不一致")
    original_lock = et._no_symlinks(r.path.parent / str(r.get("execution.lockfile")))
    _need(original_lock.read_bytes() == et._local(bundle, manifest.get("lock_path")).read_bytes(),
          "导出锁与当前 lock 不同")
    files = data.get("files")
    _need(isinstance(files, dict), "导出缺少完整文件绑定")
    actual_files = {p.relative_to(bundle).as_posix() for p in et._tree_files(bundle) if p != path}
    _need(set(files) == actual_files, "导出文件清单缺失或出现未绑定文件")
    for relative, record in files.items():
        file = et._local(bundle, relative)
        _need(record == et._record(file, bundle), f"导出文件被修改：{relative}")
    for key, relative in (("bundle_manifest", MANIFEST), ("worker_evidence", WORKER_EVIDENCE),
                          ("reference_evaluation", "reference/evaluation.json"),
                          ("reload_evaluation", "reload/eval/evaluation.json")):
        _need(data.get(key) == files[relative], f"导出关联记录不一致：{key}")
    worker = et._json(bundle / WORKER_EVIDENCE)
    process = data.get("subprocess", {})
    _need(isinstance(process, dict) and type(process.get("pid")) is int and process["pid"] > 0
          and type(process.get("parent_pid")) is int and process["parent_pid"] > 0
          and process["pid"] != process["parent_pid"] and process.get("returncode") == 0
          and process.get("offline_environment") == OFFLINE_ENV, "导出新进程身份不一致")
    _need(isinstance(worker, dict) and worker.get("kind") == "export-reload-worker"
          and worker.get("task_id") == rt.TASK_ID and worker.get("stage") == "completed"
          and worker.get("pid") == process["pid"] and worker.get("ppid") == process["parent_pid"]
          and worker.get("offline_environment") == OFFLINE_ENV
          and worker.get("evaluation_split") == rs.ACCEPTANCE_SPLIT
          and worker.get("config_id") == copied.get("execution.config_id")
          and worker.get("fingerprint") == data["fingerprint"]
          and worker.get("bundle_manifest_sha256") == et._hash(bundle / MANIFEST),
          "导出 worker 身份不一致")
    actual_path = bundle / "reload/eval/evaluation.json"
    _need(worker.get("evaluation") == files["reload/eval/evaluation.json"], "worker 关联评估不同")
    artifact = rt.artifact_manifest(bundle / "model")
    actual, actual_predictions, _ = _evaluation(
        copied, actual_path, artifact, reference["split_manifest"],
        recipe_sha=mt.sha256_json(copied.data), hashes=rs._hashes(copied))
    _same_protocol(reference, actual, predictions, actual_predictions)
    _need(worker.get("runtime") == actual["runtime"] == data.get("runtime")
          and data.get("metrics") == actual["metrics"]
          and data.get("code_sha256") == actual["code_sha256"]
          and data.get("environment_lock_sha256") == actual["environment_lock_sha256"]
          and data.get("split_sha256") == actual["split_manifest"]["sha256"]
          and data.get("model_artifact_sha256") == artifact["sha256"]
          and data.get("seed") == effective.get("train.seed")
          and data.get("config_id") == effective.get("execution.config_id")
          and data.get("evaluation_split") == rs.ACCEPTANCE_SPLIT
          and data.get("metric_comparable_to_upstream_rougeL") is False
          and data.get("concurrent_scheduling") is False
          and data.get("license") == LICENSE,
          "导出摘要与复载实际评估不一致")
    comparisons = {}
    for name, limit in et._tolerance(effective).items():
        delta = abs(reference["metrics"][name] - actual["metrics"][name])
        _need(delta <= limit, "复载指标超出当前容差")
        comparisons[name] = {"reference": reference["metrics"][name], "reload": actual["metrics"][name],
                             "absolute_delta": delta, "tolerance": limit, "within_tolerance": True}
    _need(data.get("comparison") == comparisons, "导出容差摘要未重算或已改变")
    return True


def export_bundle(r, checkpoint, evaluation_path, destination, execute=True):
    if not execute:
        try:
            rt.require_supported(r)
            supported, reason = True, None
        except mt.TaskExecutionError as exc:
            supported, reason = False, str(exc)
        return {"schema_version": 1, "kind": "export-plan", "stage": "declared", "recipe_id": r.id,
                "task_id": rt.TASK_ID, "execution_supported": supported, "unsupported_reason": reason,
                "checkpoint": str(checkpoint), "evaluation": str(evaluation_path),
                "destination": str(destination),
                "config_id": r.get("execution.config_id"),
                "config_id_source": "execute 时读取 checkpoint training.config_id 精确派生，拒绝默认配置冒充",
                "train_seed": r.get("train.seed"),
                "effective_seed_source": "execute 时读取 checkpoint training.seed；只覆盖 train.seed",
                "data_seed": r.get("train.data_seed"), "evaluation_seed": r.get("evaluation.seed"),
                "protocol": rt.EVAL_PROTOCOL, "evaluation_split": rs.ACCEPTANCE_SPLIT,
                "metrics": list(rt.METRICS),
                "tolerance": (r.get("evaluation.export_tolerance")
                              if r.has("evaluation.export_tolerance") else r.get("evaluation.tolerance")),
                # 没有基于实跑预声明的容差表时，--full 必须失败关闭；null 不表示「无需容差」。
                "tolerance_status": ("declared" if r.has("evaluation.export_tolerance")
                                     or r.has("evaluation.tolerance") else
                                     "not_declared：--full 会因缺少预声明容差失败关闭，不自动回填 expected/tolerance"),
                "offline_environment": dict(OFFLINE_ENV), "data_dependency": DATA_DEPENDENCY,
                "base_weights_bundled": False, "adapter_only": True,
                "quantization": "none", "compute_dtype": "float16",
                "metric_comparable_to_upstream_rougeL": False,
                "concurrent_scheduling": False, "generated_code_executed": False,
                "upstream_quantitative_evaluation": False,
                "license": dict(LICENSE), "baseline_promotion": "none"}
    rt.require_supported(r)
    source = et._no_symlinks(Path(checkpoint))
    output = et._no_symlinks(Path(destination))
    _need(not output.is_relative_to(source) and not source.is_relative_to(output),
          "destination 不能与 checkpoint 路径内嵌或重叠")
    _need(not output.exists() or (output.is_dir() and not any(output.iterdir())),
          "拒绝 nonempty destination 或非目录输出")
    evaluation_path = et._no_symlinks(Path(evaluation_path))
    _need(evaluation_path.is_file(), "reference evaluation 不存在")
    r = effective_recipe(r, source)
    rt.require_supported(r)
    rt.fixed_seeds(r)
    tolerance = et._tolerance(r)
    source = rt.checkpoint_path({}, source, recipe=r)
    hashes = rs._hashes(r)
    training_json = et._json(et._local(source, "training.json"))
    manifest = training_json.get("split_manifest")
    rs._manifest(r, manifest)
    training, artifact, _ = _training(r, source, hashes, manifest)
    reference, _, prediction_path = _evaluation(
        r, evaluation_path, artifact, manifest, hashes=hashes)
    _need(training.get("split_manifest") == reference["split_manifest"], "训练与评估内容 hash 划分不一致")
    lock = et._no_symlinks(r.path.parent / str(r.get("execution.lockfile")))
    et._input_hashes(r, training, reference, lock)
    lock_relative, _, recipe_path = _lock_layout(r.get("execution.lockfile"), output)
    copy_sources = {"model/" + name: et._local(source, name) for name in artifact["files"]}
    copy_sources.update({"model/training.json": et._local(source, "training.json"),
                         lock_relative: lock,
                         "reference/evaluation.json": evaluation_path,
                         "reference/source_manifest.json": et._local(
                             evaluation_path.parent, "source_manifest.json"),
                         "reference/" + rt.TOKENIZER_IDENTITY_FILE: et._local(
                             evaluation_path.parent, rt.TOKENIZER_IDENTITY_FILE),
                         "reference/" + rt.EVAL_MANIFEST_FILE: et._local(
                             evaluation_path.parent, rt.EVAL_MANIFEST_FILE)})
    prediction_name = reference["predictions_file"]["path"]
    _need(prediction_name not in ("evaluation.json", "source_manifest.json",
                                  rt.TOKENIZER_IDENTITY_FILE, rt.EVAL_MANIFEST_FILE),
          "预测路径与证据路径冲突")
    copy_sources["reference/" + prediction_name] = et._local(evaluation_path.parent, prediction_name)
    expected_sources = {relative: et._record(path, path.parent) for relative, path in copy_sources.items()}
    # 快照必须与训练时 effective_recipe.json / recipe_sha256 逐字节相同；
    # 锁文件的「位置」由 _lock_layout 的虚拟锚点处理，不写进内容身份。
    snapshot = copy.deepcopy(r.data)
    snapshot_recipe = Recipe(r.id, recipe_path, snapshot)
    _need(snapshot == r.data and mt.sha256_json(snapshot) == mt.sha256_json(r.data)
          and be.fingerprint(snapshot_recipe) == be.fingerprint(r),
          "导出包不得改写 recipe 内容身份")
    output.mkdir(parents=True, exist_ok=True)
    for relative, path in copy_sources.items():
        target = et._local(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        _need(et._hash(target) == expected_sources[relative]["sha256"], f"复制期间来源被篡改：{relative}")
    mt.save_json(output / SNAPSHOT, snapshot)
    (output / "recipe.toml").write_text(
        "id = " + json.dumps(r.id) + "\n[execution]\nlockfile = "
        + json.dumps(r.get("execution.lockfile")) + "\n", encoding="utf-8")
    mt.save_json(output / BASE_REFERENCE, _base_reference(r, reference["tokenizer_identity"]))
    files = {path.relative_to(output).as_posix(): et._record(path, output) for path in et._tree_files(output)}
    bundle_manifest = {"schema_version": 1, "kind": "export-bundle", "stage": "prepared",
                       "task_id": rt.TASK_ID, "recipe_id": r.id, "fingerprint": be.fingerprint(r),
                       "config_id": r.get("execution.config_id"),
                       "source_recipe_sha256": mt.sha256_json(snapshot), "lock_path": lock_relative,
                       "reference_recipe_sha256": reference["recipe_sha256"],
                       "model_artifact_sha256": artifact["sha256"], "files": files,
                       "tolerance": tolerance, "data_dependency": DATA_DEPENDENCY,
                       "base_weights_bundled": False, "adapter_only": True,
                       "quantization": "none", "compute_dtype": "float16",
                       "metric_comparable_to_upstream_rougeL": False,
                       "concurrent_scheduling": False, "generated_code_executed": False,
                       "upstream_quantitative_evaluation": False,
                       "license": dict(LICENSE)}
    bundle_manifest["sha256"] = mt.sha256_json(bundle_manifest)
    mt.save_json(output / MANIFEST, bundle_manifest)
    manifest_sha = et._hash(output / MANIFEST)
    _verify_bundle(output, expected_manifest_sha=manifest_sha)
    process_info = et._launch_worker(output)
    reloaded_recipe, _, copied_reference, copied_predictions = _verify_bundle(
        output, expected_manifest_sha=manifest_sha)
    worker = et._json(et._local(output, WORKER_EVIDENCE))
    _need(isinstance(worker, dict) and worker.get("schema_version") == 1
          and worker.get("kind") == "export-reload-worker" and worker.get("stage") == "completed"
          and worker.get("task_id") == rt.TASK_ID
          and worker.get("config_id") == r.get("execution.config_id")
          and worker.get("evaluation_split") == rs.ACCEPTANCE_SPLIT
          and type(worker.get("pid")) is int and worker["pid"] == process_info["pid"] != os.getpid()
          and worker.get("ppid") == os.getpid() and worker.get("fingerprint") == be.fingerprint(r)
          and worker.get("bundle_manifest_sha256") == manifest_sha
          and worker.get("offline_environment") == OFFLINE_ENV,
          "worker 身份 / pid / offline / completed 证据不一致")
    record = worker.get("evaluation", {})
    _need(record.get("path") == "reload/eval/evaluation.json" and be.hashed_file(output, record),
          "worker evaluation.json hash 不一致")
    actual_path = et._local(output, record["path"])
    actual, actual_predictions, _ = _evaluation(
        reloaded_recipe, actual_path, artifact, copied_reference["split_manifest"],
        recipe_sha=mt.sha256_json(reloaded_recipe.data), hashes=rs._hashes(reloaded_recipe))
    _need(worker.get("runtime") == actual["runtime"], "worker runtime 必须来自子进程实际 evaluation")
    _same_protocol(copied_reference, actual, copied_predictions, actual_predictions)
    comparisons = {}
    for name, limit in tolerance.items():
        delta = abs(actual["metrics"][name] - copied_reference["metrics"][name])
        _need(delta <= limit, f"复载指标 {name} 超出 export tolerance：delta={delta}, tolerance={limit}")
        comparisons[name] = {"reference": copied_reference["metrics"][name], "reload": actual["metrics"][name],
                             "absolute_delta": delta, "tolerance": limit, "within_tolerance": True}
    _verify_bundle(output, expected_manifest_sha=manifest_sha)
    evidence = {"schema_version": 1, "kind": "export-reload", "stage": "completed",
                "task_id": rt.TASK_ID, "recipe_id": r.id, "fingerprint": be.fingerprint(r),
                "config_id": r.get("execution.config_id"),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "seed": r.get("train.seed"), "data_seed": r.get("train.data_seed"),
                "evaluation_seed": r.get("evaluation.seed"), "runtime": actual["runtime"],
                "subprocess": process_info, "metrics": actual["metrics"], "comparison": comparisons,
                "model_artifact_sha256": artifact["sha256"], "code_sha256": actual["code_sha256"],
                "environment_lock_sha256": actual["environment_lock_sha256"],
                "bundle_manifest": et._record(output / MANIFEST, output),
                "worker_evidence": et._record(output / WORKER_EVIDENCE, output),
                "reference_evaluation": et._record(output / "reference" / "evaluation.json", output),
                "reload_evaluation": et._record(actual_path, output),
                "split_sha256": actual["split_manifest"]["sha256"],
                "protocol": rt.EVAL_PROTOCOL, "evaluation_split": rs.ACCEPTANCE_SPLIT,
                "metric_comparable_to_upstream_rougeL": False,
                "concurrent_scheduling": False, "generated_code_executed": False,
                "upstream_quantitative_evaluation": False,
                "license": dict(LICENSE), "data_dependency": DATA_DEPENDENCY,
                "base_weights_bundled": False, "baseline_promotion": "none",
                "training_seed_variance_measured": False,
                "files": {path.relative_to(output).as_posix(): et._record(path, output)
                          for path in et._tree_files(output)}}
    _need(not (output / FINAL).exists(), "final 导出证据已存在，拒绝覆盖")
    return mt.save_json(output / FINAL, evidence)
