"""qwen25vl-mpo-v1 的 DoRA/4bit adapter-only 导出包与 HF offline 新进程复载。

与 DPO 篇刻意分开：MPO 的发布级身份是「偏好对图像 hash 三段隔离 + 显式
image_token_id 与 completion-only 双层 label mask + processor identity +
消融 config_id 与 sigmoid/bco_pair/sft 三损失权重 + bf16=False 矛盾登记 +
逐题隐式奖励 pairwise 重算」。QLoRA/DoRA checkpoint 只是 adapter：包内显式声明
冻结 base 的 revision / 量化 / attention / chat template / 三损失配置与 ref 配置
身份以及 base_weights_bundled=false；ref 模型只记录配置身份，不打包权重。worker
必须真实调用 mpo_task.run_evaluation，不复制 JSON 冒充复载。

ADR-015「位置不是内容身份」同样适用；recipe-snapshot 与训练有效配方逐字节一致，
锁用虚拟锚点自包含。三损失消融只在同一 Qwen backbone 内，不与 SmolVLM DPO 篇混用。
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
from . import mpo_suite as xs
from . import mpo_task as xt
from . import preference_common as pc
from .model import Recipe

CODE_ROOT = et.CODE_ROOT
OFFLINE_ENV, WORKER_TIMEOUT_SECONDS = et.OFFLINE_ENV, et.WORKER_TIMEOUT_SECONDS
MANIFEST, SNAPSHOT, FINAL, WORKER_EVIDENCE = (
    et.MANIFEST, et.SNAPSHOT, et.FINAL, et.WORKER_EVIDENCE)
DATA_DEPENDENCY = ("离线复载需要预先缓存冻结 revision 的 Qwen/Qwen2.5-VL-3B-Instruct "
                   "base/processor 与 HuggingFaceH4/rlaif-v_formatted 图像偏好数据；本包只含 "
                   "DoRA/4bit adapter，不含 base 权重与图像，不是无数据依赖的安装镜像；"
                   "Qwen / RLAIF-V 的许可与商业使用、再分发权限未确认。")
BASE_REFERENCE = "frozen-base-reference.json"
LICENSE = {"status": "pending", "commercial_use_confirmed": False}
PROTOCOL_KEYS = (
    "recipe_id", "fingerprint", "task_id", "seed", "data_seed", "model_id",
    "model_revision", "dataset_id", "dataset_revision", "source_sha",
    "split_manifest", "sample_count", "evaluation_protocol", "config_id",
    "code_sha256", "environment_lock_sha256", "evaluation_manifest",
    "processor_identity", "prompt_protocol",
)
CONFIG_KEYS = xs.CONFIG_KEYS
PREDICTION_IDENTITY_KEYS = xs.PREDICTION_IDENTITY_KEYS


def _need(condition, message):
    if not condition:
        raise mt.TaskExecutionError(message)


def _lock_layout(raw, bundle):
    """虚拟锁锚点：锁是位置不是内容身份，不改写 recipe 快照（ADR-015/022）。"""
    return pc.lock_layout(raw, bundle)


def _checkpoint_recipe(r, checkpoint):
    """从 checkpoint 读取 seed 与 config_id，只覆盖这两项；其余配方保持原样。"""
    training = et._json(et._local(et._no_symlinks(Path(checkpoint)), "training.json"))
    config_id = training.get("config_id", r.get("execution.config_id", "full"))
    _need(config_id in xt.ABLATION_CONFIGS, "checkpoint config_id 非法或未知")
    effective = et.effective_recipe(r, checkpoint)
    return xt.config_recipe(effective, config_id)


def effective_recipe(r, checkpoint):
    return _checkpoint_recipe(r, checkpoint)


def _identity(r, data, hashes=None):
    hashes = hashes or xs._hashes(r)
    xs._identity(r, data, hashes)


def _training(r, checkpoint):
    clean = xs.identity_recipe(r)
    xt.checkpoint_path({}, checkpoint, recipe=clean)
    training = et._json(et._local(checkpoint, "training.json"))
    _identity(r, training)
    _need(training.get("kind") == "training" and training.get("mode") == "full"
          and training.get("trainer") == "Trainer"
          and not any(training.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "仅允许 full 且非 mock 的 completed MPO 训练证据")
    _need(be.frozen_environment(r) and be.runtime_consistent(r, training),
          "训练环境不一致或不是 frozen lock")
    config = training.get("config")
    expected = xt.config_kwargs(clean, {"recipe_id": clean.id, "task_id": xt.TASK_ID,
                                        "mode": "full"}, checkpoint)
    expected_identity = xt.training_arguments_identity(expected)
    _need(isinstance(config, dict) and training.get("config_sha256") == xt.sha256_json(config)
          and xt.training_arguments_identity(config.get("training_arguments")) == expected_identity
          and xt.training_arguments_identity(config.get("api_training_arguments")) in (
              expected_identity, xt.training_arguments_identity({
                  ("eval_strategy" if k == "evaluation_strategy" else k): v for k, v in expected.items()}))
          and config.get("preference") == clean.get("preference")
          and config.get("peft") == clean.get("peft")
          and config.get("quantization") == clean.get("quantization")
          and config.get("attention_implementation") == "eager"
          and config.get("upstream_metadata") == dict(clean.get("upstream"))
          and config.get("lora_execution", {}).get("tensor_reload_check", {}).get(
              "strict_values_verified") is True
          and config.get("processor_identity") == training.get("processor_identity")
          and type(training.get("steps")) is int
          and training["steps"] == training.get("max_steps") == expected["max_steps"] == xs.FULL_STEPS,
          "训练有效配置 / 三损失 / 双层 mask 协议 / full=200 预算不一致")
    xs._loss_binding(config, training, clean)
    artifact = pc.artifact_manifest(checkpoint)
    _need(pc.same_artifact(training.get("model_artifact"), artifact), "训练 artifact 文件或 hash 不一致")
    return training, artifact


def _evaluation(r, path, artifact, *, recipe_sha=None):
    data = et._json(path)
    _need(isinstance(data, dict) and data.get("kind") == "evaluation"
          and data.get("stage") == "completed" and data.get("mode") == "eval"
          and data.get("evaluation_protocol") == xt.EVAL_PROTOCOL
          and data.get("config_id") == r.get("execution.config_id", "full")
          and data.get("model_role") == "checkpoint"
          and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "参考评估不是已完成的 MPO 训练模型 held-out 隐式奖励评估")
    _need(bool(be.HEX64.fullmatch(str(data.get("recipe_sha256", "")))), "评估缺少 recipe_sha256")
    if recipe_sha is not None:
        _need(data.get("recipe_sha256") == recipe_sha, "评估 recipe snapshot hash 不一致")
    model = data.get("model_artifact")
    _need(isinstance(model, dict) and pc.same_artifact(model, artifact),
          "评估 model artifact 与 adapter checkpoint 不一致")
    _identity(r, data)
    manifest = data.get("split_manifest")
    xs._manifest(r, manifest)
    predictions, prediction_path, _, _ = xs._evaluation_content(r, path, data, manifest)
    return data, predictions, prediction_path


def _same_protocol(reference, actual, reference_predictions, actual_predictions):
    """参考评估与离线复载必须共享输入 / gold / 解码 / 消融协议；奖励值可不同。"""
    for key in PROTOCOL_KEYS:
        _need(actual.get(key) == reference.get(key), f"复载输入或协议不一致：{key}")
    for key in CONFIG_KEYS:
        _need(actual.get("config", {}).get(key) == reference.get("config", {}).get(key),
              f"复载 config 不一致：{key}")
    _need(pc.artifact_identity(actual.get("model_artifact"))
          == pc.artifact_identity(reference.get("model_artifact"))
          and pc.artifact_identity(actual["config"].get("model"))
          == pc.artifact_identity(reference["config"].get("model")),
          "复载 adapter 内容身份不一致")
    _need(len(reference_predictions) == len(actual_predictions), "复载逐题数量不一致")
    for before, after in zip(reference_predictions, actual_predictions):
        _need(all(after.get(k) == before.get(k) for k in PREDICTION_IDENTITY_KEYS),
              "复载逐题 prompt / gold / 图像输入或 config_id 不一致")


def _base_reference(r, identity):
    config_id = r.get("execution.config_id", "full")
    spec = xt.verify_loss_config(config_id, r.get("train.loss_types"), r.get("train.loss_weights"))
    return {"schema_version": 1, "kind": "frozen-base-reference",
            "task_id": xt.TASK_ID, "model_id": xt.MODEL,
            "revision": r.get("model.revision"),
            "quantization": r.get("quantization"), "compute_dtype": "bfloat16",
            "attention_implementation": "eager", "trust_remote_code": False,
            "processor_identity_sha256": identity["sha256"],
            "image_token_id": identity["image_token_id"],
            "chat_template_sha256": identity["chat_template_sha256"],
            "prompt_format": r.get("preference.prompt_format"),
            "config_id": config_id, "loss_types": spec["loss_types"],
            "loss_weights": spec["weights"],
            "bf16_training": False, "precision_conflict": r.get("preference.precision_conflict"),
            "loss": r.get("preference.loss"),
            "reward_source": r.get("preference.metric_definition"),
            "ref_binding": {"policy_ref_disjoint": True, "ref_trainable_parameters": 0},
            "protocol": xt.EVAL_PROTOCOL, "variant": xt.EVAL_VARIANT,
            "generated_code_executed": False,
            "upstream_quantitative_evaluation": False,
            "base_weights_bundled": False, "adapter_only": True,
            "ablation_is_within_backbone": True, "cross_backbone_attribution_valid": False,
            "data_dependency": DATA_DEPENDENCY, "license": dict(LICENSE)}


def _verify_bundle(bundle, *, expected_manifest_sha=None):
    bundle = et._no_symlinks(bundle)
    _need(bundle.is_dir(), "bundle 目录不存在")
    manifest_path = et._local(bundle, MANIFEST)
    if expected_manifest_sha is not None:
        _need(et._hash(manifest_path) == expected_manifest_sha, "bundle manifest 被篡改")
    manifest = et._json(manifest_path)
    _need(isinstance(manifest, dict) and manifest.get("schema_version") == 1
          and manifest.get("kind") == "export-bundle" and manifest.get("task_id") == xt.TASK_ID
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
    dt = xt
    dt.require_supported(r)
    _need(et._tolerance(r) == manifest.get("tolerance"), "bundle tolerance 被替换")
    checkpoint = dt.checkpoint_path({}, et._local(bundle, "model"), recipe=r)
    training, artifact = _training(r, checkpoint)
    reference_path = et._local(bundle, "reference/evaluation.json")
    reference, predictions, _ = _evaluation(
        r, reference_path, artifact, recipe_sha=manifest.get("reference_recipe_sha256"))
    base = et._json(et._local(bundle, BASE_REFERENCE))
    _need(base == _base_reference(r, reference["processor_identity"]),
          "冻结 base 引用与消融配置/bf16/双层 mask 协议不一致")
    _need(training.get("split_manifest") == reference["split_manifest"], "训练与评估图像 hash 划分不一致")
    et._input_hashes(r, training, reference, lock_path)
    _need(manifest.get("model_artifact_sha256") == artifact["sha256"], "bundle adapter 身份不一致")
    required = {SNAPSHOT, "recipe.toml", lock_relative, BASE_REFERENCE,
                "model/training.json", "reference/evaluation.json",
                "reference/source_manifest.json",
                "reference/" + xt.PROCESSOR_IDENTITY_FILE,
                "reference/" + xt.EVAL_MANIFEST_FILE,
                "reference/" + reference["predictions_file"]["path"]}
    required.update("model/" + name for name in artifact["files"])
    _need(set(records) == required, "bundle 文件清单不完整或含额外输入")
    return r, manifest, reference, predictions


def run_worker(bundle):
    """内部固定 worker；真实评估委派给延迟导入框架的 mpo_task。"""
    _need(all(os.environ.get(k) == v for k, v in OFFLINE_ENV.items()), "worker 拒绝非 offline / 非确定性环境")
    bundle = Path(bundle)
    _need(bundle.is_absolute(), "worker --bundle 必须为 ABS 路径")
    r, manifest, reference, predictions = _verify_bundle(bundle)
    _need(not (bundle / "reload").exists() and not (bundle / WORKER_EVIDENCE).exists()
          and not (bundle / FINAL).exists(), "worker 输出已存在，拒绝重用证据")
    manifest_sha = et._hash(bundle / MANIFEST)
    from .plan import build_plan
    plan = build_plan(r, mode="eval")
    plan["run"]["output_dir"] = str(bundle / "reload" / "eval")
    try:
        evaluation_path = xt.run_evaluation(r, plan, checkpoint=bundle / "model")
    except Exception as exc:
        raise mt.TaskExecutionError(f"MPO 离线真实评估失败：{exc}。{DATA_DEPENDENCY}") from exc
    _need(Path(evaluation_path) == bundle / "reload" / "eval" / "evaluation.json", "worker 评估输出路径不一致")
    _verify_bundle(bundle, expected_manifest_sha=manifest_sha)
    actual_path = et._local(bundle, "reload/eval/evaluation.json")
    artifact = pc.artifact_manifest(bundle / "model")
    actual, actual_predictions, _ = _evaluation(
        r, actual_path, artifact, recipe_sha=mt.sha256_json(r.data))
    _same_protocol(reference, actual, predictions, actual_predictions)
    for name, limit in et._tolerance(r).items():
        _need(abs(reference["metrics"][name] - actual["metrics"][name]) <= limit,
              f"复载指标 {name} 超出 export tolerance")
    evidence = {"schema_version": 1, "kind": "export-reload-worker", "stage": "completed",
                "task_id": xt.TASK_ID, "pid": os.getpid(), "ppid": os.getppid(),
                "runtime": actual["runtime"], "bundle_manifest_sha256": manifest_sha,
                "fingerprint": be.fingerprint(r),
                "evaluation": et._record(Path(evaluation_path), bundle),
                "offline_environment": dict(OFFLINE_ENV)}
    return mt.save_json(bundle / WORKER_EVIDENCE, evidence)


def validate_export_evidence(r, path):
    """C405 的 MPO 内容复核；审核布尔值不能替代 adapter / 逐题重算 / 复载内容检查。"""
    path = et._no_symlinks(Path(path))
    _need(path.name == FINAL, "导出证据必须为包内 export.json")
    bundle = path.parent
    data = et._json(path)
    _need(isinstance(data, dict) and data.get("schema_version") == 1
          and data.get("kind") == "export-reload" and data.get("stage") == "completed"
          and data.get("recipe_id") == r.id and data.get("task_id") == xt.TASK_ID
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
          and worker.get("task_id") == xt.TASK_ID and worker.get("stage") == "completed"
          and worker.get("pid") == process["pid"] and worker.get("ppid") == process["parent_pid"]
          and worker.get("offline_environment") == OFFLINE_ENV
          and worker.get("fingerprint") == data["fingerprint"]
          and worker.get("bundle_manifest_sha256") == et._hash(bundle / MANIFEST),
          "导出 worker 身份不一致")
    actual_path = bundle / "reload/eval/evaluation.json"
    _need(worker.get("evaluation") == files["reload/eval/evaluation.json"], "worker 关联评估不同")
    artifact = pc.artifact_manifest(bundle / "model")
    actual, actual_predictions, _ = _evaluation(copied, actual_path, artifact,
                                                recipe_sha=mt.sha256_json(copied.data))
    _same_protocol(reference, actual, predictions, actual_predictions)
    xs._manifest(copied, actual["split_manifest"])
    _need(worker.get("runtime") == actual["runtime"] == data.get("runtime")
          and data.get("metrics") == actual["metrics"]
          and data.get("code_sha256") == actual["code_sha256"]
          and data.get("environment_lock_sha256") == actual["environment_lock_sha256"]
          and data.get("split_sha256") == actual["split_manifest"]["sha256"]
          and data.get("model_artifact_sha256") == artifact["sha256"]
          and data.get("config_id") == effective.get("execution.config_id", "full")
          and data.get("seed") == effective.get("train.seed")
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
            xt.require_supported(r)
            supported, reason = True, None
        except mt.TaskExecutionError as exc:
            supported, reason = False, str(exc)
        return {"schema_version": 1, "kind": "export-plan", "stage": "declared", "recipe_id": r.id,
                "task_id": xt.TASK_ID, "execution_supported": supported, "unsupported_reason": reason,
                "checkpoint": str(checkpoint), "evaluation": str(evaluation_path),
                "destination": str(destination), "train_seed": r.get("train.seed"),
                "effective_seed_source": "execute 时读取 checkpoint training.seed/config_id；只覆盖这两项",
                "data_seed": r.get("train.data_seed"), "evaluation_seed": r.get("evaluation.seed"),
                "protocol": xt.EVAL_PROTOCOL, "metrics": list(xt.METRICS),
                "config_id": r.get("execution.config_id", "full"),
                "ablation_configs": list(xs.CONFIG_IDS),
                "tolerance": (r.get("evaluation.export_tolerance")
                              if r.has("evaluation.export_tolerance") else r.get("evaluation.tolerance")),
                "tolerance_status": ("declared" if r.has("evaluation.export_tolerance")
                                     or r.has("evaluation.tolerance") else
                                     "not_declared：--full 会因缺少预声明容差失败关闭，不自动回填 expected/tolerance"),
                "offline_environment": dict(OFFLINE_ENV), "data_dependency": DATA_DEPENDENCY,
                "base_weights_bundled": False, "adapter_only": True,
                "ref_weights_bundled": False, "generated_code_executed": False,
                "upstream_quantitative_evaluation": False,
                "bf16_training": False, "precision_conflict": r.get("preference.precision_conflict"),
                "ablation_is_within_backbone": True, "cross_backbone_attribution_valid": False,
                "vision_contract": {"processing_class": "processor",
                                    "image_mode": r.get("modality_contract.images_column"),
                                    "explicit_image_token_mask": True,
                                    "assistant_only_loss": True},
                "license": dict(LICENSE), "baseline_promotion": "none"}
    xt.require_supported(r)
    source = et._no_symlinks(Path(checkpoint))
    output = et._no_symlinks(Path(destination))
    _need(not output.is_relative_to(source) and not source.is_relative_to(output),
          "destination 不能与 checkpoint 路径内嵌或重叠")
    _need(not output.exists() or (output.is_dir() and not any(output.iterdir())),
          "拒绝 nonempty destination 或非目录输出")
    evaluation_path = et._no_symlinks(Path(evaluation_path))
    _need(evaluation_path.is_file(), "reference evaluation 不存在")
    r = effective_recipe(r, source)
    xt.require_supported(r)
    xt.fixed_seeds(r)
    tolerance = et._tolerance(r)
    source = xt.checkpoint_path({}, source, recipe=xs.identity_recipe(r))
    training, artifact = _training(r, source)
    reference, _, _ = _evaluation(r, evaluation_path, artifact)
    _need(training.get("split_manifest") == reference["split_manifest"], "训练与评估图像 hash 划分不一致")
    lock = et._no_symlinks(r.path.parent / str(r.get("execution.lockfile")))
    et._input_hashes(r, training, reference, lock)
    lock_relative, _, recipe_path = _lock_layout(r.get("execution.lockfile"), output)
    copy_sources = {"model/" + name: et._local(source, name) for name in artifact["files"]}
    copy_sources.update({"model/training.json": et._local(source, "training.json"),
                         lock_relative: lock,
                         "reference/evaluation.json": evaluation_path,
                         "reference/source_manifest.json": et._local(
                             evaluation_path.parent, "source_manifest.json"),
                         "reference/" + xt.PROCESSOR_IDENTITY_FILE: et._local(
                             evaluation_path.parent, xt.PROCESSOR_IDENTITY_FILE),
                         "reference/" + xt.EVAL_MANIFEST_FILE: et._local(
                             evaluation_path.parent, xt.EVAL_MANIFEST_FILE)})
    prediction_name = reference["predictions_file"]["path"]
    _need(prediction_name not in ("evaluation.json", "source_manifest.json",
                                  xt.PROCESSOR_IDENTITY_FILE, xt.EVAL_MANIFEST_FILE),
          "预测路径与证据路径冲突")
    copy_sources["reference/" + prediction_name] = et._local(evaluation_path.parent, prediction_name)
    expected_sources = {relative: et._record(path, path.parent) for relative, path in copy_sources.items()}
    # 快照必须与训练时 effective_recipe.json / recipe_sha256 逐字节相同：
    # 剥离实跑后回填的 expected/tolerance，保留预声明的 export_tolerance。
    snapshot = copy.deepcopy(xs.identity_recipe(r).data)
    snapshot_recipe = Recipe(r.id, recipe_path, snapshot)
    _need(snapshot == xs.identity_recipe(r).data
          and mt.sha256_json(snapshot) == mt.sha256_json(xs.identity_recipe(r).data)
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
    mt.save_json(output / BASE_REFERENCE, _base_reference(r, reference["processor_identity"]))
    files = {path.relative_to(output).as_posix(): et._record(path, output) for path in et._tree_files(output)}
    manifest = {"schema_version": 1, "kind": "export-bundle", "stage": "prepared",
                "task_id": xt.TASK_ID, "recipe_id": r.id, "fingerprint": be.fingerprint(r),
                "source_recipe_sha256": mt.sha256_json(snapshot), "lock_path": lock_relative,
                "reference_recipe_sha256": reference["recipe_sha256"],
                "model_artifact_sha256": artifact["sha256"], "files": files, "tolerance": tolerance,
                "config_id": r.get("execution.config_id", "full"), "ablation_configs": list(xs.CONFIG_IDS),
                "data_dependency": DATA_DEPENDENCY, "base_weights_bundled": False,
                "adapter_only": True, "ref_weights_bundled": False,
                "generated_code_executed": False, "upstream_quantitative_evaluation": False,
                "bf16_training": False, "ablation_is_within_backbone": True,
                "cross_backbone_attribution_valid": False, "license": dict(LICENSE)}
    manifest["sha256"] = mt.sha256_json(manifest)
    mt.save_json(output / MANIFEST, manifest)
    manifest_sha = et._hash(output / MANIFEST)
    _verify_bundle(output, expected_manifest_sha=manifest_sha)
    process_info = et._launch_worker(output)
    reloaded_recipe, _, copied_reference, copied_predictions = _verify_bundle(
        output, expected_manifest_sha=manifest_sha)
    worker = et._json(et._local(output, WORKER_EVIDENCE))
    _need(isinstance(worker, dict) and worker.get("schema_version") == 1
          and worker.get("kind") == "export-reload-worker" and worker.get("stage") == "completed"
          and worker.get("task_id") == xt.TASK_ID
          and type(worker.get("pid")) is int and worker["pid"] == process_info["pid"] != os.getpid()
          and worker.get("ppid") == os.getpid() and worker.get("fingerprint") == be.fingerprint(r)
          and worker.get("bundle_manifest_sha256") == manifest_sha
          and worker.get("offline_environment") == OFFLINE_ENV,
          "worker 身份 / pid / offline / completed 证据不一致")
    record = worker.get("evaluation", {})
    _need(record.get("path") == "reload/eval/evaluation.json" and be.hashed_file(output, record),
          "worker evaluation.json hash 不一致")
    actual_path = et._local(output, record["path"])
    actual, actual_predictions, _ = _evaluation(reloaded_recipe, actual_path, artifact,
                                                recipe_sha=mt.sha256_json(reloaded_recipe.data))
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
                "task_id": xt.TASK_ID, "recipe_id": r.id, "fingerprint": be.fingerprint(r),
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
                "config_id": r.get("execution.config_id", "full"),
                "protocol": xt.EVAL_PROTOCOL, "bf16_training": False,
                "generated_code_executed": False, "upstream_quantitative_evaluation": False,
                "base_weights_bundled": False, "ref_weights_bundled": False,
                "ablation_is_within_backbone": True, "cross_backbone_attribution_valid": False,
                "license": dict(LICENSE), "data_dependency": DATA_DEPENDENCY,
                "baseline_promotion": "none", "training_seed_variance_measured": False,
                "files": {path.relative_to(output).as_posix(): et._record(path, output)
                          for path in et._tree_files(output)}}
    _need(not (output / FINAL).exists(), "final 导出证据已存在，拒绝覆盖")
    return mt.save_json(output / FINAL, evidence)
