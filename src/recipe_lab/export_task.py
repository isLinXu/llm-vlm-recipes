"""数学 checkpoint 导出与新进程离线复载；导入和 dry plan 仅依赖标准库。"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import baseline_evidence as be
from . import math_task as mt
from .model import Recipe

CODE_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
WORKER_TIMEOUT_SECONDS = 3600
MANIFEST = "bundle-manifest.json"
SNAPSHOT = "recipe-snapshot.json"
LOCKFILE = "environment.lock.toml"
FINAL = "export.json"
WORKER_EVIDENCE = "reload-worker.json"
DATA_DEPENDENCY = "离线复载需要预先缓存锁定 revision 的数据集；本包不是无数据依赖的安装镜像。"


def _need(condition, message):
    if not condition:
        raise mt.TaskExecutionError(message)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path, base):
    return {"path": path.relative_to(base).as_posix(), "sha256": _hash(path),
            "bytes": path.stat().st_size}


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise mt.TaskExecutionError(f"无法读取 JSON 证据 {path}：{exc}") from exc


def _no_symlinks(path):
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    _need(not any(p.is_symlink() for p in (path, *path.parents)), f"拒绝 symlink 路径：{path}")
    return Path(os.path.abspath(path))


def _local(base, relative):
    _need(isinstance(relative, str) and bool(relative), "证据必须使用本地相对路径")
    part = Path(relative)
    _need(not part.is_absolute() and all(p not in (".", "..") for p in relative.split("/"))
          and "\\" not in relative, f"非法本地相对路径：{relative}")
    result = _no_symlinks(base / part)
    _need(result.is_relative_to(base.resolve()), f"路径越界：{relative}")
    return result


def _tree_files(base):
    result = []
    for path in sorted(base.rglob("*")):
        _need(not path.is_symlink(), f"拒绝 bundle symlink：{path}")
        _need(path.is_dir() or path.is_file(), f"拒绝非普通文件：{path}")
        if path.is_file():
            result.append(path)
    return result


def effective_recipe(r, checkpoint):
    """只覆盖训练 seed；不改原始 Recipe、数据 seed、评估 seed 或 TOML。"""
    source = _no_symlinks(Path(checkpoint))
    training = _json(_local(source, "training.json"))
    _need(isinstance(training, dict), "training.json 必须为对象")
    seed = training.get("seed")
    _need(type(seed) is int and seed >= 0, "training.seed 必须为非负整数")
    from .provenance import effective_recipe as with_train_seed
    try:
        return with_train_seed(r, seed)
    except ValueError as exc:
        raise mt.TaskExecutionError(str(exc)) from exc


def _tolerance(r):
    names = r.get("evaluation.metrics")
    _need(isinstance(names, list) and bool(names) and all(isinstance(n, str) and n for n in names)
          and len(names) == len(set(names)), "evaluation.metrics 必须为非空且不重复的指标表")
    table = r.get("evaluation.export_tolerance") if r.has("evaluation.export_tolerance") else r.get("evaluation.tolerance")
    _need(isinstance(table, dict) and set(names) <= set(table), "导出 tolerance 缺失或未覆盖全部 evaluation.metrics；不得造数")
    _need(all(be.finite_number(table[n]) and table[n] >= 0 for n in names), "导出 tolerance 必须为有限非负数值")
    return {n: table[n] for n in names}


def _input_hashes(r, training, evaluation, lock):
    lock_sha = _hash(lock)
    for label, evidence in (("训练", training), ("评估", evaluation)):
        _need(bool(be.HEX64.fullmatch(str(evidence.get("code_sha256", "")))), f"{label}缺少 code_sha256")
        _need(evidence.get("environment_lock_sha256") == lock_sha, f"{label} environment_lock_sha256 与锁字节不一致")
    from .provenance import code_hash
    _need(training["code_sha256"] == evaluation["code_sha256"] == code_hash(), "训练与评估及当前复载代码版本 hash 不一致")
    return lock_sha


def _training(r, checkpoint):
    training = _json(_local(checkpoint, "training.json"))
    _need(isinstance(training, dict) and training.get("kind") == "training"
          and training.get("stage") == "completed" and training.get("mode") == "full"
          and training.get("task_id") == mt.TASK_ID and training.get("recipe_id") == r.id
          and not any(training.get(k) is True for k in ("mock", "simulated", "dry_run")),
          "仅允许 full 且非 mock 的 completed 训练证据")
    _need(training.get("schema_version") == 1 and training.get("fingerprint") == be.fingerprint(r)
          and training.get("seed") == r.get("train.seed")
          and training.get("data_seed") == r.get("train.data_seed"), "训练来源配置或 seed 不一致")
    _need(be.frozen_environment(r) and be.runtime_consistent(r, training), "训练环境不一致或不是 frozen lock")
    config = training.get("config")
    _need(isinstance(config, dict) and training.get("config_sha256") == mt.sha256_json(config), "训练配置内容 hash 不一致")
    _need(all(config.get(k) == v for k, v in r.get("train", {}).items()), "训练超参数与有效 Recipe 不一致")
    max_steps = r.get("train.max_steps", r.get("budget.full_steps"))
    _need(type(max_steps) is int and max_steps > 0 and type(training.get("steps")) is int
          and training["steps"] == training.get("max_steps") == config.get("max_steps") == max_steps,
          "训练未完成 full 预算")
    artifact = mt.artifact_manifest(checkpoint)
    _need(training.get("model_artifact", {}).get("files") == artifact["files"]
          and training["model_artifact"].get("sha256") == artifact["sha256"], "训练 artifact 文件或 hash 不一致")
    return training


def _evaluation(r, path, artifact, *, recipe_sha=None):
    data = _json(path)
    problems = be.evaluation_problems(r, data, path, role="trained_checkpoint")
    _need(not problems, "评估证据拒绝：" + "；".join(problems))
    _need(data.get("schema_version") == 1 and data.get("task_id") == mt.TASK_ID
          and data.get("fingerprint") == be.fingerprint(r)
          and data.get("seed") == r.get("train.seed") and data.get("data_seed") == r.get("train.data_seed"),
          "评估 fingerprint / seed / 任务不一致")
    _need(bool(be.HEX64.fullmatch(str(data.get("recipe_sha256", "")))), "评估缺少 recipe_sha256")
    if recipe_sha is not None:
        _need(data.get("recipe_sha256") == recipe_sha, "评估 recipe snapshot hash 不一致")
    model = data.get("model_artifact", {})
    _need(model.get("role") == "trained_checkpoint" and model.get("sha256") == artifact["sha256"]
          and model.get("files") == artifact["files"], "评估 model artifact 与 checkpoint 不一致")
    config = data["config"]
    _need(config.get("task_id") == mt.TASK_ID and config.get("system_prompt") == mt.SYSTEM_PROMPT
          and config.get("model") == model and config.get("dataset_revision") == r.get("dataset.revision")
          and config.get("test_sha256") == data["split_manifest"]["splits"]["test"]["sha256"],
          "评估输入、prompt 或 split config 不一致")
    generation = config.get("generation", {})
    _need(generation.get("do_sample") is False and generation.get("num_beams") == 1
          and generation.get("seed") == r.get("evaluation.seed", r.get("train.seed"))
          and generation.get("max_new_tokens") == r.get("train.max_completion_length"), "评估不是相同 greedy 配置")
    prediction_record = data.get("predictions_file", {})
    prediction_path = _local(path.parent, prediction_record.get("path"))
    predictions = _json(prediction_path)
    description = data.get("predictions", {})
    _need(description.get("path") == prediction_record.get("path")
          and description.get("count") == len(predictions)
          and description.get("sha256") == mt.sha256_json(predictions), "逐题预测内容 hash 不一致")
    _need(_json(_local(path.parent, "manifest.json")) == data["split_manifest"], "manifest.json 与评估划分不一致")
    # 内容 hash 绑定 gold 与 prompt；同 sample_id 不足以防止替换答案。
    for prediction, sample in zip(predictions, data["split_manifest"]["splits"]["test"]["samples"]):
        prompt, solution = prediction.get("prompt"), prediction.get("solution")
        _need(isinstance(prompt, list) and len(prompt) == 2 and isinstance(prompt[1], dict)
              and isinstance(prompt[1].get("content"), str) and bool(prompt[1]["content"].strip())
              and prompt == mt.make_prompt(prompt[1]["content"])
              and mt.sha256_json(prompt[1]["content"]) == sample["problem_sha256"]
              and isinstance(solution, str) and bool(solution.strip())
              and mt.sha256_json(solution) == sample.get("solution_sha256")
              and isinstance(prediction.get("completion"), str), "逐题 prompt / gold / completion 非法")
        compliant = int(mt.answer_text(prediction["completion"]) is not None)
        _need(prediction.get("format_compliance") == compliant
              and (prediction.get("accuracy") != 1 or compliant == 1), "逐题格式分数与输出不一致")
    return data, predictions


def _same_protocol(reference, actual, reference_predictions, actual_predictions):
    for key in ("recipe_id", "fingerprint", "task_id", "seed", "data_seed", "model_id", "model_revision",
                "dataset_id", "dataset_revision", "split_manifest", "sample_count", "evaluation_protocol",
                "code_sha256", "environment_lock_sha256"):
        _need(actual.get(key) == reference.get(key), f"复载输入或协议不一致：{key}")
    for key in ("system_prompt", "generation", "dataset_revision", "test_sha256"):
        _need(actual["config"].get(key) == reference["config"].get(key), f"复载 config 不一致：{key}")
    for reference_row, actual_row in zip(reference_predictions, actual_predictions):
        for key in ("sample_id", "content_sha256", "prompt", "solution"):
            _need(actual_row.get(key) == reference_row.get(key), f"复载逐题输入 / gold 不一致：{key}")


def _verify_bundle(bundle, *, expected_manifest_sha=None):
    bundle = _no_symlinks(bundle)
    _need(bundle.is_dir(), "bundle 目录不存在")
    manifest_path = _local(bundle, MANIFEST)
    if expected_manifest_sha is not None:
        _need(_hash(manifest_path) == expected_manifest_sha, "bundle manifest 被篡改")
    manifest = _json(manifest_path)
    _need(isinstance(manifest, dict) and manifest.get("schema_version") == 1
          and manifest.get("kind") == "export-bundle" and manifest.get("stage") == "prepared"
          and manifest.get("sha256") == mt.sha256_json({k: v for k, v in manifest.items() if k != "sha256"}),
          "bundle manifest 结构或 hash 非法")
    records = manifest.get("files")
    _need(isinstance(records, dict) and bool(records), "bundle 未绑定文件")
    for relative, record in records.items():
        path = _local(bundle, relative)
        _need(isinstance(record, dict) and record.get("path") == relative and path.is_file()
              and record.get("bytes") == path.stat().st_size and _hash(path) == record.get("sha256"),
              f"bundle 文件 hash 不一致：{relative}")
    for path in _tree_files(bundle):
        relative = path.relative_to(bundle).as_posix()
        _need(relative in records or relative in (MANIFEST, WORKER_EVIDENCE, FINAL)
              or relative.startswith("reload/eval/"), f"bundle 包含未绑定文件：{relative}")
    snapshot = _json(_local(bundle, SNAPSHOT))
    _need(isinstance(snapshot, dict) and snapshot.get("id") == manifest.get("recipe_id"), "Recipe snapshot 非法")
    r = Recipe(manifest["recipe_id"], bundle / "recipe.toml", snapshot)
    _need(r.get("execution.lockfile") == LOCKFILE and be.fingerprint(r) == manifest.get("fingerprint"), "snapshot lock / fingerprint 不一致")
    mt.require_supported(r)
    _need(_tolerance(r) == manifest.get("tolerance"), "bundle tolerance 被替换")
    checkpoint = mt.checkpoint_path({}, _local(bundle, "model"), recipe=r)
    training = _training(r, checkpoint)
    artifact = mt.artifact_manifest(checkpoint)
    reference_path = _local(bundle, "reference/evaluation.json")
    reference, predictions = _evaluation(r, reference_path, artifact,
                                         recipe_sha=manifest.get("reference_recipe_sha256"))
    _need(training.get("split_manifest") == reference["split_manifest"], "训练与评估 split 不一致")
    _input_hashes(r, training, reference, _local(bundle, LOCKFILE))
    _need(manifest.get("model_artifact_sha256") == artifact["sha256"], "bundle 模型身份不一致")
    required = {SNAPSHOT, "recipe.toml", LOCKFILE, "model/training.json", "reference/evaluation.json",
                "reference/manifest.json", "reference/" + reference["predictions_file"]["path"]}
    required.update("model/" + name for name in artifact["files"])
    _need(set(records) == required, "bundle 文件清单不完整或含额外输入")
    return r, manifest, reference, predictions


def run_worker(bundle):
    """内部固定 worker；真实评估委派给延迟导入框架的 math_task。"""
    _need(all(os.environ.get(k) == v for k, v in OFFLINE_ENV.items()), "worker 拒绝非 offline / 非确定性环境")
    bundle = Path(bundle)
    _need(bundle.is_absolute(), "worker --bundle 必须为 ABS 路径")
    r, manifest, reference, predictions = _verify_bundle(bundle)
    _need(not (bundle / "reload").exists() and not (bundle / WORKER_EVIDENCE).exists()
          and not (bundle / FINAL).exists(), "worker 输出已存在，拒绝重用证据")
    manifest_sha = _hash(bundle / MANIFEST)
    from .plan import build_plan
    plan = build_plan(r, mode="eval")
    plan["run"]["output_dir"] = str(bundle / "reload" / "eval")
    try:
        evaluation_path = mt.run_evaluation(r, plan, checkpoint=bundle / "model")
    except Exception as exc:
        raise mt.TaskExecutionError(f"离线真实评估失败：{exc}。{DATA_DEPENDENCY}") from exc
    _need(Path(evaluation_path) == bundle / "reload" / "eval" / "evaluation.json", "worker 评估输出路径不一致")
    _verify_bundle(bundle, expected_manifest_sha=manifest_sha)
    actual, actual_predictions = _evaluation(r, Path(evaluation_path), mt.artifact_manifest(bundle / "model"),
                                             recipe_sha=mt.sha256_json(r.data))
    _same_protocol(reference, actual, predictions, actual_predictions)
    evidence = {
        "schema_version": 1, "kind": "export-reload-worker", "stage": "completed",
        "pid": os.getpid(), "ppid": os.getppid(), "runtime": actual["runtime"],
        "bundle_manifest_sha256": manifest_sha, "fingerprint": be.fingerprint(r),
        "evaluation": _record(Path(evaluation_path), bundle), "offline_environment": dict(OFFLINE_ENV),
    }
    return mt.save_json(bundle / WORKER_EVIDENCE, evidence)


def validate_export_evidence(r, path):
    """只读核验完成导出包；审核布尔值不能替代模型/复载内容检查。"""
    path = _no_symlinks(Path(path))
    _need(path.name == FINAL, "导出证据必须为包内export.json")
    bundle = path.parent
    data = _json(path)
    _need(isinstance(data, dict) and data.get("schema_version") == 1
          and data.get("kind") == "export-reload" and data.get("stage") == "completed"
          and data.get("recipe_id") == r.id and data.get("task_id") == mt.TASK_ID
          and not any(data.get(k) is True for k in ("mock", "simulated", "dry_run")), "导出完成身份非法")
    effective = effective_recipe(r, bundle / "model")
    copied, manifest, reference, predictions = _verify_bundle(bundle)
    _need(be.fingerprint(effective) == be.fingerprint(copied) == data.get("fingerprint"), "导出与当前有效recipe不一致")
    _need(_tolerance(effective) == manifest["tolerance"], "当前导出容差与包内协议不一致")
    original_lock = _no_symlinks(r.path.parent / str(r.get("execution.lockfile")))
    _need(original_lock.read_bytes() == (bundle / LOCKFILE).read_bytes(), "导出锁与当前lock不同")
    files = data.get("files")
    _need(isinstance(files, dict), "导出缺少完整文件绑定")
    actual_files = {p.relative_to(bundle).as_posix() for p in _tree_files(bundle) if p != path}
    _need(set(files) == actual_files, "导出文件清单缺失或出现未绑定文件")
    for relative, record in files.items():
        file = _local(bundle, relative)
        _need(record == _record(file, bundle), f"导出文件被修改：{relative}")
    for key, relative in (("bundle_manifest", MANIFEST), ("worker_evidence", WORKER_EVIDENCE),
                          ("reference_evaluation", "reference/evaluation.json"),
                          ("reload_evaluation", "reload/eval/evaluation.json")):
        _need(data.get(key) == files[relative], f"导出关联记录不一致：{key}")
    worker = _json(bundle / WORKER_EVIDENCE)
    process = data.get("subprocess", {})
    _need(isinstance(process, dict) and type(process.get("pid")) is int and process["pid"] > 0
          and type(process.get("parent_pid")) is int and process["parent_pid"] > 0
          and process["pid"] != process["parent_pid"] and process.get("returncode") == 0
          and process.get("offline_environment") == OFFLINE_ENV, "导出新进程身份不一致")
    _need(isinstance(worker, dict) and worker.get("kind") == "export-reload-worker"
          and worker.get("stage") == "completed" and worker.get("pid") == process["pid"]
          and worker.get("ppid") == process["parent_pid"] and worker.get("offline_environment") == OFFLINE_ENV
          and worker.get("fingerprint") == data["fingerprint"]
          and worker.get("bundle_manifest_sha256") == _hash(bundle / MANIFEST), "导出worker身份不一致")
    actual_path = bundle / "reload/eval/evaluation.json"
    _need(worker.get("evaluation") == files["reload/eval/evaluation.json"], "worker关联评估不同")
    artifact = mt.artifact_manifest(bundle / "model")
    actual, actual_predictions = _evaluation(copied, actual_path, artifact, recipe_sha=mt.sha256_json(copied.data))
    _same_protocol(reference, actual, predictions, actual_predictions)
    from .suite_task import _manifest
    _manifest(copied, actual["split_manifest"])
    _need(worker.get("runtime") == actual["runtime"] == data.get("runtime")
          and data.get("metrics") == actual["metrics"]
          and data.get("code_sha256") == actual["code_sha256"]
          and data.get("environment_lock_sha256") == actual["environment_lock_sha256"]
          and data.get("split_sha256") == actual["split_manifest"]["sha256"]
          and data.get("model_artifact_sha256") == artifact["sha256"]
          and data.get("seed") == effective.get("train.seed"), "导出摘要与复载实际评估不一致")
    comparisons = {}
    for name, limit in _tolerance(effective).items():
        delta = abs(reference["metrics"][name] - actual["metrics"][name])
        _need(delta <= limit, "复载指标超出当前容差")
        comparisons[name] = {"reference": reference["metrics"][name], "reload": actual["metrics"][name],
                             "absolute_delta": delta, "tolerance": limit, "within_tolerance": True}
    _need(data.get("comparison") == comparisons, "导出容差摘要未重算或已改变")
    return True


def _launch_worker(bundle):
    command = [sys.executable, str(CODE_ROOT / "runner" / "export.py"), "--worker", "--bundle", str(bundle)]
    env = {**os.environ, **OFFLINE_ENV}
    process = None
    try:
        process = subprocess.Popen(command, cwd=CODE_ROOT, env=env, shell=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=WORKER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise mt.TaskExecutionError("离线复载子进程 timeout；不生成 completed 导出证据") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
        raise mt.TaskExecutionError(f"离线复载子进程启动 / 通信失败：{exc}") from exc
    _need(process.returncode == 0, f"离线复载子进程失败（returncode={process.returncode}）：{stderr}")
    return {"command": command, "pid": process.pid, "parent_pid": os.getpid(), "returncode": process.returncode,
            "stdout": stdout, "stderr": stderr, "offline_environment": dict(OFFLINE_ENV)}


def export_bundle(r, checkpoint, evaluation_path, destination, execute=True):
    if not execute:
        try:
            mt.require_supported(r)
            supported, reason = True, None
        except mt.TaskExecutionError as exc:
            supported, reason = False, str(exc)
        return {
            "schema_version": 1, "kind": "export-plan", "stage": "declared", "recipe_id": r.id,
            "task_id": mt.resolve_task_id(r), "execution_supported": supported, "unsupported_reason": reason,
            "checkpoint": str(checkpoint), "evaluation": str(evaluation_path), "destination": str(destination),
            "train_seed": r.get("train.seed"), "effective_seed_source": "execute 时读取 checkpoint training.seed；只覆盖 train.seed",
            "data_seed": r.get("train.data_seed"), "evaluation_seed": r.get("evaluation.seed"),
            "tolerance": r.get("evaluation.export_tolerance") if r.has("evaluation.export_tolerance") else r.get("evaluation.tolerance"),
            "offline_environment": dict(OFFLINE_ENV), "data_dependency": DATA_DEPENDENCY,
            "license": {"status": "pending", "commercial_use_confirmed": False}, "baseline_promotion": "none",
        }
    mt.require_supported(r)
    source = _no_symlinks(Path(checkpoint))
    output = _no_symlinks(Path(destination))
    _need(not output.is_relative_to(source) and not source.is_relative_to(output), "destination 不能与 checkpoint 路径内嵌或重叠")
    _need(not output.exists() or (output.is_dir() and not any(output.iterdir())), "拒绝 nonempty destination 或非目录输出")
    evaluation_path = _no_symlinks(Path(evaluation_path))
    _need(evaluation_path.is_file(), "reference evaluation 不存在")
    r = effective_recipe(r, source)
    mt.require_supported(r)
    mt.fixed_seeds(r)
    tolerance = _tolerance(r)
    source = mt.checkpoint_path({}, source, recipe=r)
    training = _training(r, source)
    artifact = mt.artifact_manifest(source)
    reference, reference_predictions = _evaluation(r, evaluation_path, artifact)
    _need(training.get("split_manifest") == reference["split_manifest"], "训练与评估 split 不一致")
    lock = _no_symlinks(r.path.parent / str(r.get("execution.lockfile")))
    _input_hashes(r, training, reference, lock)
    copy_sources = {"model/" + name: _local(source, name) for name in artifact["files"]}
    copy_sources.update({"model/training.json": _local(source, "training.json"), LOCKFILE: lock,
                         "reference/evaluation.json": evaluation_path,
                         "reference/manifest.json": _local(evaluation_path.parent, "manifest.json")})
    prediction_name = reference["predictions_file"]["path"]
    _need(prediction_name not in ("evaluation.json", "manifest.json"), "预测路径与证据路径冲突")
    copy_sources["reference/" + prediction_name] = _local(evaluation_path.parent, prediction_name)
    expected_sources = {relative: _record(path, path.parent) for relative, path in copy_sources.items()}
    snapshot = copy.deepcopy(r.data)
    snapshot.setdefault("execution", {})["lockfile"] = LOCKFILE
    snapshot_recipe = Recipe(r.id, output / "recipe.toml", snapshot)
    _need(be.fingerprint(snapshot_recipe) == be.fingerprint(r), "lock 重定位不得改变 fingerprint")
    output.mkdir(parents=True, exist_ok=True)
    for relative, path in copy_sources.items():
        target = _local(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        _need(_hash(target) == expected_sources[relative]["sha256"], f"复制期间来源被篡改：{relative}")
    mt.save_json(output / SNAPSHOT, snapshot)
    (output / "recipe.toml").write_text('id = ' + json.dumps(r.id) + '\n[execution]\nlockfile = "' + LOCKFILE + '"\n', encoding="utf-8")
    files = {path.relative_to(output).as_posix(): _record(path, output) for path in _tree_files(output)}
    manifest = {
        "schema_version": 1, "kind": "export-bundle", "stage": "prepared", "recipe_id": r.id,
        "fingerprint": be.fingerprint(r), "source_recipe_sha256": mt.sha256_json(r.data),
        "reference_recipe_sha256": reference["recipe_sha256"],
        "model_artifact_sha256": artifact["sha256"], "files": files, "tolerance": tolerance,
        "data_dependency": DATA_DEPENDENCY, "license": {"status": "pending", "commercial_use_confirmed": False},
    }
    manifest["sha256"] = mt.sha256_json(manifest)
    mt.save_json(output / MANIFEST, manifest)
    manifest_sha = _hash(output / MANIFEST)
    _verify_bundle(output, expected_manifest_sha=manifest_sha)
    process_info = _launch_worker(output)
    reloaded_recipe, _, copied_reference, copied_predictions = _verify_bundle(output, expected_manifest_sha=manifest_sha)
    worker = _json(_local(output, WORKER_EVIDENCE))
    _need(isinstance(worker, dict) and worker.get("schema_version") == 1
          and worker.get("kind") == "export-reload-worker" and worker.get("stage") == "completed"
          and type(worker.get("pid")) is int and worker["pid"] == process_info["pid"] != os.getpid()
          and worker.get("ppid") == os.getpid() and worker.get("fingerprint") == be.fingerprint(r)
          and worker.get("bundle_manifest_sha256") == manifest_sha
          and worker.get("offline_environment") == OFFLINE_ENV, "worker 身份 / pid / offline / completed 证据不一致")
    record = worker.get("evaluation", {})
    _need(record.get("path") == "reload/eval/evaluation.json" and be.hashed_file(output, record), "worker evaluation.json hash 不一致")
    actual_path = _local(output, record["path"])
    actual, actual_predictions = _evaluation(reloaded_recipe, actual_path, artifact,
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
    evidence = {
        "schema_version": 1, "kind": "export-reload", "stage": "completed", "task_id": mt.TASK_ID,
        "recipe_id": r.id, "fingerprint": be.fingerprint(r), "completed_at": datetime.now(timezone.utc).isoformat(),
        "seed": r.get("train.seed"), "data_seed": r.get("train.data_seed"), "evaluation_seed": r.get("evaluation.seed"),
        "runtime": actual["runtime"], "subprocess": process_info, "metrics": actual["metrics"],
        "comparison": comparisons, "model_artifact_sha256": artifact["sha256"],
        "code_sha256": actual["code_sha256"], "environment_lock_sha256": actual["environment_lock_sha256"],
        "bundle_manifest": _record(output / MANIFEST, output), "worker_evidence": _record(output / WORKER_EVIDENCE, output),
        "reference_evaluation": _record(output / "reference" / "evaluation.json", output),
        "reload_evaluation": _record(actual_path, output), "split_sha256": actual["split_manifest"]["sha256"],
        "license": {"status": "pending", "commercial_use_confirmed": False},
        "data_dependency": DATA_DEPENDENCY, "baseline_promotion": "none", "training_seed_variance_measured": False,
        "files": {path.relative_to(output).as_posix(): _record(path, output) for path in _tree_files(output)},
    }
    _need(not (output / FINAL).exists(), "final 导出证据已存在，拒绝覆盖")
    return mt.save_json(output / FINAL, evidence)
