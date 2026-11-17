"""toml_edit：迁移脚本与 runner/lock.py 共用的最小侵入式改写器。

重点不是「能插进去」，而是「不破坏已有内容」——注释、多行数组、段间空行，
以及绝不能因为 [baseline_contract].state 而误改 [repro.code].state。
"""

from __future__ import annotations

import tomllib

from recipe_lab.toml_edit import quote, upsert_scalar

SRC = '''schema = 1
id = "x"
source_sha = ""

[train]
num_train_epochs = 3  # 保留这个行尾注释以外的结构
learning_rate = 2e-4

[known_issues]
items = [
  "a",
  "b",
]

[repro.code]
state = "missing"
'''


def test_insert_into_existing_section_preserves_layout() -> None:
    out = upsert_scalar(SRC, "train", "seed", "42")
    d = tomllib.loads(out)
    assert d["train"]["seed"] == 42
    assert d["train"]["num_train_epochs"] == 3
    assert d["known_issues"]["items"] == ["a", "b"]
    assert "[known_issues]\nitems = [" in out, "多行数组被破坏了"
    assert "learning_rate = 2e-4\nseed = 42" in out, "新键应追加到段尾而非段首"
    assert "[train]\nnum_train_epochs" in out, "段头后的空行不应被吞掉"


def test_replace_existing_key_is_idempotent() -> None:
    once = upsert_scalar(SRC, "train", "seed", "42")
    twice = upsert_scalar(once, "train", "seed", "42")
    assert once == twice
    changed = upsert_scalar(once, "train", "seed", "7")
    assert tomllib.loads(changed)["train"]["seed"] == 7


def test_new_section_is_appended_and_parses() -> None:
    out = upsert_scalar(SRC, "evaluation", "split", quote("test[:10%]"))
    d = tomllib.loads(out)
    assert d["evaluation"]["split"] == "test[:10%]"
    assert out.count("[evaluation]") == 1


def test_top_level_section_none() -> None:
    out = upsert_scalar(SRC, None, "source_sha", quote("abc123"))
    assert tomllib.loads(out)["source_sha"] == "abc123"
    assert out.count("source_sha =") == 1
    inserted = upsert_scalar(SRC, None, "owner", quote("team"))
    assert tomllib.loads(inserted)["owner"] == "team"


def test_state_in_baseline_contract_does_not_touch_repro_state() -> None:
    out = upsert_scalar(SRC, "baseline_contract", "state", quote("B1"))
    d = tomllib.loads(out)
    assert d["baseline_contract"]["state"] == "B1"
    assert d["repro"]["code"]["state"] == "missing"
    # 迁移脚本早期版本用全局 re.sub(r'^state = "B\d"$')，会漏改/误伤；这里锁死行为。
    again = upsert_scalar(out, "baseline_contract", "state", quote("B2"))
    assert tomllib.loads(again)["repro"]["code"]["state"] == "missing"


def test_existing_execution_table_is_reused_not_duplicated() -> None:
    """Wordle 篇已有 [execution]（HF Jobs + papermill），迁移不能造出重复表。"""
    src = """[execution]
runner = "hf-jobs"
launcher = "papermill"

[rewards]
functions = ["format_reward"]
"""
    out = src
    for key, value in (("command", "train.py"), ("entrypoint", "../../runner/train.py")):
        out = upsert_scalar(out, "execution", key, quote(value))
    d = tomllib.loads(out)
    assert out.count("[execution]") == 1
    assert d["execution"]["runner"] == "hf-jobs", "原有字段不能丢"
    assert d["execution"]["command"] == "train.py"
    assert d["rewards"]["functions"] == ["format_reward"]


def test_quote_escapes() -> None:
    assert quote('a"b\\c') == '"a\\"b\\\\c"'
    assert tomllib.loads("v = " + quote('a"b\\c'))["v"] == 'a"b\\c'
    assert tomllib.loads("v = " + quote("line1\nline2"))["v"] == "line1\nline2"
