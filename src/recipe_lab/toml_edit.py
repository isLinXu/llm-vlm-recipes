"""TOML 标量的最小侵入式读改写。

仓库刻意不引入 tomlkit（ADR-002：核心工具链零第三方依赖），但 `scripts/` 的
迁移脚本与 `runner/lock.py` 都需要在**不破坏注释、数组多行结构、段间空行**的
前提下替换或插入一个标量键。这里提供唯一实现，避免各处重复出略有差异的正则。

只支持「顶层标量」和「一级表内的标量」两种情形——这正是本仓库需要改写的全部
范围。更深的结构请直接编辑 TOML，不要用正则硬凑。
"""

from __future__ import annotations

import re


def quote(s: object) -> str:
    """生成 TOML basic string 字面量。"""
    text = str(s)
    return (
        '"'
        + text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        + '"'
    )


def upsert_scalar(text: str, section: str | None, key: str, literal: str) -> str:
    """在 ``section`` 内替换或插入 ``key = literal``。

    ``section=None`` 表示顶层标量区（第一个 ``[table]`` 之前）。
    ``literal`` 必须是**已格式化**的 TOML 值，如 ``42`` / ``"abc"`` / ``["a"]``；
    字符串请用 :func:`quote` 生成，不要手拼引号。

    段不存在时追加到文件末尾；段存在时把新键追加到该段末尾，
    原有注释、缩进、多行数组与段间空行保持不变。
    """
    key_pat = re.compile(rf"^{re.escape(key)}[ \t]*=.*$", re.M)

    if section is None:
        first = re.search(r"^\[", text, re.M)
        head_end = first.start() if first else len(text)
        head, tail = text[:head_end], text[head_end:]
        if key_pat.search(head):
            return key_pat.sub(f"{key} = {literal}", head, count=1) + tail
        return head.rstrip("\n") + f"\n{key} = {literal}\n\n" + tail.lstrip("\n")

    header = re.compile(rf"^\[{re.escape(section)}\][ \t]*$", re.M)
    m = header.search(text)
    if not m:
        return text.rstrip("\n") + f"\n\n[{section}]\n{key} = {literal}\n"

    # 下一个 [table] 作为本段边界；子表 [repro.code] 同样会正确成为边界。
    nxt = re.search(r"^\[", text[m.end():], re.M)
    block_end = m.end() + (nxt.start() if nxt else len(text) - m.end())
    block = text[m.end():block_end]

    if key_pat.search(block):
        return (
            text[:m.end()]
            + key_pat.sub(f"{key} = {literal}", block, count=1)
            + text[block_end:]
        )

    body = block.rstrip("\n")
    trailing = block[len(body):]
    new_block = f"{body}\n{key} = {literal}\n" + ("\n" if trailing else "")
    return text[:m.end()] + new_block + text[block_end:]
