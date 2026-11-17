from __future__ import annotations

from recipe_lab.matrix import render
from recipe_lab.runner import validate


def test_matrix_contains_every_recipe(ctx) -> None:
    report = validate(ctx)
    text = render(report)
    for r in ctx.recipes:
        assert f"`{r.id}`" in text
    assert "## 全量矩阵" in text
    assert "## 版本漂移" in text
    assert "## 可复现性分布" in text
    assert "repro" in text
    assert "0/14 篇" in text  # 审计结论不得被悄悄改掉


def test_matrix_lists_warnings_as_backlog(ctx) -> None:
    report = validate(ctx)
    text = render(report)
    assert "## 待办清单（warning）" in text
    assert "warning 不代表失败" in text


def test_matrix_has_no_error_section_when_clean(ctx) -> None:
    report = validate(ctx)
    assert report.errors == []
    assert "## 阻塞项（error）" not in render(report)
