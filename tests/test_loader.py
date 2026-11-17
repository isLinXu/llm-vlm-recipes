from __future__ import annotations

from pathlib import Path

from recipe_lab.loader import LoadError, load_recipes, load_taxonomy, read_toml, repo_root


def test_repo_root_found_from_tests(root: Path) -> None:
    assert (root / "registry" / "taxonomy.toml").is_file()
    assert repo_root(root / "recipes") == root


def test_all_recipes_load_and_ids_unique(ctx) -> None:
    assert len(ctx.recipes) == 14
    ids = [r.id for r in ctx.recipes]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_taxonomy_registered(ctx) -> None:
    assert "sft" in ctx.families
    assert ctx.families["grpo"]["trainer"] == "GRPOTrainer"
    assert ctx.families["mpo"]["trainer"] == "DPOTrainer"
    assert ctx.families["ppo"]["tier"] == "removed"
    assert ctx.compat, "compat.matrix 不应为空"


def test_every_recipe_id_matches_directory(ctx) -> None:
    for r in ctx.recipes:
        expected = r.path.parent.relative_to(ctx.root / "recipes").as_posix()
        assert r.id == expected


def test_bad_toml_raises_loud_error(tmp_path: Path) -> None:
    p = tmp_path / "broken.toml"
    p.write_text("this is = not = toml", encoding="utf-8")
    try:
        read_toml(p)
    except LoadError as exc:
        assert "TOML 解析失败" in str(exc)
    else:
        raise AssertionError("应抛出 LoadError")


def test_taxonomy_schema_is_int(root: Path) -> None:
    assert isinstance(load_taxonomy(root)["schema"], int)
