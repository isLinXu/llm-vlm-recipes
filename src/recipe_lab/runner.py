"""分层验证编排。"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Finding, Recipe
from .rules import GLOBAL_RULES, PER_RECIPE_RULES, Context


@dataclass
class Report:
    recipes: list[Recipe]
    findings: list[Finding]
    level: int

    def by_severity(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def errors(self) -> list[Finding]:
        return self.by_severity("error")

    @property
    def warnings(self) -> list[Finding]:
        return self.by_severity("warning")

    @property
    def infos(self) -> list[Finding]:
        return self.by_severity("info")

    def for_recipe(self, recipe_id: str) -> list[Finding]:
        return [f for f in self.findings if f.recipe_id == recipe_id]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "recipe_count": len(self.recipes),
            "summary": {
                "error": len(self.errors),
                "warning": len(self.warnings),
                "info": len(self.infos),
            },
            "ok": self.ok,
            "findings": [f.as_dict() for f in self.findings],
        }


def validate(ctx: Context) -> Report:
    findings: list[Finding] = []
    for recipe in ctx.recipes:
        for level, rule in PER_RECIPE_RULES:
            if level <= ctx.level:
                findings.extend(rule(recipe, ctx))
    for level, rule in GLOBAL_RULES:
        if level <= ctx.level:
            findings.extend(rule(ctx))

    order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (order[f.severity], f.level, f.rule, f.recipe_id))
    return Report(recipes=ctx.recipes, findings=findings, level=ctx.level)
