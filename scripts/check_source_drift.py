#!/usr/bin/env python3
"""上游来源漂移的辅助检查。

默认**不联网**：只列出未锁定 source_sha 的 recipe，可在 CI 安全运行。
加 --network 才会发 HEAD 请求检查上游页面可达性（建议本地手动跑）。

用法：
    PYTHONPATH=src python3 scripts/check_source_drift.py
    PYTHONPATH=src python3 scripts/check_source_drift.py --network
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recipe_lab.loader import load_recipes, repo_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true", help="联网检查上游页面可达性")
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    root = repo_root()
    recipes = load_recipes(root)

    unlocked = [r for r in recipes if not r.get("source_sha")]
    print(f"共 {len(recipes)} 篇 recipe，其中 {len(unlocked)} 篇未锁定上游 commit。")
    if unlocked:
        print("建议把抓取时的 commit sha 写入 source_sha，便于日后 diff 上游改动：")
        for r in unlocked:
            print(f"  - {r.id}")

    if not args.network:
        print("\n（离线模式。加 --network 可检查上游页面可达性。）")
        return 0

    failed = 0
    print("\n检查上游页面：")
    for r in recipes:
        url = str(r.get("source", ""))
        try:
            req = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "recipe-lab/0.1"}
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                status = resp.status
            print(f"  [ok]   {status} {r.id}")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            # 有些站点拒绝 HEAD，降级为 GET 再试一次。
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "recipe-lab/0.1", "Range": "bytes=0-0"}
                )
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    status = resp.status
                print(f"  [ok]   {status} {r.id} (GET 降级)")
            except Exception as exc2:  # noqa: BLE001
                failed += 1
                print(f"  [FAIL] {r.id}: HEAD {exc} / GET {exc2}")

    print(f"\n不可达 {failed} 篇。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
