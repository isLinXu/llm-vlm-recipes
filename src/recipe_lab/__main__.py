"""允许 `PYTHONPATH=src python3 -m recipe_lab ...` 直接调用。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
