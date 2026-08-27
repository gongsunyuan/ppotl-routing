"""`python -m trl_sb3` 占位入口：仅确认包可导入。

实验编排入口为 `python -m trl_sb3.run ...`（sweep/make_figures，M2-4 起落地）。
"""

import sys

from trl_sb3 import __version__


def _main() -> int:
    print(f"trl_sb3 {__version__} — experiment entrypoints: python -m trl_sb3.run ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
