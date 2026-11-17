"""recipe-lab: 配置驱动的 LLM / VLM 训练 recipe 登记与验证工具链。

设计目标：核心工具链零第三方依赖（仅用 Python 3.11+ 标准库 tomllib），
因为「用来验证别人环境的东西」自己不能装不上。
"""

__version__ = "0.1.0"
SCHEMA_VERSION = 1

STATUS_ORDER = ("cataloged", "ported", "smoke-passed", "benchmarked")
