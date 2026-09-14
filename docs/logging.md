# 日志设计与使用

设计参考 `dexe_agent/src/dexe_agent/common/log/logger.py`，基于 Python 标准库
`logging`：以包名作为父 logger，通过 `setup_logging()` 统一配置，使用
`add_logger_handler()` 扩展输出，通过环境变量启用调试日志。

业务模块使用 `get_logger(__name__)` 获取子 logger；外部适配器传入短名称时会自动
添加 `embodiedforge.` 前缀。也兼容标准库的
`logging.getLogger("embodiedforge.worker")`。配置在应用入口执行，模块导入只安装
`NullHandler`，不自动读取环境变量或创建输出流。显式配置后关闭向 root 的传播，
不修改宿主应用的 root handlers 或日志级别。

```python
from embodiedforge.logging import get_logger, setup_logging

setup_logging("INFO", with_json_format=True)
logger = get_logger("worker", run_id="run-1", backend="numpy")
logger.info("Worker started")
```

文本格式沿用参考仓库的时间、级别、logger 名、文件、函数和四位补零行号：
`%(asctime)s|%(levelname)s|%(name)s|%(filename)s-%(funcName)s-%(lineno)04d]: %(message)s`。
JSON 格式由标准库实现，无需额外依赖；每条记录独占一行，包含 UTC 时间戳、
级别、logger、消息、源码位置、进程、线程和 `context`。固定上下文只在 JSON
格式中输出，避免放入图像或传感器数组。`logger.exception()` 保留完整异常栈。

针对本项目的使用场景，保留以下行为：

- 默认输出到 **stderr**，让 CLI 的 stdout 保持可解析的工作流 JSON。
- `handlers=None` 创建默认控制台 handler；传入列表则仅使用列表中的 handlers，
  `handlers=[]` 关闭包内输出，不额外追加控制台 handler。
- 自定义 handler 保留原有级别和 formatter；未设置 formatter 时补充所选格式。
  不修改调用方的列表，也不关闭调用方提供的 handler。
- 重复配置替换旧 handlers，并关闭本模块创建且不再使用的 handler；重复添加同一
  handler 不会重复输出。配置应在应用启动时执行，不应与日志输出并发进行。

CLI 和可视化入口显式调用 `auto_configure_debug_logging()`，支持
`EMBODIEDFORGE_DEBUG=1` 与 `EMBODIEDFORGE_JSON_FORMAT_LOG=1`；也接受
`true`、`yes`、`on`（忽略大小写）。JSON 环境变量仅在 DEBUG 开关启用时生效。
普通库调用需要自行调用该函数或 `setup_logging()`。

```bash
EMBODIEDFORGE_DEBUG=1 EMBODIEDFORGE_JSON_FORMAT_LOG=1 embodiedforge plan
embodiedforge plan --log-level INFO --log-json
```

入口提供的 `--log-level` / `--log-json` 显式选项优先于环境配置。

可视化入口默认启用 INFO 生命周期日志，记录初始化、首帧生成、就绪耗时和退出。
`--duration` 从初始帧完成后计时；`--log-level WARNING` 可减少应用输出。
第三方原生 SDK 的 stdout/stderr 不会自动转换为应用 JSON 日志。
