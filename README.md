# Gemini API Proxy 2.0

Gemini API Proxy 2.0 是一个面向 Render 单实例部署的 Gemini 反向代理，实现 Gemini CLI 与 Gemini API Key 的统一调度、OpenAI 兼容接口以及可视化运维面板。项目默认使用 FastAPI 提供 API，Streamlit 构建管理后台，两者在同一进程内协作即可完成部署。

## 功能特性

- **双池账号调度**：支持 Gemini CLI 账号与 Gemini API Key 并行接入，CLI 账号优先轮询且单独计费，Key 池具备健康监控、熔断与自动清理。
- **模型映射与搜索派生**：自动暴露 `*-search` 模型；CLI 账号请求时会注入工具搜索链路，API Key 则使用标准生成路径；Preview 型号继承基础模型的速率配置。
- **DeepThink 流水线**：通过串行多轮推理整合网页搜索、代码沙盒与普通分析，模型自行决定最多 7 轮迭代并生成最终答案。
- **安全代码执行**：内置受限 Python 沙盒（禁用网络、限制库、4 秒超时），可在推理过程中执行轻量级计算与解析任务。
- **一体化管理台**：Streamlit 控制台涵盖密钥管理、模型配置、使用率仪表板、健康检测、日志查看与在线 CLI OAuth。
- **任务队列与后台作业**：APScheduler 驱动保活、健康巡检与清理任务；所有请求均通过队列控制并发，适配 Render 免费层资源限制。

## 快速开始

### 环境要求

- Python 3.10 及以上
- 可访问外网以调用 Gemini API 与搜索工具

### 安装与启动

```bash
pip install -r requirements.txt
python -m app.runtime.server
```

默认监听 `0.0.0.0:8000`，
- 管理后台：`http://127.0.0.1:8000/admin`
- OpenAI 兼容接口：`POST /v1/chat/completions`
- 原生 Gemini 接口：`POST /v1beta/models/{model}:generateContent`

首次启动会在根目录生成 `gemini_proxy.db` 作为默认 SQLite 数据库。

## Render 部署

1. Fork 或导入仓库，确保 `render.yaml` 与项目位于同一目录。
2. 在 Render 创建 Blueprint 服务，指向仓库根目录；构建命令和启动命令使用默认配置即可：
   ```yaml
   buildCommand: pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
   startCommand: python -m app.runtime.server
   ```
3. 配置关键环境变量：
   - `GEMINI_AUTH_PASSWORD`：管理后台登录密码；
   - `API_BASE_URL`（可选）：绑定自定义域名后可显式指定；
   - `QUEUE_MAX_CONCURRENCY`、`QUEUE_MAX_WAIT_MS` 等按需调整；
4. 部署完成后访问 `<Render 外网地址>/admin`，执行 CLI OAuth 或导入 API Key。

> 建议在 Render 启用持久卷或改用外部数据库，以保留 CLI 授权记录与密钥状态。

## 控制台使用指南

### 登录
- 管理台默认开启密码保护，用户名固定为 `admin`，密码为 `GEMINI_AUTH_PASSWORD`。

### 添加账号
- **Gemini CLI**：在“密钥管理”页面生成授权链接，浏览器完成 Google 登录后系统自动导入账号；亦可上传 `oauth_creds.json`。
- **Gemini API Key**：在同一页面粘贴 Key，支持批量导入并即时健康检测。

### 模型配置
- “模型配置”页可调整显示名称、RPM/RPD/TPM 限额、默认思考预算与是否返回思考内容。
- CLI `gemini-2.5-pro`、`gemini-2.5-flash` 及全部 Preview 共享默认 1000 RPD；API Key 模型保留独立限额，均可在此处修改。

### 使用率分析
- 控制台默认按单一模型展示分钟 / 日使用率，可通过下拉切换目标模型，快速定位热点负载。

### DeepThink 与搜索
- 在“系统设置 → 实验性”中启用 DeepThink；用户消息包含 `[deepthink]` 即可触发串行推理。模型会根据上下文自动决定探索轮次，最大 7 轮。
- `*-search` 模型无需额外标签即可启动工具搜索；搜索链路通过 DuckDuckGo 查询并抓取网页内容，再交由模型生成回答。

## DeepThink 与代码沙盒

- DeepThink Planner 会在每一轮选择 `search`、`code`、`analysis` 或 `final`：
  - `search`：调用 DuckDuckGo 抓取（受 `search_num_pages_per_query` 控制）；
  - `code`：在受限 Python 沙盒执行代码，返回标准输出与错误；
  - `analysis`：直接向目标模型发送分析提示；
  - `final`：生成最终答案并结束循环。
- 沙盒默认允许 `math`、`statistics`、`json`、`datetime` 等少量安全模块，执行超时为 4 秒，可按需扩展。

## 配置与限额

- 所有配置保存在 SQLite 表 `system_config` 中，可通过管理台修改或在启动前设置环境变量。
- 常用变量：
  - `QUEUE_MAX_CONCURRENCY`、`QUEUE_MAX_WAIT_MS`：控制任务队列吞吐；
  - `STREAMLIT_INTERNAL_PORT`：Streamlit 子进程端口，默认 7000；
  - `API_BASE_URL`：显式指定对外访问地址；
  - `RENDER_EXTERNAL_URL`：Render 自动注入，用于构造 CLI OAuth 回调；
  - `SEARCH_ENABLED`、`SEARCH_NUM_PAGES_PER_QUERY`：控制工具搜索行为。
- CLI pro/flash（含 Preview）在单实例模式下共享 1000 RPD；API Key 模型在 `model_configs` 中保持独立配置。

## 健康与维护

- FastAPI 提供 `/health`、`/healthz` 与 `/wake` 端点，可用于负载均衡与 Render Keep-Alive。
- “密钥管理”支持一键健康检测与删除异常 Key；日志面板提供最近 200 条后台日志。
- APScheduler 默认开启保活与清理任务，可在 `app/admin/api_routes.py` 中停用或调整。

## 测试

```bash
python -m unittest discover -s tests
```

## 目录结构

```
app/
├── admin/        # 后台 API、数据库、CLI 功能
├── core/         # 配置与日志
├── proxy/        # Gemini/OpenAI 兼容路由
├── runtime/      # 启动脚本与 Streamlit 反代
├── tools/        # 受限代码执行工具
└── server.py     # FastAPI 主入口

streamlit_app/    # Streamlit 管理界面
tests/            # 单元测试
render.yaml       # Render 部署配置
```
