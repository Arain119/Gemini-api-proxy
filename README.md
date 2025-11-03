# Gemini API Proxy 2.0

Gemini API Proxy 2.0 是一个面向 Render / 本地单实例部署的 Gemini 反向代理与运维平台。项目将 FastAPI、Streamlit、任务调度及多账号池管理整合在同一进程中，既能兼容原生 Gemini/Google Generative AI 接口，也提供 OpenAI 格式的 `/v1/chat/completions` 端点与图形化控制台，帮助团队快速落地 Gemini 能力并维持稳定运行。

---

## 核心特性

- **双账号池调度**：内置 Gemini CLI OAuth 与普通 API Key 两类凭据，按优先级与健康度自动分配请求。
- **OpenAI 兼容层**：兼容大部分 Chat Completions 能力，含流式响应、模型列表等，便于接入现有 SDK。
- **模型映射与限额管理**：支持别名、RPM/RPD/TPM 限额、思考预算、流式模式等细粒度配置，并向 CLI 账号自动暴露 `*-search` 模型。
- **健康检测与故障转移**：带有队列限流、速率缓存、自动熔断/恢复、失败任务清理与多重保活机制。
- **DeepThink 多轮推理**：可选的链式思考流程，按需串联搜索、代码沙盒和分析步骤，自动决定迭代轮次（最多 7 轮）。
- **可视化控制台**：Streamlit 仪表盘整合使用率、密钥/账号管理、模型配置、日志查看、任务运行状态与一键 CLI 登录。
- **任务调度与持久化**：APScheduler 周期触发保活、健康巡检、自动清理等任务；SQLite（或自定义数据库）持久化配置信息、使用记录与授权凭据。
- **一键部署模板**：随仓库提供 `render.yaml`，在 Render Blueprint 中即可完成持续部署。

---

## 架构概览

```
┌───────────────────────┐
│  Streamlit 控制台 (UI) │  <---> 终端用户
└─────────────┬─────────┘
              │ 反向代理 (app.runtime.streamlit)
┌─────────────▼─────────┐
│   FastAPI 主服务        │
│  ├─ / (公开 API)       │
│  ├─ /admin (管理 API)  │
│  └─ /v1/* (OpenAI 兼容)│
└─────────────┬─────────┘
              │
      ┌───────▼─────────┐
      │ 服务子模块       │
      │  ├─ 账号池/队列  │
      │  ├─ CLI OAuth    │
      │  ├─ Failover     │
      │  └─ Scheduler    │
      └────────┬────────┘
               │
        ┌──────▼──────┐
        │ 持久化层    │  (SQLite / 外部数据库)
        └─────────────┘
```

所有组件运行在同一 Python 进程内：FastAPI 负责 API，Streamlit 子进程通过内置反向代理暴露，数据库写入采用异步队列串行化，任务调度器按配置启动/关闭。

---

## 快速开始

### 环境要求

- Python 3.10+
- 可访问外网（Gemini 接口、Google OAuth、DuckDuckGo 搜索等）
- （可选）Google Cloud CLI 账号用于 CLI OAuth

### 克隆与安装

```bash
git clone https://github.com/your-org/gemini-api-proxy.git
cd gemini-api-proxy
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# 或 source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 启动服务

```bash
# 默认监听 0.0.0.0:8000，Streamlit 控制台占用 127.0.0.1:7000
python -m app.runtime.server
```

启动成功后：

- 管理后台（Streamlit）：`http://127.0.0.1:8000/admin`
- OpenAI 兼容端点：`POST http://127.0.0.1:8000/v1/chat/completions`
- 原生 Gemini 端点：`POST http://127.0.0.1:8000/v1beta/models/{model}:generateContent`
- 健康检查：`GET http://127.0.0.1:8000/health`

首次运行会在仓库根目录生成 `gemini_proxy.db` 作为默认 SQLite 数据库。

---

## Render 部署指南

1. **准备仓库**  
   - 保留仓库根目录下的 `render.yaml`、`requirements.txt`、`app/` 与 `streamlit_app/`。
2. **创建 Blueprint 服务**  
   - 在 Render Dashboard 选择 *New ➝ Blueprint*，关联 Git 仓库。
   - `render.yaml` 会自动生成 Web 服务，默认 plan 为 `free`。
3. **构建与启动命令**  
   - 构建：`pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
   - 启动：`python -m app.runtime.server`
4. **配置环境变量（示例）**
   - `GEMINI_AUTH_PASSWORD`：控制台登录口令。
   - `ADMIN_AUTH_TOKEN`：OpenAI 兼容层基础认证（用户名固定为 `admin`）。
   - `RENDER_EXTERNAL_URL`：Render 自动注入，用于生成 CLI OAuth 回调地址。
   - `QUEUE_MAX_CONCURRENCY`、`QUEUE_MAX_WAIT_MS`：按机器性能微调。
5. **首登与 CLI OAuth**
   - 部署完成后访问 `<外网域名>/admin`，使用 `admin / GEMINI_AUTH_PASSWORD` 登录。
   - 在“密钥管理”页面发起 CLI 授权或导入 API Key。

> 建议为 `gemini_proxy.db` 配置持久卷，或迁移至外部数据库，以保留 CLI 凭证与统计数据。

---

## 环境变量速查

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `APP_NAME` | `gemini-api-proxy-beta` | 应用名称（影响日志与页面标题）。 |
| `ENVIRONMENT` | `production` | 环境标识，可用于自定义逻辑。 |
| `DEBUG` | `false` | 是否开启调试模式。 |
| `HOST` | `0.0.0.0` | FastAPI 监听地址。 |
| `PORT` | `8000` | FastAPI 监听端口。 |
| `RELOAD` | `false` | 是否开启自动重载。 |
| `ADMIN_AUTH_TOKEN` | `admin-secret` | OpenAI 兼容端点的 HTTP Basic 密码。 |
| `GEMINI_AUTH_PASSWORD` | `123456` | Streamlit 控制台登录密码。 |
| `DATABASE_URL` | `sqlite:///./proxy.db` | SQLAlchemy 数据库连接字符串。 |
| `SQLITE_PATH` | `./proxy.db` | 本地 SQLite 文件路径。 |
| `QUEUE_MAX_CONCURRENCY` | `4` | 每个模型的最大并发请求数。 |
| `QUEUE_MAX_WAIT_MS` | `30000` | 请求在队列中等待的超时时间（毫秒）。 |
| `QUEUE_POLL_INTERVAL_MS` | `50` | 队列轮询间隔。 |
| `DEFAULT_THINKING_BUDGET` | `-1` | 全局默认思考 token 预算，-1 代表按模型配置。 |
| `STREAMLIT_INTERNAL_PORT` | `7000` | Streamlit 子进程绑定端口。 |
| `API_BASE_URL` | (空) | 显式指定后端 Base URL，Streamlit 将使用该值调用 API。 |
| `STREAMLIT_BASE_URL` | (空) | Streamlit 对外访问地址，供自定义域名场景使用。 |
| `RENDER_EXTERNAL_URL` | (自动) | Render 注入的公网域名，CLI OAuth 回调会使用。 |
| `CLI_CLIENT_ID` | 官方 CLI Client ID | 自定义 Google OAuth 客户端 ID，启用线上授权时必填。 |
| `CLI_CLIENT_SECRET` | 官方 CLI Client Secret | 自定义 Google OAuth 客户端密钥，需与 `CLI_CLIENT_ID` 配套。 |
| `CLI_REDIRECT_BASE_URL` | (自动推断) | 强制指定 OAuth 回调基准域名，Render/自定义域建议显式设置。 |
| `ENABLE_KEEP_ALIVE` | `true`(Render)/`false`(其他) | 是否默认启用保活任务。 |
| `KEEP_ALIVE_INTERVAL` | `10` | 保活任务触发间隔（分钟）。 |
| `CORS_ORIGINS` | `*` | 允许的 CORS 来源列表，逗号分隔。 |

更多运行时配置（如 Failover、DeepThink、搜索策略等）可在控制台页面直接修改，无需更改环境变量。

---

## 在线 OAuth 授权教程（Google CLI 凭证）

若要在 Render、Hugging Face 等公网环境完成 Google OAuth 登录，必须使用自己创建的 OAuth Client（不能继续使用官方 `Gemini Code Assist and Gemini CLI` 默认凭证）。操作步骤如下：

1. **创建 OAuth 客户端**
   - 访问 *[Google Cloud Console → APIs & Services → Credentials](https://console.cloud.google.com/apis/credentials)*。
   - 选择 *Create Credentials → OAuth client ID*（[参考文档](https://developers.google.com/identity/protocols/oauth2/web-server#creatingcred)），将应用类型设置为 **Web application**。
   - 在 *Authorized redirect URIs* 中添加：
     - 线上域名：`https://<你的域名>/admin/cli-auth/callback`（Render 例：`https://<render-service>.onrender.com/admin/cli-auth/callback`）。
     - 可选：本地调试时保留 `http://127.0.0.1:8765/oauth2callback`、`http://localhost:8765/oauth2callback`（[Loopback 说明](https://developers.google.com/identity/protocols/oauth2/native-app#redirect-uri_loopback)）。
   - 保存后记录生成的 Client ID、Client Secret。

2. **配置部署环境变量**
   - `CLI_CLIENT_ID`：填入新建的 Client ID（可在 [Credentials 页面](https://console.cloud.google.com/apis/credentials) 查看）。
   - `CLI_CLIENT_SECRET`：填入 Client Secret。
   - `CLI_REDIRECT_BASE_URL`：你的公网域名（例：`https://<render-service>.onrender.com` 或自定义域）。如使用反向代理/CDN，需与最终访问域名一致（可参考 [Render 自定义域文档](https://render.com/docs/custom-domains)）。
   - Render 用户可保留 `RENDER_EXTERNAL_URL` 辅助推断，但显式设置 `CLI_REDIRECT_BASE_URL` 可避免头信息缺失导致的回调错误。

3. **重新部署并验证日志**
   - 重启服务后，查看日志中是否出现 `Starting CLI OAuth flow: base_url=...`。若仍显示 `http://127.0.0.1/...`，说明转发头或环境变量配置不正确（Render 日志位置：[Dashboard ➝ Logs](https://dashboard.render.com/)）。

4. **在控制台发起授权**
   - 登录 `/admin` 控制台 → “密钥管理 → Gemini CLI”。（默认面板地址示例：`https://<你的域名>`）
   - 点击“在线授权”，系统会返回 Google 登录链接并在新标签页打开。
   - 完成登录后，浏览器会跳转回 `.../admin/cli-auth/callback` 并提示成功。
   - 返回控制台，通常几秒内会看到新建的 CLI 账号；若状态未刷新可手动点击右上角刷新按钮。

5. **常见问题排查**
   - **redirect_uri_mismatch**：回调地址未加入 OAuth 客户端白名单（详见 [Google 错误说明](https://developers.google.com/identity/protocols/oauth2/web-server#redirect-uri_mismatch)），返回步骤 1 检查配置。
   - **仍跳到 localhost 回调**：转发头缺失或 `CLI_REDIRECT_BASE_URL` 未设置，请检查日志中 base_url 是否正确（Render 反向代理头参考：[官方文档](https://render.com/docs/configure-ssl#forwarded-headers)）。
   - **授权成功但未生成密钥**：确认数据库可写，或在“日志”页面查看 `cli-auth` 模块输出的异常信息（可在 `/admin` → “日志”或 Render Logs 中查看）。

完成上述配置后，即可在公网环境中完成 Google OAuth 登录并自动注入 CLI 账号池。

---

## 控制台使用指南

### 登录与总体状态
- 登录地址：`/admin`，用户名固定为 `admin`，密码由 `GEMINI_AUTH_PASSWORD` 控制。
- 仪表盘展示密钥状态、核心指标、近 24 小时请求量、失败率及最近日志。

### 密钥与账号管理
- **CLI 授权**：点击“Gemini CLI”卡片中的授权按钮即可生成 OAuth 链接，完成 Google 登录后自动持久化为 `cli-account-*`，并注入模型别名与限额。
- **API Key 导入**：支持批量粘贴或单条添加；配套健康检测、状态切换及一键删除异常密钥。
- **Failover / 自动清理**：可配置失败阈值、熔断策略、每日清理时段等。

### 模型与调用策略
- 为每个模型设置别名、上下游映射、RPM/RPD/TPM 限额、思考预算、流式策略。
- CLI 专用模型（如 `gemini-2.5-pro-search`）会自动映射到真实模型并注入搜索链路。

### DeepThink 与工具搜索
- 可在“高级功能”页启用 DeepThink，配置最大轮次、是否返回思考内容等。
- 搜索链路通过 DuckDuckGo 聚合网页摘要，与代码沙盒（受限 Python 环境、4 秒超时）协同工作。

### 系统与日志
- “系统信息”页展示 CPU/内存、运行时长、支持模型、Keep-Alive 状态等。
- “请求日志”支持分页查看近 200 条 API 调用明细；支持一键触发保活、健康检查。

---

## API 速览

| 类别 | 路径 | 描述 |
| :--- | :--- | :--- |
| 健康检查 | `GET /health` / `/healthz` / `/wake` | 基础健康、唤醒与 Render Keep-Alive 接口。 |
| Gemini 原生 | `POST /v1beta/models/{model}:generateContent` | 代理到 Gemini 官方接口。 |
| Gemini 原生 | `POST /v1beta/{model}:streamGenerateContent` | 流式生成。 |
| OpenAI 兼容 | `POST /v1/chat/completions` | Chat Completions，支持流式与非流式。 |
| OpenAI 兼容 | `GET /v1/models` | 列出可用模型（去除 `models/` 前缀）。 |
| 管理 | `GET /admin/stats` 等 | 仪表盘数据、任务配置、日志、密钥 CRUD、CLI 授权等。 |
| CLI OAuth | `POST /admin/cli-auth/start` | 生成授权链接并创建会话。 |
| CLI OAuth | `GET /admin/cli-auth/status` | 查询授权状态。 |
| CLI OAuth | `POST /admin/cli-auth/import` | 导入本地保存的 OAuth 凭证 JSON。 |

所有管理端口均需内部调用或受控前端访问，请勿直接暴露给未授权用户。

---

## 背景任务与并发控制

- **请求队列**：按模型名维护独立信号量，超过 `QUEUE_MAX_CONCURRENCY` 会进入等待队列；若超过 `QUEUE_MAX_WAIT_MS` 将返回 429。
- **APScheduler 任务**  
  - `keep_alive`：按 `KEEP_ALIVE_INTERVAL` 分钟触发 `/health` 等轻量请求，防止 Render 休眠。  
  - `hourly_health_check`：每小时统计密钥健康与使用率。  
  - `daily_cleanup`：每日 02:00 自动清理长时间异常的密钥。  
  - `daily_db_cleanup`：每日 03:00 整理数据库日志与历史记录。
- **数据库写队列**：`db_queue` 将使用日志等写操作放入后台任务，避免阻塞请求线程。

---

## 数据与持久化

- 默认使用位于项目根目录的 `gemini_proxy.db`，包含密钥表、CLI 凭证、使用日志、配置项等。
- 可通过 `DATABASE_URL` 切换至 PostgreSQL、MySQL 等外部数据库。
- CLI OAuth 凭证会以加密 JSON 存储，包含 `refresh_token`、`client_id`、`client_secret` 等字段。

---

## 安全与部署建议

- 修改 `GEMINI_AUTH_PASSWORD` 与 `ADMIN_AUTH_TOKEN`，避免使用默认值。
- Render 或其他云平台部署时，务必开启 HTTPS 并限制 `/admin` 访问来源。
- 为长期运行准备持久化存储，防止实例重建或缩容导致的授权丢失。
- 在日志或第三方监控中避免暴露实际密钥，可使用项目内的 `mask_key` 工具函数。

---

## 常见问题

1. **CLI OAuth 在云端失败**  
   - 确认 `RENDER_EXTERNAL_URL` 或自定义的 `API_BASE_URL`、`STREAMLIT_BASE_URL` 指向公网 HTTPS 地址。  
   - Google OAuth 不接受纯 HTTP 回调，需使用 Render 提供的正式域名或手动配置反向代理。
2. **OpenAI 兼容接口报 401**  
   - 端点使用 HTTP Basic 认证：用户名固定为 `admin`，密码取自 `ADMIN_AUTH_TOKEN`。
3. **请求频繁返回 429**  
   - 可能是队列饱和或模型限额达到上限。检查控制台使用率与 `QUEUE_MAX_CONCURRENCY` 设置。
4. **Streamlit 控制台打不开**  
   - 查看服务器日志，确认 `streamlit` 子进程已启动且 `STREAMLIT_INTERNAL_PORT` 未被占用。

---

## 目录结构

```
├── app/
│   ├── admin/          # 管理 API、数据库层、CLI OAuth、配置服务
│   ├── core/           # 全局设置与缓存
│   ├── proxy/          # Gemini/OpenAI 路由与适配
│   ├── runtime/        # 进程启动、Streamlit 代理、守护逻辑
│   ├── services/       # 队列、工具函数等
│   └── server.py       # FastAPI 入口
├── streamlit_app/      # 控制台页面、样式与工具函数
├── requirements.txt    # 依赖列表
├── render.yaml         # Render 部署模板
└── README.md
```

---

## 开发与测试

- 代码风格遵循项目现有约定，提交前建议运行 `ruff` 或 `black`（仓库未强制）。
- 可使用 `uvicorn app.server:app --reload` 进行热重载开发。
- 项目当前未包含自动化测试，可根据需要在 `tests/` 目录中补充。

---

## 贡献

欢迎提交 Issue 或 Pull Request，协助完善 CLI 体验、更多模型映射、自动化测试以及运维工具链。
