# Gemini API Proxy 2.0

Gemini API Proxy 2.0 是一个面向 Render / 本地单实例部署的 Gemini 反向代理与运维平台。项目将 FastAPI、Streamlit、任务调度及多账号池管理整合在同一进程中，既能兼容原生 Gemini/Google Generative AI 接口，也提供 OpenAI 格式的 `/v1/chat/completions` 端点与图形化控制台。

---

## 核心特性

- **双账号池调度**：支持导入 Gemini CLI 凭证与普通 API Key，按优先级与健康度自动分配请求。
- **OpenAI 兼容层**：兼容大部分 Chat Completions 能力，含流式响应、模型列表等，便于接入现有 SDK。
- **模型映射与限额管理**：支持别名、RPM/RPD/TPM 限额、思考预算、流式模式等细粒度配置，并向 CLI 账号自动暴露 `*-search` 模型。
- **健康检测与故障转移**：带有队列限流、速率缓存、自动熔断/恢复、失败任务清理与多重保活机制。
- **DeepThink 多轮推理**：可选的链式思考流程，按需串联搜索、代码沙盒和分析步骤，自动决定迭代轮次（最多 7 轮）。
- **可视化控制台**：Streamlit 仪表盘整合使用率、密钥/账号管理、模型配置、日志查看、任务运行状态与 CLI 凭证导入。
- **任务调度与持久化**：APScheduler 周期触发保活、健康巡检、自动清理等任务；SQLite（或自定义数据库）持久化配置信息、使用记录与授权凭据。
- **一键部署模板**：随仓库提供 `render.yaml`，在 Render Blueprint 中即可完成持续部署。


## Render 部署指南

1. **准备仓库**  
   - 保留仓库根目录下的 `render.yaml`、`requirements.txt`、`app/` 与 `streamlit_app/`。
2. **创建 Blueprint 服务**  
   - 在 Render Dashboard 选择 *New ➝ Blueprint*，关联 Git 仓库。
   - `render.yaml` 会自动生成 Web 服务，默认 plan 为 `free`。
3. **构建与启动命令**  
   - 构建：`pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
   - 启动：`python -m app.runtime.server`
4. **首登与 CLI 凭证导入**
   - 部署完成后点击页面中的 https://gemini-api-proxy-xx.onrender.com 即可访问。
   - 在“密钥管理”页面点击“导入 CLI 凭证”，粘贴本地 `gemini-cli login` 导出或在线获取的 JSON；成功后即可沿用官方 CLI 的免费额度。

### 获取并导入 gemini-cli 凭证

要沿用官方 CLI 的免费额度，需要先取得一份有效的 CLI OAuth 凭证（JSON），然后在控制台导入。可以选择本地 CLI 或线上工具：

**方式 A：本地 `gemini-cli` 登录**

1. 安装 CLI：`npm install -g @google/geminicli`。
2. 执行 `gemini-cli login`，按提示完成浏览器授权。
3. 登录成功后，终端会输出 JSON 凭证（同时写入 `~/.config/geminicli/`），包含 `client_id`、`client_secret`、`refresh_token`、`token_uri`、`project_id` 等。

**方式 B：在线获取（免本地安装）**

我们通过 sukaka 老师搭建的认证网站来获取凭证。通过这个方法获取凭证，整个的操作步骤几乎被简化到极致。sukaka 老师，伟大无需多言。

1. 访问认证网站：[http://gcli-auth.sukaka.top:7861/](http://gcli-auth.sukaka.top:7861/)。
2. 输入访问密码：`pwd`，即可进入 Google OAuth 认证页面。
3. **获取认证链接**：
   - 在页面中，你可以直接点击“获取认证链接”按钮，来获取认证链接。
   - 也可以点击“高级选项”部分输入你的项目 ID（如果你不知道是什么，就不要输入，让系统自己检测）。
4. **Google 账号认证**：
   - 点击“获取认证链接”会出现一个很长的链接，点击打开。
   - 打开后进入 Google 账号登录页面，按提示完成登录。
   - 登录完成后会出现一个错误页面。此时在浏览器地址栏将 `localhost` 更改为 `gcli-auth.sukaka.top`（仅替换主机名，其余端口和参数保持不变），按回车访问。
   - 页面显示 “OAuth authentication successful!” 即表示认证成功，可直接关闭该页面。
5. **获取认证文件**：
   - 回到认证网站，点击“获取认证文件”，稍候会显示“步骤二：认证成功”。
   - 点击“下载认证文件”获取 JSON 凭证，也可直接复制页面中展示的认证内容。

**导入到控制台**

1. 打开 “密钥管理 → Gemini CLI” → “导入 CLI 凭证”。
2. 将上述获取的 JSON 原样粘贴提交。
3. 系统会创建 `cli-account-*` 记录并保存 refresh token，后续请求将继续使用官方 CLI 的 OAuth 客户端，从而继承免费额度。

> 默认限额（按单个 CLI 账号计算）：`gemini-2.5-pro` 每日 100 次；`gemini-2.5-pro-preview-*` 合计每日 1000 次；`gemini-2.5-flash` 每日 100 次，`gemini-2.5-flash-preview-*` 合计每日 1000 次。

**维护与撤销**

- 若凭证失效，可重新获取 JSON（本地或线上方式）再导入。
- 如需撤销访问，可在 Google 账号的“安全性 → 第三方应用访问权限”中移除 `Gemini Code Assist and Gemini CLI`，然后在控制台删除对应账号。


## API 速览

| 类别 | 路径 | 描述 |
| :--- | :--- | :--- |
| 健康检查 | `GET /health` / `/healthz` / `/wake` | 基础健康、唤醒与 Render Keep-Alive 接口。 |
| Gemini 原生 | `POST /v1beta/models/{model}:generateContent` | 代理到 Gemini 官方接口。 |
| Gemini 原生 | `POST /v1beta/{model}:streamGenerateContent` | 流式生成。 |
| OpenAI 兼容 | `POST /v1/chat/completions` | Chat Completions，支持流式与非流式。 |
| OpenAI 兼容 | `GET /v1/models` | 列出可用模型（去除 `models/` 前缀）。 |
| 管理 | `GET /admin/stats` 等 | 仪表盘数据、任务配置、日志、密钥 CRUD、CLI 凭证导入等。 |
| CLI 凭证 | `POST /admin/cli-auth/import` | 导入本地保存的 gemini-cli OAuth 凭证 JSON。 |

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
- CLI 凭证会以加密 JSON 存储，包含 `refresh_token`、`client_id`、`client_secret` 等字段。

---

## 安全与部署建议

- 修改 `GEMINI_AUTH_PASSWORD` 与 `ADMIN_AUTH_TOKEN`，避免使用默认值。
- Render 或其他云平台部署时，务必开启 HTTPS 并限制 `/admin` 访问来源。
- 为长期运行准备持久化存储，防止实例重建或缩容导致的授权丢失。
- 在日志或第三方监控中避免暴露实际密钥，可使用项目内的 `mask_key` 工具函数。


## 目录结构

```
├── app/
│   ├── admin/          # 管理 API、数据库层、CLI 凭证、配置服务
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

- 代码风格遵循项目现有约定，提交前建议运行 `ruff` 或 `black`。
- 可使用 `uvicorn app.server:app --reload` 进行热重载开发。
- 项目当前未包含自动化测试，可根据需要在 `tests/` 目录中补充。

---

## 贡献

欢迎提交 Issue 或 Pull Request，协助完善 CLI 体验、更多模型映射、自动化测试以及运维工具链。
