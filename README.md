# Gemini API 轮询代理服务

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

一个专为**Gemini多Key轮询**设计的 Gemini API 代理服务，通过智能轮询多个 API Key 突破单Key限制，提供 OpenAI 兼容接口和完整的管理界面。无需服务器，一键免费部署到 Render 平台，立即获得公网访问地址。


## ✨ 核心特性

- 🔄 **智能轮询与自适应策略**：多个 Gemini API Key 自动轮询，支持轮询、最少使用和自适应策略，突破单Key请求限制。
- 🛡️ **高可用性与快速故障转移**：单个Key失效不影响服务，自动进行健康检测并实现零延迟故障转移。
- 🚀 **一键部署**：Fork 仓库后直接在 Render 部署，10分钟获得公网地址。
- 💰 **完全免费**：使用 Render 免费层，无需支付内网穿透或映射服务器费用。
- 🎨 **现代管理界面**：采用玻璃拟态设计的 Streamlit 直观管理界面，完美适配移动端。
- ⚡ **高性能**：FastAPI + 异步处理，支持可控的流式与非流式响应。
- 🔐 **安全可靠**：用户密钥管理、使用统计、速率限制，并内置反自动化检测机制。
- 🖼️ **全面多模态支持**：原生支持 Gemini 2.5 的图片、音频、视频、文档处理能力，并提供文件上传API。
- 🧹 **智能维护**：自动监控API Key健康状态，并可配置自动清理异常Key。

## 🎯 快速开始

### 1. Fork 本仓库

点击页面右上角的 **Fork** 按钮，将本项目复制到你的 GitHub 账户下。

### 2. 一键部署前后端服务

#### 2.1 创建 Render 账户
1. 访问 [Render.com](https://render.com)
2. 使用 GitHub 账户登录（推荐）或邮箱注册
3. **需要MasterCard或VisaCard验证身份！但完全免费**

#### 2.2 一键部署 Blueprint
1. 在 Render 控制台点击 **"New +"** → **"Blueprint"**
2. 选择 **"Connect a repository"**
3. 找到你刚刚 Fork 的 `gemini-api-proxy` 仓库，点击 **"Connect"**
4. 配置 Blueprint 参数：

```yaml
Name: gemini-api-services  # 自定义Blueprint名称
Branch: main
```

5. Render 会自动识别 `render.yaml` 文件并显示将要创建的服务：
   - ✅ **gemini-api-proxy** (后端API服务)
   - ✅ **gemini-proxy-admin** (前端管理界面)

6. 点击 **"Apply"** 开始部署

#### 2.3 等待部署完成
- ⏱️ **首次部署时间**：约5-10分钟
- 📊 **部署进度**：可在Dashboard中实时查看两个服务的构建状态
- ✅ **完成标志**：两个服务都显示绿色的"Live"状态

### 3. 配置服务连接

由于Render免费层的限制，需要手动配置前后端连接：

#### 3.1 获取后端地址
1. 在Render Dashboard中找到 **gemini-api-proxy** 服务
2. 复制其完整URL，格式类似：`https://gemini-api-proxy-xxx.onrender.com`

#### 3.2 配置前端环境变量
1. 点击进入 **gemini-proxy-admin** 服务
2. 转到 **"Environment"** 标签页
3. 添加环境变量：
   ```
   Key: API_BASE_URL
   Value: https://gemini-api-proxy-xxx.onrender.com
   ```
4. 点击 **"Save Changes"**
5. 前端会自动重新部署（约2-3分钟）

#### 3.3 访问管理界面
访问 `https://gemini-proxy-admin-xxx.onrender.com`

你将看到 Streamlit 管理界面，现在可以开始配置 API 密钥了！

---

## ⚠️ 常见问题

**Q: 为什么不能完全自动连接前后端？**
A: Render免费层的Blueprint功能在服务引用方面有限制，无法自动构建完整的HTTPS URL。

**Q: 如果前端无法连接后端怎么办？**
A: 检查以下几点：
1. 后端服务是否正常运行（访问/health端点）
2. API_BASE_URL环境变量是否设置正确
3. URL格式是否包含https://前缀

**Q: 可以自定义服务名称吗？**
A: 可以！修改render.yaml中的name字段，然后重新同步Blueprint。

## 🔧 配置指南

### 1. 添加多个 Gemini API Key

轮询的核心是配置多个API Key，建议至少添加3-5个Key：

1. 访问前端管理界面
2. 进入 **"密钥管理"** → **"Gemini 密钥"** 页面
3. **逐个添加**多个 Gemini API Key：

```
Key 1: AIzaSyXXXXXXXXXXXXXXXXXXXXXX
Key 2: AIzaSyYYYYYYYYYYYYYYYYYYYYYY  
Key 3: AIzaSyZZZZZZZZZZZZZZZZZZZZZZ
... 更多Key
```

4. API Key 获取方式：
   - 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
   - 登录并创建新的 API Key
   - 复制密钥（格式：`AIzaSy...`）
   - **建议创建多个项目，每个项目生成一个Key**

**💡 配置建议：**
- ✅ **至少3个Key**：保证基本的轮询效果
- ✅ **5-10个Key**：获得更高的请求限制和稳定性
- ✅ **不同项目的Key**：降低同时被限制的风险

### 2. 配置轮询策略

1. 进入 **"系统设置"** → **"轮询配置"** 页面
2. 选择轮询策略：
   - **Round Robin**：按顺序轮询，负载分配均匀
   - **Least Used（推荐）**：智能选择使用量最少的Key
3. 设置故障转移：自动跳过失效的Key

### 3. 生成用户访问密钥

1. 在管理界面进入 **"密钥管理"** → **"用户密钥"** 页面
2. 点击 **"生成新密钥"**，输入密钥名称
3. **立即保存生成的密钥**（格式：`sk-...`），它不会再次显示

### 4. 配置思考模式

1. 进入 **"系统设置"** → **"思考模式"** 页面
2. 启用思考模式以获得更好的推理能力
3. 选择合适的思考预算：
   - **自动**：让模型自动决定
   - **2.5flash最大思考预算 (24k)**：快速响应
   - **2.5pro最大思考预算 (32k)**：深度思考

### 5. 在 Render 上完成 Gemini CLI 登录

Gemini CLI 登录支持两种模式：本地 loopback（默认）与远程回调。若你按照 `render.yaml` 同时部署了 FastAPI 后端和 Streamlit 管理台，只需完成下列配置即可自动启用远程模式：

1. 在 **前端（Streamlit）服务** 的环境变量中设置 `API_BASE_URL=https://<你的后端服务>.onrender.com`。
2. 后端服务会依次检查 `GEMINI_CLI_CALLBACK_URL`、`GEMINI_CLI_CALLBACK_BASE_URL`、Render 注入的 `RENDER_EXTERNAL_URL`，最后回退到 `API_BASE_URL`。只要该地址不是 `localhost`，`CliAuthManager` 会判定为云端部署并生成 `https://<后端>/admin/cli-auth/callback` 作为 Google OAuth 回调地址。【F:cli_auth.py†L298-L329】
3. 管理台的「通过 Google 登录」卡片会根据返回的 `mode` 显示“云端回调模式”，同时在提示中展示检测到的 `API_BASE_URL` 与具体回调 URL，方便确认配置是否生效。【F:app_pages.py†L360-L417】

完成授权后，Google 会直接重定向至后端的 `/admin/cli-auth/callback`，服务会保存授权结果并自动为号池创建新的 CLI 账号。你只需返回管理台点击“刷新授权状态”即可看到新账号同步成功。

## 📡 使用 API

配置完成后，你就可以使用 OpenAI SDK 访问轮询代理了。系统会自动在多个 Gemini Key 之间进行轮询，提供更高的请求限制和稳定性。

### 🔌 API 端点

#### 文件上传 API

```bash
# 上传文件
curl -X POST "http://localhost:8000/v1/files" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@image.jpg"

# 响应示例
{
  "id": "file-abc123",
  "object": "file",
  "bytes": 1024000,
  "created_at": 1703123456,
  "filename": "image.jpg",
  "purpose": "multimodal"
}
```

#### 多模态对话 API

```python
import openai

client = openai.OpenAI(
    api_key="YOUR_API_KEY",
    base_url="http://localhost:8000/v1"
)

# 文本 + 图片对话
response = client.chat.completions.create(
    model="gemini-2.5-flash-lite",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "这张图片里有什么？"
                },
                {
                    "type": "image",
                    "file_data": {
                        "file_id": "file-abc123",
                        "mime_type": "image/jpeg"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

#### 支持的文件类型

| 类型 | 支持格式 | 最大大小 | 用途 |
|------|----------|----------|------|
| 图片 | PNG, JPG, JPEG, GIF, WebP | 100MB | 图像识别、分析、描述 |
| 音频 | MP3, WAV, M4A, FLAC | 100MB | 语音转文字、音频分析 |
| 视频 | MP4, AVI, MOV, WebM | 100MB | 视频内容分析、帧提取 |
| 文档 | PDF | 100MB | 文档解析、内容提取 |

### 💬 加入交流群

遇到问题或想要交流经验？欢迎加入我们的QQ群：

[![QQ群：1055274821](https://img.shields.io/badge/QQ群-1055274821-blue?style=for-the-badge&logo=tencent-qq)](https://qm.qq.com/cgi-bin/qm/qr?k=&jump_from=webapi&authKey=&join_group=1055274821)

## 🎛️ 管理功能

### 控制台
- 📊 **轮询效率监控**：实时查看每个Key的使用分布
- 📈 **聚合请求统计**：所有Key的总体请求量图表
- 🔍 **Key状态分析**：快速识别失效或过载的Key
- 💹 **限制倍增显示**：直观展示轮询带来的限制提升

### 密钥管理
- 🔑 **多Key轮询管理**：批量添加、启用/禁用Gemini API Key
- 🔄 **轮询状态监控**：查看每个Key的轮询参与状态  
- 👤 **用户访问密钥生成**：为客户端生成访问令牌
- 🚦 **智能故障检测**：自动识别并跳过失效Key

### 轮询配置
- ⚖️ **负载均衡策略**：Round Robin / Least Used 策略切换
- 🎯 **Key权重设置**：为不同Key设置不同的使用权重
- 🛡️ **故障转移配置**：设置Key失效时的自动切换策略
- 📊 **使用率阈值**：单Key使用率预警和保护

### 模型配置
- ⚙️ **单Key限制设置**：配置每个Key的RPM/TPM限制
- 📊 **总限制计算**：自动计算轮询后的总体限制  
- 🔧 **模型状态管理**：启用/禁用特定模型的轮询

### 系统设置
- 🧠 思考模式配置
- 📝 提示词注入
- ✂️ 防截断功能开关
- 📋 系统状态监控

## 🆓 Render 免费层说明

### 免费额度
- ⏰ **运行时间**：每月 750 小时
- 📶 **带宽**：每月 100GB 出站流量
- 💾 **数据库**：1GB PostgreSQL（90天）
- 🌍 **域名**：免费 `.onrender.com` 子域名
- 🔒 **HTTPS**：自动 SSL 证书

### 限制说明
- 🛌 **休眠机制**：15分钟无请求后自动休眠
- ⚡ **冷启动**：休眠后首次请求需要15-30秒唤醒
- 🔄 **自动重启**：系统可能随时重启服务



## 🌐 自定义域名（可选）

Render 免费层支持自定义域名：

1. 在服务设置中添加自定义域名
2. 在域名提供商处添加 CNAME 记录
3. Render 自动提供 SSL 证书

## 🔧 高级配置

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `API_BASE_URL` | 后端API地址 | - |
| `PORT` | 服务端口 | 自动分配 |
| `PYTHONUNBUFFERED` | Python输出缓冲 | 1 |

## 🚨 注意事项

### 轮询最佳实践
- 🔑 **Key数量建议**：3-10个Key为最佳，过多Key管理复杂
- 🌍 **Key来源分散**：使用不同Google账号和项目创建Key
- 📊 **监控使用率**：定期检查Key使用分布，确保轮询效果
- 🔄 **及时替换**：发现失效Key立即替换，保持轮询池健康
- ⚖️ **策略选择**：高并发场景用Round Robin，日常使用Least Used

### 安全提醒
- 🔐 用户密钥仅显示一次，请立即保存
- 🚫 不要在客户端直接使用 Gemini API Key
- 🔒 定期轮换所有 API 密钥
- 🛡️ 多Key轮询降低单点安全风险

### 常见问题

**Q: 需要多少个Key才有轮询效果？**
A: 至少2个Key，推荐3-5个Key获得最佳平衡。

**Q: 某个Key失效了怎么办？**
A: 系统自动跳过失效Key，继续使用其他Key，服务不中断。

**Q: 如何知道轮询是否在工作？**
A: 在管理界面可以看到每个Key的使用分布和轮询状态。

**Q: 服务访问很慢怎么办？**
A: 这是 Render 免费层的冷启动特性，等待15-30秒即可恢复正常。

**Q: 如何避免服务休眠？**
A: 服务可能会因为无请求而休眠，首次访问需要等待唤醒。

**Q: 可以商用吗？**
A: 本项目采用 CC BY-NC 4.0 许可证，仅允许非商业使用。

## 🛠️ 本地开发

### 环境要求
- Python 3.8+
- pip

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动后端
```bash
python run_server.py
```

### 启动前端
```bash
streamlit run main.py
```

## 未来优化方向
API Key 加密存储 - 当前明文存储存在一定安全风险

数据持久化方案 - 解决 Render 重启丢失数据问题

自动 Key 健康检测 - 检查到 Key 连续三天失效自动移至移除区

优化 Key 使用策略 - 自适应负载均衡，选择最优配置

**确保部署保证简单与免费的特性**

## 📄 许可证

本项目采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可证。

- ✅ 允许：分享、修改、分发
- ❌ 禁止：商业使用
- 📝 要求：署名原作者

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request


## ⭐ 如果这个项目对你有帮助，请给个 Star ⭐️

推一推另一个好玩的App！https://github.com/Arain119/ChatApp ，超强的角色扮演能力~

## 🙏 致谢

- [Google Gemini](https://deepmind.google/technologies/gemini/) - 强大的AI模型
- [Render](https://render.com) - 优秀的免费部署平台
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Python Web框架
- [Streamlit](https://streamlit.io/) - 快速构建数据应用
