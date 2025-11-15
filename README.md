
<div align="center">

![:name](https://count.getloli.com/@astrbot_plugin_gemini_search?name=astrbot_plugin_gemini_search&theme=minecraft&padding=6&offset=0&align=top&scale=1&pixelated=1&darkmode=auto)

# astrbot_plugin_gemini_search

_✨ [astrbot](https://github.com/AstrBotDevs/AstrBot) 一个调用 Gemini 格式 API 进行联网搜索的函数工具 ✨_  

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![AstrBot](https://img.shields.io/badge/AstrBot-3.4%2B-orange.svg)](https://github.com/Soulter/AstrBot)
[![GitHub](https://img.shields.io/badge/作者-Chris-blue)](https://github.com/Chris95743)

</div>

# Gemini Search 函数工具

强力的联网检索与网页实用工具：基于 Google Generative AI（Gemini 2.0 Flash）与截图服务（默认 screenshotsnap.com），既可启用原生 Google Search 工具进行实时搜索，也可抓取网页文本、对网页截图进行分析或直接把截图发给用户。搜索结果提炼为“要点摘要 + 引用来源（标题+URL）”，并作为函数工具输出注入 AstrBot 对话，让主模型在同一轮对话中即可消化并回答。

## 功能特性
- 原生 Google Search 工具接入：无需自建爬虫，检索能力由 Gemini 托管
- 结果结构化：要点式摘要 + 参考来源列表（标题 + URL）
- 多 API Key 负载分摊：支持随机或轮询挑选 Key
- 可配置的失败重试：开关、次数、时间间隔，稳中求稳
- 即插即用：作为 LLM 函数工具自动启用（名称：`gemini_search`）
 - 新增网页工具：
	 - `web_fetch` 抓取网页纯文本内容
	 - `webshot_analyze` 截图并由 Gemini 对截图进行分析
	 - `webshot_send` 截图并直接发送给用户，可选开启 AI 审核

## 函数工具一览
- `gemini_search(query: string)`：联网检索并返回“要点摘要 + 来源列表”。
- `web_fetch(url: string)`：抓取网页纯文本（剔除 script/style），返回文本（可配置长度上限）。
- `webshot_analyze(url: string, prompt?: string)`：对网页截图并将图片交给 Gemini 分析，返回结构化文本。
- `webshot_send(url: string)`：对网页截图并直接发送图片给用户；若开启 `moderation_before_image_send`，将先由 AI 审核（BLOCK 则不发送）。

## 安装与前置
请确保你已按 AstrBot 项目文档完成环境准备：
### 方式一：插件市场安装（推荐）
1. 在AstrBot插件市场搜索 `astrbot_plugin_gemini_search`
2. 点击安装，等待安装完成
3. 重启AstrBot即可使用

### 方式二：手动克隆安装
```bash
# 进入插件目录
cd /AstrBot/data/plugins

# 克隆仓库
git clone https://github.com/Chris95743/astrbot_plugin_gemini_search

# 重启AstrBot
```

## 快速开始（3 步）
1) 打开 Dashboard → 插件 → 选择 `astrbot_plugin_gemini_search`
2) 配置以下选项（详见下节“配置项详解”），至少填写一个可用的 Google API Key
3) 回到会话，让模型尝试需要实时信息的问题，Agent 会在需要时自动调用 `gemini_search`

可显式提示模型调用：
- “请联网搜索并给出要点摘要与参考链接”
- “调用 gemini_search 查询：<你的问题>”

## 配置项详解
- api_key（list）
	- 一个或多个 Google API Key，支持轮询/随机选择
- api_base_url（string）
	- Gemini API 基础地址，默认 `https://generativelanguage.googleapis.com`
- model（string）
	- 例如 `gemini-2.0-flash`（默认），可按需替换为其他可用变体
- random_api_key_selection（bool）
	- 是否随机使用 Key，关闭则按顺序轮询
- 以下功能将在另一个分支实现。
- retry_on_error（bool）
	- 出错时是否自动重试（默认开启）
- retry_count（int）
	- 最大重试次数，不含首次调用（默认 2）
- retry_interval_seconds（float）
	- 两次调用之间的等待秒数（默认 2.0）

新增：截图与抓取相关
- screenshot_api_base（string）默认 `https://screenshotsnap.com/api/screenshot`
- screenshot_format（string）`webp|png`，默认 `webp`
- screenshot_width（int）默认 1920
- screenshot_height（int）默认 1080
- moderation_before_image_send（bool）发送截图前是否先由 AI 审核，默认 false
- fetch_timeout_seconds（float）网络请求超时，默认 20
- fetch_user_agent（string）抓取网页 UA，默认 `Mozilla/5.0 AstrBot`
- fetch_max_chars（int）`web_fetch` 最大返回字符数，默认 20000

## 依赖
- `google-genai>=0.4.0`
- `httpx>=0.27.0`
- `beautifulsoup4>=4.12.0`
这些依赖已在插件 `requirements.txt` 中声明。

## 工作原理（简述）
1) 插件注册 `@llm_tool("gemini_search")` 函数工具，并在插件初始化时激活
2) 当对话需要实时检索时，Agent 调用该工具
3) 工具通过 Google GenAI SDK 调用 Gemini 2.0 Flash，并启用原生 `GoogleSearch` 工具进行检索
4) 模型返回要点摘要与来源列表；插件将文本作为工具结果注入会话
5) 主模型据此继续组织最终回答（AstrBot 的标准 Tool-Loop 流程）

## 使用示例（对话思路）
- 用户：今年某领域的最新进展？给出要点摘要并附参考链接
- 模型（内部）：调用 `gemini_search` 工具 -> 获取摘要与来源 -> 基于结果生成最终答复
- 可以在主函数中按需求修改以下内容。使搜索返回内容不包括url或者返回内容更详细。
```main
prompt = (
	"你是检索聚合助手。请使用 Google Search 工具对下述问题进行检索，"
	"产出包含：\n"
	"1) 关键要点的条目式摘要；\n"
	"2) 参考来源的列表（标题 + URL）。\n"
	"请避免冗长描述，直接给出结论与可靠来源。\n"
	"问题：" + query
)
```

### 新增工具示例
- 抓取网页文本：
	- “调用 web_fetch：https://example.com/article”
- 截图并分析：
	- “调用 webshot_analyze：https://example.com/news，分析要点与风险提示”
- 截图并发送（如开启审核，会先审再发）：
	- “调用 webshot_send：https://www.baidu.com”

## 截图API（默认）
默认使用 `screenshotsnap.com`：
- 接口：`GET /api/screenshot?url=<url>&format=webp|png&width=1920&height=1080`
- 可通过配置项 `screenshot_api_base`、`screenshot_format`、`screenshot_width`、`screenshot_height` 进行覆盖。

## 安全与合规
- 开启 `moderation_before_image_send` 后，截图会先交由 Gemini 做合规判定，只返回 `ALLOW|BLOCK`。若为 `BLOCK`，插件将拦截发送并提示用户。
- 抓取和截图会访问第三方服务，请遵循目标站点与服务条款，不要提交敏感隐私数据。
  
## 常见问题（FAQ）
Q1: 一定要用官方域名吗？
- 不一定，可在 `api_base_url` 填写你的反代地址，但需确保兼容 `google-genai` SDK 的 API 约定

Q2: 工具没有被调用？
- 确保当前会话使用的模型/Provider 支持函数工具
- 确保插件已加载且在初始化时成功激活了 `gemini_search`

## 隐私与合规
- 工具会把你的查询交给 Google 的生成式 AI 服务处理；请勿输入敏感/个人隐私信息
- 遵循目标站点的访问与使用条款；引用来源请规范标注

## 版本与兼容
- 默认模型：`gemini-2.0-flash`
- 需要：`google-genai>=0.4.0`
- AstrBot：3.4+（参考主仓库）

## 贡献 & 反馈
- Issue/PR：见 `metadata.yaml` 中的 repo 链接
- 欢迎提交改进建议：比如支持更多模型参数、结果去重/合并、引用质量提升等

---

<div align="center">

**感谢使用 astrbot_plugin_gemini_search！**

如果觉得有用，请给个 ⭐ Star 支持一下！

</div>



