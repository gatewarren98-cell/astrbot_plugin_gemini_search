import random
import asyncio
from typing import Optional
from urllib.parse import urlencode

import astrbot.api.star as star
from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.message_event_result import MessageChain

# Google GenAI SDK
from google import genai
from google.genai import types

# HTTP & HTML parsing
try:
	import httpx
except Exception:  # pragma: no cover - 延迟导入失败时给出友好提示
	httpx = None

try:
	from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
	BeautifulSoup = None


class Main(star.Star):
	"""
	使用 Gemini 2.0 Flash + Google Search 原生工具进行联网检索。
	在函数工具中执行检索与汇总，返回结果文本，框架会把该文本作为 tool 消息注入当前对话，
	供主 LLM 后续综合使用。
	"""

	def __init__(self, context: star.Context, config=None) -> None:
		self.context = context
		# AstrBot 会根据 _conf_schema.json 构造 config（AstrBotConfig），此处按 dict 访问
		self.config = config or {}
		self._rr_index = 0  # 轮询下标
		self._clients: dict[str, genai.Client] = {}

	async def initialize(self):
		# 默认启用工具
		self.context.activate_llm_tool("gemini_search")
		self.context.activate_llm_tool("web_fetch")
		self.context.activate_llm_tool("webshot_analyze")
		self.context.activate_llm_tool("webshot_send")
		logger.info("[gemini_search] 函数工具已启用")
		logger.info("[web_fetch] 函数工具已启用")
		logger.info("[webshot_analyze] 函数工具已启用")
		logger.info("[webshot_send] 函数工具已启用")

	@llm_tool("gemini_search")
	async def gemini_search(self, event: AstrMessageEvent, query: str) -> str:
		"""这是一个“联网搜索”的函数工具（工具名：gemini_search）。当需要获取互联网上的实时/最新信息时，你必须调用本工具进行搜索。

		Args:
			query(string): 简要说明用户希望检索的查询内容

		Returns:
			str: 要点摘要与引用来源，作为 tool 消息注入上下文
		"""
		try:
			client = self._get_client()
		except Exception as e:
			logger.error(f"[gemini_search] 初始化客户端失败: {e}")
			return "请先在插件配置中填写有效的 Gemini API Key。"

		model = self.config.get("model", "gemini-2.0-flash")

		# 启用原生 Google Search 工具
		config = types.GenerateContentConfig(
			tools=[types.Tool(google_search=types.GoogleSearch())],
			temperature=0.2,
		)

		prompt = (
			"你是检索聚合助手。请使用 Google Search 工具对下述问题进行检索，"
			"产出包含：\n"
			"1) 关键要点的条目式摘要；\n"
			"2) 参考来源的列表（标题 + URL）。\n"
			"请避免冗长描述，直接给出结论与可靠来源。\n"
			"问题：" + query
		)

		try:
			resp = await client.models.generate_content(
				model=model,
				contents=prompt,
				config=config,
			)
			text = getattr(resp, "text", None) or self._extract_text(resp)
			return text.strip() if text else "未从检索中获得可用文本结果。"
		except Exception as e:
			logger.error(f"[gemini_search] 调用失败: {e}")
			return f"检索失败：{e}"

	@llm_tool("web_fetch")
	async def web_fetch(self, event: AstrMessageEvent, url: str) -> str:
		"""抓取网页文本内容（去标签纯文本）。

		Args:
			url(string): 需要抓取内容的网页 URL

		Returns:
			str: 提取的纯文本（前 20,000 字符内），失败时返回错误信息
		"""
		if httpx is None:
			return "插件缺少依赖 httpx，请在该插件 requirements.txt 中安装后重启。"
		try:
			text = await self._fetch_page_text(url)
			if not text:
				return "未能从页面中提取到有效文本。"
			max_chars = int(self.config.get("fetch_max_chars", 20000))
			return text[:max_chars]
		except Exception as e:
			logger.error(f"[web_fetch] 抓取失败: {e}")
			return f"抓取失败：{e}"

	@llm_tool("webshot_analyze")
	async def webshot_analyze(self, event: AstrMessageEvent, url: str, prompt: str = "请根据网页截图进行要点提炼与关键信息抽取，输出条目式结论与可操作建议。") -> str:
		"""对网页进行截图，并将截图发送给 Gemini 进行分析，返回结构化文本。

		Args:
			url(string): 网页 URL
			prompt(string): 对截图的分析提示词（可选）

		Returns:
			str: 模型对截图的分析文本
		"""
		if httpx is None:
			return "插件缺少依赖 httpx，请在该插件 requirements.txt 中安装后重启。"
		try:
			client = self._get_client()
		except Exception as e:
			logger.error(f"[webshot_analyze] 初始化客户端失败: {e}")
			return "请先在插件配置中填写有效的 Gemini API Key。"

		fmt = str(self.config.get("screenshot_format", "webp"))
		width = int(self.config.get("screenshot_width", 1920))
		height = int(self.config.get("screenshot_height", 1080))
		shot_url = self._build_screenshot_url(url, fmt=fmt, width=width, height=height)

		try:
			image_bytes, mime = await self._fetch_screenshot(shot_url, fmt)
		except Exception as e:
			logger.error(f"[webshot_analyze] 截图获取失败: {e}")
			return f"截图失败：{e}"

		try:
			image_part = self._make_image_part(image_bytes, mime)
			contents = [
				types.Content(
					role="user",
					parts=[image_part, types.Part.from_text(text=prompt)],
				)
			]
			model = self.config.get("model", "gemini-2.0-flash")
			resp = await client.models.generate_content(
				model=model,
				contents=contents,
			)
			text = getattr(resp, "text", None) or self._extract_text(resp)
			return text.strip() if text else "未从分析中获得可用文本结果。"
		except Exception as e:
			logger.error(f"[webshot_analyze] 调用失败: {e}")
			return f"分析失败：{e}"

	@llm_tool("webshot_send")
	async def webshot_send(self, event: AstrMessageEvent, url: str) -> str:
		"""对网页进行截图，并直接发送截图给用户。

		当配置项 `moderation_before_image_send` 为 True 时，将先调用 Gemini 对图片进行合规审核，
		若判定为 BLOCK 则不发送图片并提示用户；否则发送图片。

		Args:
			url(string): 网页 URL
		"""
		if httpx is None:
			return "插件缺少依赖 httpx，请在该插件 requirements.txt 中安装后重启。"

		fmt = str(self.config.get("screenshot_format", "webp"))
		width = int(self.config.get("screenshot_width", 1920))
		height = int(self.config.get("screenshot_height", 1080))
		shot_url = self._build_screenshot_url(url, fmt=fmt, width=width, height=height)

		should_moderate = bool(self.config.get("moderation_before_image_send", False))
		if should_moderate:
			try:
				client = self._get_client()
			except Exception as e:
				logger.error(f"[webshot_send] 初始化客户端失败: {e}")
				return "请先在插件配置中填写有效的 Gemini API Key。"
			try:
				image_bytes, mime = await self._fetch_screenshot(shot_url, fmt)
				image_part = self._make_image_part(image_bytes, mime)
				check_prompt = (
					"你是合规审核器。请仅输出一个词：ALLOW 或 BLOCK。"
					"当图片包含露骨色情、严重暴力、仇恨、隐私泄露、恶意程序二维码/链接等不适宜内容时输出 BLOCK；"
					"其余情况输出 ALLOW。"
				)
				contents = [
					types.Content(role="user", parts=[image_part, types.Part.from_text(text=check_prompt)])
				]
				model = self.config.get("model", "gemini-2.0-flash")
				resp = await client.models.generate_content(model=model, contents=contents)
				decision = (getattr(resp, "text", "") or self._extract_text(resp) or "").strip().upper()
				if "BLOCK" in decision and "ALLOW" not in decision:
					await event.send(MessageChain().message("由于合规审核未通过，图片已被拦截。"))
					return "图片已拦截（审核结果：BLOCK）。"
			except Exception as e:
				logger.error(f"[webshot_send] 审核失败: {e}")
				return f"审核失败：{e}"

		# 发送网络图片 URL（平台适配层会下载或转发）
		try:
			await event.send(MessageChain().url_image(shot_url))
			return f"截图已发送：{shot_url}"
		except Exception as e:
			logger.error(f"[webshot_send] 发送失败: {e}")
			return f"发送失败：{e}"

	def _get_client(self):
		"""根据配置选择 API Key，并创建/复用异步 client。支持随机或轮询策略。"""
		keys = self.config.get("api_key", []) or []
		if not keys:
			raise RuntimeError("请在插件配置中填写至少一个 Google API Key。")

		use_random = bool(self.config.get("random_api_key_selection", False))
		if use_random:
			key = random.choice(keys)
		else:
			key = keys[self._rr_index % len(keys)]
			self._rr_index += 1

		if key in self._clients:
			return self._clients[key]

		api_base = self.config.get(
			"api_base_url", "https://generativelanguage.googleapis.com"
		)
		if api_base.endswith("/"):
			api_base = api_base[:-1]

		http_options = types.HttpOptions(
			base_url=api_base,
		)
		client = genai.Client(api_key=key, http_options=http_options).aio
		self._clients[key] = client
		return client

	def _build_screenshot_url(self, page_url: str, fmt: str = "webp", width: int = 1920, height: int = 1080) -> str:
		"""构建截图服务 URL。默认使用 screenshotsnap.com。"""
		base = self.config.get(
			"screenshot_api_base", "https://screenshotsnap.com/api/screenshot"
		)
		params = {
			"url": page_url,
			"format": fmt,
			"width": width,
			"height": height,
		}
		return f"{base}?{urlencode(params)}"

	async def _fetch_screenshot(self, shot_url: str, fmt: str) -> tuple[bytes, str]:
		"""下载截图字节与 mime。"""
		mime = "image/webp" if fmt.lower() == "webp" else "image/png"
		timeout = float(self.config.get("fetch_timeout_seconds", 20))
		async with httpx.AsyncClient(timeout=timeout) as client:
			resp = await client.get(shot_url)
			resp.raise_for_status()
			return resp.content, mime

	async def _fetch_page_text(self, url: str) -> Optional[str]:
		"""抓取网页并提取纯文本。"""
		timeout = float(self.config.get("fetch_timeout_seconds", 20))
		headers = {"User-Agent": self.config.get("fetch_user_agent", "Mozilla/5.0 AstrBot")}
		async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
			resp = await client.get(url)
			resp.raise_for_status()
			html = resp.text
			if BeautifulSoup is None:
				# 退化处理：简单去标签
				return html
			soup = BeautifulSoup(html, "html.parser")
			# 移除脚本与样式
			for tag in soup(["script", "style", "noscript"]):
				tag.decompose()
			text = soup.get_text("\n")
			# 简单压缩空行
			lines = [ln.strip() for ln in text.splitlines()]
			return "\n".join([ln for ln in lines if ln])

	def _make_image_part(self, image_bytes: bytes, mime: str):
		"""构造 Gemini SDK 所需的图片 Part。"""
		return types.Part.from_bytes(data=image_bytes, mime_type=mime)

	@staticmethod
	def _extract_text(resp) -> Optional[str]:
		"""兼容性提取：把 candidates/parts 文本拼起来。"""
		try:
			if not resp or not getattr(resp, "candidates", None):
				return None
			parts = []
			for c in resp.candidates:
				content = getattr(c, "content", None)
				if not content or not getattr(content, "parts", None):
					continue
				for p in content.parts:
					t = getattr(p, "text", None)
					if t:
						parts.append(t)
			return "\n".join(parts) if parts else None
		except Exception:
			return None



