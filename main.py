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
		"""此工具用于对网页进行截图，根据配置选择返回图片给AI模型分析。

		Args:
			url(string): 网页 URL
			prompt(string): 对截图的分析提示词（可选）

		Returns:
			str: 模型对截图的分析文本，或图片标记（由主模型分析）
		"""
		if httpx is None:
			return "插件缺少依赖 httpx，请在该插件 requirements.txt 中安装后重启。"

		fmt = str(self.config.get("screenshot_format", "webp"))
		width = int(self.config.get("screenshot_width", 1920))
		height = int(self.config.get("screenshot_height", 1080))
		shot_urls = self._build_screenshot_urls(url, fmt=fmt, width=width, height=height)

		try:
			image_bytes, mime = await self._fetch_screenshot(shot_urls, fmt)
		except Exception as e:
			logger.error(f"[webshot_analyze] 截图获取失败: {e}")
			return f"截图失败：{e}"

		# 检查是否使用插件内Gemini分析
		use_gemini = bool(self.config.get("webshot_analyze_with_gemini", True))
		
		if not use_gemini:
			# 直接将图片发送给主模型分析
			logger.info(f"[webshot_analyze] 将截图直接转发给主模型分析")
			try:
				# 将图片转为base64发送给用户(主模型会在上下文中看到)
				import base64
				base64_str = base64.b64encode(image_bytes).decode('utf-8')
				await event.send(MessageChain().base64_image(base64_str))
				return f"已获取网页 {url} 的截图。{prompt}"
			except Exception as e:
				logger.error(f"[webshot_analyze] 发送截图失败: {e}")
				return f"发送截图失败：{e}"
		
		# 使用插件内Gemini分析
		try:
			client = self._get_client()
		except Exception as e:
			logger.error(f"[webshot_analyze] 初始化客户端失败: {e}")
			return "请先在插件配置中填写有效的 Gemini API Key。"

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

		Args:
			url(string): 网页 URL
		"""
		if httpx is None:
			return "插件缺少依赖 httpx，请在该插件 requirements.txt 中安装后重启。"

		fmt = str(self.config.get("screenshot_format", "webp"))
		width = int(self.config.get("screenshot_width", 1920))
		height = int(self.config.get("screenshot_height", 1080))
		shot_urls = self._build_screenshot_urls(url, fmt=fmt, width=width, height=height)

		should_moderate = bool(self.config.get("moderation_before_image_send", False))
		if should_moderate:
			try:
				client = self._get_client()
			except Exception as e:
				logger.error(f"[webshot_send] 初始化客户端失败: {e}")
				return "请先在插件配置中填写有效的 Gemini API Key。"
			try:
				image_bytes, mime = await self._fetch_screenshot(shot_urls, fmt)
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
		# 注意：这里发送的是第一个服务的URL，实际下载会自动轮换
		try:
			await event.send(MessageChain().url_image(shot_urls[0]))
			return f"截图已发送（使用 {len(shot_urls)} 个备用服务）"
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

	def _build_screenshot_urls(self, page_url: str, fmt: str = "webp", width: int = 1920, height: int = 1080) -> list[str]:
		"""构建所有配置的截图服务 URL 列表。"""
		bases = self.config.get(
			"screenshot_api_base", 
			["https://screenshotsnap.com/api/screenshot"]
		)
		# 兼容旧配置：如果是字符串则转为列表
		if isinstance(bases, str):
			bases = [bases]
		
		params = {
			"url": page_url,
			"format": fmt,
			"width": width,
			"height": height,
		}
		query_string = urlencode(params)
		return [f"{base}?{query_string}" for base in bases]

	async def _fetch_screenshot(self, shot_urls: list[str], fmt: str) -> tuple[bytes, str]:
		"""下载截图字节与 mime，支持多服务轮换重试。
		
		策略：每轮遍历所有配置的截图服务，失败后进入下一轮重试。
		例如：2个服务，2轮重试 = 服务1→服务2→服务1→服务2（最多4次尝试）
		"""
		mime = "image/webp" if fmt.lower() == "webp" else "image/png"
		timeout = float(self.config.get("fetch_timeout_seconds", 20))
		
		# 获取重试轮数配置（0=不重试只1轮，1-5=重试1-5轮）
		retry_rounds = int(self.config.get("screenshot_retry_rounds", 2))
		retry_rounds = max(0, min(5, retry_rounds))  # 限制在0-5之间
		total_rounds = retry_rounds + 1  # 首次尝试 + 重试轮数
		
		if not shot_urls:
			raise RuntimeError("没有配置截图服务")
		
		all_errors = []  # 记录所有失败
		total_attempts = 0
		
		# 多轮遍历所有服务
		for round_num in range(1, total_rounds + 1):
			if round_num > 1:
				logger.info(f"[webshot] ====== 开始第 {round_num}/{total_rounds} 轮重试 ======")
				await asyncio.sleep(1.5)  # 轮次间隔
			
			# 遍历所有截图服务
			for service_idx, shot_url in enumerate(shot_urls, 1):
				total_attempts += 1
				service_name = shot_url.split('//')[1].split('/')[0] if '//' in shot_url else shot_url[:30]
				
				if round_num == 1:
					logger.info(f"[webshot] 尝试服务 {service_idx}/{len(shot_urls)}: {service_name}")
				else:
					logger.info(f"[webshot] 第{round_num}轮 - 服务 {service_idx}/{len(shot_urls)}: {service_name}")
				
				try:
					async with httpx.AsyncClient(timeout=timeout) as client:
						resp = await client.get(shot_url)
						resp.raise_for_status()
						
						if total_attempts > 1:
							logger.info(f"[webshot] ✓ 截图成功！服务: {service_name}（第{round_num}轮第{service_idx}个服务）")
						return resp.content, mime
						
				except Exception as e:
					error_msg = f"第{round_num}轮-服务{service_idx}({service_name}): {str(e)[:80]}"
					all_errors.append(error_msg)
					logger.warning(f"[webshot] ✗ {error_msg}")
					await asyncio.sleep(0.5)  # 服务间短暂延迟
		
		# 所有轮次都失败了
		error_summary = "\n  ".join(all_errors[-6:])  # 显示最后6个错误
		raise RuntimeError(
			f"所有截图服务在 {total_rounds} 轮尝试中均失败（共 {total_attempts} 次尝试）。\n"
			f"最近的错误:\n  {error_summary}"
		)

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



