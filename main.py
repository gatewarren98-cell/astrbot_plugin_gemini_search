import random
import asyncio
from typing import Optional
from urllib.parse import urlencode

import astrbot.api.star as star
from astrbot.api import llm_tool, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.message.message_event_result import MessageChain
# 新增：引入事件过滤器
from astrbot.api.event.filter import event_message_type, EventMessageType

# Google GenAI SDK
from google import genai
from google.genai import types

# HTTP & HTML parsing
try:
	import httpx
except Exception:  # pragma: no cover
	httpx = None

try:
	from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
	BeautifulSoup = None


class Main(star.Star):
	"""
	使用 Gemini 2.0 Flash + Google Search 原生工具进行联网检索。
	已配置为：每次用户发消息时，自动在后台拦截并检索，若检索成功则把结果注入上下文，
	若遇到 RPM 限制或报错，则取消检索，直接让主模型回复。
	"""

	def __init__(self, context: star.Context, config=None) -> None:
		self.context = context
		self.config = config or {}
		self._rr_index = 0
		self._clients: dict[str, genai.Client] = {}

	async def initialize(self):
		# 注意：取消了 gemini_search 的工具注册，防止大模型重复调用
		self.context.activate_llm_tool("web_fetch")
		self.context.activate_llm_tool("webshot_analyze")
		self.context.activate_llm_tool("webshot_send")
		logger.info("[web_fetch] 函数工具已启用")
		logger.info("[webshot_analyze] 函数工具已启用")
		logger.info("[webshot_send] 函数工具已启用")
		logger.info("[gemini_search] 已转为全局静默拦截模式，不再作为按需工具")

	@event_message_type(EventMessageType.ALL)
	async def auto_search_for_every_message(self, event: AstrMessageEvent):
		"""
		全局消息拦截器：每次用户发送消息时，尝试触发搜索并将结果注入上下文。
		"""
		user_text = event.message_obj.message_str.strip()
		
		# 忽略过短的消息或空消息（例如纯图片）
		if not user_text or len(user_text) < 2:
			return
			
		logger.info(f"[全局检索] 正在为用户消息尝试检索: {user_text}")
		
		# 调用内部搜索方法
		search_result = await self._internal_gemini_search(user_text)
		
		# 如果检索成功且有结果，注入到用户消息中
		if search_result:
			logger.info("[全局检索] 检索成功，已将结果注入上下文")
			append_text = (
				f"\n\n=== 系统自动注入的实时网络检索结果 ===\n"
				f"{search_result}\n"
				f"======================================\n"
				f"请务必结合上述最新检索结果来回答我的问题。"
			)
			event.message_obj.message_str += append_text
		else:
			# 检索失败（例如触发 RPM 限制、网络超时等），跳过注入
			# 此时 event.message_obj 保持原样，框架会直接用原问题去请求主 LLM
			logger.warning("[全局检索] 检索失败或触发频率限制，已跳过检索，平滑回退到普通对话")

	async def _internal_gemini_search(self, query: str) -> Optional[str]:
		"""
		内部检索函数（不再使用 @llm_tool 装饰器）。
		失败时返回 None，以便拦截器判断是否需要降级。
		"""
		try:
			client = self._get_client()
		except Exception as e:
			logger.error(f"[_internal_gemini_search] 初始化客户端失败: {e}")
			return None

		model = self.config.get("model", "gemini-2.0-flash")

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
			return text.strip() if text else None
		except Exception as e:
			# 这里会捕获 429 Too Many Requests (RPM超限) 等异常
			logger.error(f"[_internal_gemini_search] 调用 API 失败 (可能触发了 RPM 限制): {e}")
			return None

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
			# 直接将图片注入到主模型的请求上下文中
			logger.info(f"[webshot_analyze] 将截图注入到主模型请求上下文中进行分析")
			try:
				import base64
				base64_str = base64.b64encode(image_bytes).decode('utf-8')
				# 获取当前的 ProviderRequest 并添加图片
				req = event.get_extra("provider_request")
				if req is not None:
					# 使用 base64:// 前缀，框架会自动处理
					req.image_urls.append(f"base64://{base64_str}")
					logger.info(f"[webshot_analyze] 成功将截图注入到 ProviderRequest.image_urls 中")
					return f"已获取网页 {url} 的截图并注入到上下文中。请根据截图内容回答用户问题。{prompt}"
				else:
					# 如果获取不到 ProviderRequest，回退到发送给用户
					logger.warning("[webshot_analyze] 无法获取 ProviderRequest，回退到发送图片给用户")
					await event.send(MessageChain().base64_image(base64_str))
					return f"已获取网页 {url} 的截图（已发送给用户）。{prompt}"
			except Exception as e:
				logger.error(f"[webshot_analyze] 注入截图失败: {e}")
				return f"注入截图失败：{e}"
		
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
		try:
			await event.send(MessageChain().url_image(shot_urls[0]))
			return f"截图已发送（使用 {len(shot_urls)} 个备用服务）"
		except Exception as e:
			logger.error(f"[webshot_send] 发送失败: {e}")
			return f"发送失败：{e}"

	def _get_client(self):
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
		bases = self.config.get(
			"screenshot_api_base", 
			["https://screenshotsnap.com/api/screenshot"]
		)
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
		mime = "image/webp" if fmt.lower() == "webp" else "image/png"
		timeout = float(self.config.get("fetch_timeout_seconds", 20))
		
		retry_rounds = int(self.config.get("screenshot_retry_rounds", 2))
		retry_rounds = max(0, min(5, retry_rounds))
		total_rounds = retry_rounds + 1
		
		if not shot_urls:
			raise RuntimeError("没有配置截图服务")
		
		all_errors = []
		total_attempts = 0
		
		for round_num in range(1, total_rounds + 1):
			if round_num > 1:
				logger.info(f"[webshot] ====== 开始第 {round_num}/{total_rounds} 轮重试 ======")
				await asyncio.sleep(1.5)
			
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
					await asyncio.sleep(0.5)
		
		error_summary = "\n  ".join(all_errors[-6:])
		raise RuntimeError(
			f"所有截图服务在 {total_rounds} 轮尝试中均失败（共 {total_attempts} 次尝试）。\n"
			f"最近的错误:\n  {error_summary}"
		)

	async def _fetch_page_text(self, url: str) -> Optional[str]:
		timeout = float(self.config.get("fetch_timeout_seconds", 20))
		headers = {"User-Agent": self.config.get("fetch_user_agent", "Mozilla/5.0 AstrBot")}
		async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
			resp = await client.get(url)
			resp.raise_for_status()
			html = resp.text
			if BeautifulSoup is None:
				return html
			soup = BeautifulSoup(html, "html.parser")
			for tag in soup(["script", "style", "noscript"]):
				tag.decompose()
			text = soup.get_text("\n")
			lines = [ln.strip() for ln in text.splitlines()]
			return "\n".join([ln for ln in lines if ln])

	def _make_image_part(self, image_bytes: bytes, mime: str):
		return types.Part.from_bytes(data=image_bytes, mime_type=mime)

	@staticmethod
	def _extract_text(resp) -> Optional[str]:
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


