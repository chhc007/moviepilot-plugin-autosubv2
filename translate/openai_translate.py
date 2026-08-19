import time
import random
from typing import List, Union

import openai
from cacheout import Cache

OpenAISessionCache = Cache(maxsize=100, ttl=3600, timer=time.time, default=None)


class OpenAi:
    _api_key: str = None
    _api_url: str = None
    _model: str = "gpt-3.5-turbo"

    def __init__(self, api_key: str = None, api_url: str = None, proxy: dict = None, model: str = None,
                 compatible: bool = False):
        self._api_key = api_key
        self._api_url = api_url
        base_url = self._api_url if compatible else self._api_url + "/v1"
        
        # 创建 OpenAI 客户端实例
        if proxy and proxy.get("https"):
            import httpx
            http_client = httpx.Client(proxies=proxy.get("https"))
            self.client = openai.OpenAI(api_key=self._api_key, base_url=base_url, http_client=http_client)
        else:
            self.client = openai.OpenAI(api_key=self._api_key, base_url=base_url)
        
        if model:
            self._model = model

    @staticmethod
    def __save_session(session_id: str, message: str):
        """
        保存会话
        :param session_id: 会话ID
        :param message: 消息
        :return:
        """
        seasion = OpenAISessionCache.get(session_id)
        if seasion:
            seasion.append({
                "role": "assistant",
                "content": message
            })
            OpenAISessionCache.set(session_id, seasion)

    @staticmethod
    def __get_session(session_id: str, message: str) -> List[dict]:
        """
        获取会话
        :param session_id: 会话ID
        :return: 会话上下文
        """
        seasion = OpenAISessionCache.get(session_id)
        if seasion:
            seasion.append({
                "role": "user",
                "content": message
            })
        else:
            seasion = [
                {
                    "role": "system",
                    "content": "请在接下来的对话中请使用中文回复，并且内容尽可能详细。"
                },
                {
                    "role": "user",
                    "content": message
                }]
            OpenAISessionCache.set(session_id, seasion)
        return seasion

    def __get_model(self, message: Union[str, List[dict]],
                    prompt: str = None,
                    user: str = "MoviePilot",
                    **kwargs):
        """
        获取模型
        """
        if not isinstance(message, list):
            if prompt:
                message = [
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            else:
                message = [
                    {
                        "role": "user",
                        "content": message
                    }
                ]
        return self.client.chat.completions.create(
            model=self._model,
            user=user,
            messages=message,
            **kwargs
        )

    @staticmethod
    def __clear_session(session_id: str):
        """
        清除会话
        :param session_id: 会话ID
        :return:
        """
        if OpenAISessionCache.get(session_id):
            OpenAISessionCache.delete(session_id)

    def translate_to_zh(self, text: str, context: str = None, max_retries: int = 3):
        """
        翻译为中文
        :param text: 输入文本
        :param context: 翻译上下文
        :param max_retries: 最大重试次数
        """
        system_prompt = """您是专业字幕翻译员。严格遵守以下铁律：

【格式铁律】
- 输入是带编号的字幕行，格式："序号. 原文"
- 输出必须保持相同数量的带编号行，格式："序号. 译文"
- 严禁合并行、拆分行、跳过行、添加行
- 每行译文必须严格对应同序号的原文，1对1翻译

【时间铁律】
- 每行字幕有固定显示时长，译文必须简洁精练，字数控制在原文的1.2倍以内
- 禁止添加解释、注释、括号补充说明
- 禁止把一行原文拆成两行输出

【翻译铁律】
- 翻译为简体中文，口语化、自然、符合影视观影习惯
- 人名、地名、专有名词保持原文或通用译名
- 语气、情感、粗口、俚语要传神，不要弱化
- 上下文已提供（如有），保持称谓和术语连贯
- 只输出译文，不输出任何解释、开场白、总结"""
        user_prompt = f"翻译上下文：\n{context}\n\n需要翻译的内容：\n{text}" if context else f"请翻译：\n{text}"
        
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                completion = self.__get_model(prompt=system_prompt,
                                              message=user_prompt,
                                              temperature=0.1,
                                              top_p=0.8)
                result = completion.choices[0].message.content.strip()
                return True, result
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    # 使用指数退避和随机抖动，避免多个请求同时重试
                    base_delay = 2 ** attempt  # 指数退避: 1s, 2s, 4s...
                    jitter = random.uniform(0.1, 0.9)  # 随机抖动: 0.1-0.9秒
                    sleep_time = base_delay + jitter
                    print(f"翻译请求失败 (第{attempt + 1}次尝试)：{last_error}，{sleep_time:.1f}秒后重试...")
                    time.sleep(sleep_time)
                else:
                    print(f"翻译请求失败 (已重试{max_retries}次)：{last_error}")
                    return False, f"{last_error}"
