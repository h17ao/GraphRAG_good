from __future__ import annotations

import numpy as np
from typing import Optional, Union
import asyncio
import time  # 添加time模块用于延迟
import tiktoken

from openai import APIConnectionError, AsyncOpenAI, AsyncStream, APIError, RateLimitError, APITimeoutError
from openai._base_client import AsyncHttpxClientWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    after_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed,
)

from Config.LLMConfig import LLMConfig, LLMType
from Core.Common.Constants import USE_CONFIG_TIMEOUT
from Core.Common.Logger import log_llm_stream, logger
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.LLMProviderRegister import register_provider
from Core.Common.Utils import  log_and_reraise,prase_json_from_response
from Core.Common.CostManager import CostManager
from Core.Utils.Exceptions import handle_exception
from Core.Utils.TokenCounter import (
    count_input_tokens,
    count_output_tokens,
    get_max_completion_tokens,
)


@register_provider(
    [
        LLMType.OPENAI,
        LLMType.FIREWORKS,
        LLMType.OPEN_LLM,
    ]
)
class OpenAILLM(BaseLLM):
    """Check https://platform.openai.com/examples for examples"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._init_client()
        self.auto_max_tokens = False
        self.cost_manager: Optional[CostManager] = None
        self._semaphore = None  #   修改semaphore，在每个新的事件循环中动态重新创建
        self._semaphore_loop = None  #   修改semaphore，在每个新的事件循环中动态重新创建
        
        # 初始化tokenizer（保留用于其他功能）
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 使用通用tokenizer，与模型差异<3%


    #   修改semaphore，在每个新的事件循环中动态重新创建
    @property
    def semaphore(self):
        """Get or create semaphore for current event loop"""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop
            current_loop = None
            
        if self._semaphore is None or self._semaphore_loop != current_loop:
            if current_loop is not None:
                self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
                self._semaphore_loop = current_loop
            else:
                # If no event loop is running, create a basic semaphore
                self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
                self._semaphore_loop = None
                
        return self._semaphore

    def _init_client(self):
        """https://github.com/openai/openai-python#async-usage"""
        self.model = self.config.model  # Used in _calc_usage & _cons_kwargs
        self.pricing_plan = self.config.pricing_plan or self.model
        kwargs = self._make_client_kwargs()
        self.aclient = AsyncOpenAI(**kwargs)

    def _make_client_kwargs(self) -> dict:
        kwargs = {"api_key": self.config.api_key, "base_url": self.config.base_url}

        # to use proxy, openai v1 needs http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs

    def _get_proxy_params(self) -> dict:
        params = {}
        if self.config.proxy:
            params = {"proxies": self.config.proxy}
            if self.config.base_url:
                params["base_url"] = self.config.base_url

        return params

    # 自定义：首次命中内容审查错误则不重试
    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        import re
        msg = str(exc) if exc is not None else ""
        if re.search(r"data_inspection_failed|inappropriate content", msg, re.IGNORECASE):
            return False
        return isinstance(exc, (
            APIError,
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            ConnectionError,
            TimeoutError,
            Exception,
        ))

    @retry(
        wait=wait_fixed(3),  # 固定等待3秒
        stop=stop_after_attempt(20),  # 最多重试20次
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception(_should_retry),
        retry_error_callback=log_and_reraise,
    )
    async def _achat_completion_stream(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None) -> str:
        response: AsyncStream[ChatCompletionChunk] = await self.aclient.chat.completions.create(
            **self._cons_kwargs(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens), stream=True
        )
        usage = None
        collected_messages = []
        has_finished = False
        async for chunk in response:
            chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
            finish_reason = (
                chunk.choices[0].finish_reason if chunk.choices and hasattr(chunk.choices[0], "finish_reason") else None
            )
            log_llm_stream(chunk_message)
            collected_messages.append(chunk_message)
            chunk_has_usage = hasattr(chunk, "usage") and chunk.usage
            if has_finished:
                # for oneapi, there has a usage chunk after finish_reason not none chunk
                if chunk_has_usage:
                    usage = CompletionUsage(**chunk.usage) if isinstance(chunk.usage, dict) else chunk.usage
            if finish_reason:
                if chunk_has_usage:
                    # Some services have usage as an attribute of the chunk, such as Fireworks
                    if isinstance(chunk.usage, CompletionUsage):
                        usage = chunk.usage
                    else:
                        usage = CompletionUsage(**chunk.usage)
                elif hasattr(chunk.choices[0], "usage"):
                    # The usage of some services is an attribute of chunk.choices[0], such as Moonshot
                    usage = CompletionUsage(**chunk.choices[0].usage)
                has_finished = True

        log_llm_stream("\n")
        full_reply_content = "".join(collected_messages)
        if not usage:
            # Some services do not provide the usage attribute, such as OpenAI or OpenLLM
            usage = self._calc_usage(messages, full_reply_content)

        self._update_costs(usage)
        return full_reply_content

    def _cons_kwargs(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None, **extra_kwargs) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self._get_max_tokens(messages),
            # "n": 1,  # Some services do not provide this parameter, such as mistral
            "stop": ["[/INST]", "<<SYS>>"] ,  # default it's None and gpt4-v can't have this one
            "temperature": self.config.temperature,
            "model": self.model,
            "timeout": self.get_timeout(timeout),
        }
        if "o1-" in self.model:
            # compatible to openai o1-series
            kwargs["temperature"] = 1
            kwargs.pop("max_tokens")
        if max_tokens != None:
            kwargs["max_tokens"] = max_tokens
        # 添加chat_template_kwargs支持，用于控制Qwen3思考模式等
        if self.config.chat_template_kwargs:
            kwargs["extra_body"] = {"chat_template_kwargs": self.config.chat_template_kwargs}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs
    


    @retry(
        wait=wait_fixed(3),  # 固定等待3秒
        stop=stop_after_attempt(20),  # 最多重试20次
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception(_should_retry),
        retry_error_callback=log_and_reraise,
    )
    async def _achat_completion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None) -> ChatCompletion:
        kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout), max_tokens=max_tokens)
        rsp: ChatCompletion = await self.aclient.chat.completions.create(**kwargs)
        self._update_costs(rsp.usage)
        return rsp

    @retry(
        wait=wait_fixed(3),  # 固定等待3秒
        stop=stop_after_attempt(20),  # 最多重试20次
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception(_should_retry),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    @retry(
        wait=wait_fixed(3),  # 固定等待3秒
        stop=stop_after_attempt(20),  # 最多重试20次
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception(_should_retry),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT, max_tokens = None, format = "text") -> str:
        """文本补全方法，带重试机制"""
        if stream:
            return await self._achat_completion_stream(messages, timeout=timeout, max_tokens = max_tokens)

        rsp = await self._achat_completion(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens)

        rsp_text = self.get_choice_text(rsp)
        if format == "json":
            return prase_json_from_response(rsp_text)
        return rsp_text


 

    def get_choice_text(self, rsp: ChatCompletion) -> str:
        """Required to provide the first text of choice"""
        return rsp.choices[0].message.content if rsp.choices else ""

    def _calc_usage(self, messages: list[dict], rsp: str) -> CompletionUsage:
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if not self.config.calc_usage:
            return usage

        try:
            usage.prompt_tokens = count_input_tokens(messages, self.pricing_plan)
            usage.completion_tokens = count_output_tokens(rsp, self.pricing_plan)
        except Exception as e:
            logger.warning(f"usage calculation failed: {e}")

        return usage

    def _get_max_tokens(self, messages: list[dict]):
        if not self.auto_max_tokens:
            return self.config.max_token
        # FIXME
        # https://community.openai.com/t/why-is-gpt-3-5-turbo-1106-max-tokens-limited-to-4096/494973/3
        return min(get_max_completion_tokens(messages, self.model, self.config.max_token), 4096)


   
    def get_maxtokens(self) -> int:
       return ['max_tokens']

    @retry(
        wait=wait_fixed(3),  # 固定等待3秒
        stop=stop_after_attempt(20),  # 最多重试20次
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception(_should_retry),
        retry_error_callback=log_and_reraise,
    )
    async def openai_embedding(self, text):
        response = await self.aclient.embeddings.create(
            model = self.model, input = text, encoding_format = "float"
        )
        return np.array([dp.embedding for dp in response.data])