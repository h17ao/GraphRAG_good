#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, List
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig
)

from Config.LLMConfig import LLMConfig, LLMType
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.LLMProviderRegister import register_provider
from Core.Common.CostManager import CostManager


@register_provider([LLMType.HUGGINGFACE])
class HuggingFaceLLM(BaseLLM):
    """HuggingFace Transformers LLM Provider for local model loading"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._init_model()
        self.cost_manager: Optional[CostManager] = None
        # Add semaphore for concurrent request limiting (required by BaseLLM)
        import asyncio
        self.semaphore = asyncio.Semaphore(1)

    def _init_model(self):
        """Initialize HuggingFace model and tokenizer"""
        model_kwargs = {
            'torch_dtype': getattr(torch, getattr(self.config, 'torch_dtype', 'float16')),
            'device_map': getattr(self.config, 'device_map', 'auto'),
        }
        
        if hasattr(self.config, 'trust_remote_code'):
            model_kwargs['trust_remote_code'] = self.config.trust_remote_code
            
        if self.config.MAX_MODEL_LEN:
            model_kwargs['max_position_embeddings'] = self.config.MAX_MODEL_LEN

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model, **model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, 
            trust_remote_code=getattr(self.config, 'trust_remote_code', True),
            use_fast=False
        )
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Generation config with thinking length limit for Qwen3
        generation_config_kwargs = {
            'max_new_tokens': self.config.max_token,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'do_sample': self.config.temperature > 0,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # 限制Qwen3的思考长度，避免过长思考（特别是在实体关系抽取时）
        if hasattr(self.config, 'max_thinking_length'):
            generation_config_kwargs['max_thinking_length'] = self.config.max_thinking_length
        elif 'qwen' in self.config.model.lower():
            # 为Qwen模型设置默认思考长度限制：512 tokens用于实体抽取，1024用于推理
            generation_config_kwargs['max_thinking_length'] = 512
            
        self.generation_config = GenerationConfig(**generation_config_kwargs)

    async def aask(self, msg: str, **kwargs) -> str:
        """Async interface to generate response"""
        return self.ask(msg, **kwargs)

    def ask(self, msg: str, **kwargs) -> str:
        """Generate response using HuggingFace model"""
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [{"role": "user", "content": msg}]
                formatted_input = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback to plain text if chat template fails
                formatted_input = msg
        else:
            formatted_input = msg

        # Tokenize
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,  # 修复: truncate -> truncation
            max_length=self.config.MAX_MODEL_LEN or 32768,
            padding=False
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            # 处理max_new_tokens参数
            if 'max_new_tokens' in kwargs and kwargs['max_new_tokens'] is not None:
                # 如果传入了max_new_tokens，动态修改generation_config
                generation_config = self.generation_config
                generation_config.max_new_tokens = kwargs['max_new_tokens']
                filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'max_new_tokens'}
            else:
                generation_config = self.generation_config
                filtered_kwargs = kwargs
                
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                **filtered_kwargs
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()

    async def aask_batch(self, msgs: List[str], **kwargs) -> List[str]:
        """Async batch interface"""
        return [self.ask(msg, **kwargs) for msg in msgs]

    @property
    def use_system_prompt(self) -> bool:
        """Whether this LLM supports system prompts"""
        return True

    def get_model(self):
        """Return the underlying HuggingFace model"""
        return self.model
        
    def get_tokenizer(self):
        """Return the tokenizer"""
        return self.tokenizer
    
    async def _achat_completion(self, messages: list[dict], timeout=None, max_tokens=None, format="text"):
        """Async chat completion implementation"""
        # Convert messages to a single prompt
        prompt = self._messages_to_prompt(messages)
        # 正确传递max_tokens作为max_new_tokens
        response = self.ask(prompt, max_new_tokens=max_tokens)
        
        # Return in OpenAI-compatible format
        return {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,  # HF doesn't provide token counts easily
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    async def _achat_completion_stream(self, messages: list[dict], timeout=None, max_tokens=None, format="text"):
        """Async chat completion stream implementation (fallback to non-stream)"""
        # For simplicity, fallback to non-streaming
        response = await self._achat_completion(messages, timeout, max_tokens, format)
        return self.get_choice_text(response)
    
    async def acompletion(self, messages: list[dict], timeout=None, max_tokens=None):
        """Async completion implementation"""
        return await self._achat_completion(messages, timeout, max_tokens)
    
    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert message list to a single prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n".join(prompt_parts)