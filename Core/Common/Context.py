#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from Option.Config2 import Config
from Config.LLMConfig import LLMConfig, LLMType
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Common.CostManager import (
    CostManager,
    FireworksCostManager,
    TokenCostManager,
)



class AttrDict(BaseModel):
    """A dict-like object that allows access to keys as attributes, compatible with Pydantic."""

    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        return self.__dict__.get(key, None)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key in self.__dict__:
            del self.__dict__[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def set(self, key, val: Any):
        self.__dict__[key] = val

    def get(self, key, default: Any = None):
        return self.__dict__.get(key, default)

    def remove(self, key):
        if key in self.__dict__:
            self.__delattr__(key)


class Context(BaseModel):


    model_config = ConfigDict(arbitrary_types_allowed=True)

    kwargs: AttrDict = AttrDict()
    config: Config = Config.default()


    src_workspace: Optional[Path] = None
    cost_manager: CostManager = CostManager()

    _llm: Optional[BaseLLM] = None

    def new_environ(self):
        """Return a new os.environ object"""
        env = os.environ.copy()
        # i = self.options
        # env.update({k: v for k, v in i.items() if isinstance(v, str)})
        return env

    def _select_costmanager(self, llm_config: LLMConfig) -> CostManager:
        """Return a CostManager instance"""
        if llm_config.api_type == LLMType.FIREWORKS:
            return FireworksCostManager()
        elif llm_config.api_type == LLMType.OPEN_LLM:
            return TokenCostManager()
        else:
            return self.cost_manager

    def llm(self) -> BaseLLM:
        """Return a LLM instance, fixme: support cache"""
        # if self._llm is None:
        self._llm = create_llm_instance(self.config.llm)
        if self._llm.cost_manager is None:
            self._llm.cost_manager = self._select_costmanager(self.config.llm)
        return self._llm

    def llm_with_cost_manager_from_llm_config(self, llm_config: LLMConfig) -> BaseLLM:
        """Return a LLM instance, fixme: support cache"""
        # if self._llm is None:
   
        llm = create_llm_instance(llm_config)
        if llm.cost_manager is None:
            llm.cost_manager = self._select_costmanager(llm_config)
        return llm

    def retrieval_llm(self) -> BaseLLM:
        """Return a retrieval-specific LLM instance"""
        if not self.config.retrieval_llm:
            raise ValueError("retrieval_llm未配置，请在配置文件中添加retrieval_llm配置")
        return self.llm_with_cost_manager_from_llm_config(self.config.retrieval_llm)
    
    def eval_llm(self) -> BaseLLM:
        """Return an evaluation-specific LLM instance"""
        if not self.config.eval_llm:
            raise ValueError("eval_llm未配置，请在配置文件中添加eval_llm配置")
        return self.llm_with_cost_manager_from_llm_config(self.config.eval_llm)

  
