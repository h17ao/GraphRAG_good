# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/01 01:02
@File    : CostManager.py
@Desc    : Refer to the MetaGPT https://github.com/geekan/MetaGPT/blob/main/metagpt/utils/cost_manager.py
"""

import re
from typing import NamedTuple

from pydantic import BaseModel

from Core.Common.Logger import logger
from Core.Utils.TokenCounter import FIREWORKS_GRADE_TOKEN_COSTS, TOKEN_COSTS


class Costs(NamedTuple):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float
    total_budget: float
    model_name: str = ""  # 添加模型名称字段


class CostManager(BaseModel):
    """Calculate the overhead of using the interface."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_budget: float = 0
    max_budget: float = 10000000.0
    total_cost: float = 0
    current_model: str = ""  # 当前使用的模型名称
    token_costs: dict[str, dict[str, float]] = TOKEN_COSTS  # different model's token cost
    stage_costs: list[Costs] = []

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        if prompt_tokens + completion_tokens == 0 or not model:
            return
        
        # 更新当前模型名称
        self.current_model = model
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        if model not in self.token_costs:
            logger.warning(f"Model {model} not found in TOKEN_COSTS.")
            return

        cost = (
            prompt_tokens * self.token_costs[model]["prompt"]
            + completion_tokens * self.token_costs[model]["completion"]
        ) / 1000
        self.total_cost += cost
        logger.info(
            f"模型: {model} | Total running cost: ${self.total_cost:.3f} | Max budget: ${self.max_budget:.3f} | "
            f"Current cost: ${cost:.3f}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}"
        )

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_costs(self) -> Costs:
        """Get all costs"""
        return Costs(
            self.total_prompt_tokens, 
            self.total_completion_tokens, 
            self.total_cost, 
            self.total_budget,
            self.current_model
        )

    def set_stage_cost(self):
        """Set the cost of the current stage."""
        self.stage_costs.append(self.get_costs())
        
        
    def get_last_stage_cost(self) -> Costs:
        """Get the cost of the last stage."""
        
        current_cost = self.get_costs()
        
        if len(self.stage_costs) == 0:
            last_cost = Costs(0, 0, 0, 0, "")
        else:
            last_cost = self.stage_costs[-1]
        
        last_stage_cost = Costs(
            current_cost.total_prompt_tokens - last_cost.total_prompt_tokens,
            current_cost.total_completion_tokens - last_cost.total_completion_tokens,
            current_cost.total_cost - last_cost.total_cost,
            current_cost.total_budget - last_cost.total_budget,
            current_cost.model_name  # 使用当前模型名称
        )
        
        self.set_stage_cost()
        
        return last_stage_cost
            
        
    

class TokenCostManager(CostManager):
    """open llm model is self-host, it's free and without cost"""

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        # 更新当前模型名称
        self.current_model = model
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(f"模型: {model} | prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}")


class FireworksCostManager(CostManager):
    def model_grade_token_costs(self, model: str) -> dict[str, float]:
        def _get_model_size(model: str) -> float:
            size = re.findall(".*-([0-9.]+)b", model)
            size = float(size[0]) if len(size) > 0 else -1
            return size

        if "mixtral-8x7b" in model:
            token_costs = FIREWORKS_GRADE_TOKEN_COSTS["mixtral-8x7b"]
        else:
            model_size = _get_model_size(model)
            if 0 < model_size <= 16:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["16"]
            elif 16 < model_size <= 80:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["80"]
            else:
                token_costs = FIREWORKS_GRADE_TOKEN_COSTS["-1"]
        return token_costs

    def update_cost(self, prompt_tokens: int, completion_tokens: int, model: str):
        """
        Refs to `https://app.fireworks.ai/pricing` **Developer pricing**
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
        model (str): The model used for the API call.
        """
        # 更新当前模型名称
        self.current_model = model
        
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        token_costs = self.model_grade_token_costs(model)
        cost = (prompt_tokens * token_costs["prompt"] + completion_tokens * token_costs["completion"]) / 1000000
        self.total_cost += cost
        logger.info(
            f"模型: {model} | Total running cost: ${self.total_cost:.4f} | "
            f"Current cost: ${cost:.4f}, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}"
        )